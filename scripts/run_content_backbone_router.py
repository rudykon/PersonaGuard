#!/usr/bin/env python3
"""Route frozen content backbones using inner-fold scores only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.content_router import choose_backbone  # noqa: E402
from merps.innovation.data import load_innovation_index  # noqa: E402
from scripts.run_av_content_prior import gain_diagnostics, model_metrics  # noqa: E402


MANAGED_FILES = {
    "decisions.json",
    "oof_predictions.npz",
    "results.json",
    "run_manifest.json",
}
AXES = ("valence", "arousal")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="NAME:PATH",
        help="Repeat for each completed content-prior artifact.",
    )
    parser.add_argument("--baseline", default="clip")
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=(0.0, 0.02, 0.05, 0.1, 0.2)
    )
    parser.add_argument("--primary-threshold", type=float, default=0.05)
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--offsets", nargs="+", type=int, default=(-3, -2, -1, 0, 1, 2, 3))
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/MER_PS_trainval")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/content_backbone_router")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_output(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.iterdir())
    if not existing:
        return
    if not overwrite:
        raise FileExistsError(f"{path} is not empty; pass --overwrite")
    unknown = [item for item in existing if item.name not in MANAGED_FILES]
    if unknown:
        raise FileExistsError(
            "Refusing to overwrite unmanaged files: "
            + ", ".join(sorted(item.name for item in unknown))
        )
    for item in existing:
        if item.is_file() or item.is_symlink():
            item.unlink()


def parse_artifact(value: str) -> tuple[str, Path]:
    pieces = str(value).split(":", 1)
    if len(pieces) != 2 or not all(pieces):
        raise ValueError(f"Invalid --artifact {value!r}; expected NAME:PATH")
    return pieces[0], resolve(Path(pieces[1]))


def record_key(record: Mapping[str, object]) -> tuple[int, int]:
    return int(record["subject_fold"]), int(record["video_fold"])


@dataclass(frozen=True)
class ContentArtifact:
    name: str
    path: Path
    arrays: Mapping[str, np.ndarray]
    records: Mapping[str, Mapping[tuple[int, int], Mapping[str, object]]]

    @classmethod
    def load(cls, name: str, path: Path) -> "ContentArtifact":
        required = path / "oof_predictions.npz"
        fold_path = path / "fold_selections.json"
        result_path = path / "results.json"
        for item in (required, fold_path, result_path):
            if not item.is_file():
                raise FileNotFoundError(item)
        with np.load(required) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        raw_records = json.loads(fold_path.read_text(encoding="utf-8"))
        grouped: dict[str, dict[tuple[int, int], Mapping[str, object]]] = {}
        for record in raw_records:
            label = str(record["run_label"])
            key = record_key(record)
            if key in grouped.setdefault(label, {}):
                raise ValueError(f"Duplicate fold record in {name}: {label} {key}")
            grouped[label][key] = record
        return cls(name=name, path=path, arrays=arrays, records=grouped)


def repeat_label(repeat: int) -> str:
    return f"offset_{0:+d}_repeat_{int(repeat)}"


def offset_label(offset: int) -> str:
    return f"offset_{int(offset):+d}_repeat_0"


def repeat_prediction_key(repeat: int) -> str:
    return "primary_content" if int(repeat) == 0 else f"repeat_{int(repeat)}_content"


def repeat_metadata_key(repeat: int) -> str:
    return "primary_metadata" if int(repeat) == 0 else f"repeat_{int(repeat)}_metadata"


def offset_prediction_key(offset: int) -> str:
    token = f"m{abs(int(offset))}" if int(offset) < 0 else f"p{int(offset)}"
    return f"offset_{token}_content"


def validate_artifacts(artifacts: Mapping[str, ContentArtifact], baseline: str) -> None:
    if baseline not in artifacts:
        raise KeyError(f"Baseline artifact {baseline!r} was not supplied")
    reference = artifacts[baseline]
    identity = ("sample_ids", "subject_numbers", "videos", "timestamps", "targets")
    for name, artifact in artifacts.items():
        for key in identity:
            if key not in artifact.arrays or key not in reference.arrays:
                raise KeyError(f"Missing identity array {key} in {name}")
            if not np.array_equal(artifact.arrays[key], reference.arrays[key]):
                raise ValueError(f"Identity mismatch for {name}: {key}")
        if set(artifact.records) != set(reference.records):
            raise ValueError(f"Run-label mismatch for {name}")
        for label, reference_records in reference.records.items():
            records = artifact.records[label]
            if set(records) != set(reference_records):
                raise ValueError(f"Fold-cell mismatch for {name}: {label}")
            for key, reference_record in reference_records.items():
                record = records[key]
                for field in (
                    "training_subjects", "validation_subjects",
                    "training_videos", "validation_videos",
                ):
                    if record[field] != reference_record[field]:
                        raise ValueError(f"Partition mismatch for {name}: {label} {key} {field}")
                for axis in AXES:
                    observed = float(record["content_selection"]["axes"][axis]["metadata_mae"])
                    expected = float(reference_record["content_selection"]["axes"][axis]["metadata_mae"])
                    if not np.isclose(observed, expected, atol=1e-8, rtol=0.0):
                        raise ValueError(f"Inner metadata score mismatch for {name}: {label} {key} {axis}")


def route_run(
    artifacts: Mapping[str, ContentArtifact],
    *,
    baseline: str,
    label: str,
    prediction_key: str,
    threshold: float,
) -> tuple[np.ndarray, list[dict[str, object]], Counter[str]]:
    reference = artifacts[baseline]
    subjects = reference.arrays["subject_numbers"]
    videos = reference.arrays["videos"]
    output = np.full_like(reference.arrays["targets"], np.nan, dtype=np.float32)
    decisions: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    reference_records = reference.records[label]
    for key in sorted(reference_records):
        record = reference_records[key]
        rows = np.flatnonzero(
            np.isin(subjects, np.asarray(record["validation_subjects"], dtype=np.int16))
            & np.isin(videos, np.asarray(record["validation_videos"], dtype=np.int16))
        )
        if not len(rows):
            raise RuntimeError(f"Empty validation cell: {label} {key}")
        for axis_index, axis in enumerate(AXES):
            scores = {
                name: float(
                    artifact.records[label][key]["content_selection"]["axes"][axis][
                        "selected_mae"
                    ]
                )
                for name, artifact in artifacts.items()
            }
            decision = choose_backbone(
                scores,
                baseline=baseline,
                minimum_inner_gain=float(threshold),
            )
            selected_prediction = artifacts[decision.selected].arrays[prediction_key]
            output[rows, axis_index] = selected_prediction[rows, axis_index]
            counts[decision.selected] += 1
            decisions.append(
                {
                    "run_label": label,
                    "subject_fold": int(key[0]),
                    "video_fold": int(key[1]),
                    "axis": axis,
                    "threshold": float(threshold),
                    "selected": decision.selected,
                    "used_alternative": decision.used_alternative,
                    "inner_gain": decision.inner_gain,
                    "inner_mae_by_model": scores,
                    "validation_rows": int(len(rows)),
                }
            )
            if decision.selected == baseline:
                baseline_values = reference.arrays[prediction_key][rows, axis_index]
                if not np.array_equal(output[rows, axis_index], baseline_values):
                    raise RuntimeError("Exact fallback propagation failed")
    if not np.isfinite(output).all():
        raise RuntimeError(f"Incomplete routed prediction: {label} threshold={threshold}")
    return output, decisions, counts


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    thresholds = tuple(sorted(set(float(value) for value in args.thresholds)))
    if any(value < 0.0 for value in thresholds):
        raise ValueError("thresholds must be non-negative")
    if float(args.primary_threshold) not in thresholds:
        raise ValueError("primary-threshold must be included in thresholds")
    parsed = [parse_artifact(value) for value in args.artifact]
    names = [name for name, _ in parsed]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("Supply at least two uniquely named artifacts")
    artifacts = {name: ContentArtifact.load(name, path) for name, path in parsed}
    validate_artifacts(artifacts, args.baseline)
    output_dir = resolve(args.output_dir)
    prepare_output(output_dir, args.overwrite)
    index = load_innovation_index(resolve(args.data_root))
    reference = artifacts[args.baseline]
    for key, observed in (
        ("subject_numbers", index.subject_numbers),
        ("videos", index.videos),
        ("timestamps", index.timestamps),
        ("targets", index.targets),
    ):
        if not np.array_equal(reference.arrays[key], observed):
            raise ValueError(f"Input artifacts do not match dataset index: {key}")

    all_decisions: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {
        key: reference.arrays[key]
        for key in ("sample_ids", "subject_numbers", "videos", "timestamps", "targets")
    }
    primary_curve: dict[str, dict[str, object]] = {}
    primary_predictions: dict[float, np.ndarray] = {}
    for threshold in thresholds:
        prediction, decisions, counts = route_run(
            artifacts,
            baseline=args.baseline,
            label=repeat_label(0),
            prediction_key=repeat_prediction_key(0),
            threshold=threshold,
        )
        primary_predictions[threshold] = prediction
        all_decisions.extend(decisions)
        primary_curve[str(threshold)] = {
            "metrics": model_metrics(index, prediction),
            "gain_vs_baseline": float(
                model_metrics(index, reference.arrays["primary_content"])["trial_macro_mae"]
                - model_metrics(index, prediction)["trial_macro_mae"]
            ),
            "selected_axis_cells": dict(sorted(counts.items())),
        }
        saved[f"threshold_{str(threshold).replace('.', 'p')}_content"] = prediction

    primary_threshold = float(args.primary_threshold)
    primary = primary_predictions[primary_threshold]
    saved["primary_content"] = primary
    saved["baseline_content"] = reference.arrays["primary_content"]
    repeat_rows = []
    repeat_predictions = []
    for repeat in range(int(args.repeat_count)):
        prediction, decisions, counts = route_run(
            artifacts,
            baseline=args.baseline,
            label=repeat_label(repeat),
            prediction_key=repeat_prediction_key(repeat),
            threshold=primary_threshold,
        )
        all_decisions.extend(decisions)
        repeat_predictions.append(prediction)
        baseline_prediction = reference.arrays[repeat_prediction_key(repeat)]
        routed_metrics = model_metrics(index, prediction)
        baseline_metrics = model_metrics(index, baseline_prediction)
        repeat_rows.append(
            {
                "repeat": repeat,
                "baseline": baseline_metrics,
                "routed": routed_metrics,
                "gain_vs_baseline": float(
                    baseline_metrics["trial_macro_mae"] - routed_metrics["trial_macro_mae"]
                ),
                "selected_axis_cells": dict(sorted(counts.items())),
            }
        )
        saved[f"repeat_{repeat}_content"] = prediction

    offset_rows: dict[str, dict[str, object]] = {}
    offset_values = []
    for offset in (int(value) for value in args.offsets):
        prediction, decisions, counts = route_run(
            artifacts,
            baseline=args.baseline,
            label=offset_label(offset),
            prediction_key=offset_prediction_key(offset),
            threshold=primary_threshold,
        )
        all_decisions.extend(decisions)
        metrics = model_metrics(index, prediction)
        offset_values.append(metrics["trial_macro_mae"])
        offset_rows[str(offset)] = {
            "metrics": metrics,
            "selected_axis_cells": dict(sorted(counts.items())),
        }
        saved[f"offset_{'m'+str(abs(offset)) if offset < 0 else 'p'+str(offset)}_content"] = prediction

    baseline_primary = reference.arrays["primary_content"]
    routed_repeat_mae = np.asarray(
        [row["routed"]["trial_macro_mae"] for row in repeat_rows], dtype=np.float64
    )
    repeated_gain = np.asarray(
        [row["gain_vs_baseline"] for row in repeat_rows], dtype=np.float64
    )
    results = {
        "schema_version": "merps-content-backbone-router-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_target": "new participant and unseen video; zero target interaction",
        "selection_boundary": (
            "Backbone choices use only inner grouped-CV MAE inside each outer "
            "participant-by-video cell; outer targets never select a backbone."
        ),
        "baseline": args.baseline,
        "candidate_artifacts": names,
        "primary_threshold": primary_threshold,
        "decision_curve": primary_curve,
        "primary_metrics": model_metrics(index, primary),
        "primary_gain_vs_baseline": float(
            model_metrics(index, baseline_primary)["trial_macro_mae"]
            - model_metrics(index, primary)["trial_macro_mae"]
        ),
        "primary_gain_diagnostics": gain_diagnostics(
            index,
            baseline_primary,
            primary,
            repeats=int(args.bootstrap_repeats),
            seed=int(args.seed),
        ),
        "repeated_grouped_cv": {
            "total_runs": len(repeat_rows),
            "runs": repeat_rows,
            "routed_trial_macro_mae_mean": float(routed_repeat_mae.mean()),
            "routed_trial_macro_mae_std": float(routed_repeat_mae.std(ddof=0)),
            "gain_vs_baseline_mean": float(repeated_gain.mean()),
            "positive_gain_runs": int(np.sum(repeated_gain > 0.0)),
        },
        "offset_sensitivity": {
            "fixed_offsets_seconds": [int(value) for value in args.offsets],
            "metrics": offset_rows,
            "trial_macro_mae_range": [float(min(offset_values)), float(max(offset_values))],
            "primary_offset_was_not_selected_on_outer_labels": True,
        },
        "exact_baseline_fallback_checked": True,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    np.savez_compressed(output_dir / "oof_predictions.npz", **saved)
    write_json(output_dir / "decisions.json", all_decisions)
    write_json(output_dir / "results.json", results)
    source_hashes: dict[str, str] = {
        "src/merps/content_router.py": file_sha256(PROJECT_ROOT / "src/merps/content_router.py"),
        "scripts/run_content_backbone_router.py": file_sha256(Path(__file__)),
    }
    for name, artifact in artifacts.items():
        for filename in ("oof_predictions.npz", "fold_selections.json", "results.json"):
            source_hashes[f"{name}:{filename}"] = file_sha256(artifact.path / filename)
    manifest = {
        "schema_version": "merps-content-backbone-router-run-v1",
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "baseline": args.baseline,
            "thresholds": list(thresholds),
            "primary_threshold": primary_threshold,
            "repeat_count": int(args.repeat_count),
            "offsets": [int(value) for value in args.offsets],
            "artifact_paths": {name: str(artifact.path) for name, artifact in artifacts.items()},
        },
        "source_hashes": source_hashes,
        "output_hashes": {
            filename: file_sha256(output_dir / filename)
            for filename in ("decisions.json", "oof_predictions.npz", "results.json")
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps({
        "primary_metrics": results["primary_metrics"],
        "primary_gain_vs_baseline": results["primary_gain_vs_baseline"],
        "repeated_grouped_cv": results["repeated_grouped_cv"],
        "offset_range": results["offset_sensitivity"]["trial_macro_mae_range"],
    }, indent=2))


if __name__ == "__main__":
    main()
