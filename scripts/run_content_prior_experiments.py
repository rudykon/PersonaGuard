#!/usr/bin/env python3
"""Evaluate metadata-conditioned temporal priors under strict double holdout.

This experiment targets deployment to a new participant watching a new video.
The outer split therefore excludes both the participant and the video whose
labels are predicted.  Hyperparameters are selected using outer-training data
only.  The resulting category/phase model is a metadata-only lower bound for
the planned audio-visual content prior; it must not be called a visual model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.content_prior import (  # noqa: E402
    DEFAULT_CATEGORY_PHASE_SPECS,
    CategoryPhaseSpec,
    doubly_held_out_category_priors,
    emotion_by_video,
    load_stimulus_manifest,
    nested_doubly_held_out_category_priors,
    trial_macro_mae,
)
from merps.innovation.data import load_innovation_index  # noqa: E402


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "stimuli" / "refed_15_videos.csv"
DEFAULT_STIMULUS_AUDIT = PROJECT_ROOT / "data" / "stimuli" / "stimulus_audit.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "content_prior_optimization"
MANAGED_FILES = {
    "fold_selections.json",
    "metrics.csv",
    "oof_predictions.npz",
    "results.json",
    "run_manifest.json",
    "trial_metrics.csv",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stimulus-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stimulus-audit", type=Path, default=DEFAULT_STIMULUS_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subject-folds", type=int, default=5)
    parser.add_argument("--video-folds", type=int, default=5)
    parser.add_argument("--inner-subject-folds", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def validate_manifest_against_index(index, records) -> dict[str, object]:
    by_video = {int(record.video_id): record for record in records}
    observed_videos = sorted(set(int(value) for value in index.videos))
    if observed_videos != sorted(by_video):
        raise ValueError(
            f"Manifest/data video mismatch: data={observed_videos}, "
            f"manifest={sorted(by_video)}"
        )
    lengths: dict[int, int] = {}
    for video in observed_videos:
        per_subject = []
        for subject in sorted(set(int(value) for value in index.subject_numbers)):
            rows = (index.subject_numbers == subject) & (index.videos == video)
            if not np.any(rows):
                raise ValueError(f"Missing subject {subject}, video {video} trial")
            timestamps = np.sort(index.timestamps[rows])
            expected = np.arange(len(timestamps), dtype=timestamps.dtype)
            if not np.array_equal(timestamps, expected):
                raise ValueError(
                    f"Non-contiguous timestamps for subject {subject}, video {video}"
                )
            per_subject.append(int(len(timestamps)))
        if len(set(per_subject)) != 1:
            raise ValueError(f"Inconsistent annotation lengths for video {video}")
        lengths[video] = per_subject[0]
        expected_length = int(by_video[video].annotation_seconds)
        if lengths[video] != expected_length:
            raise ValueError(
                f"Video {video}: observed {lengths[video]} seconds, "
                f"manifest says {expected_length}"
            )
    return {
        "status": "passed",
        "participants": len(set(int(value) for value in index.subject_numbers)),
        "videos": len(observed_videos),
        "samples": int(len(index.targets)),
        "annotation_lengths": {str(key): value for key, value in lengths.items()},
    }


def model_metrics(index, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    error = np.abs(prediction - np.asarray(index.targets, dtype=np.float64))
    participant_scores = []
    video_scores = []
    for subject in sorted(set(int(value) for value in index.subject_numbers)):
        participant_scores.append(float(error[index.subject_numbers == subject].mean()))
    for video in sorted(set(int(value) for value in index.videos)):
        video_scores.append(float(error[index.videos == video].mean()))
    return {
        "sample_mae": float(error.mean()),
        "valence_mae": float(error[:, 0].mean()),
        "arousal_mae": float(error[:, 1].mean()),
        "participant_macro_mae": float(np.mean(participant_scores)),
        "video_macro_mae": float(np.mean(video_scores)),
        "trial_macro_mae": float(trial_macro_mae(index, prediction)),
    }


def trial_error_matrix(index, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects = np.asarray(
        sorted(set(int(value) for value in index.subject_numbers)), dtype=np.int16
    )
    videos = np.asarray(sorted(set(int(value) for value in index.videos)), dtype=np.int16)
    matrix = np.empty((len(subjects), len(videos)), dtype=np.float64)
    for subject_index, subject in enumerate(subjects):
        for video_index, video in enumerate(videos):
            rows = (index.subject_numbers == subject) & (index.videos == video)
            matrix[subject_index, video_index] = float(
                np.abs(prediction[rows] - index.targets[rows]).mean()
            )
    return subjects, videos, matrix


def gain_summary(
    index,
    baseline: np.ndarray,
    comparator: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Describe trial-MAE gain with participant and crossed cluster resampling.

    The intervals condition on the fitted OOF predictions.  They represent
    participant/video sampling variation, not uncertainty from refitting the
    complete model-selection procedure.
    """

    subjects, videos, baseline_error = trial_error_matrix(index, baseline)
    _, _, comparator_error = trial_error_matrix(index, comparator)
    gain = baseline_error - comparator_error
    participant_gain = gain.mean(axis=1)
    video_gain = gain.mean(axis=0)
    rng = np.random.default_rng(seed)
    participant_bootstrap = np.empty(int(repeats), dtype=np.float64)
    crossed_bootstrap = np.empty(int(repeats), dtype=np.float64)
    for repeat in range(int(repeats)):
        sampled_subjects = rng.integers(0, len(subjects), size=len(subjects))
        sampled_videos = rng.integers(0, len(videos), size=len(videos))
        participant_bootstrap[repeat] = participant_gain[sampled_subjects].mean()
        crossed_bootstrap[repeat] = gain[
            sampled_subjects[:, None], sampled_videos[None, :]
        ].mean()
    interval = lambda values: [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]
    return {
        "trial_macro_mae_gain": float(gain.mean()),
        "positive_means_comparator_is_better": True,
        "participants_improved": int(np.sum(participant_gain > 0.0)),
        "participants_total": int(len(subjects)),
        "videos_improved": int(np.sum(video_gain > 0.0)),
        "videos_total": int(len(videos)),
        "participant_gain_median": float(np.median(participant_gain)),
        "video_gain_median": float(np.median(video_gain)),
        "participant_cluster_bootstrap_ci95": interval(participant_bootstrap),
        "participant_video_crossed_bootstrap_ci95": interval(crossed_bootstrap),
        "resamples": int(repeats),
        "uncertainty_scope": (
            "conditional_on_fitted_oof_predictions; sampling variation only; "
            "does not propagate full refit/model-selection uncertainty"
        ),
    }


def spec_to_dict(spec: CategoryPhaseSpec) -> dict[str, object]:
    temperature = float(spec.duration_temperature)
    return {
        "name": spec.name,
        "phase_weight": float(spec.phase_weight),
        "duration_temperature": "infinity" if np.isinf(temperature) else temperature,
        "smooth_radius": int(spec.smooth_radius),
    }


def build_trial_rows(index, predictions: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    subjects = sorted(set(int(value) for value in index.subject_numbers))
    videos = sorted(set(int(value) for value in index.videos))
    for subject in subjects:
        for video in videos:
            rows = (index.subject_numbers == subject) & (index.videos == video)
            record: dict[str, object] = {
                "subject": subject,
                "video": video,
                "seconds": int(np.sum(rows)),
            }
            for name, prediction in predictions.items():
                error = np.abs(prediction[rows] - index.targets[rows])
                record[f"{name}_mae"] = float(error.mean())
                record[f"{name}_valence_mae"] = float(error[:, 0].mean())
                record[f"{name}_arousal_mae"] = float(error[:, 1].mean())
            rows_out.append(record)
    return rows_out


def assert_fold_integrity(records: Sequence[Mapping[str, object]]) -> None:
    for record in records:
        train_subjects = set(int(value) for value in record["training_subjects"])
        valid_subjects = set(int(value) for value in record["validation_subjects"])
        train_videos = set(int(value) for value in record["training_videos"])
        valid_videos = set(int(value) for value in record["validation_videos"])
        if train_subjects.intersection(valid_subjects):
            raise RuntimeError("Participant leakage detected")
        if train_videos.intersection(valid_videos):
            raise RuntimeError("Video leakage detected")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = resolve(args.data_root)
    stimulus_manifest = resolve(args.stimulus_manifest)
    stimulus_audit_path = resolve(args.stimulus_audit)
    output_dir = resolve(args.output_dir)
    prepare_output(output_dir, args.overwrite)

    started = time.perf_counter()
    run_manifest: dict[str, object] = {
        "schema_version": "merps-content-prior-experiment-v1",
        "status": "running",
        "started_at_utc": utc_now(),
        "data_root": project_path(data_root),
        "stimulus_manifest": project_path(stimulus_manifest),
        "configuration": {
            "subject_folds": int(args.subject_folds),
            "video_folds": int(args.video_folds),
            "inner_subject_folds": int(args.inner_subject_folds),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "seed": int(args.seed),
            "candidate_specs": [spec_to_dict(spec) for spec in DEFAULT_CATEGORY_PHASE_SPECS],
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "source_hashes": {
            project_path(stimulus_manifest): file_sha256(stimulus_manifest),
            "src/merps/content_prior.py": file_sha256(PROJECT_ROOT / "src/merps/content_prior.py"),
            "scripts/run_content_prior_experiments.py": file_sha256(Path(__file__)),
        },
        "generated_files": [],
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    index = load_innovation_index(data_root)
    records = load_stimulus_manifest(stimulus_manifest)
    data_validation = validate_manifest_against_index(index, records)
    category_by_video = emotion_by_video(records)
    duration_by_video = {
        int(record.video_id): float(record.reported_duration_seconds)
        for record in records
    }
    run_manifest["target_index_sha256"] = array_sha256(
        index.sample_ids.astype("S"), index.targets
    )

    fixed = doubly_held_out_category_priors(
        index,
        category_by_video,
        subject_folds=args.subject_folds,
        video_folds=args.video_folds,
    )
    nested = nested_doubly_held_out_category_priors(
        index,
        category_by_video,
        duration_by_video,
        subject_folds=args.subject_folds,
        video_folds=args.video_folds,
        inner_subject_folds=args.inner_subject_folds,
    )
    assert_fold_integrity(fixed.fold_records)
    assert_fold_integrity(nested.fold_records)
    if not np.array_equal(fixed.subject_fold_ids, nested.subject_fold_ids):
        raise RuntimeError("Fixed and nested participant folds differ")
    if not np.array_equal(fixed.video_fold_ids, nested.video_fold_ids):
        raise RuntimeError("Fixed and nested video folds differ")
    for name in ("global_constant", "category_constant"):
        if not np.allclose(fixed.predictions[name], nested.predictions[name]):
            raise RuntimeError(f"Fixed and nested {name} baselines differ")

    predictions = {
        "global_constant": nested.predictions["global_constant"],
        "category_constant": nested.predictions["category_constant"],
        "category_phase_fixed": fixed.predictions["category_phase"],
        "category_phase_nested": nested.predictions["category_phase_nested"],
    }
    metric_values = {
        name: model_metrics(index, prediction)
        for name, prediction in predictions.items()
    }
    metric_rows = [{"model": name, **values} for name, values in metric_values.items()]
    write_rows(output_dir / "metrics.csv", metric_rows)
    write_rows(output_dir / "trial_metrics.csv", build_trial_rows(index, predictions))
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        sample_ids=index.sample_ids,
        subject_numbers=index.subject_numbers,
        videos=index.videos,
        timestamps=index.timestamps,
        targets=index.targets,
        subject_fold_ids=nested.subject_fold_ids,
        video_fold_ids=nested.video_fold_ids,
        **predictions,
    )
    write_json(output_dir / "fold_selections.json", list(nested.fold_records))

    selected_counts = Counter(
        str(record["selected_spec"]["name"]) for record in nested.fold_records
    )
    audit = None
    if stimulus_audit_path.exists():
        audit = json.loads(stimulus_audit_path.read_text(encoding="utf-8"))
    nested_metric = metric_values["category_phase_nested"]["trial_macro_mae"]
    global_metric = metric_values["global_constant"]["trial_macro_mae"]
    category_metric = metric_values["category_constant"]["trial_macro_mae"]
    results = {
        "schema_version": "merps-content-prior-results-v1",
        "completed_at_utc": utc_now(),
        "deployment_target": "new participant and unseen video",
        "interaction_requirement": "none from the target participant",
        "available_at_prediction_time": [
            "stimulus target category supplied by the experiment design",
            "elapsed playback second",
            "scheduled/reported stimulus duration",
        ],
        "unavailable_to_every_outer_prediction": [
            "all labels from the target participant",
            "all labels from every outer-held-out video",
        ],
        "data_validation": data_validation,
        "fold_integrity": {
            "status": "passed",
            "outer_fold_cells": int(len(nested.fold_records)),
            "participant_and_video_groups_disjoint": True,
            "hyperparameters_selected_inside_outer_training_data": True,
        },
        "metrics": metric_values,
        "primary_gain": {
            "versus_global_constant": global_metric - nested_metric,
            "relative_percent_versus_global_constant": 100.0 * (global_metric - nested_metric) / global_metric,
            "versus_category_constant": category_metric - nested_metric,
            "relative_percent_versus_category_constant": 100.0 * (category_metric - nested_metric) / category_metric,
        },
        "gain_uncertainty": {
            "nested_vs_global_constant": gain_summary(
                index,
                predictions["global_constant"],
                predictions["category_phase_nested"],
                repeats=args.bootstrap_repeats,
                seed=args.seed,
            ),
            "nested_vs_category_constant": gain_summary(
                index,
                predictions["category_constant"],
                predictions["category_phase_nested"],
                repeats=args.bootstrap_repeats,
                seed=args.seed + 1,
            ),
            "nested_vs_fixed_phase": gain_summary(
                index,
                predictions["category_phase_fixed"],
                predictions["category_phase_nested"],
                repeats=args.bootstrap_repeats,
                seed=args.seed + 2,
            ),
        },
        "selected_spec_counts": dict(sorted(selected_counts.items())),
        "candidate_count": len(DEFAULT_CATEGORY_PHASE_SPECS),
        "stimulus_media_audit": audit,
        "interpretation_boundary": [
            "This is a metadata-conditioned temporal prior, not an audio-visual model.",
            "The emotion category is assumed known from the experimental design.",
            "No target-participant or held-out-video label is used for prediction.",
            "The exact source videos are absent/unverified when the media audit is blocked.",
            "EEG/fNIRS residual learning is intentionally not credited in this run.",
            "Conditional bootstrap intervals do not include full refit/model-selection uncertainty.",
        ],
        "next_stage_contract": {
            "content_model_target": "label minus metadata temporal prior",
            "physiology_model_target": "label minus fitted content prior",
            "physiology_gates": "selected within training folds and allowed to be exactly zero",
            "formal_media_requirement": "all 15 exact edits identity-verified by SHA-256",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(output_dir / "results.json", results)

    run_manifest.update(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "elapsed_seconds": results["elapsed_seconds"],
            "generated_files": sorted(
                item.name
                for item in output_dir.iterdir()
                if item.name != "run_manifest.json"
            ),
        }
    )
    write_json(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(results["metrics"], indent=2, sort_keys=True), flush=True)
    print(json.dumps(results["primary_gain"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
