#!/usr/bin/env python3
"""Evaluate a frozen visual temporal prior with strict participant/video holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.av_content_prior import (  # noqa: E402
    DISABLED_CONTENT_SPEC,
    ContentTemplateSpec,
    FoundationFeatureArchive,
    population_curves,
    predict_content_template_prior,
)
from merps.content_prior import (  # noqa: E402
    CategoryPhaseSpec,
    emotion_by_video,
    load_stimulus_manifest,
    predict_category_phase_prior,
    round_robin_folds,
    select_category_phase_spec,
    trial_macro_mae,
)
from merps.innovation.data import load_innovation_index  # noqa: E402


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "stimuli" / "refed_15_videos.csv"
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "stimuli" / "features" / "foundation_av.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "av_content_prior"
MANAGED_FILES = {
    "ablation_metrics.csv",
    "fold_selections.json",
    "metrics.csv",
    "oof_predictions.npz",
    "results.json",
    "run_manifest.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stimulus-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--foundation-features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--content-modality",
        choices=("clip", "ast", "clip_ast", "visual", "visual_ast"),
        default="clip",
    )
    parser.add_argument("--subject-folds", type=int, default=5)
    parser.add_argument("--video-folds", type=int, default=5)
    parser.add_argument("--inner-base-subject-folds", type=int, default=3)
    parser.add_argument("--inner-content-subject-folds", type=int, default=2)
    parser.add_argument("--inner-content-video-folds", type=int, default=3)
    parser.add_argument("--minimum-inner-gain", type=float, default=0.05)
    parser.add_argument(
        "--offsets",
        nargs="+",
        type=int,
        default=(-3, -2, -1, 0, 1, 2, 3),
    )
    parser.add_argument("--primary-offset", type=int, default=0)
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=5,
        help="Total grouped-CV partitions including the fixed primary partition.",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def spec_to_dict(spec: ContentTemplateSpec) -> dict[str, object]:
    temperature = float(spec.source_temperature)
    return {
        "name": spec.name,
        "modality": spec.modality,
        "source_temperature": (
            "infinity" if np.isinf(temperature) else temperature
        ),
        "warp_band": float(spec.warp_band),
        "phase_penalty": float(spec.phase_penalty),
        "warp_gate": float(spec.warp_gate),
        "content_gate": float(spec.content_gate),
    }


def base_spec_to_dict(spec: CategoryPhaseSpec) -> dict[str, object]:
    temperature = float(spec.duration_temperature)
    return {
        "name": spec.name,
        "phase_weight": float(spec.phase_weight),
        "duration_temperature": (
            "infinity" if np.isinf(temperature) else temperature
        ),
        "smooth_radius": int(spec.smooth_radius),
    }


def candidate_specs(modality: str = "clip") -> tuple[ContentTemplateSpec, ...]:
    values = [DISABLED_CONTENT_SPEC]
    transfer = (
        (0.0, 0.0),
        (0.08, 0.50),
        (0.15, 0.50),
        (0.15, 0.75),
        (0.15, 1.00),
        (0.25, 0.75),
    )
    for temperature in (0.0, 0.01, 0.03, float("inf")):
        for band, warp_gate in transfer:
            for content_gate in (0.40, 0.60, 0.80, 1.00):
                values.append(
                    ContentTemplateSpec(
                        source_temperature=temperature,
                        warp_band=band,
                        phase_penalty=0.0,
                        warp_gate=warp_gate,
                        content_gate=content_gate,
                        modality=modality,
                    )
                )
    names = [spec.name for spec in values]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate content candidate names")
    return tuple(values)


def axis_trial_errors(
    index,
    rows: np.ndarray,
    prediction: np.ndarray,
    axis: int,
) -> list[float]:
    rows = np.asarray(rows, dtype=np.int64)
    values = np.asarray(prediction, dtype=np.float32)
    if values.shape != (len(rows), 2):
        raise ValueError("Prediction shape does not match rows")
    result = []
    subjects = sorted(set(int(value) for value in index.subject_numbers[rows]))
    videos = sorted(set(int(value) for value in index.videos[rows]))
    for subject in subjects:
        for video in videos:
            local = np.flatnonzero(
                (index.subject_numbers[rows] == subject)
                & (index.videos[rows] == video)
            )
            if len(local):
                result.append(
                    float(
                        np.abs(
                            values[local, axis]
                            - index.targets[rows[local], axis]
                        ).mean()
                    )
                )
    return result


def select_axis_specs(
    index,
    training_subjects: np.ndarray,
    training_videos: np.ndarray,
    category_by_video: Mapping[int, str],
    duration_by_video: Mapping[int, float],
    archive: FoundationFeatureArchive,
    base_spec: CategoryPhaseSpec,
    candidates: Sequence[ContentTemplateSpec],
    *,
    inner_subject_folds: int,
    inner_video_folds: int,
    minimum_gain: float,
    offset_seconds: int,
    map_cache: dict[tuple[object, ...], np.ndarray],
) -> tuple[tuple[ContentTemplateSpec, ContentTemplateSpec], dict[str, object]]:
    base_errors: list[list[float]] = [[], []]
    candidate_errors = {
        spec.name: [[], []] for spec in candidates
    }
    cells = 0
    for inner_training_subjects, inner_validation_subjects in round_robin_folds(
        training_subjects, inner_subject_folds
    ):
        for inner_training_videos, inner_validation_videos in round_robin_folds(
            training_videos, inner_video_folds
        ):
            rows = np.flatnonzero(
                np.isin(index.subject_numbers, inner_validation_subjects)
                & np.isin(index.videos, inner_validation_videos)
            )
            if not len(rows):
                continue
            base = predict_category_phase_prior(
                index,
                inner_training_subjects,
                inner_training_videos,
                rows,
                category_by_video,
                duration_by_video=duration_by_video,
                spec=base_spec,
            )
            curves = population_curves(
                index, inner_training_subjects, inner_training_videos
            )
            for axis in (0, 1):
                base_errors[axis].extend(
                    axis_trial_errors(index, rows, base, axis)
                )
            for candidate in candidates:
                prediction = predict_content_template_prior(
                    index,
                    rows,
                    base,
                    archive,
                    inner_training_subjects,
                    inner_training_videos,
                    category_by_video,
                    spec=candidate,
                    offset_seconds=offset_seconds,
                    curves=curves,
                    map_cache=map_cache,
                )
                for axis in (0, 1):
                    candidate_errors[candidate.name][axis].extend(
                        axis_trial_errors(index, rows, prediction, axis)
                    )
            cells += 1
    if not cells or any(not values for values in base_errors):
        raise RuntimeError("No valid inner content-selection cells")
    mean_base = [float(np.mean(values)) for values in base_errors]
    mean_scores = {
        name: [float(np.mean(values[axis])) for axis in (0, 1)]
        for name, values in candidate_errors.items()
    }
    by_name = {spec.name: spec for spec in candidates}
    selected = []
    rankings: dict[str, object] = {}
    for axis, axis_name in enumerate(("valence", "arousal")):
        best_name = min(
            mean_scores,
            key=lambda name: (
                mean_scores[name][axis],
                0 if name == "disabled" else 1,
                name,
            ),
        )
        gain = mean_base[axis] - mean_scores[best_name][axis]
        if best_name != "disabled" and gain < float(minimum_gain):
            best_name = "disabled"
            gain = 0.0
        selected.append(by_name[best_name])
        ordered = sorted(
            (
                {
                    "name": name,
                    "mae": scores[axis],
                    "gain_vs_metadata": mean_base[axis] - scores[axis],
                }
                for name, scores in mean_scores.items()
            ),
            key=lambda item: (float(item["mae"]), str(item["name"])),
        )
        rankings[axis_name] = {
            "metadata_mae": mean_base[axis],
            "selected": spec_to_dict(by_name[best_name]),
            "selected_mae": mean_scores[best_name][axis],
            "selected_gain": gain,
            "top_five": ordered[:5],
        }
    return (selected[0], selected[1]), {
        "inner_cells": cells,
        "axes": rankings,
    }


def shuffled_subject_folds(
    subjects: Sequence[int], n_folds: int, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    values = np.asarray(sorted(set(int(value) for value in subjects)), dtype=np.int16)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(values)
    result = []
    for fold in range(int(n_folds)):
        validation = np.sort(shuffled[fold:: int(n_folds)])
        training = values[~np.isin(values, validation)]
        result.append((training, validation))
    return tuple(result)


def stratified_video_folds(
    videos: Sequence[int],
    category_by_video: Mapping[int, str],
    n_folds: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if int(n_folds) != 5:
        return shuffled_subject_folds(videos, n_folds, seed)
    rng = np.random.default_rng(seed)
    categories = sorted(set(category_by_video[int(video)] for video in videos))
    categories = [categories[index] for index in rng.permutation(len(categories))]
    bins: list[list[int]] = [[] for _ in range(5)]
    shift = int(rng.integers(0, 5))
    for category_index, category in enumerate(categories):
        local = [
            int(video)
            for video in videos
            if category_by_video[int(video)] == category
        ]
        local = [local[index] for index in rng.permutation(len(local))]
        if len(local) > 5:
            raise ValueError("A category has more videos than folds")
        for position, video in enumerate(local):
            fold = (category_index + position + shift) % 5
            bins[fold].append(video)
    values = np.asarray(sorted(set(int(value) for value in videos)), dtype=np.int16)
    if sorted(len(group) for group in bins) != [3, 3, 3, 3, 3]:
        raise RuntimeError("Stratified video folds are not balanced")
    result = []
    for group in bins:
        validation = np.asarray(sorted(group), dtype=np.int16)
        training = values[~np.isin(values, validation)]
        result.append((training, validation))
    return tuple(result)


def run_nested(
    index,
    category_by_video: Mapping[int, str],
    duration_by_video: Mapping[int, float],
    archive: FoundationFeatureArchive,
    candidates: Sequence[ContentTemplateSpec],
    subject_splits,
    video_splits,
    *,
    inner_base_subject_folds: int,
    inner_content_subject_folds: int,
    inner_content_video_folds: int,
    minimum_inner_gain: float,
    offset_seconds: int,
    run_label: str,
) -> dict[str, object]:
    metadata_oof = np.full_like(index.targets, np.nan, dtype=np.float32)
    content_oof = np.full_like(index.targets, np.nan, dtype=np.float32)
    subject_fold_ids = np.full(len(index.targets), -1, dtype=np.int8)
    video_fold_ids = np.full(len(index.targets), -1, dtype=np.int8)
    records = []
    map_cache: dict[tuple[object, ...], np.ndarray] = {}
    total = len(subject_splits) * len(video_splits)
    position = 0
    for subject_fold, (training_subjects, validation_subjects) in enumerate(
        subject_splits
    ):
        for video_fold, (training_videos, validation_videos) in enumerate(
            video_splits
        ):
            position += 1
            print(
                f"[{run_label}] outer {position:02d}/{total} "
                f"S{subject_fold} V{video_fold}",
                flush=True,
            )
            if np.intersect1d(training_subjects, validation_subjects).size:
                raise RuntimeError("Participant leakage in outer split")
            if np.intersect1d(training_videos, validation_videos).size:
                raise RuntimeError("Video leakage in outer split")
            base_spec, base_inner_scores = select_category_phase_spec(
                index,
                training_subjects,
                training_videos,
                category_by_video,
                duration_by_video,
                inner_subject_folds=inner_base_subject_folds,
            )
            selected, selection = select_axis_specs(
                index,
                training_subjects,
                training_videos,
                category_by_video,
                duration_by_video,
                archive,
                base_spec,
                candidates,
                inner_subject_folds=inner_content_subject_folds,
                inner_video_folds=inner_content_video_folds,
                minimum_gain=minimum_inner_gain,
                offset_seconds=offset_seconds,
                map_cache=map_cache,
            )
            rows = np.flatnonzero(
                np.isin(index.subject_numbers, validation_subjects)
                & np.isin(index.videos, validation_videos)
            )
            base = predict_category_phase_prior(
                index,
                training_subjects,
                training_videos,
                rows,
                category_by_video,
                duration_by_video=duration_by_video,
                spec=base_spec,
            )
            curves = population_curves(index, training_subjects, training_videos)
            prediction_by_name = {}
            for spec in {selected[0].name: selected[0], selected[1].name: selected[1]}.values():
                prediction_by_name[spec.name] = predict_content_template_prior(
                    index,
                    rows,
                    base,
                    archive,
                    training_subjects,
                    training_videos,
                    category_by_video,
                    spec=spec,
                    offset_seconds=offset_seconds,
                    curves=curves,
                    map_cache=map_cache,
                )
            combined = base.copy()
            for axis, spec in enumerate(selected):
                combined[:, axis] = prediction_by_name[spec.name][:, axis]
            metadata_oof[rows] = base
            content_oof[rows] = combined
            subject_fold_ids[rows] = subject_fold
            video_fold_ids[rows] = video_fold
            records.append(
                {
                    "run_label": run_label,
                    "offset_seconds": int(offset_seconds),
                    "subject_fold": subject_fold,
                    "video_fold": video_fold,
                    "training_subjects": training_subjects.tolist(),
                    "validation_subjects": validation_subjects.tolist(),
                    "training_videos": training_videos.tolist(),
                    "validation_videos": validation_videos.tolist(),
                    "base_spec": base_spec_to_dict(base_spec),
                    "base_inner_mae_by_spec": base_inner_scores,
                    "content_selection": selection,
                    "validation_rows": int(len(rows)),
                }
            )
    if not np.isfinite(metadata_oof).all() or not np.isfinite(content_oof).all():
        raise RuntimeError(f"Incomplete OOF prediction for {run_label}")
    if np.any(subject_fold_ids < 0) or np.any(video_fold_ids < 0):
        raise RuntimeError(f"Incomplete fold assignment for {run_label}")
    return {
        "metadata": metadata_oof,
        "content": content_oof,
        "subject_fold_ids": subject_fold_ids,
        "video_fold_ids": video_fold_ids,
        "records": records,
        "map_cache_entries": len(map_cache),
    }


def model_metrics(index, prediction: np.ndarray) -> dict[str, float]:
    error = np.abs(
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(index.targets, dtype=np.float64)
    )
    participant = [
        float(error[index.subject_numbers == subject].mean())
        for subject in sorted(set(int(value) for value in index.subject_numbers))
    ]
    video = [
        float(error[index.videos == value].mean())
        for value in sorted(set(int(item) for item in index.videos))
    ]
    return {
        "sample_mae": float(error.mean()),
        "valence_mae": float(error[:, 0].mean()),
        "arousal_mae": float(error[:, 1].mean()),
        "participant_macro_mae": float(np.mean(participant)),
        "video_macro_mae": float(np.mean(video)),
        "trial_macro_mae": float(trial_macro_mae(index, prediction)),
    }


def gain_diagnostics(
    index,
    baseline: np.ndarray,
    comparator: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    subjects = sorted(set(int(value) for value in index.subject_numbers))
    videos = sorted(set(int(value) for value in index.videos))
    gain = np.empty((len(subjects), len(videos)), dtype=np.float64)
    for subject_index, subject in enumerate(subjects):
        for video_index, video in enumerate(videos):
            rows = (index.subject_numbers == subject) & (index.videos == video)
            gain[subject_index, video_index] = float(
                np.abs(baseline[rows] - index.targets[rows]).mean()
                - np.abs(comparator[rows] - index.targets[rows]).mean()
            )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(int(repeats), dtype=np.float64)
    for repeat in range(int(repeats)):
        sampled_subjects = rng.integers(0, len(subjects), len(subjects))
        sampled_videos = rng.integers(0, len(videos), len(videos))
        bootstrap[repeat] = gain[
            sampled_subjects[:, None], sampled_videos[None, :]
        ].mean()
    return {
        "trial_macro_mae_gain": float(gain.mean()),
        "participants_improved": int(np.sum(gain.mean(axis=1) > 0.0)),
        "participants_total": len(subjects),
        "videos_improved": int(np.sum(gain.mean(axis=0) > 0.0)),
        "videos_total": len(videos),
        "crossed_cluster_bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_repeats": int(repeats),
        "scope": (
            "participant/video sampling variation conditional on the primary "
            "nested OOF fit; refit variation is reported by repeated grouped CV"
        ),
    }


def fixed_ablation_predictions(
    index,
    primary: Mapping[str, object],
    archive: FoundationFeatureArchive,
    category_by_video: Mapping[int, str],
    visual_modality: str = "clip",
) -> dict[str, np.ndarray]:
    specs = {
        "uniform_phase": ContentTemplateSpec(
            float("inf"), 0.0, 0.0, 0.0, 1.0, visual_modality
        ),
        f"{visual_modality}_hard_phase": ContentTemplateSpec(
            0.0, 0.0, 0.0, 0.0, 1.0, visual_modality
        ),
        "ast_hard_phase": ContentTemplateSpec(
            0.0, 0.0, 0.0, 0.0, 1.0, "ast"
        ),
        f"{visual_modality}_hard_dtw": ContentTemplateSpec(
            0.0, 0.15, 0.0, 0.75, 0.80, visual_modality
        ),
    }
    if visual_modality in {"clip", "visual"}:
        joint_modality = f"{visual_modality}_ast"
        specs[f"{joint_modality}_hard_phase"] = ContentTemplateSpec(
            0.0, 0.0, 0.0, 0.0, 1.0, joint_modality
        )
    output = {
        name: np.full_like(index.targets, np.nan, dtype=np.float32)
        for name in specs
    }
    base = np.asarray(primary["metadata"], dtype=np.float32)
    map_cache: dict[tuple[object, ...], np.ndarray] = {}
    for record in primary["records"]:
        training_subjects = np.asarray(record["training_subjects"], dtype=np.int16)
        validation_subjects = np.asarray(record["validation_subjects"], dtype=np.int16)
        training_videos = np.asarray(record["training_videos"], dtype=np.int16)
        validation_videos = np.asarray(record["validation_videos"], dtype=np.int16)
        rows = np.flatnonzero(
            np.isin(index.subject_numbers, validation_subjects)
            & np.isin(index.videos, validation_videos)
        )
        curves = population_curves(index, training_subjects, training_videos)
        for name, spec in specs.items():
            output[name][rows] = predict_content_template_prior(
                index,
                rows,
                base[rows],
                archive,
                training_subjects,
                training_videos,
                category_by_video,
                spec=spec,
                offset_seconds=0,
                curves=curves,
                map_cache=map_cache,
            )
    for name, prediction in output.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Incomplete fixed ablation: {name}")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = resolve(args.data_root)
    manifest_path = resolve(args.stimulus_manifest)
    feature_path = resolve(args.foundation_features)
    output_dir = resolve(args.output_dir)
    prepare_output(output_dir, args.overwrite)
    if int(args.primary_offset) not in set(int(value) for value in args.offsets):
        raise ValueError("primary-offset must be included in offsets")
    if int(args.repeat_count) < 1:
        raise ValueError("repeat-count must be positive")

    started = time.perf_counter()
    candidates = candidate_specs(args.content_modality)
    run_manifest = {
        "schema_version": "merps-av-content-experiment-v1",
        "status": "running",
        "started_at_utc": utc_now(),
        "configuration": {
            "subject_folds": int(args.subject_folds),
            "video_folds": int(args.video_folds),
            "inner_base_subject_folds": int(args.inner_base_subject_folds),
            "inner_content_subject_folds": int(args.inner_content_subject_folds),
            "inner_content_video_folds": int(args.inner_content_video_folds),
            "minimum_inner_gain": float(args.minimum_inner_gain),
            "offsets": [int(value) for value in args.offsets],
            "primary_offset": int(args.primary_offset),
            "repeat_count": int(args.repeat_count),
            "seed": int(args.seed),
            "parallel_jobs": int(args.parallel_jobs),
            "candidate_count": len(candidates),
            "content_modality": str(args.content_modality),
        },
        "source_hashes": {
            str(manifest_path): file_sha256(manifest_path),
            str(feature_path): file_sha256(feature_path),
            "src/merps/av_content_prior.py": file_sha256(
                PROJECT_ROOT / "src" / "merps" / "av_content_prior.py"
            ),
            "scripts/run_av_content_prior.py": file_sha256(Path(__file__)),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    index = load_innovation_index(data_root)
    records = load_stimulus_manifest(manifest_path)
    archive = FoundationFeatureArchive.load(feature_path)
    category_by_video = emotion_by_video(records)
    duration_by_video = {
        int(record.video_id): float(record.reported_duration_seconds)
        for record in records
    }
    observed_videos = sorted(set(int(value) for value in index.videos))
    if observed_videos != sorted(archive.features_by_video):
        raise ValueError("Feature archive videos do not match label index")
    for record in records:
        if archive.annotation_lengths[record.video_id] != record.annotation_seconds:
            raise ValueError(f"Annotation length mismatch for video {record.video_id}")

    subjects = sorted(set(int(value) for value in index.subject_numbers))
    videos = observed_videos
    primary_subject_splits = round_robin_folds(subjects, args.subject_folds)
    primary_video_splits = round_robin_folds(videos, args.video_folds)
    task_specs: list[tuple[str, int, object, object, int, str]] = []
    for offset in [int(value) for value in args.offsets]:
        task_specs.append(
            (
                "offset",
                offset,
                primary_subject_splits,
                primary_video_splits,
                offset,
                f"offset_{offset:+d}_repeat_0",
            )
        )
    for repeat in range(1, int(args.repeat_count)):
        subject_splits = shuffled_subject_folds(
            subjects, args.subject_folds, args.seed + 1000 * repeat
        )
        video_splits = stratified_video_folds(
            videos,
            category_by_video,
            args.video_folds,
            args.seed + 1000 * repeat + 1,
        )
        task_specs.append(
            (
                "repeat",
                repeat,
                subject_splits,
                video_splits,
                int(args.primary_offset),
                f"offset_{args.primary_offset:+d}_repeat_{repeat}",
            )
        )

    def submit(executor, task):
        _, _, subject_splits, video_splits, offset, label = task
        arguments = (
            index,
            category_by_video,
            duration_by_video,
            archive,
            candidates,
            subject_splits,
            video_splits,
        )
        keywords = {
            "inner_base_subject_folds": args.inner_base_subject_folds,
            "inner_content_subject_folds": args.inner_content_subject_folds,
            "inner_content_video_folds": args.inner_content_video_folds,
            "minimum_inner_gain": args.minimum_inner_gain,
            "offset_seconds": offset,
            "run_label": label,
        }
        if executor is None:
            return run_nested(*arguments, **keywords)
        return executor.submit(run_nested, *arguments, **keywords)

    completed: dict[tuple[str, int], dict[str, object]] = {}
    jobs = max(1, min(int(args.parallel_jobs), len(task_specs)))
    if jobs == 1:
        for task in task_specs:
            completed[(task[0], task[1])] = submit(None, task)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_key = {
                submit(executor, task): (task[0], task[1])
                for task in task_specs
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                completed[key] = future.result()
                print(
                    f"[scheduler] completed {key[0]} {key[1]}",
                    flush=True,
                )

    offset_runs = {
        offset: completed[("offset", offset)]
        for offset in [int(value) for value in args.offsets]
    }
    primary = offset_runs[int(args.primary_offset)]
    repeat_runs = [primary] + [
        completed[("repeat", repeat)]
        for repeat in range(1, int(args.repeat_count))
    ]

    ablations = fixed_ablation_predictions(
        index,
        primary,
        archive,
        category_by_video,
        visual_modality=args.content_modality,
    )
    content_metric_name = f"nested_{args.content_modality}_content_prior"
    primary_metrics = {
        "metadata_temporal_prior": model_metrics(index, primary["metadata"]),
        content_metric_name: model_metrics(index, primary["content"]),
    }
    ablation_metrics = {
        name: model_metrics(index, prediction)
        for name, prediction in ablations.items()
    }
    offset_metrics = {
        str(offset): {
            "metadata": model_metrics(index, run["metadata"]),
            "content": model_metrics(index, run["content"]),
        }
        for offset, run in offset_runs.items()
    }
    repeat_metrics = [
        {
            "repeat": repeat,
            "metadata": model_metrics(index, run["metadata"]),
            "content": model_metrics(index, run["content"]),
            "gain": (
                model_metrics(index, run["metadata"])["trial_macro_mae"]
                - model_metrics(index, run["content"])["trial_macro_mae"]
            ),
        }
        for repeat, run in enumerate(repeat_runs)
    ]
    primary_gain = (
        primary_metrics["metadata_temporal_prior"]["trial_macro_mae"]
        - primary_metrics[content_metric_name]["trial_macro_mae"]
    )

    metric_rows = [
        {"model": name, **metrics}
        for name, metrics in {
            **primary_metrics,
            **{f"ablation_{name}": value for name, value in ablation_metrics.items()},
        }.items()
    ]
    write_rows(output_dir / "metrics.csv", metric_rows)
    write_rows(
        output_dir / "ablation_metrics.csv",
        [{"model": name, **metrics} for name, metrics in ablation_metrics.items()],
    )
    all_records = []
    for offset, run in offset_runs.items():
        all_records.extend(run["records"])
    for repeat, run in enumerate(repeat_runs[1:], start=1):
        all_records.extend(run["records"])
    write_json(output_dir / "fold_selections.json", all_records)

    arrays = {
        "sample_ids": index.sample_ids,
        "subject_numbers": index.subject_numbers,
        "videos": index.videos,
        "timestamps": index.timestamps,
        "targets": index.targets,
        "primary_metadata": primary["metadata"],
        "primary_content": primary["content"],
        "primary_subject_fold_ids": primary["subject_fold_ids"],
        "primary_video_fold_ids": primary["video_fold_ids"],
    }
    for offset, run in offset_runs.items():
        suffix = f"m{abs(offset)}" if offset < 0 else f"p{offset}"
        arrays[f"offset_{suffix}_content"] = run["content"]
    for repeat, run in enumerate(repeat_runs):
        arrays[f"repeat_{repeat}_content"] = run["content"]
        arrays[f"repeat_{repeat}_metadata"] = run["metadata"]
    for name, prediction in ablations.items():
        arrays[f"ablation_{name}"] = prediction
    np.savez_compressed(output_dir / "oof_predictions.npz", **arrays)

    selected_counts = {
        axis: dict(
            Counter(
                record["content_selection"]["axes"][axis]["selected"]["name"]
                for record in primary["records"]
            )
        )
        for axis in ("valence", "arousal")
    }
    content_mae_by_offset = {
        offset: metrics["content"]["trial_macro_mae"]
        for offset, metrics in (
            (int(key), value) for key, value in offset_metrics.items()
        )
    }
    results = {
        "schema_version": "merps-av-content-results-v1",
        "completed_at_utc": utc_now(),
        "deployment_target": "new participant watching an unseen but preavailable video",
        "interaction_requirement": "none from the target participant",
        "content_availability": (
            "the complete stimulus file is encoded before playback; online lookup "
            "uses a precomputed trajectory and does not require future physiology"
        ),
        "outer_exclusions": [
            "all labels from the validation participants",
            "all labels from the validation videos",
        ],
        "content_modality": str(args.content_modality),
        "feature_identity": {
            "schema_version": archive.schema_version,
            "archive_sha256": file_sha256(feature_path),
            "media_sha256": archive.media_sha256,
        },
        "candidate_count": len(candidates),
        "primary_metrics": primary_metrics,
        "primary_gain_trial_macro_mae": primary_gain,
        "primary_gain_diagnostics": gain_diagnostics(
            index,
            primary["metadata"],
            primary["content"],
            repeats=args.bootstrap_repeats,
            seed=args.seed,
        ),
        "selected_spec_counts": selected_counts,
        "offset_sensitivity": {
            "fixed_offsets_seconds": sorted(content_mae_by_offset),
            "metrics": offset_metrics,
            "content_trial_macro_mae_range": [
                float(min(content_mae_by_offset.values())),
                float(max(content_mae_by_offset.values())),
            ],
            "primary_offset_was_not_selected_on_outer_labels": True,
        },
        "repeated_grouped_cv": {
            "runs": repeat_metrics,
            "content_trial_macro_mae_mean": float(
                np.mean(
                    [
                        item["content"]["trial_macro_mae"]
                        for item in repeat_metrics
                    ]
                )
            ),
            "content_trial_macro_mae_std": float(
                np.std(
                    [
                        item["content"]["trial_macro_mae"]
                        for item in repeat_metrics
                    ],
                    ddof=1,
                )
                if len(repeat_metrics) > 1
                else 0.0
            ),
            "gain_mean": float(np.mean([item["gain"] for item in repeat_metrics])),
            "positive_gain_runs": int(
                np.sum([item["gain"] > 0.0 for item in repeat_metrics])
            ),
            "total_runs": len(repeat_metrics),
        },
        "fixed_diagnostic_ablations": ablation_metrics,
        "interpretation_boundary": [
            "The frozen feature encoders are label-free; trajectory templates use training labels.",
            "The complete target stimulus is available for preprocessing before playback.",
            "The primary estimate uses nested selection inside every outer training split.",
            "The compact search space was developed on this corpus and still needs external preregistered confirmation.",
            "Crossed bootstrap intervals are conditional on one fitted OOF run; repeated grouped CV captures split/refit variation.",
            "Offset runs are fixed sensitivity analyses and are not used to choose the reported primary offset.",
        ],
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
    print(json.dumps(primary_metrics, indent=2, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "primary_gain": primary_gain,
                "offset_mae_range": results["offset_sensitivity"][
                    "content_trial_macro_mae_range"
                ],
                "repeat_gain_mean": results["repeated_grouped_cv"]["gain_mean"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
