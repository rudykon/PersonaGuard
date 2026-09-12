#!/usr/bin/env python3
"""Test whether causal EEG/fNIRS adds value after a temporal content prior.

Every outer prediction holds out both its participant and its video.  Each
physiology axis has an independently selected modality/gate and may fall back
exactly to zero.  This is an algorithm experiment; the current base model is a
metadata-conditioned proxy until the identity-verified stimulus edits arrive.
"""

from __future__ import annotations

import argparse
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
    CategoryPhaseSpec,
    cross_fitted_category_phase_prior,
    emotion_by_video,
    load_stimulus_manifest,
    predict_category_phase_prior,
    round_robin_folds,
    trial_macro_mae,
)
from merps.innovation.data import load_innovation_index  # noqa: E402
from merps.physiology_residual import (  # noqa: E402
    causal_ema,
    fit_standardized_ridge_path,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_STIMULUS_MANIFEST = PROJECT_ROOT / "configs" / "stimuli" / "refed_15_videos.csv"
DEFAULT_CAUSAL_CACHE = PROJECT_ROOT / "data" / "innovation_cache_causal"
DEFAULT_CBRAMOD_CACHE = (
    PROJECT_ROOT / "data" / "innovation_cache_antialias" / "cbramod_causal"
)
DEFAULT_PRIOR_ARTIFACT = PROJECT_ROOT / "artifacts" / "content_prior_optimization"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "causal_physio_residual"
RIDGE_ALPHAS = (100.0, 1000.0, 10000.0, 100000.0)
GATES = (0.05, 0.10, 0.20, 0.40, 0.70, 1.00)
CAPS = (8.0, 16.0, 32.0)
EMA_DECAYS = (0.0, 0.5, 0.8)
MANAGED_FILES = {
    "metrics.json",
    "oof_predictions.npz",
    "results.json",
    "run_manifest.json",
    "selections.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stimulus-manifest", type=Path, default=DEFAULT_STIMULUS_MANIFEST)
    parser.add_argument("--causal-cache", type=Path, default=DEFAULT_CAUSAL_CACHE)
    parser.add_argument("--cbramod-cache", type=Path, default=DEFAULT_CBRAMOD_CACHE)
    parser.add_argument("--prior-artifact", type=Path, default=DEFAULT_PRIOR_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("eeg", "fnirs", "fusion", "cbramod", "cbramod_fnirs"),
        default=("eeg", "fnirs", "fusion"),
    )
    parser.add_argument("--subject-folds", type=int, default=5)
    parser.add_argument("--video-folds", type=int, default=5)
    parser.add_argument("--inner-subject-folds", type=int, default=2)
    parser.add_argument("--inner-video-folds", type=int, default=2)
    parser.add_argument("--minimum-inner-gain", type=float, default=0.10)
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


def load_features(
    cache: Path,
    cbramod_cache: Path,
    expected_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("causal") is not True:
        raise ValueError("The physiology feature cache is not certified causal")
    eeg = np.load(cache / "causal_eeg_pooled.npy", mmap_mode="r", allow_pickle=False)
    fnirs = np.load(cache / "causal_fnirs_pooled.npy", mmap_mode="r", allow_pickle=False)
    if eeg.shape != (expected_samples, 90) or fnirs.shape != (expected_samples, 180):
        raise ValueError(f"Unexpected causal feature shapes: {eeg.shape}, {fnirs.shape}")
    if not np.isfinite(eeg).all() or not np.isfinite(fnirs).all():
        raise ValueError("Causal physiology features contain non-finite values")
    cbramod_manifest_path = cbramod_cache / "manifest.json"
    cbramod_manifest = json.loads(cbramod_manifest_path.read_text(encoding="utf-8"))
    if (
        cbramod_manifest.get("status") != "complete"
        or cbramod_manifest.get("causal") is not True
    ):
        raise ValueError("The CBraMod embedding cache is not certified causal")
    cbramod = np.load(
        cbramod_cache / "pooled.npy", mmap_mode="r", allow_pickle=False
    )
    if cbramod.shape != (expected_samples, 400) or not np.isfinite(cbramod).all():
        raise ValueError(f"Unexpected causal CBraMod shape: {cbramod.shape}")
    manifest["cbramod"] = cbramod_manifest
    return eeg, fnirs, cbramod, manifest


def feature_block(
    modality: str,
    eeg: np.ndarray,
    fnirs: np.ndarray,
    cbramod: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    if modality == "eeg":
        return np.asarray(eeg[rows], dtype=np.float32)
    if modality == "fnirs":
        return np.asarray(fnirs[rows], dtype=np.float32)
    if modality == "fusion":
        return np.concatenate(
            [np.asarray(eeg[rows], dtype=np.float32), np.asarray(fnirs[rows], dtype=np.float32)],
            axis=1,
        )
    if modality == "cbramod":
        return np.asarray(cbramod[rows], dtype=np.float32)
    if modality == "cbramod_fnirs":
        return np.concatenate(
            [
                np.asarray(cbramod[rows], dtype=np.float32),
                np.asarray(fnirs[rows], dtype=np.float32),
            ],
            axis=1,
        )
    raise ValueError(modality)


def phase_spec_from_record(record: Mapping[str, object]) -> CategoryPhaseSpec:
    payload = record["selected_spec"]
    temperature = payload["duration_temperature"]
    return CategoryPhaseSpec(
        phase_weight=float(payload["phase_weight"]),
        duration_temperature=(
            float("inf") if temperature == "infinity" else float(temperature)
        ),
        smooth_radius=int(payload["smooth_radius"]),
    )


def load_prior_selections(path: Path) -> dict[tuple[int, int], Mapping[str, object]]:
    records = json.loads((path / "fold_selections.json").read_text(encoding="utf-8"))
    result = {
        (int(record["subject_fold"]), int(record["video_fold"])): record
        for record in records
    }
    if len(result) != len(records):
        raise ValueError("Duplicate prior selection fold")
    return result


def trial_axis_mae(index, rows: np.ndarray, prediction: np.ndarray, axis: int) -> float:
    rows = np.asarray(rows, dtype=np.int64)
    values = []
    for subject in sorted(set(int(value) for value in index.subject_numbers[rows])):
        for video in sorted(set(int(value) for value in index.videos[rows])):
            keep = (index.subject_numbers[rows] == subject) & (index.videos[rows] == video)
            if np.any(keep):
                values.append(
                    float(np.abs(prediction[keep, axis] - index.targets[rows[keep], axis]).mean())
                )
    return float(np.mean(values))


def trial_macro_weights(index, rows: np.ndarray) -> np.ndarray:
    """Return sample weights whose weighted MAE equals trial-macro MAE."""

    rows = np.asarray(rows, dtype=np.int64)
    weights = np.zeros(len(rows), dtype=np.float64)
    pairs = sorted(
        set(
            zip(
                (int(value) for value in index.subject_numbers[rows]),
                (int(value) for value in index.videos[rows]),
            )
        )
    )
    for subject, video in pairs:
        keep = (index.subject_numbers[rows] == subject) & (index.videos[rows] == video)
        weights[keep] = 1.0 / (len(pairs) * int(np.sum(keep)))
    if not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("Invalid trial-macro sample weights")
    return weights


def model_metrics(index, prediction: np.ndarray, prior: np.ndarray) -> dict[str, float]:
    error = np.abs(np.asarray(prediction, dtype=np.float64) - index.targets)
    participant = [
        float(error[index.subject_numbers == subject].mean())
        for subject in sorted(set(int(value) for value in index.subject_numbers))
    ]
    correction = np.asarray(prediction) - np.asarray(prior)
    return {
        "sample_mae": float(error.mean()),
        "valence_mae": float(error[:, 0].mean()),
        "arousal_mae": float(error[:, 1].mean()),
        "participant_macro_mae": float(np.mean(participant)),
        "trial_macro_mae": float(trial_macro_mae(index, prediction)),
        "mean_absolute_correction": float(np.abs(correction).mean()),
        "correction_p95": float(np.quantile(np.abs(correction), 0.95)),
    }


def inner_raw_predictions(
    index,
    eeg: np.ndarray,
    fnirs: np.ndarray,
    cbramod: np.ndarray,
    training_subjects: np.ndarray,
    training_videos: np.ndarray,
    category_by_video: Mapping[int, str],
    duration_by_video: Mapping[int, float],
    phase_spec: CategoryPhaseSpec,
    modalities: Sequence[str],
    inner_subject_folds: int,
    inner_video_folds: int,
) -> tuple[np.ndarray, dict[tuple[str, float], np.ndarray], np.ndarray]:
    source_rows = np.flatnonzero(
        np.isin(index.subject_numbers, training_subjects)
        & np.isin(index.videos, training_videos)
    )
    prior_oof = np.full((len(index.targets), 2), np.nan, dtype=np.float32)
    raw_oof = {
        (modality, alpha): np.full((len(index.targets), 2), np.nan, dtype=np.float32)
        for modality in modalities
        for alpha in RIDGE_ALPHAS
    }
    subject_splits = round_robin_folds(training_subjects, inner_subject_folds)
    video_splits = round_robin_folds(training_videos, inner_video_folds)
    for inner_training_subjects, inner_validation_subjects in subject_splits:
        for inner_training_videos, inner_validation_videos in video_splits:
            fit_rows = np.flatnonzero(
                np.isin(index.subject_numbers, inner_training_subjects)
                & np.isin(index.videos, inner_training_videos)
            )
            validation_rows = np.flatnonzero(
                np.isin(index.subject_numbers, inner_validation_subjects)
                & np.isin(index.videos, inner_validation_videos)
            )
            cross_rows, fit_prior = cross_fitted_category_phase_prior(
                index,
                inner_training_subjects,
                inner_training_videos,
                category_by_video,
                duration_by_video=duration_by_video,
                spec=phase_spec,
            )
            if not np.array_equal(cross_rows, fit_rows):
                raise RuntimeError("Cross-fitted prior rows do not match inner fit rows")
            validation_prior = predict_category_phase_prior(
                index,
                inner_training_subjects,
                inner_training_videos,
                validation_rows,
                category_by_video,
                duration_by_video=duration_by_video,
                spec=phase_spec,
            )
            prior_oof[validation_rows] = validation_prior
            residual_target = index.targets[fit_rows] - fit_prior
            for modality in modalities:
                models = fit_standardized_ridge_path(
                    feature_block(modality, eeg, fnirs, cbramod, fit_rows),
                    residual_target,
                    RIDGE_ALPHAS,
                )
                validation_features = feature_block(
                    modality, eeg, fnirs, cbramod, validation_rows
                )
                for alpha, model in models.items():
                    raw_oof[(modality, alpha)][validation_rows] = model.predict(
                        validation_features
                    )
    if not np.isfinite(prior_oof[source_rows]).all():
        raise RuntimeError("Incomplete inner prior OOF")
    for key, values in raw_oof.items():
        if not np.isfinite(values[source_rows]).all():
            raise RuntimeError(f"Incomplete inner physiology OOF for {key}")
    return prior_oof, raw_oof, source_rows


def select_axis_experts(
    index,
    source_rows: np.ndarray,
    prior_oof: np.ndarray,
    raw_oof: Mapping[tuple[str, float], np.ndarray],
    *,
    minimum_gain: float,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    local_prior = prior_oof[source_rows]
    local_target = index.targets[source_rows]
    subjects = index.subject_numbers[source_rows]
    videos = index.videos[source_rows]
    timestamps = index.timestamps[source_rows]
    weights = trial_macro_weights(index, source_rows)
    base_scores = {
        axis: float(weights @ np.abs(local_prior[:, axis] - local_target[:, axis]))
        for axis in (0, 1)
    }
    choices: list[dict[str, object]] = []
    best_scores: dict[str, float] = {}
    for axis, axis_name in enumerate(("valence", "arousal")):
        best_key = (base_scores[axis], 0, "disabled")
        best = {
            "axis": axis_name,
            "modality": "disabled",
            "alpha": 0.0,
            "gate": 0.0,
            "cap": 0.0,
            "ema_decay": 0.0,
            "inner_trial_macro_mae": base_scores[axis],
            "inner_prior_trial_macro_mae": base_scores[axis],
            "inner_gain": 0.0,
        }
        for (modality, alpha), raw_full in raw_oof.items():
            local_raw = raw_full[source_rows]
            for decay in EMA_DECAYS:
                smoothed = causal_ema(
                    local_raw, subjects, videos, timestamps, decay
                )
                for cap in CAPS:
                    for gate in GATES:
                        correction = np.clip(gate * smoothed[:, axis], -cap, cap)
                        candidate_axis = np.clip(
                            local_prior[:, axis] + correction, 1.0, 255.0
                        )
                        score = float(
                            weights
                            @ np.abs(candidate_axis - local_target[:, axis])
                        )
                        name = f"{modality}_a{alpha:g}_g{gate:g}_c{cap:g}_ema{decay:g}"
                        key = (score, 1, name)
                        if key < best_key:
                            best_key = key
                            best = {
                                "axis": axis_name,
                                "modality": modality,
                                "alpha": float(alpha),
                                "gate": float(gate),
                                "cap": float(cap),
                                "ema_decay": float(decay),
                                "inner_trial_macro_mae": float(score),
                                "inner_prior_trial_macro_mae": float(base_scores[axis]),
                                "inner_gain": float(base_scores[axis] - score),
                            }
        if float(best["inner_gain"]) < float(minimum_gain):
            best = {
                "axis": axis_name,
                "modality": "disabled",
                "alpha": 0.0,
                "gate": 0.0,
                "cap": 0.0,
                "ema_decay": 0.0,
                "inner_trial_macro_mae": float(base_scores[axis]),
                "inner_prior_trial_macro_mae": float(base_scores[axis]),
                "inner_gain": 0.0,
            }
        choices.append(best)
        best_scores[axis_name] = float(best["inner_trial_macro_mae"])
    return choices, best_scores


def final_residual_prediction(
    index,
    eeg: np.ndarray,
    fnirs: np.ndarray,
    cbramod: np.ndarray,
    training_subjects: np.ndarray,
    training_videos: np.ndarray,
    validation_rows: np.ndarray,
    validation_prior: np.ndarray,
    category_by_video: Mapping[int, str],
    duration_by_video: Mapping[int, float],
    phase_spec: CategoryPhaseSpec,
    choices: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    prediction = validation_prior.copy()
    raw_output = np.zeros_like(prediction)
    enabled = [choice for choice in choices if choice["modality"] != "disabled"]
    if not enabled:
        return prediction, raw_output
    fit_rows, fit_prior = cross_fitted_category_phase_prior(
        index,
        training_subjects,
        training_videos,
        category_by_video,
        duration_by_video=duration_by_video,
        spec=phase_spec,
    )
    residual_target = index.targets[fit_rows] - fit_prior
    fitted: dict[tuple[str, float], object] = {}
    for choice in enabled:
        key = (str(choice["modality"]), float(choice["alpha"]))
        if key not in fitted:
            fitted[key] = fit_standardized_ridge_path(
                feature_block(key[0], eeg, fnirs, cbramod, fit_rows),
                residual_target,
                [key[1]],
            )[key[1]]
    for axis, choice in enumerate(choices):
        if choice["modality"] == "disabled":
            continue
        key = (str(choice["modality"]), float(choice["alpha"]))
        raw = fitted[key].predict(
            feature_block(key[0], eeg, fnirs, cbramod, validation_rows)
        )
        smoothed = causal_ema(
            raw,
            index.subject_numbers[validation_rows],
            index.videos[validation_rows],
            index.timestamps[validation_rows],
            float(choice["ema_decay"]),
        )
        raw_output[:, axis] = smoothed[:, axis]
        correction = np.clip(
            float(choice["gate"]) * smoothed[:, axis],
            -float(choice["cap"]),
            float(choice["cap"]),
        )
        prediction[:, axis] = np.clip(
            validation_prior[:, axis] + correction, 1.0, 255.0
        )
    return prediction.astype(np.float32), raw_output.astype(np.float32)


def gain_uncertainty(
    index, prior: np.ndarray, prediction: np.ndarray, repeats: int, seed: int
) -> dict[str, object]:
    subjects = sorted(set(int(value) for value in index.subject_numbers))
    videos = sorted(set(int(value) for value in index.videos))
    gain = np.empty((len(subjects), len(videos)), dtype=np.float64)
    for i, subject in enumerate(subjects):
        for j, video in enumerate(videos):
            rows = (index.subject_numbers == subject) & (index.videos == video)
            gain[i, j] = float(
                np.abs(prior[rows] - index.targets[rows]).mean()
                - np.abs(prediction[rows] - index.targets[rows]).mean()
            )
    rng = np.random.default_rng(seed)
    participant = np.empty(repeats, dtype=np.float64)
    crossed = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        si = rng.integers(0, len(subjects), len(subjects))
        vi = rng.integers(0, len(videos), len(videos))
        participant[repeat] = gain.mean(axis=1)[si].mean()
        crossed[repeat] = gain[si[:, None], vi[None, :]].mean()
    interval = lambda values: [
        float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
    ]
    return {
        "trial_macro_mae_gain": float(gain.mean()),
        "positive_means_physiology_is_better": True,
        "participant_cluster_bootstrap_ci95": interval(participant),
        "participant_video_crossed_bootstrap_ci95": interval(crossed),
        "participants_improved": int(np.sum(gain.mean(axis=1) > 0.0)),
        "videos_improved": int(np.sum(gain.mean(axis=0) > 0.0)),
        "resamples": int(repeats),
        "uncertainty_scope": "conditional on fitted OOF predictions; no full refit propagation",
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = resolve(args.data_root)
    stimulus_manifest_path = resolve(args.stimulus_manifest)
    causal_cache = resolve(args.causal_cache)
    cbramod_cache = resolve(args.cbramod_cache)
    prior_artifact = resolve(args.prior_artifact)
    output_dir = resolve(args.output_dir)
    prepare_output(output_dir, args.overwrite)
    started = time.perf_counter()

    run_manifest = {
        "schema_version": "merps-causal-physio-residual-run-v1",
        "status": "running",
        "started_at_utc": utc_now(),
        "configuration": {
            "deployment": "new participant and unseen video",
            "modalities": list(args.modalities),
            "subject_folds": int(args.subject_folds),
            "video_folds": int(args.video_folds),
            "inner_subject_folds": int(args.inner_subject_folds),
            "inner_video_folds": int(args.inner_video_folds),
            "ridge_alphas": list(RIDGE_ALPHAS),
            "gates": list(GATES),
            "caps": list(CAPS),
            "ema_decays": list(EMA_DECAYS),
            "minimum_inner_gain": float(args.minimum_inner_gain),
            "seed": int(args.seed),
        },
        "inputs": {
            "data_root": project_path(data_root),
            "stimulus_manifest": project_path(stimulus_manifest_path),
            "causal_cache": project_path(causal_cache),
            "cbramod_cache": project_path(cbramod_cache),
            "prior_artifact": project_path(prior_artifact),
        },
        "source_hashes": {
            "src/merps/content_prior.py": file_sha256(PROJECT_ROOT / "src/merps/content_prior.py"),
            "src/merps/physiology_residual.py": file_sha256(PROJECT_ROOT / "src/merps/physiology_residual.py"),
            "scripts/run_causal_physio_residual.py": file_sha256(Path(__file__)),
            project_path(stimulus_manifest_path): file_sha256(stimulus_manifest_path),
            project_path(causal_cache / "manifest.json"): file_sha256(causal_cache / "manifest.json"),
            project_path(cbramod_cache / "manifest.json"): file_sha256(cbramod_cache / "manifest.json"),
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    index = load_innovation_index(data_root)
    records = load_stimulus_manifest(stimulus_manifest_path)
    category_by_video = emotion_by_video(records)
    duration_by_video = {
        int(record.video_id): float(record.reported_duration_seconds) for record in records
    }
    eeg, fnirs, cbramod, causal_manifest = load_features(
        causal_cache, cbramod_cache, len(index.targets)
    )
    prior_selections = load_prior_selections(prior_artifact)
    subjects = sorted(set(int(value) for value in index.subject_numbers))
    videos = sorted(set(int(value) for value in index.videos))
    subject_splits = round_robin_folds(subjects, args.subject_folds)
    video_splits = round_robin_folds(videos, args.video_folds)

    prior_oof = np.full_like(index.targets, np.nan, dtype=np.float32)
    prediction_oof = np.full_like(index.targets, np.nan, dtype=np.float32)
    raw_oof = np.zeros_like(index.targets, dtype=np.float32)
    subject_fold_ids = np.full(len(index.targets), -1, dtype=np.int8)
    video_fold_ids = np.full(len(index.targets), -1, dtype=np.int8)
    selections: list[dict[str, object]] = []
    total_cells = len(subject_splits) * len(video_splits)
    completed_cells = 0
    for subject_fold, (training_subjects, validation_subjects) in enumerate(subject_splits):
        for video_fold, (training_videos, validation_videos) in enumerate(video_splits):
            selection_record = prior_selections[(subject_fold, video_fold)]
            if set(selection_record["training_subjects"]) != set(training_subjects.tolist()):
                raise ValueError("Prior artifact participant split does not match")
            if set(selection_record["training_videos"]) != set(training_videos.tolist()):
                raise ValueError("Prior artifact video split does not match")
            phase_spec = phase_spec_from_record(selection_record)
            validation_rows = np.flatnonzero(
                np.isin(index.subject_numbers, validation_subjects)
                & np.isin(index.videos, validation_videos)
            )
            validation_prior = predict_category_phase_prior(
                index,
                training_subjects,
                training_videos,
                validation_rows,
                category_by_video,
                duration_by_video=duration_by_video,
                spec=phase_spec,
            )
            inner_prior, inner_raw, source_rows = inner_raw_predictions(
                index,
                eeg,
                fnirs,
                cbramod,
                training_subjects,
                training_videos,
                category_by_video,
                duration_by_video,
                phase_spec,
                args.modalities,
                args.inner_subject_folds,
                args.inner_video_folds,
            )
            axis_choices, inner_scores = select_axis_experts(
                index,
                source_rows,
                inner_prior,
                inner_raw,
                minimum_gain=args.minimum_inner_gain,
            )
            prediction, raw = final_residual_prediction(
                index,
                eeg,
                fnirs,
                cbramod,
                training_subjects,
                training_videos,
                validation_rows,
                validation_prior,
                category_by_video,
                duration_by_video,
                phase_spec,
                axis_choices,
            )
            prior_oof[validation_rows] = validation_prior
            prediction_oof[validation_rows] = prediction
            raw_oof[validation_rows] = raw
            subject_fold_ids[validation_rows] = subject_fold
            video_fold_ids[validation_rows] = video_fold
            selections.append(
                {
                    "subject_fold": subject_fold,
                    "video_fold": video_fold,
                    "training_subjects": training_subjects.tolist(),
                    "validation_subjects": validation_subjects.tolist(),
                    "training_videos": training_videos.tolist(),
                    "validation_videos": validation_videos.tolist(),
                    "content_prior_spec": selection_record["selected_spec"],
                    "axis_choices": axis_choices,
                    "inner_axis_scores": inner_scores,
                    "validation_rows": int(len(validation_rows)),
                }
            )
            completed_cells += 1
            print(
                f"[causal-residual] {completed_cells}/{total_cells} "
                f"subject_fold={subject_fold} video_fold={video_fold} "
                f"choices={[choice['modality'] for choice in axis_choices]}",
                flush=True,
            )

    if not np.isfinite(prior_oof).all() or not np.isfinite(prediction_oof).all():
        raise RuntimeError("Incomplete outer OOF prediction")
    if np.any(subject_fold_ids < 0) or np.any(video_fold_ids < 0):
        raise RuntimeError("Incomplete outer fold assignment")
    metrics = {
        "metadata_content_prior": model_metrics(index, prior_oof, prior_oof),
        "content_plus_causal_physiology": model_metrics(index, prediction_oof, prior_oof),
    }
    uncertainty = gain_uncertainty(
        index, prior_oof, prediction_oof, args.bootstrap_repeats, args.seed
    )
    enabled = Counter(
        f"{choice['axis']}:{choice['modality']}"
        for record in selections
        for choice in record["axis_choices"]
    )
    results = {
        "schema_version": "merps-causal-physio-residual-results-v1",
        "completed_at_utc": utc_now(),
        "deployment_target": "new participant and unseen video",
        "metrics": metrics,
        "gain_uncertainty": uncertainty,
        "axis_choice_counts": dict(sorted(enabled.items())),
        "outer_cells": int(total_cells),
        "causal_feature_contract": causal_manifest,
        "leakage_controls": [
            "outer participant groups are disjoint",
            "outer video groups are disjoint",
            "training residual priors exclude each row's participant and video",
            "physiology uses t-2,t-1,t only and EMA is causal within trial",
            "axis gates are selected by inner participant-by-video OOF and can be zero",
        ],
        "interpretation_boundary": [
            "The base is metadata-conditioned, not yet extracted audio-visual content.",
            "The target emotion category is known from the experiment design.",
            "This run estimates incremental causal physiology value only.",
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        sample_ids=index.sample_ids,
        targets=index.targets,
        metadata_content_prior=prior_oof,
        content_plus_causal_physiology=prediction_oof,
        raw_causal_physiology_residual=raw_oof,
        subject_fold_ids=subject_fold_ids,
        video_fold_ids=video_fold_ids,
    )
    write_json(output_dir / "selections.json", selections)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "results.json", results)
    run_manifest.update(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "elapsed_seconds": results["elapsed_seconds"],
            "generated_files": [
                "metrics.json", "oof_predictions.npz", "results.json", "selections.json"
            ],
        }
    )
    write_json(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(json.dumps(uncertainty, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
