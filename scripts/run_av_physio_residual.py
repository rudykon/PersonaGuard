#!/usr/bin/env python3
"""Test causal physiology residuals after the nested frozen-content prior."""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merps.av_content_prior import (  # noqa: E402
    ContentTemplateSpec,
    FoundationFeatureArchive,
    cross_fitted_content_prior,
    population_curves,
    predict_content_template_prior,
)
from merps.content_prior import (  # noqa: E402
    CategoryPhaseSpec,
    emotion_by_video,
    load_stimulus_manifest,
    predict_category_phase_prior,
    round_robin_folds,
)
from merps.innovation.data import load_innovation_index  # noqa: E402
from merps.physiology_residual import (  # noqa: E402
    causal_ema,
    fit_weighted_standardized_ridge_path,
)
from run_causal_physio_residual import (  # noqa: E402
    gain_uncertainty,
    load_features,
    model_metrics,
    prepare_output,
    project_path,
    trial_macro_weights,
    write_json,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_STIMULUS_MANIFEST = PROJECT_ROOT / "configs" / "stimuli" / "refed_15_videos.csv"
DEFAULT_FOUNDATION_FEATURES = (
    PROJECT_ROOT / "data" / "stimuli" / "features" / "foundation_av.npz"
)
DEFAULT_CONTENT_ARTIFACT = PROJECT_ROOT / "artifacts" / "av_content_prior"
DEFAULT_CAUSAL_CACHE = PROJECT_ROOT / "data" / "innovation_cache_causal"
DEFAULT_CBRAMOD_CACHE = (
    PROJECT_ROOT / "data" / "innovation_cache_antialias" / "cbramod_causal"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "av_causal_physio_residual"
RIDGE_ALPHAS = (100.0, 1000.0, 10000.0, 100000.0)
GATES = (0.02, 0.05, 0.10, 0.20, 0.40)
CAPS = (8.0, 16.0, 32.0)
EMA_DECAYS = (0.0, 0.5, 0.8, 0.95)
GAIN_THRESHOLDS = (0.0, 0.05, 0.10, 0.20, 0.50)
PRIMARY_GAIN_THRESHOLD = 0.10
PRIMARY_RUN_LABEL = "offset_+0_repeat_0"
MODALITY_CHOICES = (
    "eeg",
    "fnirs",
    "fnirs_lag3",
    "fnirs_lag6",
    "fnirs_lag9",
    "fnirs_mean5",
    "fnirs_mean10",
    "fusion",
    "cbramod",
    "cbramod_fnirs",
    "cbramod_fnirs_lag6",
    "cbramod_fnirs_lag9",
    "cbramod_fnirs_mean10",
    "cbramod_latest",
    "cbramod_token_delta",
    "cbramod_temporal",
    "cbramod_token_stats",
    "cbramod_temporal_fnirs",
)
CBRAMOD_TOKEN_MODALITIES = frozenset(
    {
        "cbramod_latest", "cbramod_token_delta", "cbramod_temporal",
        "cbramod_token_stats", "cbramod_temporal_fnirs",
    }
)



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--stimulus-manifest", type=Path, default=DEFAULT_STIMULUS_MANIFEST
    )
    parser.add_argument(
        "--foundation-features", type=Path, default=DEFAULT_FOUNDATION_FEATURES
    )
    parser.add_argument(
        "--content-artifact", type=Path, default=DEFAULT_CONTENT_ARTIFACT
    )
    parser.add_argument("--causal-cache", type=Path, default=DEFAULT_CAUSAL_CACHE)
    parser.add_argument("--cbramod-cache", type=Path, default=DEFAULT_CBRAMOD_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=MODALITY_CHOICES,
        default=("eeg", "fnirs", "fusion", "cbramod", "cbramod_fnirs"),
    )
    parser.add_argument("--inner-subject-folds", type=int, default=2)
    parser.add_argument("--inner-video-folds", type=int, default=2)
    parser.add_argument(
        "--gain-thresholds",
        nargs="+",
        type=float,
        default=GAIN_THRESHOLDS,
    )
    parser.add_argument(
        "--primary-gain-threshold",
        type=float,
        default=PRIMARY_GAIN_THRESHOLD,
    )
    parser.add_argument("--parallel-jobs", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
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


def threshold_key(value: float) -> str:
    return f"gain_{float(value):g}"


def phase_spec_from_record(record: Mapping[str, object]) -> CategoryPhaseSpec:
    payload = record["base_spec"]
    temperature = payload["duration_temperature"]
    return CategoryPhaseSpec(
        phase_weight=float(payload["phase_weight"]),
        duration_temperature=(
            float("inf") if temperature == "infinity" else float(temperature)
        ),
        smooth_radius=int(payload["smooth_radius"]),
    )


def content_spec_from_payload(payload: Mapping[str, object]) -> ContentTemplateSpec:
    temperature = payload["source_temperature"]
    return ContentTemplateSpec(
        source_temperature=(
            float("inf") if temperature == "infinity" else float(temperature)
        ),
        warp_band=float(payload["warp_band"]),
        phase_penalty=float(payload["phase_penalty"]),
        warp_gate=float(payload["warp_gate"]),
        content_gate=float(payload["content_gate"]),
        modality=str(payload["modality"]),
    )


def axis_specs_from_record(
    record: Mapping[str, object],
) -> tuple[ContentTemplateSpec, ContentTemplateSpec]:
    axes = record["content_selection"]["axes"]
    return (
        content_spec_from_payload(axes["valence"]["selected"]),
        content_spec_from_payload(axes["arousal"]["selected"]),
    )


def load_primary_fold_records(
    content_artifact: Path,
) -> dict[tuple[int, int], Mapping[str, object]]:
    records = json.loads(
        (content_artifact / "fold_selections.json").read_text(encoding="utf-8")
    )
    selected = [
        record for record in records if record["run_label"] == PRIMARY_RUN_LABEL
    ]
    result = {
        (int(record["subject_fold"]), int(record["video_fold"])): record
        for record in selected
    }
    if len(selected) != 25 or len(result) != 25:
        raise ValueError(
            f"Expected 25 unique primary content folds, found {len(selected)}"
        )
    return result


def predict_combined_content_prior(
    index,
    rows: np.ndarray,
    archive: FoundationFeatureArchive,
    source_subjects: np.ndarray,
    source_videos: np.ndarray,
    category_by_video: Mapping[int, str],
    duration_by_video: Mapping[int, float],
    base_spec: CategoryPhaseSpec,
    axis_specs: Sequence[ContentTemplateSpec],
    *,
    map_cache: dict[tuple[object, ...], np.ndarray],
) -> np.ndarray:
    metadata = predict_category_phase_prior(
        index,
        source_subjects,
        source_videos,
        rows,
        category_by_video,
        duration_by_video=duration_by_video,
        spec=base_spec,
    )
    curves = population_curves(index, source_subjects, source_videos)
    predictions: dict[str, np.ndarray] = {}
    for spec in {item.name: item for item in axis_specs}.values():
        predictions[spec.name] = predict_content_template_prior(
            index,
            rows,
            metadata,
            archive,
            source_subjects,
            source_videos,
            category_by_video,
            spec=spec,
            offset_seconds=0,
            curves=curves,
            map_cache=map_cache,
        )
    combined = metadata.copy()
    for axis, spec in enumerate(axis_specs):
        combined[:, axis] = predictions[spec.name][:, axis]
    return combined


def causal_shift_view(
    values: np.ndarray,
    index,
    lag_seconds: int,
) -> np.ndarray:
    """Shift features into the past and replicate only the trial-start edge."""

    source = np.asarray(values, dtype=np.float32)
    lag = int(lag_seconds)
    if source.ndim != 2 or len(source) != len(index.targets):
        raise ValueError("Causal shift input does not match the sample index")
    if lag < 0:
        raise ValueError("lag_seconds must be non-negative")
    output = np.empty_like(source)
    pairs = sorted(
        set(
            zip(
                (int(value) for value in index.subject_numbers),
                (int(value) for value in index.videos),
            )
        )
    )
    for subject, video in pairs:
        rows = np.flatnonzero(
            (index.subject_numbers == subject) & (index.videos == video)
        )
        rows = rows[np.argsort(index.timestamps[rows])]
        positions = np.maximum(np.arange(len(rows)) - lag, 0)
        output[rows] = source[rows[positions]]
    return output


def causal_mean_view(
    values: np.ndarray,
    index,
    window_seconds: int,
) -> np.ndarray:
    """Compute a past-only rolling mean independently inside each trial."""

    source = np.asarray(values, dtype=np.float32)
    window = int(window_seconds)
    if source.ndim != 2 or len(source) != len(index.targets):
        raise ValueError("Causal mean input does not match the sample index")
    if window < 1:
        raise ValueError("window_seconds must be positive")
    output = np.empty_like(source)
    pairs = sorted(
        set(
            zip(
                (int(value) for value in index.subject_numbers),
                (int(value) for value in index.videos),
            )
        )
    )
    for subject, video in pairs:
        rows = np.flatnonzero(
            (index.subject_numbers == subject) & (index.videos == video)
        )
        rows = rows[np.argsort(index.timestamps[rows])]
        local = np.asarray(source[rows], dtype=np.float64)
        cumulative = np.concatenate(
            [np.zeros((1, local.shape[1])), np.cumsum(local, axis=0)],
            axis=0,
        )
        end = np.arange(1, len(rows) + 1)
        start = np.maximum(end - window, 0)
        output[rows] = (
            (cumulative[end] - cumulative[start])
            / (end - start)[:, None]
        ).astype(np.float32)
    return output


def load_cbramod_patch_tokens(
    cache: Path, expected_samples: int
) -> np.ndarray:
    """Load the certified causal t-3..t CBraMod tokens by memory map."""
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    expected_shape = (int(expected_samples), 4, 200)
    if manifest.get("causal") is not True or tuple(manifest.get("patch_tokens", ())) != expected_shape:
        raise ValueError("CBraMod patch-token manifest is not complete and causal")
    tokens = np.load(cache / "patch_tokens.npy", mmap_mode="r", allow_pickle=False)
    if tokens.shape != expected_shape or not np.isfinite(tokens).all():
        raise ValueError(f"Unexpected causal CBraMod token shape: {tokens.shape}")
    return tokens


def build_feature_views(
    index,
    eeg: np.ndarray,
    fnirs: np.ndarray,
    cbramod: np.ndarray,
    modalities: Sequence[str],
    cbramod_tokens: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Materialize only requested label-free causal physiology views."""

    requested = set(str(value) for value in modalities)
    views = {
        "eeg": eeg,
        "fnirs": fnirs,
        "cbramod": cbramod,
    }
    if requested & CBRAMOD_TOKEN_MODALITIES:
        if cbramod_tokens is None:
            raise ValueError("Requested a CBraMod token view without patch tokens")
        if cbramod_tokens.shape != (len(index.targets), 4, 200):
            raise ValueError(
                f"Unexpected CBraMod patch-token shape: {cbramod_tokens.shape}"
            )
        views["cbramod_patch_tokens"] = cbramod_tokens

    for lag in (3, 6, 9):
        name = f"fnirs_lag{lag}"
        if name in requested or f"cbramod_fnirs_lag{lag}" in requested:
            views[name] = causal_shift_view(fnirs, index, lag)
    for window in (5, 10):
        name = f"fnirs_mean{window}"
        if name in requested or f"cbramod_fnirs_mean{window}" in requested:
            views[name] = causal_mean_view(fnirs, index, window)
    return views


def feature_view(
    modality: str,
    views: Mapping[str, np.ndarray],
    rows: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if modality in CBRAMOD_TOKEN_MODALITIES:
        if "cbramod_patch_tokens" not in views:
            raise KeyError("Missing causal CBraMod patch tokens")
        tokens = np.asarray(
            views["cbramod_patch_tokens"][rows], dtype=np.float32
        )
        latest = tokens[:, -1, :]
        delta = latest - tokens[:, 0, :]
        if modality == "cbramod_latest":
            return latest
        if modality == "cbramod_token_delta":
            return delta
        if modality == "cbramod_temporal":
            return np.concatenate([latest, delta], axis=1)
        if modality == "cbramod_token_stats":
            return np.concatenate(
                [tokens.mean(axis=1), tokens.std(axis=1)], axis=1
            )
        if modality == "cbramod_temporal_fnirs":
            return np.concatenate(
                [latest, delta, np.asarray(views["fnirs"][rows], dtype=np.float32)],
                axis=1,
            )
        raise RuntimeError(f"Unhandled CBraMod token modality: {modality}")
    if modality in views:
        return np.asarray(views[modality][rows], dtype=np.float32)
    if modality == "fusion":
        pieces = (views["eeg"], views["fnirs"])
    elif modality == "cbramod_fnirs":
        pieces = (views["cbramod"], views["fnirs"])
    elif modality.startswith("cbramod_fnirs_"):
        fnirs_name = modality.removeprefix("cbramod_")
        if fnirs_name not in views:
            raise KeyError(f"Missing derived feature view: {fnirs_name}")
        pieces = (views["cbramod"], views[fnirs_name])
    else:
        raise ValueError(f"Unsupported physiology modality: {modality}")
    return np.concatenate(
        [np.asarray(piece[rows], dtype=np.float32) for piece in pieces],
        axis=1,
    )


def inner_raw_predictions(
    index,
    views: Mapping[str, np.ndarray],
    archive: FoundationFeatureArchive,
    training_subjects: np.ndarray,
    training_videos: np.ndarray,
    category_by_video: Mapping[int, str],
    duration_by_video: Mapping[int, float],
    base_spec: CategoryPhaseSpec,
    axis_specs: Sequence[ContentTemplateSpec],
    modalities: Sequence[str],
    inner_subject_folds: int,
    inner_video_folds: int,
    *,
    map_cache: dict[tuple[object, ...], np.ndarray],
) -> tuple[np.ndarray, dict[tuple[str, float], np.ndarray], np.ndarray]:
    source_rows = np.flatnonzero(
        np.isin(index.subject_numbers, training_subjects)
        & np.isin(index.videos, training_videos)
    )
    prior_oof = np.full_like(index.targets, np.nan, dtype=np.float32)
    raw_oof = {
        (modality, alpha): np.full_like(
            index.targets, np.nan, dtype=np.float32
        )
        for modality in modalities
        for alpha in RIDGE_ALPHAS
    }
    for inner_training_subjects, inner_validation_subjects in round_robin_folds(
        training_subjects, inner_subject_folds
    ):
        for inner_training_videos, inner_validation_videos in round_robin_folds(
            training_videos, inner_video_folds
        ):
            fit_rows, fit_prior = cross_fitted_content_prior(
                index,
                inner_training_subjects,
                inner_training_videos,
                category_by_video,
                archive,
                base_spec,
                axis_specs,
                duration_by_video=duration_by_video,
                offset_seconds=0,
                map_cache=map_cache,
            )
            validation_rows = np.flatnonzero(
                np.isin(index.subject_numbers, inner_validation_subjects)
                & np.isin(index.videos, inner_validation_videos)
            )
            validation_prior = predict_combined_content_prior(
                index,
                validation_rows,
                archive,
                inner_training_subjects,
                inner_training_videos,
                category_by_video,
                duration_by_video,
                base_spec,
                axis_specs,
                map_cache=map_cache,
            )
            prior_oof[validation_rows] = validation_prior
            residual_target = index.targets[fit_rows] - fit_prior
            fit_weights = trial_macro_weights(index, fit_rows)
            for modality in modalities:
                models = fit_weighted_standardized_ridge_path(
                    feature_view(modality, views, fit_rows),
                    residual_target,
                    RIDGE_ALPHAS,
                    fit_weights,
                )
                validation_features = feature_view(modality, views, validation_rows)
                for alpha, model in models.items():
                    raw_oof[(modality, alpha)][validation_rows] = (
                        model.predict(validation_features)
                    )
    if not np.isfinite(prior_oof[source_rows]).all():
        raise RuntimeError("Incomplete inner content prior OOF")
    for key, values in raw_oof.items():
        if not np.isfinite(values[source_rows]).all():
            raise RuntimeError(f"Incomplete inner physiology OOF for {key}")
    return prior_oof, raw_oof, source_rows


def best_axis_candidates(
    index,
    source_rows: np.ndarray,
    prior_oof: np.ndarray,
    raw_oof: Mapping[tuple[str, float], np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    local_prior = prior_oof[source_rows]
    local_target = index.targets[source_rows]
    subjects = index.subject_numbers[source_rows]
    videos = index.videos[source_rows]
    timestamps = index.timestamps[source_rows]
    weights = trial_macro_weights(index, source_rows)
    base_scores = {
        axis: float(
            weights
            @ np.abs(local_prior[:, axis] - local_target[:, axis])
        )
        for axis in (0, 1)
    }
    best: list[dict[str, object]] = []
    for axis, axis_name in enumerate(("valence", "arousal")):
        winner = {
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
        winner_key = (base_scores[axis], 0, "disabled")
        for (modality, alpha), raw_full in raw_oof.items():
            local_raw = raw_full[source_rows]
            for decay in EMA_DECAYS:
                smoothed = causal_ema(
                    local_raw, subjects, videos, timestamps, decay
                )
                for cap in CAPS:
                    for gate in GATES:
                        candidate = np.clip(
                            local_prior[:, axis]
                            + np.clip(
                                gate * smoothed[:, axis], -cap, cap
                            ),
                            1.0,
                            255.0,
                        )
                        score = float(
                            weights @ np.abs(candidate - local_target[:, axis])
                        )
                        name = (
                            f"{modality}_a{alpha:g}_g{gate:g}"
                            f"_c{cap:g}_ema{decay:g}"
                        )
                        key = (score, 1, name)
                        if key < winner_key:
                            winner_key = key
                            winner = {
                                "axis": axis_name,
                                "modality": modality,
                                "alpha": float(alpha),
                                "gate": float(gate),
                                "cap": float(cap),
                                "ema_decay": float(decay),
                                "inner_trial_macro_mae": score,
                                "inner_prior_trial_macro_mae": base_scores[axis],
                                "inner_gain": base_scores[axis] - score,
                            }
        best.append(winner)
    return best, {
        "valence": base_scores[0],
        "arousal": base_scores[1],
    }


def threshold_choices(
    candidates: Sequence[Mapping[str, object]],
    threshold: float,
) -> list[dict[str, object]]:
    output = []
    for candidate in candidates:
        if (
            candidate["modality"] == "disabled"
            or float(candidate["inner_gain"]) < float(threshold)
        ):
            output.append(
                {
                    "axis": candidate["axis"],
                    "modality": "disabled",
                    "alpha": 0.0,
                    "gate": 0.0,
                    "cap": 0.0,
                    "ema_decay": 0.0,
                    "inner_trial_macro_mae": float(
                        candidate["inner_prior_trial_macro_mae"]
                    ),
                    "inner_prior_trial_macro_mae": float(
                        candidate["inner_prior_trial_macro_mae"]
                    ),
                    "inner_gain": 0.0,
                }
            )
        else:
            output.append(dict(candidate))
    return output


def final_predictions(
    index,
    views: Mapping[str, np.ndarray],
    archive: FoundationFeatureArchive,
    training_subjects: np.ndarray,
    training_videos: np.ndarray,
    validation_rows: np.ndarray,
    validation_prior: np.ndarray,
    category_by_video: Mapping[int, str],
    duration_by_video: Mapping[int, float],
    base_spec: CategoryPhaseSpec,
    axis_specs: Sequence[ContentTemplateSpec],
    choices_by_threshold: Mapping[float, Sequence[Mapping[str, object]]],
    *,
    map_cache: dict[tuple[object, ...], np.ndarray],
    primary_threshold: float,
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    fit_rows, fit_prior = cross_fitted_content_prior(
        index,
        training_subjects,
        training_videos,
        category_by_video,
        archive,
        base_spec,
        axis_specs,
        duration_by_video=duration_by_video,
        offset_seconds=0,
        map_cache=map_cache,
    )
    residual_target = index.targets[fit_rows] - fit_prior
    fit_weights = trial_macro_weights(index, fit_rows)
    enabled_keys = {
        (str(choice["modality"]), float(choice["alpha"]))
        for choices in choices_by_threshold.values()
        for choice in choices
        if choice["modality"] != "disabled"
    }
    fitted = {}
    raw_cache = {}
    for modality, alpha in sorted(enabled_keys):
        fitted[(modality, alpha)] = fit_weighted_standardized_ridge_path(
            feature_view(modality, views, fit_rows),
            residual_target,
            [alpha],
            fit_weights,
        )[alpha]
        raw_cache[(modality, alpha)] = fitted[(modality, alpha)].predict(
            feature_view(modality, views, validation_rows)
        )
    smoothed_cache: dict[tuple[str, float, float], np.ndarray] = {}
    predictions = {}
    primary_raw = np.zeros_like(validation_prior)
    for threshold, choices in choices_by_threshold.items():
        prediction = validation_prior.copy()
        for axis, choice in enumerate(choices):
            if choice["modality"] == "disabled":
                continue
            key = (str(choice["modality"]), float(choice["alpha"]))
            smooth_key = (*key, float(choice["ema_decay"]))
            if smooth_key not in smoothed_cache:
                smoothed_cache[smooth_key] = causal_ema(
                    raw_cache[key],
                    index.subject_numbers[validation_rows],
                    index.videos[validation_rows],
                    index.timestamps[validation_rows],
                    float(choice["ema_decay"]),
                )
            raw = smoothed_cache[smooth_key][:, axis]
            correction = np.clip(
                float(choice["gate"]) * raw,
                -float(choice["cap"]),
                float(choice["cap"]),
            )
            prediction[:, axis] = np.clip(
                validation_prior[:, axis] + correction, 1.0, 255.0
            )
            if np.isclose(float(threshold), float(primary_threshold)):
                primary_raw[:, axis] = raw
        predictions[float(threshold)] = prediction.astype(np.float32)
    return predictions, primary_raw.astype(np.float32)


def process_subject_fold(config: Mapping[str, object]) -> dict[str, object]:
    data_root = Path(str(config["data_root"]))
    stimulus_manifest = Path(str(config["stimulus_manifest"]))
    foundation_features = Path(str(config["foundation_features"]))
    content_artifact = Path(str(config["content_artifact"]))
    causal_cache = Path(str(config["causal_cache"]))
    cbramod_cache = Path(str(config["cbramod_cache"]))
    subject_fold = int(config["subject_fold"])
    modalities = tuple(str(value) for value in config["modalities"])
    thresholds = tuple(float(value) for value in config["thresholds"])
    primary_threshold = float(config["primary_threshold"])

    index = load_innovation_index(data_root)
    records = load_stimulus_manifest(stimulus_manifest)
    archive = FoundationFeatureArchive.load(foundation_features)
    category_by_video = emotion_by_video(records)
    duration_by_video = {
        int(record.video_id): float(record.reported_duration_seconds)
        for record in records
    }
    eeg, fnirs, cbramod, _ = load_features(
        causal_cache, cbramod_cache, len(index.targets)
    )
    cbramod_tokens = (
        load_cbramod_patch_tokens(cbramod_cache, len(index.targets))
        if set(modalities) & CBRAMOD_TOKEN_MODALITIES
        else None
    )
    views = build_feature_views(
        index, eeg, fnirs, cbramod, modalities, cbramod_tokens
    )
    fold_records = load_primary_fold_records(content_artifact)
    with np.load(
        content_artifact / "oof_predictions.npz", allow_pickle=False
    ) as payload:
        if not np.array_equal(payload["sample_ids"], index.sample_ids):
            raise ValueError("Content artifact sample order differs from index")
        content_oof = np.asarray(payload["primary_content"], dtype=np.float32)

    rows_out = []
    prior_out = []
    raw_out = []
    prediction_out = {threshold: [] for threshold in thresholds}
    selections = []
    map_cache: dict[tuple[object, ...], np.ndarray] = {}
    for video_fold in range(5):
        record = fold_records[(subject_fold, video_fold)]
        training_subjects = np.asarray(
            record["training_subjects"], dtype=np.int16
        )
        validation_subjects = np.asarray(
            record["validation_subjects"], dtype=np.int16
        )
        training_videos = np.asarray(
            record["training_videos"], dtype=np.int16
        )
        validation_videos = np.asarray(
            record["validation_videos"], dtype=np.int16
        )
        validation_rows = np.flatnonzero(
            np.isin(index.subject_numbers, validation_subjects)
            & np.isin(index.videos, validation_videos)
        )
        base_spec = phase_spec_from_record(record)
        axis_specs = axis_specs_from_record(record)
        reconstructed = predict_combined_content_prior(
            index,
            validation_rows,
            archive,
            training_subjects,
            training_videos,
            category_by_video,
            duration_by_video,
            base_spec,
            axis_specs,
            map_cache=map_cache,
        )
        expected_prior = content_oof[validation_rows]
        maximum_difference = float(
            np.max(np.abs(reconstructed - expected_prior))
        )
        if maximum_difference > 2e-5:
            raise RuntimeError(
                f"Content reconstruction mismatch: {maximum_difference}"
            )

        inner_prior, inner_raw, source_rows = inner_raw_predictions(
            index,
            views,
            archive,
            training_subjects,
            training_videos,
            category_by_video,
            duration_by_video,
            base_spec,
            axis_specs,
            modalities,
            int(config["inner_subject_folds"]),
            int(config["inner_video_folds"]),
            map_cache=map_cache,
        )
        candidates, inner_base_scores = best_axis_candidates(
            index, source_rows, inner_prior, inner_raw
        )
        choices_by_threshold = {
            threshold: threshold_choices(candidates, threshold)
            for threshold in thresholds
        }
        predictions, raw = final_predictions(
            index,
            views,
            archive,
            training_subjects,
            training_videos,
            validation_rows,
            expected_prior,
            category_by_video,
            duration_by_video,
            base_spec,
            axis_specs,
            choices_by_threshold,
            map_cache=map_cache,
            primary_threshold=primary_threshold,
        )
        rows_out.append(validation_rows)
        prior_out.append(expected_prior)
        raw_out.append(raw)
        for threshold in thresholds:
            prediction_out[threshold].append(predictions[threshold])
        selections.append(
            {
                "subject_fold": subject_fold,
                "video_fold": video_fold,
                "training_subjects": training_subjects.tolist(),
                "validation_subjects": validation_subjects.tolist(),
                "training_videos": training_videos.tolist(),
                "validation_videos": validation_videos.tolist(),
                "base_spec": record["base_spec"],
                "content_axis_specs": {
                    "valence": record["content_selection"]["axes"][
                        "valence"
                    ]["selected"],
                    "arousal": record["content_selection"]["axes"][
                        "arousal"
                    ]["selected"],
                },
                "best_physiology_candidates": candidates,
                "inner_content_prior_axis_mae": inner_base_scores,
                "choices_by_gain_threshold": {
                    threshold_key(threshold): choices_by_threshold[threshold]
                    for threshold in thresholds
                },
                "content_reconstruction_max_abs_difference": (
                    maximum_difference
                ),
                "validation_rows": int(len(validation_rows)),
            }
        )
        print(
            f"[av-physio] subject_fold={subject_fold} "
            f"video_fold={video_fold} "
            f"best={[item['modality'] for item in candidates]}",
            flush=True,
        )
    rows = np.concatenate(rows_out).astype(np.int64)
    order = np.argsort(rows)
    return {
        "rows": rows[order],
        "prior": np.concatenate(prior_out)[order],
        "raw": np.concatenate(raw_out)[order],
        "predictions": {
            threshold: np.concatenate(prediction_out[threshold])[order]
            for threshold in thresholds
        },
        "selections": selections,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = resolve(args.data_root)
    stimulus_manifest = resolve(args.stimulus_manifest)
    foundation_features = resolve(args.foundation_features)
    content_artifact = resolve(args.content_artifact)
    causal_cache = resolve(args.causal_cache)
    cbramod_cache = resolve(args.cbramod_cache)
    output_dir = resolve(args.output_dir)
    thresholds = tuple(sorted(set(float(value) for value in args.gain_thresholds)))
    if any(value < 0.0 for value in thresholds):
        raise ValueError("gain thresholds must be non-negative")
    primary_threshold = float(args.primary_gain_threshold)
    if primary_threshold not in thresholds:
        raise ValueError("primary gain threshold must appear in gain thresholds")
    prepare_output(output_dir, args.overwrite)
    started = time.perf_counter()

    run_manifest = {
        "schema_version": "merps-av-causal-physio-run-v1",
        "status": "running",
        "started_at_utc": utc_now(),
        "configuration": {
            "deployment": "new participant and unseen video",
            "modalities": list(args.modalities),
            "ridge_alphas": list(RIDGE_ALPHAS),
            "gates": list(GATES),
            "caps": list(CAPS),
            "ema_decays": list(EMA_DECAYS),
            "gain_thresholds": list(thresholds),
            "primary_gain_threshold": primary_threshold,
            "inner_subject_folds": int(args.inner_subject_folds),
            "inner_video_folds": int(args.inner_video_folds),
            "parallel_jobs": int(args.parallel_jobs),
            "trial_weighted_fit": True,
            "seed": int(args.seed),
        },
        "inputs": {
            "data_root": project_path(data_root),
            "stimulus_manifest": project_path(stimulus_manifest),
            "foundation_features": project_path(foundation_features),
            "content_artifact": project_path(content_artifact),
            "causal_cache": project_path(causal_cache),
            "cbramod_cache": project_path(cbramod_cache),
        },
        "source_hashes": {
            "src/merps/av_content_prior.py": file_sha256(
                PROJECT_ROOT / "src" / "merps" / "av_content_prior.py"
            ),
            "src/merps/physiology_residual.py": file_sha256(
                PROJECT_ROOT / "src" / "merps" / "physiology_residual.py"
            ),
            "scripts/run_av_physio_residual.py": file_sha256(Path(__file__)),
            "scripts/run_causal_physio_residual.py": file_sha256(
                PROJECT_ROOT / "scripts" / "run_causal_physio_residual.py"
            ),
            project_path(
                content_artifact / "oof_predictions.npz"
            ): file_sha256(content_artifact / "oof_predictions.npz"),
            project_path(
                content_artifact / "fold_selections.json"
            ): file_sha256(content_artifact / "fold_selections.json"),
            project_path(foundation_features): file_sha256(
                foundation_features
            ),
            **(
                {
                    project_path(cbramod_cache / "patch_tokens.npy"): file_sha256(
                        cbramod_cache / "patch_tokens.npy"
                    )
                }
                if set(args.modalities) & CBRAMOD_TOKEN_MODALITIES else {}
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    index = load_innovation_index(data_root)
    with np.load(
        content_artifact / "oof_predictions.npz", allow_pickle=False
    ) as payload:
        if not np.array_equal(payload["sample_ids"], index.sample_ids):
            raise ValueError("Content artifact sample order differs from index")
        expected_content_prior = np.asarray(
            payload["primary_content"], dtype=np.float32
        )

    base_config = {
        "data_root": str(data_root),
        "stimulus_manifest": str(stimulus_manifest),
        "foundation_features": str(foundation_features),
        "content_artifact": str(content_artifact),
        "causal_cache": str(causal_cache),
        "cbramod_cache": str(cbramod_cache),
        "modalities": tuple(args.modalities),
        "thresholds": thresholds,
        "primary_threshold": primary_threshold,
        "inner_subject_folds": int(args.inner_subject_folds),
        "inner_video_folds": int(args.inner_video_folds),
    }
    jobs = max(1, min(int(args.parallel_jobs), 5))
    completed = {}
    if jobs == 1:
        for subject_fold in range(5):
            completed[subject_fold] = process_subject_fold(
                {**base_config, "subject_fold": subject_fold}
            )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_fold = {
                executor.submit(
                    process_subject_fold,
                    {**base_config, "subject_fold": subject_fold},
                ): subject_fold
                for subject_fold in range(5)
            }
            for future in as_completed(future_to_fold):
                subject_fold = future_to_fold[future]
                completed[subject_fold] = future.result()
                print(
                    f"[scheduler] completed subject fold {subject_fold}",
                    flush=True,
                )

    prior_oof = np.full_like(index.targets, np.nan, dtype=np.float32)
    raw_oof = np.zeros_like(index.targets, dtype=np.float32)
    predictions = {
        threshold: np.full_like(index.targets, np.nan, dtype=np.float32)
        for threshold in thresholds
    }
    selections = []
    for subject_fold in range(5):
        result = completed[subject_fold]
        rows = result["rows"]
        prior_oof[rows] = result["prior"]
        raw_oof[rows] = result["raw"]
        for threshold in thresholds:
            predictions[threshold][rows] = result["predictions"][threshold]
        selections.extend(result["selections"])
    if not np.isfinite(prior_oof).all():
        raise RuntimeError("Incomplete outer content prior")
    if not np.array_equal(prior_oof, expected_content_prior):
        maximum = float(np.max(np.abs(prior_oof - expected_content_prior)))
        raise RuntimeError(f"Content artifact propagation mismatch: {maximum}")
    for threshold, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Incomplete physiology OOF at threshold {threshold}")

    metrics = {
        "content_prior": model_metrics(index, prior_oof, prior_oof),
        **{
            threshold_key(threshold): model_metrics(
                index, predictions[threshold], prior_oof
            )
            for threshold in thresholds
        },
    }
    decision_curve = []
    base_mae = metrics["content_prior"]["trial_macro_mae"]
    for threshold in thresholds:
        key = threshold_key(threshold)
        decision_curve.append(
            {
                "minimum_inner_gain": threshold,
                "trial_macro_mae": metrics[key]["trial_macro_mae"],
                "gain_vs_content_prior": (
                    base_mae - metrics[key]["trial_macro_mae"]
                ),
                "mean_absolute_correction": metrics[key][
                    "mean_absolute_correction"
                ],
            }
        )
    primary_key = threshold_key(primary_threshold)
    primary_prediction = predictions[primary_threshold]
    uncertainty = gain_uncertainty(
        index,
        prior_oof,
        primary_prediction,
        args.bootstrap_repeats,
        args.seed,
    )
    primary_choice_counts = Counter(
        f"{choice['axis']}:{choice['modality']}"
        for record in selections
        for choice in record["choices_by_gain_threshold"][primary_key]
    )
    results = {
        "schema_version": "merps-av-causal-physio-results-v1",
        "completed_at_utc": utc_now(),
        "deployment_target": "new participant and unseen video",
        "base_content_artifact_sha256": file_sha256(
            content_artifact / "oof_predictions.npz"
        ),
        "content_prior_exactly_propagated": True,
        "metrics": metrics,
        "primary_gain_threshold": primary_threshold,
        "primary_gain_uncertainty": uncertainty,
        "primary_axis_choice_counts": dict(sorted(primary_choice_counts.items())),
        "decision_curve": decision_curve,
        "outer_cells": 25,
        "leakage_controls": [
            "outer participant and video groups are both disjoint",
            "content prior is reconstructed exactly from the frozen outer-fold specification",
            "each physiology training residual excludes its own participant and video labels",
            "all EEG/fNIRS/CBraMod features are certified causal",
            "trial-weighted Ridge gives every participant-video trial equal fit mass",
            "axis-specific physiology can fall back exactly to zero correction",
        ],
        "interpretation_boundary": [
            "The complete stimulus is pre-encoded before playback.",
            "Physiology has access only to current and past signal windows.",
            "Thresholds form a prespecified decision curve; the primary threshold is 0.10 MAE.",
            "Conditional bootstrap does not propagate full model-selection uncertainty.",
            "Repeated grouped-CV stability is established for the upstream content prior.",
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    arrays = {
        "sample_ids": index.sample_ids,
        "targets": index.targets,
        "content_prior": prior_oof,
        "primary_prediction": primary_prediction,
        "primary_raw_residual": raw_oof,
    }
    for threshold in thresholds:
        arrays[f"prediction_{threshold_key(threshold)}"] = predictions[threshold]
    np.savez_compressed(output_dir / "oof_predictions.npz", **arrays)
    write_json(output_dir / "selections.json", selections)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "results.json", results)
    run_manifest.update(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "elapsed_seconds": results["elapsed_seconds"],
            "generated_files": [
                "metrics.json",
                "oof_predictions.npz",
                "results.json",
                "selections.json",
            ],
        }
    )
    write_json(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(json.dumps(uncertainty, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
