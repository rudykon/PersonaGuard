#!/usr/bin/env python3
"""Run construct, reliability, and calibration-cost robustness analyses."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.innovation.data import load_innovation_index, outer_subject_folds
from merps.innovation.normative_dynamics import (
    WarpConfig,
    circular_shift_null,
    decompose_trial,
    normative_trajectory,
)
from merps.innovation.phase_trait import VarianceComponents, reliability_for_videos

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "measurement_robustness"
DEFAULT_NORMATIVE_DIR = PROJECT_ROOT / "artifacts" / "normative_dynamics"
DEFAULT_TRAIT_DIR = PROJECT_ROOT / "artifacts" / "phase_trait"
DEFAULT_PERSONALIZATION_DIR = PROJECT_ROOT / "artifacts" / "phase_personalization"
MANAGED_FILES = {
    "matched_high_change.csv",
    "null_distribution.npz",
    "random_split_half.csv",
    "reliability_cost.csv",
    "run_manifest.json",
    "sensitivity.csv",
    "simulation_recovery.csv",
    "summary.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--normative-dir", type=Path, default=DEFAULT_NORMATIVE_DIR)
    parser.add_argument("--trait-dir", type=Path, default=DEFAULT_TRAIT_DIR)
    parser.add_argument(
        "--personalization-dir", type=Path, default=DEFAULT_PERSONALIZATION_DIR
    )
    parser.add_argument("--simulation-repeats", type=int, default=100)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--split-repeats", type=int, default=10000)
    parser.add_argument("--null-bank-size", type=int, default=32)
    parser.add_argument("--null-draws", type=int, default=10000)
    parser.add_argument("--skip-null", action="store_true")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
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


def interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(first: np.ndarray, second: np.ndarray, *, spearman: bool = False) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if spearman:
        first, second = rankdata(first), rankdata(second)
    if len(first) < 3 or first.std() < 1e-12 or second.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def spearman_brown(value: float) -> float:
    denominator = 1.0 + float(value)
    return float("nan") if abs(denominator) < 1e-12 else float(2.0 * value / denominator)


def matrix_inference(matrix: np.ndarray, *, repeats: int, seed: int) -> dict[str, object]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (24, 15) or not np.isfinite(values).all():
        raise ValueError("Expected a complete 24 x 15 participant-video matrix")
    rng = np.random.default_rng(seed)
    participant_boot = np.empty(repeats, dtype=np.float64)
    crossed_boot = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        participant_draw = rng.integers(0, 24, 24)
        participant_boot[repeat] = values[participant_draw].mean()
        video_draw = rng.integers(0, 15, 15)
        crossed_boot[repeat] = values[np.ix_(participant_draw, video_draw)].mean()
    participant_values = values.mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(100000, 24))
    null = (signs * participant_values[None, :]).mean(axis=1)
    observed = float(values.mean())
    return {
        "observed_mean": observed,
        "participant_bootstrap_ci95": interval(participant_boot),
        "participant_video_crossed_ci95": interval(crossed_boot),
        "participant_signflip_p_two_sided": float(
            (1 + np.sum(np.abs(null) >= abs(observed))) / 100001
        ),
        "independent_unit": "participant-by-video trial",
    }


def simulated_template(rng: np.random.Generator, length: int = 120) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length)
    phase = rng.uniform(-np.pi, np.pi, size=4)
    valence = 128.0 + 32.0 * np.sin(2 * np.pi * 1.3 * x + phase[0])
    valence += 17.0 * np.sin(2 * np.pi * 3.1 * x + phase[1])
    arousal = 128.0 + 28.0 * np.cos(2 * np.pi * 1.1 * x + phase[2])
    arousal += 15.0 * np.sin(2 * np.pi * 2.7 * x + phase[3])
    center = rng.uniform(0.25, 0.75)
    transition = np.tanh((x - center) / rng.uniform(0.025, 0.055))
    valence += rng.choice([-1.0, 1.0]) * 16.0 * transition
    arousal -= rng.choice([-1.0, 1.0]) * 12.0 * transition
    return np.stack([valence, arousal], axis=1)


def run_simulation_recovery(
    *, repeats: int, seed: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = WarpConfig(10, 0.15, 0.05, 2, True)
    scenarios = (
        ("null", 0, 0.0),
        ("pure_temporal", 4, 0.0),
        ("pure_amplitude", 0, 14.0),
        ("mixed", 4, 14.0),
    )
    noise_levels = (0.0, 2.0, 5.0)
    rng = np.random.default_rng(seed)
    raw: list[dict[str, float | str]] = []
    for scenario, lag, amplitude_scale in scenarios:
        for noise_sd in noise_levels:
            for _ in range(repeats):
                template = simulated_template(rng)
                length = len(template)
                times = np.arange(length)
                true_mapping = np.clip(times - lag, 0, length - 1)
                gradient = np.linalg.norm(np.gradient(template, axis=0), axis=1)
                usable = np.arange(12, length - 12)
                pulse_center = int(usable[np.argmax(gradient[usable])])
                pulse = np.exp(-0.5 * ((times - pulse_center) / 5.0) ** 2)
                injected = np.zeros_like(template)
                injected[:, 0] = amplitude_scale * pulse
                injected[:, 1] = -0.75 * amplitude_scale * pulse
                target = template[true_mapping] + injected
                target += rng.normal(0.0, noise_sd, size=target.shape)
                result = decompose_trial(target, template, config)
                interior = slice(config.band_seconds, length - config.band_seconds)
                true_lag = times - true_mapping
                estimated_lag = result.consensus_lag.astype(np.float64)
                slope_order = np.argsort(gradient, kind="mergesort")
                quartile = max(1, length // 4)
                low, high = slope_order[:quartile], slope_order[-quartile:]
                raw.append(
                    {
                        "scenario": scenario,
                        "noise_sd": noise_sd,
                        "lag_mae_seconds": float(
                            np.abs(estimated_lag[interior] - true_lag[interior]).mean()
                        ),
                        "estimated_mean_absolute_lag_seconds": float(
                            np.abs(estimated_lag[interior]).mean()
                        ),
                        "true_mean_absolute_lag_seconds": float(
                            np.abs(true_lag[interior]).mean()
                        ),
                        "amplitude_recovery_mae_units": float(
                            np.abs(result.amplitude_residual[interior] - injected[interior]).mean()
                        ),
                        "cross_dimension_gain_units": float(result.metrics["cross_gain_mean"]),
                        "high_minus_low_slope_false_lag_seconds": float(
                            np.abs(estimated_lag[high]).mean()
                            - np.abs(estimated_lag[low]).mean()
                        ),
                    }
                )
            print(f"simulation scenario={scenario} noise={noise_sd:.1f} complete", flush=True)
    rows: list[dict[str, object]] = []
    metric_keys = (
        "lag_mae_seconds",
        "estimated_mean_absolute_lag_seconds",
        "true_mean_absolute_lag_seconds",
        "amplitude_recovery_mae_units",
        "cross_dimension_gain_units",
        "high_minus_low_slope_false_lag_seconds",
    )
    for scenario, _, _ in scenarios:
        for noise_sd in noise_levels:
            selected = [
                item
                for item in raw
                if item["scenario"] == scenario and item["noise_sd"] == noise_sd
            ]
            row: dict[str, object] = {
                "scenario": scenario,
                "noise_sd": noise_sd,
                "repeats": repeats,
            }
            for key in metric_keys:
                values = np.asarray([float(item[key]) for item in selected])
                row[f"{key}_mean"] = float(values.mean())
                row[f"{key}_ci95_low"] = float(np.quantile(values, 0.025))
                row[f"{key}_ci95_high"] = float(np.quantile(values, 0.975))
            rows.append(row)
    pure_time = [row for row in rows if row["scenario"] == "pure_temporal"]
    pure_amplitude = [row for row in rows if row["scenario"] == "pure_amplitude"]
    summary = {
        "design": "known lag and/or slope-centered amplitude pulse across three noise levels",
        "primary_config": asdict(config),
        "pure_temporal_lag_mae_range_seconds": [
            float(min(row["lag_mae_seconds_mean"] for row in pure_time)),
            float(max(row["lag_mae_seconds_mean"] for row in pure_time)),
        ],
        "pure_amplitude_spurious_absolute_lag_range_seconds": [
            float(min(row["estimated_mean_absolute_lag_seconds_mean"] for row in pure_amplitude)),
            float(max(row["estimated_mean_absolute_lag_seconds_mean"] for row in pure_amplitude)),
        ],
        "interpretation_boundary": (
            "Recovery diagnoses the estimator; it does not identify empirical lag as uniquely affective."
        ),
    }
    return rows, summary


def trial_rows(index, subject: int, video: int) -> np.ndarray:
    rows = np.flatnonzero(
        (index.subject_numbers == int(subject)) & (index.videos == int(video))
    )
    return rows[np.argsort(index.timestamps[rows])]


def empirical_configurations() -> list[tuple[str, int, WarpConfig]]:
    primary = WarpConfig(10, 0.15, 0.05, 2, True)
    return [
        ("primary", 3, primary),
        ("band_5", 3, WarpConfig(5, 0.15, 0.05, 2, True)),
        ("band_15", 3, WarpConfig(15, 0.15, 0.05, 2, True)),
        ("warp_penalty_0.05", 3, WarpConfig(10, 0.05, 0.05, 2, True)),
        ("warp_penalty_0.30", 3, WarpConfig(10, 0.30, 0.05, 2, True)),
        ("smoothing_1", 3, WarpConfig(10, 0.15, 0.05, 1, True)),
        ("smoothing_3", 3, WarpConfig(10, 0.15, 0.05, 3, True)),
        ("template_radius_1", 1, primary),
        ("template_radius_5", 5, primary),
    ]


def run_sensitivity(
    index,
    primary_matrix: np.ndarray,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    primary_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    for position, (name, template_radius, config) in enumerate(empirical_configurations()):
        gain = np.full((24, 15), np.nan, dtype=np.float64)
        gain_v = np.full_like(gain, np.nan)
        gain_a = np.full_like(gain, np.nan)
        lag_effect = np.full_like(gain, np.nan)
        amplitude_effect = np.full_like(gain, np.nan)
        q_matrix = np.full_like(gain, np.nan)
        disagreement = np.full_like(gain, np.nan)
        for training_subjects, validation_subjects in outer_subject_folds():
            templates = {
                video: normative_trajectory(
                    index, training_subjects, video, radius=template_radius
                )[0]
                for video in range(1, 16)
            }
            for subject in validation_subjects:
                for video in range(1, 16):
                    selected_rows = trial_rows(index, int(subject), video)
                    target = index.targets[selected_rows].astype(np.float64)
                    template = templates[video]
                    result = decompose_trial(target, template, config)
                    i, j = int(subject) - 1, video - 1
                    gain[i, j] = result.metrics["cross_gain_mean"]
                    gain_v[i, j] = result.metrics["cross_gain_valence"]
                    gain_a[i, j] = result.metrics["cross_gain_arousal"]
                    lag_effect[i, j] = (
                        result.metrics["boundary_absolute_lag_seconds"]
                        - result.metrics["stable_absolute_lag_seconds"]
                    )
                    amplitude_effect[i, j] = (
                        result.metrics["boundary_amplitude_deviation"]
                        - result.metrics["stable_amplitude_deviation"]
                    )
                    q_matrix[i, j] = result.metrics["mean_absolute_lag_seconds"]
                    disagreement[i, j] = result.metrics[
                        "lag_dimension_disagreement_seconds"
                    ]
                    if name == "primary":
                        primary_inputs.append((target, template))
        gain_inference = matrix_inference(
            gain, repeats=bootstrap_repeats, seed=seed + position * 11
        )
        lag_inference = matrix_inference(
            lag_effect, repeats=bootstrap_repeats, seed=seed + position * 11 + 1
        )
        amp_inference = matrix_inference(
            amplitude_effect, repeats=bootstrap_repeats, seed=seed + position * 11 + 2
        )
        rows.append(
            {
                "configuration": name,
                "template_radius": template_radius,
                "band_seconds": config.band_seconds,
                "warp_penalty": config.warp_penalty,
                "step_penalty": config.step_penalty,
                "smoothing_radius": config.smoothing_radius,
                "cross_gain_mean_units": float(gain.mean()),
                "cross_gain_crossed_ci95_low": gain_inference[
                    "participant_video_crossed_ci95"
                ][0],
                "cross_gain_crossed_ci95_high": gain_inference[
                    "participant_video_crossed_ci95"
                ][1],
                "cross_gain_valence_units": float(gain_v.mean()),
                "cross_gain_arousal_units": float(gain_a.mean()),
                "positive_trial_fraction": float((gain > 0).mean()),
                "high_change_lag_effect_seconds": float(lag_effect.mean()),
                "high_change_lag_crossed_ci95_low": lag_inference[
                    "participant_video_crossed_ci95"
                ][0],
                "high_change_lag_crossed_ci95_high": lag_inference[
                    "participant_video_crossed_ci95"
                ][1],
                "high_change_amplitude_effect_units": float(amplitude_effect.mean()),
                "high_change_amplitude_crossed_ci95_low": amp_inference[
                    "participant_video_crossed_ci95"
                ][0],
                "high_change_amplitude_crossed_ci95_high": amp_inference[
                    "participant_video_crossed_ci95"
                ][1],
                "participant_profile_pearson_vs_primary": correlation(
                    q_matrix.mean(axis=1), primary_matrix.mean(axis=1)
                ),
                "participant_profile_spearman_vs_primary": correlation(
                    q_matrix.mean(axis=1), primary_matrix.mean(axis=1), spearman=True
                ),
                "mean_dimension_lag_disagreement_seconds": float(disagreement.mean()),
            }
        )
        print(f"sensitivity configuration={name} complete", flush=True)
    cross_values = np.asarray([float(row["cross_gain_mean_units"]) for row in rows])
    lag_values = np.asarray(
        [float(row["high_change_lag_effect_seconds"]) for row in rows]
    )
    return rows, {
        "configurations": len(rows),
        "cross_gain_range_units": [float(cross_values.min()), float(cross_values.max())],
        "high_change_lag_effect_range_seconds": [float(lag_values.min()), float(lag_values.max())],
        "all_cross_gain_estimates_positive": bool(np.all(cross_values > 0)),
        "all_high_change_lag_estimates_positive": bool(np.all(lag_values > 0)),
        "one_factor_at_a_time": True,
    }, primary_inputs


def matched_peak_pairs(template: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    length = len(template)
    gradient = np.linalg.norm(np.gradient(template, axis=0), axis=1)
    progress = np.linspace(0.0, 1.0, length)
    magnitude = np.linalg.norm((template - 128.0) / 127.0, axis=1)
    local_peak = np.zeros(length, dtype=bool)
    local_peak[1:-1] = (
        (gradient[1:-1] >= gradient[:-2]) & (gradient[1:-1] >= gradient[2:])
    )
    boundary = np.flatnonzero(local_peak & (gradient >= np.quantile(gradient, 0.75)))
    boundary = boundary[(boundary >= 5) & (boundary < length - 5)]
    if not len(boundary):
        boundary = np.asarray([int(np.argmax(gradient[5:-5])) + 5])
    excluded = np.zeros(length, dtype=bool)
    excluded[boundary] = True
    candidates = np.flatnonzero(~excluded)
    candidates = candidates[(candidates >= 5) & (candidates < length - 5)]
    features = np.stack([gradient, progress, magnitude], axis=1)
    scale = np.maximum(features.std(axis=0), 1e-6)
    controls = []
    for value in boundary:
        distance = np.square((features[candidates] - features[value]) / scale).sum(axis=1)
        selected = int(candidates[int(np.argmin(distance))])
        controls.append(selected)
    controls_array = np.asarray(controls, dtype=np.int64)
    balance = {
        "pairs": int(len(boundary)),
        "gradient_standardized_mean_difference": float(
            (gradient[boundary].mean() - gradient[controls_array].mean()) / scale[0]
        ),
        "progress_standardized_mean_difference": float(
            (progress[boundary].mean() - progress[controls_array].mean()) / scale[1]
        ),
        "magnitude_standardized_mean_difference": float(
            (magnitude[boundary].mean() - magnitude[controls_array].mean()) / scale[2]
        ),
    }
    return boundary, controls_array, balance


def run_matched_high_change(
    index, *, bootstrap_repeats: int, seed: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = WarpConfig(10, 0.15, 0.05, 2, True)
    records: list[dict[str, object]] = []
    lag_matrix = np.full((24, 15), np.nan, dtype=np.float64)
    amplitude_matrix = np.full_like(lag_matrix, np.nan)
    balance_records: list[dict[str, float]] = []
    for training_subjects, validation_subjects in outer_subject_folds():
        templates = {
            video: normative_trajectory(index, training_subjects, video, radius=3)[0]
            for video in range(1, 16)
        }
        pairs = {video: matched_peak_pairs(templates[video]) for video in range(1, 16)}
        balance_records.extend(value[2] for value in pairs.values())
        for subject in validation_subjects:
            for video in range(1, 16):
                selected_rows = trial_rows(index, int(subject), video)
                result = decompose_trial(index.targets[selected_rows], templates[video], config)
                boundary, controls, balance = pairs[video]
                absolute_lag = np.abs(result.consensus_lag.astype(np.float64))
                amplitude = np.abs(result.amplitude_residual).mean(axis=1)
                lag_effect = float(absolute_lag[boundary].mean() - absolute_lag[controls].mean())
                amplitude_effect = float(amplitude[boundary].mean() - amplitude[controls].mean())
                i, j = int(subject) - 1, video - 1
                lag_matrix[i, j] = lag_effect
                amplitude_matrix[i, j] = amplitude_effect
                records.append(
                    {
                        "subject": int(subject),
                        "video": video,
                        "matched_pairs": balance["pairs"],
                        "lag_effect_seconds": lag_effect,
                        "amplitude_effect_units": amplitude_effect,
                        "gradient_standardized_mean_difference": balance[
                            "gradient_standardized_mean_difference"
                        ],
                        "progress_standardized_mean_difference": balance[
                            "progress_standardized_mean_difference"
                        ],
                        "magnitude_standardized_mean_difference": balance[
                            "magnitude_standardized_mean_difference"
                        ],
                    }
                )
    lag_inference = matrix_inference(lag_matrix, repeats=bootstrap_repeats, seed=seed)
    amplitude_inference = matrix_inference(
        amplitude_matrix, repeats=bootstrap_repeats, seed=seed + 1
    )
    balance_keys = (
        "gradient_standardized_mean_difference",
        "progress_standardized_mean_difference",
        "magnitude_standardized_mean_difference",
    )
    return records, {
        "definition": (
            "local peaks in the top normative-gradient quartile, nearest-neighbour matched "
            "within video (with replacement) on gradient, video progress, and normative magnitude"
        ),
        "lag_effect": lag_inference,
        "amplitude_effect": amplitude_inference,
        "mean_pairs_per_fold_video": float(
            np.mean([value["pairs"] for value in balance_records])
        ),
        "mean_absolute_standardized_balance": {
            key: float(np.mean([abs(value[key]) for value in balance_records]))
            for key in balance_keys
        },
        "boundary_language": (
            "These are normative high-change peaks, not independently annotated semantic events."
        ),
    }


def run_null_bank(
    trial_inputs: list[tuple[np.ndarray, np.ndarray]],
    *,
    observed_gain: float,
    bank_size: int,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    config = WarpConfig(10, 0.15, 0.05, 2, True)
    rng = np.random.default_rng(seed)
    bank = np.empty((bank_size, len(trial_inputs)), dtype=np.float32)
    for bank_index in range(bank_size):
        for trial_index, (target, template) in enumerate(trial_inputs):
            bank[bank_index, trial_index] = circular_shift_null(
                target, template, config, rng
            )
        print(f"circular null bank={bank_index + 1}/{bank_size} complete", flush=True)
    null = np.empty(draws, dtype=np.float64)
    trial_axis = np.arange(len(trial_inputs))
    for start in range(0, draws, 500):
        end = min(start + 500, draws)
        choices = rng.integers(0, bank_size, size=(end - start, len(trial_inputs)))
        sampled = bank[choices, trial_axis[None, :]]
        null[start:end] = sampled.mean(axis=1)
    return bank, null, {
        "trial_units": len(trial_inputs),
        "independent_shift_realisations_per_trial": bank_size,
        "dataset_level_null_draws": draws,
        "observed_gain_units": observed_gain,
        "null_mean_units": float(null.mean()),
        "null_ci95_units": interval(null),
        "one_sided_p": float((1 + np.sum(null >= observed_gain)) / (draws + 1)),
        "null_preserves": "within-dimension marginal distribution and autocorrelation",
        "null_breaks": "cross-dimensional temporal synchronization",
        "permutation_unit": "independent circular shifts within each participant-video trial",
        "finite_sample_formula": "(1 + count[T_null >= T_observed]) / (B + 1)",
    }


def emotion_categories(path: Path) -> np.ndarray:
    first_line = path.read_text(encoding="utf-8").splitlines()[1]
    values = [int(value) for value in re.findall(r"\b[0-4]\b", first_line)]
    if len(values) != 15:
        raise RuntimeError("Could not parse 15 targeted-emotion categories")
    return np.asarray(values, dtype=np.int16)


def run_random_split_half(
    matrix: np.ndarray,
    categories: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(seed)
    by_category = {
        category: np.flatnonzero(categories == category)
        for category in sorted(set(int(value) for value in categories))
    }
    records: list[dict[str, object]] = []
    for repeat in range(repeats):
        double_categories = set(rng.choice(list(by_category), size=2, replace=False))
        first: list[int] = []
        second: list[int] = []
        for category, video_indices in by_category.items():
            shuffled = rng.permutation(video_indices)
            first_count = 2 if category in double_categories else 1
            first.extend(int(value) for value in shuffled[:first_count])
            second.extend(int(value) for value in shuffled[first_count:])
        first_profile = matrix[:, first].mean(axis=1)
        second_profile = matrix[:, second].mean(axis=1)
        pearson = correlation(first_profile, second_profile)
        spearman = correlation(first_profile, second_profile, spearman=True)
        records.append(
            {
                "repeat": repeat + 1,
                "first_videos": " ".join(str(value + 1) for value in sorted(first)),
                "second_videos": " ".join(str(value + 1) for value in sorted(second)),
                "pearson": pearson,
                "spearman": spearman,
                "spearman_brown_pearson": spearman_brown(pearson),
                "spearman_brown_spearman": spearman_brown(spearman),
            }
        )
    summary: dict[str, object] = {
        "repeats": repeats,
        "split": "7 versus 8 videos, stratified so every emotion category appears in both halves",
        "correction": "Spearman-Brown 2r/(1+r)",
    }
    for key in (
        "pearson",
        "spearman",
        "spearman_brown_pearson",
        "spearman_brown_spearman",
    ):
        values = np.asarray([float(row[key]) for row in records])
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "ci95": interval(values),
        }
    return records, summary


def video_durations(index) -> np.ndarray:
    durations = np.empty(15, dtype=np.float64)
    for video in range(1, 16):
        rows = np.flatnonzero(index.videos == video)
        durations[video - 1] = float(index.timestamps[rows].max() + 1)
    return durations


def run_reliability_cost(
    index,
    trait_summary: dict[str, object],
    personalization_summary: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw = trait_summary["g_study_variance_components"]
    components = VarianceComponents(
        grand_mean=float(raw["grand_mean"]),
        participant=float(raw["participant"]),
        video=float(raw["video"]),
        residual=float(raw["residual"]),
        participant_mean_square=float(raw["participant_mean_square"]),
        video_mean_square=float(raw["video_mean_square"]),
        residual_mean_square=float(raw["residual_mean_square"]),
    )
    durations = video_durations(index)
    aggregate = personalization_summary["aggregate"]
    rows: list[dict[str, object]] = []
    for count in range(1, 16):
        reliability = reliability_for_videos(components, count)
        relative_sem = float(np.sqrt(components.residual / count))
        absolute_sem = float(np.sqrt((components.video + components.residual) / count))
        row: dict[str, object] = {
            "videos": count,
            "expected_calibration_minutes": float(count * durations.mean() / 60.0),
            "minimum_calibration_minutes": float(np.sort(durations)[:count].sum() / 60.0),
            "maximum_calibration_minutes": float(np.sort(durations)[-count:].sum() / 60.0),
            "relative_g": reliability["relative_g"],
            "absolute_phi": reliability["absolute_phi"],
            "relative_sem_seconds": relative_sem,
            "absolute_sem_seconds": absolute_sem,
            "video_mean_baseline_mae_seconds": "",
            "additive_calibration_mae_seconds": "",
            "absolute_gain_seconds": "",
            "relative_improvement_percent": "",
            "gain_as_fraction_of_relative_sem": "",
        }
        if str(count) in aggregate:
            baseline = aggregate[str(count)]["video_mean_zero_shot"]
            calibrated = aggregate[str(count)]["additive_fewshot"]
            baseline_mae = float(baseline["participant_macro_mae_seconds"])
            calibrated_mae = float(calibrated["participant_macro_mae_seconds"])
            gain = float(calibrated["paired_gain_vs_video_mean_seconds"])
            row.update(
                {
                    "video_mean_baseline_mae_seconds": baseline_mae,
                    "additive_calibration_mae_seconds": calibrated_mae,
                    "absolute_gain_seconds": gain,
                    "relative_improvement_percent": float(100.0 * gain / baseline_mae),
                    "gain_as_fraction_of_relative_sem": float(gain / relative_sem),
                }
            )
        rows.append(row)
    return rows, {
        "video_duration_seconds": {
            "mean": float(durations.mean()),
            "range": [float(durations.min()), float(durations.max())],
            "total": float(durations.sum()),
        },
        "sem_definition": {
            "relative": "sqrt(sigma_residual^2 / k)",
            "absolute": "sqrt((sigma_video^2 + sigma_residual^2) / k)",
        },
        "calibration_budgets_with_measured_prediction": [
            int(key) for key in aggregate if int(key) > 0
        ],
    }


def main() -> None:
    args = parse_args()
    data_root = resolve(args.data_root)
    output_dir = resolve(args.output_dir)
    normative_dir = resolve(args.normative_dir)
    trait_dir = resolve(args.trait_dir)
    personalization_dir = resolve(args.personalization_dir)
    prepare_output(output_dir, args.overwrite)
    manifest_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "running",
            "updated_at_utc": utc_now(),
            "configuration": manifest_args,
        },
    )
    started = time.perf_counter()
    index = load_innovation_index(data_root)
    primary_matrix = np.load(
        trait_dir / "phase_trait_matrix.npz", allow_pickle=False
    )["mean_absolute_lag"].astype(np.float64)
    normative_summary = json.loads(
        (normative_dir / "summary.json").read_text(encoding="utf-8")
    )
    trait_summary = json.loads((trait_dir / "summary.json").read_text(encoding="utf-8"))
    personalization_summary = json.loads(
        (personalization_dir / "summary.json").read_text(encoding="utf-8")
    )

    simulation_rows, simulation_summary = run_simulation_recovery(
        repeats=args.simulation_repeats, seed=args.seed
    )
    write_rows(output_dir / "simulation_recovery.csv", simulation_rows)
    sensitivity_rows, sensitivity_summary, trial_inputs = run_sensitivity(
        index,
        primary_matrix,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 1000,
    )
    write_rows(output_dir / "sensitivity.csv", sensitivity_rows)
    matched_rows, matched_summary = run_matched_high_change(
        index, bootstrap_repeats=args.bootstrap_repeats, seed=args.seed + 2000
    )
    write_rows(output_dir / "matched_high_change.csv", matched_rows)
    split_rows, split_summary = run_random_split_half(
        primary_matrix,
        emotion_categories(data_root / "Targeted_emotions.txt"),
        repeats=args.split_repeats,
        seed=args.seed + 3000,
    )
    write_rows(output_dir / "random_split_half.csv", split_rows)
    cost_rows, cost_summary = run_reliability_cost(
        index, trait_summary, personalization_summary
    )
    write_rows(output_dir / "reliability_cost.csv", cost_rows)

    if args.skip_null:
        null_summary: dict[str, object] = {"status": "skipped", "reason": "--skip-null"}
    else:
        bank, null, null_summary = run_null_bank(
            trial_inputs,
            observed_gain=float(
                normative_summary["primary_cross_dimension_gain"]["observed_mean"]
            ),
            bank_size=args.null_bank_size,
            draws=args.null_draws,
            seed=args.seed + 4000,
        )
        np.savez_compressed(
            output_dir / "null_distribution.npz",
            trial_null_bank=bank,
            dataset_level_null=null,
            observed_gain=np.asarray(
                normative_summary["primary_cross_dimension_gain"]["observed_mean"]
            ),
        )

    summary = {
        "research_question": (
            "How identifiable, parameter-robust, repeatable, and practically useful is "
            "interface-contingent temporal deviation in continuous affect reports?"
        ),
        "simulation_recovery": simulation_summary,
        "parameter_sensitivity": sensitivity_summary,
        "matched_high_change": matched_summary,
        "circular_shift_null": null_summary,
        "random_balanced_split_half": split_summary,
        "reliability_and_cost": cost_summary,
        "independent_units": {
            "empirical_effects": "participant-by-video trial",
            "reliability": "participant crossed with video",
            "simulation": "independently generated synthetic trajectory",
            "null": "independent circular shifts within participant-video trial",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "configuration": manifest_args,
            "software": {"python": platform.python_version(), "numpy": np.__version__},
            "generated_files": sorted(item.name for item in output_dir.iterdir()),
            "elapsed_seconds": summary["elapsed_seconds"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
