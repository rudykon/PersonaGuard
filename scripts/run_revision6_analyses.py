#!/usr/bin/env python3
"""Revision-6 analyses for matched donor transfer and sensing specification.

The script repairs the cross-video donor control by estimating every donor
residual from the receiver's exact calibration-video set, tests category
heterogeneity with participant-blocked permutations, and makes the strict
anti-aliased EEG plus reservation-aware fNIRS pipeline the primary sensing
analysis.  Only aggregate results are written; participant-level targets,
predictions, and profiles remain in memory and are not added to the anonymous
artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merps.innovation.data import (  # noqa: E402
    load_innovation_index,
    outer_subject_folds,
)
from merps.innovation.normative_dynamics import WarpConfig  # noqa: E402
from merps.innovation.phase_sensing import (  # noqa: E402
    RIDGE_ALPHAS,
    build_outer_phase_targets,
)
from run_revision2_analyses import emotion_categories, trial_rows  # noqa: E402
from run_revision3_analyses import METHODS, run_nested_sensing  # noqa: E402
from run_revision4_analyses import (  # noqa: E402
    build_reservation_aware_fnirs,
    paired_crossed_interval,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_FEATURE_CACHE = PROJECT_ROOT / "data" / "feature_cache"
DEFAULT_INNOVATION_CACHE = PROJECT_ROOT / "data" / "innovation_cache"
DEFAULT_ANTI_ALIAS_CACHE = PROJECT_ROOT / "data" / "innovation_cache_antialias"
DEFAULT_TRAIT_DIR = PROJECT_ROOT / "artifacts" / "phase_trait"
DEFAULT_REVISION4_DIR = PROJECT_ROOT / "artifacts" / "revision4"
DEFAULT_REVISION5_DIR = PROJECT_ROOT / "artifacts" / "revision5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "revision6"

CALIBRATION_BUDGETS = (1, 2, 4, 8)
CATEGORY_NAMES = ("neutral", "happy", "fear", "sad", "relaxed")
ALL_VIDEOS = tuple(range(1, 16))
DECISION_THRESHOLDS = tuple(float(value) for value in np.linspace(0.0, 0.2, 41))
MANAGED_FILES = {
    "category_heterogeneity.csv",
    "decision_threshold_curves.csv",
    "matched_identity_transfer.csv",
    "primary_antialias_sensing_results.csv",
    "repeated_grouped_cv.csv",
    "run_manifest.json",
    "sensing_model_table.csv",
    "summary.json",
}

MODEL_METADATA: dict[str, dict[str, object]] = {
    "video_mean": {
        "label": "Video mean",
        "inputs": "Video identity",
        "dimension": None,
        "contains_eeg": False,
    },
    "context_only": {
        "label": "Context",
        "inputs": "Fold-safe reference/video context",
        "dimension": 14,
        "contains_eeg": False,
    },
    "cbramod_only": {
        "label": "CBraMod",
        "inputs": "EEG foundation features",
        "dimension": 800,
        "contains_eeg": True,
    },
    "handcrafted_eeg_only": {
        "label": "Handcrafted EEG",
        "inputs": "EEG spectral/Hjorth features",
        "dimension": 180,
        "contains_eeg": True,
    },
    "eeg_only": {
        "label": "Combined EEG",
        "inputs": "CBraMod + handcrafted EEG",
        "dimension": 980,
        "contains_eeg": True,
    },
    "fnirs_only": {
        "label": "fNIRS",
        "inputs": "Reservation-aware distributed-lag summaries",
        "dimension": 1260,
        "contains_eeg": False,
    },
    "eeg_fnirs_single_penalty": {
        "label": "EEG+fNIRS",
        "inputs": "Combined EEG + fNIRS",
        "dimension": 2240,
        "contains_eeg": True,
    },
    "context_eeg": {
        "label": "Context+EEG",
        "inputs": "Context + combined EEG",
        "dimension": 994,
        "contains_eeg": True,
    },
    "context_fnirs": {
        "label": "Context+fNIRS",
        "inputs": "Context + fNIRS",
        "dimension": 1274,
        "contains_eeg": False,
    },
    "context_eeg_fnirs_single_penalty": {
        "label": "Context+joint",
        "inputs": "Context + combined EEG + fNIRS",
        "dimension": 2254,
        "contains_eeg": True,
    },
    "context_plus_blockwise_residual": {
        "label": "Blockwise residual stack",
        "inputs": "Context + inner-selected sensor residual (disabled allowed)",
        "dimension": None,
        "contains_eeg": True,
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument(
        "--innovation-cache", type=Path, default=DEFAULT_INNOVATION_CACHE
    )
    parser.add_argument(
        "--anti-alias-cache", type=Path, default=DEFAULT_ANTI_ALIAS_CACHE
    )
    parser.add_argument("--trait-dir", type=Path, default=DEFAULT_TRAIT_DIR)
    parser.add_argument("--revision4-dir", type=Path, default=DEFAULT_REVISION4_DIR)
    parser.add_argument("--revision5-dir", type=Path, default=DEFAULT_REVISION5_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--permutation-repeats", type=int, default=10000)
    parser.add_argument(
        "--grouped-cv-repeats", type=int, default=5,
        help="Total deterministic 5-fold participant-grouped CV partitions, including the primary split.",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--skip-sensing", action="store_true")
    parser.add_argument("--skip-fold-sensitivity", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def portable_path(path: Path) -> str:
    resolved = resolve(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def participant_bootstrap(
    values: dict[int, float], *, repeats: int, seed: int
) -> tuple[float, list[float]]:
    participants = np.asarray(sorted(values), dtype=np.int16)
    observed = float(np.mean([values[int(subject)] for subject in participants]))
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        selected = rng.choice(participants, size=len(participants), replace=True)
        draws[repeat] = float(
            np.mean([values[int(subject)] for subject in selected])
        )
    return observed, percentile_interval(draws)


def wilson_interval(count: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Wilson score interval for a participant-level binary proportion."""

    if total <= 0 or count < 0 or count > total:
        raise ValueError("Invalid count for Wilson interval")
    proportion = count / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))]


def threshold_decision_rows(
    gains: Sequence[float],
    *,
    analysis: str,
    route: str,
    unit: str,
    budget: int | None = None,
    thresholds: Sequence[float] = DECISION_THRESHOLDS,
) -> list[dict[str, object]]:
    """Describe benefit and degradation rates across hypothetical gain thresholds."""

    values = np.asarray(gains, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Threshold-sensitivity gains must be a finite one-dimensional array")
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        if threshold < 0:
            raise ValueError("Decision thresholds must be non-negative")
        counts = {
            "meets_gain_threshold": int(np.sum(values >= float(threshold))),
            "exceeds_degradation_threshold": int(
                np.sum(values <= -float(threshold))
            ),
        }
        for direction, count in counts.items():
            interval = wilson_interval(count, len(values))
            row: dict[str, object] = {
                "analysis": analysis,
                "route": route,
                "unit": unit,
                "threshold": float(threshold),
                "direction": direction,
                "participants": int(len(values)),
                "count": count,
                "proportion": float(count / len(values)),
                "wilson_ci95_low": interval[0],
                "wilson_ci95_high": interval[1],
            }
            if budget is not None:
                row["budget"] = int(budget)
            rows.append(row)
    return rows


def calibration_threshold_decision_rows(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    """Build aggregate calibration threshold sensitivities without identities."""

    grouped: dict[int, dict[int, float]] = defaultdict(dict)
    for row in rows:
        budget = int(row["budget"])
        subject = int(row["subject"])
        if subject in grouped[budget]:
            raise RuntimeError(
                f"Duplicate calibration practical-value row for budget={budget}, subject={subject}"
            )
        grouped[budget][subject] = float(row["calibrated_gain_vs_video_units"])
    output: list[dict[str, object]] = []
    for budget in CALIBRATION_BUDGETS:
        participants = grouped[budget]
        if len(participants) != 24:
            raise RuntimeError(
                f"Expected 24 calibration participants for budget {budget}, found {len(participants)}"
            )
        output.extend(
            threshold_decision_rows(
                [participants[subject] for subject in sorted(participants)],
                analysis="signed_calibration_vs_video_only",
                route="behavioral_calibration",
                unit="joystick_label_units",
                budget=budget,
            )
        )
    return output


def random_derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("A derangement needs at least two identities")
    original = np.arange(size)
    for _ in range(1000):
        candidate = rng.permutation(size)
        if np.all(candidate != original):
            return candidate
    return np.roll(original, 1)


def signed_lookup(
    signed_rows: Sequence[dict[str, str]],
) -> dict[tuple[int, int, int], tuple[float, float]]:
    """Return fold/subject/video -> (true b, training-video mean b)."""

    output: dict[tuple[int, int, int], tuple[float, float]] = {}
    for row in signed_rows:
        key = (int(row["fold"]), int(row["subject"]), int(row["video"]))
        value = (float(row["true_b_seconds"]), float(row["video_mean_b_seconds"]))
        if key in output and not np.allclose(output[key], value, atol=1e-10):
            raise RuntimeError(f"Inconsistent signed target lookup at {key}")
        output[key] = value
    return output


def calibration_videos(rows: Sequence[dict[str, str]], budget: int) -> tuple[int, ...]:
    evaluation = {int(row["video"]) for row in rows}
    calibration = tuple(video for video in ALL_VIDEOS if video not in evaluation)
    if len(calibration) != budget:
        raise RuntimeError(
            f"Expected {budget} calibration videos, found {len(calibration)}"
        )
    return calibration


def residual_offset(
    lookup: dict[tuple[int, int, int], tuple[float, float]],
    *,
    fold: int,
    subject: int,
    calibration: Sequence[int],
    shrinkage: float,
) -> float:
    residual_sum = sum(
        lookup[(fold, subject, int(video))][0]
        - lookup[(fold, subject, int(video))][1]
        for video in calibration
    )
    return float(residual_sum / (len(calibration) + float(shrinkage)))


def _participant_aggregate(
    detail: Sequence[dict[str, object]],
) -> dict[tuple[int, str, int], dict[str, float]]:
    metrics = (
        "video_only_b_mae_seconds",
        "own_identity_b_mae_seconds",
        "all_donor_mean_b_mae_seconds",
        "deranged_identity_b_mae_seconds",
        "own_gain_vs_video_seconds",
        "own_gain_vs_all_donors_seconds",
        "own_gain_vs_deranged_seconds",
    )
    grouped: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in detail:
        grouped[(int(row["budget"]), str(row["scope"]), int(row["subject"]))].append(
            row
        )
    return {
        key: {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in metrics
        }
        for key, rows in grouped.items()
    }


def matched_identity_transfer(
    signed_rows: Sequence[dict[str, str]],
    categories: np.ndarray,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
]:
    """Compare own and donor residuals using identical calibration videos."""

    lookup = signed_lookup(signed_rows)
    grouped: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in signed_rows:
        grouped[(int(row["fold"]), int(row["budget"]), int(row["repeat"]))].append(
            row
        )

    rng = np.random.default_rng(seed)
    detail: list[dict[str, object]] = []
    skipped: dict[tuple[int, str], int] = defaultdict(int)
    matched_offset_checks = 0
    for (fold, budget, repeat), group in sorted(grouped.items()):
        by_subject: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in group:
            by_subject[int(row["subject"])].append(row)
        subjects = sorted(by_subject)
        assignment = random_derangement(len(subjects), rng)
        deranged = {
            subjects[position]: subjects[int(assignment[position])]
            for position in range(len(subjects))
        }

        for subject in subjects:
            evaluation_rows = by_subject[subject]
            calibration = calibration_videos(evaluation_rows, budget)
            shrinkages = {float(row["selected_shrinkage"]) for row in evaluation_rows}
            if len(shrinkages) != 1:
                raise RuntimeError("Selected shrinkage varies within a receiver cell")
            shrinkage = shrinkages.pop()
            own_offset = residual_offset(
                lookup,
                fold=fold,
                subject=subject,
                calibration=calibration,
                shrinkage=shrinkage,
            )
            saved_offsets = np.asarray(
                [
                    float(row["calibrated_b_seconds"])
                    - float(row["video_mean_b_seconds"])
                    for row in evaluation_rows
                ],
                dtype=np.float64,
            )
            if not np.allclose(saved_offsets, own_offset, atol=1e-9):
                raise RuntimeError("Rebuilt own residual does not match signed correction")
            matched_offset_checks += 1

            other_subjects = [value for value in subjects if value != subject]
            donor_offsets = {
                donor: residual_offset(
                    lookup,
                    fold=fold,
                    subject=donor,
                    calibration=calibration,
                    shrinkage=shrinkage,
                )
                for donor in other_subjects
            }
            scopes: list[tuple[str, list[dict[str, str]]]] = [
                ("all", evaluation_rows)
            ]
            for category, name in enumerate(CATEGORY_NAMES):
                selected = [
                    row
                    for row in evaluation_rows
                    if int(categories[int(row["video"]) - 1]) == category
                ]
                if selected:
                    scopes.append((name, selected))
                else:
                    skipped[(budget, name)] += 1

            for scope, selected in scopes:
                target = np.asarray(
                    [float(row["true_b_seconds"]) for row in selected],
                    dtype=np.float64,
                )
                video_mean = np.asarray(
                    [float(row["video_mean_b_seconds"]) for row in selected],
                    dtype=np.float64,
                )
                video_mae = float(np.abs(video_mean - target).mean())
                own_mae = float(np.abs(video_mean + own_offset - target).mean())
                donor_mae = float(
                    np.mean(
                        [
                            np.abs(video_mean + donor_offsets[donor] - target).mean()
                            for donor in other_subjects
                        ]
                    )
                )
                deranged_subject = deranged[subject]
                deranged_offset = donor_offsets[deranged_subject]
                deranged_mae = float(
                    np.abs(video_mean + deranged_offset - target).mean()
                )
                detail.append(
                    {
                        "fold": fold,
                        "budget": budget,
                        "repeat": repeat,
                        "subject": subject,
                        "scope": scope,
                        "evaluation_videos": len(selected),
                        "calibration_videos": calibration,
                        "selected_shrinkage": shrinkage,
                        "video_only_b_mae_seconds": video_mae,
                        "own_identity_b_mae_seconds": own_mae,
                        "all_donor_mean_b_mae_seconds": donor_mae,
                        "deranged_identity_b_mae_seconds": deranged_mae,
                        "own_gain_vs_video_seconds": video_mae - own_mae,
                        "own_gain_vs_all_donors_seconds": donor_mae - own_mae,
                        "own_gain_vs_deranged_seconds": deranged_mae - own_mae,
                    }
                )

    participant_values = _participant_aggregate(detail)
    aggregate_rows: list[dict[str, object]] = []
    summary_budgets: dict[str, object] = {}
    metrics = (
        "video_only_b_mae_seconds",
        "own_identity_b_mae_seconds",
        "all_donor_mean_b_mae_seconds",
        "deranged_identity_b_mae_seconds",
    )
    effects = (
        "own_gain_vs_video_seconds",
        "own_gain_vs_all_donors_seconds",
        "own_gain_vs_deranged_seconds",
    )
    for budget in CALIBRATION_BUDGETS:
        summary_budgets[str(budget)] = {}
        for scope in ("all", *CATEGORY_NAMES):
            keys = [
                key
                for key in participant_values
                if key[0] == budget and key[1] == scope
            ]
            if not keys:
                continue
            by_subject = {key[2]: participant_values[key] for key in keys}
            row: dict[str, object] = {
                "budget": budget,
                "scope": scope,
                "participants": len(by_subject),
                "participant_repeat_cells": int(
                    np.sum(
                        [
                            int(item["budget"]) == budget
                            and str(item["scope"]) == scope
                            for item in detail
                        ]
                    )
                ),
                "skipped_fully_calibrated_category_cells": (
                    skipped[(budget, scope)] if scope != "all" else 0
                ),
            }
            scope_summary: dict[str, object] = {
                "participants": len(by_subject),
                "skipped_fully_calibrated_category_cells": row[
                    "skipped_fully_calibrated_category_cells"
                ],
            }
            for metric in metrics:
                value = float(
                    np.mean([subject_values[metric] for subject_values in by_subject.values()])
                )
                row[metric] = value
                scope_summary[metric] = value
            for effect_position, effect in enumerate(effects):
                values = {
                    subject: subject_values[effect]
                    for subject, subject_values in by_subject.items()
                }
                mean, ci = participant_bootstrap(
                    values,
                    repeats=bootstrap_repeats,
                    seed=(
                        seed
                        + budget * 1000
                        + effect_position * 100
                        + len(scope)
                    ),
                )
                row[effect] = mean
                row[f"{effect}_ci95_low"] = ci[0]
                row[f"{effect}_ci95_high"] = ci[1]
                row[f"{effect}_participants_positive"] = int(
                    np.sum(np.asarray(list(values.values())) > 0)
                )
                scope_summary[effect] = {
                    "mean": mean,
                    "participant_bootstrap_percentile_ci95": ci,
                    "participants_positive": row[
                        f"{effect}_participants_positive"
                    ],
                }
            aggregate_rows.append(row)
            summary_budgets[str(budget)][scope] = scope_summary

    summary = {
        "target": "fold-safe signed bias b_pv on held-out videos",
        "inference_unit": "participant; calibration repetitions are averaged within participant",
        "donor_matching": (
            "For each receiver repetition, every donor residual uses the receiver's "
            "exact calibration-video set, the same outer fold, and the same frozen "
            "fold/budget shrinkage value. Donors are other outer-held-out participants."
        ),
        "category_empty_cell_rule": (
            "A participant-repeat-category cell is skipped when all three videos in "
            "that category are calibration videos; skipped counts are reported."
        ),
        "matched_own_offset_checks": matched_offset_checks,
        "budgets": summary_budgets,
        "boundary": (
            "This tests cross-video transfer of a constant signed residual. It does "
            "not establish persistence of local w(t), cross-session identity, or a "
            "person-intrinsic timing trait."
        ),
    }
    return aggregate_rows, summary, detail


def category_statistic(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    category_means = values.mean(axis=0)
    return float(np.square(category_means - category_means.mean()).sum())


def blocked_category_permutation(
    matrix: np.ndarray, *, repeats: int, seed: int
) -> dict[str, object]:
    """Permutation test of category heterogeneity within participant blocks."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CATEGORY_NAMES):
        raise ValueError("Expected participant x five-category matrix")
    if not np.isfinite(values).all():
        raise ValueError("Category matrix contains missing or non-finite values")
    observed = category_statistic(values)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(repeats):
        permuted = np.stack([rng.permutation(row) for row in values])
        exceedances += int(category_statistic(permuted) >= observed - 1e-15)
    return {
        "participants": int(values.shape[0]),
        "observed_sum_squared_category_mean_deviation": observed,
        "permutation_repeats": repeats,
        "participant_blocked_permutation_p": float(
            (1 + exceedances) / (repeats + 1)
        ),
        "category_mean_gains_seconds": {
            category: float(values[:, position].mean())
            for position, category in enumerate(CATEGORY_NAMES)
        },
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float((total - rank) * values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def category_heterogeneity(
    detail: Sequence[dict[str, object]], *, repeats: int, seed: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in detail:
        scope = str(row["scope"])
        if scope not in CATEGORY_NAMES:
            continue
        grouped[(int(row["subject"]), int(row["budget"]), scope)].append(
            float(row["own_gain_vs_video_seconds"])
        )
    participant_budget = {
        key: float(np.mean(values)) for key, values in grouped.items()
    }
    participants = sorted({key[0] for key in participant_budget})
    budget_results: list[dict[str, object]] = []
    budget_p_values: list[float] = []
    for position, budget in enumerate(CALIBRATION_BUDGETS):
        matrix = np.asarray(
            [
                [
                    participant_budget[(subject, budget, category)]
                    for category in CATEGORY_NAMES
                ]
                for subject in participants
            ],
            dtype=np.float64,
        )
        result = blocked_category_permutation(
            matrix, repeats=repeats, seed=seed + position
        )
        result["analysis"] = "budget_specific_exploratory"
        result["budget"] = budget
        budget_results.append(result)
        budget_p_values.append(float(result["participant_blocked_permutation_p"]))
    adjusted = holm_adjust(budget_p_values)
    for result, value in zip(budget_results, adjusted):
        result["holm_adjusted_p_across_four_budgets"] = value

    global_matrix = np.asarray(
        [
            [
                np.mean(
                    [
                        participant_budget[(subject, budget, category)]
                        for budget in CALIBRATION_BUDGETS
                    ]
                )
                for category in CATEGORY_NAMES
            ]
            for subject in participants
        ],
        dtype=np.float64,
    )
    global_result = blocked_category_permutation(
        global_matrix, repeats=repeats, seed=seed + 100
    )
    global_result["analysis"] = "global_across_budgets"
    global_result["budget"] = "pooled"
    global_result["holm_adjusted_p_across_four_budgets"] = None

    results = [global_result, *budget_results]
    rows: list[dict[str, object]] = []
    for result in results:
        row = {
            key: value
            for key, value in result.items()
            if key != "category_mean_gains_seconds"
        }
        for category, value in result["category_mean_gains_seconds"].items():
            row[f"{category}_mean_gain_vs_video_seconds"] = value
        rows.append(row)
    interpretation = (
        "heterogeneity_detected_in_empirical_library"
        if float(global_result["participant_blocked_permutation_p"]) < 0.05
        else "category_stratified_estimates_varied_descriptively"
    )
    return rows, {
        "primary_effect": "own residual gain versus video-only prediction",
        "calibration_repetitions": "averaged within participant before inference",
        "test": (
            "10,000 within-participant category-label permutations of the sum of "
            "squared deviations among five category mean gains"
        ),
        "global": global_result,
        "budget_specific_exploratory": budget_results,
        "interpretation": interpretation,
        "boundary": (
            "The test concerns heterogeneity across the five categories represented "
            "by this empirical video library; it does not establish stable category traits."
        ),
    }


def sensing_model_rows() -> list[dict[str, object]]:
    rows = []
    for method in METHODS:
        metadata = MODEL_METADATA[method]
        rows.append(
            {
                "method": method,
                "model": metadata["label"],
                "inputs": metadata["inputs"],
                "dimension": (
                    metadata["dimension"]
                    if metadata["dimension"] is not None
                    else "fold-selected / not fixed"
                ),
                "ridge_alpha_candidates": (
                    "" if method == "video_mean" else ";".join(f"{x:g}" for x in RIDGE_ALPHAS)
                ),
            }
        )
    return rows


def _compact_selections(
    selections: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "outer_fold": int(selection["outer_fold"]),
            "selected_alphas": selection["selected_alphas"],
            "blockwise_residual_selection": selection[
                "blockwise_residual_selection"
            ],
            "feature_dimensions": selection["feature_dimensions"],
        }
        for selection in selections
    ]


def participant_fold_plan(
    repeat: int, *, seed: int, n_folds: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the primary split or a deterministic shuffled participant split."""

    if repeat == 0:
        return outer_subject_folds()
    subjects = np.arange(1, 25, dtype=np.int16)
    rng = np.random.default_rng(seed + repeat * 104729)
    shuffled = rng.permutation(subjects)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        validation = np.sort(shuffled[fold::n_folds]).astype(np.int16)
        training = np.sort(subjects[~np.isin(subjects, validation)]).astype(
            np.int16
        )
        folds.append((training, validation))
    held_out = np.concatenate([validation for _, validation in folds])
    if sorted(int(value) for value in held_out) != list(range(1, 25)):
        raise RuntimeError("Repeated grouped-CV fold plan is incomplete")
    return folds


def fold_plan_digest(
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> str:
    payload = [
        [int(value) for value in np.asarray(validation)]
        for _, validation in folds
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_fold_target_cache(
    index,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    cache: Path,
) -> np.ndarray:
    """Build fold-specific q targets for an alternate participant partition."""

    target_dir = cache / "phase_sensing_targets"
    target_dir.mkdir(parents=True, exist_ok=False)
    matrix = np.full((24, 15), np.nan, dtype=np.float64)
    assigned = np.zeros(24, dtype=bool)
    warp_config = WarpConfig()
    for fold, (training, validation) in enumerate(folds):
        values = build_outer_phase_targets(
            index,
            training,
            validation,
            warp_config=warp_config,
            template_radius=3,
        )
        lag = np.asarray(values["lag"], dtype=np.float32)
        np.savez_compressed(target_dir / f"fold_{fold}.npz", lag=lag)
        for subject in validation:
            subject_index = int(subject) - 1
            if assigned[subject_index]:
                raise RuntimeError("Participant appears in two outer folds")
            for video in ALL_VIDEOS:
                rows = trial_rows(index, int(subject), int(video))
                matrix[subject_index, video - 1] = float(
                    np.abs(lag[rows]).mean()
                )
            assigned[subject_index] = True
    if not assigned.all() or not np.isfinite(matrix).all():
        raise RuntimeError("Repeated grouped-CV target matrix is incomplete")
    return matrix


def link_sensing_features(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    os.symlink(source / "cbramod", destination / "cbramod")
    os.symlink(
        source / "handcrafted_eeg_pooled.npy",
        destination / "handcrafted_eeg_pooled.npy",
    )
    os.symlink(
        source / "handcrafted_fnirs_pooled.npy",
        destination / "handcrafted_fnirs_pooled.npy",
    )


def prediction_summary_rows(
    predictions: np.ndarray,
    target: np.ndarray,
    selections: Sequence[dict[str, object]],
    *,
    repeat: int,
    partition_sha256: str,
    bootstrap_repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    video_position = METHODS.index("video_mean")
    video_error = np.abs(predictions[video_position] - target)
    rows: list[dict[str, object]] = []
    for position, method in enumerate(METHODS):
        error = np.abs(predictions[position] - target)
        participant_gain = (video_error - error).mean(axis=1)
        gain, interval = participant_bootstrap(
            {
                participant + 1: float(value)
                for participant, value in enumerate(participant_gain)
            },
            repeats=bootstrap_repeats,
            seed=seed + repeat * 1000 + position,
        )
        row: dict[str, object] = {
            "repeat": repeat,
            "partition_sha256": partition_sha256,
            "outer_folds": 5,
            "inner_folds": 4,
            "method": method,
            "model": MODEL_METADATA[method]["label"],
            "participant_macro_mae_seconds": float(error.mean(axis=1).mean()),
            "gain_vs_video_mean_seconds": gain,
            "gain_vs_video_mean_ci95_low": interval[0],
            "gain_vs_video_mean_ci95_high": interval[1],
            "participants_positive_gain": int(np.sum(participant_gain > 0)),
            "participant_gain_min_seconds": float(participant_gain.min()),
            "participant_gain_median_seconds": float(
                np.median(participant_gain)
            ),
            "participant_gain_max_seconds": float(participant_gain.max()),
        }
        if method == "context_plus_blockwise_residual":
            row["disabled_sensor_increment_outer_folds"] = int(
                np.sum(
                    [
                        selection["blockwise_residual_selection"]["block"]
                        == "disabled"
                        for selection in selections
                    ]
                )
            )
        rows.append(row)
    return rows


def repeated_grouped_cv_sensitivity(
    index,
    primary_matrix: np.ndarray,
    primary_predictions: np.ndarray,
    primary_selections: Sequence[dict[str, object]],
    feature_cache: Path,
    *,
    repeats: int,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Repeat fully participant-grouped nested CV under alternate fold allocations."""

    if repeats < 1:
        raise ValueError("grouped_cv_repeats must be at least one")
    inference_repeats = max(100, min(1000, int(bootstrap_repeats)))
    rows: list[dict[str, object]] = []
    primary_folds = participant_fold_plan(0, seed=seed)
    rows.extend(
        prediction_summary_rows(
            primary_predictions,
            primary_matrix,
            primary_selections,
            repeat=0,
            partition_sha256=fold_plan_digest(primary_folds),
            bootstrap_repeats=inference_repeats,
            seed=seed,
        )
    )
    for repeat in range(1, repeats):
        folds = participant_fold_plan(repeat, seed=seed)
        repeat_cache = feature_cache / f"grouped_cv_repeat_{repeat}"
        link_sensing_features(feature_cache, repeat_cache)
        target = build_fold_target_cache(index, folds, repeat_cache)
        predictions, _, _, selections, _ = run_nested_sensing(
            index,
            target,
            repeat_cache,
            bootstrap_repeats=inference_repeats,
            outer_folds=folds,
            seed=seed + repeat * 10000,
        )
        rows.extend(
            prediction_summary_rows(
                predictions,
                target,
                selections,
                repeat=repeat,
                partition_sha256=fold_plan_digest(folds),
                bootstrap_repeats=inference_repeats,
                seed=seed,
            )
        )
        print(
            f"repeated participant-grouped CV partition {repeat + 1}/{repeats} complete",
            flush=True,
        )

    stack_rows = [
        row
        for row in rows
        if row["method"] == "context_plus_blockwise_residual"
    ]
    gains = np.asarray(
        [float(row["gain_vs_video_mean_seconds"]) for row in stack_rows],
        dtype=np.float64,
    )
    return rows, {
        "design": (
            f"{repeats} repetitions of five-fold outer participant-grouped CV "
            "with four-fold inner participant-grouped model selection; repeat 0 "
            "is the primary allocation and later repeats use deterministic shuffles"
        ),
        "reference_and_target_rebuilt_per_partition": True,
        "participants": 24,
        "videos_per_participant": 15,
        "repeats": repeats,
        "outer_folds_per_repeat": 5,
        "inner_folds": 4,
        "aggregate_interval_bootstrap_repeats": inference_repeats,
        "primary_route": "context_plus_blockwise_residual",
        "primary_route_gain_vs_video_mean_seconds": {
            "minimum_across_partitions": float(gains.min()),
            "maximum_across_partitions": float(gains.max()),
            "partitions_positive": int(np.sum(gains > 0)),
            "partitions_total": int(len(gains)),
        },
        "boundary": (
            "This is fold-allocation sensitivity for the released 24 participants, "
            "not evidence of equivalence, external generalization, or user benefit."
        ),
    }


def primary_antialias_sensing(
    index,
    primary_matrix: np.ndarray,
    feature_cache: Path,
    innovation_cache: Path,
    anti_alias_cache: Path,
    revision4_dir: Path,
    *,
    bootstrap_repeats: int,
    grouped_cv_repeats: int,
    run_fold_sensitivity: bool,
    seed: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Run strict anti-aliased EEG with reservation-aware fNIRS."""

    with tempfile.TemporaryDirectory(prefix="revision6_sensing_", dir="/tmp") as raw:
        cache = Path(raw)
        os.symlink(anti_alias_cache / "cbramod", cache / "cbramod")
        os.symlink(
            anti_alias_cache / "handcrafted_eeg_pooled.npy",
            cache / "handcrafted_eeg_pooled.npy",
        )
        os.symlink(
            innovation_cache / "phase_sensing_targets",
            cache / "phase_sensing_targets",
        )
        build_reservation_aware_fnirs(
            index,
            feature_cache,
            innovation_cache,
            cache / "handcrafted_fnirs_pooled.npy",
        )
        predictions, _, nested_summary, selections, _ = run_nested_sensing(
            index,
            primary_matrix,
            cache,
            bootstrap_repeats=bootstrap_repeats,
            seed=seed,
        )

        stack_position = METHODS.index("context_plus_blockwise_residual")
        video_position = METHODS.index("video_mean")
        stack_participant_gain = (
            np.abs(predictions[video_position] - primary_matrix)
            - np.abs(predictions[stack_position] - primary_matrix)
        ).mean(axis=1)
        sensing_decision_rows = threshold_decision_rows(
            stack_participant_gain,
            analysis="sensing_increment_vs_video_mean",
            route="context_plus_blockwise_residual",
            unit="seconds_mae_gain",
        )
        repeated_rows: list[dict[str, object]] = []
        repeated_summary: dict[str, object] | None = None
        if run_fold_sensitivity:
            repeated_rows, repeated_summary = repeated_grouped_cv_sensitivity(
                index,
                primary_matrix,
                predictions,
                selections,
                cache,
                repeats=grouped_cv_repeats,
                bootstrap_repeats=bootstrap_repeats,
                seed=seed + 4000,
            )

    revision4 = json.loads(
        (revision4_dir / "summary.json").read_text(encoding="utf-8")
    )
    boxcar = revision4["reservation_aware_sensing"][
        "nested_matched_target_sensing"
    ]
    boxcar_methods = boxcar["methods"]
    boxcar_selections = boxcar["selections"]
    video_error = np.abs(predictions[METHODS.index("video_mean")] - primary_matrix)
    rows: list[dict[str, object]] = []
    for position, method in enumerate(METHODS):
        metadata = MODEL_METADATA[method]
        error = np.abs(predictions[position] - primary_matrix)
        gain = video_error - error
        ci = paired_crossed_interval(
            gain,
            repeats=bootstrap_repeats,
            seed=seed + 1000 + position,
        )
        boxcar_mae = float(boxcar_methods[method]["participant_macro_mae_seconds"])
        primary_mae = float(error.mean(axis=1).mean())
        rows.append(
            {
                "method": method,
                "model": metadata["label"],
                "inputs": metadata["inputs"],
                "dimension": (
                    metadata["dimension"]
                    if metadata["dimension"] is not None
                    else "fold-selected / not fixed"
                ),
                "primary_antialias_reservation_mae_seconds": primary_mae,
                "gain_vs_video_mean_seconds": float(gain.mean()),
                "gain_vs_video_mean_ci95_low": ci[0],
                "gain_vs_video_mean_ci95_high": ci[1],
                "boxcar_reservation_mae_seconds": boxcar_mae,
                "primary_minus_boxcar_mae_seconds": primary_mae - boxcar_mae,
            }
        )

    primary_blocks = [
        selection["blockwise_residual_selection"]["block"]
        for selection in selections
    ]
    boxcar_blocks = [
        selection["blockwise_residual_selection"]["block"]
        for selection in boxcar_selections
    ]
    eeg_differences = [
        abs(float(row["primary_minus_boxcar_mae_seconds"]))
        for row in rows
        if bool(MODEL_METADATA[str(row["method"])]["contains_eeg"])
    ]
    non_eeg_differences = [
        abs(float(row["primary_minus_boxcar_mae_seconds"]))
        for row in rows
        if not bool(MODEL_METADATA[str(row["method"])]["contains_eeg"])
    ]
    if max(non_eeg_differences) > 1e-12:
        raise RuntimeError(
            "A non-EEG comparator changed between the anti-alias and boxcar runs; "
            "the target, fNIRS features, or fold protocol is not matched"
        )
    stack_row = next(
        row for row in rows if row["method"] == "context_plus_blockwise_residual"
    )
    summary = {
        "primary_pipeline": (
            "Strict 80-Hz-passband/100-Hz-stopband anti-aliased EEG decimation "
            "combined with reservation-aware fNIRS pooling."
        ),
        "implementation_sensitivity": "five-sample boxcar reduction with the same reservation-aware fNIRS",
        "target": "fold-matched primary q matrix from phase_trait_matrix.npz",
        "bootstrap_repeats": bootstrap_repeats,
        "ridge_alpha_candidates": list(RIDGE_ALPHAS),
        "nested_matched_target_sensing": {
            **nested_summary,
            "feature_pooling": {
                "cbramod": "anti-aliased array-order CBraMod trial mean and s.d. (800-D)",
                "handcrafted_eeg": "anti-aliased spectral/Hjorth trial mean and s.d. (180-D)",
                "fnirs": (
                    "reservation-aware trial mean/s.d. plus distributed-lag "
                    "cross-moments against fold-safe reference-gradient energy (1,260-D)"
                ),
            },
            "selections": _compact_selections(selections),
        },
        "blockwise_residual_selection_primary": primary_blocks,
        "blockwise_residual_selection_boxcar": boxcar_blocks,
        "fold_block_selections_unchanged": primary_blocks == boxcar_blocks,
        "maximum_absolute_mae_change_eeg_affected_methods_seconds": float(
            max(eeg_differences)
        ),
        "blockwise_stack_gain_vs_video_mean_seconds": stack_row[
            "gain_vs_video_mean_seconds"
        ],
        "blockwise_stack_gain_vs_video_mean_ci95": [
            stack_row["gain_vs_video_mean_ci95_low"],
            stack_row["gain_vs_video_mean_ci95_high"],
        ],
        "decision_threshold_curve": {
            "route": "context_plus_blockwise_residual",
            "thresholds_seconds_mae_gain": list(DECISION_THRESHOLDS),
            "interval": "participant-level Wilson 95% interval",
            "interpretation": (
                "Conditional benefit and degradation rates across hypothetical "
                "thresholds; no threshold was prespecified and the curve is not "
                "an equivalence or practical-utility analysis."
            ),
        },
        "repeated_grouped_cv_sensitivity": repeated_summary,
        "decision": "NO_DEMONSTRATED_INCREMENT_FOR_THE_EVALUATED_PIPELINE",
        "boundary": (
            "The result is specific to released arrays, array-order adaptation, "
            "reservation masks, evaluated features/models, q as target, and the "
            "participant-held-out same-trial offline protocol."
        ),
    }
    return rows, summary, sensing_decision_rows, repeated_rows


def main() -> None:
    args = parse_args()
    data_root = resolve(args.data_root)
    feature_cache = resolve(args.feature_cache)
    innovation_cache = resolve(args.innovation_cache)
    anti_alias_cache = resolve(args.anti_alias_cache)
    trait_dir = resolve(args.trait_dir)
    revision4_dir = resolve(args.revision4_dir)
    revision5_dir = resolve(args.revision5_dir)
    output_dir = resolve(args.output_dir)
    prepare_output(output_dir, args.overwrite)
    started = time.perf_counter()
    configuration = {
        key: portable_path(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "running",
            "started_at_utc": utc_now(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "configuration": configuration,
        },
    )

    index = load_innovation_index(data_root)
    categories = emotion_categories(data_root / "Targeted_emotions.txt")
    signed_rows = read_rows(revision4_dir / "signed_correction.csv")
    calibration_practical_rows = read_rows(
        revision5_dir / "calibration_practical_value.csv"
    )
    decision_rows = calibration_threshold_decision_rows(calibration_practical_rows)


    matched_rows, matched_summary, detail = matched_identity_transfer(
        signed_rows,
        categories,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 1000,
    )
    write_rows(output_dir / "matched_identity_transfer.csv", matched_rows)

    category_rows, category_summary = category_heterogeneity(
        detail,
        repeats=args.permutation_repeats,
        seed=args.seed + 2000,
    )
    write_rows(output_dir / "category_heterogeneity.csv", category_rows)

    model_rows = sensing_model_rows()
    write_rows(output_dir / "sensing_model_table.csv", model_rows)
    sensing_summary: dict[str, object] | None = None
    repeated_rows: list[dict[str, object]] = []
    if not args.skip_sensing:
        target_cache = np.load(
            trait_dir / "phase_trait_matrix.npz", allow_pickle=False
        )
        primary_matrix = target_cache["mean_absolute_lag"].astype(np.float64)
        (
            sensing_rows,
            sensing_summary,
            sensing_decision_rows,
            repeated_rows,
        ) = primary_antialias_sensing(
            index,
            primary_matrix,
            feature_cache,
            innovation_cache,
            anti_alias_cache,
            revision4_dir,
            bootstrap_repeats=args.bootstrap_repeats,
            grouped_cv_repeats=args.grouped_cv_repeats,
            run_fold_sensitivity=not args.skip_fold_sensitivity,
            seed=args.seed + 3000,
        )
        write_rows(
            output_dir / "primary_antialias_sensing_results.csv", sensing_rows
        )

        decision_rows.extend(sensing_decision_rows)

    write_rows(output_dir / "decision_threshold_curves.csv", decision_rows)
    if repeated_rows:
        write_rows(output_dir / "repeated_grouped_cv.csv", repeated_rows)

    summary = {
        "matched_cross_video_identity_transfer": matched_summary,
        "category_heterogeneity": category_summary,
        "schema_version": "revision6-analysis-v2",
        "revision": 6,
        "decision_threshold_curves": {
            "source_csv": "artifacts/revision6/decision_threshold_curves.csv",
            "threshold_grid": list(DECISION_THRESHOLDS),
            "analyses": [
                "signed_calibration_vs_video_only",
                *(
                    []
                    if args.skip_sensing
                    else ["sensing_increment_vs_video_mean"]
                ),
            ],
            "interval": "participant-level Wilson 95% interval",
            "boundary": (
                "Thresholds are a transparent sensitivity grid because no smallest "
                "worthwhile improvement or acceptable degradation threshold was "
                "specified before analysis. The curves do not establish equivalence "
                "or practical utility."
            ),
        },
        "primary_antialias_reservation_sensing": sensing_summary,
        "reporting_boundary": (
            "Only aggregate Revision-6 results are written. Participant-level "
            "targets, predictions, calibration profiles, and participant-indexed "
            "manifests are excluded from the anonymous artifact."
        ),
    }
    write_json(output_dir / "summary.json", summary)

    generated = sorted(
        item
        for item in MANAGED_FILES
        if item != "run_manifest.json" and (output_dir / item).exists()
    )
    artifact_registry = [
        {
            "uri": (output_dir / name).relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(output_dir / name),
        }
        for name in generated
    ]
    previous = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "complete",
            "started_at_utc": previous["started_at_utc"],
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "configuration": configuration,
            "generated_files": generated,
            "artifact_registry": artifact_registry,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
