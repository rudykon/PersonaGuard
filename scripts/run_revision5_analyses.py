#!/usr/bin/env python3
"""Revision-5 audits for identity transfer, reference uncertainty, and utility.

The script does not overwrite earlier revision artifacts.  It reuses their
committed fold-safe targets, rebuilds the reference-aware bootstrap with
original-identity exclusion, and adds only analyses that the released data can
support.  No new-session, demographic, hardware, or provider-side QC evidence
is inferred.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
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
from merps.innovation.normative_dynamics import WarpConfig, decompose_trial  # noqa: E402
from run_revision2_analyses import emotion_categories, safe_correlation, trial_rows  # noqa: E402
from run_revision3_analyses import balanced_variance_components, reliability_for_k  # noqa: E402
from run_revision4_analyses import (  # noqa: E402
    category_order,
    category_reliability,
    category_variance_components,
    reference_from_positions,
    target_cube,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_INNOVATION_CACHE = PROJECT_ROOT / "data" / "innovation_cache"
DEFAULT_REVISION2_DIR = PROJECT_ROOT / "artifacts" / "revision2"
DEFAULT_REVISION3_DIR = PROJECT_ROOT / "artifacts" / "revision3"
DEFAULT_REVISION4_DIR = PROJECT_ROOT / "artifacts" / "revision4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "revision5"
CALIBRATION_BUDGETS = (1, 2, 4, 8)
CATEGORY_NAMES = ("neutral", "happy", "fear", "sad", "relaxed")
MANAGED_FILES = {
    "calibration_practical_value.csv",
    "formula_target_audit.csv",
    "full_pipeline_bootstrap.csv",
    "identity_transfer.csv",
    "reference_ruler_audit.csv",
    "run_manifest.json",
    "summary.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--innovation-cache", type=Path, default=DEFAULT_INNOVATION_CACHE
    )
    parser.add_argument("--revision2-dir", type=Path, default=DEFAULT_REVISION2_DIR)
    parser.add_argument("--revision3-dir", type=Path, default=DEFAULT_REVISION3_DIR)
    parser.add_argument("--revision4-dir", type=Path, default=DEFAULT_REVISION4_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--full-pipeline-repeats", type=int, default=1000)
    parser.add_argument("--mc-repeats", type=int, default=1000)
    parser.add_argument(
        "--full-pipeline-workers",
        type=int,
        default=min(24, max(1, (os.cpu_count() or 2) // 2)),
    )
    parser.add_argument("--seed", type=int, default=20260729)
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


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
        draws[repeat] = float(np.mean([values[int(subject)] for subject in selected]))
    return observed, percentile_interval(draws)


def formula_target_audit(
    index, innovation_cache: Path, revision3_dir: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    committed = {
        (int(row["fold"]), int(row["subject"]), int(row["video"])): row
        for row in read_rows(revision3_dir / "target_definition_audit.csv")
        if row["role"] == "outer_validation_training_reference"
    }
    rows: list[dict[str, object]] = []
    maximum_q_difference = 0.0
    maximum_b_difference = 0.0
    violations = 0
    for fold, (_, validation_subjects) in enumerate(outer_subject_folds()):
        lag = np.load(
            innovation_cache / "phase_sensing_targets" / f"fold_{fold}.npz",
            allow_pickle=False,
        )["lag"].astype(np.float64)
        for subject in validation_subjects:
            for video in range(1, 16):
                selected = lag[trial_rows(index, int(subject), video)]
                manual_q = float(np.mean(np.abs(selected)))
                manual_b = float(np.mean(selected))
                saved = committed[(fold, int(subject), video)]
                saved_q = float(saved["q_pv_seconds"])
                saved_b = float(saved["b_pv_seconds"])
                q_difference = abs(manual_q - saved_q)
                b_difference = abs(manual_b - saved_b)
                invariant = manual_q >= abs(manual_b) - 1e-10
                violations += int(not invariant)
                maximum_q_difference = max(maximum_q_difference, q_difference)
                maximum_b_difference = max(maximum_b_difference, b_difference)
                rows.append(
                    {
                        "fold": fold,
                        "subject": int(subject),
                        "video": video,
                        "manual_q_seconds": manual_q,
                        "saved_q_seconds": saved_q,
                        "absolute_q_difference_seconds": q_difference,
                        "manual_b_seconds": manual_b,
                        "saved_b_seconds": saved_b,
                        "absolute_b_difference_seconds": b_difference,
                        "q_ge_abs_b": invariant,
                    }
                )
    if len(rows) != 360 or maximum_q_difference > 1e-10 or violations:
        raise RuntimeError(
            "Formula/target audit failed: "
            f"rows={len(rows)} max_q_diff={maximum_q_difference} "
            f"violations={violations}"
        )
    return rows, {
        "status": "VERIFIED_NO_SOURCE_CHANGE_REQUIRED",
        "interpretation": (
            "The reviewer's missing absolute-value concern was a PDF/text-"
            "extraction misread. Source equations and saved targets use "
            "q_pv = mean_t |t-w_pv(t)| and trace MAE uses an absolute value."
        ),
        "outer_validation_trials_checked": len(rows),
        "maximum_manual_saved_q_difference_seconds": maximum_q_difference,
        "maximum_manual_saved_b_difference_seconds": maximum_b_difference,
        "q_ge_abs_b_violations": violations,
    }


def random_derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("A derangement needs at least two identities")
    original = np.arange(size)
    for _ in range(1000):
        candidate = rng.permutation(size)
        if np.all(candidate != original):
            return candidate
    # Deterministic fallback, also useful for very small test cases.
    return np.roll(original, 1)


def identity_transfer(
    signed_rows: list[dict[str, str]],
    categories: np.ndarray,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in signed_rows:
        grouped[(int(row["fold"]), int(row["budget"]), int(row["repeat"]))].append(row)

    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for (fold, budget, repeat), group in sorted(grouped.items()):
        by_subject: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in group:
            by_subject[int(row["subject"])].append(row)
        subjects = sorted(by_subject)
        offsets: dict[int, float] = {}
        for subject in subjects:
            candidate = np.asarray(
                [
                    float(row["calibrated_b_seconds"])
                    - float(row["video_mean_b_seconds"])
                    for row in by_subject[subject]
                ],
                dtype=np.float64,
            )
            if np.ptp(candidate) > 1e-9:
                raise RuntimeError("Calibrated additive offset is not constant")
            offsets[subject] = float(np.median(candidate))
        assignment = random_derangement(len(subjects), rng)
        shuffled = {
            subjects[position]: subjects[int(assignment[position])]
            for position in range(len(subjects))
        }

        for subject in subjects:
            evaluation = by_subject[subject]
            other_subjects = [value for value in subjects if value != subject]
            scopes: list[tuple[str, list[dict[str, str]]]] = [("all", evaluation)]
            for category, name in enumerate(CATEGORY_NAMES):
                selected = [
                    row
                    for row in evaluation
                    if int(categories[int(row["video"]) - 1]) == category
                ]
                if selected:
                    scopes.append((name, selected))
            for scope, selected in scopes:
                target = np.asarray(
                    [float(row["true_b_seconds"]) for row in selected]
                )
                video_mean = np.asarray(
                    [float(row["video_mean_b_seconds"]) for row in selected]
                )
                own_prediction = video_mean + offsets[subject]
                donor_errors = [
                    float(np.abs(video_mean + offsets[donor] - target).mean())
                    for donor in other_subjects
                ]
                shuffled_subject = shuffled[subject]
                video_mae = float(np.abs(video_mean - target).mean())
                own_mae = float(np.abs(own_prediction - target).mean())
                donor_mae = float(np.mean(donor_errors))
                shuffled_mae = float(
                    np.abs(video_mean + offsets[shuffled_subject] - target).mean()
                )
                output.append(
                    {
                        "fold": fold,
                        "budget": budget,
                        "repeat": repeat,
                        "subject": subject,
                        "scope": scope,
                        "evaluation_videos": len(selected),
                        "selected_shrinkage": float(selected[0]["selected_shrinkage"]),
                        "own_residual_offset_seconds": offsets[subject],
                        "shuffled_donor_subject": shuffled_subject,
                        "video_only_b_mae_seconds": video_mae,
                        "own_identity_b_mae_seconds": own_mae,
                        "all_donor_mean_b_mae_seconds": donor_mae,
                        "shuffled_identity_b_mae_seconds": shuffled_mae,
                        "own_gain_vs_video_seconds": video_mae - own_mae,
                        "own_gain_vs_all_donors_seconds": donor_mae - own_mae,
                        "own_gain_vs_shuffled_seconds": shuffled_mae - own_mae,
                    }
                )

    summary: dict[str, object] = {}
    for budget in CALIBRATION_BUDGETS:
        summary[str(budget)] = {}
        for scope in ("all", *CATEGORY_NAMES):
            selected = [
                row
                for row in output
                if int(row["budget"]) == budget and row["scope"] == scope
            ]
            if not selected:
                continue
            participant_values: dict[int, dict[str, float]] = {}
            for subject in sorted({int(row["subject"]) for row in selected}):
                subject_rows = [
                    row for row in selected if int(row["subject"]) == subject
                ]
                participant_values[subject] = {
                    name: float(np.mean([float(row[name]) for row in subject_rows]))
                    for name in (
                        "video_only_b_mae_seconds",
                        "own_identity_b_mae_seconds",
                        "all_donor_mean_b_mae_seconds",
                        "shuffled_identity_b_mae_seconds",
                        "own_gain_vs_video_seconds",
                        "own_gain_vs_all_donors_seconds",
                        "own_gain_vs_shuffled_seconds",
                    )
                }
            effects: dict[str, object] = {}
            for effect_position, effect in enumerate(
                (
                    "own_gain_vs_video_seconds",
                    "own_gain_vs_all_donors_seconds",
                    "own_gain_vs_shuffled_seconds",
                )
            ):
                values = {
                    subject: metrics[effect]
                    for subject, metrics in participant_values.items()
                }
                mean, ci = participant_bootstrap(
                    values,
                    repeats=bootstrap_repeats,
                    seed=seed + budget * 1000 + effect_position * 100 + len(scope),
                )
                effects[effect] = {
                    "mean": mean,
                    "participant_bootstrap_percentile_ci95": ci,
                    "participants_positive": int(
                        np.sum(np.asarray(list(values.values())) > 0)
                    ),
                    "participants": len(values),
                }
            summary[str(budget)][scope] = {
                "participant_macro_mae_seconds": {
                    name: float(
                        np.mean([metrics[name] for metrics in participant_values.values()])
                    )
                    for name in (
                        "video_only_b_mae_seconds",
                        "own_identity_b_mae_seconds",
                        "all_donor_mean_b_mae_seconds",
                        "shuffled_identity_b_mae_seconds",
                    )
                },
                **effects,
            }
    return output, {
        "target": "fold-safe signed bias b_pv on held-out videos",
        "inference_unit": "participant",
        "design": (
            "Each participant's additive residual is estimated from independent "
            "calibration videos and evaluated on that participant's held-out "
            "videos. It is compared with the video-only prediction, every other "
            "held-out participant's residual, and a within-fold deranged identity."
        ),
        "boundary": (
            "This tests cross-video transfer of a constant signed residual. It "
            "does not establish persistence of the local same-trial warp w(t), "
            "cross-session identity, or a person-intrinsic affective clock."
        ),
        "budgets": summary,
    }


def calibration_burden(
    sampling_rows: list[dict[str, str]], budget: int
) -> dict[str, float]:
    durations = []
    for video in range(1, 16):
        values = [
            float(row["label_seconds"])
            for row in sampling_rows
            if int(row["video"]) == video
        ]
        durations.append(float(np.median(values)))
    totals = np.asarray(
        [sum(durations[index] for index in combination) for combination in itertools.combinations(range(15), budget)],
        dtype=np.float64,
    )
    return {
        "expected_minutes": float(totals.mean() / 60.0),
        "minimum_minutes": float(totals.min() / 60.0),
        "maximum_minutes": float(totals.max() / 60.0),
        "p05_minutes": float(np.quantile(totals, 0.05) / 60.0),
        "p95_minutes": float(np.quantile(totals, 0.95) / 60.0),
    }


def calibration_practical_value(
    signed_rows: list[dict[str, str]],
    sampling_rows: list[dict[str, str]],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    output: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    metrics = (
        "no_correction_trace_mae_units",
        "video_only_trace_mae_units",
        "calibrated_trace_mae_units",
        "oracle_b_trace_mae_units",
    )
    for budget in CALIBRATION_BUDGETS:
        selected = [row for row in signed_rows if int(row["budget"]) == budget]
        participants: dict[int, dict[str, float]] = {}
        for subject in range(1, 25):
            subject_rows = [row for row in selected if int(row["subject"]) == subject]
            values = {
                name: float(np.mean([float(row[name]) for row in subject_rows]))
                for name in metrics
            }
            gain = (
                values["video_only_trace_mae_units"]
                - values["calibrated_trace_mae_units"]
            )
            values["calibrated_gain_vs_video_units"] = gain
            values["calibrated_relative_gain_vs_video_percent"] = float(
                100.0 * gain / max(values["video_only_trace_mae_units"], 1e-12)
            )
            values["oracle_gap_units"] = (
                values["calibrated_trace_mae_units"]
                - values["oracle_b_trace_mae_units"]
            )
            participants[subject] = values
            output.append({"budget": budget, "subject": subject, **values})
        gain_values = {
            subject: values["calibrated_gain_vs_video_units"]
            for subject, values in participants.items()
        }
        mean_gain, gain_ci = participant_bootstrap(
            gain_values,
            repeats=bootstrap_repeats,
            seed=seed + budget * 100,
        )
        burden = calibration_burden(sampling_rows, budget)
        mean_metrics = {
            name: float(np.mean([values[name] for values in participants.values()]))
            for name in metrics
        }
        summary[str(budget)] = {
            **mean_metrics,
            "calibrated_gain_vs_video_units": mean_gain,
            "calibrated_gain_vs_video_percent_of_video_mae": float(
                100.0
                * mean_gain
                / max(mean_metrics["video_only_trace_mae_units"], 1e-12)
            ),
            "calibrated_gain_vs_video_participant_bootstrap_percentile_ci95": gain_ci,
            "calibrated_oracle_gap_units": float(
                np.mean([values["oracle_gap_units"] for values in participants.values()])
            ),
            "participants_improved_vs_video": int(
                np.sum(np.asarray(list(gain_values.values())) > 0)
            ),
            "participants_degraded_vs_video": int(
                np.sum(np.asarray(list(gain_values.values())) < 0)
            ),
            "worst_participant_gain_vs_video_units": float(
                np.min(list(gain_values.values()))
            ),
            "participant_median_gain_vs_video_units": float(
                np.median(list(gain_values.values()))
            ),
            "continuous_annotation_burden": burden,
            "smallest_worthwhile_improvement_units": None,
            "status": "PARTIAL_PRACTICAL_VALUE_UNDETERMINED",
        }
    return output, {
        "scale": "joystick labels on the released [1,255] range",
        "burden_scope": (
            "Continuous annotation time computed from released video durations; "
            "setup, instructions, breaks, and training time are unavailable and excluded."
        ),
        "decision": (
            "The incremental comparison is calibrated versus video-only correction. "
            "Because no smallest worthwhile improvement was specified before seeing "
            "the data, statistical support does not establish practical value."
        ),
        "budgets": summary,
    }


def icc_consistency(first: np.ndarray, second: np.ndarray) -> float:
    values = np.column_stack(
        [np.asarray(first, dtype=np.float64).reshape(-1), np.asarray(second, dtype=np.float64).reshape(-1)]
    )
    rows, columns = values.shape
    row_mean = values.mean(axis=1)
    column_mean = values.mean(axis=0)
    grand = float(values.mean())
    ms_rows = float(columns * np.square(row_mean - grand).sum() / (rows - 1))
    residual = values - row_mean[:, None] - column_mean[None, :] + grand
    ms_error = float(np.square(residual).sum() / ((rows - 1) * (columns - 1)))
    denominator = ms_rows + (columns - 1) * ms_error
    return float((ms_rows - ms_error) / denominator) if denominator > 0 else 0.0


def reference_ruler_audit(
    revision2_dir: Path, revision3_dir: Path, revision4_dir: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    loo = np.load(
        revision4_dir / "estimand_targets.npz", allow_pickle=False
    )["consensus"].astype(np.float64)
    common = np.load(
        revision3_dir / "matched_target_predictions.npz", allow_pickle=False
    )["target"].astype(np.float64)
    loo_profile = loo.mean(axis=1)
    common_profile = common.mean(axis=1)
    rows = [
        {
            "comparison": "leave-one-participant-out_23_vs_outer-training-common_19-or-20",
            "trial_mae_seconds": float(np.abs(loo - common).mean()),
            "trial_pearson": safe_correlation(loo, common),
            "trial_spearman": safe_correlation(loo, common, spearman=True),
            "trial_icc_consistency": icc_consistency(loo, common),
            "profile_mae_seconds": float(np.abs(loo_profile - common_profile).mean()),
            "profile_pearson": safe_correlation(loo_profile, common_profile),
            "profile_spearman": safe_correlation(
                loo_profile, common_profile, spearman=True
            ),
            "profile_icc_consistency": icc_consistency(
                loo_profile, common_profile
            ),
            "maximum_profile_shift_seconds": float(
                np.abs(loo_profile - common_profile).max()
            ),
        }
    ]
    revision2_summary = json.loads(
        (revision2_dir / "summary.json").read_text(encoding="utf-8")
    )
    size_summary = revision2_summary["reference_composition_sensitivity"]
    for size, values in size_summary["metrics"].items():
        rows.append(
            {
                "comparison": f"outer-training-subsample_{size}_vs_primary",
                "reference_participants": int(size),
                **{
                    f"{name}_mean": metric["mean"]
                    for name, metric in values.items()
                },
            }
        )
    return rows, {
        "common_panel_comparison": rows[0],
        "reference_size_sensitivity": size_summary,
        "external_reference_status": "NOT_AVAILABLE_IN_RELEASE",
        "boundary": (
            "The fold-common panel is independent of each held-out participant but "
            "is not an external cohort. Demographic and recruitment composition of "
            "the reference population is unavailable, preventing subgroup and "
            "transportability assessment."
        ),
    }


_FULL_CUBES: list[np.ndarray] | None = None
_FULL_GROUPS: list[np.ndarray] | None = None


def _init_full_worker(cubes: list[np.ndarray], groups: list[np.ndarray]) -> None:
    global _FULL_CUBES, _FULL_GROUPS
    _FULL_CUBES = cubes
    _FULL_GROUPS = groups


def cluster_reference_sources(selected: np.ndarray, focal: int) -> np.ndarray:
    selected = np.asarray(selected, dtype=np.int16)
    return selected[selected != selected[int(focal)]]


def _bootstrap_metrics(q: np.ndarray, sampled_groups: list[np.ndarray]) -> dict[str, float]:
    videos = np.concatenate(sampled_groups)
    simple_components = balanced_variance_components(q[:, videos])
    simple = reliability_for_k(simple_components, 15)
    category_cube = np.stack([q[:, group] for group in sampled_groups], axis=1)
    category_components = category_variance_components(category_cube)
    category = category_reliability(category_components, 3)
    return {
        "relative_g_15": simple["relative_g"],
        "absolute_phi_15": simple["absolute_phi"],
        "category_relative_g_15": category["relative_g"],
        "category_absolute_phi_15": category["absolute_phi"],
        "raw_participant_variance": simple_components["raw_participant"],
        "raw_video_variance": simple_components["raw_video"],
        "raw_category_participant_variance": category_components["raw_participant"],
        "raw_video_within_category_variance": category_components[
            "raw_video_within_category"
        ],
        "raw_participant_by_category_variance": category_components[
            "raw_participant_by_category"
        ],
    }


def _full_pipeline_draw(seed: int) -> dict[str, object]:
    if _FULL_CUBES is None or _FULL_GROUPS is None:
        raise RuntimeError("Full-pipeline worker is not initialized")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, 24, 24, dtype=np.int16)
    while len(np.unique(selected)) < 2:
        selected = rng.integers(0, 24, 24, dtype=np.int16)
    config = WarpConfig()
    q_legacy = np.empty((24, 15), dtype=np.float64)
    q_cluster = np.empty((24, 15), dtype=np.float64)
    pseudo_positions = np.arange(24)
    cluster_sizes = []
    contaminated = False
    for pseudo in range(24):
        legacy_sources = selected[pseudo_positions != pseudo]
        cluster_sources = cluster_reference_sources(selected, pseudo)
        contaminated = contaminated or bool(
            np.any(legacy_sources == selected[pseudo])
        )
        if not len(cluster_sources):
            raise RuntimeError("Cluster exclusion produced an empty reference")
        cluster_sizes.append(len(cluster_sources))
        for video, cube in enumerate(_FULL_CUBES):
            target = cube[selected[pseudo]]
            legacy_template = reference_from_positions(cube, legacy_sources)
            cluster_template = reference_from_positions(cube, cluster_sources)
            legacy = decompose_trial(target, legacy_template, config)
            cluster = decompose_trial(target, cluster_template, config)
            q_legacy[pseudo, video] = float(np.abs(legacy.consensus_lag).mean())
            q_cluster[pseudo, video] = float(np.abs(cluster.consensus_lag).mean())
    sampled_groups = [
        rng.choice(group, size=3, replace=True) for group in _FULL_GROUPS
    ]
    return {
        "legacy": _bootstrap_metrics(q_legacy, sampled_groups),
        "cluster_exclusion": _bootstrap_metrics(q_cluster, sampled_groups),
        "legacy_identity_contamination": contaminated,
        "unique_original_participants": int(len(np.unique(selected))),
        "cluster_reference_size_mean": float(np.mean(cluster_sizes)),
        "cluster_reference_size_minimum": int(np.min(cluster_sizes)),
        "cluster_reference_size_maximum": int(np.max(cluster_sizes)),
    }


def quantile_mc_error(
    values: np.ndarray, *, repeats: int, seed: int
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    low = np.empty(repeats, dtype=np.float64)
    high = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        draw = rng.choice(values, size=len(values), replace=True)
        low[repeat], high[repeat] = np.quantile(draw, (0.025, 0.975))
    return {
        "lower_endpoint_mcse": float(low.std(ddof=1)),
        "upper_endpoint_mcse": float(high.std(ddof=1)),
    }


def full_pipeline_bootstrap(
    index,
    categories: np.ndarray,
    *,
    repeats: int,
    workers: int,
    mc_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cubes = target_cube(index)
    groups = category_order(categories)
    seeds = [seed + repeat * 104729 for repeat in range(repeats)]
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        initializer=_init_full_worker,
        initargs=(cubes, groups),
    ) as executor:
        for draw, result in enumerate(
            executor.map(_full_pipeline_draw, seeds, chunksize=1), start=1
        ):
            diagnostics.append(result)
            for protocol in ("legacy", "cluster_exclusion"):
                rows.append(
                    {
                        "draw": draw,
                        "protocol": protocol,
                        **result[protocol],
                        "legacy_identity_contamination": result[
                            "legacy_identity_contamination"
                        ],
                        "unique_original_participants": result[
                            "unique_original_participants"
                        ],
                        "cluster_reference_size_mean": result[
                            "cluster_reference_size_mean"
                        ],
                        "cluster_reference_size_minimum": result[
                            "cluster_reference_size_minimum"
                        ],
                        "cluster_reference_size_maximum": result[
                            "cluster_reference_size_maximum"
                        ],
                    }
                )
            if draw % max(1, repeats // 10) == 0:
                print(f"revision5 full-pipeline bootstrap={draw}/{repeats}", flush=True)

    metrics = (
        "relative_g_15",
        "absolute_phi_15",
        "category_relative_g_15",
        "category_absolute_phi_15",
    )
    summary: dict[str, object] = {}
    for protocol_position, protocol in enumerate(("legacy", "cluster_exclusion")):
        selected_rows = [row for row in rows if row["protocol"] == protocol]
        protocol_summary: dict[str, object] = {}
        for metric_position, metric in enumerate(metrics):
            values = np.asarray([float(row[metric]) for row in selected_rows])
            protocol_summary[metric] = {
                "median": float(np.median(values)),
                "percentile_ci95": percentile_interval(values),
                "percentile_endpoint_monte_carlo_se": quantile_mc_error(
                    values,
                    repeats=mc_repeats,
                    seed=seed + protocol_position * 10000 + metric_position * 100,
                ),
                "above_070_fraction": float(np.mean(values > 0.70)),
            }
        raw_names = (
            "raw_participant_variance",
            "raw_video_variance",
            "raw_category_participant_variance",
            "raw_video_within_category_variance",
            "raw_participant_by_category_variance",
        )
        protocol_summary["negative_raw_component_counts"] = {
            name: int(
                np.sum(np.asarray([float(row[name]) for row in selected_rows]) < 0)
            )
            for name in raw_names
        }
        summary[protocol] = protocol_summary

    paired: dict[str, object] = {}
    for metric in metrics:
        legacy = np.asarray(
            [float(row[metric]) for row in rows if row["protocol"] == "legacy"]
        )
        cluster = np.asarray(
            [
                float(row[metric])
                for row in rows
                if row["protocol"] == "cluster_exclusion"
            ]
        )
        difference = cluster - legacy
        paired[metric] = {
            "cluster_minus_legacy_mean": float(difference.mean()),
            "paired_draw_percentile_interval": percentile_interval(difference),
            "maximum_absolute_draw_difference": float(np.abs(difference).max()),
        }
    return rows, {
        "draws": repeats,
        "workers": workers,
        "interval_type": "nonparametric percentile interval",
        "protocols": summary,
        "paired_protocol_sensitivity": paired,
        "legacy_identity_contamination_draw_fraction": float(
            np.mean(
                [bool(value["legacy_identity_contamination"]) for value in diagnostics]
            )
        ),
        "unique_original_participants": {
            "mean": float(
                np.mean([int(value["unique_original_participants"]) for value in diagnostics])
            ),
            "minimum": int(
                np.min([int(value["unique_original_participants"]) for value in diagnostics])
            ),
            "maximum": int(
                np.max([int(value["unique_original_participants"]) for value in diagnostics])
            ),
        },
        "cluster_exclusion_rule": (
            "For a focal pseudo-participant, every resampled copy of the same "
            "original participant is removed from the reference. Other original "
            "identities retain their cluster-bootstrap multiplicity."
        ),
    }


def load_reservation_aware_sensing_methods(revision4_dir: Path) -> dict[str, object]:
    revision4 = json.loads((revision4_dir / "summary.json").read_text(encoding="utf-8"))
    methods = revision4["reservation_aware_sensing"]["nested_matched_target_sensing"]["methods"]
    if not isinstance(methods, dict):
        raise RuntimeError("Revision 4 reservation-aware sensing methods are missing")
    return methods


def sanitize_cbramod_manifest(manifest: dict[str, object]) -> dict[str, object]:
    sanitized = dict(manifest)
    checkpoint = sanitized.get("checkpoint")
    if checkpoint is not None:
        sanitized["checkpoint"] = portable_path(Path(str(checkpoint)))
    return sanitized


def signal_pipeline_summary(
    revision4_dir: Path, innovation_cache: Path
) -> dict[str, object]:
    formal_methods = load_reservation_aware_sensing_methods(revision4_dir)
    resampling = json.loads(
        (PROJECT_ROOT / "artifacts" / "revision3_eeg_resampling" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    anti_alias_manifest = json.loads(
        (PROJECT_ROOT / "data" / "innovation_cache_antialias" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    cbramod_manifest = sanitize_cbramod_manifest(
        json.loads(
            (PROJECT_ROOT / "data" / "innovation_cache_antialias" / "cbramod" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
    )
    fnirs_pool = np.load(
        innovation_cache / "handcrafted_fnirs_pooled.npy", mmap_mode="r", allow_pickle=False
    )
    if fnirs_pool.shape[1] != 180:
        raise RuntimeError(f"Unexpected reservation-aware fNIRS pool: {fnirs_pool.shape}")
    return {
        "eeg_resampling": {
            **resampling,
            "formal_sensitivity_cache": portable_path(
                PROJECT_ROOT / "data" / "innovation_cache_antialias"
            ),
            "anti_alias_manifest": anti_alias_manifest["eeg"],
        },
        "cbramod": {
            "input": (
                "Centered four-second windows, 64 released array-order channels, "
                "four one-second patches of 200 microvolt samples; boundary windows "
                "use edge repetition. No electrode remapping or channel imputation "
                "was possible because channel semantics were unavailable."
            ),
            "normalization": (
                "Provider baseline mean was subtracted before caching; no additional "
                "per-window z-scoring was applied before the frozen official model."
            ),
            "pooling": (
                "The frozen model returns 200-D channel-by-patch tokens. Mean and "
                "maximum over channel and patch yield 400 dimensions per label second; "
                "trial mean and standard deviation yield the reported 800 dimensions."
            ),
            "manifest": cbramod_manifest,
            "adaptation_label": "array-order adaptation",
        },
        "fnirs": {
            "released_signal_types": [
                "HbO",
                "HbR",
                "HbT",
                "Abs780",
                "Abs805",
                "Abs830",
            ],
            "baseline": "five-second provider array mean subtracted per signal type and channel",
            "resampling": "linear interpolation to four samples per label second",
            "per_channel_features": (
                "6 signal types x 5 statistics (mean, s.d., slope, skewness, "
                "excess kurtosis) x 3-second radius-one context = 90"
            ),
            "reservation_pooling": (
                "masked channel mean and s.d. of 90 features = 180 per label second"
            ),
            "trial_dimension": (
                "trial mean+s.d. of 180 features (360) plus five 180-D lagged "
                "cross-moments at 2/4/6/8/10 s (900) = 1,260"
            ),
            "motion_screen": (
                "within-array first-difference threshold = channel/type median "
                "absolute difference + 6 MAD; extreme-level screen uses |robust z|>8"
            ),
            "reservation_semantics": "source-detector/chromophore semantics unavailable",
            "short_separation_status": "not documented in the release",
            "detrending_or_bandpass": "not applied in the evaluated secondary pipeline",
        },
        "all_evaluated_methods": formal_methods,
        "decision_label": "NO_DEMONSTRATED_INCREMENT_FOR_THE_EVALUATED_PIPELINE",
        "futility_margin": None,
    }


def main() -> None:
    args = parse_args()
    data_root = resolve(args.data_root)
    innovation_cache = resolve(args.innovation_cache)
    revision2_dir = resolve(args.revision2_dir)
    revision3_dir = resolve(args.revision3_dir)
    revision4_dir = resolve(args.revision4_dir)
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
    sampling_rows = read_rows(revision4_dir / "sampling_rate_audit.csv")

    formula_rows, formula_summary = formula_target_audit(
        index, innovation_cache, revision3_dir
    )
    write_rows(output_dir / "formula_target_audit.csv", formula_rows)

    identity_rows, identity_summary = identity_transfer(
        signed_rows,
        categories,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 1000,
    )
    write_rows(output_dir / "identity_transfer.csv", identity_rows)

    practical_rows, practical_summary = calibration_practical_value(
        signed_rows,
        sampling_rows,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 2000,
    )
    write_rows(output_dir / "calibration_practical_value.csv", practical_rows)

    reference_rows, reference_summary = reference_ruler_audit(
        revision2_dir, revision3_dir, revision4_dir
    )
    write_rows(output_dir / "reference_ruler_audit.csv", reference_rows)

    full_rows, full_summary = full_pipeline_bootstrap(
        index,
        categories,
        repeats=args.full_pipeline_repeats,
        workers=args.full_pipeline_workers,
        mc_repeats=args.mc_repeats,
        seed=args.seed + 3000,
    )
    write_rows(output_dir / "full_pipeline_bootstrap.csv", full_rows)

    summary = {
        "formula_target_audit": formula_summary,
        "cross_video_identity_transfer": identity_summary,
        "signed_calibration_practical_value": practical_summary,
        "reference_ruler_audit": reference_summary,
        "full_pipeline_reference_bootstrap": full_summary,
        "generalizability_universe": {
            "finite_library_claim": (
                "The observed profile is descriptive for exactly the 15 released "
                "videos. No unseen-video population generalization is claimed."
            ),
            "random_facet_sensitivity": (
                "G/Phi are reported as method-of-moments ANOVA sensitivities to "
                "repeated participant and finite-library/category-stratified video "
                "resampling, with the five released emotion categories fixed."
            ),
            "finite_population_note": (
                "When all 15 fixed videos are treated as a census, video-sampling "
                "error is not an unseen-item uncertainty. The coefficients therefore "
                "must not be read as evidence for future videos or persistence."
            ),
            "variance_component_estimator": (
                "Balanced random-effects ANOVA method-of-moments. Raw negative "
                "participant/video components are recorded; truncation to zero is "
                "used only when forming reliability coefficients. This is not ML or REML."
            ),
        },
        "signal_pipeline": signal_pipeline_summary(revision4_dir, innovation_cache),
        "claim_boundary": (
            "The added analyses use the same released session. They do not provide "
            "test-retest persistence, unseen-video validation, demographic transport, "
            "external-reference validation, hardware acquisition metadata, or user utility."
        ),
    }
    write_json(output_dir / "summary.json", summary)
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
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
