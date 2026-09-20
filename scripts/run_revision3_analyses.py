#!/usr/bin/env python3
"""Revision-3 analyses for stricter claim, target, and fold-safety auditing.

The script addresses the third manuscript review without changing the released
observations. It verifies the q_pv target definition, rebuilds targets and
context inside every inner participant split, reports modality-specific sensing
ablations, places the G-study on a unified leave-one-participant-out reference,
cross-fits the descriptive geometry adjustment, and propagates calibration-set
as well as participant uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merps.innovation.data import (
    build_context_features,
    build_cross_fitted_context,
    grouped_subject_folds,
    load_innovation_index,
    outer_subject_folds,
)
from merps.innovation.normative_dynamics import (
    WarpConfig,
    decompose_trial,
    normative_trajectory,
)
from merps.innovation.phase_sensing import RIDGE_ALPHAS
from run_revision2_analyses import (
    DualRidgeDesign,
    interval,
    matrix_inference,
    safe_correlation,
    trial_rows,
)

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "innovation_cache"
DEFAULT_TRAIT_DIR = PROJECT_ROOT / "artifacts" / "phase_trait"
DEFAULT_PERSONALIZATION_DIR = PROJECT_ROOT / "artifacts" / "phase_personalization"
DEFAULT_REVISION2_DIR = PROJECT_ROOT / "artifacts" / "revision2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "revision3"

MANAGED_FILES = {
    "calibration_uncertainty.csv",
    "fold_reference_manifest.json",
    "geometry_crossfit.csv",
    "matched_target_predictions.npz",
    "matched_target_results.csv",
    "nested_reference_targets.npz",
    "run_manifest.json",
    "summary.json",
    "target_definition_audit.csv",
    "unified_lopo_targets.csv",
}

MODEL_SPECS: dict[str, tuple[str, ...]] = {
    "context_only": ("context",),
    "cbramod_only": ("cbramod",),
    "handcrafted_eeg_only": ("handcrafted_eeg",),
    "eeg_only": ("cbramod", "handcrafted_eeg"),
    "fnirs_only": ("fnirs_hrf",),
    "eeg_fnirs_single_penalty": (
        "cbramod",
        "handcrafted_eeg",
        "fnirs_hrf",
    ),
    "context_eeg": ("context", "cbramod", "handcrafted_eeg"),
    "context_fnirs": ("context", "fnirs_hrf"),
    "context_eeg_fnirs_single_penalty": (
        "context",
        "cbramod",
        "handcrafted_eeg",
        "fnirs_hrf",
    ),
}
METHODS = (
    "video_mean",
    *MODEL_SPECS.keys(),
    "context_plus_blockwise_residual",
)
RESIDUAL_SPECS: dict[str, tuple[str, ...]] = {
    "cbramod": ("cbramod",),
    "handcrafted_eeg": ("handcrafted_eeg",),
    "eeg": ("cbramod", "handcrafted_eeg"),
    "fnirs_hrf": ("fnirs_hrf",),
    "eeg_fnirs": ("cbramod", "handcrafted_eeg", "fnirs_hrf"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--innovation-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--trait-dir", type=Path, default=DEFAULT_TRAIT_DIR)
    parser.add_argument(
        "--eeg-feature-cache",
        type=Path,
        default=None,
        help="Optional cache supplying only CBraMod and handcrafted EEG blocks.",
    )
    parser.add_argument(
        "--personalization-dir", type=Path, default=DEFAULT_PERSONALIZATION_DIR
    )
    parser.add_argument("--revision2-dir", type=Path, default=DEFAULT_REVISION2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--geometry-bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def manifest_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def build_reference_relative_matrices(
    index,
    source_subjects: Sequence[int],
    target_subjects: Sequence[int],
    *,
    leave_target_out: bool,
    radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct q and b using only the declared reference participants."""

    source_subjects = np.asarray(source_subjects, dtype=np.int16)
    target_subjects = np.asarray(target_subjects, dtype=np.int16)
    config = WarpConfig()
    q = np.empty((len(target_subjects), 15), dtype=np.float64)
    b = np.empty_like(q)
    shared_templates: dict[int, np.ndarray] = {}
    if not leave_target_out:
        shared_templates = {
            video: normative_trajectory(index, source_subjects, video, radius=radius)[0]
            for video in range(1, 16)
        }

    for subject_position, subject in enumerate(target_subjects):
        if leave_target_out:
            reference_subjects = source_subjects[source_subjects != int(subject)]
            if not len(reference_subjects):
                raise ValueError("A leave-one-out reference needs another participant")
            templates = {
                video: normative_trajectory(
                    index, reference_subjects, video, radius=radius
                )[0]
                for video in range(1, 16)
            }
        else:
            templates = shared_templates
        for video in range(1, 16):
            rows = trial_rows(index, int(subject), video)
            result = decompose_trial(
                index.targets[rows].astype(np.float64),
                templates[video],
                config,
            )
            lag = result.consensus_lag.astype(np.float64)
            q[subject_position, video - 1] = float(np.abs(lag).mean())
            b[subject_position, video - 1] = float(lag.mean())
    return q, b


def target_definition_audit(
    index,
    primary_matrix: np.ndarray,
    innovation_cache: Path,
) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray]:
    rows_out: list[dict[str, object]] = []
    validation_b = np.full((24, 15), np.nan, dtype=np.float64)
    validation_q = np.full_like(validation_b, np.nan)
    minimum_margin = np.inf
    violations = 0
    validation_mismatch = 0

    for fold, (training_subjects, validation_subjects) in enumerate(
        outer_subject_folds()
    ):
        cache = np.load(
            innovation_cache / "phase_sensing_targets" / f"fold_{fold}.npz",
            allow_pickle=False,
        )
        lag = cache["lag"].astype(np.float64)
        for role, subjects in (
            ("outer_training_leave_one_out", training_subjects),
            ("outer_validation_training_reference", validation_subjects),
        ):
            for subject in subjects:
                for video in range(1, 16):
                    selected = trial_rows(index, int(subject), video)
                    values = lag[selected]
                    q_pv = float(np.abs(values).mean())
                    b_pv = float(values.mean())
                    margin = q_pv - abs(b_pv)
                    minimum_margin = min(minimum_margin, margin)
                    if margin < -1e-9:
                        violations += 1
                    if role.startswith("outer_validation"):
                        validation_q[int(subject) - 1, video - 1] = q_pv
                        validation_b[int(subject) - 1, video - 1] = b_pv
                        if not np.isclose(
                            q_pv,
                            primary_matrix[int(subject) - 1, video - 1],
                            atol=1e-6,
                        ):
                            validation_mismatch += 1
                    rows_out.append(
                        {
                            "fold": fold,
                            "role": role,
                            "subject": int(subject),
                            "video": video,
                            "q_pv_seconds": q_pv,
                            "b_pv_seconds": b_pv,
                            "q_minus_abs_b_seconds": margin,
                            "invariant_pass": margin >= -1e-9,
                        }
                    )

    if not np.isfinite(validation_q).all() or not np.isfinite(validation_b).all():
        raise RuntimeError("Outer-validation target audit is incomplete")
    if violations or validation_mismatch:
        raise RuntimeError(
            f"Target audit failed: invariant={violations}, mismatch={validation_mismatch}"
        )
    return rows_out, {
        "equation_5": "q_pv = mean_t |delta_pv(t)|",
        "equation_4": "b_pv = mean_t delta_pv(t)",
        "checked_fold_trial_targets": len(rows_out),
        "q_ge_abs_b_violations": violations,
        "minimum_q_minus_abs_b_seconds": float(minimum_margin),
        "outer_validation_q_mismatches": validation_mismatch,
        "status": "pass",
    }, validation_b


def lagged_fnirs_cross_moments(
    fnirs_trial: np.ndarray,
    context_trial: np.ndarray,
    lags: Sequence[int] = (2, 4, 6, 8, 10),
) -> np.ndarray:
    """Fold-safe distributed-lag summaries against reference-gradient energy."""

    values = np.asarray(fnirs_trial, dtype=np.float64)
    normalized = (values - values.mean(axis=0, keepdims=True)) / np.maximum(
        values.std(axis=0, keepdims=True), 1e-6
    )
    gradient = np.linalg.norm(
        np.asarray(context_trial, dtype=np.float64)[:, 4:6], axis=1
    )
    output = []
    for lag in lags:
        if len(values) <= lag + 2:
            output.append(np.zeros(values.shape[1], dtype=np.float64))
            continue
        driver = gradient[:-lag]
        driver = (driver - driver.mean()) / max(float(driver.std()), 1e-6)
        response = normalized[lag:]
        output.append(np.mean(response * driver[:, None], axis=0))
    return np.concatenate(output).astype(np.float32)


def aggregate_trial_blocks(
    index,
    selected_rows: np.ndarray,
    context: np.ndarray,
    cbramod: np.ndarray,
    handcrafted_eeg: np.ndarray,
    fnirs: np.ndarray,
    subjects: Sequence[int],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    local = np.full(len(index.targets), -1, dtype=np.int64)
    local[selected_rows] = np.arange(len(selected_rows))
    blocks: dict[str, list[np.ndarray]] = {
        "context": [],
        "cbramod": [],
        "handcrafted_eeg": [],
        "fnirs_hrf": [],
    }
    subject_rows: list[int] = []
    video_rows: list[int] = []

    for subject in subjects:
        for video in range(1, 16):
            rows = trial_rows(index, int(subject), video)
            positions = local[rows]
            if np.any(positions < 0):
                raise RuntimeError("Trial rows are missing from selected context")
            context_trial = np.asarray(context[positions], dtype=np.float64)
            cbramod_trial = np.asarray(cbramod[rows], dtype=np.float64)
            handcrafted_trial = np.asarray(handcrafted_eeg[rows], dtype=np.float64)
            fnirs_trial = np.asarray(fnirs[rows], dtype=np.float64)
            blocks["context"].append(
                np.concatenate(
                    [context_trial.mean(axis=0), context_trial.std(axis=0)]
                )
            )
            blocks["cbramod"].append(
                np.concatenate(
                    [cbramod_trial.mean(axis=0), cbramod_trial.std(axis=0)]
                )
            )
            blocks["handcrafted_eeg"].append(
                np.concatenate(
                    [handcrafted_trial.mean(axis=0), handcrafted_trial.std(axis=0)]
                )
            )
            fnirs_summary = np.concatenate(
                [fnirs_trial.mean(axis=0), fnirs_trial.std(axis=0)]
            )
            fnirs_lagged = lagged_fnirs_cross_moments(fnirs_trial, context_trial)
            blocks["fnirs_hrf"].append(
                np.concatenate([fnirs_summary, fnirs_lagged])
            )
            subject_rows.append(int(subject))
            video_rows.append(video)

    return (
        {
            key: np.asarray(value, dtype=np.float32)
            for key, value in blocks.items()
        },
        np.asarray(subject_rows, dtype=np.int16),
        np.asarray(video_rows, dtype=np.int16),
    )


def combine_blocks(
    blocks: dict[str, np.ndarray], specification: Sequence[str]
) -> np.ndarray:
    return np.concatenate([blocks[name] for name in specification], axis=1)


def fit_ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    alpha: float,
    *,
    clip_output: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    design = DualRidgeDesign.fit(train_x)
    train_prediction = design.predict(train_x, train_y, alpha)
    validation_prediction = design.predict(validation_x, train_y, alpha)
    if clip_output:
        train_prediction = np.clip(train_prediction, 0.0, 10.0)
        validation_prediction = np.clip(
            validation_prediction, 0.0, 10.0
        )
    return train_prediction, validation_prediction


def select_alpha(
    inner_splits: list[dict[str, object]],
    specification: Sequence[str],
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for alpha in RIDGE_ALPHAS:
        errors = []
        for split in inner_splits:
            train_x = combine_blocks(split["train_blocks"], specification)
            validation_x = combine_blocks(
                split["validation_blocks"], specification
            )
            _, prediction = fit_ridge_predict(
                train_x,
                split["train_y"],
                validation_x,
                float(alpha),
            )
            errors.append(np.abs(prediction - split["validation_y"]))
        scores[str(float(alpha))] = float(np.concatenate(errors).mean())
    best = min(
        (float(value) for value in RIDGE_ALPHAS),
        key=lambda value: (scores[str(value)], value),
    )
    return best, scores


def select_blockwise_residual(
    inner_splits: list[dict[str, object]],
    context_alpha: float,
) -> tuple[str, float | None, dict[str, float]]:
    scores: dict[str, float] = {}
    disabled_errors = []
    for split in inner_splits:
        train_x = combine_blocks(split["train_blocks"], ("context",))
        validation_x = combine_blocks(
            split["validation_blocks"], ("context",)
        )
        _, context_validation = fit_ridge_predict(
            train_x,
            split["train_y"],
            validation_x,
            context_alpha,
        )
        disabled_errors.append(
            np.abs(context_validation - split["validation_y"])
        )
    scores["disabled"] = float(np.concatenate(disabled_errors).mean())

    for name, specification in RESIDUAL_SPECS.items():
        for alpha in RIDGE_ALPHAS:
            errors = []
            for split in inner_splits:
                context_train_x = combine_blocks(
                    split["train_blocks"], ("context",)
                )
                context_validation_x = combine_blocks(
                    split["validation_blocks"], ("context",)
                )
                context_train, context_validation = fit_ridge_predict(
                    context_train_x,
                    split["train_y"],
                    context_validation_x,
                    context_alpha,
                )
                residual_target = split["train_y"] - context_train
                residual_train_x = combine_blocks(
                    split["train_blocks"], specification
                )
                residual_validation_x = combine_blocks(
                    split["validation_blocks"], specification
                )
                _, increment = fit_ridge_predict(
                    residual_train_x,
                    residual_target,
                    residual_validation_x,
                    float(alpha),
                    clip_output=False,
                )
                prediction = np.clip(
                    context_validation + increment,
                    0.0,
                    10.0,
                )
                errors.append(np.abs(prediction - split["validation_y"]))
            scores[f"{name}:{float(alpha)}"] = float(
                np.concatenate(errors).mean()
            )

    best = min(
        scores,
        key=lambda key: (
            scores[key],
            0 if key == "disabled" else 1,
            key,
        ),
    )
    if best == "disabled":
        return "disabled", None, scores
    name, alpha = best.split(":", 1)
    return name, float(alpha), scores


def run_nested_sensing(
    index,
    primary_matrix: np.ndarray,
    innovation_cache: Path,
    *,
    bootstrap_repeats: int,
    eeg_feature_cache: Path | None = None,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    seed: int,
) -> tuple[
    np.ndarray,
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
    dict[str, np.ndarray],
]:
    eeg_feature_cache = innovation_cache if eeg_feature_cache is None else eeg_feature_cache
    cbramod = np.load(
        eeg_feature_cache / "cbramod" / "pooled.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    handcrafted_eeg = np.load(
        eeg_feature_cache / "handcrafted_eeg_pooled.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    fnirs = np.load(
        innovation_cache / "handcrafted_fnirs_pooled.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    predictions = {
        method: np.full((24, 15), np.nan, dtype=np.float64)
        for method in METHODS
    }
    selections: list[dict[str, object]] = []
    target_cache: dict[str, np.ndarray] = {}

    fold_plan = outer_subject_folds() if outer_folds is None else list(outer_folds)
    for outer_fold, (outer_training, outer_validation) in enumerate(fold_plan):
        lag_cache = np.load(
            innovation_cache
            / "phase_sensing_targets"
            / f"fold_{outer_fold}.npz",
            allow_pickle=False,
        )
        lag = lag_cache["lag"].astype(np.float64)
        outer_train_y = np.asarray(
            [
                float(np.abs(lag[trial_rows(index, int(subject), video)]).mean())
                for subject in outer_training
                for video in range(1, 16)
            ],
            dtype=np.float64,
        )
        outer_validation_y = np.asarray(
            [
                primary_matrix[int(subject) - 1, video - 1]
                for subject in outer_validation
                for video in range(1, 16)
            ],
            dtype=np.float64,
        )
        target_cache[f"outer_{outer_fold}_training_q"] = outer_train_y
        target_cache[f"outer_{outer_fold}_validation_q"] = outer_validation_y

        inner_splits: list[dict[str, object]] = []
        inner_manifest = []
        for inner_fold, (inner_training, inner_validation) in enumerate(
            grouped_subject_folds(outer_training, 4)
        ):
            train_q, train_b = build_reference_relative_matrices(
                index,
                inner_training,
                inner_training,
                leave_target_out=True,
                radius=3,
            )
            validation_q, validation_b = build_reference_relative_matrices(
                index,
                inner_training,
                inner_validation,
                leave_target_out=False,
                radius=3,
            )
            train_y = train_q.reshape(-1)
            validation_y = validation_q.reshape(-1)
            target_cache[
                f"outer_{outer_fold}_inner_{inner_fold}_training_q"
            ] = train_y
            target_cache[
                f"outer_{outer_fold}_inner_{inner_fold}_validation_q"
            ] = validation_y
            target_cache[
                f"outer_{outer_fold}_inner_{inner_fold}_training_b"
            ] = train_b.reshape(-1)
            target_cache[
                f"outer_{outer_fold}_inner_{inner_fold}_validation_b"
            ] = validation_b.reshape(-1)

            train_indices = index.indices_for_subjects(inner_training)
            validation_indices = index.indices_for_subjects(inner_validation)
            _, train_context = build_cross_fitted_context(
                index,
                inner_training,
                train_indices,
                radius=3,
            )
            _, validation_context = build_context_features(
                index,
                inner_training,
                validation_indices,
                radius=3,
            )
            train_blocks, train_subject_ids, _ = aggregate_trial_blocks(
                index,
                train_indices,
                train_context,
                cbramod,
                handcrafted_eeg,
                fnirs,
                inner_training,
            )
            validation_blocks, validation_subject_ids, _ = aggregate_trial_blocks(
                index,
                validation_indices,
                validation_context,
                cbramod,
                handcrafted_eeg,
                fnirs,
                inner_validation,
            )
            if not np.array_equal(
                np.unique(train_subject_ids), np.asarray(inner_training)
            ):
                raise RuntimeError("Inner-training trial ordering is invalid")
            if not np.array_equal(
                np.unique(validation_subject_ids),
                np.asarray(inner_validation),
            ):
                raise RuntimeError("Inner-validation trial ordering is invalid")
            if np.intersect1d(inner_training, inner_validation).size:
                raise RuntimeError("Inner subject split is not disjoint")
            inner_splits.append(
                {
                    "train_blocks": train_blocks,
                    "validation_blocks": validation_blocks,
                    "train_y": train_y,
                    "validation_y": validation_y,
                }
            )
            inner_manifest.append(
                {
                    "inner_fold": inner_fold,
                    "training_subjects": inner_training.tolist(),
                    "validation_subjects": inner_validation.tolist(),
                    "training_target_reference": (
                        "inner-training participants excluding the target participant"
                    ),
                    "validation_target_reference": "inner-training participants only",
                    "training_context_reference": (
                        "inner-training participants excluding the target participant"
                    ),
                    "validation_context_reference": "inner-training participants only",
                    "validation_participants_in_training_reference": False,
                }
            )
            print(
                f"outer={outer_fold} inner={inner_fold} nested targets complete",
                flush=True,
            )

        selected_alphas: dict[str, float] = {}
        alpha_scores: dict[str, dict[str, float]] = {}
        for method, specification in MODEL_SPECS.items():
            selected, scores = select_alpha(inner_splits, specification)
            selected_alphas[method] = selected
            alpha_scores[method] = scores
        residual_name, residual_alpha, residual_scores = (
            select_blockwise_residual(
                inner_splits,
                selected_alphas["context_only"],
            )
        )

        outer_training_indices = index.indices_for_subjects(outer_training)
        outer_validation_indices = index.indices_for_subjects(outer_validation)
        _, outer_train_context = build_cross_fitted_context(
            index,
            outer_training,
            outer_training_indices,
            radius=3,
        )
        _, outer_validation_context = build_context_features(
            index,
            outer_training,
            outer_validation_indices,
            radius=3,
        )
        outer_train_blocks, _, _ = aggregate_trial_blocks(
            index,
            outer_training_indices,
            outer_train_context,
            cbramod,
            handcrafted_eeg,
            fnirs,
            outer_training,
        )
        outer_validation_blocks, _, _ = aggregate_trial_blocks(
            index,
            outer_validation_indices,
            outer_validation_context,
            cbramod,
            handcrafted_eeg,
            fnirs,
            outer_validation,
        )

        fold_predictions: dict[str, np.ndarray] = {
            "video_mean": np.tile(
                outer_train_y.reshape(len(outer_training), 15).mean(axis=0),
                len(outer_validation),
            )
        }
        for method, specification in MODEL_SPECS.items():
            train_x = combine_blocks(outer_train_blocks, specification)
            validation_x = combine_blocks(
                outer_validation_blocks, specification
            )
            _, fold_predictions[method] = fit_ridge_predict(
                train_x,
                outer_train_y,
                validation_x,
                selected_alphas[method],
            )

        context_train_x = combine_blocks(
            outer_train_blocks, ("context",)
        )
        context_validation_x = combine_blocks(
            outer_validation_blocks, ("context",)
        )
        context_train, context_validation = fit_ridge_predict(
            context_train_x,
            outer_train_y,
            context_validation_x,
            selected_alphas["context_only"],
        )
        if residual_name == "disabled":
            residual_prediction = context_validation
        else:
            residual_specification = RESIDUAL_SPECS[residual_name]
            residual_train_x = combine_blocks(
                outer_train_blocks, residual_specification
            )
            residual_validation_x = combine_blocks(
                outer_validation_blocks, residual_specification
            )
            _, increment = fit_ridge_predict(
                residual_train_x,
                outer_train_y - context_train,
                residual_validation_x,
                float(residual_alpha),
                clip_output=False,
            )
            residual_prediction = np.clip(
                context_validation + increment, 0.0, 10.0
            )
        fold_predictions[
            "context_plus_blockwise_residual"
        ] = residual_prediction

        for method, values in fold_predictions.items():
            for position, subject in enumerate(outer_validation):
                predictions[method][int(subject) - 1] = values[
                    position * 15 : (position + 1) * 15
                ]

        selections.append(
            {
                "outer_fold": outer_fold,
                "outer_training_subjects": outer_training.tolist(),
                "outer_validation_subjects": outer_validation.tolist(),
                "inner_folds": inner_manifest,
                "selected_alphas": selected_alphas,
                "alpha_scores": alpha_scores,
                "blockwise_residual_selection": {
                    "block": residual_name,
                    "alpha": residual_alpha,
                    "scores": residual_scores,
                    "disabled_candidate_included": True,
                },
                "feature_dimensions": {
                    key: int(value.shape[1])
                    for key, value in outer_train_blocks.items()
                },
            }
        )
        print(
            f"outer={outer_fold} residual={residual_name}:{residual_alpha}",
            flush=True,
        )

    for method, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete revision-3 predictions: {method}")

    context_error = np.abs(predictions["context_only"] - primary_matrix)
    summary_methods: dict[str, object] = {}
    result_rows: list[dict[str, object]] = []
    for position, method in enumerate(METHODS):
        error = np.abs(predictions[method] - primary_matrix)
        gain = context_error - error
        inference = matrix_inference(
            gain,
            repeats=bootstrap_repeats,
            seed=seed + position,
        )
        summary_methods[method] = {
            "participant_macro_mae_seconds": float(error.mean(axis=1).mean()),
            "participant_macro_spearman": float(
                np.mean(
                    [
                        safe_correlation(
                            predictions[method][participant],
                            primary_matrix[participant],
                            spearman=True,
                        )
                        for participant in range(24)
                    ]
                )
            ),
            "gain_over_context_seconds": inference,
        }
        for participant in range(24):
            for video in range(15):
                result_rows.append(
                    {
                        "subject": participant + 1,
                        "video": video + 1,
                        "method": method,
                        "target_q_seconds": float(
                            primary_matrix[participant, video]
                        ),
                        "prediction_q_seconds": float(
                            predictions[method][participant, video]
                        ),
                        "absolute_error_seconds": float(
                            error[participant, video]
                        ),
                        "context_error_minus_method_error_seconds": float(
                            gain[participant, video]
                        ),
                    }
                )

    summary = {
        "target": "trial-level absolute deviation magnitude q_pv",
        "prediction_scope": (
            "participant-held-out, same-trial, offline estimation; not cold-start "
            "onboarding and not real-time inference"
        ),
        "inner_reference_protocol": (
            "for every inner split, inner-training targets use A\\{p}; "
            "inner-validation targets use A; all context, scaling and penalties "
            "are fitted without inner-validation participants"
        ),
        "feature_pooling": {
            "cbramod": "per-trial mean and standard deviation of frozen 400-D embeddings",
            "handcrafted_eeg": (
                "per-trial mean and standard deviation of the committed "
                "handcrafted EEG feature cache"
            ),
            "fnirs": (
                "per-trial mean/std plus 2,4,6,8,10-s distributed-lag "
                "cross-moments against fold-safe reference-gradient energy"
            ),
        },
        "methods": summary_methods,
        "selections": selections,
    }
    return (
        np.stack([predictions[method] for method in METHODS]),
        result_rows,
        summary,
        selections,
        target_cache,
    )


def balanced_variance_components(
    matrix: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    participants, videos = values.shape
    grand = float(values.mean())
    row_mean = values.mean(axis=1)
    column_mean = values.mean(axis=0)
    residual = (
        values
        - row_mean[:, None]
        - column_mean[None, :]
        + grand
    )
    ms_participant = float(
        videos * np.square(row_mean - grand).sum() / (participants - 1)
    )
    ms_video = float(
        participants
        * np.square(column_mean - grand).sum()
        / (videos - 1)
    )
    ms_residual = float(
        np.square(residual).sum()
        / ((participants - 1) * (videos - 1))
    )
    raw_participant = (ms_participant - ms_residual) / videos
    raw_video = (ms_video - ms_residual) / participants
    return {
        "participant": float(max(raw_participant, 0.0)),
        "video": float(max(raw_video, 0.0)),
        "residual": float(max(ms_residual, 0.0)),
        "raw_participant": float(raw_participant),
        "raw_video": float(raw_video),
        "negative_participant_truncated": bool(raw_participant < 0),
        "negative_video_truncated": bool(raw_video < 0),
    }


def reliability_for_k(
    components: dict[str, float], k: int
) -> dict[str, float]:
    participant = components["participant"]
    video = components["video"]
    residual = components["residual"]
    relative_denominator = participant + residual / k
    absolute_denominator = participant + (video + residual) / k
    return {
        "relative_g": float(
            participant / relative_denominator
            if relative_denominator > 0
            else 0.0
        ),
        "absolute_phi": float(
            participant / absolute_denominator
            if absolute_denominator > 0
            else 0.0
        ),
    }


def threshold_crossing(
    components: dict[str, float],
    coefficient: str,
    threshold: float,
    maximum: int = 100,
) -> int | None:
    for k in range(1, maximum + 1):
        if reliability_for_k(components, k)[coefficient] >= threshold:
            return k
    return None


def run_unified_lopo_gstudy(
    index,
    primary_matrix: np.ndarray,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray, np.ndarray]:
    subjects = np.arange(1, 25, dtype=np.int16)
    q_lopo, b_lopo = build_reference_relative_matrices(
        index,
        subjects,
        subjects,
        leave_target_out=True,
        radius=3,
    )
    components = balanced_variance_components(q_lopo)
    reliability_15 = reliability_for_k(components, 15)
    rng = np.random.default_rng(seed)
    g15 = np.empty(bootstrap_repeats, dtype=np.float64)
    phi15 = np.empty_like(g15)
    g70 = np.empty(bootstrap_repeats, dtype=np.float64)
    phi70 = np.empty_like(g70)
    negative_participant = 0
    negative_video = 0
    for repeat in range(bootstrap_repeats):
        participant_draw = rng.integers(0, 24, 24)
        video_draw = rng.integers(0, 15, 15)
        draw = q_lopo[np.ix_(participant_draw, video_draw)]
        draw_components = balanced_variance_components(draw)
        negative_participant += int(
            draw_components["negative_participant_truncated"]
        )
        negative_video += int(draw_components["negative_video_truncated"])
        reliability = reliability_for_k(draw_components, 15)
        g15[repeat] = reliability["relative_g"]
        phi15[repeat] = reliability["absolute_phi"]
        g_cross = threshold_crossing(
            draw_components, "relative_g", 0.70, maximum=100
        )
        phi_cross = threshold_crossing(
            draw_components, "absolute_phi", 0.70, maximum=100
        )
        g70[repeat] = np.nan if g_cross is None else g_cross
        phi70[repeat] = np.nan if phi_cross is None else phi_cross

    rows = [
        {
            "subject": subject + 1,
            "video": video + 1,
            "q_pv_seconds": float(q_lopo[subject, video]),
            "b_pv_seconds": float(b_lopo[subject, video]),
            "reference": "all other 23 participants",
        }
        for subject in range(24)
        for video in range(15)
    ]
    summary = {
        "reference": (
            "one unified leave-one-participant-out reference: every participant "
            "is compared with the other 23 participants"
        ),
        "estimator": (
            "balanced two-facet ANOVA method-of-moments; negative participant "
            "or video variance estimates are recorded and truncated to zero "
            "only when computing reliability"
        ),
        "bootstrap": (
            "participants and videos are independently resampled; duplicate "
            "draws are treated as newly indexed participant/video facets"
        ),
        "variance_components_seconds_squared": components,
        "relative_g_15": reliability_15["relative_g"],
        "relative_g_15_ci95": interval(g15),
        "absolute_phi_15": reliability_15["absolute_phi"],
        "absolute_phi_15_ci95": interval(phi15),
        "g_70_point_crossing_videos": threshold_crossing(
            components, "relative_g", 0.70
        ),
        "phi_70_point_crossing_videos": threshold_crossing(
            components, "absolute_phi", 0.70
        ),
        "g_70_bootstrap_crossing": {
            "median": float(np.nanmedian(g70)),
            "ci95": interval(g70[np.isfinite(g70)]),
            "not_crossed_by_15_fraction": float(
                np.mean(~np.isfinite(g70) | (g70 > 15))
            ),
        },
        "phi_70_bootstrap_crossing": {
            "median": float(np.nanmedian(phi70)),
            "ci95": interval(phi70[np.isfinite(phi70)]),
            "not_crossed_by_15_fraction": float(
                np.mean(~np.isfinite(phi70) | (phi70 > 15))
            ),
        },
        "negative_bootstrap_variance_counts": {
            "participant": negative_participant,
            "video": negative_video,
            "draws": bootstrap_repeats,
        },
        "comparison_with_outer_fold_scale": {
            "trial_mae_seconds": float(np.abs(q_lopo - primary_matrix).mean()),
            "profile_spearman": safe_correlation(
                q_lopo.mean(axis=1),
                primary_matrix.mean(axis=1),
                spearman=True,
            ),
        },
        "observed_video_limit": 15,
        "crossings_above_15_are_extrapolations": True,
    }
    return rows, summary, q_lopo, b_lopo


GEOMETRY_FEATURES = (
    "trajectory_range",
    "trajectory_variance",
    "joystick_path_length_per_second",
    "mean_joystick_speed",
    "axis_correlation",
    "group_reference_gradient",
    "magnitude_residual",
    "distance_from_center",
    "floor_ceiling_occupancy",
)


def crossfit_geometry(
    y: np.ndarray,
    x: np.ndarray,
    included_features: Sequence[int] | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if included_features is None:
        included_features = tuple(range(x.shape[2]))
    residual = np.empty_like(y)
    positions = np.arange(y.shape[0])
    for fold in range(5):
        validation = positions[fold::5]
        training = positions[~np.isin(positions, validation)]
        train_x = x[training][:, :, included_features].reshape(
            -1, len(included_features)
        )
        validation_x = x[validation][:, :, included_features].reshape(
            -1, len(included_features)
        )
        mean = train_x.mean(axis=0, keepdims=True)
        scale = np.maximum(train_x.std(axis=0, keepdims=True), 1e-8)
        train_x = (train_x - mean) / scale
        validation_x = (validation_x - mean) / scale
        train_videos = np.tile(np.arange(15), len(training))
        validation_videos = np.tile(np.arange(15), len(validation))
        train_dummies = np.column_stack(
            [(train_videos == video).astype(float) for video in range(1, 15)]
        )
        validation_dummies = np.column_stack(
            [
                (validation_videos == video).astype(float)
                for video in range(1, 15)
            ]
        )
        train_design = np.column_stack(
            [np.ones(len(train_x)), train_x, train_dummies]
        )
        validation_design = np.column_stack(
            [np.ones(len(validation_x)), validation_x, validation_dummies]
        )
        coefficient = np.linalg.lstsq(
            train_design,
            y[training].reshape(-1),
            rcond=None,
        )[0]
        residual[validation] = (
            y[validation].reshape(-1)
            - validation_design @ coefficient
        ).reshape(len(validation), 15)
    return residual


def variance_inflation_factors(x: np.ndarray) -> dict[str, float]:
    flat = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
    flat = (flat - flat.mean(axis=0)) / np.maximum(flat.std(axis=0), 1e-8)
    videos = np.tile(np.arange(15), x.shape[0])
    dummies = np.column_stack(
        [(videos == video).astype(float) for video in range(1, 15)]
    )
    output: dict[str, float] = {}
    for feature in range(flat.shape[1]):
        target = flat[:, feature]
        others = np.delete(flat, feature, axis=1)
        design = np.column_stack([np.ones(len(flat)), others, dummies])
        prediction = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        denominator = float(np.square(target - target.mean()).sum())
        r_squared = 1.0 - float(np.square(target - prediction).sum()) / max(
            denominator, 1e-12
        )
        output[GEOMETRY_FEATURES[feature]] = float(
            1.0 / max(1.0 - r_squared, 1e-8)
        )
    return output


def run_geometry_crossfit(
    q_lopo: np.ndarray,
    geometry_csv: Path,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = read_rows(geometry_csv)
    source.sort(key=lambda row: (int(row["subject"]), int(row["video"])))
    x = np.asarray(
        [
            [float(row[name]) for name in GEOMETRY_FEATURES]
            for row in source
        ],
        dtype=np.float64,
    ).reshape(24, 15, len(GEOMETRY_FEATURES))
    residual = crossfit_geometry(q_lopo, x)
    raw_components = balanced_variance_components(q_lopo)
    residual_components = balanced_variance_components(residual)
    retained = (
        residual_components["participant"]
        / max(raw_components["participant"], 1e-12)
    )

    rng = np.random.default_rng(seed)
    retained_boot = np.empty(bootstrap_repeats, dtype=np.float64)
    adjusted_g = np.empty_like(retained_boot)
    for repeat in range(bootstrap_repeats):
        participant_draw = rng.integers(0, 24, 24)
        video_draw = rng.integers(0, 15, 15)
        y_draw = q_lopo[np.ix_(participant_draw, video_draw)]
        x_draw = x[participant_draw][:, video_draw]
        residual_draw = crossfit_geometry(y_draw, x_draw)
        raw_draw = balanced_variance_components(y_draw)
        adjusted_draw = balanced_variance_components(residual_draw)
        retained_boot[repeat] = (
            adjusted_draw["participant"]
            / max(raw_draw["participant"], 1e-12)
        )
        adjusted_g[repeat] = reliability_for_k(
            adjusted_draw, 15
        )["relative_g"]

    leave_one_out: dict[str, object] = {}
    for feature, name in enumerate(GEOMETRY_FEATURES):
        included = [value for value in range(len(GEOMETRY_FEATURES)) if value != feature]
        candidate = crossfit_geometry(q_lopo, x, included)
        candidate_components = balanced_variance_components(candidate)
        leave_one_out[name] = {
            "participant_variance": candidate_components["participant"],
            "retained_fraction": (
                candidate_components["participant"]
                / max(raw_components["participant"], 1e-12)
            ),
        }

    rows_out = [
        {
            "subject": subject + 1,
            "video": video + 1,
            "q_lopo_seconds": float(q_lopo[subject, video]),
            "cross_fitted_geometry_residual_seconds": float(
                residual[subject, video]
            ),
        }
        for subject in range(24)
        for video in range(15)
    ]
    summary = {
        "interpretation": (
            "geometry covaried with q_pv; the analysis is descriptive and does "
            "not identify a causal motor mechanism"
        ),
        "protocol": (
            "five participant-held-out cross-fitting folds; standardization and "
            "OLS coefficients are fitted only on training participants"
        ),
        "raw_participant_variance": raw_components["participant"],
        "cross_fitted_adjusted_participant_variance": residual_components[
            "participant"
        ],
        "participant_variance_retained_fraction": float(retained),
        "participant_variance_retained_ci95": interval(retained_boot),
        "cross_fitted_adjusted_relative_g_15": reliability_for_k(
            residual_components, 15
        )["relative_g"],
        "cross_fitted_adjusted_relative_g_15_ci95": interval(adjusted_g),
        "profile_rank_spearman_raw_vs_adjusted": safe_correlation(
            q_lopo.mean(axis=1),
            residual.mean(axis=1),
            spearman=True,
        ),
        "variance_inflation_factors": variance_inflation_factors(x),
        "leave_one_covariate_out": leave_one_out,
        "bootstrap": (
            "the participant-by-video crossed bootstrap redraws both facets and "
            "refits the entire participant-cross-fitted residualization"
        ),
    }
    return rows_out, summary


def joint_participant_subset_bootstrap(
    values: dict[int, np.ndarray],
    *,
    repeats: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(values), dtype=int)
    output = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        output[repeat] = float(
            np.mean(
                [
                    rng.choice(values[int(subject)])
                    for subject in draw
                ]
            )
        )
    return interval(output)


def calibration_uncertainty(
    personalization_dir: Path,
    revision2_dir: Path,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    random_rows = [
        row
        for row in read_rows(personalization_dir / "results.csv")
        if row["method"] == "additive_fewshot"
        and int(row["budget"]) in (1, 2, 4, 8)
    ]
    balanced_rows = read_rows(
        revision2_dir / "category_balanced_calibration.csv"
    )
    revision2_summary = json.loads(
        (revision2_dir / "summary.json").read_text(encoding="utf-8")
    )
    rows_out: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for design, rows in (
        ("random", random_rows),
        ("category_balanced", balanced_rows),
    ):
        summary[design] = {}
        for budget in (1, 2, 4, 8):
            selected = [row for row in rows if int(row["budget"]) == budget]
            by_subject: dict[int, np.ndarray] = {}
            for subject in range(1, 25):
                if design == "random":
                    values = np.asarray(
                        [
                            float(row["delta_vs_video_mean_seconds"])
                            for row in selected
                            if int(row["subject"]) == subject
                        ],
                        dtype=np.float64,
                    )
                else:
                    values = np.asarray(
                        [
                            float(row["gain_vs_video_mean_seconds"])
                            for row in selected
                            if int(row["subject"]) == subject
                        ],
                        dtype=np.float64,
                    )
                by_subject[subject] = values
            participant_means = np.asarray(
                [by_subject[subject].mean() for subject in range(1, 25)]
            )
            if design == "random":
                base = revision2_summary[
                    "matched_noncalibration_video_comparison"
                ][str(budget)]["video_mean"]["participant_macro_mae_seconds"]
            else:
                base = (
                    revision2_summary["category_balanced_calibration"][
                        "aggregate"
                    ][str(budget)]["participant_macro_mae_seconds"]
                    + participant_means.mean()
                )
            combined_ci = joint_participant_subset_bootstrap(
                by_subject,
                repeats=bootstrap_repeats,
                seed=seed + budget + (100 if design == "category_balanced" else 0),
            )
            record = {
                "design": design,
                "budget_videos": budget,
                "mean_gain_seconds_scaled_profile_error": float(
                    participant_means.mean()
                ),
                "relative_gain_percent": float(
                    100.0 * participant_means.mean() / max(base, 1e-12)
                ),
                "participant_and_subset_bootstrap_ci95_low": combined_ci[0],
                "participant_and_subset_bootstrap_ci95_high": combined_ci[1],
                "participants_improved": int(np.sum(participant_means > 0)),
                "participant_gain_median_seconds": float(
                    np.median(participant_means)
                ),
                "worst_participant_predictive_degradation_seconds": float(
                    participant_means.min()
                ),
            }
            rows_out.append(record)
            summary[design][str(budget)] = record
    return rows_out, {
        "metric_boundary": (
            "gains are seconds-scaled aggregate q_pv prediction improvements, "
            "not millisecond local-alignment precision"
        ),
        "uncertainty": (
            "bootstrap resamples participants and, within each sampled "
            "participant, one calibration-set repetition"
        ),
        "budgets": summary,
    }


def build_fold_manifest() -> dict[str, object]:
    folds = []
    for outer_fold, (outer_training, outer_validation) in enumerate(
        outer_subject_folds()
    ):
        inner = []
        for inner_fold, (inner_training, inner_validation) in enumerate(
            grouped_subject_folds(outer_training, 4)
        ):
            inner.append(
                {
                    "inner_fold": inner_fold,
                    "training_subjects": inner_training.tolist(),
                    "validation_subjects": inner_validation.tolist(),
                    "training_target_reference_rule": "A minus target participant",
                    "validation_target_reference_rule": "A only",
                    "training_context_reference_rule": "A minus target participant",
                    "validation_context_reference_rule": "A only",
                    "validation_subjects_used_by_training_reference": False,
                }
            )
        folds.append(
            {
                "outer_fold": outer_fold,
                "training_subjects": outer_training.tolist(),
                "validation_subjects": outer_validation.tolist(),
                "outer_training_target_reference_rule": (
                    "outer training minus target participant"
                ),
                "outer_validation_target_reference_rule": "outer training only",
                "inner_folds": inner,
            }
        )
    return {
        "target": "q_pv = mean_t |delta_pv(t)|",
        "context_features": (
            "reference valence, reference arousal, robust uncertainty for both "
            "axes, reference gradients for both axes, and normalized video progress"
        ),
        "reference_smoothing_radius_seconds": 3,
        "participant_trajectory_smoothing_radius_seconds": 2,
        "folds": folds,
    }


def main() -> None:
    args = parse_args()
    data_root = resolve(args.data_root)
    innovation_cache = resolve(args.innovation_cache)
    trait_dir = resolve(args.trait_dir)
    eeg_feature_cache = (
        innovation_cache
        if args.eeg_feature_cache is None
        else resolve(args.eeg_feature_cache)
    )
    personalization_dir = resolve(args.personalization_dir)
    revision2_dir = resolve(args.revision2_dir)
    output_dir = resolve(args.output_dir)
    prepare_output(output_dir, args.overwrite)
    started = time.perf_counter()

    index = load_innovation_index(data_root)
    primary_matrix = np.load(
        trait_dir / "phase_trait_matrix.npz", allow_pickle=False
    )["mean_absolute_lag"].astype(np.float64)

    target_rows, target_summary, signed_primary = target_definition_audit(
        index,
        primary_matrix,
        innovation_cache,
    )
    write_rows(output_dir / "target_definition_audit.csv", target_rows)
    write_json(output_dir / "fold_reference_manifest.json", build_fold_manifest())

    (
        prediction_stack,
        prediction_rows,
        sensing_summary,
        selections,
        target_cache,
    ) = run_nested_sensing(
        index,
        primary_matrix,
        innovation_cache,
        bootstrap_repeats=args.bootstrap_repeats,
        eeg_feature_cache=eeg_feature_cache,
        seed=args.seed + 100,
    )
    np.savez_compressed(
        output_dir / "matched_target_predictions.npz",
        methods=np.asarray(METHODS),
        predictions=prediction_stack.astype(np.float32),
        target=primary_matrix.astype(np.float32),
    )
    np.savez_compressed(
        output_dir / "nested_reference_targets.npz",
        **{key: value.astype(np.float32) for key, value in target_cache.items()},
    )
    write_rows(output_dir / "matched_target_results.csv", prediction_rows)

    lopo_rows, lopo_summary, q_lopo, b_lopo = run_unified_lopo_gstudy(
        index,
        primary_matrix,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 200,
    )
    write_rows(output_dir / "unified_lopo_targets.csv", lopo_rows)

    geometry_rows, geometry_summary = run_geometry_crossfit(
        q_lopo,
        revision2_dir / "geometry_covariates.csv",
        bootstrap_repeats=args.geometry_bootstrap_repeats,
        seed=args.seed + 300,
    )
    write_rows(output_dir / "geometry_crossfit.csv", geometry_rows)

    calibration_rows, calibration_summary = calibration_uncertainty(
        personalization_dir,
        revision2_dir,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 400,
    )
    write_rows(output_dir / "calibration_uncertainty.csv", calibration_rows)

    summary = {
        "target_definition_audit": target_summary,
        "nested_matched_target_sensing": sensing_summary,
        "unified_lopo_gstudy": lopo_summary,
        "geometry_crossfit": geometry_summary,
        "calibration_uncertainty": calibration_summary,
        "signed_bias_scope": {
            "b_pv_matrix_mean_seconds": float(signed_primary.mean()),
            "b_pv_matrix_mean_absolute_seconds": float(
                np.abs(signed_primary).mean()
            ),
            "unified_lopo_b_profile_spearman_vs_outer_fold": safe_correlation(
                b_lopo.mean(axis=1),
                signed_primary.mean(axis=1),
                spearman=True,
            ),
            "utility_boundary": (
                "b_pv preserves direction, but neither scalar b_pv nor q_pv "
                "recovers the local warp; downstream trace correction remains "
                "a separate consequential-utility test"
            ),
        },
        "claim_boundary": (
            "The available data support a candidate within-session, "
            "reference-relative profile. They do not establish test-retest "
            "persistence, cross-interface portability, or consequential utility."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "configuration": {
                "data_root": manifest_path(data_root),
                "innovation_cache": manifest_path(innovation_cache),
                "trait_dir": manifest_path(trait_dir),
                "eeg_feature_cache": manifest_path(eeg_feature_cache),
                "personalization_dir": manifest_path(personalization_dir),
                "revision2_dir": manifest_path(revision2_dir),
                "output_dir": manifest_path(output_dir),
                "bootstrap_repeats": args.bootstrap_repeats,
                "geometry_bootstrap_repeats": args.geometry_bootstrap_repeats,
                "seed": args.seed,
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "generated_files": sorted(MANAGED_FILES),
            "path_policy": "project-relative paths only",
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
