#!/usr/bin/env python3
"""Supplementary analyses requested by the second manuscript review.

This script keeps the established five-fold participant-held-out protocol and
uses the same outer-training-only group reference as the primary analyses. It
adds participant-specific warp controls, matched-target trial-level sensing,
reference-composition sensitivity, trajectory-geometry adjustment, and
category-balanced behavioral calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.innovation.data import (
    build_context_features,
    build_cross_fitted_context,
    grouped_subject_folds,
    load_innovation_index,
    outer_subject_folds,
)
from merps.innovation.normative_dynamics import (
    WarpConfig,
    constrained_dtw_mapping,
    normative_trajectory,
)
from merps.innovation.personalization import additive_personalization
from merps.innovation.phase_sensing import RIDGE_ALPHAS
from merps.innovation.phase_trait import (
    estimate_variance_components,
    reliability_for_videos,
)

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_INNOVATION_CACHE = PROJECT_ROOT / "data" / "innovation_cache"
DEFAULT_TRAIT_DIR = PROJECT_ROOT / "artifacts" / "phase_trait"
DEFAULT_PERSONALIZATION_DIR = PROJECT_ROOT / "artifacts" / "phase_personalization"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "revision2"
MANAGED_FILES = {
    "category_balanced_calibration.csv",
    "geometry_covariates.csv",
    "matched_target_predictions.npz",
    "matched_target_results.csv",
    "own_other_warp.csv",
    "own_other_warp_null.npz",
    "reference_sensitivity.csv",
    "run_manifest.json",
    "summary.json",
}
SENSING_METHODS = (
    "zero_lag",
    "video_mean",
    "context_only",
    "physiology_only",
    "context_physiology_single_penalty",
    "context_plus_residual_physiology",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--innovation-cache", type=Path, default=DEFAULT_INNOVATION_CACHE
    )
    parser.add_argument("--trait-dir", type=Path, default=DEFAULT_TRAIT_DIR)
    parser.add_argument(
        "--personalization-dir", type=Path, default=DEFAULT_PERSONALIZATION_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--permutation-repeats", type=int, default=10000)
    parser.add_argument("--reference-repeats", type=int, default=20)
    parser.add_argument("--calibration-repeats", type=int, default=100)
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
    fieldnames = sorted({key for row in rows for key in row})
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


def interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def safe_correlation(
    first: np.ndarray, second: np.ndarray, *, spearman: bool = False
) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if len(first) < 3 or first.std() < 1e-12 or second.std() < 1e-12:
        return 0.0
    if spearman:
        first = np.argsort(np.argsort(first, kind="mergesort"), kind="mergesort")
        second = np.argsort(np.argsort(second, kind="mergesort"), kind="mergesort")
    return float(np.corrcoef(first, second)[0, 1])


def trial_rows(index, subject: int, video: int) -> np.ndarray:
    rows = np.flatnonzero(
        (index.subject_numbers == int(subject)) & (index.videos == int(video))
    )
    return rows[np.argsort(index.timestamps[rows])]


def matrix_inference(
    matrix: np.ndarray, *, repeats: int, seed: int
) -> dict[str, object]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (24, 15):
        raise ValueError("Expected a complete 24 x 15 matrix")
    rng = np.random.default_rng(seed)
    crossed = np.empty(repeats, dtype=np.float64)
    participant_values = values.mean(axis=1)
    signflip = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        participant_draw = rng.integers(0, 24, 24)
        video_draw = rng.integers(0, 15, 15)
        crossed[repeat] = values[np.ix_(participant_draw, video_draw)].mean()
        signs = rng.choice((-1.0, 1.0), size=24)
        signflip[repeat] = float(np.mean(signs * participant_values))
    observed = float(values.mean())
    return {
        "mean": observed,
        "participant_video_crossed_ci95": interval(crossed),
        "participant_signflip_p_two_sided": float(
            (1 + np.sum(np.abs(signflip) >= abs(observed))) / (repeats + 1)
        ),
        "participant_mean_median": float(np.median(participant_values)),
        "participants_positive": int(np.sum(participant_values > 0)),
    }


def icc_consistency(first: np.ndarray, second: np.ndarray) -> float:
    values = np.column_stack(
        [
            np.asarray(first, dtype=np.float64).reshape(-1),
            np.asarray(second, dtype=np.float64).reshape(-1),
        ]
    )
    n, k = values.shape
    row_mean = values.mean(axis=1)
    column_mean = values.mean(axis=0)
    grand = float(values.mean())
    ms_rows = float(k * np.square(row_mean - grand).sum() / (n - 1))
    residual = values - row_mean[:, None] - column_mean[None, :] + grand
    ms_error = float(np.square(residual).sum() / ((n - 1) * (k - 1)))
    denominator = ms_rows + (k - 1) * ms_error
    return float((ms_rows - ms_error) / denominator) if denominator > 0 else 0.0


def run_own_other_warp(
    index,
    *,
    bootstrap_repeats: int,
    permutation_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray]:
    config = WarpConfig()
    rows_out: list[dict[str, object]] = []
    combined = np.full((24, 15), np.nan, dtype=np.float64)
    valence = np.full_like(combined, np.nan)
    arousal = np.full_like(combined, np.nan)
    donor_error_blocks: list[tuple[np.ndarray, np.ndarray]] = []

    for fold, (training_subjects, validation_subjects) in enumerate(
        outer_subject_folds()
    ):
        for video in range(1, 16):
            template = normative_trajectory(
                index, training_subjects, video, radius=3
            )[0]
            targets = []
            mappings_v = []
            mappings_a = []
            for subject in validation_subjects:
                target = index.targets[trial_rows(index, int(subject), video)].astype(
                    np.float64
                )
                targets.append(target)
                mappings_v.append(
                    constrained_dtw_mapping(target[:, 0], template[:, 0], config)
                )
                mappings_a.append(
                    constrained_dtw_mapping(target[:, 1], template[:, 1], config)
                )
            n_validation = len(validation_subjects)
            donor_errors = np.empty((n_validation, n_validation, 2), dtype=np.float64)
            baselines = np.empty((n_validation, 2), dtype=np.float64)
            for receiver in range(n_validation):
                target = targets[receiver]
                baselines[receiver, 0] = np.abs(target[:, 0] - template[:, 0]).mean()
                baselines[receiver, 1] = np.abs(target[:, 1] - template[:, 1]).mean()
                for donor in range(n_validation):
                    donor_errors[receiver, donor, 0] = np.abs(
                        target[:, 0] - template[mappings_a[donor], 0]
                    ).mean()
                    donor_errors[receiver, donor, 1] = np.abs(
                        target[:, 1] - template[mappings_v[donor], 1]
                    ).mean()
            donor_error_blocks.append((validation_subjects.copy(), donor_errors))
            for receiver, subject in enumerate(validation_subjects):
                donors = np.arange(n_validation) != receiver
                own_gain_v = baselines[receiver, 0] - donor_errors[receiver, receiver, 0]
                own_gain_a = baselines[receiver, 1] - donor_errors[receiver, receiver, 1]
                other_gain_v = float(
                    baselines[receiver, 0] - donor_errors[receiver, donors, 0].mean()
                )
                other_gain_a = float(
                    baselines[receiver, 1] - donor_errors[receiver, donors, 1].mean()
                )
                delta_v = own_gain_v - other_gain_v
                delta_a = own_gain_a - other_gain_a
                delta = float((delta_v + delta_a) / 2.0)
                combined[int(subject) - 1, video - 1] = delta
                valence[int(subject) - 1, video - 1] = delta_v
                arousal[int(subject) - 1, video - 1] = delta_a
                rows_out.append(
                    {
                        "fold": fold,
                        "subject": int(subject),
                        "video": video,
                        "own_gain_valence_from_arousal_label_units": float(own_gain_v),
                        "other_gain_valence_from_arousal_label_units": float(other_gain_v),
                        "own_minus_other_valence_label_units": float(delta_v),
                        "own_gain_arousal_from_valence_label_units": float(own_gain_a),
                        "other_gain_arousal_from_valence_label_units": float(other_gain_a),
                        "own_minus_other_arousal_label_units": float(delta_a),
                        "own_minus_other_mean_label_units": delta,
                        "other_donors": int(n_validation - 1),
                    }
                )

    if not np.isfinite(combined).all():
        raise RuntimeError("Own-versus-other matrix is incomplete")

    rng = np.random.default_rng(seed)
    null = np.empty(permutation_repeats, dtype=np.float64)
    for repeat in range(permutation_repeats):
        differences = []
        for _, donor_errors in donor_error_blocks:
            n_validation = donor_errors.shape[0]
            order = rng.permutation(n_validation)
            shift = int(rng.integers(1, n_validation))
            donors = np.empty(n_validation, dtype=np.int16)
            donors[order] = np.roll(order, shift)
            receivers = np.arange(n_validation)
            own_error = donor_errors[receivers, receivers].mean(axis=1)
            other_error = donor_errors[receivers, donors].mean(axis=1)
            differences.extend((other_error - own_error).tolist())
        null[repeat] = float(np.mean(differences))

    observed = float(combined.mean())
    summary = {
        "estimand": (
            "cross-dimensional alignment gain from a receiver's own source-axis "
            "warp minus the mean gain from other held-out participants' warps in "
            "the same video and outer fold"
        ),
        "unit": "released joystick label units",
        "combined": matrix_inference(
            combined, repeats=bootstrap_repeats, seed=seed + 1
        ),
        "valence_from_arousal": matrix_inference(
            valence, repeats=bootstrap_repeats, seed=seed + 2
        ),
        "arousal_from_valence": matrix_inference(
            arousal, repeats=bootstrap_repeats, seed=seed + 3
        ),
        "within_video_donor_reassignment": {
            "draws": permutation_repeats,
            "mean_advantage_label_units": float(null.mean()),
            "ci95_label_units": interval(null),
            "p_random_other_as_good_or_better_than_own": float(
                (1 + np.sum(null <= 0)) / (permutation_repeats + 1)
            ),
            "scheme": (
                "within each outer-fold/video block, a random one-to-one "
                "derangement assigns every receiver another held-out donor"
            ),
        },
    }
    return rows_out, summary, null


@dataclass
class DualRidgeDesign:
    x_mean: np.ndarray
    x_scale: np.ndarray
    normalized_training: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "DualRidgeDesign":
        x = np.asarray(x, dtype=np.float64)
        x_mean = x.mean(axis=0, keepdims=True)
        x_scale = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
        normalized = (x - x_mean) / x_scale
        gram = normalized @ normalized.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        return cls(
            x_mean=x_mean,
            x_scale=x_scale,
            normalized_training=normalized,
            eigenvalues=np.maximum(eigenvalues, 0.0),
            eigenvectors=eigenvectors,
        )

    def predict(
        self, x: np.ndarray, target: np.ndarray, alpha: float
    ) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        target_mean = float(target.mean())
        projected = self.eigenvectors.T @ (target - target_mean)
        dual = self.eigenvectors @ (
            projected / (self.eigenvalues + float(alpha))
        )
        normalized_test = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_scale
        kernel = normalized_test @ self.normalized_training.T
        return np.asarray(kernel @ dual + target_mean, dtype=np.float64)


def aggregate_trial_features(
    index,
    selected_rows: np.ndarray,
    context: np.ndarray,
    eeg: np.ndarray,
    fnirs: np.ndarray,
    subjects: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    local = np.full(len(index.targets), -1, dtype=np.int64)
    local[selected_rows] = np.arange(len(selected_rows))
    context_rows = []
    physiology_rows = []
    subject_rows = []
    video_rows = []
    for subject in subjects:
        for video in range(1, 16):
            rows = trial_rows(index, int(subject), video)
            positions = local[rows]
            if np.any(positions < 0):
                raise RuntimeError("Trial rows are missing from selected feature rows")
            context_trial = np.asarray(context[positions], dtype=np.float64)
            eeg_trial = np.asarray(eeg[rows], dtype=np.float64)
            fnirs_trial = np.asarray(fnirs[rows], dtype=np.float64)
            context_rows.append(
                np.concatenate([context_trial.mean(axis=0), context_trial.std(axis=0)])
            )
            physiology_rows.append(
                np.concatenate(
                    [
                        eeg_trial.mean(axis=0),
                        eeg_trial.std(axis=0),
                        fnirs_trial.mean(axis=0),
                        fnirs_trial.std(axis=0),
                    ]
                )
            )
            subject_rows.append(int(subject))
            video_rows.append(video)
    return (
        np.asarray(context_rows, dtype=np.float32),
        np.asarray(physiology_rows, dtype=np.float32),
        np.asarray(subject_rows, dtype=np.int16),
        np.asarray(video_rows, dtype=np.int16),
    )


def trial_targets(matrix: np.ndarray, subjects: Sequence[int]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(matrix[int(subject) - 1], dtype=np.float64) for subject in subjects]
    )


def lag_trial_matrix(index, lag: np.ndarray, subjects: Sequence[int]) -> np.ndarray:
    matrix = np.empty((len(subjects), 15), dtype=np.float64)
    for subject_position, subject in enumerate(subjects):
        for video in range(1, 16):
            rows = trial_rows(index, int(subject), video)
            matrix[subject_position, video - 1] = float(np.abs(lag[rows]).mean())
    return matrix


def select_ridge_predictions(
    designs: list[tuple[np.ndarray, np.ndarray, DualRidgeDesign]],
    target: np.ndarray,
    validation_positions: list[np.ndarray],
    feature_validation: list[np.ndarray],
) -> tuple[float, dict[str, float], np.ndarray]:
    predictions = {
        float(alpha): np.full(len(target), np.nan, dtype=np.float64)
        for alpha in RIDGE_ALPHAS
    }
    for (training, _, design), validation, x_validation in zip(
        designs, validation_positions, feature_validation
    ):
        for alpha in RIDGE_ALPHAS:
            predictions[float(alpha)][validation] = np.clip(
                design.predict(x_validation, target[training], float(alpha)),
                0.0,
                10.0,
            )
    scores = {
        str(alpha): float(np.mean(np.abs(predictions[float(alpha)] - target)))
        for alpha in RIDGE_ALPHAS
    }
    best = min(RIDGE_ALPHAS, key=lambda alpha: (scores[str(alpha)], alpha))
    return float(best), scores, predictions[float(best)]


def run_matched_target_sensing(
    index,
    primary_matrix: np.ndarray,
    innovation_cache: Path,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    eeg = np.load(
        innovation_cache / "cbramod" / "pooled.npy",
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
        for method in SENSING_METHODS
    }
    selections: list[dict[str, object]] = []

    for fold, (training_subjects, validation_subjects) in enumerate(
        outer_subject_folds()
    ):
        lag_cache = np.load(
            innovation_cache / "phase_sensing_targets" / f"fold_{fold}.npz",
            allow_pickle=False,
        )
        lag = lag_cache["lag"]
        train_idx = index.indices_for_subjects(training_subjects)
        validation_idx = index.indices_for_subjects(validation_subjects)
        _, train_context = build_cross_fitted_context(
            index, training_subjects, train_idx, radius=3
        )
        _, validation_context = build_context_features(
            index, training_subjects, validation_idx, radius=3
        )
        train_c, train_p, train_subject_ids, _ = aggregate_trial_features(
            index, train_idx, train_context, eeg, fnirs, training_subjects
        )
        validation_c, validation_p, _, _ = aggregate_trial_features(
            index,
            validation_idx,
            validation_context,
            eeg,
            fnirs,
            validation_subjects,
        )
        training_matrix = lag_trial_matrix(index, lag, training_subjects)
        validation_matrix = lag_trial_matrix(index, lag, validation_subjects)
        train_y = training_matrix.reshape(-1)
        validation_y = validation_matrix.reshape(-1)
        expected_validation = trial_targets(primary_matrix, validation_subjects)
        if not np.allclose(validation_y, expected_validation, atol=1e-6):
            raise RuntimeError("Fold validation q does not match the primary held-out matrix")
        inner_records = []
        c_designs = []
        p_designs = []
        j_designs = []
        validation_positions = []
        c_validation = []
        p_validation = []
        j_validation = []
        for inner_fold, (inner_training_subjects, inner_validation_subjects) in enumerate(
            grouped_subject_folds(training_subjects, 4)
        ):
            inner_training = np.flatnonzero(
                np.isin(train_subject_ids, inner_training_subjects)
            )
            inner_validation = np.flatnonzero(
                np.isin(train_subject_ids, inner_validation_subjects)
            )
            c_design = DualRidgeDesign.fit(train_c[inner_training])
            p_design = DualRidgeDesign.fit(train_p[inner_training])
            joint_training = np.concatenate(
                [train_c[inner_training], train_p[inner_training]], axis=1
            )
            joint_validation = np.concatenate(
                [train_c[inner_validation], train_p[inner_validation]], axis=1
            )
            j_design = DualRidgeDesign.fit(joint_training)
            c_designs.append((inner_training, inner_validation, c_design))
            p_designs.append((inner_training, inner_validation, p_design))
            j_designs.append((inner_training, inner_validation, j_design))
            validation_positions.append(inner_validation)
            c_validation.append(train_c[inner_validation])
            p_validation.append(train_p[inner_validation])
            j_validation.append(joint_validation)
            inner_records.append(
                {
                    "fold": inner_fold,
                    "training_subjects": inner_training_subjects.tolist(),
                    "validation_subjects": inner_validation_subjects.tolist(),
                }
            )

        alpha_c, scores_c, oof_context = select_ridge_predictions(
            c_designs, train_y, validation_positions, c_validation
        )
        alpha_p, scores_p, _ = select_ridge_predictions(
            p_designs, train_y, validation_positions, p_validation
        )
        alpha_j, scores_j, _ = select_ridge_predictions(
            j_designs, train_y, validation_positions, j_validation
        )

        residual_predictions = {
            "disabled": oof_context.copy(),
            **{
                str(alpha): np.full(len(train_y), np.nan, dtype=np.float64)
                for alpha in RIDGE_ALPHAS
            },
        }
        for (
            (inner_training, _, c_design),
            (_, _, p_design),
            inner_validation,
            x_c_validation,
            x_p_validation,
        ) in zip(
            c_designs,
            p_designs,
            validation_positions,
            c_validation,
            p_validation,
        ):
            context_train = c_design.predict(
                train_c[inner_training], train_y[inner_training], alpha_c
            )
            context_validation_prediction = c_design.predict(
                x_c_validation, train_y[inner_training], alpha_c
            )
            residual_target = train_y[inner_training] - context_train
            for alpha in RIDGE_ALPHAS:
                residual_increment = p_design.predict(
                    x_p_validation, residual_target, float(alpha)
                )
                residual_predictions[str(alpha)][inner_validation] = np.clip(
                    context_validation_prediction + residual_increment,
                    0.0,
                    10.0,
                )
        residual_scores = {
            key: float(np.mean(np.abs(value - train_y)))
            for key, value in residual_predictions.items()
        }
        best_residual = min(
            residual_scores,
            key=lambda key: (
                residual_scores[key],
                0.0 if key == "disabled" else float(key),
            ),
        )

        c_outer = DualRidgeDesign.fit(train_c)
        p_outer = DualRidgeDesign.fit(train_p)
        joint_train = np.concatenate([train_c, train_p], axis=1)
        joint_validation = np.concatenate([validation_c, validation_p], axis=1)
        j_outer = DualRidgeDesign.fit(joint_train)
        context_prediction = np.clip(
            c_outer.predict(validation_c, train_y, alpha_c), 0.0, 10.0
        )
        physiology_prediction = np.clip(
            p_outer.predict(validation_p, train_y, alpha_p), 0.0, 10.0
        )
        joint_prediction = np.clip(
            j_outer.predict(joint_validation, train_y, alpha_j), 0.0, 10.0
        )
        if best_residual == "disabled":
            residual_prediction = context_prediction.copy()
        else:
            context_train = c_outer.predict(train_c, train_y, alpha_c)
            residual_target = train_y - context_train
            residual_increment = p_outer.predict(
                validation_p, residual_target, float(best_residual)
            )
            residual_prediction = np.clip(
                context_prediction + residual_increment, 0.0, 10.0
            )
        video_mean = training_matrix.mean(axis=0)
        fold_predictions = {
            "zero_lag": np.zeros_like(validation_y),
            "video_mean": np.tile(video_mean, len(validation_subjects)),
            "context_only": context_prediction,
            "physiology_only": physiology_prediction,
            "context_physiology_single_penalty": joint_prediction,
            "context_plus_residual_physiology": residual_prediction,
        }
        for method, values in fold_predictions.items():
            for position, subject in enumerate(validation_subjects):
                predictions[method][int(subject) - 1] = values[
                    position * 15 : (position + 1) * 15
                ]
        selections.append(
            {
                "fold": fold,
                "training_subjects": training_subjects.tolist(),
                "validation_subjects": validation_subjects.tolist(),
                "context_alpha": alpha_c,
                "physiology_alpha": alpha_p,
                "single_penalty_joint_alpha": alpha_j,
                "residual_physiology_alpha": best_residual,
                "context_scores": scores_c,
                "physiology_scores": scores_p,
                "single_penalty_joint_scores": scores_j,
                "residual_scores": residual_scores,
                "inner_folds": inner_records,
                "feature_dimensions": {
                    "context": int(train_c.shape[1]),
                    "physiology": int(train_p.shape[1]),
                },
            }
        )
        print(
            f"matched-target fold={fold} context={alpha_c:g} phys={alpha_p:g} "
            f"joint={alpha_j:g} residual={best_residual}",
            flush=True,
        )

    for method, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete matched-target predictions: {method}")

    summary: dict[str, object] = {
        "target": "trial-level absolute deviation magnitude q_pv",
        "outer_protocol": (
            "five participant-held-out folds; validation q and context use only "
            "outer-training participants"
        ),
        "feature_pooling": (
            "per-trial mean and standard deviation of frozen 400-D CBraMod "
            "embeddings and 180-D released-array fNIRS summaries"
        ),
        "methods": {},
        "residual_increment_selection": {
            "fold_selected_sensor_penalty": [
                record["residual_physiology_alpha"] for record in selections
            ],
            "disabled_candidate_included": True,
        },
    }
    result_rows: list[dict[str, object]] = []
    context_error = np.abs(predictions["context_only"] - primary_matrix)
    for method in SENSING_METHODS:
        error = np.abs(predictions[method] - primary_matrix)
        participant_mae = error.mean(axis=1)
        participant_spearman = np.asarray(
            [
                safe_correlation(
                    predictions[method][participant],
                    primary_matrix[participant],
                    spearman=True,
                )
                for participant in range(24)
            ]
        )
        gain_matrix = context_error - error
        gain_inference = matrix_inference(
            gain_matrix, repeats=bootstrap_repeats, seed=seed + len(result_rows) + 10
        )
        summary["methods"][method] = {
            "participant_macro_mae_seconds": float(participant_mae.mean()),
            "participant_macro_spearman": float(participant_spearman.mean()),
            "gain_over_context_seconds": gain_inference,
        }
        for participant in range(24):
            for video in range(15):
                result_rows.append(
                    {
                        "subject": participant + 1,
                        "video": video + 1,
                        "method": method,
                        "target_q_seconds": float(primary_matrix[participant, video]),
                        "prediction_q_seconds": float(
                            predictions[method][participant, video]
                        ),
                        "absolute_error_seconds": float(error[participant, video]),
                        "context_error_minus_method_error_seconds": float(
                            gain_matrix[participant, video]
                        ),
                    }
                )
    return (
        np.stack([predictions[method] for method in SENSING_METHODS]),
        result_rows,
        summary,
        selections,
    )


def calibration_split_comparison(
    prediction_stack: np.ndarray,
    primary_matrix: np.ndarray,
    personalization_dir: Path,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    prediction_lookup = {
        method: prediction_stack[position]
        for position, method in enumerate(SENSING_METHODS)
    }
    with (personalization_dir / "results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    additive = {
        (int(row["subject"]), int(row["budget"]), int(row["repeat"])): row
        for row in source_rows
        if row["method"] == "additive_fewshot" and int(row["budget"]) > 0
    }
    rows: list[dict[str, object]] = []
    for (subject, budget, repeat), source in additive.items():
        calibration = {
            int(value) - 1
            for value in source["calibration_videos"].split()
            if value
        }
        evaluation = np.asarray(
            [video for video in range(15) if video not in calibration], dtype=int
        )
        target = primary_matrix[subject - 1, evaluation]
        for method in SENSING_METHODS:
            prediction = prediction_lookup[method][subject - 1, evaluation]
            rows.append(
                {
                    "subject": subject,
                    "budget": budget,
                    "repeat": repeat,
                    "method": method,
                    "mae_seconds": float(np.abs(prediction - target).mean()),
                }
            )
        rows.append(
            {
                "subject": subject,
                "budget": budget,
                "repeat": repeat,
                "method": "additive_behavioral_calibration",
                "mae_seconds": float(source["mae_seconds"]),
            }
        )

    summary: dict[str, object] = {}
    rng = np.random.default_rng(seed)
    methods = list(SENSING_METHODS) + ["additive_behavioral_calibration"]
    for budget in (1, 2, 4, 8):
        summary[str(budget)] = {}
        selected_budget = [row for row in rows if int(row["budget"]) == budget]
        for method in methods:
            selected = [row for row in selected_budget if row["method"] == method]
            participant_mae = np.asarray(
                [
                    np.mean(
                        [
                            float(row["mae_seconds"])
                            for row in selected
                            if int(row["subject"]) == subject
                        ]
                    )
                    for subject in range(1, 25)
                ]
            )
            baseline_rows = [
                row for row in selected_budget if row["method"] == "video_mean"
            ]
            baseline = np.asarray(
                [
                    np.mean(
                        [
                            float(row["mae_seconds"])
                            for row in baseline_rows
                            if int(row["subject"]) == subject
                        ]
                    )
                    for subject in range(1, 25)
                ]
            )
            gain = baseline - participant_mae
            boot = np.empty(bootstrap_repeats, dtype=np.float64)
            for repeat in range(bootstrap_repeats):
                draw = rng.integers(0, 24, 24)
                boot[repeat] = gain[draw].mean()
            summary[str(budget)][method] = {
                "participant_macro_mae_seconds": float(participant_mae.mean()),
                "gain_vs_video_mean_seconds": float(gain.mean()),
                "participant_bootstrap_gain_ci95": interval(boot),
                "participants_improved": int(np.sum(gain > 0)),
                "participant_gain_median_seconds": float(np.median(gain)),
                "participant_gain_iqr_seconds": [
                    float(np.quantile(gain, 0.25)),
                    float(np.quantile(gain, 0.75)),
                ],
                "worst_participant_gain_seconds": float(gain.min()),
            }
    return rows, summary


def run_reference_sensitivity(
    index,
    primary_matrix: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(seed)
    config = WarpConfig()
    rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    for size in (8, 12, 16):
        size_rows = []
        for repeat in range(repeats):
            candidate = np.full((24, 15), np.nan, dtype=np.float64)
            for fold, (training_subjects, validation_subjects) in enumerate(
                outer_subject_folds()
            ):
                reference_subjects = np.sort(
                    rng.choice(training_subjects, size=size, replace=False)
                )
                for video in range(1, 16):
                    template = normative_trajectory(
                        index, reference_subjects, video, radius=3
                    )[0]
                    for subject in validation_subjects:
                        target = index.targets[
                            trial_rows(index, int(subject), video)
                        ].astype(np.float64)
                        mapping_v = constrained_dtw_mapping(
                            target[:, 0], template[:, 0], config
                        )
                        mapping_a = constrained_dtw_mapping(
                            target[:, 1], template[:, 1], config
                        )
                        mapping = np.floor(
                            (mapping_v.astype(float) + mapping_a.astype(float)) / 2.0
                            + 0.5
                        ).astype(np.int16)
                        mapping = np.maximum.accumulate(
                            np.clip(mapping, 0, len(template) - 1)
                        )
                        lag = np.arange(len(target)) - mapping
                        candidate[int(subject) - 1, video - 1] = float(
                            np.abs(lag).mean()
                        )
            profile_primary = primary_matrix.mean(axis=1)
            profile_candidate = candidate.mean(axis=1)
            record = {
                "reference_participants": size,
                "repeat": repeat + 1,
                "trial_pearson": safe_correlation(primary_matrix, candidate),
                "trial_spearman": safe_correlation(
                    primary_matrix, candidate, spearman=True
                ),
                "trial_icc_consistency": icc_consistency(primary_matrix, candidate),
                "trial_mae_seconds": float(np.abs(primary_matrix - candidate).mean()),
                "profile_pearson": safe_correlation(
                    profile_primary, profile_candidate
                ),
                "profile_spearman": safe_correlation(
                    profile_primary, profile_candidate, spearman=True
                ),
                "profile_icc_consistency": icc_consistency(
                    profile_primary, profile_candidate
                ),
                "profile_mae_seconds": float(
                    np.abs(profile_primary - profile_candidate).mean()
                ),
                "maximum_profile_shift_seconds": float(
                    np.abs(profile_primary - profile_candidate).max()
                ),
            }
            rows.append(record)
            size_rows.append(record)
            print(
                f"reference size={size} repeat={repeat + 1}/{repeats}",
                flush=True,
            )
        summaries[str(size)] = {
            key: {
                "mean": float(
                    np.mean([float(record[key]) for record in size_rows])
                ),
                "ci95": interval(
                    np.asarray([float(record[key]) for record in size_rows])
                ),
            }
            for key in (
                "trial_pearson",
                "trial_spearman",
                "trial_icc_consistency",
                "trial_mae_seconds",
                "profile_pearson",
                "profile_spearman",
                "profile_icc_consistency",
                "profile_mae_seconds",
                "maximum_profile_shift_seconds",
            )
        }
    return rows, {
        "primary_reference_size": (
            "19 outer-training participants in four folds and 20 in one fold"
        ),
        "subsample_sizes": [8, 12, 16],
        "repeats_per_size": repeats,
        "metrics": summaries,
    }


def run_geometry_adjustment(
    index,
    primary_matrix: np.ndarray,
    innovation_cache: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    feature_names = (
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
    for fold, (training_subjects, validation_subjects) in enumerate(
        outer_subject_folds()
    ):
        cache = np.load(
            innovation_cache / "phase_sensing_targets" / f"fold_{fold}.npz",
            allow_pickle=False,
        )
        amplitude = cache["amplitude"]
        for video in range(1, 16):
            template = normative_trajectory(
                index, training_subjects, video, radius=3
            )[0]
            template_gradient = float(
                np.linalg.norm(np.gradient(template, axis=0), axis=1).mean()
            )
            for subject in validation_subjects:
                indices = trial_rows(index, int(subject), video)
                target = index.targets[indices].astype(np.float64)
                difference = np.diff(target, axis=0)
                speed = np.linalg.norm(difference, axis=1)
                axis_correlation = safe_correlation(target[:, 0], target[:, 1])
                record = {
                    "subject": int(subject),
                    "video": video,
                    "q_seconds": float(primary_matrix[int(subject) - 1, video - 1]),
                    "trajectory_range": float(
                        np.ptp(target, axis=0).mean()
                    ),
                    "trajectory_variance": float(target.var(axis=0).mean()),
                    "joystick_path_length_per_second": float(
                        speed.sum() / max(len(target), 1)
                    ),
                    "mean_joystick_speed": float(speed.mean()),
                    "axis_correlation": axis_correlation,
                    "group_reference_gradient": template_gradient,
                    "magnitude_residual": float(
                        np.abs(amplitude[indices]).mean()
                    ),
                    "distance_from_center": float(
                        np.linalg.norm(target - 128.0, axis=1).mean()
                    ),
                    "floor_ceiling_occupancy": float(
                        np.mean((target <= 13.0) | (target >= 243.0))
                    ),
                }
                rows.append(record)

    rows.sort(key=lambda row: (int(row["subject"]), int(row["video"])))
    y = np.asarray([float(row["q_seconds"]) for row in rows])
    geometry = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows]
    )
    geometry = (geometry - geometry.mean(axis=0)) / np.maximum(
        geometry.std(axis=0), 1e-8
    )
    videos = np.asarray([int(row["video"]) for row in rows])
    video_dummies = np.column_stack(
        [(videos == video).astype(float) for video in range(2, 16)]
    )
    design = np.column_stack([np.ones(len(rows)), geometry, video_dummies])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    residual_matrix = residual.reshape(24, 15)
    raw_components = estimate_variance_components(primary_matrix)
    adjusted_components = estimate_variance_components(residual_matrix)
    raw_profile = primary_matrix.mean(axis=1)
    adjusted_profile = residual_matrix.mean(axis=1)
    correlations = {
        name: {
            "pearson": safe_correlation(
                y, np.asarray([float(row[name]) for row in rows])
            ),
            "spearman": safe_correlation(
                y,
                np.asarray([float(row[name]) for row in rows]),
                spearman=True,
            ),
        }
        for name in feature_names
    }
    return rows, {
        "model": (
            "OLS q_pv on standardized trajectory-geometry covariates plus video "
            "fixed effects; participant variance is re-estimated from residuals"
        ),
        "covariate_correlations": correlations,
        "raw_participant_variance": float(raw_components.participant),
        "geometry_adjusted_participant_variance": float(
            adjusted_components.participant
        ),
        "participant_variance_retained_fraction": float(
            adjusted_components.participant / raw_components.participant
        ),
        "raw_relative_g_15": reliability_for_videos(
            raw_components, 15
        )["relative_g"],
        "geometry_adjusted_relative_g_15": reliability_for_videos(
            adjusted_components, 15
        )["relative_g"],
        "profile_rank_spearman_raw_vs_adjusted": safe_correlation(
            raw_profile, adjusted_profile, spearman=True
        ),
        "coefficients": {
            name: float(coefficients[position + 1])
            for position, name in enumerate(feature_names)
        },
    }


def emotion_categories(path: Path) -> np.ndarray:
    import re

    first_line = path.read_text(encoding="utf-8").splitlines()[1]
    values = [int(value) for value in re.findall(r"\b[0-4]\b", first_line)]
    if len(values) != 15:
        raise RuntimeError("Could not parse 15 targeted-emotion categories")
    return np.asarray(values, dtype=np.int16)


def category_balanced_set(
    categories: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    by_category = {
        category: np.flatnonzero(categories == category)
        for category in sorted(set(int(value) for value in categories))
    }
    selected: list[int] = []
    base = budget // len(by_category)
    remainder = budget % len(by_category)
    for category, videos in by_category.items():
        if base:
            selected.extend(
                int(value)
                for value in rng.choice(videos, size=base, replace=False)
            )
    extra_categories = (
        rng.choice(list(by_category), size=remainder, replace=False)
        if remainder
        else []
    )
    for category in extra_categories:
        available = np.setdiff1d(
            by_category[int(category)], np.asarray(selected, dtype=int)
        )
        selected.append(int(rng.choice(available)))
    return np.asarray(sorted(selected), dtype=int)


def run_category_balanced_calibration(
    index,
    primary_matrix: np.ndarray,
    data_root: Path,
    innovation_cache: Path,
    personalization_dir: Path,
    *,
    repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    selections = json.loads(
        (personalization_dir / "fold_selections.json").read_text(encoding="utf-8")
    )
    categories = emotion_categories(data_root / "Targeted_emotions.txt")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for fold, (training_subjects, validation_subjects) in enumerate(
        outer_subject_folds()
    ):
        lag = np.load(
            innovation_cache / "phase_sensing_targets" / f"fold_{fold}.npz",
            allow_pickle=False,
        )["lag"]
        training_matrix = lag_trial_matrix(index, lag, training_subjects)
        video_mean = training_matrix.mean(axis=0)
        for budget in (1, 2, 4, 8):
            shrinkage = float(
                selections[fold]["budgets"][str(budget)]["additive_shrinkage"]
            )
            for subject in validation_subjects:
                target = primary_matrix[int(subject) - 1]
                for repeat in range(repeats):
                    calibration = category_balanced_set(categories, budget, rng)
                    evaluation = np.setdiff1d(np.arange(15), calibration)
                    prediction = additive_personalization(
                        video_mean,
                        calibration,
                        target[calibration],
                        shrinkage,
                    )
                    baseline_mae = float(
                        np.abs(video_mean[evaluation] - target[evaluation]).mean()
                    )
                    calibrated_mae = float(
                        np.abs(prediction[evaluation] - target[evaluation]).mean()
                    )
                    rows.append(
                        {
                            "fold": fold,
                            "subject": int(subject),
                            "budget": budget,
                            "repeat": repeat,
                            "calibration_videos": " ".join(
                                str(int(value) + 1) for value in calibration
                            ),
                            "categories_covered": int(
                                len(set(int(categories[value]) for value in calibration))
                            ),
                            "mae_seconds": calibrated_mae,
                            "gain_vs_video_mean_seconds": baseline_mae - calibrated_mae,
                        }
                    )

    summary: dict[str, object] = {}
    for budget in (1, 2, 4, 8):
        selected = [row for row in rows if int(row["budget"]) == budget]
        participant_gain = np.asarray(
            [
                np.mean(
                    [
                        float(row["gain_vs_video_mean_seconds"])
                        for row in selected
                        if int(row["subject"]) == subject
                    ]
                )
                for subject in range(1, 25)
            ]
        )
        participant_mae = np.asarray(
            [
                np.mean(
                    [
                        float(row["mae_seconds"])
                        for row in selected
                        if int(row["subject"]) == subject
                    ]
                )
                for subject in range(1, 25)
            ]
        )
        summary[str(budget)] = {
            "participant_macro_mae_seconds": float(participant_mae.mean()),
            "gain_vs_video_mean_seconds": float(participant_gain.mean()),
            "participants_improved": int(np.sum(participant_gain > 0)),
            "participant_gain_median_seconds": float(np.median(participant_gain)),
            "participant_gain_iqr_seconds": [
                float(np.quantile(participant_gain, 0.25)),
                float(np.quantile(participant_gain, 0.75)),
            ],
            "worst_participant_gain_seconds": float(participant_gain.min()),
            "mean_categories_covered": float(
                np.mean([float(row["categories_covered"]) for row in selected])
            ),
        }
    return rows, {
        "selection": (
            "allocate videos as evenly as possible across the five released "
            "target-emotion categories; use the same fold-selected additive "
            "shrinkage as the random-calibration analysis"
        ),
        "aggregate": summary,
    }


def main() -> None:
    args = parse_args()
    data_root = resolve(args.data_root)
    innovation_cache = resolve(args.innovation_cache)
    trait_dir = resolve(args.trait_dir)
    personalization_dir = resolve(args.personalization_dir)
    output_dir = resolve(args.output_dir)
    prepare_output(output_dir, args.overwrite)
    started = time.perf_counter()

    index = load_innovation_index(data_root)
    primary_matrix = np.load(
        trait_dir / "phase_trait_matrix.npz", allow_pickle=False
    )["mean_absolute_lag"].astype(np.float64)

    own_rows, own_summary, own_null = run_own_other_warp(
        index,
        bootstrap_repeats=args.bootstrap_repeats,
        permutation_repeats=args.permutation_repeats,
        seed=args.seed,
    )
    write_rows(output_dir / "own_other_warp.csv", own_rows)
    np.savez_compressed(
        output_dir / "own_other_warp_null.npz",
        donor_reassignment_advantage=own_null.astype(np.float32),
    )

    prediction_stack, sensing_rows, sensing_summary, selections = (
        run_matched_target_sensing(
            index,
            primary_matrix,
            innovation_cache,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed + 10000,
        )
    )
    write_rows(output_dir / "matched_target_results.csv", sensing_rows)
    np.savez_compressed(
        output_dir / "matched_target_predictions.npz",
        predictions=prediction_stack.astype(np.float32),
        methods=np.asarray(SENSING_METHODS),
        target=primary_matrix.astype(np.float32),
    )
    matched_rows, matched_summary = calibration_split_comparison(
        prediction_stack,
        primary_matrix,
        personalization_dir,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 20000,
    )

    reference_rows, reference_summary = run_reference_sensitivity(
        index,
        primary_matrix,
        repeats=args.reference_repeats,
        seed=args.seed + 30000,
    )
    write_rows(output_dir / "reference_sensitivity.csv", reference_rows)

    geometry_rows, geometry_summary = run_geometry_adjustment(
        index, primary_matrix, innovation_cache
    )
    write_rows(output_dir / "geometry_covariates.csv", geometry_rows)

    category_rows, category_summary = run_category_balanced_calibration(
        index,
        primary_matrix,
        data_root,
        innovation_cache,
        personalization_dir,
        repeats=args.calibration_repeats,
        seed=args.seed + 40000,
    )
    write_rows(output_dir / "category_balanced_calibration.csv", category_rows)

    summary = {
        "protocol": {
            "outer_folds": 5,
            "reference": "outer-training-only group-reference trajectory",
            "participants": 24,
            "videos": 15,
            "target": (
                "q_pv = trial mean of the absolute local signed temporal "
                "deviation |delta_pv(t)|"
            ),
        },
        "own_other_warp": own_summary,
        "matched_target_trial_sensing": sensing_summary,
        "matched_noncalibration_video_comparison": matched_summary,
        "reference_composition_sensitivity": reference_summary,
        "trajectory_geometry_adjustment": geometry_summary,
        "category_balanced_calibration": category_summary,
        "model_selections": selections,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "configuration": {
                "bootstrap_repeats": args.bootstrap_repeats,
                "permutation_repeats": args.permutation_repeats,
                "reference_repeats": args.reference_repeats,
                "calibration_repeats": args.calibration_repeats,
                "seed": args.seed,
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "generated_files": sorted(MANAGED_FILES),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
