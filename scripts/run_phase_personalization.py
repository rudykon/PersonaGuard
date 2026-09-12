#!/usr/bin/env python3
"""Compare few-shot phase calibration with zero-shot context and physiology.

The target is each participant-video trial's mean absolute temporal deviation
from a fold-safe normative affect trajectory. The experiment treats a new
participant as a cold-start user: k calibration videos are observed, then the
remaining videos are predicted with additive or low-rank collaborative
filtering learned only from outer-training participants.
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

from merps.innovation.data import (
    build_context_features,
    build_cross_fitted_context,
    grouped_subject_folds,
    load_innovation_index,
    outer_subject_folds,
)
from merps.innovation.personalization import (
    additive_personalization,
    fit_low_rank_profile,
    low_rank_personalization,
)
from merps.innovation.phase_sensing import (
    RIDGE_ALPHAS,
    ensure_pooled_node_features,
    fit_ridge_path,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MER_PS_trainval"
DEFAULT_FEATURE_CACHE = PROJECT_ROOT / "data" / "feature_cache"
DEFAULT_INNOVATION_CACHE = PROJECT_ROOT / "data" / "innovation_cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "phase_personalization"
BUDGETS = (0, 1, 2, 4, 8)
ADDITIVE_SHRINKAGE = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
LOWRANK_RANKS = (1, 2, 3, 4)
LOWRANK_RIDGES = (0.1, 1.0, 10.0, 100.0)
METHODS = (
    "video_mean_zero_shot",
    "context_zero_shot",
    "joint_physiology_zero_shot",
    "additive_fewshot",
    "lowrank_fewshot",
)
MANAGED_FILES = {
    "fold_selections.json",
    "results.csv",
    "run_manifest.json",
    "summary.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument(
        "--innovation-cache", type=Path, default=DEFAULT_INNOVATION_CACHE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--inner-repeats", type=int, default=30)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260726)
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


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 3 or first.std() < 1e-12 or second.std() < 1e-12:
        return 0.0
    first_rank = np.argsort(np.argsort(first, kind="mergesort"), kind="mergesort")
    second_rank = np.argsort(
        np.argsort(second, kind="mergesort"), kind="mergesort"
    )
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def trial_matrix(
    index,
    values: np.ndarray,
    subjects: Sequence[int],
) -> np.ndarray:
    subjects = np.asarray(subjects, dtype=np.int16)
    matrix = np.empty((len(subjects), 15), dtype=np.float64)
    for subject_position, subject in enumerate(subjects):
        for video in range(1, 16):
            rows = np.flatnonzero(
                (index.subject_numbers == int(subject)) & (index.videos == video)
            )
            matrix[subject_position, video - 1] = float(
                np.abs(values[rows]).mean()
            )
    return matrix


def trial_prediction_vector(
    index,
    sample_prediction: np.ndarray,
    subject: int,
) -> np.ndarray:
    output = np.empty(15, dtype=np.float64)
    for video in range(1, 16):
        rows = np.flatnonzero(
            (index.subject_numbers == int(subject)) & (index.videos == video)
        )
        output[video - 1] = float(sample_prediction[rows].mean())
    return output


def zero_shot_feature_matrices(
    rows: np.ndarray,
    context: np.ndarray,
    eeg_foundation: np.ndarray,
    fnirs: np.ndarray,
) -> dict[str, np.ndarray]:
    context = np.asarray(context, dtype=np.float32)
    eeg = np.asarray(eeg_foundation[rows], dtype=np.float32)
    fnirs_values = np.asarray(fnirs[rows], dtype=np.float32)
    return {
        "context_zero_shot": context,
        "joint_physiology_zero_shot": np.concatenate(
            [context, eeg, fnirs_values], axis=1
        ),
    }


def trial_profile_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    videos: np.ndarray,
) -> float:
    errors = []
    for subject in sorted(set(int(value) for value in subjects)):
        for video in range(1, 16):
            keep = (subjects == subject) & (videos == video)
            errors.append(
                abs(float(prediction[keep].mean()) - float(target[keep].mean()))
            )
    return float(np.mean(errors))


def select_zero_shot_alpha(
    x: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    videos: np.ndarray,
    outer_training_subjects: np.ndarray,
    inner_folds: int,
) -> dict[str, object]:
    predictions = {
        alpha: np.full(len(x), np.nan, dtype=np.float32)
        for alpha in RIDGE_ALPHAS
    }
    folds = []
    for fold, (training_subjects, validation_subjects) in enumerate(
        grouped_subject_folds(outer_training_subjects, inner_folds)
    ):
        train = np.flatnonzero(np.isin(subjects, training_subjects))
        validation = np.flatnonzero(np.isin(subjects, validation_subjects))
        model = fit_ridge_path(x[train], target[train])
        for alpha in RIDGE_ALPHAS:
            prediction = model.predict(x[validation], alpha).reshape(-1)
            predictions[alpha][validation] = np.clip(prediction, 0.0, 10.0)
        folds.append(
            {
                "fold": fold,
                "training_subjects": training_subjects.tolist(),
                "validation_subjects": validation_subjects.tolist(),
            }
        )
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("Zero-shot inner OOF predictions are incomplete")
    scores = {
        str(alpha): trial_profile_mae(
            prediction, target, subjects, videos
        )
        for alpha, prediction in predictions.items()
    }
    best = min(RIDGE_ALPHAS, key=lambda alpha: (scores[str(alpha)], alpha))
    return {
        "alpha": float(best),
        "trial_profile_mae_scores": scores,
        "inner_folds": folds,
    }


def calibration_sets(
    *,
    participants: int,
    budget: int,
    repeats: int,
    seed: int,
) -> dict[tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    output = {}
    for participant in range(participants):
        for repeat in range(repeats):
            output[(participant, repeat)] = np.sort(
                rng.choice(15, size=budget, replace=False)
            )
    return output


def select_hyperparameters(
    matrix: np.ndarray,
    *,
    budget: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    if budget == 0:
        return {
            "additive_shrinkage": 0.0,
            "lowrank_rank": 1,
            "lowrank_ridge": 1.0,
            "additive_scores": {},
            "lowrank_scores": {},
        }
    subsets = calibration_sets(
        participants=len(matrix), budget=budget, repeats=repeats, seed=seed
    )
    additive_scores = {value: [] for value in ADDITIVE_SHRINKAGE}
    lowrank_scores = {
        (rank, ridge): []
        for rank in LOWRANK_RANKS
        for ridge in LOWRANK_RIDGES
    }
    for participant in range(len(matrix)):
        reference = np.delete(matrix, participant, axis=0)
        video_mean = reference.mean(axis=0)
        profiles = {
            rank: fit_low_rank_profile(reference, rank)
            for rank in LOWRANK_RANKS
        }
        target = matrix[participant]
        for repeat in range(repeats):
            calibration = subsets[(participant, repeat)]
            evaluation = np.setdiff1d(np.arange(15), calibration)
            for shrinkage in ADDITIVE_SHRINKAGE:
                prediction = additive_personalization(
                    video_mean,
                    calibration,
                    target[calibration],
                    shrinkage,
                )
                additive_scores[shrinkage].append(
                    float(np.abs(prediction[evaluation] - target[evaluation]).mean())
                )
            for rank in LOWRANK_RANKS:
                for ridge in LOWRANK_RIDGES:
                    prediction = low_rank_personalization(
                        profiles[rank],
                        calibration,
                        target[calibration],
                        ridge,
                    )
                    lowrank_scores[(rank, ridge)].append(
                        float(
                            np.abs(
                                prediction[evaluation] - target[evaluation]
                            ).mean()
                        )
                    )
    best_additive = min(
        ADDITIVE_SHRINKAGE,
        key=lambda value: (np.mean(additive_scores[value]), value),
    )
    best_lowrank = min(
        lowrank_scores,
        key=lambda value: (
            np.mean(lowrank_scores[value]),
            value[0],
            value[1],
        ),
    )
    return {
        "additive_shrinkage": float(best_additive),
        "lowrank_rank": int(best_lowrank[0]),
        "lowrank_ridge": float(best_lowrank[1]),
        "additive_scores": {
            str(key): float(np.mean(value))
            for key, value in additive_scores.items()
        },
        "lowrank_scores": {
            f"rank={key[0]},ridge={key[1]}": float(np.mean(value))
            for key, value in lowrank_scores.items()
        },
    }


def summarize_rows(
    rows: list[dict[str, object]],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    summary = {}
    for budget in BUDGETS:
        summary[str(budget)] = {}
        for method in METHODS:
            selected = [
                row
                for row in rows
                if int(row["budget"]) == budget and row["method"] == method
            ]
            participant_mae = []
            participant_delta_video = []
            participant_delta_context = []
            participant_delta_joint = []
            participant_spearman = []
            for subject in range(1, 25):
                subject_rows = [
                    row for row in selected if int(row["subject"]) == subject
                ]
                participant_mae.append(
                    float(np.mean([float(row["mae_seconds"]) for row in subject_rows]))
                )
                participant_delta_video.append(
                    float(
                        np.mean(
                            [
                                float(row["delta_vs_video_mean_seconds"])
                                for row in subject_rows
                            ]
                        )
                    )
                )
                participant_delta_context.append(
                    float(
                        np.mean(
                            [
                                float(row["delta_vs_context_seconds"])
                                for row in subject_rows
                            ]
                        )
                    )
                )
                participant_delta_joint.append(
                    float(
                        np.mean(
                            [
                                float(row["delta_vs_joint_physiology_seconds"])
                                for row in subject_rows
                            ]
                        )
                    )
                )
                participant_spearman.append(
                    float(np.mean([float(row["spearman"]) for row in subject_rows]))
                )
            participant_mae = np.asarray(participant_mae)
            participant_delta_video = np.asarray(participant_delta_video)
            participant_delta_context = np.asarray(participant_delta_context)
            participant_delta_joint = np.asarray(participant_delta_joint)
            bootstrap_video = np.empty(bootstrap_repeats, dtype=np.float64)
            bootstrap_context = np.empty(bootstrap_repeats, dtype=np.float64)
            bootstrap_joint = np.empty(bootstrap_repeats, dtype=np.float64)
            for repeat in range(bootstrap_repeats):
                draw = rng.integers(0, 24, 24)
                bootstrap_video[repeat] = participant_delta_video[draw].mean()
                bootstrap_context[repeat] = participant_delta_context[draw].mean()
                bootstrap_joint[repeat] = participant_delta_joint[draw].mean()
            summary[str(budget)][method] = {
                "participant_macro_mae_seconds": float(participant_mae.mean()),
                "participant_macro_spearman": float(
                    np.mean(participant_spearman)
                ),
                "paired_gain_vs_video_mean_seconds": float(
                    participant_delta_video.mean()
                ),
                "participant_gain_heterogeneity_vs_video_mean": {
                    "participants_improved": int(np.sum(participant_delta_video > 0)),
                    "participants_harmed": int(np.sum(participant_delta_video < 0)),
                    "proportion_improved": float(np.mean(participant_delta_video > 0)),
                    "median_seconds": float(np.median(participant_delta_video)),
                    "iqr_seconds": [
                        float(np.quantile(participant_delta_video, 0.25)),
                        float(np.quantile(participant_delta_video, 0.75)),
                    ],
                    "range_seconds": [
                        float(participant_delta_video.min()),
                        float(participant_delta_video.max()),
                    ],
                    "participant_values_seconds": {
                        str(subject): float(participant_delta_video[subject - 1])
                        for subject in range(1, 25)
                    },
                },
                "participant_bootstrap_gain_ci95": [
                    float(np.quantile(bootstrap_video, 0.025)),
                    float(np.quantile(bootstrap_video, 0.975)),
                ],
                "paired_gain_vs_context_seconds": float(
                    participant_delta_context.mean()
                ),
                "participant_bootstrap_gain_vs_context_ci95": [
                    float(np.quantile(bootstrap_context, 0.025)),
                    float(np.quantile(bootstrap_context, 0.975)),
                ],
                "paired_gain_vs_joint_physiology_seconds": float(
                    participant_delta_joint.mean()
                ),
                "participant_bootstrap_gain_vs_joint_physiology_ci95": [
                    float(np.quantile(bootstrap_joint, 0.025)),
                    float(np.quantile(bootstrap_joint, 0.975)),
                ],
            }
    return summary


def main() -> None:
    args = parse_args()
    data_root = resolve(args.data_root)
    feature_cache = resolve(args.feature_cache)
    innovation_cache = resolve(args.innovation_cache)
    output_dir = resolve(args.output_dir)
    prepare_output(output_dir, args.overwrite)
    started = time.perf_counter()
    index = load_innovation_index(data_root)
    fnirs = ensure_pooled_node_features(
        feature_cache / "fnirs.npy",
        innovation_cache / "handcrafted_fnirs_pooled.npy",
        expected_samples=len(index.targets),
        expected_nodes=51,
        expected_features=90,
    )
    eeg_foundation = np.load(
        innovation_cache / "cbramod" / "pooled.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    zero_shot_predictions = {
        name: np.full(len(index.targets), np.nan, dtype=np.float32)
        for name in ("context_zero_shot", "joint_physiology_zero_shot")
    }
    result_rows: list[dict[str, object]] = []
    selections = []

    for fold, (training_subjects, validation_subjects) in enumerate(
        outer_subject_folds()
    ):
        target_cache = np.load(
            innovation_cache / "phase_sensing_targets" / f"fold_{fold}.npz",
            allow_pickle=False,
        )
        if (
            "target_version" not in target_cache.files
            or int(target_cache["target_version"]) != 2
        ):
            raise RuntimeError("Phase target cache is stale; rerun phase sensing")
        lag = target_cache["lag"]
        train_idx = index.indices_for_subjects(training_subjects)
        validation_idx = index.indices_for_subjects(validation_subjects)
        training_matrix = trial_matrix(index, lag, training_subjects)
        validation_matrix = trial_matrix(index, lag, validation_subjects)
        video_mean = training_matrix.mean(axis=0)
        _, train_context = build_cross_fitted_context(
            index, training_subjects, train_idx, radius=3
        )
        _, validation_context = build_context_features(
            index, training_subjects, validation_idx, radius=3
        )
        train_features = zero_shot_feature_matrices(
            train_idx, train_context, eeg_foundation, fnirs
        )
        validation_features = zero_shot_feature_matrices(
            validation_idx, validation_context, eeg_foundation, fnirs
        )
        zero_shot_selection = {}
        for feature_name in ("context_zero_shot", "joint_physiology_zero_shot"):
            selected = select_zero_shot_alpha(
                train_features[feature_name],
                np.abs(lag[train_idx]),
                index.subject_numbers[train_idx],
                index.videos[train_idx],
                training_subjects,
                args.inner_folds,
            )
            model = fit_ridge_path(
                train_features[feature_name], np.abs(lag[train_idx])
            )
            prediction = model.predict(
                validation_features[feature_name], float(selected["alpha"])
            ).reshape(-1)
            zero_shot_predictions[feature_name][validation_idx] = np.clip(
                prediction, 0.0, 10.0
            )
            zero_shot_selection[feature_name] = selected
        fold_selection = {
            "fold": fold,
            "training_subjects": training_subjects.tolist(),
            "validation_subjects": validation_subjects.tolist(),
            "zero_shot_models": zero_shot_selection,
            "budgets": {},
        }
        for budget in BUDGETS:
            selection = select_hyperparameters(
                training_matrix,
                budget=budget,
                repeats=args.inner_repeats,
                seed=args.seed + fold * 1000 + budget * 10,
            )
            fold_selection["budgets"][str(budget)] = selection
            profile = fit_low_rank_profile(
                training_matrix, int(selection["lowrank_rank"])
            )
            outer_subsets = calibration_sets(
                participants=len(validation_subjects),
                budget=budget,
                repeats=1 if budget == 0 else args.repeats,
                seed=args.seed + 50000 + fold * 1000 + budget * 10,
            )
            for subject_position, subject in enumerate(validation_subjects):
                target = validation_matrix[subject_position]
                context_trial = trial_prediction_vector(
                    index, zero_shot_predictions["context_zero_shot"], int(subject)
                )
                joint_trial = trial_prediction_vector(
                    index,
                    zero_shot_predictions["joint_physiology_zero_shot"],
                    int(subject),
                )
                repeat_count = 1 if budget == 0 else args.repeats
                for repeat in range(repeat_count):
                    calibration = outer_subsets[(subject_position, repeat)]
                    evaluation = np.setdiff1d(np.arange(15), calibration)
                    additive = additive_personalization(
                        video_mean,
                        calibration,
                        target[calibration],
                        float(selection["additive_shrinkage"]),
                    )
                    lowrank = low_rank_personalization(
                        profile,
                        calibration,
                        target[calibration],
                        float(selection["lowrank_ridge"]),
                    )
                    predictions = {
                        "video_mean_zero_shot": video_mean,
                        "context_zero_shot": context_trial,
                        "joint_physiology_zero_shot": joint_trial,
                        "additive_fewshot": additive,
                        "lowrank_fewshot": lowrank,
                    }
                    baseline_mae = float(
                        np.abs(video_mean[evaluation] - target[evaluation]).mean()
                    )
                    context_mae = float(
                        np.abs(context_trial[evaluation] - target[evaluation]).mean()
                    )
                    joint_mae = float(
                        np.abs(joint_trial[evaluation] - target[evaluation]).mean()
                    )
                    for method, prediction in predictions.items():
                        mae = float(
                            np.abs(prediction[evaluation] - target[evaluation]).mean()
                        )
                        result_rows.append(
                            {
                                "fold": fold,
                                "subject": int(subject),
                                "budget": budget,
                                "repeat": repeat,
                                "method": method,
                                "calibration_videos": " ".join(
                                    str(int(value) + 1) for value in calibration
                                ),
                                "evaluation_videos": int(len(evaluation)),
                                "mae_seconds": mae,
                                "delta_vs_video_mean_seconds": baseline_mae - mae,
                                "delta_vs_context_seconds": context_mae - mae,
                                "delta_vs_joint_physiology_seconds": joint_mae - mae,
                                "spearman": safe_spearman(
                                    prediction[evaluation], target[evaluation]
                                ),
                            }
                        )
            print(
                f"fold={fold} budget={budget} "
                f"additive={selection['additive_shrinkage']} "
                f"rank={selection['lowrank_rank']} "
                f"ridge={selection['lowrank_ridge']}",
                flush=True,
            )
        selections.append(fold_selection)

    aggregate = summarize_rows(
        result_rows,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed + 90000,
    )
    passing = []
    for budget in (1, 2, 4):
        for method in ("additive_fewshot", "lowrank_fewshot"):
            value = aggregate[str(budget)][method]
            ci_video = float(value["participant_bootstrap_gain_ci95"][0])
            ci_context = float(
                value["participant_bootstrap_gain_vs_context_ci95"][0]
            )
            ci_joint = float(
                value[
                    "participant_bootstrap_gain_vs_joint_physiology_ci95"
                ][0]
            )
            if min(ci_video, ci_context, ci_joint) > 0:
                passing.append({"budget": budget, "method": method})
    summary = {
        "research_question": (
            "Can a brief behavioral calibration predict a new participant's "
            "phase profile better than zero-shot context or EEG-fNIRS sensing?"
        ),
        "aggregate": aggregate,
        "personalization_gate": {
            "status": "pass" if passing else "stop",
            "rule": (
                "for budget <= 4, participant-bootstrap paired-gain CI lower is "
                "> 0 against video mean, matched-target context, and matched-target "
                "joint EEG-fNIRS baselines"
            ),
            "passing_configurations": passing,
            "next_step": (
                "frame a calibration-over-sensing interaction contribution"
                if passing
                else "retain phase measurement result without a personalization claim"
            ),
        },
        "configuration": {
            "outer_folds": 5,
            "budgets": list(BUDGETS),
            "outer_repeats": args.repeats,
            "inner_repeats": args.inner_repeats,
            "inner_folds": args.inner_folds,
            "additive_shrinkage_candidates": list(ADDITIVE_SHRINKAGE),
            "lowrank_rank_candidates": list(LOWRANK_RANKS),
            "lowrank_ridge_candidates": list(LOWRANK_RIDGES),
            "participant_disjoint": True,
            "zero_shot_target": (
                "sample absolute lag, evaluated as trial mean absolute lag"
            ),
            "zero_shot_ridge_alphas": list(RIDGE_ALPHAS),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    with (output_dir / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    write_json(output_dir / "fold_selections.json", selections)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "configuration": summary["configuration"],
            "personalization_gate": summary["personalization_gate"],
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "generated_files": sorted(MANAGED_FILES),
            "rows": len(result_rows),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
