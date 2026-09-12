"""Leakage-safe targets and linear probes for physiological phase sensing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from merps.innovation.data import InnovationIndex
from merps.innovation.normative_dynamics import WarpConfig, decompose_trial, smooth_curve


RIDGE_ALPHAS = (10.0, 100.0, 1000.0, 10000.0, 100000.0)


@dataclass
class RidgePath:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_cross: np.ndarray

    def predict(self, x: np.ndarray, alpha: float) -> np.ndarray:
        values = (np.asarray(x, dtype=np.float32) - self.x_mean) / self.x_scale
        weights = self.eigenvectors @ (
            self.projected_cross / (self.eigenvalues[:, None] + float(alpha))
        )
        return np.asarray(values @ weights + self.y_mean, dtype=np.float32)


def fit_ridge_path(x: np.ndarray, y: np.ndarray) -> RidgePath:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    x_mean = x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    x_scale = np.maximum(
        x.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32), 1e-6
    )
    y_mean = y.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    normalized = np.asarray((x - x_mean) / x_scale, dtype=np.float64)
    centered_target = np.asarray(y - y_mean, dtype=np.float64)
    gram = normalized.T @ normalized
    cross = normalized.T @ centered_target
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_cross = eigenvectors.T @ cross
    return RidgePath(
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        eigenvalues=eigenvalues.astype(np.float64),
        eigenvectors=eigenvectors.astype(np.float64),
        projected_cross=projected_cross.astype(np.float64),
    )


def ensure_pooled_node_features(
    source_path: Path,
    output_path: Path,
    *,
    expected_samples: int,
    expected_nodes: int,
    expected_features: int,
) -> np.ndarray:
    expected_shape = (expected_samples, expected_features * 2)
    if output_path.exists():
        cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
        if cached.shape == expected_shape:
            return cached
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.shape != (expected_samples, expected_nodes, expected_features):
        raise ValueError(f"Unexpected feature shape at {source_path}: {source.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pooled = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=expected_shape
    )
    for start in range(0, expected_samples, 512):
        end = min(start + 512, expected_samples)
        block = np.asarray(source[start:end], dtype=np.float32)
        pooled[start:end, :expected_features] = block.mean(axis=1)
        pooled[start:end, expected_features:] = block.std(axis=1)
    pooled.flush()
    return np.load(output_path, mmap_mode="r", allow_pickle=False)


def _video_cube(
    index: InnovationIndex, subjects: Sequence[int], video: int
) -> tuple[np.ndarray, np.ndarray]:
    subject_values = np.asarray(subjects, dtype=np.int16)
    trajectories = []
    reference_rows = None
    for subject in subject_values:
        rows = np.flatnonzero(
            (index.subject_numbers == int(subject)) & (index.videos == int(video))
        )
        rows = rows[np.argsort(index.timestamps[rows])]
        trajectories.append(index.targets[rows].astype(np.float64))
        if reference_rows is None:
            reference_rows = rows
    cube = np.stack(trajectories, axis=0)
    return cube, np.asarray(reference_rows, dtype=np.int64)


def _template(cube: np.ndarray, radius: int) -> np.ndarray:
    return smooth_curve(np.median(cube, axis=0), radius)


def build_outer_phase_targets(
    index: InnovationIndex,
    training_subjects: Sequence[int],
    validation_subjects: Sequence[int],
    *,
    warp_config: WarpConfig,
    template_radius: int = 3,
) -> dict[str, np.ndarray]:
    """Build train cross-fitted and validation held-out phase targets.

    No outer-validation label contributes to a training template. Each
    training participant is also excluded from the template defining their own
    target.
    """

    training_subjects = np.asarray(training_subjects, dtype=np.int16)
    validation_subjects = np.asarray(validation_subjects, dtype=np.int16)
    selected = np.isin(
        index.subject_numbers,
        np.concatenate([training_subjects, validation_subjects]),
    )
    lag = np.full(len(index.targets), np.nan, dtype=np.float32)
    amplitude = np.full((len(index.targets), 2), np.nan, dtype=np.float32)
    boundary = np.zeros(len(index.targets), dtype=bool)
    mapping_v = np.full(len(index.targets), -1, dtype=np.int16)
    mapping_a = np.full(len(index.targets), -1, dtype=np.int16)
    for video in range(1, 16):
        training_cube, _ = _video_cube(index, training_subjects, video)
        validation_template = _template(training_cube, template_radius)
        for local_subject, subject in enumerate(training_subjects):
            keep = np.arange(len(training_subjects)) != local_subject
            subject_template = _template(training_cube[keep], template_radius)
            rows = np.flatnonzero(
                (index.subject_numbers == int(subject)) & (index.videos == video)
            )
            rows = rows[np.argsort(index.timestamps[rows])]
            result = decompose_trial(index.targets[rows], subject_template, warp_config)
            lag[rows] = result.consensus_lag
            amplitude[rows] = result.amplitude_residual
            boundary[rows] = result.boundary_mask
            mapping_v[rows] = result.mapping_from_valence
            mapping_a[rows] = result.mapping_from_arousal
        for subject in validation_subjects:
            rows = np.flatnonzero(
                (index.subject_numbers == int(subject)) & (index.videos == video)
            )
            rows = rows[np.argsort(index.timestamps[rows])]
            result = decompose_trial(index.targets[rows], validation_template, warp_config)
            lag[rows] = result.consensus_lag
            amplitude[rows] = result.amplitude_residual
            boundary[rows] = result.boundary_mask
            mapping_v[rows] = result.mapping_from_valence
            mapping_a[rows] = result.mapping_from_arousal
    rows = np.flatnonzero(selected)
    if (
        not np.isfinite(lag[rows]).all()
        or not np.isfinite(amplitude[rows]).all()
        or np.any(mapping_v[rows] < 0)
        or np.any(mapping_a[rows] < 0)
    ):
        raise RuntimeError("Outer phase targets are incomplete")
    return {
        "lag": lag,
        "amplitude": amplitude,
        "boundary": boundary,
        "mapping_from_valence": mapping_v,
        "mapping_from_arousal": mapping_a,
    }


def participant_macro_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
) -> float:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    subjects = np.asarray(subjects)
    return float(
        np.mean(
            [
                np.abs(prediction[subjects == subject] - target[subjects == subject]).mean()
                for subject in sorted(set(int(value) for value in subjects))
            ]
        )
    )


def trial_macro_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    videos: np.ndarray,
) -> float:
    scores = []
    for subject in sorted(set(int(value) for value in subjects)):
        for video in sorted(set(int(value) for value in videos)):
            keep = (subjects == subject) & (videos == video)
            if np.any(keep):
                scores.append(float(np.abs(prediction[keep] - target[keep]).mean()))
    return float(np.mean(scores))


def binary_auroc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    positive = score[target]
    negative = score[~target]
    if not len(positive) or not len(negative):
        return 0.5
    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=np.float64)
    start = 0
    while start < len(combined):
        end = start + 1
        while end < len(combined) and combined[order[end]] == combined[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = ranks[: len(positive)].sum()
    return float(
        (rank_sum - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    ranked = target[order]
    positives = int(ranked.sum())
    if positives == 0:
        return 0.0
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)
