"""Few-shot cold-start personalization for participant phase profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LowRankProfile:
    video_mean: np.ndarray
    video_factors: np.ndarray


def fit_low_rank_profile(matrix: np.ndarray, rank: int) -> LowRankProfile:
    matrix = np.asarray(matrix, dtype=np.float64)
    video_mean = matrix.mean(axis=0)
    centered = matrix - video_mean[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = max(1, min(int(rank), vt.shape[0], vt.shape[1]))
    # Columns are video embeddings used to infer a new participant factor.
    video_factors = vt[:rank].T
    return LowRankProfile(video_mean, video_factors)


def additive_personalization(
    video_mean: np.ndarray,
    observed_video_indices: np.ndarray,
    observed_values: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    video_mean = np.asarray(video_mean, dtype=np.float64)
    observed_video_indices = np.asarray(observed_video_indices, dtype=np.int64)
    observed_values = np.asarray(observed_values, dtype=np.float64)
    if len(observed_video_indices) == 0:
        return video_mean.copy()
    residual = observed_values - video_mean[observed_video_indices]
    offset = residual.sum() / (len(residual) + float(shrinkage))
    return video_mean + offset


def low_rank_personalization(
    profile: LowRankProfile,
    observed_video_indices: np.ndarray,
    observed_values: np.ndarray,
    ridge: float,
) -> np.ndarray:
    observed_video_indices = np.asarray(observed_video_indices, dtype=np.int64)
    observed_values = np.asarray(observed_values, dtype=np.float64)
    if len(observed_video_indices) == 0:
        return profile.video_mean.copy()
    factors = profile.video_factors[observed_video_indices]
    design = np.concatenate(
        [np.ones((len(factors), 1), dtype=np.float64), factors], axis=1
    )
    target = observed_values - profile.video_mean[observed_video_indices]
    gram = design.T @ design
    regularization = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    # Intercept is shrunk less strongly than the interaction factors.
    regularization[0, 0] *= 0.25
    coefficient = np.linalg.solve(
        gram + regularization, design.T @ target
    )
    full_design = np.concatenate(
        [
            np.ones((len(profile.video_mean), 1), dtype=np.float64),
            profile.video_factors,
        ],
        axis=1,
    )
    return profile.video_mean + full_design @ coefficient
