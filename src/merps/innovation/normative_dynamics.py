"""Temporal registration and post-registration residuals for continuous reports.

The central distinction comes from elastic functional data analysis: two
trajectories can differ because their events occur at different times (phase)
or because their values differ after temporal registration (amplitude).

To avoid claiming the mechanical in-sample gain of dynamic time warping as a
scientific result, the primary test estimates a warp from one affect dimension
and evaluates whether it improves alignment in the held-out dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from merps.innovation.data import InnovationIndex


@dataclass(frozen=True)
class WarpConfig:
    band_seconds: int = 10
    warp_penalty: float = 0.15
    step_penalty: float = 0.05
    smoothing_radius: int = 2
    use_derivative: bool = True


@dataclass
class TrialDecomposition:
    mapping_from_valence: np.ndarray
    mapping_from_arousal: np.ndarray
    lag_from_valence: np.ndarray
    lag_from_arousal: np.ndarray
    consensus_mapping: np.ndarray
    consensus_lag: np.ndarray
    amplitude_residual: np.ndarray
    boundary_mask: np.ndarray
    metrics: dict[str, float]


def smooth_curve(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if radius <= 0:
        return values.copy()
    output = np.empty_like(values)
    for position in range(len(values)):
        start = max(0, position - radius)
        end = min(len(values), position + radius + 1)
        output[position] = values[start:end].mean(axis=0)
    return output


def robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    if scale < 1e-6:
        scale = float(values.std())
    if scale < 1e-6:
        return np.zeros_like(values)
    return (values - median) / scale


def _alignment_signal(values: np.ndarray, config: WarpConfig) -> np.ndarray:
    smoothed = smooth_curve(np.asarray(values, dtype=np.float64), config.smoothing_radius)
    signal = np.gradient(smoothed) if config.use_derivative else smoothed
    return robust_zscore(signal)


def constrained_dtw_mapping(
    participant_curve: np.ndarray,
    normative_curve: np.ndarray,
    config: WarpConfig,
) -> np.ndarray:
    """Map each participant second to one normative second.

    Positive lag, defined later as participant_time - reference_time, means the
    participant report follows the group-reference trajectory with a delay.
    """

    participant = _alignment_signal(participant_curve, config)
    normative = _alignment_signal(normative_curve, config)
    n, m = len(participant), len(normative)
    if n < 2 or m < 2:
        return np.zeros(n, dtype=np.int16)
    band = max(abs(n - m), int(config.band_seconds))
    cost = np.full((n, m), np.inf, dtype=np.float64)
    back = np.full((n, m), -1, dtype=np.int8)
    for i in range(n):
        start = max(0, i - band)
        end = min(m, i + band + 1)
        for j in range(start, end):
            local = abs(participant[i] - normative[j])
            local += config.warp_penalty * abs(i - j) / max(band, 1)
            if i == 0 and j == 0:
                cost[i, j] = local
                back[i, j] = 0
                continue
            best = np.inf
            direction = -1
            if i > 0 and j > 0 and cost[i - 1, j - 1] < best:
                best = cost[i - 1, j - 1]
                direction = 0
            if i > 0 and cost[i - 1, j] + config.step_penalty < best:
                best = cost[i - 1, j] + config.step_penalty
                direction = 1
            if j > 0 and cost[i, j - 1] + config.step_penalty < best:
                best = cost[i, j - 1] + config.step_penalty
                direction = 2
            if np.isfinite(best):
                cost[i, j] = local + best
                back[i, j] = direction
    if not np.isfinite(cost[-1, -1]):
        raise RuntimeError("Constrained DTW failed to connect both trajectory endpoints")
    path: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        direction = int(back[i, j])
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise RuntimeError("DTW backtracking reached an invalid cell")
    path.reverse()
    matches: list[list[int]] = [[] for _ in range(n)]
    for participant_time, normative_time in path:
        matches[participant_time].append(normative_time)
    mapping = np.empty(n, dtype=np.int16)
    previous = 0
    for position, values in enumerate(matches):
        if values:
            previous = int(np.floor(float(np.median(values)) + 0.5))
        mapping[position] = previous
    return np.maximum.accumulate(np.clip(mapping, 0, m - 1)).astype(np.int16)


def normative_trajectory(
    index: InnovationIndex,
    training_subjects: Sequence[int],
    video: int,
    *,
    radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(index.subject_numbers, np.asarray(training_subjects))
    mask &= index.videos == int(video)
    rows = np.flatnonzero(mask)
    if not len(rows):
        raise ValueError(f"No training labels for video {video}")
    length = int(index.timestamps[rows].max()) + 1
    trajectory = np.empty((length, 2), dtype=np.float64)
    uncertainty = np.empty_like(trajectory)
    for timestamp in range(length):
        values = index.targets[mask & (index.timestamps == timestamp)]
        median = np.median(values, axis=0)
        trajectory[timestamp] = median
        uncertainty[timestamp] = 1.4826 * np.median(
            np.abs(values - median[None, :]), axis=0
        )
    return smooth_curve(trajectory, radius), smooth_curve(uncertainty, radius)


def _event_boundary_mask(template: np.ndarray, quantile: float = 0.75) -> np.ndarray:
    gradient = np.linalg.norm(np.gradient(template, axis=0), axis=1)
    # Select an exact top fraction instead of thresholding with >=. Flat
    # trajectories can have many ties at the quantile and would otherwise mark
    # every second as a boundary.
    fraction = float(np.clip(1.0 - quantile, 0.0, 1.0))
    count = max(1, int(np.ceil(len(gradient) * fraction)))
    count = min(count, max(len(gradient) - 1, 1))
    order = np.argsort(-gradient, kind="mergesort")
    mask = np.zeros(len(gradient), dtype=bool)
    mask[order[:count]] = True
    return mask


def decompose_trial(
    target: np.ndarray,
    template: np.ndarray,
    config: WarpConfig,
) -> TrialDecomposition:
    target = np.asarray(target, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    if target.shape != template.shape or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target and template must have matching [time, 2] shapes")
    mapping_v = constrained_dtw_mapping(target[:, 0], template[:, 0], config)
    mapping_a = constrained_dtw_mapping(target[:, 1], template[:, 1], config)
    times = np.arange(len(target), dtype=np.int16)
    lag_v = times - mapping_v
    lag_a = times - mapping_a
    consensus_mapping = np.floor(
        (mapping_v.astype(np.float64) + mapping_a.astype(np.float64)) / 2.0 + 0.5
    ).astype(np.int16)
    consensus_mapping = np.maximum.accumulate(
        np.clip(consensus_mapping, 0, len(template) - 1)
    )
    consensus_lag = times - consensus_mapping
    amplitude_residual = target - template[consensus_mapping]
    boundary_mask = _event_boundary_mask(template)
    stable_mask = ~boundary_mask

    baseline_v = float(np.abs(target[:, 0] - template[:, 0]).mean())
    baseline_a = float(np.abs(target[:, 1] - template[:, 1]).mean())
    # Primary cross-dimensional validation:
    # arousal is evaluated with a warp estimated from valence and vice versa.
    aligned_a_from_v = float(
        np.abs(target[:, 1] - template[mapping_v, 1]).mean()
    )
    aligned_v_from_a = float(
        np.abs(target[:, 0] - template[mapping_a, 0]).mean()
    )
    self_aligned_v = float(
        np.abs(target[:, 0] - template[mapping_v, 0]).mean()
    )
    self_aligned_a = float(
        np.abs(target[:, 1] - template[mapping_a, 1]).mean()
    )
    cross_gain_v = baseline_v - aligned_v_from_a
    cross_gain_a = baseline_a - aligned_a_from_v
    amplitude_magnitude = np.abs(amplitude_residual).mean(axis=1)
    metrics = {
        "baseline_valence_mae": baseline_v,
        "baseline_arousal_mae": baseline_a,
        "aligned_valence_from_arousal_mae": aligned_v_from_a,
        "aligned_arousal_from_valence_mae": aligned_a_from_v,
        "cross_gain_valence": cross_gain_v,
        "cross_gain_arousal": cross_gain_a,
        "cross_gain_mean": float((cross_gain_v + cross_gain_a) / 2.0),
        "self_gain_valence_upper_bound": baseline_v - self_aligned_v,
        "self_gain_arousal_upper_bound": baseline_a - self_aligned_a,
        "median_lag_seconds": float(np.median(consensus_lag)),
        "mean_absolute_lag_seconds": float(np.abs(consensus_lag).mean()),
        "lag_dimension_disagreement_seconds": float(np.abs(lag_v - lag_a).mean()),
        "amplitude_mae_after_consensus_alignment": float(
            np.abs(amplitude_residual).mean()
        ),
        "boundary_absolute_lag_seconds": float(
            np.abs(consensus_lag[boundary_mask]).mean()
        ),
        "stable_absolute_lag_seconds": float(
            np.abs(consensus_lag[stable_mask]).mean()
        ),
        "boundary_amplitude_deviation": float(amplitude_magnitude[boundary_mask].mean()),
        "stable_amplitude_deviation": float(amplitude_magnitude[stable_mask].mean()),
    }
    return TrialDecomposition(
        mapping_from_valence=mapping_v,
        mapping_from_arousal=mapping_a,
        lag_from_valence=lag_v,
        lag_from_arousal=lag_a,
        consensus_mapping=consensus_mapping,
        consensus_lag=consensus_lag,
        amplitude_residual=amplitude_residual.astype(np.float32),
        boundary_mask=boundary_mask,
        metrics=metrics,
    )


def circular_shift_null(
    target: np.ndarray,
    template: np.ndarray,
    config: WarpConfig,
    rng: np.random.Generator,
) -> float:
    """Cross-dimensional gain after breaking shared timing by circular shift."""

    length = len(target)
    minimum = min(config.band_seconds + 1, max(length // 4, 1))
    candidates = np.arange(minimum, max(length - minimum, minimum + 1))
    shift_v = int(rng.choice(candidates)) if len(candidates) else max(1, length // 2)
    shift_a = int(rng.choice(candidates)) if len(candidates) else max(1, length // 2)
    shifted_v = np.roll(target[:, 0], shift_v)
    shifted_a = np.roll(target[:, 1], shift_a)
    mapping_v = constrained_dtw_mapping(shifted_v, template[:, 0], config)
    mapping_a = constrained_dtw_mapping(shifted_a, template[:, 1], config)
    baseline_v = np.abs(target[:, 0] - template[:, 0]).mean()
    baseline_a = np.abs(target[:, 1] - template[:, 1]).mean()
    aligned_v = np.abs(target[:, 0] - template[mapping_a, 0]).mean()
    aligned_a = np.abs(target[:, 1] - template[mapping_v, 1]).mean()
    return float(((baseline_v - aligned_v) + (baseline_a - aligned_a)) / 2.0)
