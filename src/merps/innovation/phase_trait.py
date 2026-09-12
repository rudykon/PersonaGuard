"""Psychometric analysis of participant temporal-deviation traits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class VarianceComponents:
    grand_mean: float
    participant: float
    video: float
    residual: float
    participant_mean_square: float
    video_mean_square: float
    residual_mean_square: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def estimate_variance_components(matrix: np.ndarray) -> VarianceComponents:
    """Estimate a balanced participant x video random-effects G-study.

    With one trial summary per cell, the residual component contains the
    participant-by-video interaction plus trial-level measurement error.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("matrix must contain at least two participants and videos")
    if not np.isfinite(values).all():
        raise ValueError("matrix contains non-finite values")
    participants, videos = values.shape
    grand = float(values.mean())
    participant_means = values.mean(axis=1)
    video_means = values.mean(axis=0)
    residual = (
        values
        - participant_means[:, None]
        - video_means[None, :]
        + grand
    )
    participant_ms = float(
        videos * np.square(participant_means - grand).sum() / (participants - 1)
    )
    video_ms = float(
        participants * np.square(video_means - grand).sum() / (videos - 1)
    )
    residual_ms = float(
        np.square(residual).sum() / ((participants - 1) * (videos - 1))
    )
    participant_variance = max(
        (participant_ms - residual_ms) / videos, 0.0
    )
    video_variance = max((video_ms - residual_ms) / participants, 0.0)
    return VarianceComponents(
        grand_mean=grand,
        participant=participant_variance,
        video=video_variance,
        residual=max(residual_ms, 0.0),
        participant_mean_square=participant_ms,
        video_mean_square=video_ms,
        residual_mean_square=residual_ms,
    )


def reliability_for_videos(
    components: VarianceComponents, video_count: int
) -> dict[str, float]:
    """Return relative G and absolute dependability for k random videos."""

    count = int(video_count)
    if count < 1:
        raise ValueError("video_count must be positive")
    participant = float(components.participant)
    relative_error = float(components.residual) / count
    absolute_error = (
        float(components.video) + float(components.residual)
    ) / count
    relative_denominator = participant + relative_error
    absolute_denominator = participant + absolute_error
    return {
        "videos": count,
        "relative_g": (
            participant / relative_denominator
            if relative_denominator > 0
            else 0.0
        ),
        "absolute_phi": (
            participant / absolute_denominator
            if absolute_denominator > 0
            else 0.0
        ),
    }


def reliability_curve(
    components: VarianceComponents, maximum_videos: int
) -> list[dict[str, float]]:
    return [
        reliability_for_videos(components, count)
        for count in range(1, int(maximum_videos) + 1)
    ]


def videos_required(
    components: VarianceComponents,
    threshold: float,
    *,
    coefficient: str = "relative_g",
    maximum_videos: int = 100,
) -> int | None:
    if coefficient not in {"relative_g", "absolute_phi"}:
        raise ValueError("unknown reliability coefficient")
    for count in range(1, int(maximum_videos) + 1):
        if reliability_for_videos(components, count)[coefficient] >= threshold:
            return count
    return None


def corrected_item_total_correlations(matrix: np.ndarray) -> np.ndarray:
    """Correlation of each video residual with the other-video trait score."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("matrix must contain at least three videos")
    centered = values - values.mean(axis=0, keepdims=True)
    scores = np.empty(values.shape[1], dtype=np.float64)
    for video in range(values.shape[1]):
        other = np.delete(centered, video, axis=1).mean(axis=1)
        item = centered[:, video]
        if item.std() < 1e-12 or other.std() < 1e-12:
            scores[video] = 0.0
        else:
            scores[video] = float(np.corrcoef(item, other)[0, 1])
    return scores


def participant_label_permutation_test(
    matrix: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float | list[float]]:
    """Destroy cross-video participant identity while preserving each video."""

    values = np.asarray(matrix, dtype=np.float64)
    observed_components = estimate_variance_components(values)
    observed = reliability_for_videos(
        observed_components, values.shape[1]
    )["relative_g"]
    rng = np.random.default_rng(seed)
    null = np.empty(int(repeats), dtype=np.float64)
    permuted = np.empty_like(values)
    for repeat in range(int(repeats)):
        for video in range(values.shape[1]):
            permuted[:, video] = values[rng.permutation(values.shape[0]), video]
        components = estimate_variance_components(permuted)
        null[repeat] = reliability_for_videos(
            components, values.shape[1]
        )["relative_g"]
    return {
        "observed_relative_g": float(observed),
        "null_mean": float(null.mean()),
        "null_ci95": [
            float(np.quantile(null, 0.025)),
            float(np.quantile(null, 0.975)),
        ],
        "p_one_sided": float(
            (1 + np.sum(null >= observed)) / (int(repeats) + 1)
        ),
    }


def crossed_bootstrap(
    matrix: np.ndarray,
    *,
    repeats: int,
    seed: int,
    maximum_videos: int,
    crossing_thresholds: tuple[float, ...] = (0.60, 0.70, 0.80),
    crossing_search_maximum: int = 100,
) -> dict[str, object]:
    """Resample participants and videos for uncertainty over the G-study."""

    values = np.asarray(matrix, dtype=np.float64)
    rng = np.random.default_rng(seed)
    participant_variance = np.empty(int(repeats), dtype=np.float64)
    video_variance = np.empty(int(repeats), dtype=np.float64)
    residual_variance = np.empty(int(repeats), dtype=np.float64)
    relative = np.empty((int(repeats), int(maximum_videos)), dtype=np.float64)
    absolute = np.empty_like(relative)
    crossing_relative = {
        float(threshold): np.full(int(repeats), np.nan, dtype=np.float64)
        for threshold in crossing_thresholds
    }
    crossing_absolute = {
        float(threshold): np.full(int(repeats), np.nan, dtype=np.float64)
        for threshold in crossing_thresholds
    }
    for repeat in range(int(repeats)):
        participant_draw = rng.integers(0, values.shape[0], values.shape[0])
        video_draw = rng.integers(0, values.shape[1], values.shape[1])
        sample = values[np.ix_(participant_draw, video_draw)]
        components = estimate_variance_components(sample)
        participant_variance[repeat] = components.participant
        video_variance[repeat] = components.video
        residual_variance[repeat] = components.residual
        for count in range(1, int(maximum_videos) + 1):
            reliability = reliability_for_videos(components, count)
            relative[repeat, count - 1] = reliability["relative_g"]
            absolute[repeat, count - 1] = reliability["absolute_phi"]
        for threshold in crossing_thresholds:
            relative_count = videos_required(
                components,
                float(threshold),
                coefficient="relative_g",
                maximum_videos=crossing_search_maximum,
            )
            absolute_count = videos_required(
                components,
                float(threshold),
                coefficient="absolute_phi",
                maximum_videos=crossing_search_maximum,
            )
            if relative_count is not None:
                crossing_relative[float(threshold)][repeat] = relative_count
            if absolute_count is not None:
                crossing_absolute[float(threshold)][repeat] = absolute_count

    def interval(values: np.ndarray) -> list[float]:
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    def crossing_summary(values: np.ndarray) -> dict[str, object]:
        finite = values[np.isfinite(values)]
        return {
            "median_videos": float(np.median(finite)) if len(finite) else None,
            "ci95_videos": interval(finite) if len(finite) else [None, None],
            "proportion_not_reached_by_observed_15_videos": float(
                np.mean(~np.isfinite(values) | (values > 15))
            ),
            "proportion_not_reached_by_search_maximum": float(
                np.mean(~np.isfinite(values))
            ),
            "search_maximum_videos": int(crossing_search_maximum),
        }

    return {
        "variance_ci95": {
            "participant": interval(participant_variance),
            "video": interval(video_variance),
            "residual": interval(residual_variance),
        },
        "reliability_ci95": [
            {
                "videos": count,
                "relative_g": interval(relative[:, count - 1]),
                "absolute_phi": interval(absolute[:, count - 1]),
            }
            for count in range(1, int(maximum_videos) + 1)
        ],
        "threshold_crossing_uncertainty": {
            str(threshold): {
                "relative_g": crossing_summary(crossing_relative[float(threshold)]),
                "absolute_phi": crossing_summary(crossing_absolute[float(threshold)]),
            }
            for threshold in crossing_thresholds
        },
    }
