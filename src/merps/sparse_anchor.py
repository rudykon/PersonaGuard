"""Leakage-controlled sparse-anchor reconstruction of affect trajectories.

The zero-interaction MER-PS baseline estimates a population trajectory for a
known video.  This module adds an explicitly different deployment mode: after
the video, a participant supplies one SAM valence/arousal rating.  The rating
is mapped to the continuous-label scale and may make only a bounded correction
to the population trajectory.  The bound prevents a single coarse rating from
overwriting the time-resolved population shape.

All prior and correction choices can be selected inside participant-disjoint
inner folds.  No held-out continuous trajectory is used to construct a prior,
select a hyperparameter, or scale an anchor.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from merps.innovation.data import (
    InnovationIndex,
    grouped_subject_folds,
    outer_subject_folds,
)


SAM_MIN = 1.0
SAM_MAX = 9.0
LABEL_MIN = 1.0
LABEL_MAX = 255.0
PRIOR_METHODS = ("current", "pooled_median", "triangular")


@dataclass(frozen=True)
class PriorSpec:
    """Population-trajectory estimator selected inside training participants."""

    method: str
    radius: int

    def __post_init__(self) -> None:
        if self.method not in PRIOR_METHODS:
            raise ValueError(f"Unknown prior method: {self.method}")
        if int(self.radius) < 0:
            raise ValueError("Prior radius must be non-negative")

    @property
    def name(self) -> str:
        return f"{self.method}_r{int(self.radius)}"


@dataclass(frozen=True)
class AnchorSpec:
    """Conservative center calibration and shape shrinkage from a sparse rating."""

    center_weight: float
    shape_weight: float
    cap: float

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.center_weight) <= 1.0:
            raise ValueError("Anchor center weight must be in [0, 1]")
        if not 0.0 <= float(self.shape_weight) <= 1.0:
            raise ValueError("Anchor shape weight must be in [0, 1]")
        if float(self.center_weight) == 0.0 and float(self.shape_weight) == 0.0:
            if float(self.cap) != 0.0:
                raise ValueError("The disabled anchor must have cap=0")
        elif float(self.cap) <= 0.0:
            raise ValueError("An enabled anchor needs a positive cap")

    @property
    def name(self) -> str:
        if self.center_weight == 0.0 and self.shape_weight == 0.0:
            return "disabled"
        return (
            f"center{self.center_weight:g}_shape{self.shape_weight:g}"
            f"_cap{self.cap:g}"
        )


@dataclass(frozen=True)
class ConditionalSpec:
    """Anchor-conditioned retrieval of source trajectories for one video."""

    neighbors: int
    mix: float
    mode: str = "axis"
    bandwidth: float = float("inf")

    def __post_init__(self) -> None:
        if int(self.neighbors) < 0:
            raise ValueError("Conditional neighbors must be non-negative")
        if not 0.0 <= float(self.mix) <= 1.0:
            raise ValueError("Conditional mix must be in [0, 1]")
        if self.mode not in {"axis", "joint"}:
            raise ValueError("Conditional mode must be 'axis' or 'joint'")
        if (int(self.neighbors) == 0) != (float(self.mix) == 0.0):
            raise ValueError("The disabled conditional prior must be (0, 0)")
        if int(self.neighbors) == 0 and self.mode != "axis":
            raise ValueError("The disabled conditional prior uses axis mode")

        if not (float(self.bandwidth) > 0.0):
            raise ValueError("Conditional bandwidth must be positive")
    @property
    def name(self) -> str:
        if self.neighbors == 0:
            return "disabled"
        base = (
            f"{self.mode}_nearest{int(self.neighbors)}"
            f"_mix{float(self.mix):g}"
        )
        if np.isinf(float(self.bandwidth)):
            return base
        return f"{base}_bw{float(self.bandwidth):g}"


def gaussian_neighbor_weights(distance: np.ndarray, bandwidth: float) -> np.ndarray:
    """Numerically stable Gaussian weights; infinity reproduces uniform KNN."""
    values = np.asarray(distance, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("distance must be a non-empty finite vector")
    bandwidth = float(bandwidth)
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    if np.isinf(bandwidth):
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    log_weight = -0.5 * np.square(values / bandwidth)
    log_weight -= log_weight.max()
    weight = np.exp(log_weight)
    return weight / weight.sum()


@dataclass(frozen=True)
class FunctionalSpec:
    """Shrinkage map from one SAM anchor to a complete residual function."""

    ridge: float
    slope_radius: int
    mix: float
    cap: float

    def __post_init__(self) -> None:
        if float(self.ridge) < 0.0:
            raise ValueError("Functional ridge must be non-negative")
        if int(self.slope_radius) < 0:
            raise ValueError("Functional slope radius must be non-negative")
        if not 0.0 <= float(self.mix) <= 1.0:
            raise ValueError("Functional mix must be in [0, 1]")
        if float(self.mix) > 0.0 and float(self.cap) <= 0.0:
            raise ValueError("An enabled functional correction needs a positive cap")

    @property
    def name(self) -> str:
        return (
            f"functional_ridge{float(self.ridge):g}"
            f"_radius{int(self.slope_radius)}_mix{float(self.mix):g}"
            f"_cap{float(self.cap):g}"
        )


@dataclass(frozen=True)
class TrialTable:
    """Participant-by-video trial rows and their two sparse SAM anchors."""

    subjects: np.ndarray
    videos: np.ndarray
    rows: tuple[np.ndarray, ...]
    anchors: np.ndarray
    row_to_trial: np.ndarray

    def indices_for_subjects(self, subjects: Sequence[int]) -> np.ndarray:
        return np.flatnonzero(
            np.isin(self.subjects, np.asarray(subjects, dtype=np.int16))
        )


@dataclass(frozen=True)
class NestedSparseAnchorResult:
    predictions: dict[str, np.ndarray]
    selections: tuple[dict[str, object], ...]


DEFAULT_PRIOR_CANDIDATES = tuple(
    [PriorSpec("current", radius) for radius in (0, 1, 3, 5, 7, 10, 15)]
    + [
        PriorSpec("pooled_median", radius)
        for radius in (1, 3, 5, 7, 10, 15)
    ]
    + [PriorSpec("triangular", radius) for radius in (1, 3, 5, 7, 10, 15)]
)

# These are regularization constraints.  At least 80% of the time-resolved
# population shape is retained before clipping, the sparse anchor cannot move
# the population center by more than 30% of their discrepancy, and any
# one-second correction is at most 20/255 label units.
DEFAULT_ANCHOR_CANDIDATES = tuple(
    [AnchorSpec(0.0, 0.0, 0.0)]
    + [
        AnchorSpec(center_weight, shape_weight, cap)
        for center_weight in (0.15, 0.20, 0.25, 0.30)
        for shape_weight in (0.0, 0.10, 0.20)
        for cap in (10.0, 15.0, 20.0)
    ]
)

# A deliberately small, ordered search space.  The disabled model is first so
# exact ties prefer the simpler unconditional trajectory.  Each enabled model
# retrieves source trajectories from the same video using only the target
# trial's released post-trial SAM ratings; candidates compare axis-specific
# retrieval against one shared two-axis neighborhood.
DEFAULT_CONDITIONAL_CANDIDATES = (
    ConditionalSpec(0, 0.0),
    ConditionalSpec(3, 0.50),
    ConditionalSpec(5, 0.50),
    ConditionalSpec(7, 0.75),
    ConditionalSpec(10, 0.75),
    ConditionalSpec(10, 1.00),
    ConditionalSpec(15, 1.00),
    ConditionalSpec(7, 0.75, "joint"),
    ConditionalSpec(10, 0.75, "joint"),
    ConditionalSpec(10, 1.00, "joint"),
    ConditionalSpec(15, 1.00, "joint"),
    # Distance-weighted REFED variants; infinity above preserves uniform KNN.
    ConditionalSpec(10, 0.50, "axis", 16.0),
    ConditionalSpec(15, 0.50, "axis", 16.0),
    ConditionalSpec(15, 0.75, "axis", 16.0),
    ConditionalSpec(10, 0.75, "axis", 32.0),
    ConditionalSpec(15, 0.75, "axis", 32.0),
    ConditionalSpec(15, 0.75, "joint", 32.0),
)

# A compact development family motivated by the REFED setting: one global
# post-trial SAM pair supervises a smooth, video-specific residual function.
# It deliberately complements local KNN retrieval rather than replacing it.
DEFAULT_FUNCTIONAL_CANDIDATES = (
    FunctionalSpec(0.0, 10, 0.75, 40.0),
    FunctionalSpec(1000.0, 10, 0.75, 40.0),
    FunctionalSpec(0.0, 3, 0.75, 40.0),
    FunctionalSpec(0.0, 10, 0.50, 20.0),
    FunctionalSpec(10000.0, 10, 0.50, 20.0),
)
DEFAULT_ENSEMBLE_WEIGHTS = (1.0, 0.75, 0.50, 0.25, 0.0)


def sam_to_continuous(rating: np.ndarray | Sequence[float]) -> np.ndarray:
    """Map the released 1--9 SAM scale to the 1--255 continuous scale."""

    values = np.asarray(rating, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("SAM ratings must be finite")
    if np.any(values < SAM_MIN) or np.any(values > SAM_MAX):
        raise ValueError("SAM ratings must be inside [1, 9]")
    scaled = LABEL_MIN + (values - SAM_MIN) * (
        (LABEL_MAX - LABEL_MIN) / (SAM_MAX - SAM_MIN)
    )
    return scaled.astype(np.float32)


def _read_sam_lookup(path: str | Path) -> dict[tuple[str, int], np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    lookup: dict[tuple[str, int], np.ndarray] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject = str(row.get("sub_id", "")).strip()
            if not subject:
                raise ValueError("SAM_score.csv needs a sub_id column")
            for video in range(1, 16):
                keys = (
                    f"Video_{video}_Valence",
                    f"Video_{video}_Arousal",
                )
                if any(key not in row for key in keys):
                    raise ValueError(f"SAM_score.csv is missing {keys}")
                raw = np.asarray([float(row[key]) for key in keys])
                lookup[(subject, video)] = sam_to_continuous(raw)
    if not lookup:
        raise ValueError(f"No SAM ratings found in {path}")
    return lookup


def build_trial_table(
    index: InnovationIndex,
    sam_path: str | Path,
) -> TrialTable:
    """Create a canonical trial table and validate one SAM anchor per trial."""

    lookup = _read_sam_lookup(sam_path)
    subjects: list[int] = []
    videos: list[int] = []
    rows: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    row_to_trial = np.full(len(index.targets), -1, dtype=np.int32)

    for subject in sorted(set(int(value) for value in index.subject_numbers)):
        subject_name = f"test_{subject}"
        subject_videos = sorted(
            set(int(value) for value in index.videos[index.subject_numbers == subject])
        )
        for video in subject_videos:
            selected = np.flatnonzero(
                (index.subject_numbers == subject) & (index.videos == video)
            )
            selected = selected[np.argsort(index.timestamps[selected])]
            if not len(selected):
                continue
            if (subject_name, video) not in lookup:
                raise ValueError(
                    f"Missing SAM anchor for {subject_name}, video {video}"
                )
            trial = len(rows)
            row_to_trial[selected] = trial
            subjects.append(subject)
            videos.append(video)
            rows.append(selected.astype(np.int64))
            anchors.append(lookup[(subject_name, video)])

    if np.any(row_to_trial < 0):
        raise RuntimeError("At least one labelled second is not assigned to a trial")
    return TrialTable(
        subjects=np.asarray(subjects, dtype=np.int16),
        videos=np.asarray(videos, dtype=np.int16),
        rows=tuple(rows),
        anchors=np.asarray(anchors, dtype=np.float32),
        row_to_trial=row_to_trial,
    )


def _video_cube(
    index: InnovationIndex,
    source_subjects: Sequence[int],
    video: int,
) -> np.ndarray:
    trajectories: list[np.ndarray] = []
    for subject in source_subjects:
        rows = np.flatnonzero(
            (index.subject_numbers == int(subject)) & (index.videos == int(video))
        )
        rows = rows[np.argsort(index.timestamps[rows])]
        if len(rows):
            trajectories.append(np.asarray(index.targets[rows], dtype=np.float64))
    if not trajectories:
        raise ValueError(f"No source trajectory for video {video}")
    lengths = {len(values) for values in trajectories}
    if len(lengths) != 1:
        raise ValueError(f"Source trajectories differ in length for video {video}")
    return np.stack(trajectories, axis=0)


def estimate_population_trajectory(
    trajectories: np.ndarray,
    spec: PriorSpec,
) -> np.ndarray:
    """Estimate one robust population trajectory from source participants."""

    values = np.asarray(trajectories, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("Expected trajectories with shape [subject, time, 2]")
    pointwise = np.median(values, axis=0)
    radius = int(spec.radius)
    if radius == 0:
        return np.clip(pointwise, LABEL_MIN, LABEL_MAX).astype(np.float32)

    output = np.empty_like(pointwise)
    for timestamp in range(pointwise.shape[0]):
        start = max(0, timestamp - radius)
        end = min(pointwise.shape[0], timestamp + radius + 1)
        if spec.method == "current":
            output[timestamp] = pointwise[start:end].mean(axis=0)
        elif spec.method == "pooled_median":
            output[timestamp] = np.median(
                values[:, start:end, :].reshape(-1, 2), axis=0
            )
        elif spec.method == "triangular":
            distance = np.abs(np.arange(start, end) - timestamp)
            weight = (radius + 1 - distance).astype(np.float64)
            output[timestamp] = np.average(
                pointwise[start:end], axis=0, weights=weight
            )
        else:  # guarded by PriorSpec, retained for defensive programming
            raise ValueError(f"Unknown prior method: {spec.method}")
    return np.clip(output, LABEL_MIN, LABEL_MAX).astype(np.float32)

def smooth_pointwise_trajectory(
    pointwise: np.ndarray, spec: PriorSpec
) -> np.ndarray:
    """Apply the selected temporal smoother to a pre-aggregated trajectory."""
    values = np.asarray(pointwise, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("pointwise must have shape [time, 2]")
    radius = int(spec.radius)
    if radius == 0:
        return np.clip(values, LABEL_MIN, LABEL_MAX).astype(np.float32)
    output = np.empty_like(values)
    for timestamp in range(len(values)):
        start = max(0, timestamp - radius)
        end = min(len(values), timestamp + radius + 1)
        if spec.method == "current":
            output[timestamp] = values[start:end].mean(axis=0)
        elif spec.method == "pooled_median":
            output[timestamp] = np.median(values[start:end], axis=0)
        elif spec.method == "triangular":
            distance = np.abs(np.arange(start, end) - timestamp)
            weight = (radius + 1 - distance).astype(np.float64)
            output[timestamp] = np.average(values[start:end], axis=0, weights=weight)
        else:
            raise ValueError(f"Unknown prior method: {spec.method}")
    return np.clip(output, LABEL_MIN, LABEL_MAX).astype(np.float32)



def predict_population_prior(
    index: InnovationIndex,
    source_subjects: Sequence[int],
    prediction_rows: np.ndarray,
    spec: PriorSpec,
) -> np.ndarray:
    """Predict selected rows without using any target-participant label."""

    prediction_rows = np.asarray(prediction_rows, dtype=np.int64)
    sources = np.asarray(source_subjects, dtype=np.int16)
    if not len(sources):
        raise ValueError("Population prior needs source participants")
    if np.intersect1d(sources, index.subject_numbers[prediction_rows]).size:
        raise ValueError("Source and prediction participants must be disjoint")

    lookup: dict[int, np.ndarray] = {}
    for video in sorted(set(int(value) for value in index.videos[prediction_rows])):
        lookup[video] = estimate_population_trajectory(
            _video_cube(index, sources, video), spec
        )

    prediction = np.empty((len(prediction_rows), 2), dtype=np.float32)
    for local, row in enumerate(prediction_rows):
        video = int(index.videos[row])
        timestamp = int(index.timestamps[row])
        trajectory = lookup[video]
        prediction[local] = trajectory[
            min(max(timestamp, 0), len(trajectory) - 1)
        ]
    return prediction


def predict_conditional_population_prior(
    index: InnovationIndex,
    table: TrialTable,
    source_subjects: Sequence[int],
    prediction_rows: np.ndarray,
    population_spec: PriorSpec,
    conditional_spec: ConditionalSpec,
    base_prior: np.ndarray,
) -> np.ndarray:
    """Retrieve same-video source trajectories near the target SAM rating.

    Axis mode ranks valence/arousal trajectories by their matching SAM axis;
    joint mode uses one Euclidean neighborhood in the two SAM axes.  The
    retrieved robust trajectory is shrunk toward ``base_prior`` by ``mix``.
    Only source-participant continuous labels are used.
    """

    prediction_rows = np.asarray(prediction_rows, dtype=np.int64)
    base_values = np.asarray(base_prior, dtype=np.float32)
    if base_values.shape != (len(prediction_rows), 2):
        raise ValueError("base_prior must align with prediction_rows")
    if conditional_spec.neighbors == 0:
        return base_values.copy()

    sources = np.asarray(source_subjects, dtype=np.int16)
    if not len(sources):
        raise ValueError("Conditional prior needs source participants")
    if np.intersect1d(sources, index.subject_numbers[prediction_rows]).size:
        raise ValueError("Source and prediction participants must be disjoint")

    local = np.full(len(index.targets), -1, dtype=np.int64)
    local[prediction_rows] = np.arange(len(prediction_rows))
    source_trial_mask = np.isin(table.subjects, sources)
    output = np.full_like(base_values, np.nan, dtype=np.float32)
    prediction_trials = sorted(
        set(int(value) for value in table.row_to_trial[prediction_rows])
    )
    for target_trial in prediction_trials:
        target_rows = table.rows[target_trial]
        positions = local[target_rows]
        if np.any(positions < 0):
            raise ValueError("prediction_rows must contain complete trials")
        video = int(table.videos[target_trial])
        source_trials = np.flatnonzero(
            source_trial_mask & (table.videos == video)
        )
        if not len(source_trials):
            raise ValueError(f"No source SAM trials for video {video}")

        joint_selected: np.ndarray | None = None
        joint_distance: np.ndarray | None = None
        if conditional_spec.mode == "joint":
            # Both axes share the same released scale and affine transform.
            distance = np.linalg.norm(
                table.anchors[source_trials]
                - table.anchors[target_trial][None, :],
                axis=1,
            )
            order = np.lexsort((table.subjects[source_trials], distance))
            selected_positions = order[
                : min(int(conditional_spec.neighbors), len(order))
            ]
            joint_selected = source_trials[selected_positions]
            joint_distance = distance[selected_positions]

        for dimension in range(2):
            if joint_selected is None:
                distance = np.abs(
                    table.anchors[source_trials, dimension]
                    - table.anchors[target_trial, dimension]
                )
                order = np.lexsort((table.subjects[source_trials], distance))
                selected_positions = order[
                    : min(int(conditional_spec.neighbors), len(order))
                ]
                selected_trials = source_trials[selected_positions]
                selected_distance = distance[selected_positions]
            else:
                selected_trials = joint_selected
                assert joint_distance is not None
                selected_distance = joint_distance
            trajectories = np.stack(
                [index.targets[table.rows[int(trial)]] for trial in selected_trials]
            )
            if np.isinf(float(conditional_spec.bandwidth)):
                retrieved = estimate_population_trajectory(
                    trajectories, population_spec
                )[:, dimension]
            else:
                weights = gaussian_neighbor_weights(
                    selected_distance, conditional_spec.bandwidth
                )
                weighted_pointwise = np.average(
                    trajectories, axis=0, weights=weights
                )
                retrieved = smooth_pointwise_trajectory(
                    weighted_pointwise, population_spec
                )[:, dimension]
            output[positions, dimension] = (
                (1.0 - float(conditional_spec.mix))
                * base_values[positions, dimension]
                + float(conditional_spec.mix) * retrieved
            )

    if not np.isfinite(output).all():
        raise RuntimeError("Conditional population prediction is incomplete")
    return np.clip(output, LABEL_MIN, LABEL_MAX).astype(np.float32)


def predict_functional_anchor_prior(
    index: InnovationIndex,
    table: TrialTable,
    source_subjects: Sequence[int],
    prediction_rows: np.ndarray,
    population_spec: PriorSpec,
    functional_spec: FunctionalSpec,
    base_prior: np.ndarray,
) -> np.ndarray:
    """Map a global SAM displacement to a smooth time-varying correction.

    For each video and axis, source participants estimate a ridge-shrunk slope
    function from their post-trial SAM displacement to their dense residual
    around the population trajectory.  The slope is temporally smoothed, then
    applied to the target trial's SAM displacement with a hard correction cap.
    Target-participant continuous labels are never consulted.
    """

    prediction_rows = np.asarray(prediction_rows, dtype=np.int64)
    base_values = np.asarray(base_prior, dtype=np.float32)
    if base_values.shape != (len(prediction_rows), 2):
        raise ValueError("base_prior must align with prediction_rows")
    sources = np.asarray(source_subjects, dtype=np.int16)
    if not len(sources):
        raise ValueError("Functional prior needs source participants")
    if np.intersect1d(sources, index.subject_numbers[prediction_rows]).size:
        raise ValueError("Source and prediction participants must be disjoint")

    local = np.full(len(index.targets), -1, dtype=np.int64)
    local[prediction_rows] = np.arange(len(prediction_rows))
    source_trial_mask = np.isin(table.subjects, sources)
    output = np.full_like(base_values, np.nan, dtype=np.float32)
    prediction_trials = sorted(
        set(int(value) for value in table.row_to_trial[prediction_rows])
    )
    model_by_video: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for target_trial in prediction_trials:
        video = int(table.videos[target_trial])
        if video not in model_by_video:
            source_trials = np.flatnonzero(
                source_trial_mask & (table.videos == video)
            )
            if not len(source_trials):
                raise ValueError(f"No functional source trials for video {video}")
            trajectories = np.stack(
                [index.targets[table.rows[int(trial)]] for trial in source_trials]
            ).astype(np.float64)
            anchors = table.anchors[source_trials].astype(np.float64)
            population = estimate_population_trajectory(
                trajectories, population_spec
            ).astype(np.float64)
            anchor_center = np.median(anchors, axis=0)
            centered_anchor = anchors - anchor_center[None, :]
            slope = np.empty_like(population)
            for dimension in range(2):
                denominator = (
                    np.square(centered_anchor[:, dimension]).sum()
                    + float(functional_spec.ridge)
                )
                numerator = (
                    centered_anchor[:, dimension, None]
                    * (
                        trajectories[:, :, dimension]
                        - population[None, :, dimension]
                    )
                ).sum(axis=0)
                slope[:, dimension] = numerator / max(denominator, 1e-12)
            radius = int(functional_spec.slope_radius)
            if radius > 0:
                smoothed = np.empty_like(slope)
                for timestamp in range(len(slope)):
                    smoothed[timestamp] = slope[
                        max(0, timestamp - radius) : min(
                            len(slope), timestamp + radius + 1
                        )
                    ].mean(axis=0)
                slope = smoothed
            model_by_video[video] = (anchor_center, slope)

        target_rows = table.rows[target_trial]
        positions = local[target_rows]
        if np.any(positions < 0):
            raise ValueError("prediction_rows must contain complete trials")
        anchor_center, slope = model_by_video[video]
        raw = slope * (
            table.anchors[target_trial].astype(np.float64) - anchor_center
        )[None, :]
        correction = np.clip(
            float(functional_spec.mix) * raw,
            -float(functional_spec.cap),
            float(functional_spec.cap),
        )
        output[positions] = np.clip(
            base_values[positions] + correction, LABEL_MIN, LABEL_MAX
        )
    if not np.isfinite(output).all():
        raise RuntimeError("Functional population prediction is incomplete")
    return output.astype(np.float32)


def apply_bounded_anchor(
    prior: np.ndarray,
    anchor: np.ndarray | Sequence[float],
    spec: AnchorSpec,
) -> np.ndarray:
    """Calibrate trajectory center and shrink shape without exceeding ``cap``.

    Let ``m`` be the trial median of the population trajectory and ``a`` the
    sparse anchor.  The correction is

    ``center_weight * (a - m) - shape_weight * (prior_t - m)``.

    Using separate terms avoids forcing the strength of center calibration to
    equal the amount of temporal-amplitude contraction.
    """

    prior_values = np.asarray(prior, dtype=np.float32)
    anchor_values = np.asarray(anchor, dtype=np.float32)
    if prior_values.ndim != 2 or prior_values.shape[1] != 2:
        raise ValueError("prior must have shape [time, 2]")
    if anchor_values.shape != (2,):
        raise ValueError("anchor must have shape [2]")
    if spec.center_weight == 0.0 and spec.shape_weight == 0.0:
        return prior_values.copy()
    center = np.median(prior_values, axis=0)
    correction = np.clip(
        float(spec.center_weight) * (anchor_values[None, :] - center[None, :])
        - float(spec.shape_weight) * (prior_values - center[None, :]),
        -float(spec.cap),
        float(spec.cap),
    )
    return np.clip(
        prior_values + correction, LABEL_MIN, LABEL_MAX
    ).astype(np.float32)


def _validation_rows(
    index: InnovationIndex, subjects: Sequence[int]
) -> np.ndarray:
    return index.indices_for_subjects(subjects)


def _trial_error_vector(
    index: InnovationIndex,
    table: TrialTable,
    validation_subjects: Sequence[int],
    validation_rows: np.ndarray,
    prior: np.ndarray,
    anchor_spec: AnchorSpec,
) -> np.ndarray:
    local = np.full(len(index.targets), -1, dtype=np.int64)
    local[validation_rows] = np.arange(len(validation_rows))
    errors: list[float] = []
    for trial in table.indices_for_subjects(validation_subjects):
        rows = table.rows[int(trial)]
        positions = local[rows]
        if np.any(positions < 0):
            raise RuntimeError("Validation prior is missing trial rows")
        prediction = apply_bounded_anchor(
            prior[positions], table.anchors[int(trial)], anchor_spec
        )
        errors.append(float(np.abs(prediction - index.targets[rows]).mean()))
    return np.asarray(errors, dtype=np.float64)


def _apply_anchor_to_complete_trials(
    index: InnovationIndex,
    table: TrialTable,
    validation_subjects: Sequence[int],
    validation_rows: np.ndarray,
    prior: np.ndarray,
    anchor_spec: AnchorSpec,
) -> np.ndarray:
    output = np.empty_like(prior)
    local = np.full(len(index.targets), -1, dtype=np.int64)
    local[validation_rows] = np.arange(len(validation_rows))
    for trial in table.indices_for_subjects(validation_subjects):
        rows = table.rows[int(trial)]
        positions = local[rows]
        if np.any(positions < 0):
            raise RuntimeError("Validation prior is missing trial rows")
        output[positions] = apply_bounded_anchor(
            prior[positions], table.anchors[int(trial)], anchor_spec
        )
    return output


def nested_sparse_anchor_oof(
    index: InnovationIndex,
    table: TrialTable,
    *,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    inner_folds: int = 4,
    prior_candidates: Sequence[PriorSpec] = DEFAULT_PRIOR_CANDIDATES,
    anchor_candidates: Sequence[AnchorSpec] = DEFAULT_ANCHOR_CANDIDATES,
    conditional_candidates: Sequence[
        ConditionalSpec
    ] = DEFAULT_CONDITIONAL_CANDIDATES,
    functional_candidates: Sequence[
        FunctionalSpec
    ] = DEFAULT_FUNCTIONAL_CANDIDATES,
    ensemble_weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
    canonical_prior: PriorSpec = PriorSpec("current", 3),
) -> NestedSparseAnchorResult:
    """Run participant-disjoint nested selection and return complete OOF traces.

    Prior-only selection and joint prior-plus-anchor selection are independent.
    ``anchor_reference_prior`` uses the exact prior selected jointly with the
    sparse anchor.  A second inner stage then selects SAM-conditioned
    same-video retrieval plus a bounded correction while holding that prior
    fixed; an explicit disabled retrieval remains available.  A final inner
    stage combines local trajectory retrieval with a smooth functional SAM
    regression; weight 1.0 is an exact fallback to the conditional model.
    """

    priors = tuple(prior_candidates)
    anchors = tuple(anchor_candidates)
    conditionals = tuple(conditional_candidates)
    functionals = tuple(functional_candidates)
    weights = tuple(float(value) for value in ensemble_weights)
    if not priors or not anchors or not conditionals or not functionals or not weights:
        raise ValueError("Candidate lists cannot be empty")
    if not any(
        anchor.center_weight == 0.0 and anchor.shape_weight == 0.0
        for anchor in anchors
    ):
        raise ValueError("anchor_candidates must include a zero-correction fallback")
    if not any(condition.neighbors == 0 for condition in conditionals):
        raise ValueError(
            "conditional_candidates must include an unconditional fallback"
        )
    if any(not 0.0 <= value <= 1.0 for value in weights):
        raise ValueError("ensemble_weights must be in [0, 1]")
    if 1.0 not in weights:
        raise ValueError("ensemble_weights must include conditional-only weight 1")

    outputs = {
        "canonical_prior": np.full_like(index.targets, np.nan, dtype=np.float32),
        "independent_robust_prior": np.full_like(
            index.targets, np.nan, dtype=np.float32
        ),
        "anchor_reference_prior": np.full_like(
            index.targets, np.nan, dtype=np.float32
        ),
        "bounded_sparse_anchor": np.full_like(
            index.targets, np.nan, dtype=np.float32
        ),
        "conditional_reference_prior": np.full_like(
            index.targets, np.nan, dtype=np.float32
        ),
        "conditional_sparse_anchor": np.full_like(
            index.targets, np.nan, dtype=np.float32
        ),
        "functional_sparse_anchor": np.full_like(
            index.targets, np.nan, dtype=np.float32
        ),
        "ensemble_sparse_anchor": np.full_like(
            index.targets, np.nan, dtype=np.float32
        ),
    }
    selections: list[dict[str, object]] = []
    fold_plan = list(outer_folds or outer_subject_folds())

    for outer_fold, (outer_training, outer_validation) in enumerate(fold_plan):
        inner_records: list[dict[str, object]] = []
        for inner_training, inner_validation in grouped_subject_folds(
            outer_training, inner_folds
        ):
            validation_rows = _validation_rows(index, inner_validation)
            prior_cache = {
                spec: predict_population_prior(
                    index, inner_training, validation_rows, spec
                )
                for spec in priors
            }
            inner_records.append(
                {
                    "training_subjects": np.asarray(inner_training),
                    "validation_subjects": np.asarray(inner_validation),
                    "validation_rows": validation_rows,
                    "priors": prior_cache,
                }
            )

        prior_scores: dict[str, float] = {}
        best_prior: tuple[float, int, PriorSpec] | None = None
        disabled = AnchorSpec(0.0, 0.0, 0.0)
        for prior_position, prior_spec in enumerate(priors):
            score = float(
                np.concatenate(
                    tuple(
                        _trial_error_vector(
                            index,
                            table,
                            record["validation_subjects"],
                            record["validation_rows"],
                            record["priors"][prior_spec],
                            disabled,
                        )
                        for record in inner_records
                    )
                ).mean()
            )
            prior_scores[prior_spec.name] = score
            candidate = (score, prior_position, prior_spec)
            if best_prior is None or candidate[:2] < best_prior[:2]:
                best_prior = candidate
        assert best_prior is not None

        joint_scores: dict[str, float] = {}
        best_joint: tuple[float, int, int, PriorSpec, AnchorSpec] | None = None
        for prior_position, prior_spec in enumerate(priors):
            for anchor_position, anchor_spec in enumerate(anchors):
                score = float(
                    np.concatenate(
                        tuple(
                            _trial_error_vector(
                                index,
                                table,
                                record["validation_subjects"],
                                record["validation_rows"],
                                record["priors"][prior_spec],
                                anchor_spec,
                            )
                            for record in inner_records
                        )
                    ).mean()
                )
                key = f"{prior_spec.name}+{anchor_spec.name}"
                joint_scores[key] = score
                candidate = (
                    score,
                    prior_position,
                    anchor_position,
                    prior_spec,
                    anchor_spec,
                )
                if best_joint is None or candidate[:3] < best_joint[:3]:
                    best_joint = candidate
        assert best_joint is not None

        selected_prior = best_prior[2]
        joint_prior = best_joint[3]
        joint_anchor = best_joint[4]
        for record in inner_records:
            record["conditional_priors"] = {
                condition: predict_conditional_population_prior(
                    index,
                    table,
                    record["training_subjects"],
                    record["validation_rows"],
                    joint_prior,
                    condition,
                    record["priors"][joint_prior],
                )
                for condition in conditionals
            }

        conditional_scores: dict[str, float] = {}
        best_conditional: tuple[
            float, int, int, ConditionalSpec, AnchorSpec
        ] | None = None
        for condition_position, condition_spec in enumerate(conditionals):
            for anchor_position, anchor_spec in enumerate(anchors):
                score = float(
                    np.concatenate(
                        tuple(
                            _trial_error_vector(
                                index,
                                table,
                                record["validation_subjects"],
                                record["validation_rows"],
                                record["conditional_priors"][condition_spec],
                                anchor_spec,
                            )
                            for record in inner_records
                        )
                    ).mean()
                )
                key = f"{condition_spec.name}+{anchor_spec.name}"
                conditional_scores[key] = score
                candidate = (
                    score,
                    condition_position,
                    anchor_position,
                    condition_spec,
                    anchor_spec,
                )
                if (
                    best_conditional is None
                    or candidate[:3] < best_conditional[:3]
                ):
                    best_conditional = candidate
        assert best_conditional is not None
        selected_condition = best_conditional[3]
        selected_conditional_anchor = best_conditional[4]

        for record in inner_records:
            conditional_prior = record["conditional_priors"][selected_condition]
            record["conditional_anchored"] = _apply_anchor_to_complete_trials(
                index,
                table,
                record["validation_subjects"],
                record["validation_rows"],
                conditional_prior,
                selected_conditional_anchor,
            )
            record["functional_predictions"] = {
                functional: predict_functional_anchor_prior(
                    index,
                    table,
                    record["training_subjects"],
                    record["validation_rows"],
                    joint_prior,
                    functional,
                    record["priors"][joint_prior],
                )
                for functional in functionals
            }

        ensemble_scores: dict[str, float] = {}
        best_ensemble: tuple[
            float, int, int, FunctionalSpec, float
        ] | None = None
        for functional_position, functional_spec in enumerate(functionals):
            for weight_position, conditional_weight in enumerate(weights):
                errors = []
                for record in inner_records:
                    blended = np.clip(
                        conditional_weight * record["conditional_anchored"]
                        + (1.0 - conditional_weight)
                        * record["functional_predictions"][functional_spec],
                        LABEL_MIN,
                        LABEL_MAX,
                    )
                    errors.append(
                        _trial_error_vector(
                            index,
                            table,
                            record["validation_subjects"],
                            record["validation_rows"],
                            blended,
                            disabled,
                        )
                    )
                score = float(np.concatenate(tuple(errors)).mean())
                key = (
                    f"conditional_weight{conditional_weight:g}+"
                    f"{functional_spec.name}"
                )
                ensemble_scores[key] = score
                candidate = (
                    score,
                    functional_position,
                    weight_position,
                    functional_spec,
                    conditional_weight,
                )
                if best_ensemble is None or candidate[:3] < best_ensemble[:3]:
                    best_ensemble = candidate
        assert best_ensemble is not None
        selected_functional = best_ensemble[3]
        selected_conditional_weight = best_ensemble[4]

        validation_rows = _validation_rows(index, outer_validation)
        canonical_prediction = predict_population_prior(
            index, outer_training, validation_rows, canonical_prior
        )
        robust_prediction = predict_population_prior(
            index, outer_training, validation_rows, selected_prior
        )
        joint_prior_prediction = predict_population_prior(
            index, outer_training, validation_rows, joint_prior
        )
        conditional_prediction = predict_conditional_population_prior(
            index,
            table,
            outer_training,
            validation_rows,
            joint_prior,
            selected_condition,
            joint_prior_prediction,
        )
        functional_prediction = predict_functional_anchor_prior(
            index,
            table,
            outer_training,
            validation_rows,
            joint_prior,
            selected_functional,
            joint_prior_prediction,
        )

        anchored_prediction = _apply_anchor_to_complete_trials(
            index,
            table,
            outer_validation,
            validation_rows,
            joint_prior_prediction,
            joint_anchor,
        )
        conditional_anchored_prediction = _apply_anchor_to_complete_trials(
            index,
            table,
            outer_validation,
            validation_rows,
            conditional_prediction,
            selected_conditional_anchor,
        )
        ensemble_prediction = np.clip(
            selected_conditional_weight * conditional_anchored_prediction
            + (1.0 - selected_conditional_weight) * functional_prediction,
            LABEL_MIN,
            LABEL_MAX,
        ).astype(np.float32)

        outputs["canonical_prior"][validation_rows] = canonical_prediction
        outputs["independent_robust_prior"][validation_rows] = robust_prediction
        outputs["anchor_reference_prior"][validation_rows] = joint_prior_prediction
        outputs["bounded_sparse_anchor"][validation_rows] = anchored_prediction
        outputs["conditional_reference_prior"][
            validation_rows
        ] = conditional_prediction
        outputs["conditional_sparse_anchor"][
            validation_rows
        ] = conditional_anchored_prediction
        outputs["functional_sparse_anchor"][validation_rows] = functional_prediction
        outputs["ensemble_sparse_anchor"][validation_rows] = ensemble_prediction
        selections.append(
            {
                "outer_fold": int(outer_fold),
                "training_subjects": [int(value) for value in outer_training],
                "validation_subjects": [int(value) for value in outer_validation],
                "independent_prior": {
                    "method": selected_prior.method,
                    "radius": int(selected_prior.radius),
                    "inner_trial_macro_mae": float(best_prior[0]),
                    "scores": prior_scores,
                },
                "joint_sparse_anchor": {
                    "prior_method": joint_prior.method,
                    "prior_radius": int(joint_prior.radius),
                    "anchor_center_weight": float(joint_anchor.center_weight),
                    "anchor_shape_weight": float(joint_anchor.shape_weight),
                    "anchor_cap": float(joint_anchor.cap),
                    "inner_trial_macro_mae": float(best_joint[0]),
                    "scores": joint_scores,
                },
                "conditional_sparse_anchor": {
                    "base_prior_method": joint_prior.method,
                    "base_prior_radius": int(joint_prior.radius),
                    "retrieval_mode": selected_condition.mode,
                    "neighbors": int(selected_condition.neighbors),
                    "conditional_mix": float(selected_condition.mix),
                    "conditional_bandwidth": (
                        "infinity" if np.isinf(selected_condition.bandwidth)
                        else float(selected_condition.bandwidth)
                    ),
                    "anchor_center_weight": float(
                        selected_conditional_anchor.center_weight
                    ),
                    "anchor_shape_weight": float(
                        selected_conditional_anchor.shape_weight
                    ),
                    "anchor_cap": float(selected_conditional_anchor.cap),
                    "inner_trial_macro_mae": float(best_conditional[0]),
                    "scores": conditional_scores,
                },
                "functional_ensemble": {
                    "functional_ridge": float(selected_functional.ridge),
                    "functional_slope_radius": int(
                        selected_functional.slope_radius
                    ),
                    "functional_mix": float(selected_functional.mix),
                    "functional_cap": float(selected_functional.cap),
                    "conditional_weight": float(selected_conditional_weight),
                    "inner_trial_macro_mae": float(best_ensemble[0]),
                    "scores": ensemble_scores,
                },
            }
        )

    for name, prediction in outputs.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Incomplete OOF prediction for {name}")
    return NestedSparseAnchorResult(outputs, tuple(selections))


def trial_macro_mae(
    index: InnovationIndex,
    table: TrialTable,
    prediction: np.ndarray,
) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(
        np.mean(
            [
                np.abs(prediction[rows] - index.targets[rows]).mean()
                for rows in table.rows
            ]
        )
    )


def trial_gain_matrix(
    index: InnovationIndex,
    table: TrialTable,
    reference: np.ndarray,
    comparator: np.ndarray,
) -> np.ndarray:
    """Return subject-by-video MAE gain; positive favors ``comparator``."""

    subjects = sorted(set(int(value) for value in table.subjects))
    videos = sorted(set(int(value) for value in table.videos))
    output = np.full((len(subjects), len(videos)), np.nan, dtype=np.float64)
    subject_position = {value: position for position, value in enumerate(subjects)}
    video_position = {value: position for position, value in enumerate(videos)}
    for trial, rows in enumerate(table.rows):
        subject = int(table.subjects[trial])
        video = int(table.videos[trial])
        reference_error = np.abs(reference[rows] - index.targets[rows]).mean()
        comparator_error = np.abs(comparator[rows] - index.targets[rows]).mean()
        output[subject_position[subject], video_position[video]] = (
            reference_error - comparator_error
        )
    if not np.isfinite(output).all():
        raise RuntimeError("Trial gain matrix is incomplete")
    return output


def repeated_outer_folds(
    subjects: Iterable[int],
    *,
    repeats: int,
    folds: int,
    seed: int,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Return canonical folds followed by deterministic shuffled allocations."""

    values = np.asarray(sorted(set(int(value) for value in subjects)), dtype=np.int16)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if folds < 2 or folds > len(values):
        raise ValueError("Invalid fold count")
    plans: list[list[tuple[np.ndarray, np.ndarray]]] = []
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        ordered = values.copy()
        if repeat > 0:
            rng.shuffle(ordered)
        plan: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in range(folds):
            validation = ordered[fold::folds]
            training = values[~np.isin(values, validation)]
            plan.append((training, validation))
        plans.append(plan)
    return plans
