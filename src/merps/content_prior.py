"""Leakage-controlled stimulus-conditioned priors and residual contracts.

The strongest MER-PS baseline memorises a population trajectory for each known
video.  That is useful for a new viewer of a familiar stimulus, but it cannot
predict a genuinely unseen video.  This module adds a stricter bridge:

1. estimate one population trajectory per *training* video;
2. phase-normalise trajectories from videos with the same stimulus category;
3. predict a held-out video's normative trajectory without using any label
   from that video or participant; and
4. permit content and physiology to predict only bounded residuals around that
   prior, with an explicit zero gate.

The category-phase model is deliberately modest.  It is a metadata-only lower
bound for the audio/visual content model that can be fitted after the exact
stimulus files are obtained.  It must never be reported as a visual model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


LABEL_MIN = 1.0
LABEL_MAX = 255.0
VALID_TARGET_CODES = {"MVMA", "HVHA", "LVHA", "LVLA", "HVLA"}


@dataclass(frozen=True)
class StimulusRecord:
    """One authoritative manifest row for a REFED/MER-PS stimulus."""

    video_id: int
    target_code: str
    target_label: str
    source_dataset: str
    title: str
    reported_duration_seconds: int
    annotation_seconds: int
    local_filename: str
    source_reference: str
    source_excerpt: str
    expected_sha256: str
    access_status: str


@dataclass(frozen=True)
class DoubleHeldOutResult:
    """OOF predictions with one participant fold and one video fold per row."""

    predictions: dict[str, np.ndarray]
    subject_fold_ids: np.ndarray
    video_fold_ids: np.ndarray
    fold_records: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class CategoryPhaseSpec:
    """A metadata-only temporal transfer rule selected inside training data."""

    phase_weight: float = 1.0
    duration_temperature: float = float("inf")
    smooth_radius: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.phase_weight) <= 1.0:
            raise ValueError("phase_weight must be in [0, 1]")
        temperature = float(self.duration_temperature)
        if not np.isinf(temperature) and temperature <= 0.0:
            raise ValueError("duration_temperature must be positive or infinity")
        if int(self.smooth_radius) < 0:
            raise ValueError("smooth_radius must be non-negative")

    @property
    def name(self) -> str:
        temperature = (
            "uniform"
            if np.isinf(float(self.duration_temperature))
            else f"tau{float(self.duration_temperature):g}"
        )
        return (
            f"phase{float(self.phase_weight):g}_{temperature}"
            f"_smooth{int(self.smooth_radius)}"
        )


@dataclass(frozen=True)
class ResidualStackSpec:
    """Non-negative, bounded gates for content/EEG/fNIRS corrections."""

    content_gate: float = 0.0
    eeg_gate: float = 0.0
    fnirs_gate: float = 0.0
    max_total_correction: float = 0.0

    def __post_init__(self) -> None:
        gates = (self.content_gate, self.eeg_gate, self.fnirs_gate)
        if any(not 0.0 <= float(value) <= 1.0 for value in gates):
            raise ValueError("Residual gates must be in [0, 1]")
        if float(self.max_total_correction) < 0.0:
            raise ValueError("max_total_correction must be non-negative")
        if any(float(value) > 0.0 for value in gates):
            if float(self.max_total_correction) <= 0.0:
                raise ValueError("An enabled residual stack needs a positive cap")


DEFAULT_CATEGORY_PHASE_SPECS = tuple(
    CategoryPhaseSpec(phase_weight, temperature, radius)
    for phase_weight in (0.25, 0.50, 0.75, 1.00)
    for temperature in (30.0, 60.0, float("inf"))
    for radius in (0, 1, 3)
)


def load_stimulus_manifest(path: str | Path) -> tuple[StimulusRecord, ...]:
    """Read and validate a stimulus manifest without guessing missing fields."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[StimulusRecord] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = set(StimulusRecord.__dataclass_fields__)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Stimulus manifest is missing columns: {sorted(missing)}")
        for row in reader:
            record = StimulusRecord(
                video_id=int(row["video_id"]),
                target_code=str(row["target_code"]).strip(),
                target_label=str(row["target_label"]).strip(),
                source_dataset=str(row["source_dataset"]).strip(),
                title=str(row["title"]).strip(),
                reported_duration_seconds=int(row["reported_duration_seconds"]),
                annotation_seconds=int(row["annotation_seconds"]),
                local_filename=str(row["local_filename"]).strip(),
                source_reference=str(row["source_reference"]).strip(),
                source_excerpt=str(row["source_excerpt"]).strip(),
                expected_sha256=str(row["expected_sha256"]).strip().lower(),
                access_status=str(row["access_status"]).strip(),
            )
            _validate_record(record)
            records.append(record)
    if not records:
        raise ValueError(f"No stimulus rows found in {path}")
    ids = [record.video_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Stimulus video_id values must be unique")
    if sorted(ids) != list(range(1, len(ids) + 1)):
        raise ValueError("Stimulus video_id values must be contiguous from 1")
    return tuple(sorted(records, key=lambda record: record.video_id))


def _validate_record(record: StimulusRecord) -> None:
    if record.video_id < 1:
        raise ValueError("video_id must be positive")
    if record.target_code not in VALID_TARGET_CODES:
        raise ValueError(f"Unknown target code: {record.target_code}")
    if not record.target_label or not record.title or not record.local_filename:
        raise ValueError("target_label, title and local_filename are required")
    if record.reported_duration_seconds <= 0 or record.annotation_seconds <= 0:
        raise ValueError("Stimulus durations must be positive")
    if record.annotation_seconds > record.reported_duration_seconds + 2:
        raise ValueError("Annotation duration is implausibly longer than the clip")
    if record.expected_sha256:
        valid = len(record.expected_sha256) == 64 and all(
            character in "0123456789abcdef" for character in record.expected_sha256
        )
        if not valid:
            raise ValueError("expected_sha256 must be blank or a lowercase SHA-256")


def emotion_by_video(
    records: Sequence[StimulusRecord],
) -> dict[int, str]:
    return {int(record.video_id): str(record.target_label) for record in records}


def round_robin_folds(
    values: Sequence[int], n_folds: int = 5
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    unique = np.asarray(sorted(set(int(value) for value in values)), dtype=np.int16)
    if len(unique) < 2:
        raise ValueError("At least two groups are required")
    folds = max(2, min(int(n_folds), len(unique)))
    result = []
    for fold in range(folds):
        validation = unique[fold::folds]
        training = unique[~np.isin(unique, validation)]
        result.append((training, validation))
    return tuple(result)


def phase_resample(curve: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample a two-axis trajectory on normalised stimulus phase."""

    values = np.asarray(curve, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 1:
        raise ValueError("curve must have shape [time, 2]")
    if int(length) < 1:
        raise ValueError("length must be positive")
    if len(values) == int(length):
        return values.copy()
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = np.linspace(0.0, 1.0, int(length), dtype=np.float64)
    return np.stack(
        [np.interp(target, source, values[:, axis]) for axis in range(2)], axis=1
    ).astype(np.float32)


def absolute_time_resample(curve: np.ndarray, length: int) -> np.ndarray:
    """Transfer a trajectory on elapsed seconds, extending its final state."""

    values = np.asarray(curve, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 1:
        raise ValueError("curve must have shape [time, 2]")
    if int(length) < 1:
        raise ValueError("length must be positive")
    source = np.minimum(np.arange(int(length), dtype=np.int64), len(values) - 1)
    return values[source].copy()


def smooth_curve(curve: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(curve, dtype=np.float32)
    if int(radius) <= 0:
        return values.copy()
    output = np.empty_like(values)
    for timestamp in range(len(values)):
        start = max(0, timestamp - int(radius))
        end = min(len(values), timestamp + int(radius) + 1)
        output[timestamp] = values[start:end].mean(axis=0)
    return output


def _category_template(
    curves: Mapping[int, np.ndarray],
    source_videos: Sequence[int],
    *,
    target_length: int,
    target_duration: float,
    duration_by_video: Mapping[int, float],
    spec: CategoryPhaseSpec,
) -> np.ndarray:
    if not source_videos:
        raise ValueError("A category template needs at least one source video")
    pieces = []
    weights = []
    for video in source_videos:
        curve = smooth_curve(curves[int(video)], spec.smooth_radius)
        phase = phase_resample(curve, target_length)
        absolute = absolute_time_resample(curve, target_length)
        pieces.append(
            float(spec.phase_weight) * phase
            + (1.0 - float(spec.phase_weight)) * absolute
        )
        temperature = float(spec.duration_temperature)
        if np.isinf(temperature):
            weights.append(1.0)
        else:
            distance = abs(float(duration_by_video[int(video)]) - float(target_duration))
            weights.append(float(np.exp(-distance / temperature)))
    return np.average(
        np.stack(pieces), axis=0, weights=np.asarray(weights, dtype=np.float64)
    ).astype(np.float32)


def _population_curve(
    index,
    training_subjects: Sequence[int],
    video: int,
) -> np.ndarray:
    source = np.isin(
        index.subject_numbers, np.asarray(training_subjects, dtype=np.int16)
    ) & (index.videos == int(video))
    if not np.any(source):
        raise ValueError(f"No source labels for video {video}")
    source_timestamps = index.timestamps[source]
    source_targets = index.targets[source]
    length = int(source_timestamps.max()) + 1
    curve = np.empty((length, 2), dtype=np.float32)
    for timestamp in range(length):
        rows = source_timestamps == timestamp
        if not np.any(rows):
            raise ValueError(f"Missing timestamp {timestamp} for video {video}")
        curve[timestamp] = np.median(source_targets[rows], axis=0)
    return curve


def predict_category_constant_prior(
    index,
    training_subjects: Sequence[int],
    training_videos: Sequence[int],
    prediction_indices: np.ndarray,
    category_by_video: Mapping[int, str],
) -> np.ndarray:
    """Predict a held-out video with its training-video category median."""

    prediction_indices = np.asarray(prediction_indices, dtype=np.int64)
    source_base = np.isin(index.subject_numbers, training_subjects) & np.isin(
        index.videos, training_videos
    )
    if not np.any(source_base):
        raise ValueError("The category prior needs source labels")
    global_value = np.median(index.targets[source_base], axis=0)
    prediction = np.empty((len(prediction_indices), 2), dtype=np.float32)
    for video in sorted(set(int(index.videos[row]) for row in prediction_indices)):
        local = np.flatnonzero(index.videos[prediction_indices] == video)
        category = category_by_video[int(video)]
        source_videos = [
            int(candidate)
            for candidate in training_videos
            if category_by_video[int(candidate)] == category
            and int(candidate) != int(video)
        ]
        source = source_base & np.isin(index.videos, source_videos)
        prediction[local] = (
            np.median(index.targets[source], axis=0) if np.any(source) else global_value
        )
    return np.clip(prediction, LABEL_MIN, LABEL_MAX)


def predict_category_phase_prior(
    index,
    training_subjects: Sequence[int],
    training_videos: Sequence[int],
    prediction_indices: np.ndarray,
    category_by_video: Mapping[int, str],
    *,
    reducer: str = "mean",
    duration_by_video: Mapping[int, float] | None = None,
    spec: CategoryPhaseSpec = CategoryPhaseSpec(),
) -> np.ndarray:
    """Predict from same-category *other-video* phase-normalised trajectories.

    Labels from prediction videos are never consulted.  ``index.timestamps``
    supplies only the requested output position/length, which is known from the
    stimulus schedule at deployment time.
    """

    if reducer not in {"mean", "median"}:
        raise ValueError("reducer must be 'mean' or 'median'")
    training_subjects = np.asarray(training_subjects, dtype=np.int16)
    training_videos = np.asarray(training_videos, dtype=np.int16)
    prediction_indices = np.asarray(prediction_indices, dtype=np.int64)
    if not len(prediction_indices):
        return np.empty((0, 2), dtype=np.float32)
    curves = {
        int(video): _population_curve(index, training_subjects, int(video))
        for video in training_videos
    }
    source_mask = np.isin(index.subject_numbers, training_subjects) & np.isin(
        index.videos, training_videos
    )
    if not np.any(source_mask):
        raise ValueError("The category-phase prior needs source labels")
    fallback = np.median(index.targets[source_mask], axis=0).astype(np.float32)
    reduce_fn = np.mean if reducer == "mean" else np.median
    durations = {
        int(video): float(duration_by_video[int(video)])
        if duration_by_video is not None
        else float(len(curve))
        for video, curve in curves.items()
    }
    prediction = np.empty((len(prediction_indices), 2), dtype=np.float32)
    requested_videos = sorted(
        set(int(index.videos[row]) for row in prediction_indices)
    )
    for video in requested_videos:
        local = np.flatnonzero(index.videos[prediction_indices] == video)
        rows = prediction_indices[local]
        length = int(index.timestamps[rows].max()) + 1
        category = category_by_video[video]
        source_videos = [
            int(candidate)
            for candidate in training_videos
            if category_by_video[int(candidate)] == category
            and int(candidate) != video
        ]
        if source_videos:
            target_duration = (
                float(duration_by_video[video])
                if duration_by_video is not None
                else float(length)
            )
            if reducer == "mean":
                template = _category_template(
                    curves,
                    source_videos,
                    target_length=length,
                    target_duration=target_duration,
                    duration_by_video=durations,
                    spec=spec,
                )
            else:
                aligned = []
                for candidate in source_videos:
                    curve = smooth_curve(curves[candidate], spec.smooth_radius)
                    aligned.append(
                        float(spec.phase_weight) * phase_resample(curve, length)
                        + (1.0 - float(spec.phase_weight))
                        * absolute_time_resample(curve, length)
                    )
                template = reduce_fn(np.stack(aligned), axis=0).astype(np.float32)
        else:
            template = np.repeat(fallback[None, :], length, axis=0)
        prediction[local] = template[index.timestamps[rows]]
    return np.clip(prediction, LABEL_MIN, LABEL_MAX)


def cross_fitted_category_phase_prior(
    index,
    training_subjects: Sequence[int],
    training_videos: Sequence[int],
    category_by_video: Mapping[int, str],
    *,
    duration_by_video: Mapping[int, float] | None = None,
    spec: CategoryPhaseSpec = CategoryPhaseSpec(),
) -> tuple[np.ndarray, np.ndarray]:
    """Build training residual targets without a row's participant or video.

    Every returned prior excludes all labels from the predicted participant;
    :func:`predict_category_phase_prior` additionally excludes the predicted
    video.  This mirrors the new-participant/new-video deployment boundary for
    residual-model fitting rather than letting a training row explain itself.
    """

    subjects = np.asarray(sorted(set(int(value) for value in training_subjects)))
    videos = np.asarray(sorted(set(int(value) for value in training_videos)))
    if len(subjects) < 2:
        raise ValueError("Cross-fitted priors require at least two participants")
    rows_out: list[np.ndarray] = []
    prior_out: list[np.ndarray] = []
    for subject in subjects:
        rows = np.flatnonzero(
            (index.subject_numbers == int(subject)) & np.isin(index.videos, videos)
        )
        source_subjects = subjects[subjects != int(subject)]
        prior = predict_category_phase_prior(
            index,
            source_subjects,
            videos,
            rows,
            category_by_video,
            duration_by_video=duration_by_video,
            spec=spec,
        )
        rows_out.append(rows)
        prior_out.append(prior)
    rows = np.concatenate(rows_out).astype(np.int64)
    order = np.argsort(rows)
    return rows[order], np.concatenate(prior_out).astype(np.float32)[order]


def select_category_phase_spec(
    index,
    training_subjects: Sequence[int],
    training_videos: Sequence[int],
    category_by_video: Mapping[int, str],
    duration_by_video: Mapping[int, float],
    *,
    candidates: Sequence[CategoryPhaseSpec] = DEFAULT_CATEGORY_PHASE_SPECS,
    inner_subject_folds: int = 3,
) -> tuple[CategoryPhaseSpec, dict[str, float]]:
    """Select transfer geometry using held-out subjects and videos internally."""

    if not candidates:
        raise ValueError("At least one category-phase candidate is required")
    training_subjects = np.asarray(training_subjects, dtype=np.int16)
    training_videos = np.asarray(training_videos, dtype=np.int16)
    scores: dict[str, list[float]] = {spec.name: [] for spec in candidates}
    for inner_training_subjects, inner_validation_subjects in round_robin_folds(
        training_subjects, inner_subject_folds
    ):
        curves = {
            int(video): _population_curve(
                index, inner_training_subjects, int(video)
            )
            for video in training_videos
        }
        for validation_video in training_videos:
            source_videos = [
                int(video)
                for video in training_videos
                if int(video) != int(validation_video)
                and category_by_video[int(video)]
                == category_by_video[int(validation_video)]
            ]
            if not source_videos:
                continue
            rows = np.flatnonzero(
                np.isin(index.subject_numbers, inner_validation_subjects)
                & (index.videos == int(validation_video))
            )
            if not len(rows):
                continue
            length = int(index.timestamps[rows].max()) + 1
            target = index.targets[rows]
            timestamps = index.timestamps[rows]
            for spec in candidates:
                template = _category_template(
                    curves,
                    source_videos,
                    target_length=length,
                    target_duration=float(duration_by_video[int(validation_video)]),
                    duration_by_video=duration_by_video,
                    spec=spec,
                )
                scores[spec.name].append(
                    float(np.abs(template[timestamps] - target).mean())
                )
    mean_scores = {
        name: float(np.mean(values)) if values else float("inf")
        for name, values in scores.items()
    }
    best = min(
        candidates,
        key=lambda candidate: (mean_scores[candidate.name], candidate.name),
    )
    if not np.isfinite(mean_scores[best.name]):
        raise RuntimeError("No valid inner category-phase comparison")
    return best, mean_scores


def nested_doubly_held_out_category_priors(
    index,
    category_by_video: Mapping[int, str],
    duration_by_video: Mapping[int, float],
    *,
    candidates: Sequence[CategoryPhaseSpec] = DEFAULT_CATEGORY_PHASE_SPECS,
    subject_folds: int = 5,
    video_folds: int = 5,
    inner_subject_folds: int = 3,
) -> DoubleHeldOutResult:
    """Double-held-out OOF prediction with inner-only transfer selection."""

    subjects = sorted(set(int(value) for value in index.subject_numbers))
    videos = sorted(set(int(value) for value in index.videos))
    missing_categories = set(videos).difference(category_by_video)
    missing_durations = set(videos).difference(duration_by_video)
    if missing_categories or missing_durations:
        raise ValueError(
            f"Missing metadata: categories={sorted(missing_categories)}, "
            f"durations={sorted(missing_durations)}"
        )
    predictions = {
        "global_constant": np.full_like(index.targets, np.nan, dtype=np.float32),
        "category_constant": np.full_like(index.targets, np.nan, dtype=np.float32),
        "category_phase_nested": np.full_like(index.targets, np.nan, dtype=np.float32),
    }
    subject_fold_ids = np.full(len(index.targets), -1, dtype=np.int8)
    video_fold_ids = np.full(len(index.targets), -1, dtype=np.int8)
    records: list[dict[str, object]] = []
    subject_splits = round_robin_folds(subjects, subject_folds)
    video_splits = round_robin_folds(videos, video_folds)
    for subject_fold, (training_subjects, validation_subjects) in enumerate(
        subject_splits
    ):
        for video_fold, (training_videos, validation_videos) in enumerate(video_splits):
            spec, inner_scores = select_category_phase_spec(
                index,
                training_subjects,
                training_videos,
                category_by_video,
                duration_by_video,
                candidates=candidates,
                inner_subject_folds=inner_subject_folds,
            )
            validation = np.flatnonzero(
                np.isin(index.subject_numbers, validation_subjects)
                & np.isin(index.videos, validation_videos)
            )
            source = np.isin(index.subject_numbers, training_subjects) & np.isin(
                index.videos, training_videos
            )
            predictions["global_constant"][validation] = np.median(
                index.targets[source], axis=0
            )
            predictions["category_constant"][validation] = (
                predict_category_constant_prior(
                    index,
                    training_subjects,
                    training_videos,
                    validation,
                    category_by_video,
                )
            )
            predictions["category_phase_nested"][validation] = (
                predict_category_phase_prior(
                    index,
                    training_subjects,
                    training_videos,
                    validation,
                    category_by_video,
                    duration_by_video=duration_by_video,
                    spec=spec,
                )
            )
            subject_fold_ids[validation] = subject_fold
            video_fold_ids[validation] = video_fold
            records.append(
                {
                    "subject_fold": subject_fold,
                    "video_fold": video_fold,
                    "training_subjects": training_subjects.tolist(),
                    "validation_subjects": validation_subjects.tolist(),
                    "training_videos": training_videos.tolist(),
                    "validation_videos": validation_videos.tolist(),
                    "selected_spec": {
                        "name": spec.name,
                        "phase_weight": float(spec.phase_weight),
                        "duration_temperature": (
                            "infinity"
                            if np.isinf(float(spec.duration_temperature))
                            else float(spec.duration_temperature)
                        ),
                        "smooth_radius": int(spec.smooth_radius),
                    },
                    "inner_mae_by_spec": inner_scores,
                    "validation_rows": int(len(validation)),
                }
            )
    if np.any(subject_fold_ids < 0) or np.any(video_fold_ids < 0):
        raise RuntimeError("Incomplete nested double-held-out fold assignment")
    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete nested OOF prediction: {name}")
    return DoubleHeldOutResult(
        predictions=predictions,
        subject_fold_ids=subject_fold_ids,
        video_fold_ids=video_fold_ids,
        fold_records=tuple(records),
    )


def doubly_held_out_category_priors(
    index,
    category_by_video: Mapping[int, str],
    *,
    subject_folds: int = 5,
    video_folds: int = 5,
) -> DoubleHeldOutResult:
    """Hold out participants and videos simultaneously for every OOF row."""

    subjects = sorted(set(int(value) for value in index.subject_numbers))
    videos = sorted(set(int(value) for value in index.videos))
    missing = set(videos).difference(category_by_video)
    if missing:
        raise ValueError(f"Missing categories for videos: {sorted(missing)}")
    subject_splits = round_robin_folds(subjects, subject_folds)
    video_splits = round_robin_folds(videos, video_folds)
    predictions = {
        "global_constant": np.full_like(index.targets, np.nan, dtype=np.float32),
        "category_constant": np.full_like(index.targets, np.nan, dtype=np.float32),
        "category_phase": np.full_like(index.targets, np.nan, dtype=np.float32),
    }
    subject_fold_ids = np.full(len(index.targets), -1, dtype=np.int8)
    video_fold_ids = np.full(len(index.targets), -1, dtype=np.int8)
    records: list[dict[str, object]] = []
    for subject_fold, (training_subjects, validation_subjects) in enumerate(
        subject_splits
    ):
        for video_fold, (training_videos, validation_videos) in enumerate(video_splits):
            validation = np.flatnonzero(
                np.isin(index.subject_numbers, validation_subjects)
                & np.isin(index.videos, validation_videos)
            )
            source = np.isin(index.subject_numbers, training_subjects) & np.isin(
                index.videos, training_videos
            )
            global_value = np.median(index.targets[source], axis=0)
            predictions["global_constant"][validation] = global_value
            predictions["category_constant"][validation] = (
                predict_category_constant_prior(
                    index,
                    training_subjects,
                    training_videos,
                    validation,
                    category_by_video,
                )
            )
            predictions["category_phase"][validation] = predict_category_phase_prior(
                index,
                training_subjects,
                training_videos,
                validation,
                category_by_video,
            )
            subject_fold_ids[validation] = subject_fold
            video_fold_ids[validation] = video_fold
            records.append(
                {
                    "subject_fold": subject_fold,
                    "video_fold": video_fold,
                    "training_subjects": training_subjects.tolist(),
                    "validation_subjects": validation_subjects.tolist(),
                    "training_videos": training_videos.tolist(),
                    "validation_videos": validation_videos.tolist(),
                    "validation_rows": int(len(validation)),
                }
            )
    if np.any(subject_fold_ids < 0) or np.any(video_fold_ids < 0):
        raise RuntimeError("Incomplete double-held-out fold assignment")
    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete OOF prediction: {name}")
    return DoubleHeldOutResult(
        predictions=predictions,
        subject_fold_ids=subject_fold_ids,
        video_fold_ids=video_fold_ids,
        fold_records=tuple(records),
    )


def residual_target(target: np.ndarray, prior: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)
    if target.shape != prior.shape or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target and prior must both have shape [samples, 2]")
    if not np.isfinite(target).all() or not np.isfinite(prior).all():
        raise ValueError("target and prior must be finite")
    return (target - prior).astype(np.float32)


def apply_residual_stack(
    prior: np.ndarray,
    *,
    content_residual: np.ndarray | None = None,
    eeg_residual: np.ndarray | None = None,
    fnirs_residual: np.ndarray | None = None,
    spec: ResidualStackSpec = ResidualStackSpec(),
) -> np.ndarray:
    """Apply gated residuals while preserving an exact prior-only fallback."""

    prior = np.asarray(prior, dtype=np.float32)
    if prior.ndim != 2 or prior.shape[1] != 2 or not np.isfinite(prior).all():
        raise ValueError("prior must be finite with shape [samples, 2]")
    correction = np.zeros_like(prior)
    for residual, gate, name in (
        (content_residual, spec.content_gate, "content"),
        (eeg_residual, spec.eeg_gate, "eeg"),
        (fnirs_residual, spec.fnirs_gate, "fnirs"),
    ):
        if float(gate) == 0.0:
            continue
        if residual is None:
            raise ValueError(f"Enabled {name} gate needs a residual")
        values = np.asarray(residual, dtype=np.float32)
        if values.shape != prior.shape or not np.isfinite(values).all():
            raise ValueError(f"{name}_residual must match prior and be finite")
        correction += float(gate) * values
    cap = float(spec.max_total_correction)
    if cap > 0.0:
        correction = np.clip(correction, -cap, cap)
    elif np.any(correction != 0.0):
        raise ValueError("A non-zero correction needs a positive cap")
    return np.clip(prior + correction, LABEL_MIN, LABEL_MAX).astype(np.float32)


def trial_macro_mae(index, prediction: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape != index.targets.shape:
        raise ValueError("prediction shape does not match index targets")
    errors = []
    pairs = sorted(
        set(
            zip(
                (int(value) for value in index.subject_numbers),
                (int(value) for value in index.videos),
                strict=True,
            )
        )
    )
    for subject, video in pairs:
        rows = (index.subject_numbers == subject) & (index.videos == video)
        errors.append(float(np.abs(prediction[rows] - index.targets[rows]).mean()))
    return float(np.mean(errors))
