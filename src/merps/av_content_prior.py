"""Content-selected and content-aligned temporal priors.

The model is designed for a known stimulus file and a new participant. Frozen
CLIP features choose among same-category population trajectories learned from
training videos. A constrained monotonic alignment can then move trajectory
events to content-matched playback times. All labels from the target video and
participant remain unavailable, and a zero content gate exactly recovers the
metadata prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


LABEL_MIN = 1.0
LABEL_MAX = 255.0


@dataclass(frozen=True)
class ContentTemplateSpec:
    source_temperature: float
    warp_band: float
    phase_penalty: float
    warp_gate: float
    content_gate: float
    modality: str = "clip"

    def __post_init__(self) -> None:
        temperature = float(self.source_temperature)
        if temperature < 0.0:
            raise ValueError("source_temperature must be non-negative")
        if not 0.0 <= float(self.warp_band) <= 1.0:
            raise ValueError("warp_band must be in [0, 1]")
        if float(self.phase_penalty) < 0.0:
            raise ValueError("phase_penalty must be non-negative")
        if not 0.0 <= float(self.warp_gate) <= 1.0:
            raise ValueError("warp_gate must be in [0, 1]")
        if not 0.0 <= float(self.content_gate) <= 1.0:
            raise ValueError("content_gate must be in [0, 1]")
        if self.modality not in {
            "clip",
            "ast",
            "clip_ast",
            "visual",
            "visual_ast",
        }:
            raise ValueError(
                "modality must be clip, ast, clip_ast, visual or visual_ast"
            )
        if float(self.warp_gate) > 0.0 and float(self.warp_band) <= 0.0:
            raise ValueError("enabled warping needs a positive warp_band")

    @property
    def name(self) -> str:
        if float(self.content_gate) == 0.0:
            return "disabled"
        if np.isinf(float(self.source_temperature)):
            temperature = "uniform"
        elif float(self.source_temperature) == 0.0:
            temperature = "hard"
        else:
            temperature = f"tau{float(self.source_temperature):g}"
        warp = (
            "linear"
            if float(self.warp_gate) == 0.0
            else (
                f"dtw{float(self.warp_band):g}"
                f"_p{float(self.phase_penalty):g}"
                f"_wg{float(self.warp_gate):g}"
            )
        )
        return (
            f"{self.modality}_{temperature}_{warp}"
            f"_cg{float(self.content_gate):g}"
        )


DISABLED_CONTENT_SPEC = ContentTemplateSpec(
    source_temperature=float("inf"),
    warp_band=0.0,
    phase_penalty=0.0,
    warp_gate=0.0,
    content_gate=0.0,
)


@dataclass(frozen=True)
class FoundationFeatureArchive:
    feature_names: tuple[str, ...]
    features_by_video: Mapping[int, np.ndarray]
    annotation_lengths: Mapping[int, int]
    schema_version: str
    media_sha256: Mapping[int, str]

    @classmethod
    def load(cls, path: str | Path) -> "FoundationFeatureArchive":
        with np.load(Path(path), allow_pickle=False) as payload:
            names = tuple(str(value) for value in payload["feature_names"])
            values = np.asarray(payload["features"], dtype=np.float32)
            videos = np.asarray(payload["video_ids"], dtype=np.int16)
            timestamps = np.asarray(payload["timestamps"], dtype=np.int16)
            record_videos = np.asarray(payload["record_video_ids"], dtype=np.int16)
            annotation = np.asarray(payload["annotation_seconds"], dtype=np.int16)
            hashes = np.asarray(payload["media_sha256"])
            schema = str(payload["schema_version"])
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("Invalid foundation feature matrix")
        if len(values) != len(videos) or len(values) != len(timestamps):
            raise ValueError("Foundation feature index lengths differ")
        if not np.isfinite(values).all():
            raise ValueError("Foundation features must be finite")
        if len(record_videos) != len(annotation) or len(record_videos) != len(hashes):
            raise ValueError("Foundation record metadata lengths differ")
        by_video: dict[int, np.ndarray] = {}
        for video in record_videos:
            rows = np.flatnonzero(videos == int(video))
            order = np.argsort(timestamps[rows])
            rows = rows[order]
            expected = np.arange(len(rows), dtype=timestamps.dtype)
            if not np.array_equal(timestamps[rows], expected):
                raise ValueError(f"Non-contiguous foundation timestamps for video {video}")
            by_video[int(video)] = values[rows].copy()
        return cls(
            feature_names=names,
            features_by_video=by_video,
            annotation_lengths={
                int(video): int(length)
                for video, length in zip(record_videos, annotation)
            },
            schema_version=schema,
            media_sha256={
                int(video): str(digest)
                for video, digest in zip(record_videos, hashes)
            },
        )

    def _columns(self, modality: str) -> np.ndarray:
        if modality not in {
            "clip",
            "ast",
            "clip_ast",
            "visual",
            "visual_ast",
        }:
            raise ValueError(f"Unsupported content modality: {modality}")
        selected = [
            index
            for index, name in enumerate(self.feature_names)
            if (
                modality in {"clip_ast", "visual_ast"}
                or (modality == "clip" and name.startswith("clip_"))
                or (modality == "visual" and name.startswith("visual_"))
                or (modality == "ast" and name.startswith("ast_"))
            )
        ]
        if not selected:
            raise ValueError(f"No {modality} columns in feature archive")
        return np.asarray(selected, dtype=np.int64)

    def sequence(
        self,
        video: int,
        *,
        length: int | None = None,
        offset_seconds: int = 0,
        modality: str = "clip",
    ) -> np.ndarray:
        video = int(video)
        if video not in self.features_by_video:
            raise KeyError(f"Missing foundation features for video {video}")
        values = np.asarray(self.features_by_video[video], dtype=np.float32)
        requested = (
            int(self.annotation_lengths[video]) if length is None else int(length)
        )
        if requested < 1:
            raise ValueError("Requested sequence length must be positive")
        positions = np.clip(
            np.arange(requested, dtype=np.int64) + int(offset_seconds),
            0,
            len(values) - 1,
        )
        return values[positions][:, self._columns(modality)].copy()

    def descriptor(
        self,
        video: int,
        *,
        length: int | None = None,
        offset_seconds: int = 0,
        modality: str = "clip",
    ) -> np.ndarray:
        vector = self.sequence(
            video,
            length=length,
            offset_seconds=offset_seconds,
            modality=modality,
        ).mean(axis=0, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"Degenerate content descriptor for video {video}")
        return (vector / norm).astype(np.float32)


def phase_resample_matrix(values: np.ndarray, length: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) < 1:
        raise ValueError("values must have shape [time, dimensions]")
    if int(length) < 1:
        raise ValueError("length must be positive")
    if len(matrix) == int(length):
        return matrix.copy()
    positions = np.linspace(0.0, len(matrix) - 1, int(length))
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(matrix) - 1)
    weight = (positions - lower).astype(np.float32)[:, None]
    return ((1.0 - weight) * matrix[lower] + weight * matrix[upper]).astype(
        np.float32
    )


def monotonic_content_map(
    target_features: np.ndarray,
    source_features: np.ndarray,
    *,
    band: float,
    phase_penalty: float = 0.0,
) -> np.ndarray:
    """Map every target second to a source position with constrained DTW."""

    target = np.asarray(target_features, dtype=np.float32)
    source = np.asarray(source_features, dtype=np.float32)
    if target.ndim != 2 or source.ndim != 2 or target.shape[1] != source.shape[1]:
        raise ValueError("Target/source features must have matching 2-D shapes")
    if not 0.0 < float(band) <= 1.0:
        raise ValueError("band must be in (0, 1]")
    if float(phase_penalty) < 0.0:
        raise ValueError("phase_penalty must be non-negative")
    target_norm = target / np.maximum(
        np.linalg.norm(target, axis=1, keepdims=True), 1e-8
    )
    source_norm = source / np.maximum(
        np.linalg.norm(source, axis=1, keepdims=True), 1e-8
    )
    n_target, n_source = len(target), len(source)
    target_phase = np.arange(n_target) / max(n_target - 1, 1)
    source_phase = np.arange(n_source) / max(n_source - 1, 1)
    phase_distance = np.abs(target_phase[:, None] - source_phase[None, :])
    cost = (
        1.0
        - np.asarray(target_norm @ source_norm.T, dtype=np.float64)
        + float(phase_penalty) * phase_distance
    )
    cost[phase_distance > float(band)] = np.inf
    cumulative = np.full((n_target, n_source), np.inf, dtype=np.float64)
    predecessor = np.zeros((n_target, n_source), dtype=np.int8)
    cumulative[0, 0] = cost[0, 0]
    for target_index in range(n_target):
        for source_index in range(n_source):
            if target_index == 0 and source_index == 0:
                continue
            candidates: list[tuple[float, int]] = []
            if target_index > 0:
                candidates.append((cumulative[target_index - 1, source_index], 1))
            if source_index > 0:
                candidates.append((cumulative[target_index, source_index - 1], 2))
            if target_index > 0 and source_index > 0:
                candidates.append(
                    (cumulative[target_index - 1, source_index - 1], 3)
                )
            previous, code = min(candidates)
            cumulative[target_index, source_index] = (
                cost[target_index, source_index] + previous
            )
            predecessor[target_index, source_index] = code
    if not np.isfinite(cumulative[-1, -1]):
        raise RuntimeError("No feasible monotonic content path")
    target_index = n_target - 1
    source_index = n_source - 1
    path: list[tuple[int, int]] = []
    while True:
        path.append((target_index, source_index))
        if target_index == 0 and source_index == 0:
            break
        code = int(predecessor[target_index, source_index])
        if code in {1, 3}:
            target_index -= 1
        if code in {2, 3}:
            source_index -= 1
        if code == 0:
            raise RuntimeError("Broken monotonic content path")
    mapping = np.empty(n_target, dtype=np.float32)
    for target_index in range(n_target):
        positions = [
            source_index
            for path_target, source_index in path
            if path_target == target_index
        ]
        mapping[target_index] = float(np.mean(positions))
    return mapping


def sample_curve_at_positions(curve: np.ndarray, positions: np.ndarray) -> np.ndarray:
    values = np.asarray(curve, dtype=np.float32)
    requested = np.asarray(positions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 1:
        raise ValueError("curve must have shape [time, 2]")
    clipped = np.clip(requested, 0.0, len(values) - 1)
    lower = np.floor(clipped).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = (clipped - lower)[:, None]
    return ((1.0 - weight) * values[lower] + weight * values[upper]).astype(
        np.float32
    )


def population_curves(
    index,
    subjects: Sequence[int],
    videos: Sequence[int],
) -> dict[int, np.ndarray]:
    subject_values = np.asarray(subjects, dtype=np.int16)
    result: dict[int, np.ndarray] = {}
    for video in videos:
        video = int(video)
        rows = np.isin(index.subject_numbers, subject_values) & (
            index.videos == video
        )
        if not np.any(rows):
            raise ValueError(f"No population labels for video {video}")
        length = int(index.timestamps[rows].max()) + 1
        curve = np.empty((length, 2), dtype=np.float32)
        for timestamp in range(length):
            cell = rows & (index.timestamps == timestamp)
            if not np.any(cell):
                raise ValueError(f"Missing timestamp {timestamp} for video {video}")
            curve[timestamp] = np.median(index.targets[cell], axis=0)
        result[video] = curve
    return result


def source_similarity_weights(
    archive: FoundationFeatureArchive,
    target_video: int,
    source_videos: Sequence[int],
    *,
    temperature: float,
    offset_seconds: int = 0,
    modality: str = "clip",
) -> np.ndarray:
    sources = [int(value) for value in source_videos]
    if not sources:
        raise ValueError("At least one source video is required")
    target = archive.descriptor(
        int(target_video), offset_seconds=offset_seconds, modality=modality
    )
    similarities = np.asarray(
        [
            float(
                target
                @ archive.descriptor(
                    source, offset_seconds=offset_seconds, modality=modality
                )
            )
            for source in sources
        ],
        dtype=np.float64,
    )
    temperature = float(temperature)
    if np.isinf(temperature):
        return np.full(len(sources), 1.0 / len(sources), dtype=np.float32)
    if temperature == 0.0:
        weights = np.zeros(len(sources), dtype=np.float32)
        weights[int(np.argmax(similarities))] = 1.0
        return weights
    logits = (similarities - similarities.max()) / temperature
    weights = np.exp(logits)
    return (weights / weights.sum()).astype(np.float32)


def content_template_curve(
    archive: FoundationFeatureArchive,
    target_video: int,
    source_curves: Mapping[int, np.ndarray],
    source_videos: Sequence[int],
    *,
    target_length: int,
    spec: ContentTemplateSpec,
    offset_seconds: int = 0,
    map_cache: dict[tuple[object, ...], np.ndarray] | None = None,
) -> np.ndarray:
    sources = [int(value) for value in source_videos]
    weights = source_similarity_weights(
        archive,
        int(target_video),
        sources,
        temperature=spec.source_temperature,
        offset_seconds=offset_seconds,
        modality=spec.modality,
    )
    transferred = []
    for source in sources:
        curve = np.asarray(source_curves[source], dtype=np.float32)
        linear = phase_resample_matrix(curve, int(target_length))
        if float(spec.warp_gate) == 0.0:
            transferred.append(linear)
            continue
        key = (
            int(target_video),
            int(source),
            int(offset_seconds),
            float(spec.warp_band),
            float(spec.phase_penalty),
            str(spec.modality),
        )
        mapping = None if map_cache is None else map_cache.get(key)
        if mapping is None:
            target_features = archive.sequence(
                int(target_video),
                length=int(target_length),
                offset_seconds=offset_seconds,
                modality=spec.modality,
            )
            source_features = archive.sequence(
                int(source),
                length=len(curve),
                offset_seconds=offset_seconds,
                modality=spec.modality,
            )
            mapping = monotonic_content_map(
                target_features,
                source_features,
                band=spec.warp_band,
                phase_penalty=spec.phase_penalty,
            )
            if map_cache is not None:
                map_cache[key] = mapping
        warped = sample_curve_at_positions(curve, mapping)
        transferred.append(
            (
                (1.0 - float(spec.warp_gate)) * linear
                + float(spec.warp_gate) * warped
            ).astype(np.float32)
        )
    return np.average(
        np.stack(transferred), axis=0, weights=weights
    ).astype(np.float32)


def blend_content_prior(
    metadata_prior: np.ndarray,
    template: np.ndarray,
    content_gate: float,
) -> np.ndarray:
    prior = np.asarray(metadata_prior, dtype=np.float32)
    content = np.asarray(template, dtype=np.float32)
    if prior.shape != content.shape or prior.ndim != 2 or prior.shape[1] != 2:
        raise ValueError("metadata_prior/template must both have shape [samples, 2]")
    if not 0.0 <= float(content_gate) <= 1.0:
        raise ValueError("content_gate must be in [0, 1]")
    if float(content_gate) == 0.0:
        return prior.copy()
    return np.clip(
        (1.0 - float(content_gate)) * prior
        + float(content_gate) * content,
        LABEL_MIN,
        LABEL_MAX,
    ).astype(np.float32)


def predict_content_template_prior(
    index,
    prediction_indices: np.ndarray,
    metadata_prior: np.ndarray,
    archive: FoundationFeatureArchive,
    training_subjects: Sequence[int],
    training_videos: Sequence[int],
    category_by_video: Mapping[int, str],
    *,
    spec: ContentTemplateSpec,
    offset_seconds: int = 0,
    curves: Mapping[int, np.ndarray] | None = None,
    map_cache: dict[tuple[object, ...], np.ndarray] | None = None,
) -> np.ndarray:
    rows = np.asarray(prediction_indices, dtype=np.int64)
    base = np.asarray(metadata_prior, dtype=np.float32)
    if base.shape != (len(rows), 2):
        raise ValueError("metadata_prior shape does not match prediction rows")
    if float(spec.content_gate) == 0.0:
        return base.copy()
    training_videos = [int(value) for value in training_videos]
    source_curves = (
        population_curves(index, training_subjects, training_videos)
        if curves is None
        else curves
    )
    prediction = np.empty_like(base)
    for video in sorted(set(int(index.videos[row]) for row in rows)):
        local = np.flatnonzero(index.videos[rows] == video)
        video_rows = rows[local]
        length = int(index.timestamps[video_rows].max()) + 1
        sources = [
            source
            for source in training_videos
            if category_by_video[source] == category_by_video[video]
            and int(source) != int(video)
        ]
        if not sources:
            prediction[local] = base[local]
            continue
        template = content_template_curve(
            archive,
            video,
            source_curves,
            sources,
            target_length=length,
            spec=spec,
            offset_seconds=offset_seconds,
            map_cache=map_cache,
        )
        prediction[local] = blend_content_prior(
            base[local],
            template[index.timestamps[video_rows]],
            spec.content_gate,
        )
    return prediction


def cross_fitted_content_prior(
    index,
    training_subjects: Sequence[int],
    training_videos: Sequence[int],
    category_by_video: Mapping[int, str],
    archive: FoundationFeatureArchive,
    base_spec,
    axis_specs: Sequence[ContentTemplateSpec],
    *,
    duration_by_video: Mapping[int, float],
    offset_seconds: int = 0,
    map_cache: dict[tuple[object, ...], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict training rows without labels from their participant or video."""

    from merps.content_prior import predict_category_phase_prior

    subjects = np.asarray(sorted(set(int(value) for value in training_subjects)))
    videos = np.asarray(sorted(set(int(value) for value in training_videos)))
    if len(subjects) < 2 or len(videos) < 2:
        raise ValueError("Cross-fitting needs at least two participants and videos")
    if len(axis_specs) != 2:
        raise ValueError("axis_specs must contain valence and arousal specifications")
    rows_out: list[np.ndarray] = []
    prior_out: list[np.ndarray] = []
    cache = {} if map_cache is None else map_cache
    for subject in subjects:
        rows = np.flatnonzero(
            (index.subject_numbers == int(subject))
            & np.isin(index.videos, videos)
        )
        source_subjects = subjects[subjects != int(subject)]
        metadata = predict_category_phase_prior(
            index,
            source_subjects,
            videos,
            rows,
            category_by_video,
            duration_by_video=duration_by_video,
            spec=base_spec,
        )
        curves = population_curves(index, source_subjects, videos)
        combined = metadata.copy()
        predictions: dict[str, np.ndarray] = {}
        for spec in {item.name: item for item in axis_specs}.values():
            predictions[spec.name] = predict_content_template_prior(
                index,
                rows,
                metadata,
                archive,
                source_subjects,
                videos,
                category_by_video,
                spec=spec,
                offset_seconds=offset_seconds,
                curves=curves,
                map_cache=cache,
            )
        for axis, spec in enumerate(axis_specs):
            combined[:, axis] = predictions[spec.name][:, axis]
        rows_out.append(rows)
        prior_out.append(combined)
    output_rows = np.concatenate(rows_out).astype(np.int64)
    output_prior = np.concatenate(prior_out).astype(np.float32)
    order = np.argsort(output_rows)
    output_rows = output_rows[order]
    output_prior = output_prior[order]
    expected = np.flatnonzero(
        np.isin(index.subject_numbers, subjects)
        & np.isin(index.videos, videos)
    )
    if not np.array_equal(output_rows, expected):
        raise RuntimeError("Cross-fitted content rows do not match requested training block")
    if not np.isfinite(output_prior).all():
        raise RuntimeError("Cross-fitted content prior is incomplete")
    return output_rows, output_prior
