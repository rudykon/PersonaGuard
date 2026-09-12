"""Raw multi-timescale data access for CLaM-R experiments.

The established MER-PS pipeline caches handcrafted one-second statistics.
The innovation experiments instead need raw four-second EEG patches and
long, causal fNIRS windows. This module converts the large MATLAB files into
small per-trial arrays once, then exposes leakage-safe context features and
random-access windows for training.
"""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from merps.features import (
    _append_second_context,
    _eeg_segment_features,
    _label_matrix,
    _load_mat,
    _resample_eeg_segments,
    _subtract_eeg_baseline,
    _subtract_fnirs_baseline,
    _video_keys,
    discover_subjects,
)


EEG_CACHE_RATE = 200
EEG_RESAMPLE_METHODS = ("block_average", "fft_antialias")
EEG_ANTIALIAS_PASSBAND_HZ = 80.0
EEG_ANTIALIAS_STOPBAND_HZ = 100.0
FNIRS_CACHE_RATE = 4
FNIRS_SIGNAL_TYPES = ("HbO", "HbR", "HbT", "Abs780", "Abs805", "Abs830")
CONTEXT_DIM = 7
QUALITY_DIM = 9


@dataclass(frozen=True)
class InnovationIndex:
    """One row per labelled second in annotation order."""

    sample_ids: np.ndarray
    subjects: np.ndarray
    subject_numbers: np.ndarray
    videos: np.ndarray
    timestamps: np.ndarray
    targets: np.ndarray

    def indices_for_subjects(self, subjects: Sequence[int]) -> np.ndarray:
        values = np.asarray(subjects, dtype=np.int16)
        return np.flatnonzero(np.isin(self.subject_numbers, values))


def load_innovation_index(data_root: str | Path) -> InnovationIndex:
    data_root = Path(data_root)
    sample_ids: list[str] = []
    subjects: list[str] = []
    subject_numbers: list[int] = []
    videos: list[int] = []
    timestamps: list[int] = []
    targets: list[np.ndarray] = []

    for subject in discover_subjects(data_root):
        subject_number = int(subject.split("_", 1)[1])
        labels = _load_mat(data_root / "annotations" / f"{subject}_label.mat")
        for video_key in _video_keys(labels):
            video = int(video_key.split("_", 1)[1])
            trial_target = _label_matrix(labels[video_key]).T.astype(np.float32)
            for timestamp in range(len(trial_target)):
                sample_ids.append(f"{subject}_V{video:02d}_T{timestamp:03d}")
                subjects.append(subject)
                subject_numbers.append(subject_number)
                videos.append(video)
                timestamps.append(timestamp)
            targets.append(trial_target)

    if not targets:
        raise ValueError(f"No labels found under {data_root / 'annotations'}")
    return InnovationIndex(
        sample_ids=np.asarray(sample_ids),
        subjects=np.asarray(subjects),
        subject_numbers=np.asarray(subject_numbers, dtype=np.int16),
        videos=np.asarray(videos, dtype=np.int16),
        timestamps=np.asarray(timestamps, dtype=np.int16),
        targets=np.concatenate(targets, axis=0).astype(np.float32),
    )


def outer_subject_folds() -> list[tuple[np.ndarray, np.ndarray]]:
    subjects = np.arange(1, 25, dtype=np.int16)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(5):
        validation = subjects[fold::5]
        training = subjects[~np.isin(subjects, validation)]
        folds.append((training, validation))
    return folds


def grouped_subject_folds(
    values: Sequence[int], n_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.asarray(sorted(set(int(value) for value in values)), dtype=np.int16)
    n_folds = max(2, min(int(n_folds), len(unique)))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        validation = unique[fold::n_folds]
        training = unique[~np.isin(unique, validation)]
        folds.append((training, validation))
    return folds


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float32, copy=True)
    output = np.empty_like(values, dtype=np.float32)
    for timestamp in range(len(values)):
        start = max(0, timestamp - radius)
        end = min(len(values), timestamp + radius + 1)
        output[timestamp] = values[start:end].mean(axis=0)
    return output


def _context_lookup(
    index: InnovationIndex,
    source_subjects: Sequence[int],
    radius: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    source_mask = np.isin(
        index.subject_numbers, np.asarray(source_subjects, dtype=np.int16)
    )
    if not np.any(source_mask):
        raise ValueError("Context construction needs at least one source participant")

    global_target = index.targets[source_mask]
    global_prior = np.median(global_target, axis=0).astype(np.float32)
    global_uncertainty = (
        1.4826
        * np.median(np.abs(global_target - global_prior[None, :]), axis=0)
    ).astype(np.float32)
    prior_lookup: dict[int, np.ndarray] = {}
    uncertainty_lookup: dict[int, np.ndarray] = {}

    for video in sorted(set(int(value) for value in index.videos)):
        video_mask = source_mask & (index.videos == video)
        if not np.any(video_mask):
            continue
        max_timestamp = int(index.timestamps[video_mask].max())
        prior = np.empty((max_timestamp + 1, 2), dtype=np.float32)
        uncertainty = np.empty_like(prior)
        for timestamp in range(max_timestamp + 1):
            cell = video_mask & (index.timestamps == timestamp)
            values = index.targets[cell]
            median = np.median(values, axis=0)
            prior[timestamp] = median
            uncertainty[timestamp] = (
                1.4826 * np.median(np.abs(values - median[None, :]), axis=0)
            )
        prior_lookup[video] = _smooth(prior, radius)
        uncertainty_lookup[video] = _smooth(uncertainty, radius)
    return prior_lookup, uncertainty_lookup, global_prior, global_uncertainty


def build_context_features(
    index: InnovationIndex,
    source_subjects: Sequence[int],
    prediction_indices: np.ndarray,
    *,
    radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw prior predictions and normalized context tokens.

    Context order is prior[2], uncertainty[2], prior_gradient[2], progress.
    Every statistic is constructed only from source_subjects.
    """

    prediction_indices = np.asarray(prediction_indices, dtype=np.int64)
    prior_lookup, uncertainty_lookup, global_prior, global_uncertainty = (
        _context_lookup(index, source_subjects, radius)
    )
    priors = np.empty((len(prediction_indices), 2), dtype=np.float32)
    context = np.empty((len(prediction_indices), CONTEXT_DIM), dtype=np.float32)
    gradients = {
        video: np.gradient(values, axis=0).astype(np.float32)
        for video, values in prior_lookup.items()
    }

    for local, row in enumerate(prediction_indices):
        video = int(index.videos[row])
        timestamp = int(index.timestamps[row])
        if video in prior_lookup:
            trajectory = prior_lookup[video]
            position = min(max(timestamp, 0), len(trajectory) - 1)
            prior = trajectory[position]
            uncertainty = uncertainty_lookup[video][position]
            gradient = gradients[video][position]
            progress = position / max(len(trajectory) - 1, 1)
        else:
            prior = global_prior
            uncertainty = global_uncertainty
            gradient = np.zeros(2, dtype=np.float32)
            progress = 0.0
        priors[local] = prior
        context[local] = np.concatenate(
            [
                np.clip((prior - 128.0) / 127.0, -1.0, 1.0),
                np.clip(uncertainty / 64.0, 0.0, 4.0),
                np.clip(gradient / 32.0, -4.0, 4.0),
                np.asarray([progress], dtype=np.float32),
            ]
        )
    return np.clip(priors, 1.0, 255.0), context


def build_cross_fitted_context(
    index: InnovationIndex,
    training_subjects: Sequence[int],
    training_indices: np.ndarray,
    *,
    radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct training context without using each participant's own labels."""

    training_indices = np.asarray(training_indices, dtype=np.int64)
    priors = np.empty((len(training_indices), 2), dtype=np.float32)
    context = np.empty((len(training_indices), CONTEXT_DIM), dtype=np.float32)
    training_subjects = [int(value) for value in training_subjects]
    for subject in training_subjects:
        local = np.flatnonzero(
            index.subject_numbers[training_indices] == subject
        )
        if not local.size:
            continue
        source_subjects = [value for value in training_subjects if value != subject]
        subject_prior, subject_context = build_context_features(
            index,
            source_subjects,
            training_indices[local],
            radius=radius,
        )
        priors[local] = subject_prior
        context[local] = subject_context
    return priors, context


def load_reservation_masks(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    masks: dict[str, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        channel_fields = [f"ch-{index}" for index in range(1, 52)]
        for row in reader:
            subject = str(row.get("sub_id", "")).strip()
            if not subject:
                continue
            masks[subject] = np.asarray(
                [float(row.get(field, 0) or 0) for field in channel_fields],
                dtype=np.float32,
            )
    if not masks:
        raise ValueError(f"No reservation masks found in {path}")
    return masks


def _fft_antialias_decimate(
    eeg: np.ndarray,
    source_rate: int,
    target_rate: int,
    *,
    passband_hz: float = EEG_ANTIALIAS_PASSBAND_HZ,
    stopband_hz: float = EEG_ANTIALIAS_STOPBAND_HZ,
    channel_chunk: int = 16,
) -> np.ndarray:
    """Zero-phase FFT low-pass followed by integer decimation.

    A two-second reflection pad limits trial-edge wraparound. Frequencies at
    and above the target Nyquist are removed, with a raised-cosine transition
    from passband_hz to stopband_hz. This is the strict anti-alias sensitivity
    path; the historical cache remains available as ``block_average``.
    """

    values = np.asarray(eeg, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected EEG [channels, samples], got {values.shape}")
    if source_rate % target_rate:
        raise ValueError("FFT anti-alias decimation needs an integer rate ratio")
    nyquist = target_rate / 2.0
    if not 0.0 < passband_hz < stopband_hz <= nyquist:
        raise ValueError("Invalid anti-alias passband/stopband")

    factor = source_rate // target_rate
    pad = min(2 * source_rate, values.shape[-1] - 1)
    padded_length = values.shape[-1] + 2 * pad
    frequencies = np.fft.rfftfreq(padded_length, d=1.0 / source_rate)
    response = np.ones_like(frequencies, dtype=np.float64)
    transition = (frequencies > passband_hz) & (frequencies < stopband_hz)
    phase = (frequencies[transition] - passband_hz) / (
        stopband_hz - passband_hz
    )
    response[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    response[frequencies >= stopband_hz] = 0.0

    target_length = values.shape[-1] // factor
    output = np.empty((values.shape[0], target_length), dtype=np.float32)
    for start in range(0, values.shape[0], channel_chunk):
        end = min(start + channel_chunk, values.shape[0])
        padded = np.pad(
            values[start:end], ((0, 0), (pad, pad)), mode="reflect"
        )
        spectrum = np.fft.rfft(padded, axis=-1)
        filtered = np.fft.irfft(
            spectrum * response[None, :], n=padded_length, axis=-1
        )
        output[start:end] = filtered[
            :, pad : pad + values.shape[-1] : factor
        ].astype(np.float32)
    return output


def _resample_eeg_trial(
    eeg: np.ndarray,
    n_seconds: int,
    *,
    method: str = "block_average",
) -> np.ndarray:
    channels, n_samples = eeg.shape
    source_rate = int(round(n_samples / float(n_seconds)))
    if source_rate * n_seconds != n_samples:
        raise ValueError(
            f"EEG length {n_samples} is not an integer rate for {n_seconds} seconds"
        )
    if method not in EEG_RESAMPLE_METHODS:
        raise ValueError(
            f"Unknown EEG resampling method {method!r}; expected one of "
            f"{EEG_RESAMPLE_METHODS}"
        )
    if method == "block_average":
        segments = eeg.reshape(channels, n_seconds, source_rate).transpose(
            1, 0, 2
        )
        return _resample_eeg_segments(
            segments, float(source_rate)
        ).astype(np.float32)

    downsampled = _fft_antialias_decimate(
        eeg, source_rate, EEG_CACHE_RATE
    )
    expected = n_seconds * EEG_CACHE_RATE
    if downsampled.shape[-1] != expected:
        raise RuntimeError(
            f"Anti-aliased EEG length {downsampled.shape[-1]} != {expected}"
        )
    return downsampled.reshape(
        channels, n_seconds, EEG_CACHE_RATE
    ).transpose(1, 0, 2)


def _resample_fnirs_trial(
    fnirs: np.ndarray, n_seconds: int, target_rate: int
) -> np.ndarray:
    types, channels, n_samples = fnirs.shape
    target_length = n_seconds * target_rate
    positions = np.linspace(
        0.0, n_samples - 1.0, target_length, dtype=np.float64
    )
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, n_samples - 1)
    weight = (positions - left).astype(np.float32)
    flat = fnirs.reshape(types * channels, n_samples).astype(np.float32)
    sampled = (
        flat[:, left] * (1.0 - weight[None, :])
        + flat[:, right] * weight[None, :]
    )
    return sampled.reshape(
        types, channels, n_seconds, target_rate
    ).transpose(2, 0, 1, 3)


def _eeg_quality(eeg: np.ndarray) -> np.ndarray:
    """Compute five scale-robust one-second EEG quality summaries."""

    centered = eeg - eeg.mean(axis=-1, keepdims=True)
    standard_deviation = centered.std(axis=-1)
    peak_to_peak = np.ptp(centered, axis=-1)
    derivative_rms = np.sqrt(
        np.mean(np.diff(centered, axis=-1) ** 2, axis=-1) + 1e-8
    )
    spectrum = np.abs(np.fft.rfft(centered, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(
        centered.shape[-1], d=1.0 / EEG_CACHE_RATE
    )
    broad = spectrum[
        ..., (frequencies >= 1.0) & (frequencies <= 45.0)
    ].sum(axis=-1) + 1e-8
    line = spectrum[
        ..., (frequencies >= 49.0) & (frequencies <= 51.0)
    ].sum(axis=-1)
    line_ratio = np.median(line / broad, axis=1)
    flat_ratio = np.mean(standard_deviation < 0.1, axis=1)
    bad_ratio = np.mean(
        (standard_deviation < 0.1) | (peak_to_peak > 500.0), axis=1
    )
    return np.stack(
        [
            np.log1p(np.median(peak_to_peak, axis=1)),
            flat_ratio,
            np.log1p(np.median(derivative_rms, axis=1)),
            np.log1p(line_ratio * 1000.0),
            bad_ratio,
        ],
        axis=1,
    ).astype(np.float32)


def _fnirs_quality(
    fnirs: np.ndarray, reservation: np.ndarray
) -> np.ndarray:
    """Compute four one-second fNIRS quality summaries."""

    values = np.asarray(fnirs, dtype=np.float32)
    valid = reservation.astype(bool)[None, None, :, None]
    flattened = values.transpose(1, 2, 0, 3).reshape(6, 51, -1)
    median = np.median(flattened, axis=-1, keepdims=True)
    mad = (
        1.4826
        * np.median(np.abs(flattened - median), axis=-1, keepdims=True)
        + 1e-6
    )
    z = np.abs(
        (values - median[None, :, :, :])
        / mad[None, :, :, :]
    )
    valid_fraction = max(float(reservation.mean()), 1e-6)
    abnormal = (
        np.mean((z > 8.0) & valid, axis=(1, 2, 3)) / valid_fraction
    )

    differences = np.abs(np.diff(flattened, axis=-1))
    diff_median = np.median(differences, axis=-1, keepdims=True)
    diff_mad = (
        1.4826
        * np.median(
            np.abs(differences - diff_median), axis=-1, keepdims=True
        )
        + 1e-6
    )
    threshold = diff_median + 6.0 * diff_mad
    local_differences = np.abs(np.diff(values, axis=-1))
    motion = (
        np.mean(
            (local_differences > threshold[None, :, :, :]) & valid,
            axis=(1, 2, 3),
        )
        / valid_fraction
    )
    local_std = values.std(axis=-1)
    flat_ratio = (
        np.mean(
            (local_std < 1e-8) & reservation[None, None, :].astype(bool),
            axis=(1, 2),
        )
        / valid_fraction
    )
    valid_ratio = np.full(len(values), reservation.mean(), dtype=np.float32)
    return np.stack(
        [valid_ratio, motion, abnormal, flat_ratio], axis=1
    ).astype(np.float32)


def _trial_stem(subject: str, video: int) -> str:
    return f"{subject}_V{video:02d}"


def build_signal_cache(
    data_root: str | Path,
    cache_dir: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    eeg_rate: int = EEG_CACHE_RATE,
    fnirs_rate: int = FNIRS_CACHE_RATE,
    eeg_resample_method: str = "block_average",
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    """Build baseline-corrected, downsampled raw trial caches.

    EEG is stored in microvolts as [second, channel, 200 samples]. fNIRS is
    stored as [second, signal_type, channel, 4 samples]. Float16 storage keeps
    the complete cache close to one gigabyte while model input is cast back to
    float32.
    """

    if eeg_rate != EEG_CACHE_RATE:
        raise ValueError(
            f"The CBraMod cache rate must be {EEG_CACHE_RATE} Hz"
        )
    data_root = Path(data_root)
    if eeg_resample_method not in EEG_RESAMPLE_METHODS:
        raise ValueError(
            f"Unknown EEG resampling method {eeg_resample_method!r}"
        )
    cache_dir = Path(cache_dir)
    trial_dir = cache_dir / "trials"
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            cached_method = manifest.get("eeg", {}).get(
                "resampling_method", "block_average"
            )
            if cached_method != eeg_resample_method:
                raise ValueError(
                    f"Cache uses EEG resampling {cached_method!r}, requested "
                    f"{eeg_resample_method!r}; use a separate cache or --overwrite"
                )
            return manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    trial_dir.mkdir(parents=True, exist_ok=True)
    subject_names = (
        list(subjects)
        if subjects is not None
        else discover_subjects(data_root)
    )
    reservations = load_reservation_masks(
        data_root / "fNIRS_reservations.csv"
    )
    trial_count = 0
    sample_count = 0

    for subject_index, subject in enumerate(subject_names, start=1):
        if verbose:
            print(
                f"[innovation-cache] subject "
                f"{subject_index}/{len(subject_names)}: {subject}",
                flush=True,
            )
        subject_dir = data_root / "data" / subject
        labels = _load_mat(
            data_root / "annotations" / f"{subject}_label.mat"
        )
        eeg_videos = _load_mat(subject_dir / "EEG_videos.mat")
        fnirs_videos = _load_mat(subject_dir / "fNIRS_videos.mat")
        eeg_baselines = _load_mat(subject_dir / "EEG_baselines.mat")
        fnirs_baselines = _load_mat(subject_dir / "fNIRS_baselines.mat")
        reservation = reservations[subject]

        for video_key in _video_keys(labels):
            video = int(video_key.split("_", 1)[1])
            stem = _trial_stem(subject, video)
            paths = {
                "eeg": trial_dir / f"{stem}_eeg.npy",
                "fnirs": trial_dir / f"{stem}_fnirs.npy",
                "quality": trial_dir / f"{stem}_quality.npy",
            }
            n_seconds = int(
                _label_matrix(labels[video_key]).shape[1]
            )
            if (
                all(path.exists() for path in paths.values())
                and not overwrite
            ):
                trial_count += 1
                sample_count += n_seconds
                continue

            eeg = _subtract_eeg_baseline(
                np.asarray(eeg_videos[video_key], dtype=np.float32),
                eeg_baselines.get(video_key),
            )
            fnirs = _subtract_fnirs_baseline(
                np.asarray(fnirs_videos[video_key], dtype=np.float32),
                fnirs_baselines.get(video_key),
            )
            eeg_seconds = _resample_eeg_trial(
                eeg,
                n_seconds,
                method=eeg_resample_method,
            ) * 1_000_000.0
            fnirs_seconds = _resample_fnirs_trial(
                fnirs, n_seconds, fnirs_rate
            )
            quality = np.concatenate(
                [
                    _eeg_quality(eeg_seconds),
                    _fnirs_quality(fnirs_seconds, reservation),
                ],
                axis=1,
            )
            np.save(paths["eeg"], eeg_seconds.astype(np.float16))
            np.save(paths["fnirs"], fnirs_seconds.astype(np.float16))
            np.save(paths["quality"], quality.astype(np.float32))
            trial_count += 1
            sample_count += n_seconds

    np.save(
        cache_dir / "reservation_masks.npy",
        np.stack([reservations[name] for name in subject_names]),
    )
    (cache_dir / "subjects.txt").write_text(
        "\n".join(subject_names) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "subjects": subject_names,
        "trials": int(trial_count),
        "samples": int(sample_count),
        "eeg": {
            "rate_hz": int(eeg_rate),
            "channels": 64,
            "storage": "float16_microvolts_by_second",
            "resampling_method": eeg_resample_method,
            "source_rate_inferred_per_trial": True,
            "anti_alias_passband_hz": (
                EEG_ANTIALIAS_PASSBAND_HZ
                if eeg_resample_method == "fft_antialias"
                else None
            ),
            "anti_alias_stopband_hz": (
                EEG_ANTIALIAS_STOPBAND_HZ
                if eeg_resample_method == "fft_antialias"
                else None
            ),
        },
        "fnirs": {
            "rate_hz": int(fnirs_rate),
            "channels": 51,
            "signal_types": list(FNIRS_SIGNAL_TYPES),
            "storage": "float16_baseline_corrected_by_second",
        },
        "quality_dim": QUALITY_DIM,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_handcrafted_eeg_pool(
    index: InnovationIndex,
    cache_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Rebuild the 90-D pooled handcrafted EEG block from a signal cache."""

    cache_dir = Path(cache_dir)
    output_path = Path(
        output_path or cache_dir / "handcrafted_eeg_pooled.npy"
    )
    if output_path.exists() and not overwrite:
        existing = np.load(output_path, mmap_mode="r", allow_pickle=False)
        if existing.shape != (len(index.targets), 90):
            raise ValueError(
                f"Unexpected pooled EEG shape at {output_path}: {existing.shape}"
            )
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pooled = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(index.targets), 90),
    )
    store = SignalWindowStore(cache_dir)
    subjects = sorted(set(int(value) for value in index.subject_numbers))
    for subject_position, subject_number in enumerate(subjects, start=1):
        subject = f"test_{subject_number}"
        if verbose:
            print(
                f"[handcrafted-eeg] subject {subject_position}/{len(subjects)}: "
                f"{subject}",
                flush=True,
            )
        videos = sorted(
            set(int(value) for value in index.videos[index.subjects == subject])
        )
        for video in videos:
            rows = np.flatnonzero(
                (index.subjects == subject) & (index.videos == video)
            )
            rows = rows[np.argsort(index.timestamps[rows])]
            eeg = np.asarray(
                store._array(subject, video, "eeg"), dtype=np.float32
            ) / 1_000_000.0
            if len(eeg) != len(rows):
                raise ValueError(
                    f"EEG/annotation length mismatch for {subject} V{video:02d}"
                )
            features = _append_second_context(
                _eeg_segment_features(eeg), radius=1
            )
            pooled[rows, :45] = features.mean(axis=1)
            pooled[rows, 45:] = features.std(axis=1)
    pooled.flush()

    signal_manifest_path = cache_dir / "manifest.json"
    signal_manifest = (
        json.loads(signal_manifest_path.read_text(encoding="utf-8"))
        if signal_manifest_path.exists()
        else {}
    )
    derived_manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_signal_cache": str(cache_dir),
        "source_eeg_resampling": signal_manifest.get("eeg", {}).get(
            "resampling_method", "unknown"
        ),
        "samples": int(len(index.targets)),
        "shape": [int(len(index.targets)), 90],
        "definition": (
            "microvolts converted back to volts; one-second 15-D per-channel "
            "spectral/Hjorth features with "
            "radius-one temporal context, pooled as channel mean and std"
        ),
    }
    output_path.with_name(output_path.stem + "_manifest.json").write_text(
        json.dumps(derived_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


class SignalWindowStore:
    """LRU-backed random access to cached multi-timescale trial windows."""

    def __init__(
        self, cache_dir: str | Path, *, max_open_trials: int = 48
    ):
        self.cache_dir = Path(cache_dir)
        self.trial_dir = self.cache_dir / "trials"
        self.max_open_trials = int(max_open_trials)
        self._arrays: OrderedDict[
            tuple[str, int, str], np.ndarray
        ] = OrderedDict()
        subjects = (
            self.cache_dir / "subjects.txt"
        ).read_text(encoding="utf-8").strip().splitlines()
        masks = np.load(
            self.cache_dir / "reservation_masks.npy",
            allow_pickle=False,
        )
        self.reservation_masks = {
            subject: masks[index].astype(np.float32)
            for index, subject in enumerate(subjects)
        }

    def _array(
        self, subject: str, video: int, kind: str
    ) -> np.ndarray:
        key = (subject, int(video), kind)
        if key in self._arrays:
            self._arrays.move_to_end(key)
            return self._arrays[key]
        path = (
            self.trial_dir
            / f"{_trial_stem(subject, int(video))}_{kind}.npy"
        )
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        self._arrays[key] = array
        while len(self._arrays) > self.max_open_trials:
            self._arrays.popitem(last=False)
        return array

    @staticmethod
    def _window_indices(
        timestamp: int,
        length: int,
        seconds: int,
        *,
        causal: bool,
    ) -> np.ndarray:
        start = (
            timestamp - seconds + 1
            if causal
            else timestamp - seconds // 2
        )
        return np.clip(
            np.arange(start, start + seconds), 0, length - 1
        )

    def eeg_window(
        self,
        subject: str,
        video: int,
        timestamp: int,
        *,
        seconds: int = 4,
        causal: bool = False,
    ) -> np.ndarray:
        eeg = self._array(subject, video, "eeg")
        rows = self._window_indices(
            timestamp, len(eeg), seconds, causal=causal
        )
        return np.asarray(
            eeg[rows], dtype=np.float32
        ).transpose(1, 0, 2)

    def fnirs_window(
        self,
        subject: str,
        video: int,
        timestamp: int,
        *,
        seconds: int = 24,
        causal: bool = True,
    ) -> np.ndarray:
        fnirs = self._array(subject, video, "fnirs")
        rows = self._window_indices(
            timestamp, len(fnirs), seconds, causal=causal
        )
        selected = np.asarray(fnirs[rows], dtype=np.float32)
        return selected.transpose(0, 3, 1, 2).reshape(
            seconds * FNIRS_CACHE_RATE, 6, 51
        )

    def quality_features(
        self,
        subject: str,
        video: int,
        timestamp: int,
        *,
        eeg_seconds: int = 4,
        fnirs_seconds: int = 24,
    ) -> np.ndarray:
        quality = self._array(subject, video, "quality")
        eeg_rows = self._window_indices(
            timestamp, len(quality), eeg_seconds, causal=False
        )
        fnirs_rows = self._window_indices(
            timestamp, len(quality), fnirs_seconds, causal=True
        )
        return np.concatenate(
            [
                np.asarray(
                    quality[eeg_rows, :5], dtype=np.float32
                ).mean(axis=0),
                np.asarray(
                    quality[fnirs_rows, 5:], dtype=np.float32
                ).mean(axis=0),
            ]
        ).astype(np.float32)

    def reservation_mask(self, subject: str) -> np.ndarray:
        return self.reservation_masks[subject].copy()


def trial_chunks(
    index: InnovationIndex,
    sample_indices: np.ndarray,
    *,
    chunk_seconds: int = 32,
    stride: int | None = None,
    drop_short: bool = False,
) -> list[np.ndarray]:
    """Partition participant-video trials into contiguous label chunks."""

    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    available = np.zeros(len(index.targets), dtype=bool)
    available[sample_indices] = True
    stride = int(chunk_seconds if stride is None else stride)
    chunks: list[np.ndarray] = []
    subjects = sorted(
        set(int(value) for value in index.subject_numbers[sample_indices])
    )
    videos = sorted(
        set(int(value) for value in index.videos[sample_indices])
    )
    for subject in subjects:
        for video in videos:
            rows = np.flatnonzero(
                available
                & (index.subject_numbers == subject)
                & (index.videos == video)
            )
            if not len(rows):
                continue
            rows = rows[np.argsort(index.timestamps[rows])]
            for start in range(0, len(rows), stride):
                chunk = rows[start : start + chunk_seconds]
                if len(chunk) < chunk_seconds and drop_short:
                    continue
                chunks.append(chunk)
    return chunks
