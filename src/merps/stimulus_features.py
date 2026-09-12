"""Audited, timestamp-aligned low-level audio/visual stimulus features.

Exact REFED stimulus files are not redistributed with MER-PS.  This module
therefore separates identity auditing from feature extraction.  A file with a
matching title or approximate duration is not treated as the original edit;
formal identity requires an authoritative SHA-256 in the stimulus manifest.

The extractor intentionally depends only on NumPy and the ffmpeg command-line
tools.  Its packed NPZ schema can also carry frozen CLIP/VideoMAE/audio
embeddings produced by a future backend, so downstream prior code need not
change when higher-level content representations become available.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from merps.content_prior import StimulusRecord


SCHEMA_VERSION = "merps-stimulus-features-v1"
FRAME_WIDTH = 64
FRAME_HEIGHT = 36
AUDIO_RATE = 16_000
LUMA_HISTOGRAM_BINS = 8


@dataclass(frozen=True)
class MediaProbe:
    duration_seconds: float
    width: int
    height: int
    has_video: bool
    has_audio: bool


@dataclass(frozen=True)
class VideoFeatureRecord:
    video_id: int
    features: np.ndarray
    feature_names: tuple[str, ...]
    media_sha256: str
    identity_status: str
    backend: str = "ffmpeg_low_level_v1"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def probe_media(path: str | Path, ffprobe: str = "ffprobe") -> MediaProbe:
    executable = shutil.which(ffprobe)
    if executable is None:
        raise FileNotFoundError(f"ffprobe executable not found: {ffprobe}")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height",
            "-of",
            "json",
            str(Path(path)),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    duration = float(payload.get("format", {}).get("duration", "nan"))
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"Could not determine positive media duration for {path}")
    first_video = video_streams[0] if video_streams else {}
    return MediaProbe(
        duration_seconds=duration,
        width=int(first_video.get("width", 0) or 0),
        height=int(first_video.get("height", 0) or 0),
        has_video=bool(video_streams),
        has_audio=has_audio,
    )


def audit_stimuli(
    records: Sequence[StimulusRecord],
    media_dir: str | Path,
    *,
    duration_tolerance_seconds: float = 1.0,
    ffprobe: str = "ffprobe",
) -> list[dict[str, object]]:
    """Audit presence, duration and identity without accepting weak matches."""

    media_dir = Path(media_dir)
    results: list[dict[str, object]] = []
    for record in records:
        path = media_dir / record.local_filename
        item: dict[str, object] = {
            "video_id": int(record.video_id),
            "title": record.title,
            "path": str(path),
            "present": path.is_file(),
            "manifest_access_status": record.access_status,
            "expected_sha256_available": bool(record.expected_sha256),
        }
        if not path.is_file():
            item.update(
                {
                    "status": "missing",
                    "duration_match": False,
                    "identity_verified": False,
                }
            )
            results.append(item)
            continue
        try:
            probe = probe_media(path, ffprobe=ffprobe)
            digest = sha256_file(path)
        except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            item.update(
                {
                    "status": "probe_failed",
                    "error": f"{type(error).__name__}: {error}",
                    "duration_match": False,
                    "identity_verified": False,
                }
            )
            results.append(item)
            continue
        lower = float(record.annotation_seconds) - float(duration_tolerance_seconds)
        upper = float(record.reported_duration_seconds) + float(duration_tolerance_seconds)
        duration_match = lower <= probe.duration_seconds <= upper
        hash_match = bool(record.expected_sha256) and digest == record.expected_sha256
        if record.expected_sha256 and not hash_match:
            status = "hash_mismatch"
        elif not duration_match:
            status = "duration_mismatch"
        elif hash_match:
            status = "identity_verified"
        else:
            status = "duration_only_unverified"
        item.update(
            {
                "status": status,
                "duration_seconds": probe.duration_seconds,
                "duration_match": duration_match,
                "reported_duration_seconds": int(record.reported_duration_seconds),
                "annotation_seconds": int(record.annotation_seconds),
                "sha256": digest,
                "hash_match": hash_match,
                "identity_verified": status == "identity_verified",
                "has_video": probe.has_video,
                "has_audio": probe.has_audio,
                "width": probe.width,
                "height": probe.height,
            }
        )
        results.append(item)
    return results


def visual_feature_names() -> tuple[str, ...]:
    names = [
        "visual_rgb_mean_r",
        "visual_rgb_mean_g",
        "visual_rgb_mean_b",
        "visual_rgb_std_r",
        "visual_rgb_std_g",
        "visual_rgb_std_b",
        "visual_luma_mean",
        "visual_luma_std",
        "visual_saturation_mean",
        "visual_saturation_std",
        "visual_edge_horizontal",
        "visual_edge_vertical",
        "visual_motion_mean",
        "visual_motion_std",
    ]
    names.extend(f"visual_luma_hist_{index}" for index in range(LUMA_HISTOGRAM_BINS))
    return tuple(names)


def audio_feature_names() -> tuple[str, ...]:
    return (
        "audio_rms",
        "audio_log_rms",
        "audio_peak_abs",
        "audio_zero_crossing_rate",
        "audio_spectral_centroid",
        "audio_spectral_bandwidth",
        "audio_rolloff_85",
        "audio_band_0_250",
        "audio_band_250_1000",
        "audio_band_1000_4000",
        "audio_band_4000_8000",
        "audio_spectral_flatness",
        "audio_spectral_flux",
    )


def visual_feature_vector(
    frame: np.ndarray, previous_frame: np.ndarray | None = None
) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("frame must have shape [height, width, 3]")
    if values.max(initial=0.0) > 1.0:
        values = values / 255.0
    values = np.clip(values, 0.0, 1.0)
    rgb_mean = values.mean(axis=(0, 1))
    rgb_std = values.std(axis=(0, 1))
    luma = (
        0.2126 * values[:, :, 0]
        + 0.7152 * values[:, :, 1]
        + 0.0722 * values[:, :, 2]
    )
    saturation = values.max(axis=2) - values.min(axis=2)
    horizontal = float(np.abs(np.diff(luma, axis=1)).mean()) if values.shape[1] > 1 else 0.0
    vertical = float(np.abs(np.diff(luma, axis=0)).mean()) if values.shape[0] > 1 else 0.0
    if previous_frame is None:
        motion_mean = 0.0
        motion_std = 0.0
    else:
        previous = np.asarray(previous_frame, dtype=np.float32)
        if previous.max(initial=0.0) > 1.0:
            previous = previous / 255.0
        if previous.shape != values.shape:
            raise ValueError("previous_frame must match frame shape")
        motion = np.abs(values - previous).mean(axis=2)
        motion_mean = float(motion.mean())
        motion_std = float(motion.std())
    histogram, _ = np.histogram(
        luma, bins=LUMA_HISTOGRAM_BINS, range=(0.0, 1.0)
    )
    histogram = histogram.astype(np.float32) / max(float(luma.size), 1.0)
    result = np.concatenate(
        [
            rgb_mean,
            rgb_std,
            np.asarray(
                [
                    luma.mean(),
                    luma.std(),
                    saturation.mean(),
                    saturation.std(),
                    horizontal,
                    vertical,
                    motion_mean,
                    motion_std,
                ],
                dtype=np.float32,
            ),
            histogram,
        ]
    ).astype(np.float32)
    if result.shape != (len(visual_feature_names()),):
        raise RuntimeError("Visual feature schema mismatch")
    return result


def audio_feature_vector(
    samples: np.ndarray,
    *,
    sample_rate: int = AUDIO_RATE,
    previous_spectrum: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(signal) < 2:
        signal = np.pad(signal, (0, max(0, 2 - len(signal))))
    signal = np.nan_to_num(signal, copy=False)
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal), initial=0.0))
    zero_crossing = float(np.mean((signal[:-1] * signal[1:]) < 0.0))
    window = np.hanning(len(signal)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(signal * window)).astype(np.float64)
    power = spectrum**2
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / float(sample_rate))
    total = float(power.sum()) + 1e-12
    centroid = float((frequencies * power).sum() / total)
    bandwidth = float(np.sqrt((((frequencies - centroid) ** 2) * power).sum() / total))
    cumulative = np.cumsum(power)
    rolloff_index = min(int(np.searchsorted(cumulative, 0.85 * total)), len(frequencies) - 1)
    rolloff = float(frequencies[rolloff_index])
    bands = []
    for low, high in ((0.0, 250.0), (250.0, 1000.0), (1000.0, 4000.0), (4000.0, 8000.0)):
        selected = (frequencies >= low) & (frequencies < high)
        bands.append(float(power[selected].sum() / total))
    positive = np.maximum(power[1:], 1e-12)
    flatness = float(np.exp(np.mean(np.log(positive))) / np.mean(positive)) if len(positive) else 0.0
    normalised = spectrum / (float(np.linalg.norm(spectrum)) + 1e-12)
    if previous_spectrum is None or previous_spectrum.shape != normalised.shape:
        flux = 0.0
    else:
        flux = float(np.sqrt(np.mean((normalised - previous_spectrum) ** 2)))
    nyquist = max(float(sample_rate) / 2.0, 1.0)
    result = np.asarray(
        [
            rms,
            np.log1p(rms),
            peak,
            zero_crossing,
            centroid / nyquist,
            bandwidth / nyquist,
            rolloff / nyquist,
            *bands,
            flatness,
            flux,
        ],
        dtype=np.float32,
    )
    if result.shape != (len(audio_feature_names()),):
        raise RuntimeError("Audio feature schema mismatch")
    return result, normalised.astype(np.float32)


def _decode_rgb_frames(
    path: Path,
    *,
    ffmpeg: str,
) -> np.ndarray:
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg}")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-i",
            str(path),
            "-an",
            "-vf",
            f"fps=1,scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=area,format=rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    frame_bytes = FRAME_WIDTH * FRAME_HEIGHT * 3
    count = len(completed.stdout) // frame_bytes
    if count < 1:
        raise ValueError(f"No video frames decoded from {path}")
    usable = completed.stdout[: count * frame_bytes]
    return np.frombuffer(usable, dtype=np.uint8).reshape(
        count, FRAME_HEIGHT, FRAME_WIDTH, 3
    )


def _decode_audio(path: Path, *, ffmpeg: str) -> np.ndarray:
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg}")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32)


def extract_low_level_features(
    path: str | Path,
    *,
    annotation_seconds: int,
    has_audio: bool = True,
    ffmpeg: str = "ffmpeg",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract one deterministic content vector per annotation second."""

    path = Path(path)
    length = int(annotation_seconds)
    if length < 1:
        raise ValueError("annotation_seconds must be positive")
    frames = _decode_rgb_frames(path, ffmpeg=ffmpeg)
    if len(frames) < length:
        raise ValueError(
            f"Decoded {len(frames)} visual seconds, fewer than required {length}"
        )
    frames = frames[:length]
    visual = []
    previous = None
    for frame in frames:
        visual.append(visual_feature_vector(frame, previous))
        previous = frame
    visual_values = np.stack(visual)

    audio_values = np.zeros((length, len(audio_feature_names())), dtype=np.float32)
    if has_audio:
        audio = _decode_audio(path, ffmpeg=ffmpeg)
        required = length * AUDIO_RATE
        if len(audio) < required:
            deficit = required - len(audio)
            if deficit > AUDIO_RATE:
                raise ValueError(
                    f"Decoded audio is {deficit / AUDIO_RATE:.2f}s shorter than annotations"
                )
            audio = np.pad(audio, (0, deficit))
        audio = audio[:required].reshape(length, AUDIO_RATE)
        previous_spectrum = None
        for second in range(length):
            audio_values[second], previous_spectrum = audio_feature_vector(
                audio[second], previous_spectrum=previous_spectrum
            )
    names = visual_feature_names() + audio_feature_names()
    features = np.concatenate([visual_values, audio_values], axis=1).astype(np.float32)
    if features.shape != (length, len(names)) or not np.isfinite(features).all():
        raise RuntimeError("Invalid extracted stimulus feature matrix")
    return features, names


def pack_feature_records(
    records: Sequence[VideoFeatureRecord],
) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("At least one feature record is required")
    names = records[0].feature_names
    if any(record.feature_names != names for record in records):
        raise ValueError("All feature records must share one schema")
    features = []
    video_ids = []
    timestamps = []
    for record in sorted(records, key=lambda item: item.video_id):
        values = np.asarray(record.features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("Feature record has an invalid shape")
        features.append(values)
        video_ids.extend([int(record.video_id)] * len(values))
        timestamps.extend(range(len(values)))
    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "features": np.concatenate(features, axis=0).astype(np.float32),
        "video_ids": np.asarray(video_ids, dtype=np.int16),
        "timestamps": np.asarray(timestamps, dtype=np.int16),
        "feature_names": np.asarray(names),
        "record_video_ids": np.asarray(
            [record.video_id for record in records], dtype=np.int16
        ),
        "media_sha256": np.asarray([record.media_sha256 for record in records]),
        "identity_status": np.asarray([record.identity_status for record in records]),
        "backends": np.asarray([record.backend for record in records]),
    }
