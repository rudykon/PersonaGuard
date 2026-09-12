"""MATLAB parsing and physiological features used by the final analyses."""


from __future__ import annotations
import re
import struct
import zlib
from pathlib import Path
import numpy as np


EEG_TARGET_RATE = 200


EEG_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta_low", 13.0, 20.0),
    ("beta_high", 20.0, 30.0),
    ("gamma", 30.0, 45.0),
)


CONTEXT_RADIUS_SECONDS = 1


EPS = 1e-12


VIDEO_RE = re.compile(r"^video_(\d+)$")


SUBJECT_RE = re.compile(r"^test_(\d+)$")


MI_INT8 = 1


MI_UINT8 = 2


MI_INT16 = 3


MI_UINT16 = 4


MI_INT32 = 5


MI_UINT32 = 6


MI_SINGLE = 7


MI_DOUBLE = 9


MI_INT64 = 12


MI_UINT64 = 13


MI_MATRIX = 14


MI_COMPRESSED = 15


MI_TO_DTYPE = {
    MI_INT8: np.dtype("i1"),
    MI_UINT8: np.dtype("u1"),
    MI_INT16: np.dtype("<i2"),
    MI_UINT16: np.dtype("<u2"),
    MI_INT32: np.dtype("<i4"),
    MI_UINT32: np.dtype("<u4"),
    MI_SINGLE: np.dtype("<f4"),
    MI_DOUBLE: np.dtype("<f8"),
    MI_INT64: np.dtype("<i8"),
    MI_UINT64: np.dtype("<u8"),
}


def discover_subjects(data_root: str | Path) -> list[str]:
    data_dir = Path(data_root) / "data"
    subjects = [
        path.name
        for path in data_dir.iterdir()
        if path.is_dir() and SUBJECT_RE.fullmatch(path.name)
    ]
    return sorted(subjects, key=lambda item: int(item.split("_", 1)[1]))


def _load_mat(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_mat_v5(path)


def read_mat_v5(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        header = handle.read(128)
        if len(header) != 128 or b"MATLAB 5.0 MAT-file" not in header[:116]:
            raise ValueError(f"Unsupported .mat file format: {path}")
        payload = handle.read()

    arrays: dict[str, np.ndarray] = {}
    for data_type, data in _iter_elements(payload):
        if data_type == MI_COMPRESSED:
            for inner_type, inner_data in _iter_elements(zlib.decompress(data)):
                if inner_type == MI_MATRIX:
                    name, array = _parse_matrix(inner_data)
                    arrays[name] = array
        elif data_type == MI_MATRIX:
            name, array = _parse_matrix(data)
            arrays[name] = array
    return arrays


def _iter_elements(buffer: bytes):
    offset = 0
    size = len(buffer)
    while offset + 8 <= size:
        data_type, data, offset = _read_element(buffer, offset)
        if data_type == 0 and len(data) == 0:
            break
        yield data_type, data


def _read_element(buffer: bytes, offset: int) -> tuple[int, bytes, int]:
    raw = struct.unpack_from("<I", buffer, offset)[0]
    small_nbytes = raw >> 16
    if small_nbytes:
        data_type = raw & 0xFFFF
        data_start = offset + 4
        data_end = data_start + small_nbytes
        return data_type, buffer[data_start:data_end], offset + 8

    data_type, nbytes = struct.unpack_from("<II", buffer, offset)
    data_start = offset + 8
    data_end = data_start + nbytes
    next_offset = data_end + ((8 - (nbytes % 8)) % 8)
    return data_type, buffer[data_start:data_end], next_offset


def _parse_matrix(data: bytes) -> tuple[str, np.ndarray]:
    offset = 0
    _, _, offset = _read_element(data, offset)

    dim_type, dim_data, offset = _read_element(data, offset)
    if dim_type not in (MI_INT32, MI_UINT32):
        raise ValueError("Unsupported MATLAB dimension element")
    dims = np.frombuffer(dim_data, dtype=MI_TO_DTYPE[dim_type]).astype(np.int64)

    name_type, name_data, offset = _read_element(data, offset)
    if name_type not in (MI_INT8, MI_UINT8):
        raise ValueError("Unsupported MATLAB variable name element")
    name = bytes(name_data).decode("utf-8").rstrip("\x00")

    real_type, real_data, _ = _read_element(data, offset)
    if real_type not in MI_TO_DTYPE:
        raise ValueError(f"Unsupported MATLAB numeric type: {real_type}")
    dtype = MI_TO_DTYPE[real_type]
    array = np.frombuffer(real_data, dtype=dtype)
    if dims.size:
        array = array.reshape(tuple(int(dim) for dim in dims), order="F")
    return name, array


def _video_keys(mat: dict[str, np.ndarray]) -> list[str]:
    keys = [key for key in mat if VIDEO_RE.fullmatch(key)]
    return sorted(keys, key=lambda item: int(item.split("_", 1)[1]))


def _label_matrix(value: np.ndarray) -> np.ndarray:
    label = np.asarray(value, dtype=np.float32)
    if label.ndim != 2:
        raise ValueError(f"Expected label matrix with 2 dims, got shape {label.shape}")
    if label.shape[0] != 2 and label.shape[1] == 2:
        label = label.T
    if label.shape[0] != 2:
        raise ValueError(f"Expected label shape [2, time], got {label.shape}")
    return label


def _subtract_eeg_baseline(eeg: np.ndarray, baseline: np.ndarray | None) -> np.ndarray:
    if baseline is None:
        return eeg
    base = np.asarray(baseline, dtype=np.float32)
    if base.ndim == 2 and base.shape[0] == eeg.shape[0]:
        return eeg - base.mean(axis=1, keepdims=True)
    return eeg


def _subtract_fnirs_baseline(fnirs: np.ndarray, baseline: np.ndarray | None) -> np.ndarray:
    if baseline is None:
        return fnirs
    base = np.asarray(baseline, dtype=np.float32)
    if base.ndim == 3 and base.shape[:2] == fnirs.shape[:2]:
        return fnirs - base.mean(axis=2, keepdims=True)
    return fnirs


def _resample_eeg_segments(segments: np.ndarray, source_rate: float) -> np.ndarray:
    length = segments.shape[-1]
    if length == EEG_TARGET_RATE:
        return np.asarray(segments, dtype=np.float32)

    ratio = length / float(EEG_TARGET_RATE)
    if abs(ratio - round(ratio)) < 1e-6 and int(round(ratio)) > 1:
        factor = int(round(ratio))
        usable = EEG_TARGET_RATE * factor
        return segments[..., :usable].reshape(*segments.shape[:-1], EEG_TARGET_RATE, factor).mean(axis=-1).astype(np.float32)

    old_x = np.linspace(0.0, 1.0, num=length, endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=EEG_TARGET_RATE, endpoint=False)
    flat = segments.reshape(-1, length)
    out = np.empty((flat.shape[0], EEG_TARGET_RATE), dtype=np.float32)
    for idx, row in enumerate(flat):
        out[idx] = np.interp(new_x, old_x, row).astype(np.float32)
    return out.reshape(*segments.shape[:-1], EEG_TARGET_RATE)


def _eeg_segment_features(segments: np.ndarray) -> np.ndarray:
    segments = np.asarray(segments, dtype=np.float32)
    if segments.ndim == 2:
        segments = segments[None, :, :]
    centered = segments - segments.mean(axis=-1, keepdims=True)
    length = centered.shape[-1]
    if length < 3:
        return np.zeros((*centered.shape[:2], len(EEG_BANDS) * 2 + 3), dtype=np.float32)

    bandpower, de = _eeg_frequency_features(centered, sampling_rate=float(EEG_TARGET_RATE))
    hjorth = _hjorth_parameters(centered)
    return np.concatenate([bandpower, de, hjorth], axis=-1).astype(np.float32)


def _eeg_frequency_features(
    centered: np.ndarray,
    sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    length = centered.shape[-1]
    window = np.hanning(length).astype(np.float32)
    windowed_spectrum = np.abs(np.fft.rfft(centered * window, axis=-1)) ** 2
    raw_spectrum = np.fft.rfft(centered, axis=-1)
    freqs = np.fft.rfftfreq(length, d=1.0 / sampling_rate)

    valid = (freqs >= EEG_BANDS[0][1]) & (freqs < EEG_BANDS[-1][2])
    total_power = windowed_spectrum[..., valid].sum(axis=-1) + EPS

    bandpower_parts = []
    de_parts = []
    for _, low, high in EEG_BANDS:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            power = np.zeros_like(total_power)
            variance = np.full_like(total_power, EPS)
        else:
            power = windowed_spectrum[..., mask].sum(axis=-1)
            band_fft = np.zeros_like(raw_spectrum)
            band_fft[..., mask] = raw_spectrum[..., mask]
            band_signal = np.fft.irfft(band_fft, n=length, axis=-1)
            variance = np.var(band_signal, axis=-1)
        bandpower_parts.append(np.log(np.maximum(power / total_power, EPS)))
        de_parts.append(0.5 * np.log(2.0 * np.pi * np.e * np.maximum(variance, EPS)))

    return (
        np.stack(bandpower_parts, axis=-1).astype(np.float32),
        np.stack(de_parts, axis=-1).astype(np.float32),
    )


def _hjorth_parameters(segments: np.ndarray) -> np.ndarray:
    activity = np.var(segments, axis=-1)
    diff1 = np.diff(segments, axis=-1)
    diff2 = np.diff(diff1, axis=-1)
    var_diff1 = np.var(diff1, axis=-1)
    var_diff2 = np.var(diff2, axis=-1)
    mobility = np.sqrt(var_diff1 / np.maximum(activity, EPS))
    mobility_diff = np.sqrt(var_diff2 / np.maximum(var_diff1, EPS))
    complexity = mobility_diff / np.maximum(mobility, EPS)
    return np.stack([activity, mobility, complexity], axis=-1).astype(np.float32)


def _fnirs_segment_features(segment: np.ndarray) -> np.ndarray:
    types, channels, length = segment.shape
    mean = segment.mean(axis=-1)
    centered = segment - mean[..., None]
    std = segment.std(axis=-1)
    safe_std = np.maximum(std, 1e-6)

    if length >= 2:
        t = np.arange(length, dtype=np.float32)
        t -= t.mean()
        slope = (centered * t).sum(axis=-1) / np.maximum(float((t * t).sum()), EPS)
    else:
        slope = np.zeros((types, channels), dtype=np.float32)

    z = centered / safe_std[..., None]
    skewness = np.mean(z**3, axis=-1)
    kurtosis = np.mean(z**4, axis=-1) - 3.0
    constant = std < 1e-6
    skewness = np.where(constant, 0.0, skewness)
    kurtosis = np.where(constant, 0.0, kurtosis)

    parts = [mean, std, slope, skewness, kurtosis]
    return np.concatenate([part.T for part in parts], axis=1).astype(np.float32)


def _append_second_context(features: np.ndarray, radius: int = CONTEXT_RADIUS_SECONDS) -> np.ndarray:
    if radius <= 0:
        return np.asarray(features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(f"Expected [seconds, channels, features], got {features.shape}")

    padded = np.pad(features, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    n = features.shape[0]
    windows = [padded[offset : offset + n] for offset in range(2 * radius + 1)]
    return np.concatenate(windows, axis=-1).astype(np.float32)
