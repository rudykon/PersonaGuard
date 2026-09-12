"""Causal physiology features and conservative residual regression.

The original handcrafted MER-PS cache concatenates ``t-1, t, t+1``.  That is
useful offline but violates a real-time claim.  This module rebuilds the same
feature families with ``t-2, t-1, t`` only and provides a small Ridge residual
model whose correction can be gated all the way to exactly zero.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from merps.features import _eeg_segment_features, _fnirs_segment_features
from merps.innovation.data import InnovationIndex, SignalWindowStore


@dataclass(frozen=True)
class ResidualRidgeSpec:
    modality: str
    alpha: float
    gate: float
    cap: float
    ema_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.modality not in {"disabled", "eeg", "fnirs", "fusion"}:
            raise ValueError(f"Unsupported modality: {self.modality}")
        if float(self.alpha) < 0.0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= float(self.gate) <= 1.0:
            raise ValueError("gate must be in [0, 1]")
        if float(self.cap) < 0.0:
            raise ValueError("cap must be non-negative")
        if not 0.0 <= float(self.ema_decay) < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.modality == "disabled" and (self.gate != 0.0 or self.cap != 0.0):
            raise ValueError("disabled residual must have zero gate and cap")
        if self.modality != "disabled" and (self.gate <= 0.0 or self.cap <= 0.0):
            raise ValueError("enabled residual needs positive gate and cap")

    @property
    def name(self) -> str:
        if self.modality == "disabled":
            return "disabled"
        return (
            f"{self.modality}_a{self.alpha:g}_g{self.gate:g}"
            f"_c{self.cap:g}_ema{self.ema_decay:g}"
        )


@dataclass(frozen=True)
class StandardizedRidge:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    coefficient: np.ndarray
    clip_z: float = 8.0

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        standardized = np.clip(
            (values - self.x_mean) / self.x_scale,
            -float(self.clip_z),
            float(self.clip_z),
        )
        return np.asarray(
            standardized @ self.coefficient + self.y_mean, dtype=np.float32
        )


def causal_context(features: np.ndarray, lookback: int = 2) -> np.ndarray:
    """Concatenate current and past seconds; never read a future second."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("features must have shape [seconds, channels, features]")
    if int(lookback) < 0:
        raise ValueError("lookback must be non-negative")
    indices = np.arange(len(values), dtype=np.int64)
    pieces = [values[np.maximum(indices - lag, 0)] for lag in range(lookback, -1, -1)]
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def _manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_causal_physio_pools(
    index: InnovationIndex,
    signal_cache: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """Build causal EEG/fNIRS features from baseline-corrected signal shards."""

    signal_cache = Path(signal_cache)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eeg_path = output_dir / "causal_eeg_pooled.npy"
    fnirs_path = output_dir / "causal_fnirs_pooled.npy"
    manifest_path = output_dir / "manifest.json"
    expected_shapes = {
        eeg_path: (len(index.targets), 90),
        fnirs_path: (len(index.targets), 180),
    }
    if not overwrite and all(path.exists() for path in expected_shapes):
        for path, shape in expected_shapes.items():
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(f"Invalid causal pool at {path}: {array.shape}/{array.dtype}")
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("causal") is not True:
            raise ValueError("Causal feature manifest is incomplete")
        return {"eeg": eeg_path, "fnirs": fnirs_path, "manifest": manifest_path}

    eeg_output = np.lib.format.open_memmap(
        eeg_path, mode="w+", dtype=np.float32, shape=expected_shapes[eeg_path]
    )
    fnirs_output = np.lib.format.open_memmap(
        fnirs_path, mode="w+", dtype=np.float32, shape=expected_shapes[fnirs_path]
    )
    store = SignalWindowStore(signal_cache)
    subjects = sorted(set(int(value) for value in index.subject_numbers))
    for position, subject_number in enumerate(subjects, start=1):
        subject = f"test_{subject_number}"
        if verbose:
            print(f"[causal-features] {position}/{len(subjects)} {subject}", flush=True)
        reservation = store.reservation_mask(subject).astype(np.float32)
        videos = sorted(set(int(value) for value in index.videos[index.subjects == subject]))
        for video in videos:
            rows = np.flatnonzero((index.subjects == subject) & (index.videos == video))
            rows = rows[np.argsort(index.timestamps[rows])]
            eeg = np.asarray(store._array(subject, video, "eeg"), dtype=np.float32)
            fnirs = np.asarray(store._array(subject, video, "fnirs"), dtype=np.float32)
            if len(eeg) != len(rows) or len(fnirs) != len(rows):
                raise ValueError(f"Signal/label length mismatch: {subject} V{video:02d}")

            eeg_second = _eeg_segment_features(eeg / 1_000_000.0)
            eeg_context = causal_context(eeg_second, lookback=2)
            eeg_output[rows, :45] = eeg_context.mean(axis=1)
            eeg_output[rows, 45:] = eeg_context.std(axis=1)

            fnirs_second = np.stack(
                [_fnirs_segment_features(second) for second in fnirs], axis=0
            ).astype(np.float32)
            fnirs_context = causal_context(fnirs_second, lookback=2)
            weights = reservation[None, :, None]
            count = max(float(reservation.sum()), 1.0)
            mean = (fnirs_context * weights).sum(axis=1) / count
            variance = (
                np.square(fnirs_context - mean[:, None, :]) * weights
            ).sum(axis=1) / count
            fnirs_output[rows, :90] = mean
            fnirs_output[rows, 90:] = np.sqrt(np.maximum(variance, 0.0))
    eeg_output.flush()
    fnirs_output.flush()

    source_manifest = signal_cache / "manifest.json"
    payload = {
        "schema_version": "merps-causal-physiology-features-v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "causal": True,
        "temporal_support": "t-2,t-1,t with edge replication at trial start",
        "samples": int(len(index.targets)),
        "eeg": {
            "path": eeg_path.name,
            "shape": list(expected_shapes[eeg_path]),
            "definition": (
                "per-second spectral/Hjorth features; causal three-second context; "
                "channel mean and standard deviation"
            ),
        },
        "fnirs": {
            "path": fnirs_path.name,
            "shape": list(expected_shapes[fnirs_path]),
            "definition": (
                "per-second six-signal summary statistics; causal three-second context; "
                "reservation-aware channel mean and standard deviation"
            ),
        },
        "source_signal_manifest": str(source_manifest),
        "source_signal_manifest_sha256": (
            _manifest_hash(source_manifest) if source_manifest.exists() else None
        ),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"eeg": eeg_path, "fnirs": fnirs_path, "manifest": manifest_path}


def fit_standardized_ridge(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
    *,
    clip_z: float = 8.0,
) -> StandardizedRidge:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or y.shape[1] != 2:
        raise ValueError("features/target shapes are incompatible")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and target must be finite")
    x_mean = x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    x_scale = np.maximum(
        x.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32), 1e-6
    )
    standardized = np.clip((x - x_mean) / x_scale, -clip_z, clip_z)
    y_mean = y.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    centered_y = y - y_mean
    gram = np.asarray(standardized.T @ standardized, dtype=np.float64)
    cross = np.asarray(standardized.T @ centered_y, dtype=np.float64)
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    coefficient = np.linalg.solve(gram, cross).astype(np.float32)
    return StandardizedRidge(x_mean, x_scale, y_mean, coefficient, float(clip_z))


def fit_standardized_ridge_path(
    features: np.ndarray,
    target: np.ndarray,
    alphas: Sequence[float],
    *,
    clip_z: float = 8.0,
) -> dict[float, StandardizedRidge]:
    """Fit several Ridge penalties with one standardized Gram eigendecomposition."""

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    alpha_values = tuple(float(value) for value in alphas)
    if not alpha_values or any(value < 0.0 for value in alpha_values):
        raise ValueError("alphas must contain non-negative values")
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or y.shape[1] != 2:
        raise ValueError("features/target shapes are incompatible")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and target must be finite")
    x_mean = x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    x_scale = np.maximum(
        x.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32), 1e-6
    )
    standardized = np.clip((x - x_mean) / x_scale, -clip_z, clip_z)
    y_mean = y.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    centered_y = y - y_mean
    gram = np.asarray(standardized.T @ standardized, dtype=np.float64)
    cross = np.asarray(standardized.T @ centered_y, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ cross
    result: dict[float, StandardizedRidge] = {}
    for alpha in alpha_values:
        denominator = np.maximum(eigenvalues + alpha, 1e-12)
        coefficient = (
            eigenvectors @ (projected / denominator[:, None])
        ).astype(np.float32)
        result[alpha] = StandardizedRidge(
            x_mean, x_scale, y_mean, coefficient, float(clip_z)
        )
    return result


def causal_ema(
    values: np.ndarray,
    subjects: np.ndarray,
    videos: np.ndarray,
    timestamps: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Smooth corrections causally and reset state at every trial boundary."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("values must have shape [samples, 2]")
    if not 0.0 <= float(decay) < 1.0:
        raise ValueError("decay must be in [0, 1)")
    if float(decay) == 0.0:
        return values.copy()
    output = np.empty_like(values)
    pairs = sorted(set(zip((int(x) for x in subjects), (int(x) for x in videos))))
    for subject, video in pairs:
        rows = np.flatnonzero((subjects == subject) & (videos == video))
        rows = rows[np.argsort(timestamps[rows])]
        state = values[rows[0]].copy()
        output[rows[0]] = state
        for row in rows[1:]:
            state = float(decay) * state + (1.0 - float(decay)) * values[row]
            output[row] = state
    return output


def apply_ridge_residual(
    prior: np.ndarray,
    raw_residual: np.ndarray,
    *,
    gate: float,
    cap: float,
) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float32)
    residual = np.asarray(raw_residual, dtype=np.float32)
    if prior.shape != residual.shape or prior.ndim != 2 or prior.shape[1] != 2:
        raise ValueError("prior and residual must both have shape [samples, 2]")
    if not 0.0 <= float(gate) <= 1.0 or float(cap) < 0.0:
        raise ValueError("invalid residual gate/cap")
    correction = np.clip(float(gate) * residual, -float(cap), float(cap))
    return np.clip(prior + correction, 1.0, 255.0).astype(np.float32)


def fit_weighted_standardized_ridge_path(
    features: np.ndarray,
    target: np.ndarray,
    alphas: Sequence[float],
    sample_weight: np.ndarray,
    *,
    clip_z: float = 8.0,
) -> dict[float, StandardizedRidge]:
    """Fit a Ridge path with weights normalized to preserve alpha scale."""

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    alpha_values = tuple(float(value) for value in alphas)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or y.shape[1] != 2:
        raise ValueError("features/target shapes are incompatible")
    if weights.shape != (len(x),):
        raise ValueError("sample_weight must have one value per row")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("features, target and sample_weight must be finite")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("sample_weight must be non-negative with positive sum")
    if not alpha_values or any(value < 0.0 for value in alpha_values):
        raise ValueError("alphas must contain non-negative values")

    weights = weights * (len(weights) / float(weights.sum()))
    normalizer = float(weights.sum())
    x_mean = (
        (weights[:, None] * x).sum(axis=0, keepdims=True) / normalizer
    ).astype(np.float32)
    centered_x = x - x_mean
    x_scale = np.maximum(
        np.sqrt(
            (weights[:, None] * np.square(centered_x)).sum(
                axis=0, keepdims=True
            )
            / normalizer
        ).astype(np.float32),
        1e-6,
    )
    standardized = np.clip(centered_x / x_scale, -clip_z, clip_z)
    y_mean = (
        (weights[:, None] * y).sum(axis=0, keepdims=True) / normalizer
    ).astype(np.float32)
    centered_y = y - y_mean
    root_weight = np.sqrt(weights)[:, None]
    weighted_x = standardized * root_weight
    weighted_y = centered_y * root_weight
    gram = np.asarray(weighted_x.T @ weighted_x, dtype=np.float64)
    cross = np.asarray(weighted_x.T @ weighted_y, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ cross
    result: dict[float, StandardizedRidge] = {}
    for alpha in alpha_values:
        coefficient = (
            eigenvectors
            @ (projected / np.maximum(eigenvalues + alpha, 1e-12)[:, None])
        ).astype(np.float32)
        result[alpha] = StandardizedRidge(
            x_mean, x_scale, y_mean, coefficient, float(clip_z)
        )
    return result
