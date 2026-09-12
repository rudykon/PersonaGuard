"""Frozen, causal audio-visual representations for REFED stimuli.

The feature extractor is label-free. A CLIP image embedding is computed once
per playback second and an AudioSet-pretrained AST embedding is computed from
a causal audio window ending at that second. Full media support is retained so
timestamp offsets can be evaluated without re-encoding videos.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "merps-foundation-stimulus-features-v1"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16_000
FRAME_SIZE = 224
TIMESTAMP_CONVENTION = (
    "feature index t uses the visual frame emitted by ffmpeg fps=1 for second t; "
    "audio uses [max(0,t-window+1),t+1) with zero left-padding"
)


def _executable(name: str) -> str:
    value = shutil.which(name)
    if value is None:
        raise FileNotFoundError(f"Executable not found: {name}")
    return value


def decode_visual_seconds(
    path: str | Path,
    *,
    seconds: int,
    ffmpeg: str = "ffmpeg",
    frame_size: int = FRAME_SIZE,
) -> np.ndarray:
    """Decode one centre-cropped RGB frame per media second."""

    if int(seconds) < 1:
        raise ValueError("seconds must be positive")
    size = int(frame_size)
    completed = subprocess.run(
        [
            _executable(ffmpeg),
            "-v",
            "error",
            "-i",
            str(Path(path)),
            "-an",
            "-vf",
            (
                "fps=fps=1:start_time=0,"
                f"scale={size}:{size}:force_original_aspect_ratio=increase,"
                f"crop={size}:{size},format=rgb24"
            ),
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    frame_bytes = size * size * 3
    count = len(completed.stdout) // frame_bytes
    if count < 1:
        raise ValueError(f"No visual frames decoded from {path}")
    frames = np.frombuffer(
        completed.stdout[: count * frame_bytes], dtype=np.uint8
    ).reshape(count, size, size, 3)
    required = int(seconds)
    if len(frames) < required:
        frames = np.concatenate(
            [frames, np.repeat(frames[-1:], required - len(frames), axis=0)],
            axis=0,
        )
    return frames[:required].copy()


def decode_mono_audio(
    path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    completed = subprocess.run(
        [
            _executable(ffmpeg),
            "-v",
            "error",
            "-i",
            str(Path(path)),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(int(sample_rate)),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    values = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32)
    if len(values) < 1:
        raise ValueError(f"No audio decoded from {path}")
    return values


def causal_audio_window(
    audio: np.ndarray,
    second: int,
    *,
    window_seconds: int = 10,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return fixed-length audio ending at second + 1 without future data."""

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    rate = int(sample_rate)
    width = int(window_seconds) * rate
    if width < rate:
        raise ValueError("window_seconds must be positive")
    end = min((int(second) + 1) * rate, len(values))
    start = max(0, end - width)
    result = values[start:end]
    if len(result) < width:
        result = np.pad(result, (width - len(result), 0))
    return np.asarray(result, dtype=np.float32)


def load_foundation_models(
    *,
    device: str,
    clip_model_id: str = CLIP_MODEL_ID,
    ast_model_id: str = AST_MODEL_ID,
    local_files_only: bool = False,
):
    """Load processors and frozen encoders without a global Transformers import."""

    from transformers import (
        ASTModel,
        AutoFeatureExtractor,
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    clip_processor = CLIPImageProcessor.from_pretrained(
        clip_model_id, local_files_only=local_files_only
    )
    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        clip_model_id, local_files_only=local_files_only
    ).eval()
    ast_processor = AutoFeatureExtractor.from_pretrained(
        ast_model_id, local_files_only=local_files_only
    )
    ast_model = ASTModel.from_pretrained(
        ast_model_id, local_files_only=local_files_only
    ).eval()
    for model in (clip_model, ast_model):
        model.requires_grad_(False)
        model.to(device)
    return clip_processor, clip_model, ast_processor, ast_model


def _l2_normalize(values):
    import torch

    return torch.nn.functional.normalize(values.float(), p=2, dim=-1)


def encode_clip_frames(
    frames: np.ndarray,
    processor,
    model,
    *,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    import torch

    output: list[np.ndarray] = []
    use_amp = str(device).startswith("cuda")
    for start in range(0, len(frames), int(batch_size)):
        batch = [frame for frame in frames[start : start + int(batch_size)]]
        pixel_values = processor(images=batch, return_tensors="pt")[
            "pixel_values"
        ].to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if use_amp else "cpu",
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            embedding = model(pixel_values=pixel_values).image_embeds
        output.append(_l2_normalize(embedding).cpu().numpy().astype(np.float32))
    return np.concatenate(output, axis=0)


def encode_ast_audio(
    audio: np.ndarray,
    seconds: int,
    processor,
    model,
    *,
    device: str,
    batch_size: int = 8,
    window_seconds: int = 10,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    import torch

    output: list[np.ndarray] = []
    use_amp = str(device).startswith("cuda")
    for start in range(0, int(seconds), int(batch_size)):
        indices = range(start, min(start + int(batch_size), int(seconds)))
        windows = [
            causal_audio_window(
                audio,
                second,
                window_seconds=window_seconds,
                sample_rate=sample_rate,
            )
            for second in indices
        ]
        input_values = processor(
            windows, sampling_rate=int(sample_rate), return_tensors="pt"
        )["input_values"].to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if use_amp else "cpu",
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            embedding = model(input_values=input_values).pooler_output
        output.append(_l2_normalize(embedding).cpu().numpy().astype(np.float32))
    return np.concatenate(output, axis=0)


def extract_foundation_features(
    path: str | Path,
    *,
    media_duration_seconds: float,
    clip_processor,
    clip_model,
    ast_processor,
    ast_model,
    device: str,
    visual_batch_size: int = 64,
    audio_batch_size: int = 8,
    audio_window_seconds: int = 10,
    ffmpeg: str = "ffmpeg",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract normalized CLIP and causal AST vectors over the full clip."""

    seconds = max(1, int(math.ceil(float(media_duration_seconds))))
    frames = decode_visual_seconds(path, seconds=seconds, ffmpeg=ffmpeg)
    audio = decode_mono_audio(path, ffmpeg=ffmpeg)
    visual = encode_clip_frames(
        frames,
        clip_processor,
        clip_model,
        device=device,
        batch_size=visual_batch_size,
    )
    acoustic = encode_ast_audio(
        audio,
        seconds,
        ast_processor,
        ast_model,
        device=device,
        batch_size=audio_batch_size,
        window_seconds=audio_window_seconds,
    )
    features = np.concatenate([visual, acoustic], axis=1).astype(np.float32)
    names = tuple(
        [f"clip_{index:03d}" for index in range(visual.shape[1])]
        + [f"ast_{index:03d}" for index in range(acoustic.shape[1])]
    )
    if features.shape != (seconds, len(names)) or not np.isfinite(features).all():
        raise RuntimeError("Invalid frozen foundation feature matrix")
    return features, names


def model_revision(model, model_id: str | None = None) -> str:
    revision = str(getattr(model.config, "_commit_hash", "") or "")
    if revision or not model_id:
        return revision or "unknown"
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        from huggingface_hub.file_download import repo_folder_name

        reference = (
            Path(HF_HUB_CACHE)
            / repo_folder_name(repo_id=model_id, repo_type="model")
            / "refs"
            / "main"
        )
        return reference.read_text(encoding="utf-8").strip() or "unknown"
    except (ImportError, OSError):
        return "unknown"
