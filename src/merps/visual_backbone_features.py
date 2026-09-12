"""Frozen per-second visual embeddings for stimulus-content experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np


SCHEMA_VERSION = "merps-visual-backbone-features-v1"
BACKBONE_MODEL_IDS = {
    "siglip": "google/siglip-base-patch16-224",
    "dinov2": "facebook/dinov2-base",
}


def load_visual_backbone(
    model_id: str,
    *,
    device: str,
    local_files_only: bool = False,
):
    """Load a frozen Hugging Face vision encoder and its image processor."""

    from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

    processor = AutoImageProcessor.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    model_class = (
        SiglipVisionModel
        if "siglip" in str(model_id).lower()
        else AutoModel
    )
    model = model_class.from_pretrained(
        model_id, local_files_only=local_files_only
    ).eval()
    model.requires_grad_(False)
    model.to(device)
    return processor, model


def _pooled_embedding(outputs):
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None or hidden.ndim != 3:
        raise ValueError("Vision backbone exposes neither pooler nor token embeddings")
    return hidden[:, 0]


def encode_visual_frames(
    frames: np.ndarray,
    processor,
    model,
    *,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode and L2-normalize one vector per frame."""

    import torch

    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3 or len(values) < 1:
        raise ValueError("frames must have shape [time, height, width, 3]")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    output: list[np.ndarray] = []
    use_amp = str(device).startswith("cuda")
    for start in range(0, len(values), int(batch_size)):
        batch: Sequence[np.ndarray] = [
            frame for frame in values[start : start + int(batch_size)]
        ]
        pixel_values = processor(images=batch, return_tensors="pt")[
            "pixel_values"
        ].to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if use_amp else "cpu",
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            embedding = _pooled_embedding(model(pixel_values=pixel_values))
        normalized = torch.nn.functional.normalize(
            embedding.float(), p=2, dim=-1
        )
        output.append(normalized.cpu().numpy().astype(np.float32))
    result = np.concatenate(output, axis=0)
    if result.ndim != 2 or len(result) != len(values) or not np.isfinite(result).all():
        raise RuntimeError("Invalid visual backbone embedding matrix")
    return result
