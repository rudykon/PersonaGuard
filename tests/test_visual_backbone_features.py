from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from merps.visual_backbone_features import _pooled_embedding, encode_visual_frames


class ToyProcessor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        values = np.stack(images).astype(np.float32)
        return {"pixel_values": torch.from_numpy(values).permute(0, 3, 1, 2)}


class ToyModel:
    def __call__(self, *, pixel_values):
        pooled = pixel_values.mean(dim=(2, 3))
        return SimpleNamespace(pooler_output=pooled, last_hidden_state=None)


def test_pooled_embedding_falls_back_to_cls_token() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    output = _pooled_embedding(
        SimpleNamespace(pooler_output=None, last_hidden_state=hidden)
    )
    torch.testing.assert_close(output, hidden[:, 0])


def test_visual_embeddings_are_unit_normalized() -> None:
    frames = np.ones((3, 4, 4, 3), dtype=np.uint8)
    output = encode_visual_frames(
        frames,
        ToyProcessor(),
        ToyModel(),
        device="cpu",
        batch_size=2,
    )
    assert output.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-6)
