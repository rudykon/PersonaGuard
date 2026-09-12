from __future__ import annotations

import unittest
from pathlib import Path

import torch

from merps.innovation.cbramod import (
    CBraModBackbone,
    CBraModResidualHead,
    load_pretrained_cbramod,
    pool_cbramod_features,
)


class CBraModAdapterTest(unittest.TestCase):
    def test_small_backbone_accepts_arbitrary_channel_count(self):
        model = CBraModBackbone(
            d_model=200,
            out_dim=200,
            dim_feedforward=400,
            n_layer=1,
            nhead=4,
        )
        output = model(torch.randn(2, 5, 4, 200))
        self.assertEqual(tuple(output.shape), (2, 5, 4, 200))
        self.assertEqual(tuple(pool_cbramod_features(output).shape), (2, 400))

    def test_frozen_adapter_stays_below_two_million_parameters(self):
        backbone = CBraModBackbone(n_layer=1)
        model = CBraModResidualHead(backbone)
        self.assertTrue(all(not p.requires_grad for p in backbone.parameters()))
        self.assertLess(model.trainable_parameter_count(), 2_000_000)
        model.train()
        self.assertFalse(model.backbone.training)

    def test_official_checkpoint_loads_strictly_when_available(self):
        path = Path("checkpoints/innovation/pretrained_weights.pth")
        if not path.exists():
            self.skipTest("Official CBraMod checkpoint is not available locally")
        model = load_pretrained_cbramod(path)
        self.assertEqual(len(model.encoder.layers), 12)
        self.assertEqual(model.proj_out[0].out_features, 200)


if __name__ == "__main__":
    unittest.main()
