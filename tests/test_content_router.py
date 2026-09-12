from __future__ import annotations

import unittest

from merps.content_router import choose_backbone


class ContentRouterTests(unittest.TestCase):
    def test_selects_alternative_when_inner_gain_clears_gate(self) -> None:
        decision = choose_backbone(
            {"clip": 10.0, "siglip": 9.8},
            baseline="clip",
            minimum_inner_gain=0.1,
        )
        self.assertEqual(decision.selected, "siglip")
        self.assertTrue(decision.used_alternative)
        self.assertAlmostEqual(decision.inner_gain, 0.2)

    def test_exactly_falls_back_when_gain_is_too_small(self) -> None:
        decision = choose_backbone(
            {"clip": 10.0, "siglip": 9.95},
            baseline="clip",
            minimum_inner_gain=0.1,
        )
        self.assertEqual(decision.selected, "clip")
        self.assertFalse(decision.used_alternative)

    def test_tie_prefers_baseline(self) -> None:
        decision = choose_backbone(
            {"siglip": 10.0, "clip": 10.0},
            baseline="clip",
            minimum_inner_gain=0.0,
        )
        self.assertEqual(decision.selected, "clip")

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(KeyError):
            choose_backbone({"siglip": 1.0}, baseline="clip", minimum_inner_gain=0.0)
        with self.assertRaises(ValueError):
            choose_backbone({"clip": float("nan")}, baseline="clip", minimum_inner_gain=0.0)
        with self.assertRaises(ValueError):
            choose_backbone({"clip": 1.0}, baseline="clip", minimum_inner_gain=-0.1)


if __name__ == "__main__":
    unittest.main()
