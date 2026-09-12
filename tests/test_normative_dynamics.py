from __future__ import annotations

import unittest

import numpy as np

from merps.innovation.normative_dynamics import (
    WarpConfig,
    _event_boundary_mask,
    constrained_dtw_mapping,
    decompose_trial,
)


class NormativeDynamicsTests(unittest.TestCase):
    def test_cross_dimension_warp_recovers_shared_delay(self):
        time = np.linspace(0.0, 4.0 * np.pi, 100)
        template = np.stack([np.sin(time), np.cos(time * 0.7)], axis=1)
        delay = 4
        target = np.empty_like(template)
        target[:delay] = template[0]
        target[delay:] = template[:-delay]
        result = decompose_trial(
            target,
            template,
            WarpConfig(
                band_seconds=8,
                warp_penalty=0.05,
                step_penalty=0.02,
                smoothing_radius=1,
            ),
        )
        self.assertGreater(result.metrics["cross_gain_mean"], 0.0)
        self.assertGreater(result.metrics["median_lag_seconds"], 1.5)

    def test_mapping_is_monotonic_and_banded(self):
        source = np.sin(np.linspace(0, 8, 60))
        template = np.sin(np.linspace(0, 8, 60))
        config = WarpConfig(band_seconds=5)
        mapping = constrained_dtw_mapping(source, template, config)
        self.assertTrue(np.all(np.diff(mapping) >= 0))
        self.assertLessEqual(int(np.abs(np.arange(60) - mapping).max()), 5)

    def test_identical_curves_keep_near_diagonal_mapping(self):
        curve = np.sin(np.linspace(0, 6, 80))
        mapping = constrained_dtw_mapping(
            curve,
            curve,
            WarpConfig(band_seconds=8, warp_penalty=0.3),
        )
        self.assertLess(float(np.abs(np.arange(80) - mapping).mean()), 0.5)

    def test_boundary_mask_keeps_stable_samples_under_ties(self):
        template = np.zeros((60, 2), dtype=np.float64)
        mask = _event_boundary_mask(template)
        self.assertEqual(int(mask.sum()), 15)
        self.assertTrue(np.any(~mask))

    def test_absolute_trial_deviation_dominates_signed_bias(self):
        """Equation 5 uses |delta|, so q must be at least |b|."""
        lag = np.asarray([-4, -2, 0, 1, 5], dtype=np.float64)
        signed_bias = float(lag.mean())
        absolute_deviation = float(np.abs(lag).mean())
        self.assertGreaterEqual(absolute_deviation, abs(signed_bias))

        time = np.linspace(0.0, 3.0 * np.pi, 90)
        template = np.stack([np.sin(time), np.cos(time)], axis=1)
        target = np.roll(template, 3, axis=0)
        target[:3] = template[0]
        result = decompose_trial(target, template, WarpConfig(band_seconds=8))
        q_pv = float(result.metrics["mean_absolute_lag_seconds"])
        b_pv = float(result.consensus_lag.mean())
        self.assertGreaterEqual(q_pv + 1e-12, abs(b_pv))


if __name__ == "__main__":
    unittest.main()
