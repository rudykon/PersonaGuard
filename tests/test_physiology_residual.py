import unittest
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.physiology_residual import (
    ResidualRidgeSpec,
    apply_ridge_residual,
    causal_context,
    causal_ema,
    fit_standardized_ridge,
    fit_standardized_ridge_path,
    fit_weighted_standardized_ridge_path,
)


class PhysiologyResidualTest(unittest.TestCase):
    def test_causal_context_cannot_read_future_seconds(self):
        values = np.arange(6, dtype=np.float32)[:, None, None]
        first = causal_context(values, lookback=2)
        changed = values.copy()
        changed[4:] = 1000.0
        second = causal_context(changed, lookback=2)
        np.testing.assert_array_equal(first[:4], second[:4])
        np.testing.assert_array_equal(first[0, 0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(first[3, 0], [1.0, 2.0, 3.0])

    def test_causal_ema_resets_at_trial_boundary(self):
        values = np.asarray([[0, 0], [10, 10], [100, 100], [0, 0]], dtype=np.float32)
        subjects = np.asarray([1, 1, 1, 1])
        videos = np.asarray([1, 1, 2, 2])
        timestamps = np.asarray([0, 1, 0, 1])
        actual = causal_ema(values, subjects, videos, timestamps, decay=0.5)
        np.testing.assert_allclose(actual, [[0, 0], [5, 5], [100, 100], [50, 50]])

    def test_ridge_fits_residual_and_gate_has_exact_zero_fallback(self):
        x = np.linspace(-2.0, 2.0, 50, dtype=np.float32)[:, None]
        y = np.concatenate([2.0 * x, -3.0 * x], axis=1)
        model = fit_standardized_ridge(x, y, alpha=0.01)
        raw = model.predict(x)
        self.assertLess(float(np.abs(raw - y).mean()), 0.01)
        prior = np.full_like(y, 128.0)
        np.testing.assert_array_equal(
            apply_ridge_residual(prior, raw, gate=0.0, cap=0.0), prior
        )
        corrected = apply_ridge_residual(prior, raw, gate=1.0, cap=2.0)
        self.assertLessEqual(float(np.abs(corrected - prior).max()), 2.0)

    def test_disabled_spec_is_the_only_zero_correction_configuration(self):
        self.assertEqual(ResidualRidgeSpec("disabled", 0, 0, 0).name, "disabled")
        with self.assertRaises(ValueError):
            ResidualRidgeSpec("eeg", 100, 0, 0)
        with self.assertRaises(ValueError):
            ResidualRidgeSpec("disabled", 0, 0.1, 8)

    def test_ridge_path_matches_independent_fits(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(80, 6)).astype(np.float32)
        y = rng.normal(size=(80, 2)).astype(np.float32)
        path = fit_standardized_ridge_path(x, y, [10.0, 100.0])
        for alpha, model in path.items():
            independent = fit_standardized_ridge(x, y, alpha)
            np.testing.assert_allclose(
                model.predict(x), independent.predict(x), atol=2e-5
            )

    def test_uniform_weighted_ridge_matches_unweighted_path(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(60, 5)).astype(np.float32)
        y = rng.normal(size=(60, 2)).astype(np.float32)
        ordinary = fit_standardized_ridge_path(x, y, [10.0, 100.0])
        weighted = fit_weighted_standardized_ridge_path(
            x, y, [10.0, 100.0], np.ones(len(x))
        )
        for alpha in ordinary:
            np.testing.assert_allclose(
                ordinary[alpha].predict(x),
                weighted[alpha].predict(x),
                atol=2e-5,
            )



if __name__ == "__main__":
    unittest.main()
