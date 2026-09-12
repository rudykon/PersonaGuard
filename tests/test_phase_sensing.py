from __future__ import annotations

import unittest

import numpy as np

from merps.innovation.phase_sensing import (
    average_precision,
    binary_auroc,
    fit_ridge_path,
)


class PhaseSensingTests(unittest.TestCase):
    def test_ridge_path_predicts_linear_target(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(200, 5)).astype(np.float32)
        y = (x[:, :1] * 2.0 - x[:, 1:2]).astype(np.float32)
        model = fit_ridge_path(x, y)
        prediction = model.predict(x, 1.0)
        self.assertLess(float(np.abs(prediction - y).mean()), 0.05)

    def test_binary_metrics_are_perfect_for_ordered_scores(self):
        target = np.asarray([0, 0, 1, 1], dtype=bool)
        score = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(binary_auroc(target, score), 1.0)
        self.assertAlmostEqual(average_precision(target, score), 1.0)


if __name__ == "__main__":
    unittest.main()
