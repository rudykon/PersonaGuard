from __future__ import annotations

import unittest

import numpy as np

from merps.innovation.personalization import (
    additive_personalization,
    fit_low_rank_profile,
    low_rank_personalization,
)


class PersonalizationTests(unittest.TestCase):
    def test_additive_calibration_recovers_user_offset(self):
        mean = np.asarray([1.0, 2.0, 3.0, 4.0])
        target = mean + 2.0
        prediction = additive_personalization(
            mean, np.asarray([0, 2]), target[[0, 2]], shrinkage=0.0
        )
        self.assertTrue(np.allclose(prediction, target))

    def test_low_rank_calibration_recovers_interaction_direction(self):
        users = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.2, 2.1, 2.9, 3.8],
                [3.8, 2.9, 2.1, 1.2],
            ]
        )
        profile = fit_low_rank_profile(users, rank=1)
        target = np.asarray([1.1, 2.0, 3.0, 3.9])
        prediction = low_rank_personalization(
            profile, np.asarray([0, 3]), target[[0, 3]], ridge=0.01
        )
        self.assertLess(float(np.abs(prediction - target).mean()), 0.3)


if __name__ == "__main__":
    unittest.main()
