import unittest
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merps.innovation.data import InnovationIndex
from run_av_physio_residual import (
    build_feature_views,
    causal_mean_view,
    causal_shift_view,
    feature_view,
)


def make_index() -> InnovationIndex:
    return InnovationIndex(
        sample_ids=np.asarray(["a0", "a1", "a2", "b0", "b1", "b2"]),
        subjects=np.asarray(["s1"] * 6),
        subject_numbers=np.asarray([1] * 6, dtype=np.int16),
        videos=np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int16),
        timestamps=np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int16),
        targets=np.zeros((6, 2), dtype=np.float32),
    )


class AVPhysioViewTest(unittest.TestCase):
    def test_shift_is_past_only_and_resets_at_trial_boundary(self):
        index = make_index()
        values = np.arange(6, dtype=np.float32)[:, None]
        shifted = causal_shift_view(values, index, 1)
        np.testing.assert_array_equal(
            shifted[:, 0], [0, 0, 1, 3, 3, 4]
        )

    def test_mean_is_past_only_and_resets_at_trial_boundary(self):
        index = make_index()
        values = np.arange(6, dtype=np.float32)[:, None]
        mean = causal_mean_view(values, index, 2)
        np.testing.assert_allclose(
            mean[:, 0], [0, 0.5, 1.5, 3, 3.5, 4.5]
        )
        changed = values.copy()
        changed[2] = 1000
        changed_mean = causal_mean_view(changed, index, 2)
        np.testing.assert_array_equal(mean[:2], changed_mean[:2])

    def test_requested_multiscale_views_and_fusion_shapes(self):
        index = make_index()
        eeg = np.ones((6, 2), dtype=np.float32)
        fnirs = np.arange(18, dtype=np.float32).reshape(6, 3)
        cbramod = np.ones((6, 4), dtype=np.float32)
        views = build_feature_views(
            index,
            eeg,
            fnirs,
            cbramod,
            ("fnirs_lag6", "cbramod_fnirs_mean10"),
        )
        self.assertIn("fnirs_lag6", views)
        self.assertIn("fnirs_mean10", views)
        rows = np.asarray([0, 1, 2])
        self.assertEqual(
            feature_view("cbramod_fnirs_mean10", views, rows).shape,
            (3, 7),
        )


    def test_cbramod_token_views_preserve_causal_patch_order(self):
        index = make_index()
        eeg = np.ones((6, 2), dtype=np.float32)
        fnirs = np.arange(18, dtype=np.float32).reshape(6, 3)
        cbramod = np.ones((6, 4), dtype=np.float32)
        tokens = np.arange(6 * 4 * 200, dtype=np.float32).reshape(6, 4, 200)
        modalities = (
            "cbramod_latest",
            "cbramod_token_delta",
            "cbramod_temporal",
            "cbramod_token_stats",
            "cbramod_temporal_fnirs",
        )
        views = build_feature_views(
            index, eeg, fnirs, cbramod, modalities, tokens
        )
        rows = np.asarray([0, 2, 3])
        latest = feature_view("cbramod_latest", views, rows)
        delta = feature_view("cbramod_token_delta", views, rows)
        np.testing.assert_array_equal(latest, tokens[rows, -1, :])
        np.testing.assert_array_equal(
            delta, tokens[rows, -1, :] - tokens[rows, 0, :]
        )
        self.assertEqual(
            feature_view("cbramod_temporal", views, rows).shape, (3, 400)
        )
        self.assertEqual(
            feature_view("cbramod_token_stats", views, rows).shape, (3, 400)
        )
        self.assertEqual(
            feature_view("cbramod_temporal_fnirs", views, rows).shape,
            (3, 400 + fnirs.shape[1]),
        )


if __name__ == "__main__":
    unittest.main()
