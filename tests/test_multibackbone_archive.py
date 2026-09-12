from __future__ import annotations

import unittest

import numpy as np

from scripts.build_multibackbone_archive import (
    causal_embedding_difference,
    normalize_rows,
    parse_weight,
)


class MultiBackboneArchiveTests(unittest.TestCase):
    def test_row_normalization_is_finite_for_zero_rows(self) -> None:
        values = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        normalized = normalize_rows(values)
        np.testing.assert_allclose(normalized[0], [0.6, 0.8], atol=1e-7)
        np.testing.assert_allclose(normalized[1], [0.0, 0.0], atol=0.0)

    def test_causal_difference_uses_only_current_and_past_rows(self) -> None:
        values = np.arange(12, dtype=np.float32).reshape(6, 2)
        observed = causal_embedding_difference(values, lag=2)
        np.testing.assert_allclose(observed[:2], values[:2] - values[[0, 0]])
        np.testing.assert_allclose(observed[2:], values[2:] - values[:-2])

        changed_future = values.copy()
        changed_future[4:] += 10_000.0
        changed = causal_embedding_difference(changed_future, lag=2)
        np.testing.assert_allclose(changed[:4], observed[:4])

    def test_each_call_resets_the_video_boundary(self) -> None:
        first = np.asarray([[1.0], [2.0]], dtype=np.float32)
        second = np.asarray([[100.0], [103.0]], dtype=np.float32)
        np.testing.assert_allclose(causal_embedding_difference(first, 1)[0], [0.0])
        np.testing.assert_allclose(causal_embedding_difference(second, 1)[0], [0.0])

    def test_nonpositive_lag_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            causal_embedding_difference(np.ones((2, 2), dtype=np.float32), 0)


    def test_visual_weight_parser_accepts_only_positive_finite_values(self) -> None:
        self.assertEqual(parse_weight("siglip:0.5"), ("siglip", 0.5))
        for value in ("siglip:0", "siglip:-1", "siglip:nan", "bad"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_weight(value)

    def test_equal_raw_weights_reproduce_equal_norm_scaling(self) -> None:
        weights = np.asarray([1.0, 1.0], dtype=np.float32)
        weights /= np.linalg.norm(weights)
        np.testing.assert_allclose(weights, np.full(2, 1.0 / np.sqrt(2.0)))
if __name__ == "__main__":
    unittest.main()
