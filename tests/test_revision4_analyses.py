from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merps.innovation.normative_dynamics import WarpConfig
from scripts.run_revision4_analyses import (
    audited_dtw,
    baseline_comparison_from_predictions,
    category_reliability,
    category_variance_components,
    consensus_mapping,
    manifest_configuration,
    parse_args,
    shifted_trace_mae,
)


class Revision4AnalysisTests(unittest.TestCase):
    def test_manifest_configuration_uses_portable_anonymous_paths(self):
        args = parse_args([])
        configuration = manifest_configuration(args)
        self.assertEqual(configuration["data_root"], "data/MER_PS_trainval")
        self.assertEqual(configuration["output_dir"], "artifacts/revision4")
        self.assertNotIn(str(PROJECT_ROOT), str(configuration))

    def test_consensus_projection_is_noop_for_monotone_mappings(self):
        first = np.asarray([0, 0, 2, 3, 5, 5], dtype=np.int16)
        second = np.asarray([0, 1, 1, 4, 4, 5], dtype=np.int16)
        mapping, changes = consensus_mapping(first, second)
        expected = np.floor((first + second) / 2.0 + 0.5).astype(np.int16)
        np.testing.assert_array_equal(mapping, expected)
        self.assertEqual(changes, 0)
        self.assertTrue(np.all(np.diff(mapping) >= 0))

    def test_multivariate_dtw_keeps_identical_curve_near_diagonal(self):
        time = np.linspace(0.0, 3.0 * np.pi, 80)
        curve = np.column_stack([np.sin(time), np.cos(0.7 * time)])
        result = audited_dtw(
            curve,
            curve,
            WarpConfig(band_seconds=8, warp_penalty=0.2, step_penalty=0.05),
        )
        self.assertTrue(np.isfinite(result.total_cost))
        self.assertLess(float(np.abs(np.arange(len(curve)) - result.mapping).mean()), 0.5)

    def test_path_mean_objective_is_finite_and_not_worse_than_direct_average(self):
        time = np.linspace(0.0, 4.0 * np.pi, 90)
        template = np.column_stack([np.sin(time), np.cos(0.6 * time)])
        target = np.roll(template, 3, axis=0)
        target[:3] = template[0]
        config = WarpConfig(band_seconds=8, warp_penalty=0.1, step_penalty=0.05)
        direct = audited_dtw(target, template, config, objective="direct_step")
        path_mean = audited_dtw(target, template, config, objective="path_mean")
        direct_average = direct.total_cost / direct.path_length
        path_average = path_mean.total_cost / path_mean.path_length
        self.assertTrue(np.isfinite(path_average))
        self.assertLessEqual(path_average, direct_average + 1e-9)

    def test_mixed_step_mean_optimizes_its_declared_objective(self):
        time = np.linspace(0.0, 3.0 * np.pi, 45)
        template = np.column_stack([np.sin(time), np.cos(0.8 * time)])
        target = np.roll(template, 2, axis=0)
        target[:2] = template[0]
        config = WarpConfig(band_seconds=6, warp_penalty=0.1, step_penalty=0.05)
        mixed = audited_dtw(
            target, template, config, objective="mixed_step_mean"
        )
        step_zero = audited_dtw(
            target, template, config, objective="step_zero"
        )
        zero_path_under_mixed = (
            step_zero.local_cost
            + step_zero.warp_cost
            + config.step_penalty
            * step_zero.non_diagonal_steps
            / step_zero.path_length
        )
        self.assertTrue(np.isfinite(mixed.total_cost))
        self.assertLessEqual(mixed.total_cost, zero_path_under_mixed + 1e-9)
        self.assertAlmostEqual(
            mixed.step_cost,
            config.step_penalty
            * mixed.non_diagonal_steps
            / mixed.path_length,
        )

    def test_category_components_recover_balanced_additive_structure(self):
        participant = np.linspace(-1.5, 1.5, 24)
        category = np.asarray([-0.8, -0.4, 0.0, 0.4, 0.8])
        nested_video = np.asarray([-0.3, 0.0, 0.3])
        cube = (
            participant[:, None, None]
            + category[None, :, None]
            + nested_video[None, None, :]
        )
        components = category_variance_components(cube)
        self.assertGreater(components["participant"], 0.0)
        self.assertAlmostEqual(components["participant_by_category"], 0.0, places=10)
        self.assertAlmostEqual(
            components["residual_participant_by_video"], 0.0, places=10
        )
        reliability = category_reliability(components, videos_per_category=3)
        self.assertAlmostEqual(reliability["relative_g"], 1.0, places=10)
        self.assertLess(reliability["absolute_phi"], 1.0)

    def test_baseline_comparison_uses_the_supplied_prediction_stack(self):
        methods = [
            "video_mean",
            "context_only",
            "context_plus_blockwise_residual",
        ]
        target = np.zeros((3, 4), dtype=np.float64)
        predictions = np.stack(
            [
                np.full_like(target, 0.5),
                np.full_like(target, 0.6),
                np.full_like(target, 0.8),
            ]
        )
        rows, summary = baseline_comparison_from_predictions(
            methods,
            predictions,
            target,
            bootstrap_repeats=20,
            seed=7,
        )
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(
            summary["video_mean_advantage_over_context_seconds"], 0.1
        )
        self.assertAlmostEqual(
            summary["sensor_gain_over_video_mean_seconds"], -0.3
        )

    def test_positive_signed_lag_uses_positive_timestamp_correction(self):
        time = np.linspace(-2.0, 2.0, 80)
        template = np.column_stack(
            [
                np.exp(-np.square(time + 0.3)) + 0.1 * time,
                np.tanh(1.3 * time) + 0.15 * np.sin(4.0 * time),
            ]
        )
        delay = 4
        target = np.empty_like(template)
        target[:delay] = template[0]
        target[delay:] = template[:-delay]
        corrected = shifted_trace_mae(target, template, float(delay))
        uncorrected = shifted_trace_mae(target, template, 0.0)
        wrong_direction = shifted_trace_mae(target, template, float(-delay))
        self.assertLess(corrected, uncorrected)
        self.assertLess(corrected, wrong_direction)


if __name__ == "__main__":
    unittest.main()
