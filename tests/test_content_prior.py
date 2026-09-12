import unittest
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.content_prior import (
    CategoryPhaseSpec,
    ResidualStackSpec,
    absolute_time_resample,
    apply_residual_stack,
    cross_fitted_category_phase_prior,
    doubly_held_out_category_priors,
    load_stimulus_manifest,
    nested_doubly_held_out_category_priors,
    phase_resample,
    predict_category_phase_prior,
    residual_target,
    trial_macro_mae,
)
from merps.innovation.data import InnovationIndex


def make_index(subjects=6):
    lengths = {1: 4, 2: 6, 3: 5, 4: 7}
    categories = {1: "positive", 2: "positive", 3: "negative", 4: "negative"}
    sample_ids = []
    subject_names = []
    subject_numbers = []
    videos = []
    timestamps = []
    targets = []
    for subject in range(1, subjects + 1):
        subject_offset = float(subject - 3) * 0.25
        for video, length in lengths.items():
            phase = np.linspace(0.0, 1.0, length)
            sign = 1.0 if categories[video] == "positive" else -1.0
            for timestamp in range(length):
                sample_ids.append(f"test_{subject}_V{video:02d}_T{timestamp:03d}")
                subject_names.append(f"test_{subject}")
                subject_numbers.append(subject)
                videos.append(video)
                timestamps.append(timestamp)
                targets.append(
                    [
                        128.0 + sign * 50.0 * phase[timestamp] + subject_offset,
                        128.0 + sign * 25.0 * phase[timestamp] + subject_offset,
                    ]
                )
    return (
        InnovationIndex(
            sample_ids=np.asarray(sample_ids),
            subjects=np.asarray(subject_names),
            subject_numbers=np.asarray(subject_numbers, dtype=np.int16),
            videos=np.asarray(videos, dtype=np.int16),
            timestamps=np.asarray(timestamps, dtype=np.int16),
            targets=np.asarray(targets, dtype=np.float32),
        ),
        categories,
    )


class ContentPriorTest(unittest.TestCase):
    def test_project_manifest_has_all_fifteen_verified_annotation_lengths(self):
        records = load_stimulus_manifest(
            PROJECT_ROOT / "configs" / "stimuli" / "refed_15_videos.csv"
        )
        self.assertEqual([record.video_id for record in records], list(range(1, 16)))
        self.assertEqual(
            [record.annotation_seconds for record in records],
            [135, 137, 90, 79, 135, 93, 122, 107, 70, 60, 61, 63, 111, 103, 170],
        )
        self.assertTrue(
            all(len(record.expected_sha256) == 64 for record in records)
        )
        self.assertEqual(len({record.expected_sha256 for record in records}), 15)

    def test_phase_resample_preserves_endpoints(self):
        curve = np.asarray([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32)
        actual = phase_resample(curve, 5)
        np.testing.assert_array_equal(actual[0], curve[0])
        np.testing.assert_array_equal(actual[-1], curve[-1])
        absolute = absolute_time_resample(curve, 5)
        np.testing.assert_array_equal(absolute[-1], curve[-1])

    def test_category_phase_prior_does_not_use_held_out_trace_or_video(self):
        index, categories = make_index()
        held_out = np.flatnonzero(
            (index.subject_numbers == 6) & (index.videos == 2)
        )
        first = predict_category_phase_prior(
            index,
            training_subjects=[1, 2, 3, 4, 5],
            training_videos=[1, 3, 4],
            prediction_indices=held_out,
            category_by_video=categories,
        )
        changed = index.targets.copy()
        changed[index.videos == 2] = 255.0 - changed[index.videos == 2]
        changed_index = InnovationIndex(
            sample_ids=index.sample_ids,
            subjects=index.subjects,
            subject_numbers=index.subject_numbers,
            videos=index.videos,
            timestamps=index.timestamps,
            targets=changed,
        )
        second = predict_category_phase_prior(
            changed_index,
            training_subjects=[1, 2, 3, 4, 5],
            training_videos=[1, 3, 4],
            prediction_indices=held_out,
            category_by_video=categories,
        )
        np.testing.assert_array_equal(first, second)

    def test_double_holdout_is_complete_and_improves_over_constant(self):
        index, categories = make_index(subjects=6)
        result = doubly_held_out_category_priors(
            index, categories, subject_folds=3, video_folds=2
        )
        self.assertTrue(np.isfinite(result.predictions["category_phase"]).all())
        self.assertTrue(np.all(result.subject_fold_ids >= 0))
        self.assertTrue(np.all(result.video_fold_ids >= 0))
        self.assertLess(
            trial_macro_mae(index, result.predictions["category_phase"]),
            trial_macro_mae(index, result.predictions["global_constant"]),
        )
        for record in result.fold_records:
            self.assertTrue(
                set(record["training_subjects"]).isdisjoint(
                    record["validation_subjects"]
                )
            )
            self.assertTrue(
                set(record["training_videos"]).isdisjoint(
                    record["validation_videos"]
                )
            )

    def test_nested_transfer_selection_stays_inside_outer_groups(self):
        index, categories = make_index(subjects=6)
        durations = {1: 4.0, 2: 6.0, 3: 5.0, 4: 7.0}
        candidates = (
            CategoryPhaseSpec(1.0, float("inf"), 0),
            CategoryPhaseSpec(0.5, 30.0, 1),
        )
        result = nested_doubly_held_out_category_priors(
            index,
            categories,
            durations,
            candidates=candidates,
            subject_folds=3,
            video_folds=4,
            inner_subject_folds=2,
        )
        self.assertTrue(
            np.isfinite(result.predictions["category_phase_nested"]).all()
        )
        for record in result.fold_records:
            self.assertIn(record["selected_spec"]["name"], {item.name for item in candidates})
            self.assertTrue(
                set(record["training_subjects"]).isdisjoint(
                    record["validation_subjects"]
                )
            )
            self.assertTrue(
                set(record["training_videos"]).isdisjoint(
                    record["validation_videos"]
                )
            )

    def test_residual_stack_has_exact_zero_fallback_and_cap(self):
        prior = np.asarray([[100.0, 200.0], [250.0, 5.0]], dtype=np.float32)
        residual = np.full_like(prior, 100.0)
        np.testing.assert_array_equal(apply_residual_stack(prior), prior)
        corrected = apply_residual_stack(
            prior,
            content_residual=residual,
            eeg_residual=-residual,
            spec=ResidualStackSpec(
                content_gate=1.0,
                eeg_gate=0.5,
                max_total_correction=10.0,
            ),
        )
        self.assertLessEqual(float(np.abs(corrected - prior).max()), 10.0)
        np.testing.assert_array_equal(
            residual_target(prior + 2.0, prior), np.full_like(prior, 2.0)
        )
        with self.assertRaises(ValueError):
            ResidualStackSpec(eeg_gate=0.1, max_total_correction=0.0)

    def test_cross_fitted_training_prior_excludes_own_participant(self):
        index, categories = make_index(subjects=6)
        rows, first = cross_fitted_category_phase_prior(
            index,
            training_subjects=[1, 2, 3, 4, 5],
            training_videos=[1, 2, 3, 4],
            category_by_video=categories,
        )
        changed_targets = index.targets.copy()
        changed_targets[index.subject_numbers == 1] += 40.0
        changed = InnovationIndex(
            sample_ids=index.sample_ids,
            subjects=index.subjects,
            subject_numbers=index.subject_numbers,
            videos=index.videos,
            timestamps=index.timestamps,
            targets=changed_targets,
        )
        changed_rows, second = cross_fitted_category_phase_prior(
            changed,
            training_subjects=[1, 2, 3, 4, 5],
            training_videos=[1, 2, 3, 4],
            category_by_video=categories,
        )
        np.testing.assert_array_equal(rows, changed_rows)
        own = index.subject_numbers[rows] == 1
        np.testing.assert_array_equal(first[own], second[own])


if __name__ == "__main__":
    unittest.main()
