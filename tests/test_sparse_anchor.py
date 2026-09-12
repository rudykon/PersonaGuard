import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.innovation.data import InnovationIndex
from merps.sparse_anchor import (
    AnchorSpec,
    ConditionalSpec,
    FunctionalSpec,
    PriorSpec,
    TrialTable,
    gaussian_neighbor_weights,
    apply_bounded_anchor,
    build_trial_table,
    estimate_population_trajectory,
    nested_sparse_anchor_oof,
    predict_conditional_population_prior,
    predict_functional_anchor_prior,
    predict_population_prior,
    repeated_outer_folds,
    sam_to_continuous,
    trial_macro_mae,
)


def make_index(subjects=6, videos=2, seconds=4):
    sample_ids = []
    subject_names = []
    subject_numbers = []
    video_ids = []
    timestamps = []
    targets = []
    trial_rows = []
    trial_subjects = []
    trial_videos = []
    anchors = []
    row_to_trial = []
    row = 0
    for subject in range(1, subjects + 1):
        offset = -24.0 if subject % 2 else 24.0
        for video in range(1, videos + 1):
            rows = []
            base = 80.0 + 25.0 * video
            for timestamp in range(seconds):
                sample_ids.append(f"test_{subject}_V{video:02d}_T{timestamp:03d}")
                subject_names.append(f"test_{subject}")
                subject_numbers.append(subject)
                video_ids.append(video)
                timestamps.append(timestamp)
                targets.append(
                    [base + timestamp + offset, 256.0 - base - timestamp + offset]
                )
                rows.append(row)
                row_to_trial.append(len(trial_rows))
                row += 1
            trial_rows.append(np.asarray(rows, dtype=np.int64))
            trial_subjects.append(subject)
            trial_videos.append(video)
            target = np.asarray(targets)[rows]
            anchors.append(np.median(target, axis=0))
    index = InnovationIndex(
        sample_ids=np.asarray(sample_ids),
        subjects=np.asarray(subject_names),
        subject_numbers=np.asarray(subject_numbers, dtype=np.int16),
        videos=np.asarray(video_ids, dtype=np.int16),
        timestamps=np.asarray(timestamps, dtype=np.int16),
        targets=np.asarray(targets, dtype=np.float32),
    )
    table = TrialTable(
        subjects=np.asarray(trial_subjects, dtype=np.int16),
        videos=np.asarray(trial_videos, dtype=np.int16),
        rows=tuple(trial_rows),
        anchors=np.asarray(anchors, dtype=np.float32),
        row_to_trial=np.asarray(row_to_trial, dtype=np.int32),
    )
    return index, table


class SparseAnchorTest(unittest.TestCase):
    def test_sam_scale_maps_endpoints_exactly(self):
        actual = sam_to_continuous([1.0, 5.0, 9.0])
        np.testing.assert_allclose(actual, [1.0, 128.0, 255.0])
        with self.assertRaises(ValueError):
            sam_to_continuous([0.0])

    def test_bounded_anchor_preserves_fallback_and_enforces_cap(self):
        prior = np.asarray([[20.0, 240.0], [100.0, 120.0]], dtype=np.float32)
        anchor = np.asarray([255.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(
            apply_bounded_anchor(prior, anchor, AnchorSpec(0.0, 0.0, 0.0)),
            prior,
        )
        corrected = apply_bounded_anchor(
            prior, anchor, AnchorSpec(0.30, 0.10, 10.0)
        )
        self.assertLessEqual(float(np.abs(corrected - prior).max()), 10.0)
        self.assertTrue(np.all(corrected[:, 0] >= prior[:, 0]))
        self.assertTrue(np.all(corrected[:, 1] <= prior[:, 1]))

    def test_pooled_median_is_robust_to_one_extreme_subject(self):
        trajectories = np.asarray(
            [
                [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]],
                [[11.0, 21.0], [11.0, 21.0], [11.0, 21.0]],
                [[250.0, 250.0], [250.0, 250.0], [250.0, 250.0]],
            ]
        )
        result = estimate_population_trajectory(
            trajectories, PriorSpec("pooled_median", 1)
        )
        self.assertTrue(np.all(result[:, 0] < 20.0))
        self.assertTrue(np.all(result[:, 1] < 30.0))

    def test_prior_does_not_use_held_out_labels(self):
        index, _ = make_index()
        held_out = np.flatnonzero(index.subject_numbers == 6)
        first = predict_population_prior(
            index, [1, 2, 3, 4, 5], held_out, PriorSpec("current", 1)
        )
        changed_targets = index.targets.copy()
        changed_targets[held_out] = 255.0 - changed_targets[held_out]
        changed = InnovationIndex(
            sample_ids=index.sample_ids,
            subjects=index.subjects,
            subject_numbers=index.subject_numbers,
            videos=index.videos,
            timestamps=index.timestamps,
            targets=changed_targets,
        )
        second = predict_population_prior(
            changed, [1, 2, 3, 4, 5], held_out, PriorSpec("current", 1)
        )
        np.testing.assert_array_equal(first, second)

    def test_conditional_prior_uses_anchor_but_not_held_out_trace(self):
        index, table = make_index()
        held_out = np.flatnonzero(index.subject_numbers == 6)
        base = predict_population_prior(
            index, [1, 2, 3, 4, 5], held_out, PriorSpec("current", 0)
        )
        first = predict_conditional_population_prior(
            index,
            table,
            [1, 2, 3, 4, 5],
            held_out,
            PriorSpec("current", 0),
            ConditionalSpec(3, 1.0),
            base,
        )
        changed_targets = index.targets.copy()
        changed_targets[held_out] = 255.0 - changed_targets[held_out]
        changed = InnovationIndex(
            sample_ids=index.sample_ids,
            subjects=index.subjects,
            subject_numbers=index.subject_numbers,
            videos=index.videos,
            timestamps=index.timestamps,
            targets=changed_targets,
        )
        second = predict_conditional_population_prior(
            changed,
            table,
            [1, 2, 3, 4, 5],
            held_out,
            PriorSpec("current", 0),
            ConditionalSpec(3, 1.0),
            base,
        )
        np.testing.assert_array_equal(first, second)
        disabled = predict_conditional_population_prior(
            index,
            table,
            [1, 2, 3, 4, 5],
            held_out,
            PriorSpec("current", 0),
            ConditionalSpec(0, 0.0),
            base,
        )
        np.testing.assert_array_equal(disabled, base)

    def test_gaussian_neighbor_weights_are_stable_and_uniform_at_infinity(self):
        uniform = gaussian_neighbor_weights(
            np.asarray([0.0, 1000.0, 2000.0]), float("inf")
        )
        np.testing.assert_array_equal(uniform, np.full(3, 1.0 / 3.0))
        extreme = gaussian_neighbor_weights(
            np.asarray([1000.0, 1001.0]), 0.01
        )
        self.assertTrue(np.isfinite(extreme).all())
        self.assertAlmostEqual(float(extreme.sum()), 1.0)
        self.assertGreater(float(extreme[0]), float(extreme[1]))

    def test_kernel_conditional_prior_uses_no_held_out_trace(self):
        index, table = make_index()
        held_out = np.flatnonzero(index.subject_numbers == 6)
        sources = [1, 2, 3, 4, 5]
        base = predict_population_prior(
            index, sources, held_out, PriorSpec("current", 0)
        )
        spec = ConditionalSpec(3, 1.0, "axis", 16.0)
        first = predict_conditional_population_prior(
            index, table, sources, held_out, PriorSpec("current", 0), spec, base
        )
        changed_targets = index.targets.copy()
        changed_targets[held_out] = 255.0 - changed_targets[held_out]
        changed = InnovationIndex(
            sample_ids=index.sample_ids, subjects=index.subjects,
            subject_numbers=index.subject_numbers, videos=index.videos,
            timestamps=index.timestamps, targets=changed_targets,
        )
        second = predict_conditional_population_prior(
            changed, table, sources, held_out, PriorSpec("current", 0), spec, base
        )
        np.testing.assert_array_equal(first, second)

    def test_functional_prior_uses_sam_but_not_held_out_trace(self):
        index, table = make_index()
        held_out = np.flatnonzero(index.subject_numbers == 6)
        base = predict_population_prior(
            index, [1, 2, 3, 4, 5], held_out, PriorSpec("current", 0)
        )
        spec = FunctionalSpec(100.0, 1, 0.75, 20.0)
        first = predict_functional_anchor_prior(
            index,
            table,
            [1, 2, 3, 4, 5],
            held_out,
            PriorSpec("current", 0),
            spec,
            base,
        )
        changed_targets = index.targets.copy()
        changed_targets[held_out] = 255.0 - changed_targets[held_out]
        changed = InnovationIndex(
            sample_ids=index.sample_ids,
            subjects=index.subjects,
            subject_numbers=index.subject_numbers,
            videos=index.videos,
            timestamps=index.timestamps,
            targets=changed_targets,
        )
        second = predict_functional_anchor_prior(
            changed,
            table,
            [1, 2, 3, 4, 5],
            held_out,
            PriorSpec("current", 0),
            spec,
            base,
        )
        np.testing.assert_array_equal(first, second)

    def test_nested_sparse_anchor_returns_complete_improved_oof(self):
        index, table = make_index()
        outer = repeated_outer_folds(range(1, 7), repeats=1, folds=3, seed=3)[0]
        result = nested_sparse_anchor_oof(
            index,
            table,
            outer_folds=outer,
            inner_folds=2,
            prior_candidates=(PriorSpec("current", 0),),
            anchor_candidates=(
                AnchorSpec(0.0, 0.0, 0.0),
                AnchorSpec(0.30, 0.10, 30.0),
            ),
            conditional_candidates=(
                ConditionalSpec(0, 0.0),
                ConditionalSpec(3, 1.0),
            ),
            canonical_prior=PriorSpec("current", 0),
        )
        for prediction in result.predictions.values():
            self.assertTrue(np.isfinite(prediction).all())
        baseline = trial_macro_mae(
            index, table, result.predictions["anchor_reference_prior"]
        )
        anchored = trial_macro_mae(
            index, table, result.predictions["bounded_sparse_anchor"]
        )
        self.assertLess(anchored, baseline)
        for selection in result.selections:
            self.assertTrue(
                set(selection["training_subjects"]).isdisjoint(
                    selection["validation_subjects"]
                )
            )

    def test_trial_table_reads_released_sam_schema(self):
        index, _ = make_index(subjects=1, videos=2, seconds=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SAM_score.csv"
            fields = ["sub_id"]
            for video in range(1, 16):
                fields.extend(
                    [
                        f"Video_{video}_Valence",
                        f"Video_{video}_Arousal",
                        f"Video_{video}_Dominance",
                        f"Video_{video}_Familiarity",
                    ]
                )
            values = ["test_1"]
            for _ in range(15):
                values.extend(["5", "5", "5", "5"])
            path.write_text(
                ",".join(fields) + "\n" + ",".join(values) + "\n",
                encoding="utf-8",
            )
            table = build_trial_table(index, path)
            self.assertEqual(len(table.rows), 2)
            np.testing.assert_allclose(table.anchors, 128.0)


if __name__ == "__main__":
    unittest.main()
