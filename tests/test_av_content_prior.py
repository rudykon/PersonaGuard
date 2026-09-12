import unittest
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from merps.av_content_prior import (
    ContentTemplateSpec,
    FoundationFeatureArchive,
    blend_content_prior,
    cross_fitted_content_prior,
    content_template_curve,
    monotonic_content_map,
    phase_resample_matrix,
    source_similarity_weights,
)
from merps.content_prior import CategoryPhaseSpec
from tests.test_content_prior import make_index


class AVContentPriorTest(unittest.TestCase):
    def archive(self) -> FoundationFeatureArchive:
        names = ("clip_000", "clip_001", "ast_000")
        return FoundationFeatureArchive(
            feature_names=names,
            features_by_video={
                1: np.asarray(
                    [[1, 0, 0], [0.9, 0.1, 1], [0.8, 0.2, 2], [0.7, 0.3, 3]],
                    dtype=np.float32,
                ),
                2: np.asarray(
                    [[1, 0, 4], [1, 0, 5], [1, 0, 6], [1, 0, 7]],
                    dtype=np.float32,
                ),
                3: np.asarray(
                    [[0, 1, 8], [0, 1, 9], [0, 1, 10], [0, 1, 11]],
                    dtype=np.float32,
                ),
            },
            annotation_lengths={1: 3, 2: 3, 3: 3},
            schema_version="test",
            media_sha256={1: "a", 2: "b", 3: "c"},
        )

    def test_offset_alignment_clamps_only_at_media_boundaries(self):
        archive = self.archive()
        negative = archive.sequence(
            1, length=3, offset_seconds=-1, modality="ast"
        )
        positive = archive.sequence(
            1, length=3, offset_seconds=2, modality="ast"
        )
        np.testing.assert_array_equal(negative[:, 0], [0, 0, 1])
        np.testing.assert_array_equal(positive[:, 0], [2, 3, 3])

    def test_visual_prefix_and_visual_ast_modalities(self):
        archive = FoundationFeatureArchive(
            feature_names=("visual_0000", "visual_0001", "ast_000"),
            features_by_video={
                1: np.asarray([[1, 0, 3], [0, 1, 4]], dtype=np.float32)
            },
            annotation_lengths={1: 2},
            schema_version="test",
            media_sha256={1: "a"},
        )
        self.assertEqual(archive.sequence(1, modality="visual").shape, (2, 2))
        self.assertEqual(
            archive.sequence(1, modality="visual_ast").shape, (2, 3)
        )

    def test_phase_resampling_preserves_endpoints(self):
        values = np.asarray([[0, 10], [10, 20]], dtype=np.float32)
        actual = phase_resample_matrix(values, 5)
        np.testing.assert_array_equal(actual[0], values[0])
        np.testing.assert_array_equal(actual[-1], values[-1])
        np.testing.assert_allclose(actual[2], [5, 15])

    def test_identity_content_has_identity_monotonic_map(self):
        features = np.eye(5, dtype=np.float32)
        mapping = monotonic_content_map(
            features, features, band=1.0, phase_penalty=0.0
        )
        np.testing.assert_allclose(mapping, np.arange(5), atol=1e-6)

    def test_hard_clip_selection_chooses_most_similar_source(self):
        archive = self.archive()
        weights = source_similarity_weights(
            archive, 1, [2, 3], temperature=0.0, modality="clip"
        )
        np.testing.assert_array_equal(weights, [1.0, 0.0])

    def test_hard_template_uses_selected_curve(self):
        archive = self.archive()
        curves = {
            2: np.asarray([[10, 20], [20, 30], [30, 40]], dtype=np.float32),
            3: np.asarray([[100, 110], [110, 120], [120, 130]], dtype=np.float32),
        }
        spec = ContentTemplateSpec(0.0, 0.0, 0.0, 0.0, 1.0)
        actual = content_template_curve(
            archive,
            1,
            curves,
            [2, 3],
            target_length=3,
            spec=spec,
        )
        np.testing.assert_array_equal(actual, curves[2])

    def test_zero_content_gate_is_exact_metadata_fallback(self):
        prior = np.asarray([[30, 40], [50, 60]], dtype=np.float32)
        template = np.asarray([[200, 210], [220, 230]], dtype=np.float32)
        actual = blend_content_prior(prior, template, 0.0)
        np.testing.assert_array_equal(actual, prior)
        self.assertIsNot(actual, prior)

    def test_cross_fitted_prior_excludes_own_participant_and_video(self):
        index, categories = make_index(subjects=6)
        lengths = {
            video: int(index.timestamps[index.videos == video].max()) + 1
            for video in range(1, 5)
        }
        archive = FoundationFeatureArchive(
            feature_names=("clip_000", "clip_001"),
            features_by_video={
                video: np.tile(
                    np.asarray([[float(video), 1.0]], dtype=np.float32),
                    (length, 1),
                )
                for video, length in lengths.items()
            },
            annotation_lengths=lengths,
            schema_version="test",
            media_sha256={video: str(video) for video in lengths},
        )
        spec = ContentTemplateSpec(0.0, 0.0, 0.0, 0.0, 1.0)
        durations = {video: float(length) for video, length in lengths.items()}
        rows, first = cross_fitted_content_prior(
            index,
            training_subjects=[1, 2, 3, 4, 5],
            training_videos=[1, 2, 3, 4],
            category_by_video=categories,
            archive=archive,
            base_spec=CategoryPhaseSpec(),
            axis_specs=(spec, spec),
            duration_by_video=durations,
        )
        changed_targets = index.targets.copy()
        changed_targets[index.subject_numbers == 1] += 30.0
        changed_targets[index.videos == 1] += 40.0
        changed = type(index)(
            sample_ids=index.sample_ids,
            subjects=index.subjects,
            subject_numbers=index.subject_numbers,
            videos=index.videos,
            timestamps=index.timestamps,
            targets=changed_targets,
        )
        changed_rows, second = cross_fitted_content_prior(
            changed,
            training_subjects=[1, 2, 3, 4, 5],
            training_videos=[1, 2, 3, 4],
            category_by_video=categories,
            archive=archive,
            base_spec=CategoryPhaseSpec(),
            axis_specs=(spec, spec),
            duration_by_video=durations,
        )
        np.testing.assert_array_equal(rows, changed_rows)
        own_trial = (
            (index.subject_numbers[rows] == 1)
            & (index.videos[rows] == 1)
        )
        np.testing.assert_array_equal(first[own_trial], second[own_trial])



if __name__ == "__main__":
    unittest.main()
