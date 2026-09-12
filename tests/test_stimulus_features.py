import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merps.content_prior import StimulusRecord
from merps.stimulus_features import (
    AUDIO_RATE,
    SCHEMA_VERSION,
    VideoFeatureRecord,
    audio_feature_names,
    audio_feature_vector,
    audit_stimuli,
    pack_feature_records,
    visual_feature_names,
    visual_feature_vector,
)
from prepare_stimulus_features import portable_path


class StimulusFeatureTest(unittest.TestCase):
    def test_visual_features_are_deterministic_and_motion_uses_previous_frame(self):
        first = np.zeros((8, 12, 3), dtype=np.uint8)
        second = np.full((8, 12, 3), 255, dtype=np.uint8)
        no_motion = visual_feature_vector(first)
        motion = visual_feature_vector(second, first)
        self.assertEqual(no_motion.shape, (len(visual_feature_names()),))
        self.assertTrue(np.isfinite(motion).all())
        self.assertEqual(float(no_motion[-10]), 0.0)
        self.assertGreater(float(motion[-10]), 0.0)

    def test_audio_features_capture_energy_and_frequency(self):
        time = np.arange(AUDIO_RATE, dtype=np.float32) / AUDIO_RATE
        signal = 0.25 * np.sin(2.0 * np.pi * 440.0 * time)
        features, spectrum = audio_feature_vector(signal)
        repeated, _ = audio_feature_vector(signal, previous_spectrum=spectrum)
        self.assertEqual(features.shape, (len(audio_feature_names()),))
        self.assertGreater(float(features[0]), 0.1)
        self.assertGreater(float(features[4]), 0.04)
        self.assertLess(float(features[4]), 0.08)
        self.assertAlmostEqual(float(repeated[-1]), 0.0, places=6)

    def test_missing_media_is_not_identity_verified(self):
        record = StimulusRecord(
            video_id=1,
            target_code="MVMA",
            target_label="neutral",
            source_dataset="test",
            title="missing",
            reported_duration_seconds=10,
            annotation_seconds=9,
            local_filename="missing.mp4",
            source_reference="",
            source_excerpt="",
            expected_sha256="",
            access_status="not_obtained",
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = audit_stimuli([record], tmp)
        self.assertEqual(result[0]["status"], "missing")
        self.assertFalse(result[0]["identity_verified"])

    def test_audit_paths_are_portable(self):
        self.assertEqual(
            portable_path(PROJECT_ROOT / "configs" / "stimuli" / "manifest.csv"),
            "configs/stimuli/manifest.csv",
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(portable_path(Path(tmp) / "clip.mp4"), "<external>/clip.mp4")

    def test_feature_records_pack_without_pickle_objects(self):
        names = ("one", "two")
        records = [
            VideoFeatureRecord(
                video_id=2,
                features=np.ones((3, 2), dtype=np.float32),
                feature_names=names,
                media_sha256="b" * 64,
                identity_status="duration_only_unverified",
            ),
            VideoFeatureRecord(
                video_id=1,
                features=np.zeros((2, 2), dtype=np.float32),
                feature_names=names,
                media_sha256="a" * 64,
                identity_status="identity_verified",
            ),
        ]
        packed = pack_feature_records(records)
        self.assertEqual(str(packed["schema_version"]), SCHEMA_VERSION)
        np.testing.assert_array_equal(packed["video_ids"], [1, 1, 2, 2, 2])
        self.assertEqual(packed["features"].dtype, np.float32)
        self.assertNotEqual(packed["features"].dtype, object)


if __name__ == "__main__":
    unittest.main()
