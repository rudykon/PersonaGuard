from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from merps.innovation.data import (
    InnovationIndex,
    SignalWindowStore,
    _eeg_quality,
    _fnirs_quality,
    build_context_features,
    _resample_eeg_trial,
    build_cross_fitted_context,
    trial_chunks,
)


def synthetic_index() -> InnovationIndex:
    sample_ids = []
    subjects = []
    subject_numbers = []
    videos = []
    timestamps = []
    targets = []
    for subject in range(1, 5):
        for video in (1, 2):
            for timestamp in range(5):
                sample_ids.append(f"test_{subject}_V{video:02d}_T{timestamp:03d}")
                subjects.append(f"test_{subject}")
                subject_numbers.append(subject)
                videos.append(video)
                timestamps.append(timestamp)
                targets.append(
                    [100.0 + 2 * timestamp + subject, 140.0 - timestamp + subject]
                )
    return InnovationIndex(
        sample_ids=np.asarray(sample_ids),
        subjects=np.asarray(subjects),
        subject_numbers=np.asarray(subject_numbers, dtype=np.int16),
        videos=np.asarray(videos, dtype=np.int16),
        timestamps=np.asarray(timestamps, dtype=np.int16),
        targets=np.asarray(targets, dtype=np.float32),
    )


class InnovationDataTest(unittest.TestCase):
    def test_context_uses_only_source_subjects(self):
        index = synthetic_index()
        rows = np.flatnonzero(index.subject_numbers == 4)
        prior, context = build_context_features(index, [1, 2, 3], rows, radius=0)
        first = rows[0]
        expected = np.median(
            index.targets[
                (index.subject_numbers <= 3)
                & (index.videos == index.videos[first])
                & (index.timestamps == index.timestamps[first])
            ],
            axis=0,
        )
        np.testing.assert_allclose(prior[0], expected)
        self.assertEqual(context.shape, (len(rows), 7))

    def test_cross_fitted_context_excludes_own_labels(self):
        index = synthetic_index()
        rows = np.arange(len(index.targets))
        prior, _ = build_cross_fitted_context(index, [1, 2, 3, 4], rows, radius=0)
        row = np.flatnonzero(
            (index.subject_numbers == 1)
            & (index.videos == 1)
            & (index.timestamps == 0)
        )[0]
        local = int(np.flatnonzero(rows == row)[0])
        expected = np.median(
            index.targets[
                (index.subject_numbers != 1)
                & (index.videos == 1)
                & (index.timestamps == 0)
            ],
            axis=0,
        )
        np.testing.assert_allclose(prior[local], expected)

    def test_quality_shapes_are_stable(self):
        rng = np.random.default_rng(7)
        eeg = rng.normal(size=(3, 64, 200)).astype(np.float32)
        fnirs = rng.normal(size=(3, 6, 51, 4)).astype(np.float32)
        reservation = np.ones(51, dtype=np.float32)
        self.assertEqual(_eeg_quality(eeg).shape, (3, 5))
        self.assertEqual(_fnirs_quality(fnirs, reservation).shape, (3, 4))

    def test_fft_antialias_suppresses_out_of_band_alias(self):
        source_rate = 1000
        seconds = 8
        time = np.arange(source_rate * seconds) / source_rate
        low = np.sin(2.0 * np.pi * 30.0 * time).astype(np.float32)
        high = np.sin(2.0 * np.pi * 170.0 * time).astype(np.float32)

        low_antialias = _resample_eeg_trial(
            low[None, :], seconds, method="fft_antialias"
        )[2:-2, 0].reshape(-1)
        high_boxcar = _resample_eeg_trial(
            high[None, :], seconds, method="block_average"
        )[2:-2, 0].reshape(-1)
        high_antialias = _resample_eeg_trial(
            high[None, :], seconds, method="fft_antialias"
        )[2:-2, 0].reshape(-1)

        low_rms = float(np.sqrt(np.mean(low_antialias**2)))
        boxcar_alias_rms = float(np.sqrt(np.mean(high_boxcar**2)))
        antialias_rms = float(np.sqrt(np.mean(high_antialias**2)))
        self.assertGreater(low_rms, 0.68)
        self.assertLess(low_rms, 0.73)
        self.assertGreater(boxcar_alias_rms, 0.05)
        self.assertLess(antialias_rms, 0.01)
        self.assertLess(antialias_rms, boxcar_alias_rms / 10.0)

    def test_window_store_pads_trial_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trials = root / "trials"
            trials.mkdir()
            (root / "subjects.txt").write_text("test_1\n", encoding="utf-8")
            np.save(root / "reservation_masks.npy", np.ones((1, 51), dtype=np.float32))
            eeg = np.arange(5 * 64 * 200, dtype=np.float32).reshape(5, 64, 200)
            fnirs = np.arange(5 * 6 * 51 * 4, dtype=np.float32).reshape(5, 6, 51, 4)
            quality = np.arange(5 * 9, dtype=np.float32).reshape(5, 9)
            np.save(trials / "test_1_V01_eeg.npy", eeg)
            np.save(trials / "test_1_V01_fnirs.npy", fnirs)
            np.save(trials / "test_1_V01_quality.npy", quality)
            store = SignalWindowStore(root)
            self.assertEqual(store.eeg_window("test_1", 1, 0).shape, (64, 4, 200))
            self.assertEqual(
                store.fnirs_window("test_1", 1, 0, seconds=3).shape,
                (12, 6, 51),
            )
            self.assertEqual(store.quality_features("test_1", 1, 0).shape, (9,))

    def test_trial_chunks_never_cross_trials(self):
        index = synthetic_index()
        chunks = trial_chunks(
            index, np.arange(len(index.targets)), chunk_seconds=3, stride=3
        )
        for chunk in chunks:
            self.assertEqual(len(set(index.subject_numbers[chunk].tolist())), 1)
            self.assertEqual(len(set(index.videos[chunk].tolist())), 1)


if __name__ == "__main__":
    unittest.main()
