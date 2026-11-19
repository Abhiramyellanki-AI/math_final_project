"""Milestone M1 tests for EEG data contract, filtering, epoch shapes, labels, and split isolation.

Includes:
1. Labeled synthetic MNE Raw fixtures verifying exact [0.5, 2.5) s sample counts (320 @ 160 Hz),
   T0 exclusion, T1->0 / T2->1 mapping, independent run filtering, and dev/test split isolation.
2. Real Subject 1 (Runs 4, 8, 12) data contract verification when local EDF files are accessible.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np

from eeg_p3.config import load_config
from eeg_p3.data import combine_run_results, prepare_subject_dataset
from eeg_p3.plotting import plot_labeled_motor_imagery_trials, plot_raw_vs_filtered_trace
from eeg_p3.preprocessing import (
    count_run_annotations,
    extract_epochs_from_filtered_run,
    filter_raw_run_independently,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "subject01.yaml"


def make_synthetic_mne_raw(
    run_id: int,
    n_channels: int = 64,
    sfreq: float = 160.0,
    duration_sec: float = 35.0,
    seed: int = 42,
) -> mne.io.RawArray:
    """Create a clearly labeled synthetic MNE RawArray fixture for unit testing ONLY."""
    rng = np.random.default_rng(seed + run_id)
    n_samples = int(round(duration_sec * sfreq))
    times = np.arange(n_samples, dtype=np.float64) / sfreq

    # Synthesize 10 Hz (mu) + 20 Hz (beta) + 2 Hz slow drift in realistic Volt scale (~20e-6 V)
    signals = np.zeros((n_channels, n_samples), dtype=np.float64)
    for ch in range(n_channels):
        mu_component = 15e-6 * np.sin(2.0 * np.pi * 10.0 * times + ch * 0.1)
        beta_component = 8e-6 * np.cos(2.0 * np.pi * 20.0 * times + ch * 0.05)
        drift_component = 25e-6 * np.sin(2.0 * np.pi * 1.5 * times)
        noise = rng.normal(0.0, 3e-6, size=n_samples)
        signals[ch] = mu_component + beta_component + drift_component + noise

    base_names = ["C3", "Cz", "C4"] + [f"EEG{i:03d}" for i in range(4, n_channels + 1)]
    info = mne.create_info(ch_names=base_names[:n_channels], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(signals, info, verbose="ERROR")

    # Add T0 (rest), T1 (left), T2 (right) annotations spaced 4.2 s apart
    onsets = [1.0, 5.2, 9.4, 13.6, 17.8, 22.0, 26.2]
    durations = [4.0] * len(onsets)
    descriptions = ["T0", "T1", "T0", "T2", "T1", "T0", "T2"]
    raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))
    return raw


class TestM1SyntheticDataContract(unittest.TestCase):
    """Unit tests using labeled synthetic MNE fixtures to verify M1 processing rules."""

    def test_independent_filtering_and_exact_320_sample_epochs(self) -> None:
        """Filtering and epoch extraction produce (n_trials, 64, 320) and exclude T0."""
        raw = make_synthetic_mne_raw(run_id=4)
        counts = count_run_annotations(raw)
        self.assertEqual(counts, {"T0": 3, "T1": 2, "T2": 2})

        filtered = filter_raw_run_independently(raw, low_hz=8.0, high_hz=30.0)
        res = extract_epochs_from_filtered_run(
            raw_filtered=filtered,
            subject_id=1,
            run_id=4,
            tmin_sec=0.5,
            tmax_sec=2.5,
            expected_n_times=320,
        )

        # 2 T1 + 2 T2 = 4 motor imagery trials; all 3 T0 rest events excluded
        self.assertEqual(res.X.shape, (4, 64, 320))
        self.assertEqual(res.y.tolist(), [0, 1, 0, 1])
        self.assertTrue(np.all(res.groups == 4))
        self.assertEqual(res.signal_unit, "V")
        self.assertAlmostEqual(float(res.times_sec[0]), 0.5, places=5)
        self.assertAlmostEqual(float(res.times_sec[-1]), 2.5 - 1.0 / 160.0, places=5)

    def test_split_bundle_preserves_run_ids_and_prevents_test_leakage(self) -> None:
        """Development runs (4, 8) and held-out test run (12) remain strictly separated."""
        per_run = {}
        for rid in (4, 8, 12):
            raw = make_synthetic_mne_raw(run_id=rid)
            filt = filter_raw_run_independently(raw, low_hz=8.0, high_hz=30.0)
            per_run[rid] = extract_epochs_from_filtered_run(filt, subject_id=1, run_id=rid)

        dev_bundle = combine_run_results("development", (4, 8), per_run)
        test_bundle = combine_run_results("held_out_test", (12,), per_run)

        self.assertEqual(dev_bundle.X.shape, (8, 64, 320))
        self.assertEqual(test_bundle.X.shape, (4, 64, 320))
        self.assertEqual(sorted(set(dev_bundle.groups.tolist())), [4, 8])
        self.assertEqual(sorted(set(test_bundle.groups.tolist())), [12])
        self.assertNotIn(12, dev_bundle.groups.tolist())

    def test_plotting_functions_create_valid_png_files(self) -> None:
        """Waveform and labeled trial plotting functions save non-empty PNG files."""
        raw = make_synthetic_mne_raw(run_id=4)
        filt = filter_raw_run_independently(raw, low_hz=8.0, high_hz=30.0)
        res = extract_epochs_from_filtered_run(filt, subject_id=1, run_id=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            p1 = plot_raw_vs_filtered_trace(raw, tmp_path / "trace.png", channel_name="C3", duration_sec=8.0)
            p2 = plot_labeled_motor_imagery_trials(res, tmp_path / "trials.png")
            self.assertTrue(p1.is_file() and p1.stat().st_size > 1000)
            self.assertTrue(p2.is_file() and p2.stat().st_size > 1000)


class TestM1RealSubject01DataContract(unittest.TestCase):
    """Integration tests on real PhysioNet Subject 1 (Runs 4, 8, 12) EDF files."""

    def test_real_subject01_runs_4_8_12_contract(self) -> None:
        """Verify actual Subject 1 Runs 4, 8, 12 shapes, 64 channels, 160 Hz, and T1/T2 labels."""
        cfg = load_config(DEFAULT_CONFIG)
        dataset = prepare_subject_dataset(cfg)

        # Verify dev split (Runs 4 & 8)
        self.assertEqual(dataset.dev.run_ids, (4, 8))
        self.assertEqual(dataset.dev.X.ndim, 3)
        self.assertEqual(dataset.dev.X.shape[1:], (64, 320))
        self.assertEqual(dataset.dev.sfreq, 160.0)
        self.assertEqual(set(dataset.dev.y.tolist()), {0, 1})
        self.assertEqual(set(dataset.dev.groups.tolist()), {4, 8})

        # Verify held-out test split (Run 12)
        self.assertEqual(dataset.test.run_ids, (12,))
        self.assertEqual(dataset.test.X.ndim, 3)
        self.assertEqual(dataset.test.X.shape[1:], (64, 320))
        self.assertEqual(set(dataset.test.y.tolist()), {0, 1})
        self.assertEqual(set(dataset.test.groups.tolist()), {12})

        # Verify physiological Volt scale (typical scalp EEG std is ~1e-6 to 1e-4 V)
        std_volts = float(np.std(dataset.dev.X))
        self.assertGreater(std_volts, 1e-7)
        self.assertLess(std_volts, 1e-2)


if __name__ == "__main__":
    unittest.main()
