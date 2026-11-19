"""Milestone M2 unit tests for deterministic mu/beta log-bandpower feature extraction."""

from __future__ import annotations

import unittest

import numpy as np

from eeg_p3.features import MuBetaLogBandpowerTransformer, extract_mu_beta_log_bandpower


class TestM2BandpowerFeatures(unittest.TestCase):
    """Tests verifying shape, determinism, finiteness, and spectral specificity."""

    def test_output_shape_and_determinism(self) -> None:
        """A (n_trials, 64, 320) tensor yields identical (n_trials, 128) features on repeat calls."""
        rng = np.random.default_rng(123)
        X = rng.normal(0.0, 20e-6, size=(6, 64, 320))

        feat1 = extract_mu_beta_log_bandpower(X, sfreq=160.0)
        feat2 = extract_mu_beta_log_bandpower(X, sfreq=160.0)

        self.assertEqual(feat1.shape, (6, 128))
        self.assertTrue(np.all(np.isfinite(feat1)))
        np.testing.assert_array_equal(feat1, feat2)

    def test_single_trial_independence(self) -> None:
        """Feature vector of trial i depends ONLY on trial i, never on other trials in the batch."""
        rng = np.random.default_rng(999)
        X_batch = rng.normal(0.0, 15e-6, size=(5, 64, 320))
        feat_batch = extract_mu_beta_log_bandpower(X_batch, sfreq=160.0)

        # Perturb trials 1..4 massively; trial 0's extracted features must be bit-identical
        X_perturbed = X_batch.copy()
        X_perturbed[1:] = 999.0
        feat_perturbed = extract_mu_beta_log_bandpower(X_perturbed, sfreq=160.0)

        np.testing.assert_allclose(feat_batch[0], feat_perturbed[0], rtol=0.0, atol=1e-15)

    def test_spectral_specificity_mu_vs_beta(self) -> None:
        """A 10 Hz sine wave increases mu (8-13 Hz) power; a 20 Hz sine wave increases beta (13-30 Hz) power."""
        sfreq = 160.0
        n_channels = 4
        n_times = 320
        t = np.arange(n_times, dtype=np.float64) / sfreq

        # Trial 0: pure 10 Hz (mu rhythm); Trial 1: pure 20 Hz (beta rhythm)
        X = np.zeros((2, n_channels, n_times), dtype=np.float64)
        X[0, :, :] = 20e-6 * np.sin(2.0 * np.pi * 10.0 * t)
        X[1, :, :] = 20e-6 * np.sin(2.0 * np.pi * 20.0 * t)

        feats = extract_mu_beta_log_bandpower(X, sfreq=sfreq)
        mu_feats = feats[:, :n_channels]
        beta_feats = feats[:, n_channels:]

        # Trial 0 must have higher mu than beta; Trial 1 must have higher beta than mu
        self.assertTrue(np.all(mu_feats[0] > beta_feats[0] + 1.5))
        self.assertTrue(np.all(beta_feats[1] > mu_feats[1] + 1.5))

    def test_sklearn_transformer_wrapper(self) -> None:
        """MuBetaLogBandpowerTransformer implements fit/transform with 2 * n_channels output."""
        rng = np.random.default_rng(7)
        X = rng.normal(0.0, 10e-6, size=(4, 64, 320))
        transformer = MuBetaLogBandpowerTransformer(sfreq=160.0)
        out = transformer.fit_transform(X)
        self.assertEqual(out.shape, (4, 128))
        self.assertEqual(transformer.n_features_out_, 128)


if __name__ == "__main__":
    unittest.main()
