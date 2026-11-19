"""Milestone M4 unit tests for Random Kitchen Sinks (RBFSampler) shapes, seed determinism, and accounting."""

from __future__ import annotations

import unittest

import numpy as np

from eeg_p3.rks import RKSFeatureMapper, compute_rks_storage_accounting


class TestM4RandomKitchenSinks(unittest.TestCase):
    """Tests verifying RKS output feature count D, fixed-seed reproducibility, and train-only scaling."""

    def test_output_shape_and_fixed_seed_reproducibility(self) -> None:
        """RKSFeatureMapper outputs (N, D) and produces identical features for fixed random_state."""
        rng = np.random.default_rng(77)
        X_train = rng.normal(0.0, 2.0, size=(15, 128))
        X_val = rng.normal(0.0, 2.0, size=(15, 128))

        for d_val in (128, 256, 512):
            m1 = RKSFeatureMapper(n_components=d_val, gamma=0.01, random_state=42)
            m2 = RKSFeatureMapper(n_components=d_val, gamma=0.01, random_state=42)

            Z_train_1 = m1.fit_transform(X_train)
            Z_train_2 = m2.fit_transform(X_train)
            Z_val_1 = m1.transform(X_val)
            Z_val_2 = m2.transform(X_val)

            self.assertEqual(Z_train_1.shape, (15, d_val))
            self.assertEqual(Z_val_1.shape, (15, d_val))
            np.testing.assert_array_equal(Z_train_1, Z_train_2)
            np.testing.assert_array_equal(Z_val_1, Z_val_2)

    def test_different_seeds_change_random_projection(self) -> None:
        """Changing random_state changes the randomized Fourier weights W and offsets b."""
        rng = np.random.default_rng(88)
        X_train = rng.normal(0.0, 1.0, size=(10, 128))

        m1 = RKSFeatureMapper(n_components=256, gamma=0.01, random_state=42).fit(X_train)
        m2 = RKSFeatureMapper(n_components=256, gamma=0.01, random_state=99).fit(X_train)

        self.assertFalse(np.allclose(m1.transform(X_train), m2.transform(X_train)))

    def test_rks_storage_accounting_flags_expansion_vs_bandpower(self) -> None:
        """Storage accounting explicitly records feature expansion when D > 128."""
        acc_128 = compute_rks_storage_accounting(input_dim=128, n_components=128)
        acc_512 = compute_rks_storage_accounting(input_dim=128, n_components=512)

        self.assertAlmostEqual(acc_128["expansion_ratio_vs_bandpower"], 1.0)
        self.assertFalse(acc_128["is_dimensionality_expansion_vs_bandpower"])

        self.assertAlmostEqual(acc_512["expansion_ratio_vs_bandpower"], 4.0)
        self.assertTrue(acc_512["is_dimensionality_expansion_vs_bandpower"])
        self.assertAlmostEqual(acc_512["compression_ratio_vs_raw"], 40.0)
        self.assertEqual(acc_512["feature_bytes_per_trial"], 512 * 8)

    def test_out_of_sample_invariance_to_validation_trials(self) -> None:
        """Perturbing validation trials never changes the training scaler or RBFSampler weights."""
        rng = np.random.default_rng(91)
        X_all = rng.normal(0.0, 3.0, size=(20, 128))
        train_idx = np.arange(0, 12)
        val_idx = np.arange(12, 20)

        m_orig = RKSFeatureMapper(n_components=256, gamma=0.01, random_state=42).fit(X_all[train_idx])
        val0_orig = m_orig.transform(X_all[val_idx])[0]

        X_corrupted = X_all.copy()
        X_corrupted[13:] = 1e6
        m_after = RKSFeatureMapper(n_components=256, gamma=0.01, random_state=42).fit(X_corrupted[train_idx])
        val0_after = m_after.transform(X_corrupted[val_idx])[0]

        np.testing.assert_allclose(val0_orig, val0_after, rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
