"""Milestone M3 unit tests for train-only HOSVD subspace learning, projection, and storage accounting."""

from __future__ import annotations

import unittest

import numpy as np

from eeg_p3.hosvd import (
    HOSVDFeatureExtractor,
    compute_hosvd_storage_accounting,
    compute_relative_frobenius_error,
    fit_hosvd_bases,
    project_hosvd_trials,
    reconstruct_hosvd_trials,
)


class TestM3HOSVDProjectionAndNoLeakage(unittest.TestCase):
    """Tests verifying (C, T) -> (r_c * r_t,) projection, determinism, and zero out-of-sample leakage."""

    def test_projection_shape_and_exact_bilinear_agreement(self) -> None:
        """project_hosvd_trials matches explicit U_c.T @ X_i @ U_t loop and outputs (r_c * r_t,) per trial."""
        rng = np.random.default_rng(101)
        X_train = rng.normal(0.0, 20e-6, size=(10, 64, 320))
        X_test = rng.normal(0.0, 20e-6, size=(4, 64, 320))

        r_c, r_t = 8, 20
        bases = fit_hosvd_bases(X_train, r_c=r_c, r_t=r_t)

        # Verify orthonormal columns
        np.testing.assert_allclose(bases.U_c.T @ bases.U_c, np.eye(r_c), atol=1e-12)
        np.testing.assert_allclose(bases.U_t.T @ bases.U_t, np.eye(r_t), atol=1e-12)

        # Single-trial 2D projection (64, 320) -> (160,)
        z0 = project_hosvd_trials(X_test[0], bases.U_c, bases.U_t, flatten=True)
        self.assertEqual(z0.shape, (r_c * r_t,))

        # Batch 3D projection (4, 64, 320) -> (4, 160)
        Z_batch = project_hosvd_trials(X_test, bases.U_c, bases.U_t, flatten=True)
        self.assertEqual(Z_batch.shape, (4, r_c * r_t))

        for i in range(X_test.shape[0]):
            expected_core = bases.U_c.T @ X_test[i] @ bases.U_t
            np.testing.assert_allclose(Z_batch[i], expected_core.reshape(-1), rtol=1e-12, atol=1e-15)

    def test_deterministic_repeated_fit_and_transform(self) -> None:
        """Repeated HOSVD fitting on identical inputs produces bit-identical bases and features."""
        rng = np.random.default_rng(303)
        X_train = rng.normal(0.0, 15e-6, size=(8, 64, 320))

        ext1 = HOSVDFeatureExtractor(r_c=4, r_t=10).fit(X_train)
        ext2 = HOSVDFeatureExtractor(r_c=4, r_t=10).fit(X_train)

        np.testing.assert_array_equal(ext1.U_c_, ext2.U_c_)
        np.testing.assert_array_equal(ext1.U_t_, ext2.U_t_)
        np.testing.assert_array_equal(ext1.transform(X_train), ext2.transform(X_train))

        # Confirm no trial-mode factor matrix attribute exists
        self.assertFalse(hasattr(ext1, "U_trials_"))
        self.assertFalse(hasattr(ext1.bases_, "U_trials"))

    def test_out_of_sample_invariance_no_leakage(self) -> None:
        """Held-out trials never influence U_c or U_t, and trial i's features never depend on trial j."""
        rng = np.random.default_rng(404)
        X_all = rng.normal(0.0, 20e-6, size=(16, 64, 320))
        train_idx = np.arange(0, 10)
        test_idx = np.arange(10, 16)

        ext_orig = HOSVDFeatureExtractor(r_c=8, r_t=20).fit(X_all[train_idx])
        test_feat_0_orig = ext_orig.transform(X_all[test_idx])[0]

        # Massively corrupt held-out trials 11..15; re-fit on train_idx and transform test_idx
        X_corrupted = X_all.copy()
        X_corrupted[11:] = -1e5
        ext_after = HOSVDFeatureExtractor(r_c=8, r_t=20).fit(X_corrupted[train_idx])
        test_feat_0_after = ext_after.transform(X_corrupted[test_idx])[0]

        np.testing.assert_allclose(ext_orig.U_c_, ext_after.U_c_, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(ext_orig.U_t_, ext_after.U_t_, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(test_feat_0_orig, test_feat_0_after, rtol=0.0, atol=1e-15)

    def test_reconstruction_error_monotonicity_and_full_rank_zero(self) -> None:
        """Relative Frobenius reconstruction error decreases monotonically with rank and vanishes at full rank."""
        rng = np.random.default_rng(505)
        X = rng.normal(0.0, 10e-6, size=(6, 16, 40))

        b_low = fit_hosvd_bases(X, r_c=4, r_t=10)
        b_mid = fit_hosvd_bases(X, r_c=8, r_t=20)
        b_full = fit_hosvd_bases(X, r_c=16, r_t=40)

        err_low = compute_relative_frobenius_error(X, b_low.U_c, b_low.U_t)
        err_mid = compute_relative_frobenius_error(X, b_mid.U_c, b_mid.U_t)
        err_full = compute_relative_frobenius_error(X, b_full.U_c, b_full.U_t)

        self.assertGreater(err_low, err_mid)
        self.assertGreater(err_mid, err_full)
        self.assertLess(err_full, 1e-12)

        X_hat_full = reconstruct_hosvd_trials(X, b_full.U_c, b_full.U_t)
        np.testing.assert_allclose(X, X_hat_full, rtol=1e-11, atol=1e-15)

    def test_storage_accounting_distinguishes_trial_and_tensor_compression(self) -> None:
        """Storage accounting reports both per-trial feature compression and Tucker-2 tensor compression."""
        acc = compute_hosvd_storage_accounting(
            n_trials=15,
            n_channels=64,
            n_times=320,
            r_c=8,
            r_t=20,
        )
        self.assertEqual(acc["n_features"], 160)
        self.assertEqual(acc["raw_trial_elements"], 20480)
        self.assertAlmostEqual(acc["compression_ratio_vs_raw"], 128.0)
        self.assertEqual(acc["feature_bytes_per_trial"], 1280)
        # Reusable bases: (64*8 + 320*20) = 512 + 6400 = 6912 elements -> 55,296 bytes
        self.assertEqual(acc["reusable_bases_elements"], 6912)
        self.assertEqual(acc["reusable_bases_bytes"], 55296)


if __name__ == "__main__":
    unittest.main()
