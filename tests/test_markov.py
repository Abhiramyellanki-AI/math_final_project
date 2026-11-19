"""Milestone M4b unit tests for Intra-Trial Observable Markov Chain modeling."""

from __future__ import annotations

import unittest

import numpy as np

from eeg_p3.markov import (
    IntraTrialMarkovFeatureExtractor,
    compute_markov_storage_accounting,
    compute_sequence_log_likelihood,
    estimate_empirical_transition_matrix,
    fit_class_conditional_markov_model,
    slice_trial_subwindows,
)


class TestM4bIntraTrialMarkovChain(unittest.TestCase):
    """Tests verifying intra-trial boundaries, row-stochastic matrices, smoothing, and zero leakage."""

    def test_intra_trial_subwindow_slicing_boundaries(self) -> None:
        """320 samples with 40-sample window and 20-sample stride yields 15 intra-trial steps."""
        rng = np.random.default_rng(11)
        X = rng.normal(0.0, 10e-6, size=(5, 64, 320))
        sub = slice_trial_subwindows(X, subwindow_samples=40, subwindow_stride=20)
        self.assertEqual(sub.shape, (5, 15, 64, 40))

        # Verify first and last sub-windows of trial 2 match exact slices of trial 2 only
        np.testing.assert_array_equal(sub[2, 0], X[2, :, 0:40])
        np.testing.assert_array_equal(sub[2, 14], X[2, :, 280:320])

    def test_empirical_transition_matrix_is_row_stochastic_with_unseen_states(self) -> None:
        """Laplace smoothing ensures valid row-stochastic P_hat (rows sum to 1) even if states are unvisited."""
        # Sequence of length 15 that only visits states 0 and 1 out of K=4 states
        seq = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1], dtype=np.int64)
        P_hat, pi_hat = estimate_empirical_transition_matrix(seq, n_states=4, laplace_alpha=0.5)

        self.assertEqual(P_hat.shape, (4, 4))
        self.assertEqual(pi_hat.shape, (4,))
        self.assertTrue(np.all(P_hat > 0.0))
        self.assertTrue(np.all(pi_hat > 0.0))
        np.testing.assert_allclose(P_hat.sum(axis=1), np.ones(4), rtol=0.0, atol=1e-14)
        self.assertAlmostEqual(float(pi_hat.sum()), 1.0, places=14)

        # Unvisited states 2 and 3 must have uniform smoothed transition rows [0.25, 0.25, 0.25, 0.25]
        np.testing.assert_allclose(P_hat[2], np.full(4, 0.25), atol=1e-14)
        np.testing.assert_allclose(P_hat[3], np.full(4, 0.25), atol=1e-14)

    def test_class_conditional_markov_log_likelihood_discriminates_patterns(self) -> None:
        """Class-conditional Markov chains assign higher likelihood to their own transition pattern."""
        # Class 0 sequences alternate 0 -> 1 -> 0 -> 1; Class 1 sequences stay in self-loops 0..0 or 2..2
        seqs_0 = np.tile(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64), (6, 1))
        seqs_1 = np.tile(np.array([2, 2, 2, 2, 0, 0, 0, 0], dtype=np.int64), (6, 1))

        m0 = fit_class_conditional_markov_model(seqs_0, n_states=3, laplace_alpha=0.5)
        m1 = fit_class_conditional_markov_model(seqs_1, n_states=3, laplace_alpha=0.5)

        np.testing.assert_allclose(m0.transition_matrix.sum(axis=1), np.ones(3), atol=1e-14)
        np.testing.assert_allclose(m1.transition_matrix.sum(axis=1), np.ones(3), atol=1e-14)

        ll_0_on_0 = compute_sequence_log_likelihood(seqs_0[0], m0)
        ll_1_on_0 = compute_sequence_log_likelihood(seqs_0[0], m1)
        self.assertGreater(ll_0_on_0, ll_1_on_0 + 2.0)

        ll_1_on_1 = compute_sequence_log_likelihood(seqs_1[0], m1)
        ll_0_on_1 = compute_sequence_log_likelihood(seqs_1[0], m0)
        self.assertGreater(ll_1_on_1, ll_0_on_1 + 2.0)

    def test_extractor_feature_dimension_and_no_cross_trial_leakage(self) -> None:
        """IntraTrialMarkovFeatureExtractor outputs K^2 + K + 3 features/trial with zero out-of-sample leakage."""
        rng = np.random.default_rng(1234)
        X_train = rng.normal(0.0, 20e-6, size=(12, 64, 320))
        y_train = np.array([0, 1] * 6, dtype=np.int64)
        X_val = rng.normal(0.0, 20e-6, size=(6, 64, 320))

        K = 4
        ext = IntraTrialMarkovFeatureExtractor(
            n_states=K,
            subwindow_samples=40,
            subwindow_stride=20,
            laplace_alpha=0.5,
            random_state=42,
        )
        F_train = ext.fit_transform(X_train, y_train)
        F_val = ext.transform(X_val)

        expected_dim = K * K + K + 3  # 16 + 4 + 3 = 23
        self.assertEqual(F_train.shape, (12, expected_dim))
        self.assertEqual(F_val.shape, (6, expected_dim))
        self.assertTrue(np.all(np.isfinite(F_train)))
        self.assertTrue(np.all(np.isfinite(F_val)))

        # Corrupt validation trials 1..5; validation trial 0's Markov vector must remain identical
        X_val_corrupted = X_val.copy()
        X_val_corrupted[1:] = -999.0
        F_val_corrupted = ext.transform(X_val_corrupted)
        np.testing.assert_allclose(F_val[0], F_val_corrupted[0], rtol=0.0, atol=1e-14)

        acc = compute_markov_storage_accounting(n_states=K)
        self.assertEqual(acc["n_features"], 23)


if __name__ == "__main__":
    unittest.main()
