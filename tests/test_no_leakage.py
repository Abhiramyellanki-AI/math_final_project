"""Milestone M2 anti-leakage tests verifying train-fold-only scaling and held-out Run 12 isolation."""

from __future__ import annotations

import unittest

import numpy as np

from eeg_p3.evaluation import build_run_swapped_splits, evaluate_baseline_spec_on_fold
from eeg_p3.features import extract_mu_beta_log_bandpower
from eeg_p3.models import BaselineModelSpec, build_baseline_pipeline


class TestM2NoLeakage(unittest.TestCase):
    """Tests proving no validation or test-run data leaks into scaler or SVM fitting."""

    def test_run_swapped_splits_reject_held_out_run_12(self) -> None:
        """Passing groups containing held-out Run 12 into build_run_swapped_splits raises ValueError."""
        groups_with_leak = np.array([4] * 15 + [8] * 15 + [12] * 15, dtype=np.int64)
        with self.assertRaises(ValueError):
            build_run_swapped_splits(groups_with_leak, development_runs=(4, 8), forbidden_test_run=12)

    def test_run_swapped_splits_exact_fold_partition(self) -> None:
        """Run-swapped splits partition Run 4 and Run 8 cleanly with zero index overlap."""
        groups = np.array([4] * 15 + [8] * 15, dtype=np.int64)
        splits = build_run_swapped_splits(groups, development_runs=(4, 8), forbidden_test_run=12)
        self.assertEqual(len(splits), 2)

        # Fold 0: train on 4, validate on 8
        self.assertEqual(splits[0].train_run_id, 4)
        self.assertEqual(splits[0].val_run_id, 8)
        self.assertTrue(set(splits[0].train_indices.tolist()).isdisjoint(set(splits[0].val_indices.tolist())))

        # Fold 1: train on 8, validate on 4
        self.assertEqual(splits[1].train_run_id, 8)
        self.assertEqual(splits[1].val_run_id, 4)
        np.testing.assert_array_equal(splits[0].train_indices, splits[1].val_indices)
        np.testing.assert_array_equal(splits[0].val_indices, splits[1].train_indices)

    def test_scaler_and_classifier_fit_are_invariant_to_validation_fold(self) -> None:
        """Perturbing validation trials never alters the learned StandardScaler or SVM parameters."""
        rng = np.random.default_rng(2026)
        X_dev = rng.normal(0.0, 20e-6, size=(20, 64, 320))
        y_dev = np.array([0, 1] * 10, dtype=np.int64)
        groups_dev = np.array([4] * 10 + [8] * 10, dtype=np.int64)

        fold0 = build_run_swapped_splits(groups_dev)[0]
        spec = BaselineModelSpec(
            model_family="baseline_linear_svc",
            config_id="LinearSVC_C=1",
            params={"C": 1.0},
            random_seed=42,
            max_iter=5000,
        )

        # Fit pipeline on training indices of original X_dev
        X_train_feat_orig = extract_mu_beta_log_bandpower(X_dev[fold0.train_indices])
        pipe_orig = build_baseline_pipeline(spec)
        pipe_orig.fit(X_train_feat_orig, y_dev[fold0.train_indices])

        # Corrupt validation fold (Run 8) with extreme values and refit on training fold
        X_dev_corrupted_val = X_dev.copy()
        X_dev_corrupted_val[fold0.val_indices] = 1e4
        X_train_feat_after = extract_mu_beta_log_bandpower(X_dev_corrupted_val[fold0.train_indices])
        pipe_after = build_baseline_pipeline(spec)
        pipe_after.fit(X_train_feat_after, y_dev[fold0.train_indices])

        np.testing.assert_allclose(
            pipe_orig.named_steps["scaler"].mean_,
            pipe_after.named_steps["scaler"].mean_,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            pipe_orig.named_steps["scaler"].scale_,
            pipe_after.named_steps["scaler"].scale_,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            pipe_orig.named_steps["clf"].coef_,
            pipe_after.named_steps["clf"].coef_,
            rtol=0.0,
            atol=1e-15,
        )

        # Verify evaluate_baseline_spec_on_fold returns all required faculty metrics
        fold_res = evaluate_baseline_spec_on_fold(
            spec=spec,
            X_raw_dev=X_dev,
            y_dev=y_dev,
            fold=fold0,
            sfreq=160.0,
            mu_band_hz=(8.0, 13.0),
            beta_band_hz=(13.0, 30.0),
            log_epsilon=1e-12,
        )
        self.assertEqual(fold_res["n_features"], 128)
        self.assertAlmostEqual(fold_res["compression_ratio_vs_raw"], 160.0)
        self.assertEqual(fold_res["feature_bytes_per_trial"], 1024)
        self.assertGreater(fold_res["model_storage_bytes"], 0)

    def test_m5_locked_selection_and_dev_refit_are_invariant_to_test_run_12(self) -> None:
        """Corrupting held-out Run 12 never changes locked config IDs or dev CV scores in M5."""
        import copy
        from dataclasses import replace
        from pathlib import Path
        from eeg_p3.config import load_config
        from eeg_p3.data import prepare_subject_dataset
        from eeg_p3.evaluation import run_m5_locked_evaluation

        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "subject01.yaml"
        base_cfg = load_config(cfg_path)
        compact_raw = copy.deepcopy(base_cfg.raw)
        compact_raw["baselines"]["linear_svc_c_grid"] = [0.1, 1.0]
        compact_raw["baselines"]["rbf_svc_c_grid"] = [1.0, 10.0]
        compact_raw["baselines"]["rbf_svc_gamma_grid"] = [0.01]
        compact_raw["hosvd"]["channel_ranks"] = [4, 16]
        compact_raw["hosvd"]["time_ranks"] = [10]
        compact_raw["hosvd"]["linear_svc_c_grid"] = [0.1]
        compact_raw["rks"]["n_components_grid"] = [128]
        compact_raw["rks"]["gamma_grid"] = [0.01, 0.1]
        compact_raw["rks"]["linear_svc_c_grid"] = [10.0]
        compact_raw["markov"]["n_states_grid"] = [3]
        compact_raw["markov"]["linear_svc_c_grid"] = [0.1, 1.0]
        cfg = replace(base_cfg, raw=compact_raw)

        dataset_orig = prepare_subject_dataset(cfg)
        locked_df_orig, rank_df_orig, _ = run_m5_locked_evaluation(dataset_orig, cfg)

        # Create a copy of dataset with held-out Run 12 signal and labels completely inverted/corrupted
        corrupted_test = replace(
            dataset_orig.test,
            X=dataset_orig.test.X * -50.0 + 1e-3,
            y=1 - dataset_orig.test.y,
        )
        dataset_corrupted = replace(dataset_orig, test=corrupted_test)

        locked_df_after, rank_df_after, _ = run_m5_locked_evaluation(dataset_corrupted, cfg)

        # Every locked configuration ID, dev CV accuracy/F1, and train reconstruction error must be identical
        self.assertEqual(
            locked_df_orig["locked_config_id"].tolist(),
            locked_df_after["locked_config_id"].tolist(),
        )
        np.testing.assert_allclose(
            locked_df_orig["dev_cv_mean_accuracy"].to_numpy(),
            locked_df_after["dev_cv_mean_accuracy"].to_numpy(),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            locked_df_orig["dev_cv_mean_f1_macro"].to_numpy(),
            locked_df_after["dev_cv_mean_f1_macro"].to_numpy(),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            rank_df_orig["train_runs_4_8_reconstruction_error"].to_numpy(),
            rank_df_after["train_runs_4_8_reconstruction_error"].to_numpy(),
            rtol=0.0,
            atol=1e-15,
        )


if __name__ == "__main__":
    unittest.main()
