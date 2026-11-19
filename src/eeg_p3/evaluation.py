"""Leakage-free run-swapped validation and faculty metric accounting (Milestones M2-M4b).

Non-negotiable rules enforced:
- Validation uses ONLY development runs (Runs 4 and 8), swapping train/validation roles:
  * Fold 1: Train on Run 4 -> Validate on Run 8
  * Fold 2: Train on Run 8 -> Validate on Run 4
- Held-out test Run 12 is strictly forbidden during development validation.
- Feature extraction/transform time and SVM classifier fit/predict time are measured separately.
- StandardScaler, HOSVD subspaces (U_c, U_t), RKS projections, Markov quantizers/transition models,
  and SVM classifiers are fit strictly on the training fold inside each split.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from eeg_p3.config import ProjectConfig
from eeg_p3.data import PreparedEEGDataset
from eeg_p3.features import extract_mu_beta_log_bandpower
from eeg_p3.hosvd import (
    HOSVDFeatureExtractor,
    compute_hosvd_storage_accounting,
    compute_relative_frobenius_error,
)
from eeg_p3.markov import (
    IntraTrialMarkovFeatureExtractor,
    compute_markov_storage_accounting,
)
from eeg_p3.models import (
    BaselineModelSpec,
    HOSVDModelSpec,
    MarkovModelSpec,
    RKSModelSpec,
    build_baseline_pipeline,
    build_hosvd_classifier_pipeline,
    build_markov_classifier_pipeline,
    build_rks_classifier_,
    get_baseline_model_specs,
    get_hosvd_model_specs,
    get_markov_model_specs,
    get_rks_model_specs,
)
from eeg_p3.rks import RKSFeatureMapper, compute_rks_storage_accounting


@dataclass(frozen=True)
class RunFoldSplit:
    """Single run-aware training/validation split."""

    fold_index: int
    train_run_id: int
    val_run_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray


def build_run_swapped_splits(
    groups: np.ndarray,
    development_runs: tuple[int, ...] = (4, 8),
    forbidden_test_run: int = 12,
) -> list[RunFoldSplit]:
    """Construct run-swapped validation folds across development runs (4 -> 8 and 8 -> 4).

    Raises:
        ValueError: If forbidden_test_run appears in groups or any development run is missing.
    """
    unique_groups = set(int(g) for g in np.unique(groups))
    if forbidden_test_run in unique_groups:
        raise ValueError(
            f"Data leakage violation: held-out test run {forbidden_test_run} passed into "
            "development cross-validation!"
        )

    for r in development_runs:
        if r not in unique_groups:
            raise ValueError(f"Expected development run {r} not found in groups {sorted(unique_groups)}.")

    if len(development_runs) != 2:
        raise ValueError(f"Run-swapped split expects exactly 2 development runs, got {development_runs}.")

    run_a, run_b = development_runs
    pairs = [(0, run_a, run_b), (1, run_b, run_a)]
    splits: list[RunFoldSplit] = []

    for fold_idx, train_run, val_run in pairs:
        train_idx = np.where(groups == train_run)[0]
        val_idx = np.where(groups == val_run)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Empty fold indices for train_run={train_run}, val_run={val_run}.")
        splits.append(
            RunFoldSplit(
                fold_index=fold_idx,
                train_run_id=int(train_run),
                val_run_id=int(val_run),
                train_indices=train_idx,
                val_indices=val_idx,
            )
        )

    return splits


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute accuracy, per-class and macro precision/recall/F1, and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_class_0_left": float(p_cls[0]),
        "recall_class_0_left": float(r_cls[0]),
        "f1_class_0_left": float(f1_cls[0]),
        "support_class_0_left": int(support_cls[0]),
        "precision_class_1_right": float(p_cls[1]),
        "recall_class_1_right": float(r_cls[1]),
        "f1_class_1_right": float(f1_cls[1]),
        "support_class_1_right": int(support_cls[1]),
        "confusion_matrix": cm.tolist(),
    }


def estimate_fitted_pipeline_storage_bytes(pipeline: Pipeline) -> int:
    """Compute exact byte footprint of fitted StandardScaler and SVM parameters."""
    total_bytes = 0
    for _, step in pipeline.named_steps.items():
        for attr in ("mean_", "scale_", "var_", "coef_", "intercept_", "support_vectors_", "dual_coef_"):
            val = getattr(step, attr, None)
            if isinstance(val, np.ndarray):
                total_bytes += int(val.nbytes)
    return total_bytes


def estimate_linear_svc_storage_bytes(clf: Any) -> int:
    """Compute exact byte footprint of fitted LinearSVC coef_ and intercept_."""
    total_bytes = 0
    for attr in ("coef_", "intercept_"):
        val = getattr(clf, attr, None)
        if isinstance(val, np.ndarray):
            total_bytes += int(val.nbytes)
    return total_bytes


def evaluate_baseline_spec_on_fold(
    spec: BaselineModelSpec,
    X_raw_dev: np.ndarray,
    y_dev: np.ndarray,
    fold: RunFoldSplit,
    sfreq: float,
    mu_band_hz: tuple[float, float],
    beta_band_hz: tuple[float, float],
    log_epsilon: float,
) -> dict[str, Any]:
    """Evaluate one baseline configuration on one run-swapped fold with separate timings."""
    X_train_raw = X_raw_dev[fold.train_indices]
    y_train = y_dev[fold.train_indices]
    X_val_raw = X_raw_dev[fold.val_indices]
    y_val = y_dev[fold.val_indices]

    t0 = time.perf_counter()
    X_train_feat = extract_mu_beta_log_bandpower(
        X_train_raw,
        sfreq=sfreq,
        mu_band_hz=mu_band_hz,
        beta_band_hz=beta_band_hz,
        log_epsilon=log_epsilon,
    )
    train_feat_time_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    X_val_feat = extract_mu_beta_log_bandpower(
        X_val_raw,
        sfreq=sfreq,
        mu_band_hz=mu_band_hz,
        beta_band_hz=beta_band_hz,
        log_epsilon=log_epsilon,
    )
    val_feat_time_sec = time.perf_counter() - t1

    pipeline = build_baseline_pipeline(spec)
    t2 = time.perf_counter()
    pipeline.fit(X_train_feat, y_train)
    svm_train_time_sec = time.perf_counter() - t2

    t3 = time.perf_counter()
    y_pred = pipeline.predict(X_val_feat)
    svm_predict_time_sec = time.perf_counter() - t3

    metrics = compute_classification_metrics(y_val, y_pred)
    n_features = int(X_train_feat.shape[1])
    raw_elements_per_trial = int(X_raw_dev.shape[1] * X_raw_dev.shape[2])
    compression_ratio_vs_raw = float(raw_elements_per_trial / n_features)
    feature_bytes_per_trial = int(n_features * 8)
    model_storage_bytes = estimate_fitted_pipeline_storage_bytes(pipeline)

    return {
        "model_family": spec.model_family,
        "config_id": spec.config_id,
        "params_json": json.dumps(spec.params, sort_keys=True),
        "fold_index": fold.fold_index,
        "train_run": fold.train_run_id,
        "val_run": fold.val_run_id,
        "n_train_trials": int(len(y_train)),
        "n_val_trials": int(len(y_val)),
        "n_features": n_features,
        "compression_ratio_vs_raw": compression_ratio_vs_raw,
        "hosvd_reconstruction_error": None,
        "feature_bytes_per_trial": feature_bytes_per_trial,
        "model_storage_bytes": model_storage_bytes,
        "train_feature_time_sec": float(train_feat_time_sec),
        "svm_train_time_sec": float(svm_train_time_sec),
        "val_feature_time_sec": float(val_feat_time_sec),
        "svm_predict_time_sec": float(svm_predict_time_sec),
        "total_prediction_time_sec": float(val_feat_time_sec + svm_predict_time_sec),
        **metrics,
    }


def run_m2_baselines_validation(
    dataset: PreparedEEGDataset,
    cfg: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run M2 baseline evaluations strictly on development Runs 4 and 8 (Run 12 untouched)."""
    splits = build_run_swapped_splits(
        groups=dataset.dev.groups,
        development_runs=cfg.data.development_runs,
        forbidden_test_run=cfg.data.held_out_test_run,
    )

    feat_cfg = cfg.raw.get("features", {})
    mu_band = tuple(float(x) for x in feat_cfg.get("mu_band_hz", [8.0, 13.0]))
    beta_band = tuple(float(x) for x in feat_cfg.get("beta_band_hz", [13.0, 30.0]))
    log_eps = float(feat_cfg.get("log_epsilon", 1e-12))

    specs = get_baseline_model_specs(cfg)
    fold_records: list[dict[str, Any]] = []

    for spec in specs:
        for fold in splits:
            record = evaluate_baseline_spec_on_fold(
                spec=spec,
                X_raw_dev=dataset.dev.X,
                y_dev=dataset.dev.y,
                fold=fold,
                sfreq=dataset.dev.sfreq,
                mu_band_hz=(mu_band[0], mu_band[1]),
                beta_band_hz=(beta_band[0], beta_band[1]),
                log_epsilon=log_eps,
            )
            fold_records.append(record)

    fold_df = pd.DataFrame(fold_records)

    summary_records: list[dict[str, Any]] = []
    for config_id, grp in fold_df.groupby("config_id", sort=False):
        first = grp.iloc[0]
        summary_records.append(
            {
                "model_family": str(first["model_family"]),
                "config_id": str(config_id),
                "params_json": str(first["params_json"]),
                "n_features": int(first["n_features"]),
                "compression_ratio_vs_raw": float(first["compression_ratio_vs_raw"]),
                "feature_bytes_per_trial": int(first["feature_bytes_per_trial"]),
                "model_storage_bytes_mean": float(grp["model_storage_bytes"].mean()),
                "mean_accuracy": float(grp["accuracy"].mean()),
                "std_accuracy": float(grp["accuracy"].std(ddof=0)),
                "fold1_run4to8_accuracy": float(grp.loc[grp["fold_index"] == 0, "accuracy"].iloc[0]),
                "fold2_run8to4_accuracy": float(grp.loc[grp["fold_index"] == 1, "accuracy"].iloc[0]),
                "mean_precision_macro": float(grp["precision_macro"].mean()),
                "mean_recall_macro": float(grp["recall_macro"].mean()),
                "mean_f1_macro": float(grp["f1_macro"].mean()),
                "std_f1_macro": float(grp["f1_macro"].std(ddof=0)),
                "mean_train_feature_time_sec": float(grp["train_feature_time_sec"].mean()),
                "mean_svm_train_time_sec": float(grp["svm_train_time_sec"].mean()),
                "mean_val_feature_time_sec": float(grp["val_feature_time_sec"].mean()),
                "mean_svm_predict_time_sec": float(grp["svm_predict_time_sec"].mean()),
                "mean_total_prediction_time_sec": float(grp["total_prediction_time_sec"].mean()),
            }
        )

    summary_df = pd.DataFrame(summary_records)

    best_linear = (
        summary_df[summary_df["model_family"] == "baseline_linear_svc"]
        .sort_values(by=["mean_f1_macro", "mean_accuracy"], ascending=False)
        .iloc[0]
        .to_dict()
    )
    best_rbf = (
        summary_df[summary_df["model_family"] == "baseline_rbf_svc"]
        .sort_values(by=["mean_f1_macro", "mean_accuracy"], ascending=False)
        .iloc[0]
        .to_dict()
    )

    report_dict: dict[str, Any] = {
        "milestone": "M2 (Deterministic Spectral Baselines — Development Runs 4 & 8 Only)",
        "subject_id": dataset.subject_id,
        "development_runs_used": list(dataset.dev.run_ids),
        "held_out_test_run_untouched": True,
        "held_out_test_run_id": cfg.data.held_out_test_run,
        "feature_representation": {
            "type": "mu_beta_log10_bandpower_welch",
            "mu_band_hz": list(mu_band),
            "beta_band_hz": list(beta_band),
            "n_channels": len(dataset.dev.channel_names),
            "n_features_per_trial": 2 * len(dataset.dev.channel_names),
            "raw_elements_per_trial": int(dataset.dev.X.shape[1] * dataset.dev.X.shape[2]),
            "compression_ratio_vs_raw": float(
                (dataset.dev.X.shape[1] * dataset.dev.X.shape[2]) / (2 * len(dataset.dev.channel_names))
            ),
        },
        "validation_protocol": (
            "Run-swapped grouped validation on Runs 4 and 8 "
            "(Fold 0: Train Run 4 -> Val Run 8; Fold 1: Train Run 8 -> Val Run 4). "
            "Caution: With only 2 development runs (15 trials/run), validation variance is high."
        ),
        "best_linear_svc_on_dev": best_linear,
        "best_rbf_svc_on_dev": best_rbf,
        "summary_table": summary_records,
        "per_fold_records": fold_records,
    }

    return fold_df, summary_df, report_dict


def evaluate_hosvd_spec_on_fold(
    spec: HOSVDModelSpec,
    X_raw_dev: np.ndarray,
    y_dev: np.ndarray,
    fold: RunFoldSplit,
) -> dict[str, Any]:
    """Evaluate one HOSVD + LinearSVC configuration on one run-swapped fold with full accounting."""
    X_train_raw = X_raw_dev[fold.train_indices]
    y_train = y_dev[fold.train_indices]
    X_val_raw = X_raw_dev[fold.val_indices]
    y_val = y_dev[fold.val_indices]

    extractor = HOSVDFeatureExtractor(r_c=spec.r_c, r_t=spec.r_t)
    t0 = time.perf_counter()
    X_train_feat = extractor.fit_transform(X_train_raw)
    train_feat_time_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    X_val_feat = extractor.transform(X_val_raw)
    val_feat_time_sec = time.perf_counter() - t1

    train_recon_err = compute_relative_frobenius_error(X_train_raw, extractor.U_c_, extractor.U_t_)
    val_recon_err = compute_relative_frobenius_error(X_val_raw, extractor.U_c_, extractor.U_t_)

    pipeline = build_hosvd_classifier_pipeline(spec)
    t2 = time.perf_counter()
    pipeline.fit(X_train_feat, y_train)
    svm_train_time_sec = time.perf_counter() - t2

    t3 = time.perf_counter()
    y_pred = pipeline.predict(X_val_feat)
    svm_predict_time_sec = time.perf_counter() - t3

    metrics = compute_classification_metrics(y_val, y_pred)
    storage = compute_hosvd_storage_accounting(
        n_trials=len(y_train),
        n_channels=int(X_raw_dev.shape[1]),
        n_times=int(X_raw_dev.shape[2]),
        r_c=spec.r_c,
        r_t=spec.r_t,
    )
    scaler_svm_bytes = estimate_fitted_pipeline_storage_bytes(pipeline)
    total_deployed_bytes = int(storage["reusable_bases_bytes"] + scaler_svm_bytes)

    return {
        "model_family": spec.model_family,
        "config_id": spec.config_id,
        "r_c": spec.r_c,
        "r_t": spec.r_t,
        "C": spec.c_val,
        "params_json": json.dumps(spec.params, sort_keys=True),
        "fold_index": fold.fold_index,
        "train_run": fold.train_run_id,
        "val_run": fold.val_run_id,
        "n_train_trials": int(len(y_train)),
        "n_val_trials": int(len(y_val)),
        "n_features": int(storage["n_features"]),
        "compression_ratio_vs_raw": float(storage["compression_ratio_vs_raw"]),
        "tensor_decomposition_compression_ratio": float(
            storage["tensor_decomposition_compression_ratio"]
        ),
        "hosvd_train_reconstruction_error": float(train_recon_err),
        "hosvd_reconstruction_error": float(val_recon_err),
        "feature_bytes_per_trial": int(storage["feature_bytes_per_trial"]),
        "reusable_bases_bytes": int(storage["reusable_bases_bytes"]),
        "scaler_svm_storage_bytes": int(scaler_svm_bytes),
        "model_storage_bytes": total_deployed_bytes,
        "decomposed_tensor_total_bytes": int(storage["decomposed_tensor_total_bytes"]),
        "train_feature_time_sec": float(train_feat_time_sec),
        "svm_train_time_sec": float(svm_train_time_sec),
        "val_feature_time_sec": float(val_feat_time_sec),
        "svm_predict_time_sec": float(svm_predict_time_sec),
        "total_prediction_time_sec": float(val_feat_time_sec + svm_predict_time_sec),
        **metrics,
    }


def run_m3_hosvd_validation(
    dataset: PreparedEEGDataset,
    cfg: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run M3 HOSVD rank sweep and LinearSVC validation on Runs 4 and 8 (Run 12 untouched)."""
    splits = build_run_swapped_splits(
        groups=dataset.dev.groups,
        development_runs=cfg.data.development_runs,
        forbidden_test_run=cfg.data.held_out_test_run,
    )

    specs = get_hosvd_model_specs(cfg)
    fold_records: list[dict[str, Any]] = []

    for spec in specs:
        for fold in splits:
            record = evaluate_hosvd_spec_on_fold(
                spec=spec,
                X_raw_dev=dataset.dev.X,
                y_dev=dataset.dev.y,
                fold=fold,
            )
            fold_records.append(record)

    fold_df = pd.DataFrame(fold_records)

    summary_records: list[dict[str, Any]] = []
    for config_id, grp in fold_df.groupby("config_id", sort=False):
        first = grp.iloc[0]
        summary_records.append(
            {
                "model_family": str(first["model_family"]),
                "config_id": str(config_id),
                "r_c": int(first["r_c"]),
                "r_t": int(first["r_t"]),
                "C": float(first["C"]),
                "params_json": str(first["params_json"]),
                "n_features": int(first["n_features"]),
                "compression_ratio_vs_raw": float(first["compression_ratio_vs_raw"]),
                "tensor_decomposition_compression_ratio": float(
                    first["tensor_decomposition_compression_ratio"]
                ),
                "mean_hosvd_train_reconstruction_error": float(
                    grp["hosvd_train_reconstruction_error"].mean()
                ),
                "mean_hosvd_val_reconstruction_error": float(
                    grp["hosvd_reconstruction_error"].mean()
                ),
                "feature_bytes_per_trial": int(first["feature_bytes_per_trial"]),
                "reusable_bases_bytes": int(first["reusable_bases_bytes"]),
                "scaler_svm_storage_bytes_mean": float(grp["scaler_svm_storage_bytes"].mean()),
                "model_storage_bytes_mean": float(grp["model_storage_bytes"].mean()),
                "decomposed_tensor_total_bytes": int(first["decomposed_tensor_total_bytes"]),
                "mean_accuracy": float(grp["accuracy"].mean()),
                "std_accuracy": float(grp["accuracy"].std(ddof=0)),
                "fold1_run4to8_accuracy": float(grp.loc[grp["fold_index"] == 0, "accuracy"].iloc[0]),
                "fold2_run8to4_accuracy": float(grp.loc[grp["fold_index"] == 1, "accuracy"].iloc[0]),
                "mean_precision_macro": float(grp["precision_macro"].mean()),
                "mean_recall_macro": float(grp["recall_macro"].mean()),
                "mean_f1_macro": float(grp["f1_macro"].mean()),
                "std_f1_macro": float(grp["f1_macro"].std(ddof=0)),
                "mean_train_feature_time_sec": float(grp["train_feature_time_sec"].mean()),
                "mean_svm_train_time_sec": float(grp["svm_train_time_sec"].mean()),
                "mean_val_feature_time_sec": float(grp["val_feature_time_sec"].mean()),
                "mean_svm_predict_time_sec": float(grp["svm_predict_time_sec"].mean()),
                "mean_total_prediction_time_sec": float(grp["total_prediction_time_sec"].mean()),
            }
        )

    summary_df = pd.DataFrame(summary_records)

    rank_records: list[dict[str, Any]] = []
    for (r_c, r_t), rgrp in summary_df.groupby(["r_c", "r_t"], sort=False):
        best_for_rank = rgrp.sort_values(by=["mean_f1_macro", "mean_accuracy"], ascending=False).iloc[0]
        rank_records.append(
            {
                "r_c": int(r_c),
                "r_t": int(r_t),
                "n_features": int(best_for_rank["n_features"]),
                "compression_ratio_vs_raw": float(best_for_rank["compression_ratio_vs_raw"]),
                "tensor_decomposition_compression_ratio": float(
                    best_for_rank["tensor_decomposition_compression_ratio"]
                ),
                "mean_hosvd_train_reconstruction_error": float(
                    best_for_rank["mean_hosvd_train_reconstruction_error"]
                ),
                "mean_hosvd_val_reconstruction_error": float(
                    best_for_rank["mean_hosvd_val_reconstruction_error"]
                ),
                "feature_bytes_per_trial": int(best_for_rank["feature_bytes_per_trial"]),
                "reusable_bases_bytes": int(best_for_rank["reusable_bases_bytes"]),
                "decomposed_tensor_total_bytes": int(best_for_rank["decomposed_tensor_total_bytes"]),
                "best_C": float(best_for_rank["C"]),
                "best_mean_accuracy": float(best_for_rank["mean_accuracy"]),
                "best_mean_f1_macro": float(best_for_rank["mean_f1_macro"]),
            }
        )
    rank_df = pd.DataFrame(rank_records)

    best_hosvd = (
        summary_df.sort_values(
            by=["mean_f1_macro", "mean_accuracy", "compression_ratio_vs_raw"],
            ascending=[False, False, False],
        )
        .iloc[0]
        .to_dict()
    )

    report_dict: dict[str, Any] = {
        "milestone": "M3 (Train-Only HOSVD Subspace Learning & Rank Sweep — Development Runs 4 & 8 Only)",
        "subject_id": dataset.subject_id,
        "development_runs_used": list(dataset.dev.run_ids),
        "held_out_test_run_untouched": True,
        "held_out_test_run_id": cfg.data.held_out_test_run,
        "storage_accounting_definitions": {
            "per_trial_deployed_extractor": (
                "Per-trial projected feature vector (8 * r_c * r_t bytes) plus reusable "
                "projection bases U_c and U_t (8 * (C*r_c + T*r_t) bytes) and StandardScaler + LinearSVC weights."
            ),
            "tensor_decomposition_storage": (
                "N_train projected cores (8 * N * r_c * r_t bytes) plus factor matrices "
                "U_c and U_t (8 * (C*r_c + T*r_t) bytes), compared against raw N*C*T tensor."
            ),
        },
        "best_hosvd_on_dev": best_hosvd,
        "rank_sweep_summary": rank_records,
        "summary_table": summary_records,
        "per_fold_records": fold_records,
    }

    return fold_df, summary_df, rank_df, report_dict


def evaluate_rks_spec_on_fold(
    spec: RKSModelSpec,
    X_raw_dev: np.ndarray,
    y_dev: np.ndarray,
    fold: RunFoldSplit,
    sfreq: float,
    mu_band_hz: tuple[float, float],
    beta_band_hz: tuple[float, float],
    log_epsilon: float,
) -> dict[str, Any]:
    """Evaluate one RKS (StandardScaler -> RBFSampler -> LinearSVC) config on one fold."""
    X_train_raw = X_raw_dev[fold.train_indices]
    y_train = y_dev[fold.train_indices]
    X_val_raw = X_raw_dev[fold.val_indices]
    y_val = y_dev[fold.val_indices]

    # 1. Measure training feature extraction (bandpower + RKS random Fourier mapping)
    rks_mapper = RKSFeatureMapper(
        n_components=spec.n_components,
        gamma=spec.gamma,
        random_state=spec.random_seed,
    )
    t0 = time.perf_counter()
    X_train_bp = extract_mu_beta_log_bandpower(
        X_train_raw,
        sfreq=sfreq,
        mu_band_hz=mu_band_hz,
        beta_band_hz=beta_band_hz,
        log_epsilon=log_epsilon,
    )
    X_train_rks = rks_mapper.fit_transform(X_train_bp)
    train_feat_time_sec = time.perf_counter() - t0

    # 2. Measure validation feature extraction (bandpower + frozen RKS transform)
    t1 = time.perf_counter()
    X_val_bp = extract_mu_beta_log_bandpower(
        X_val_raw,
        sfreq=sfreq,
        mu_band_hz=mu_band_hz,
        beta_band_hz=beta_band_hz,
        log_epsilon=log_epsilon,
    )
    X_val_rks = rks_mapper.transform(X_val_bp)
    val_feat_time_sec = time.perf_counter() - t1

    # 3. Fit LinearSVC on D-dimensional RKS features
    clf = build_rks_classifier_(spec)
    t2 = time.perf_counter()
    clf.fit(X_train_rks, y_train)
    svm_train_time_sec = time.perf_counter() - t2

    # 4. Predict on validation RKS features
    t3 = time.perf_counter()
    y_pred = clf.predict(X_val_rks)
    svm_predict_time_sec = time.perf_counter() - t3

    metrics = compute_classification_metrics(y_val, y_pred)
    raw_elements_per_trial = int(X_raw_dev.shape[1] * X_raw_dev.shape[2])
    storage = compute_rks_storage_accounting(
        input_dim=int(X_train_bp.shape[1]),
        n_components=spec.n_components,
        raw_trial_elements=raw_elements_per_trial,
    )
    svm_bytes = estimate_linear_svc_storage_bytes(clf)
    total_model_storage_bytes = int(rks_mapper.storage_bytes() + svm_bytes)

    return {
        "model_family": spec.model_family,
        "config_id": spec.config_id,
        "n_components": spec.n_components,
        "gamma": spec.gamma,
        "C": spec.c_val,
        "params_json": json.dumps(spec.params, sort_keys=True),
        "fold_index": fold.fold_index,
        "train_run": fold.train_run_id,
        "val_run": fold.val_run_id,
        "n_train_trials": int(len(y_train)),
        "n_val_trials": int(len(y_val)),
        "input_bandpower_features": int(storage["input_bandpower_features"]),
        "n_features": int(storage["n_features"]),
        "expansion_ratio_vs_bandpower": float(storage["expansion_ratio_vs_bandpower"]),
        "compression_ratio_vs_raw": float(storage["compression_ratio_vs_raw"]),
        "hosvd_reconstruction_error": None,
        "feature_bytes_per_trial": int(storage["feature_bytes_per_trial"]),
        "rks_weights_bytes": int(storage["rks_weights_bytes"]),
        "svm_storage_bytes": int(svm_bytes),
        "model_storage_bytes": total_model_storage_bytes,
        "train_feature_time_sec": float(train_feat_time_sec),
        "svm_train_time_sec": float(svm_train_time_sec),
        "val_feature_time_sec": float(val_feat_time_sec),
        "svm_predict_time_sec": float(svm_predict_time_sec),
        "total_prediction_time_sec": float(val_feat_time_sec + svm_predict_time_sec),
        **metrics,
    }


def run_m4_rks_validation(
    dataset: PreparedEEGDataset,
    cfg: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run M4 Random Kitchen Sinks parameter sweep on Runs 4 and 8 (Run 12 untouched)."""
    splits = build_run_swapped_splits(
        groups=dataset.dev.groups,
        development_runs=cfg.data.development_runs,
        forbidden_test_run=cfg.data.held_out_test_run,
    )

    feat_cfg = cfg.raw.get("features", {})
    mu_band = tuple(float(x) for x in feat_cfg.get("mu_band_hz", [8.0, 13.0]))
    beta_band = tuple(float(x) for x in feat_cfg.get("beta_band_hz", [13.0, 30.0]))
    log_eps = float(feat_cfg.get("log_epsilon", 1e-12))

    specs = get_rks_model_specs(cfg)
    fold_records: list[dict[str, Any]] = []

    for spec in specs:
        for fold in splits:
            record = evaluate_rks_spec_on_fold(
                spec=spec,
                X_raw_dev=dataset.dev.X,
                y_dev=dataset.dev.y,
                fold=fold,
                sfreq=dataset.dev.sfreq,
                mu_band_hz=(mu_band[0], mu_band[1]),
                beta_band_hz=(beta_band[0], beta_band[1]),
                log_epsilon=log_eps,
            )
            fold_records.append(record)

    fold_df = pd.DataFrame(fold_records)

    summary_records: list[dict[str, Any]] = []
    for config_id, grp in fold_df.groupby("config_id", sort=False):
        first = grp.iloc[0]
        summary_records.append(
            {
                "model_family": str(first["model_family"]),
                "config_id": str(config_id),
                "n_components": int(first["n_components"]),
                "gamma": float(first["gamma"]),
                "C": float(first["C"]),
                "params_json": str(first["params_json"]),
                "input_bandpower_features": int(first["input_bandpower_features"]),
                "n_features": int(first["n_features"]),
                "expansion_ratio_vs_bandpower": float(first["expansion_ratio_vs_bandpower"]),
                "compression_ratio_vs_raw": float(first["compression_ratio_vs_raw"]),
                "feature_bytes_per_trial": int(first["feature_bytes_per_trial"]),
                "rks_weights_bytes": int(first["rks_weights_bytes"]),
                "model_storage_bytes_mean": float(grp["model_storage_bytes"].mean()),
                "mean_accuracy": float(grp["accuracy"].mean()),
                "std_accuracy": float(grp["accuracy"].std(ddof=0)),
                "fold1_run4to8_accuracy": float(grp.loc[grp["fold_index"] == 0, "accuracy"].iloc[0]),
                "fold2_run8to4_accuracy": float(grp.loc[grp["fold_index"] == 1, "accuracy"].iloc[0]),
                "mean_precision_macro": float(grp["precision_macro"].mean()),
                "mean_recall_macro": float(grp["recall_macro"].mean()),
                "mean_f1_macro": float(grp["f1_macro"].mean()),
                "std_f1_macro": float(grp["f1_macro"].std(ddof=0)),
                "mean_train_feature_time_sec": float(grp["train_feature_time_sec"].mean()),
                "mean_svm_train_time_sec": float(grp["svm_train_time_sec"].mean()),
                "mean_val_feature_time_sec": float(grp["val_feature_time_sec"].mean()),
                "mean_svm_predict_time_sec": float(grp["svm_predict_time_sec"].mean()),
                "mean_total_prediction_time_sec": float(grp["total_prediction_time_sec"].mean()),
            }
        )

    summary_df = pd.DataFrame(summary_records)

    best_rks = (
        summary_df.sort_values(
            by=["mean_f1_macro", "mean_accuracy"],
            ascending=[False, False],
        )
        .iloc[0]
        .to_dict()
    )

    report_dict: dict[str, Any] = {
        "milestone": "M4 (Random Kitchen Sinks / Random Fourier Features — Development Runs 4 & 8 Only)",
        "subject_id": dataset.subject_id,
        "development_runs_used": list(dataset.dev.run_ids),
        "held_out_test_run_untouched": True,
        "held_out_test_run_id": cfg.data.held_out_test_run,
        "random_seed": cfg.random_seed,
        "rks_note": (
            "RKS (RBFSampler) maps 128-D standardized log-bandpower features into D in {128, 256, 512} "
            "explicit random Fourier features to approximate an RBF kernel for LinearSVC. "
            "It is kernel approximation (feature expansion when D > 128), not inherently dimensionality reduction."
        ),
        "best_rks_on_dev": best_rks,
        "summary_table": summary_records,
        "per_fold_records": fold_records,
    }

    return fold_df, summary_df, report_dict


def evaluate_markov_spec_on_fold(
    spec: MarkovModelSpec,
    X_raw_dev: np.ndarray,
    y_dev: np.ndarray,
    fold: RunFoldSplit,
    sfreq: float,
    mu_band_hz: tuple[float, float],
    beta_band_hz: tuple[float, float],
    log_epsilon: float,
) -> dict[str, Any]:
    """Evaluate one standalone or hybrid Intra-Trial Markov Chain configuration on one fold."""
    X_train_raw = X_raw_dev[fold.train_indices]
    y_train = y_dev[fold.train_indices]
    X_val_raw = X_raw_dev[fold.val_indices]
    y_val = y_dev[fold.val_indices]

    markov_ext = IntraTrialMarkovFeatureExtractor(
        n_states=spec.n_states,
        subwindow_samples=spec.subwindow_samples,
        subwindow_stride=spec.subwindow_stride,
        spatial_rank_sub=4,
        laplace_alpha=spec.laplace_alpha,
        random_state=spec.random_seed,
    )
    markov_storage = compute_markov_storage_accounting(
        n_states=spec.n_states,
        n_channels=int(X_raw_dev.shape[1]),
        spatial_rank_sub=4,
        raw_trial_elements=int(X_raw_dev.shape[1] * X_raw_dev.shape[2]),
    )

    hosvd_val_recon_err: float | None = None
    extra_extractor_bytes = int(markov_storage["markov_extractor_bytes"])

    # 1. Fit feature extractors strictly on the training fold
    t0 = time.perf_counter()
    X_train_markov = markov_ext.fit_transform(X_train_raw, y_train)

    if spec.fusion_mode == "standalone":
        X_train_feat = X_train_markov
    elif spec.fusion_mode == "hosvd_fusion":
        assert spec.r_c is not None and spec.r_t is not None
        hosvd_ext = HOSVDFeatureExtractor(r_c=spec.r_c, r_t=spec.r_t)
        X_train_hosvd = hosvd_ext.fit_transform(X_train_raw)
        X_train_feat = np.concatenate([X_train_hosvd, X_train_markov], axis=1)
        hosvd_storage = compute_hosvd_storage_accounting(
            n_trials=len(y_train),
            n_channels=int(X_raw_dev.shape[1]),
            n_times=int(X_raw_dev.shape[2]),
            r_c=spec.r_c,
            r_t=spec.r_t,
        )
        extra_extractor_bytes += int(hosvd_storage["reusable_bases_bytes"])
    elif spec.fusion_mode == "rks_fusion":
        assert spec.rks_dim is not None and spec.rks_gamma is not None
        X_train_bp = extract_mu_beta_log_bandpower(
            X_train_raw,
            sfreq=sfreq,
            mu_band_hz=mu_band_hz,
            beta_band_hz=beta_band_hz,
            log_epsilon=log_epsilon,
        )
        rks_ext = RKSFeatureMapper(
            n_components=spec.rks_dim,
            gamma=spec.rks_gamma,
            random_state=spec.random_seed,
        )
        X_train_rks = rks_ext.fit_transform(X_train_bp)
        X_train_feat = np.concatenate([X_train_rks, X_train_markov], axis=1)
        extra_extractor_bytes += int(rks_ext.storage_bytes())
    else:
        raise ValueError(f"Unknown fusion_mode: '{spec.fusion_mode}'")

    train_feat_time_sec = time.perf_counter() - t0

    # 2. Transform validation fold using frozen training extractors
    t1 = time.perf_counter()
    X_val_markov = markov_ext.transform(X_val_raw)

    if spec.fusion_mode == "standalone":
        X_val_feat = X_val_markov
    elif spec.fusion_mode == "hosvd_fusion":
        X_val_hosvd = hosvd_ext.transform(X_val_raw)
        X_val_feat = np.concatenate([X_val_hosvd, X_val_markov], axis=1)
    elif spec.fusion_mode == "rks_fusion":
        X_val_bp = extract_mu_beta_log_bandpower(
            X_val_raw,
            sfreq=sfreq,
            mu_band_hz=mu_band_hz,
            beta_band_hz=beta_band_hz,
            log_epsilon=log_epsilon,
        )
        X_val_rks = rks_ext.transform(X_val_bp)
        X_val_feat = np.concatenate([X_val_rks, X_val_markov], axis=1)

    val_feat_time_sec = time.perf_counter() - t1

    if spec.fusion_mode == "hosvd_fusion":
        hosvd_val_recon_err = compute_relative_frobenius_error(
            X_val_raw, hosvd_ext.U_c_, hosvd_ext.U_t_
        )

    # 3. Fit train-only StandardScaler + LinearSVC
    pipeline = build_markov_classifier_pipeline(spec)
    t2 = time.perf_counter()
    pipeline.fit(X_train_feat, y_train)
    svm_train_time_sec = time.perf_counter() - t2

    # 4. Predict on validation fold
    t3 = time.perf_counter()
    y_pred = pipeline.predict(X_val_feat)
    svm_predict_time_sec = time.perf_counter() - t3

    metrics = compute_classification_metrics(y_val, y_pred)
    n_features = int(X_train_feat.shape[1])
    raw_elements_per_trial = int(X_raw_dev.shape[1] * X_raw_dev.shape[2])
    compression_ratio_vs_raw = float(raw_elements_per_trial / n_features)
    feature_bytes_per_trial = int(n_features * 8)
    scaler_svm_bytes = estimate_fitted_pipeline_storage_bytes(pipeline)
    total_model_storage_bytes = int(extra_extractor_bytes + scaler_svm_bytes)

    return {
        "model_family": spec.model_family,
        "config_id": spec.config_id,
        "fusion_mode": spec.fusion_mode,
        "n_states": spec.n_states,
        "C": spec.c_val,
        "params_json": json.dumps(spec.params, sort_keys=True),
        "fold_index": fold.fold_index,
        "train_run": fold.train_run_id,
        "val_run": fold.val_run_id,
        "n_train_trials": int(len(y_train)),
        "n_val_trials": int(len(y_val)),
        "n_features": n_features,
        "compression_ratio_vs_raw": compression_ratio_vs_raw,
        "hosvd_reconstruction_error": hosvd_val_recon_err,
        "feature_bytes_per_trial": feature_bytes_per_trial,
        "extractor_storage_bytes": int(extra_extractor_bytes),
        "scaler_svm_storage_bytes": int(scaler_svm_bytes),
        "model_storage_bytes": total_model_storage_bytes,
        "train_feature_time_sec": float(train_feat_time_sec),
        "svm_train_time_sec": float(svm_train_time_sec),
        "val_feature_time_sec": float(val_feat_time_sec),
        "svm_predict_time_sec": float(svm_predict_time_sec),
        "total_prediction_time_sec": float(val_feat_time_sec + svm_predict_time_sec),
        **metrics,
    }


def run_m4b_markov_validation(
    dataset: PreparedEEGDataset,
    cfg: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run M4b Intra-Trial Markov Chain standalone and hybrid validations on Runs 4 & 8."""
    splits = build_run_swapped_splits(
        groups=dataset.dev.groups,
        development_runs=cfg.data.development_runs,
        forbidden_test_run=cfg.data.held_out_test_run,
    )

    feat_cfg = cfg.raw.get("features", {})
    mu_band = tuple(float(x) for x in feat_cfg.get("mu_band_hz", [8.0, 13.0]))
    beta_band = tuple(float(x) for x in feat_cfg.get("beta_band_hz", [13.0, 30.0]))
    log_eps = float(feat_cfg.get("log_epsilon", 1e-12))

    specs = get_markov_model_specs(cfg)
    fold_records: list[dict[str, Any]] = []

    for spec in specs:
        for fold in splits:
            record = evaluate_markov_spec_on_fold(
                spec=spec,
                X_raw_dev=dataset.dev.X,
                y_dev=dataset.dev.y,
                fold=fold,
                sfreq=dataset.dev.sfreq,
                mu_band_hz=(mu_band[0], mu_band[1]),
                beta_band_hz=(beta_band[0], beta_band[1]),
                log_epsilon=log_eps,
            )
            fold_records.append(record)

    fold_df = pd.DataFrame(fold_records)

    summary_records: list[dict[str, Any]] = []
    for config_id, grp in fold_df.groupby("config_id", sort=False):
        first = grp.iloc[0]
        recon_vals = grp["hosvd_reconstruction_error"].dropna()
        mean_recon = float(recon_vals.mean()) if len(recon_vals) > 0 else None
        summary_records.append(
            {
                "model_family": str(first["model_family"]),
                "config_id": str(config_id),
                "fusion_mode": str(first["fusion_mode"]),
                "n_states": int(first["n_states"]),
                "C": float(first["C"]),
                "params_json": str(first["params_json"]),
                "n_features": int(first["n_features"]),
                "compression_ratio_vs_raw": float(first["compression_ratio_vs_raw"]),
                "mean_hosvd_val_reconstruction_error": mean_recon,
                "feature_bytes_per_trial": int(first["feature_bytes_per_trial"]),
                "model_storage_bytes_mean": float(grp["model_storage_bytes"].mean()),
                "mean_accuracy": float(grp["accuracy"].mean()),
                "std_accuracy": float(grp["accuracy"].std(ddof=0)),
                "fold1_run4to8_accuracy": float(grp.loc[grp["fold_index"] == 0, "accuracy"].iloc[0]),
                "fold2_run8to4_accuracy": float(grp.loc[grp["fold_index"] == 1, "accuracy"].iloc[0]),
                "mean_precision_macro": float(grp["precision_macro"].mean()),
                "mean_recall_macro": float(grp["recall_macro"].mean()),
                "mean_f1_macro": float(grp["f1_macro"].mean()),
                "std_f1_macro": float(grp["f1_macro"].std(ddof=0)),
                "mean_train_feature_time_sec": float(grp["train_feature_time_sec"].mean()),
                "mean_svm_train_time_sec": float(grp["svm_train_time_sec"].mean()),
                "mean_val_feature_time_sec": float(grp["val_feature_time_sec"].mean()),
                "mean_svm_predict_time_sec": float(grp["svm_predict_time_sec"].mean()),
                "mean_total_prediction_time_sec": float(grp["total_prediction_time_sec"].mean()),
            }
        )

    summary_df = pd.DataFrame(summary_records)

    best_by_family: dict[str, Any] = {}
    for fam, fgrp in summary_df.groupby("model_family", sort=False):
        best_by_family[str(fam)] = (
            fgrp.sort_values(by=["mean_f1_macro", "mean_accuracy"], ascending=False)
            .iloc[0]
            .to_dict()
        )

    report_dict: dict[str, Any] = {
        "milestone": "M4b (Intra-Trial Observable Markov Chain & Hybrid Ablations — Runs 4 & 8 Only)",
        "subject_id": dataset.subject_id,
        "development_runs_used": list(dataset.dev.run_ids),
        "held_out_test_run_untouched": True,
        "held_out_test_run_id": cfg.data.held_out_test_run,
        "markov_design": {
            "type": "First-Order Discrete-Time Observable Markov Chain (Option A — NOT an HMM)",
            "sequence_boundaries": "Intra-trial sub-windows (40 samples = 250 ms, stride 20 = 125 ms -> 15 steps, 14 transitions/trial). Never crosses trial or run boundaries.",
            "state_construction": "Train-only HOSVD spatial sub-basis (r_sub=4) log-energy -> train-only StandardScaler -> train-only KMeans(n_clusters=K, random_state=42).",
            "smoothing": "Laplace (Dirichlet) smoothing alpha=0.5 on both per-trial empirical transition matrices and class-conditional transition matrices P^(0), P^(1).",
            "features_per_trial": "K^2 (flattened transition matrix) + K (state occupancy) + 3 (L_0, L_1, LLR) = K^2 + K + 3 features.",
        },
        "best_by_family_on_dev": best_by_family,
        "summary_table": summary_records,
        "per_fold_records": fold_records,
    }

    return fold_df, summary_df, report_dict


def _evaluate_locked_baseline_on_test(
    spec: BaselineModelSpec,
    dev_row: dict[str, Any],
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
    mu_band_hz: tuple[float, float],
    beta_band_hz: tuple[float, float],
    log_epsilon: float,
) -> dict[str, Any]:
    """Refit locked baseline pipeline on Runs 4+8 and evaluate once on held-out Run 12."""
    t0 = time.perf_counter()
    X_dev_feat = extract_mu_beta_log_bandpower(
        X_dev,
        sfreq=sfreq,
        mu_band_hz=mu_band_hz,
        beta_band_hz=beta_band_hz,
        log_epsilon=log_epsilon,
    )
    train_feat_time_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    X_test_feat = extract_mu_beta_log_bandpower(
        X_test,
        sfreq=sfreq,
        mu_band_hz=mu_band_hz,
        beta_band_hz=beta_band_hz,
        log_epsilon=log_epsilon,
    )
    test_feat_time_sec = time.perf_counter() - t1

    pipeline = build_baseline_pipeline(spec)
    t2 = time.perf_counter()
    pipeline.fit(X_dev_feat, y_dev)
    svm_train_time_sec = time.perf_counter() - t2

    t3 = time.perf_counter()
    y_pred = pipeline.predict(X_test_feat)
    svm_predict_time_sec = time.perf_counter() - t3

    metrics = compute_classification_metrics(y_test, y_pred)
    n_features = int(X_dev_feat.shape[1])
    raw_elements = int(X_dev.shape[1] * X_dev.shape[2])
    model_bytes = estimate_fitted_pipeline_storage_bytes(pipeline)

    return {
        "model_family": spec.model_family,
        "locked_config_id": spec.config_id,
        "params_json": json.dumps(spec.params, sort_keys=True),
        "input_representation": "128-D mu/beta log10-bandpower (Welch)",
        "dev_cv_mean_accuracy": float(dev_row["mean_accuracy"]),
        "dev_cv_std_accuracy": float(dev_row["std_accuracy"]),
        "dev_cv_mean_f1_macro": float(dev_row["mean_f1_macro"]),
        "test_accuracy": float(metrics["accuracy"]),
        "test_precision_macro": float(metrics["precision_macro"]),
        "test_recall_macro": float(metrics["recall_macro"]),
        "test_f1_macro": float(metrics["f1_macro"]),
        "test_precision_class_0_left": float(metrics["precision_class_0_left"]),
        "test_recall_class_0_left": float(metrics["recall_class_0_left"]),
        "test_f1_class_0_left": float(metrics["f1_class_0_left"]),
        "test_support_class_0_left": int(metrics["support_class_0_left"]),
        "test_precision_class_1_right": float(metrics["precision_class_1_right"]),
        "test_recall_class_1_right": float(metrics["recall_class_1_right"]),
        "test_f1_class_1_right": float(metrics["f1_class_1_right"]),
        "test_support_class_1_right": int(metrics["support_class_1_right"]),
        "test_confusion_matrix": metrics["confusion_matrix"],
        "n_train_trials_runs_4_8": int(len(y_dev)),
        "n_test_trials_run_12": int(len(y_test)),
        "n_features": n_features,
        "compression_ratio_vs_raw": float(raw_elements / n_features),
        "hosvd_train_reconstruction_error": None,
        "hosvd_test_reconstruction_error": None,
        "feature_bytes_per_trial": int(n_features * 8),
        "model_storage_bytes": int(model_bytes),
        "train_feature_time_sec": float(train_feat_time_sec),
        "svm_train_time_sec": float(svm_train_time_sec),
        "test_feature_time_sec": float(test_feat_time_sec),
        "svm_predict_time_sec": float(svm_predict_time_sec),
        "total_prediction_time_sec": float(test_feat_time_sec + svm_predict_time_sec),
        "prediction_time_per_trial_ms": float(
            (test_feat_time_sec + svm_predict_time_sec) * 1000.0 / len(y_test)
        ),
    }


def _evaluate_locked_hosvd_on_test(
    spec: HOSVDModelSpec,
    dev_row: dict[str, Any],
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Refit locked HOSVD + LinearSVC pipeline on Runs 4+8 and evaluate once on Run 12."""
    extractor = HOSVDFeatureExtractor(r_c=spec.r_c, r_t=spec.r_t)
    t0 = time.perf_counter()
    X_dev_feat = extractor.fit_transform(X_dev)
    train_feat_time_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    X_test_feat = extractor.transform(X_test)
    test_feat_time_sec = time.perf_counter() - t1

    train_recon_err = compute_relative_frobenius_error(X_dev, extractor.U_c_, extractor.U_t_)
    test_recon_err = compute_relative_frobenius_error(X_test, extractor.U_c_, extractor.U_t_)

    pipeline = build_hosvd_classifier_pipeline(spec)
    t2 = time.perf_counter()
    pipeline.fit(X_dev_feat, y_dev)
    svm_train_time_sec = time.perf_counter() - t2

    t3 = time.perf_counter()
    y_pred = pipeline.predict(X_test_feat)
    svm_predict_time_sec = time.perf_counter() - t3

    metrics = compute_classification_metrics(y_test, y_pred)
    storage = compute_hosvd_storage_accounting(
        n_trials=len(y_dev),
        n_channels=int(X_dev.shape[1]),
        n_times=int(X_dev.shape[2]),
        r_c=spec.r_c,
        r_t=spec.r_t,
    )
    scaler_svm_bytes = estimate_fitted_pipeline_storage_bytes(pipeline)
    total_model_bytes = int(storage["reusable_bases_bytes"] + scaler_svm_bytes)

    return {
        "model_family": spec.model_family,
        "locked_config_id": spec.config_id,
        "params_json": json.dumps(spec.params, sort_keys=True),
        "input_representation": f"3D Epoch Tensor (64x320) -> HOSVD Bilinear Projection (r_c={spec.r_c}, r_t={spec.r_t})",
        "dev_cv_mean_accuracy": float(dev_row["mean_accuracy"]),
        "dev_cv_std_accuracy": float(dev_row["std_accuracy"]),
        "dev_cv_mean_f1_macro": float(dev_row["mean_f1_macro"]),
        "test_accuracy": float(metrics["accuracy"]),
        "test_precision_macro": float(metrics["precision_macro"]),
        "test_recall_macro": float(metrics["recall_macro"]),
        "test_f1_macro": float(metrics["f1_macro"]),
        "test_precision_class_0_left": float(metrics["precision_class_0_left"]),
        "test_recall_class_0_left": float(metrics["recall_class_0_left"]),
        "test_f1_class_0_left": float(metrics["f1_class_0_left"]),
        "test_support_class_0_left": int(metrics["support_class_0_left"]),
        "test_precision_class_1_right": float(metrics["precision_class_1_right"]),
        "test_recall_class_1_right": float(metrics["recall_class_1_right"]),
        "test_f1_class_1_right": float(metrics["f1_class_1_right"]),
        "test_support_class_1_right": int(metrics["support_class_1_right"]),
        "test_confusion_matrix": metrics["confusion_matrix"],
        "n_train_trials_runs_4_8": int(len(y_dev)),
        "n_test_trials_run_12": int(len(y_test)),
        "n_features": int(storage["n_features"]),
        "compression_ratio_vs_raw": float(storage["compression_ratio_vs_raw"]),
        "hosvd_train_reconstruction_error": float(train_recon_err),
        "hosvd_test_reconstruction_error": float(test_recon_err),
        "feature_bytes_per_trial": int(storage["feature_bytes_per_trial"]),
        "model_storage_bytes": total_model_bytes,
        "train_feature_time_sec": float(train_feat_time_sec),
        "svm_train_time_sec": float(svm_train_time_sec),
        "test_feature_time_sec": float(test_feat_time_sec),
        "svm_predict_time_sec": float(svm_predict_time_sec),
        "total_prediction_time_sec": float(test_feat_time_sec + svm_predict_time_sec),
        "prediction_time_per_trial_ms": float(
            (test_feat_time_sec + svm_predict_time_sec) * 1000.0 / len(y_test)
        ),
    }


def _evaluate_locked_rks_on_test(
    spec: RKSModelSpec,
    dev_row: dict[str, Any],
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
    mu_band_hz: tuple[float, float],
    beta_band_hz: tuple[float, float],
    log_epsilon: float,
) -> dict[str, Any]:
    """Refit locked RKS + LinearSVC pipeline on Runs 4+8 and evaluate once on Run 12."""
    rks_mapper = RKSFeatureMapper(
        n_components=spec.n_components,
        gamma=spec.gamma,
        random_state=spec.random_seed,
    )
    t0 = time.perf_counter()
    X_dev_bp = extract_mu_beta_log_bandpower(
        X_dev,
        sfreq=sfreq,
        mu_band_hz=mu_band_hz,
        beta_band_hz=beta_band_hz,
        log_epsilon=log_epsilon,
    )
    X_dev_rks = rks_mapper.fit_transform(X_dev_bp)
    train_feat_time_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    X_test_bp = extract_mu_beta_log_bandpower(
        X_test,
        sfreq=sfreq,
        mu_band_hz=mu_band_hz,
        beta_band_hz=beta_band_hz,
        log_epsilon=log_epsilon,
    )
    X_test_rks = rks_mapper.transform(X_test_bp)
    test_feat_time_sec = time.perf_counter() - t1

    clf = build_rks_classifier_(spec)
    t2 = time.perf_counter()
    clf.fit(X_dev_rks, y_dev)
    svm_train_time_sec = time.perf_counter() - t2

    t3 = time.perf_counter()
    y_pred = clf.predict(X_test_rks)
    svm_predict_time_sec = time.perf_counter() - t3

    metrics = compute_classification_metrics(y_test, y_pred)
    raw_elements = int(X_dev.shape[1] * X_dev.shape[2])
    storage = compute_rks_storage_accounting(
        input_dim=int(X_dev_bp.shape[1]),
        n_components=spec.n_components,
        raw_trial_elements=raw_elements,
    )
    svm_bytes = estimate_linear_svc_storage_bytes(clf)
    total_model_bytes = int(rks_mapper.storage_bytes() + svm_bytes)

    return {
        "model_family": spec.model_family,
        "locked_config_id": spec.config_id,
        "params_json": json.dumps(spec.params, sort_keys=True),
        "input_representation": f"128-D Bandpower -> StandardScaler -> RBFSampler (D={spec.n_components}, gamma={spec.gamma:g})",
        "dev_cv_mean_accuracy": float(dev_row["mean_accuracy"]),
        "dev_cv_std_accuracy": float(dev_row["std_accuracy"]),
        "dev_cv_mean_f1_macro": float(dev_row["mean_f1_macro"]),
        "test_accuracy": float(metrics["accuracy"]),
        "test_precision_macro": float(metrics["precision_macro"]),
        "test_recall_macro": float(metrics["recall_macro"]),
        "test_f1_macro": float(metrics["f1_macro"]),
        "test_precision_class_0_left": float(metrics["precision_class_0_left"]),
        "test_recall_class_0_left": float(metrics["recall_class_0_left"]),
        "test_f1_class_0_left": float(metrics["f1_class_0_left"]),
        "test_support_class_0_left": int(metrics["support_class_0_left"]),
        "test_precision_class_1_right": float(metrics["precision_class_1_right"]),
        "test_recall_class_1_right": float(metrics["recall_class_1_right"]),
        "test_f1_class_1_right": float(metrics["f1_class_1_right"]),
        "test_support_class_1_right": int(metrics["support_class_1_right"]),
        "test_confusion_matrix": metrics["confusion_matrix"],
        "n_train_trials_runs_4_8": int(len(y_dev)),
        "n_test_trials_run_12": int(len(y_test)),
        "n_features": int(storage["n_features"]),
        "compression_ratio_vs_raw": float(storage["compression_ratio_vs_raw"]),
        "hosvd_train_reconstruction_error": None,
        "hosvd_test_reconstruction_error": None,
        "feature_bytes_per_trial": int(storage["feature_bytes_per_trial"]),
        "model_storage_bytes": total_model_bytes,
        "train_feature_time_sec": float(train_feat_time_sec),
        "svm_train_time_sec": float(svm_train_time_sec),
        "test_feature_time_sec": float(test_feat_time_sec),
        "svm_predict_time_sec": float(svm_predict_time_sec),
        "total_prediction_time_sec": float(test_feat_time_sec + svm_predict_time_sec),
        "prediction_time_per_trial_ms": float(
            (test_feat_time_sec + svm_predict_time_sec) * 1000.0 / len(y_test)
        ),
    }


def _evaluate_locked_markov_on_test(
    spec: MarkovModelSpec,
    dev_row: dict[str, Any],
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sfreq: float,
    mu_band_hz: tuple[float, float],
    beta_band_hz: tuple[float, float],
    log_epsilon: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refit locked Markov/hybrid pipeline on Runs 4+8 and evaluate once on Run 12."""
    markov_ext = IntraTrialMarkovFeatureExtractor(
        n_states=spec.n_states,
        subwindow_samples=spec.subwindow_samples,
        subwindow_stride=spec.subwindow_stride,
        spatial_rank_sub=4,
        laplace_alpha=spec.laplace_alpha,
        random_state=spec.random_seed,
    )
    markov_storage = compute_markov_storage_accounting(
        n_states=spec.n_states,
        n_channels=int(X_dev.shape[1]),
        spatial_rank_sub=4,
        raw_trial_elements=int(X_dev.shape[1] * X_dev.shape[2]),
    )

    hosvd_train_recon_err: float | None = None
    hosvd_test_recon_err: float | None = None
    extra_extractor_bytes = int(markov_storage["markov_extractor_bytes"])

    t0 = time.perf_counter()
    X_dev_markov = markov_ext.fit_transform(X_dev, y_dev)

    if spec.fusion_mode == "standalone":
        X_dev_feat = X_dev_markov
        repr_desc = f"Intra-Trial Markov Chain (K={spec.n_states} states, M=15 sub-windows)"
    elif spec.fusion_mode == "hosvd_fusion":
        assert spec.r_c is not None and spec.r_t is not None
        hosvd_ext = HOSVDFeatureExtractor(r_c=spec.r_c, r_t=spec.r_t)
        X_dev_hosvd = hosvd_ext.fit_transform(X_dev)
        X_dev_feat = np.concatenate([X_dev_hosvd, X_dev_markov], axis=1)
        hosvd_storage = compute_hosvd_storage_accounting(
            n_trials=len(y_dev),
            n_channels=int(X_dev.shape[1]),
            n_times=int(X_dev.shape[2]),
            r_c=spec.r_c,
            r_t=spec.r_t,
        )
        extra_extractor_bytes += int(hosvd_storage["reusable_bases_bytes"])
        repr_desc = f"HOSVD (r_c={spec.r_c}, r_t={spec.r_t}) + Intra-Trial Markov (K={spec.n_states})"
    elif spec.fusion_mode == "rks_fusion":
        assert spec.rks_dim is not None and spec.rks_gamma is not None
        X_dev_bp = extract_mu_beta_log_bandpower(
            X_dev,
            sfreq=sfreq,
            mu_band_hz=mu_band_hz,
            beta_band_hz=beta_band_hz,
            log_epsilon=log_epsilon,
        )
        rks_ext = RKSFeatureMapper(
            n_components=spec.rks_dim,
            gamma=spec.rks_gamma,
            random_state=spec.random_seed,
        )
        X_dev_rks = rks_ext.fit_transform(X_dev_bp)
        X_dev_feat = np.concatenate([X_dev_rks, X_dev_markov], axis=1)
        extra_extractor_bytes += int(rks_ext.storage_bytes())
        repr_desc = f"RKS (D={spec.rks_dim}, gamma={spec.rks_gamma:g}) + Intra-Trial Markov (K={spec.n_states})"
    else:
        raise ValueError(f"Unknown fusion_mode: '{spec.fusion_mode}'")

    train_feat_time_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    X_test_markov = markov_ext.transform(X_test)

    if spec.fusion_mode == "standalone":
        X_test_feat = X_test_markov
    elif spec.fusion_mode == "hosvd_fusion":
        X_test_hosvd = hosvd_ext.transform(X_test)
        X_test_feat = np.concatenate([X_test_hosvd, X_test_markov], axis=1)
    elif spec.fusion_mode == "rks_fusion":
        X_test_bp = extract_mu_beta_log_bandpower(
            X_test,
            sfreq=sfreq,
            mu_band_hz=mu_band_hz,
            beta_band_hz=beta_band_hz,
            log_epsilon=log_epsilon,
        )
        X_test_rks = rks_ext.transform(X_test_bp)
        X_test_feat = np.concatenate([X_test_rks, X_test_markov], axis=1)

    test_feat_time_sec = time.perf_counter() - t1

    if spec.fusion_mode == "hosvd_fusion":
        hosvd_train_recon_err = compute_relative_frobenius_error(
            X_dev, hosvd_ext.U_c_, hosvd_ext.U_t_
        )
        hosvd_test_recon_err = compute_relative_frobenius_error(
            X_test, hosvd_ext.U_c_, hosvd_ext.U_t_
        )

    pipeline = build_markov_classifier_pipeline(spec)
    t2 = time.perf_counter()
    pipeline.fit(X_dev_feat, y_dev)
    svm_train_time_sec = time.perf_counter() - t2

    t3 = time.perf_counter()
    y_pred = pipeline.predict(X_test_feat)
    svm_predict_time_sec = time.perf_counter() - t3

    metrics = compute_classification_metrics(y_test, y_pred)
    n_features = int(X_dev_feat.shape[1])
    raw_elements = int(X_dev.shape[1] * X_dev.shape[2])
    scaler_svm_bytes = estimate_fitted_pipeline_storage_bytes(pipeline)
    total_model_bytes = int(extra_extractor_bytes + scaler_svm_bytes)

    learned_markov_models = {
        "n_states": int(spec.n_states),
        "laplace_alpha": float(spec.laplace_alpha),
        "class_0_left_initial_probs": markov_ext.model_class0_.initial_probs.tolist(),
        "class_0_left_transition_matrix": markov_ext.model_class0_.transition_matrix.tolist(),
        "class_1_right_initial_probs": markov_ext.model_class1_.initial_probs.tolist(),
        "class_1_right_transition_matrix": markov_ext.model_class1_.transition_matrix.tolist(),
    }

    result_row = {
        "model_family": spec.model_family,
        "locked_config_id": spec.config_id,
        "params_json": json.dumps(spec.params, sort_keys=True),
        "input_representation": repr_desc,
        "dev_cv_mean_accuracy": float(dev_row["mean_accuracy"]),
        "dev_cv_std_accuracy": float(dev_row["std_accuracy"]),
        "dev_cv_mean_f1_macro": float(dev_row["mean_f1_macro"]),
        "test_accuracy": float(metrics["accuracy"]),
        "test_precision_macro": float(metrics["precision_macro"]),
        "test_recall_macro": float(metrics["recall_macro"]),
        "test_f1_macro": float(metrics["f1_macro"]),
        "test_precision_class_0_left": float(metrics["precision_class_0_left"]),
        "test_recall_class_0_left": float(metrics["recall_class_0_left"]),
        "test_f1_class_0_left": float(metrics["f1_class_0_left"]),
        "test_support_class_0_left": int(metrics["support_class_0_left"]),
        "test_precision_class_1_right": float(metrics["precision_class_1_right"]),
        "test_recall_class_1_right": float(metrics["recall_class_1_right"]),
        "test_f1_class_1_right": float(metrics["f1_class_1_right"]),
        "test_support_class_1_right": int(metrics["support_class_1_right"]),
        "test_confusion_matrix": metrics["confusion_matrix"],
        "n_train_trials_runs_4_8": int(len(y_dev)),
        "n_test_trials_run_12": int(len(y_test)),
        "n_features": n_features,
        "compression_ratio_vs_raw": float(raw_elements / n_features),
        "hosvd_train_reconstruction_error": hosvd_train_recon_err,
        "hosvd_test_reconstruction_error": hosvd_test_recon_err,
        "feature_bytes_per_trial": int(n_features * 8),
        "model_storage_bytes": total_model_bytes,
        "train_feature_time_sec": float(train_feat_time_sec),
        "svm_train_time_sec": float(svm_train_time_sec),
        "test_feature_time_sec": float(test_feat_time_sec),
        "svm_predict_time_sec": float(svm_predict_time_sec),
        "total_prediction_time_sec": float(test_feat_time_sec + svm_predict_time_sec),
        "prediction_time_per_trial_ms": float(
            (test_feat_time_sec + svm_predict_time_sec) * 1000.0 / len(y_test)
        ),
    }

    return result_row, learned_markov_models


def compute_m5_hosvd_rank_curve_on_test(
    X_dev: np.ndarray,
    X_test: np.ndarray,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    """Measure HOSVD reconstruction error and compression on Runs 4+8 vs. Run 12 across rank pairs.

    Channel and time bases (U_c, U_t) are fitted strictly on combined development Runs 4+8.
    """
    hosvd_cfg = cfg.raw.get("hosvd", {})
    r_c_grid = [int(r) for r in hosvd_cfg.get("channel_ranks", [4, 8, 16])]
    r_t_grid = [int(r) for r in hosvd_cfg.get("time_ranks", [10, 20, 40])]

    records: list[dict[str, Any]] = []
    for r_c in r_c_grid:
        for r_t in r_t_grid:
            ext = HOSVDFeatureExtractor(r_c=r_c, r_t=r_t).fit(X_dev)
            train_err = compute_relative_frobenius_error(X_dev, ext.U_c_, ext.U_t_)
            test_err = compute_relative_frobenius_error(X_test, ext.U_c_, ext.U_t_)
            storage = compute_hosvd_storage_accounting(
                n_trials=int(X_dev.shape[0]),
                n_channels=int(X_dev.shape[1]),
                n_times=int(X_dev.shape[2]),
                r_c=r_c,
                r_t=r_t,
            )
            records.append(
                {
                    "r_c": r_c,
                    "r_t": r_t,
                    "n_features": int(storage["n_features"]),
                    "compression_ratio_vs_raw": float(storage["compression_ratio_vs_raw"]),
                    "tensor_decomposition_compression_ratio": float(
                        storage["tensor_decomposition_compression_ratio"]
                    ),
                    "train_runs_4_8_reconstruction_error": float(train_err),
                    "test_run_12_reconstruction_error": float(test_err),
                    "feature_bytes_per_trial": int(storage["feature_bytes_per_trial"]),
                    "reusable_bases_bytes": int(storage["reusable_bases_bytes"]),
                }
            )
    return pd.DataFrame(records)


def run_m5_locked_evaluation(
    dataset: PreparedEEGDataset,
    cfg: ProjectConfig,
    dependency_versions: dict[str, str | None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select final model settings strictly from Runs 4 & 8 CV, refit on Runs 4+8, evaluate Run 12 once."""
    # 1. Run development CV on Runs 4 and 8 to lock best config per model family
    _, base_sum_df, base_rep = run_m2_baselines_validation(dataset, cfg)
    _, hosvd_sum_df, _, hosvd_rep = run_m3_hosvd_validation(dataset, cfg)
    _, rks_sum_df, rks_rep = run_m4_rks_validation(dataset, cfg)
    _, markov_sum_df, markov_rep = run_m4b_markov_validation(dataset, cfg)

    feat_cfg = cfg.raw.get("features", {})
    mu_band = tuple(float(x) for x in feat_cfg.get("mu_band_hz", [8.0, 13.0]))
    beta_band = tuple(float(x) for x in feat_cfg.get("beta_band_hz", [13.0, 30.0]))
    log_eps = float(feat_cfg.get("log_epsilon", 1e-12))

    X_dev, y_dev = dataset.dev.X, dataset.dev.y
    X_test, y_test = dataset.test.X, dataset.test.y

    # Lookup maps from config_id to spec object
    base_specs_map = {s.config_id: s for s in get_baseline_model_specs(cfg)}
    hosvd_specs_map = {s.config_id: s for s in get_hosvd_model_specs(cfg)}
    rks_specs_map = {s.config_id: s for s in get_rks_model_specs(cfg)}
    markov_specs_map = {s.config_id: s for s in get_markov_model_specs(cfg)}

    locked_results: list[dict[str, Any]] = []
    learned_markov_transitions: dict[str, Any] = {}

    # Lock 1: Baseline LinearSVC
    best_lin_row = base_rep["best_linear_svc_on_dev"]
    locked_results.append(
        _evaluate_locked_baseline_on_test(
            spec=base_specs_map[str(best_lin_row["config_id"])],
            dev_row=best_lin_row,
            X_dev=X_dev,
            y_dev=y_dev,
            X_test=X_test,
            y_test=y_test,
            sfreq=dataset.dev.sfreq,
            mu_band_hz=(mu_band[0], mu_band[1]),
            beta_band_hz=(beta_band[0], beta_band[1]),
            log_epsilon=log_eps,
        )
    )

    # Lock 2: Baseline RBF SVC
    best_rbf_row = base_rep["best_rbf_svc_on_dev"]
    locked_results.append(
        _evaluate_locked_baseline_on_test(
            spec=base_specs_map[str(best_rbf_row["config_id"])],
            dev_row=best_rbf_row,
            X_dev=X_dev,
            y_dev=y_dev,
            X_test=X_test,
            y_test=y_test,
            sfreq=dataset.dev.sfreq,
            mu_band_hz=(mu_band[0], mu_band[1]),
            beta_band_hz=(beta_band[0], beta_band[1]),
            log_epsilon=log_eps,
        )
    )

    # Lock 3: HOSVD + LinearSVC
    best_hosvd_row = hosvd_rep["best_hosvd_on_dev"]
    locked_results.append(
        _evaluate_locked_hosvd_on_test(
            spec=hosvd_specs_map[str(best_hosvd_row["config_id"])],
            dev_row=best_hosvd_row,
            X_dev=X_dev,
            y_dev=y_dev,
            X_test=X_test,
            y_test=y_test,
        )
    )

    # Lock 4: RKS + LinearSVC
    best_rks_row = rks_rep["best_rks_on_dev"]
    locked_results.append(
        _evaluate_locked_rks_on_test(
            spec=rks_specs_map[str(best_rks_row["config_id"])],
            dev_row=best_rks_row,
            X_dev=X_dev,
            y_dev=y_dev,
            X_test=X_test,
            y_test=y_test,
            sfreq=dataset.dev.sfreq,
            mu_band_hz=(mu_band[0], mu_band[1]),
            beta_band_hz=(beta_band[0], beta_band[1]),
            log_epsilon=log_eps,
        )
    )

    # Lock 5, 6, 7: Standalone Markov, HOSVD+Markov, RKS+Markov
    for fam in ("markov_linear_svc", "hosvd_markov_linear_svc", "rks_markov_linear_svc"):
        best_m_row = markov_rep["best_by_family_on_dev"][fam]
        m_spec = markov_specs_map[str(best_m_row["config_id"])]
        res_row, m_transitions = _evaluate_locked_markov_on_test(
            spec=m_spec,
            dev_row=best_m_row,
            X_dev=X_dev,
            y_dev=y_dev,
            X_test=X_test,
            y_test=y_test,
            sfreq=dataset.dev.sfreq,
            mu_band_hz=(mu_band[0], mu_band[1]),
            beta_band_hz=(beta_band[0], beta_band[1]),
            log_epsilon=log_eps,
        )
        locked_results.append(res_row)
        learned_markov_transitions[fam] = m_transitions

    locked_df = pd.DataFrame(locked_results)
    rank_curve_df = compute_m5_hosvd_rank_curve_on_test(X_dev, X_test, cfg)

    report_dict: dict[str, Any] = {
        "milestone": "M5 (Locked Comparison on Untouched Held-Out Run 12 Across All 8 Faculty Measures)",
        "subject_id": dataset.subject_id,
        "random_seed": cfg.random_seed,
        "development_runs_refit": list(dataset.dev.run_ids),
        "held_out_test_run_evaluated_once": int(cfg.data.held_out_test_run),
        "class_balance": {
            "train_runs_4_8": {
                "T1_left_0": int(np.sum(y_dev == 0)),
                "T2_right_1": int(np.sum(y_dev == 1)),
                "total": int(len(y_dev)),
            },
            "test_run_12": {
                "T1_left_0": int(np.sum(y_test == 0)),
                "T2_right_1": int(np.sum(y_test == 1)),
                "total": int(len(y_test)),
            },
        },
        "software_versions": dependency_versions or {},
        "methodological_disclosures_and_limitations": [
            "1. Single-subject, cross-run evaluation: Models are trained on Subject 1 Runs 4+8 (30 trials) and tested on Subject 1 Run 12 (15 trials); results do not claim cross-subject generalization.",
            "2. High validation variance: With only 2 development runs (15 trials/run) and 15 test trials, each trial corresponds to 6.67% accuracy.",
            "3. End-to-end pipeline comparison: Baselines and RKS transform 128-D mu/beta log-bandpower features, whereas HOSVD projects 3D band-pass filtered epoch tensors (64x320) and Markov models intra-trial sub-window spatial-energy state transitions.",
            "4. Offline non-causal filtering: Zero-phase 8-30 Hz FIR filtering is used offline; no real-time streaming, device control, clinical, or implanted BCI claims are made.",
            "5. RKS dimension note: RBFSampler is kernel approximation (feature expansion when D > 128), not inherently dimensionality reduction.",
        ],
        "locked_test_results": locked_results,
        "hosvd_test_rank_curve": rank_curve_df.to_dict(orient="records"),
        "learned_markov_models_on_runs_4_8": learned_markov_transitions,
    }

    return locked_df, rank_curve_df, report_dict

