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
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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
    fit_full_hosvd_singular_spaces,
    project_hosvd_trials,
    slice_hosvd_bases,
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
    """Single run-aware training/validation split (single-subject or multi-subject cohort)."""

    fold_index: int
    train_run_id: int
    val_run_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_recording_ids: tuple[str, ...] = ()
    val_recording_ids: tuple[str, ...] = ()


def build_run_swapped_splits(
    groups: np.ndarray,
    development_runs: tuple[int, ...] = (4, 8),
    forbidden_test_run: int = 12,
    recording_ids: tuple[str, ...] | None = None,
    subject_ids: np.ndarray | None = None,
) -> list[RunFoldSplit]:
    """Construct run-swapped validation folds across development runs (4 -> 8 and 8 -> 4).

    When used on a multi-participant cohort (Option A):
    - Fold 0 trains on Run 4 across all eligible participants and validates on Run 8.
    - Fold 1 trains on Run 8 across all eligible participants and validates on Run 4.
    Verifies that no recording_id is ever split across training and validation.

    Raises:
        ValueError: If forbidden_test_run appears in groups, any development run is missing,
            or any recording_id appears in both train and validation folds.
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

    if recording_ids is not None and len(recording_ids) != len(groups):
        raise ValueError(
            f"recording_ids length ({len(recording_ids)}) does not match groups length ({len(groups)})."
        )

    run_a, run_b = development_runs
    pairs = [(0, run_a, run_b), (1, run_b, run_a)]
    splits: list[RunFoldSplit] = []

    for fold_idx, train_run, val_run in pairs:
        train_idx = np.where(groups == train_run)[0]
        val_idx = np.where(groups == val_run)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Empty fold indices for train_run={train_run}, val_run={val_run}.")

        train_recs: tuple[str, ...] = ()
        val_recs: tuple[str, ...] = ()
        if recording_ids is not None:
            train_rec_set = {recording_ids[int(i)] for i in train_idx}
            val_rec_set = {recording_ids[int(i)] for i in val_idx}
            if not train_rec_set.isdisjoint(val_rec_set):
                overlap = sorted(train_rec_set & val_rec_set)
                raise ValueError(
                    f"Recording leakage violation in fold {fold_idx}: recordings {overlap} "
                    "appear in both training and validation!"
                )
            train_recs = tuple(sorted(train_rec_set))
            val_recs = tuple(sorted(val_rec_set))

        if subject_ids is not None:
            train_subj_set = set(int(s) for s in subject_ids[train_idx])
            val_subj_set = set(int(s) for s in subject_ids[val_idx])
            if train_subj_set != val_subj_set:
                raise ValueError(
                    f"Participant mismatch between train ({sorted(train_subj_set)}) and "
                    f"validation ({sorted(val_subj_set)}) in Option A fold {fold_idx}."
                )

        splits.append(
            RunFoldSplit(
                fold_index=fold_idx,
                train_run_id=int(train_run),
                val_run_id=int(val_run),
                train_indices=train_idx,
                val_indices=val_idx,
                train_recording_ids=train_recs,
                val_recording_ids=val_recs,
            )
        )

    return splits


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute accuracy, balanced accuracy, per-class & macro precision/recall/F1, and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
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


def compute_participant_grouped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute participant-clustered metrics and 95% confidence intervals across participants.

    Treats each participant (rather than each correlated trial within a participant)
    as the primary cluster unit for uncertainty estimation.
    """
    unique_subjects = sorted(int(s) for s in np.unique(subject_ids))
    per_subject_rows: list[dict[str, Any]] = []

    for sid in unique_subjects:
        mask = subject_ids == sid
        yt_s = y_true[mask]
        yp_s = y_pred[mask]
        m_s = compute_classification_metrics(yt_s, yp_s)
        per_subject_rows.append(
            {
                "subject_id": sid,
                "n_trials": int(len(yt_s)),
                "accuracy": float(m_s["accuracy"]),
                "balanced_accuracy": float(m_s["balanced_accuracy"]),
                "precision_macro": float(m_s["precision_macro"]),
                "recall_macro": float(m_s["recall_macro"]),
                "f1_macro": float(m_s["f1_macro"]),
            }
        )

    n_subj = len(per_subject_rows)
    accs = np.array([r["accuracy"] for r in per_subject_rows], dtype=np.float64)
    bal_accs = np.array([r["balanced_accuracy"] for r in per_subject_rows], dtype=np.float64)
    precs = np.array([r["precision_macro"] for r in per_subject_rows], dtype=np.float64)
    recs = np.array([r["recall_macro"] for r in per_subject_rows], dtype=np.float64)
    f1s = np.array([r["f1_macro"] for r in per_subject_rows], dtype=np.float64)

    def _cluster_summary(vals: np.ndarray) -> tuple[float, float, float, list[float]]:
        mean_v = float(np.mean(vals))
        if len(vals) <= 1:
            return mean_v, 0.0, 0.0, [mean_v, mean_v]
        std_v = float(np.std(vals, ddof=1))
        se_v = float(std_v / np.sqrt(len(vals)))
        t_crit = float(student_t.ppf(0.975, df=len(vals) - 1))
        ci = [float(mean_v - t_crit * se_v), float(mean_v + t_crit * se_v)]
        return mean_v, std_v, se_v, ci

    acc_mean, acc_std, acc_se, acc_ci = _cluster_summary(accs)
    bal_mean, bal_std, bal_se, bal_ci = _cluster_summary(bal_accs)
    f1_mean, f1_std, f1_se, f1_ci = _cluster_summary(f1s)

    q25, q75 = np.percentile(accs, [25.0, 75.0])
    summary = {
        "n_participants": n_subj,
        "participant_mean_accuracy": acc_mean,
        "participant_std_accuracy": acc_std,
        "participant_se_accuracy": acc_se,
        "participant_ci95_accuracy": acc_ci,
        "participant_median_accuracy": float(np.median(accs)),
        "participant_iqr_accuracy": float(q75 - q25),
        "participant_min_accuracy": float(np.min(accs)),
        "participant_max_accuracy": float(np.max(accs)),
        "participant_mean_balanced_accuracy": bal_mean,
        "participant_std_balanced_accuracy": bal_std,
        "participant_se_balanced_accuracy": bal_se,
        "participant_ci95_balanced_accuracy": bal_ci,
        "participant_mean_precision_macro": float(np.mean(precs)),
        "participant_mean_recall_macro": float(np.mean(recs)),
        "participant_mean_f1_macro": f1_mean,
        "participant_std_f1_macro": f1_std,
        "participant_se_f1_macro": f1_se,
        "participant_ci95_f1_macro": f1_ci,
    }
    return summary, per_subject_rows


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


def _build_test_participant_fields(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    subject_ids_test: np.ndarray | None,
    model_family: str,
    locked_config_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute participant-grouped test metrics and per-participant test rows."""
    subj_arr = (
        np.asarray(subject_ids_test, dtype=np.int64)
        if subject_ids_test is not None
        else np.ones(len(y_test), dtype=np.int64)
    )
    summary, per_subj = compute_participant_grouped_metrics(y_test, y_pred, subj_arr)
    prefixed_summary = {f"test_{k}": v for k, v in summary.items()}
    per_subj_rows = [
        {
            "model_family": model_family,
            "locked_config_id": locked_config_id,
            **row,
        }
        for row in per_subj
    ]
    return prefixed_summary, per_subj_rows


def _extract_dev_reference_fields(dev_row: dict[str, Any]) -> dict[str, Any]:
    """Extract development CV summary fields (including participant CIs when available)."""
    fields: dict[str, Any] = {
        "dev_cv_mean_accuracy": float(dev_row["mean_accuracy"]),
        "dev_cv_std_accuracy": float(dev_row["std_accuracy"]),
        "dev_cv_mean_f1_macro": float(dev_row["mean_f1_macro"]),
    }
    if "mean_balanced_accuracy" in dev_row:
        fields["dev_cv_mean_balanced_accuracy"] = float(dev_row["mean_balanced_accuracy"])
    if "participant_mean_accuracy" in dev_row:
        fields["dev_participant_mean_accuracy"] = float(dev_row["participant_mean_accuracy"])
        fields["dev_participant_ci95_accuracy"] = dev_row["participant_ci95_accuracy"]
    if "participant_mean_f1_macro" in dev_row:
        fields["dev_participant_mean_f1_macro"] = float(dev_row["participant_mean_f1_macro"])
        fields["dev_participant_ci95_f1_macro"] = dev_row["participant_ci95_f1_macro"]
    return fields


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
    subject_ids_test: np.ndarray | None = None,
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
    part_summary, part_rows = _build_test_participant_fields(
        y_test=y_test,
        y_pred=y_pred,
        subject_ids_test=subject_ids_test,
        model_family=spec.model_family,
        locked_config_id=spec.config_id,
    )
    n_features = int(X_dev_feat.shape[1])
    raw_elements = int(X_dev.shape[1] * X_dev.shape[2])
    model_bytes = estimate_fitted_pipeline_storage_bytes(pipeline)

    return {
        "model_family": spec.model_family,
        "locked_config_id": spec.config_id,
        "params_json": json.dumps(spec.params, sort_keys=True),
        "input_representation": "128-D mu/beta log10-bandpower (Welch)",
        **_extract_dev_reference_fields(dev_row),
        "test_accuracy": float(metrics["accuracy"]),
        "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
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
        **part_summary,
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
        "_per_participant_test_records": part_rows,
    }


def _evaluate_locked_hosvd_on_test(
    spec: HOSVDModelSpec,
    dev_row: dict[str, Any],
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subject_ids_test: np.ndarray | None = None,
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
    part_summary, part_rows = _build_test_participant_fields(
        y_test=y_test,
        y_pred=y_pred,
        subject_ids_test=subject_ids_test,
        model_family=spec.model_family,
        locked_config_id=spec.config_id,
    )
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
        **_extract_dev_reference_fields(dev_row),
        "test_accuracy": float(metrics["accuracy"]),
        "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
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
        **part_summary,
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
        "_per_participant_test_records": part_rows,
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
    subject_ids_test: np.ndarray | None = None,
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
    part_summary, part_rows = _build_test_participant_fields(
        y_test=y_test,
        y_pred=y_pred,
        subject_ids_test=subject_ids_test,
        model_family=spec.model_family,
        locked_config_id=spec.config_id,
    )
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
        **_extract_dev_reference_fields(dev_row),
        "test_accuracy": float(metrics["accuracy"]),
        "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
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
        **part_summary,
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
        "_per_participant_test_records": part_rows,
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
    subject_ids_test: np.ndarray | None = None,
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
    part_summary, part_rows = _build_test_participant_fields(
        y_test=y_test,
        y_pred=y_pred,
        subject_ids_test=subject_ids_test,
        model_family=spec.model_family,
        locked_config_id=spec.config_id,
    )
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
        **_extract_dev_reference_fields(dev_row),
        "test_accuracy": float(metrics["accuracy"]),
        "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
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
        **part_summary,
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
        "_per_participant_test_records": part_rows,
    }

    return result_row, learned_markov_models


def compute_m5_hosvd_rank_curve_on_test(
    X_dev: np.ndarray,
    X_test: np.ndarray,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    """Measure HOSVD reconstruction error and compression on Runs 4+8 vs. Run 12 across rank pairs.

    Channel and time bases (U_c, U_t) are fitted strictly once on combined development Runs 4+8
    and sliced to (r_c, r_t). Clearly labeled as an exploratory post-hoc rank curve diagnostic.
    """
    hosvd_cfg = cfg.raw.get("hosvd", {})
    r_c_grid = [int(r) for r in hosvd_cfg.get("channel_ranks", [4, 8, 16])]
    r_t_grid = [int(r) for r in hosvd_cfg.get("time_ranks", [10, 20, 40])]

    U_c_full, s_c_full, U_t_full, s_t_full = fit_full_hosvd_singular_spaces(X_dev)

    records: list[dict[str, Any]] = []
    for r_c in r_c_grid:
        for r_t in r_t_grid:
            bases = slice_hosvd_bases(
                U_c_full, s_c_full, U_t_full, s_t_full, r_c=r_c, r_t=r_t
            )
            train_err = compute_relative_frobenius_error(X_dev, bases.U_c, bases.U_t)
            test_err = compute_relative_frobenius_error(X_test, bases.U_c, bases.U_t)
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
                    "analysis_type": "exploratory_rank_curve_bases_fitted_on_dev_runs_4_8_only",
                }
            )
    return pd.DataFrame(records)


def run_m5_locked_evaluation(
    dataset: PreparedEEGDataset,
    cfg: ProjectConfig,
    dependency_versions: dict[str, str | None] | None = None,
    precomputed_e2_report: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select final model settings strictly from Runs 4 & 8 CV, refit on Runs 4+8, evaluate Run 12 once."""
    e2_report: dict[str, Any] | None = precomputed_e2_report
    if e2_report is not None or cfg.data.is_multi_subject:
        if e2_report is None:
            _, _, _, _, e2_report = run_e2_cohort_development_validation(dataset, cfg)
        best_map = {
            str(r["model_family"]): r for r in e2_report["locked_best_by_family_on_dev"]
        }
        best_lin_row = best_map["baseline_linear_svc"]
        best_rbf_row = best_map["baseline_rbf_svc"]
        best_hosvd_row = best_map["hosvd_linear_svc"]
        best_rks_row = best_map["rks_linear_svc"]
        markov_best_by_fam = {
            fam: best_map[fam]
            for fam in ("markov_linear_svc", "hosvd_markov_linear_svc", "rks_markov_linear_svc")
        }
    else:
        _, _, base_rep = run_m2_baselines_validation(dataset, cfg)
        _, _, _, hosvd_rep = run_m3_hosvd_validation(dataset, cfg)
        _, _, rks_rep = run_m4_rks_validation(dataset, cfg)
        _, _, markov_rep = run_m4b_markov_validation(dataset, cfg)
        best_lin_row = base_rep["best_linear_svc_on_dev"]
        best_rbf_row = base_rep["best_rbf_svc_on_dev"]
        best_hosvd_row = hosvd_rep["best_hosvd_on_dev"]
        best_rks_row = rks_rep["best_rks_on_dev"]
        markov_best_by_fam = markov_rep["best_by_family_on_dev"]

    feat_cfg = cfg.raw.get("features", {})
    mu_band = tuple(float(x) for x in feat_cfg.get("mu_band_hz", [8.0, 13.0]))
    beta_band = tuple(float(x) for x in feat_cfg.get("beta_band_hz", [13.0, 30.0]))
    log_eps = float(feat_cfg.get("log_epsilon", 1e-12))

    X_dev, y_dev = dataset.dev.X, dataset.dev.y
    X_test, y_test = dataset.test.X, dataset.test.y
    subject_ids_test = dataset.test.subject_ids

    # Lookup maps from config_id to spec object
    base_specs_map = {s.config_id: s for s in get_baseline_model_specs(cfg)}
    hosvd_specs_map = {s.config_id: s for s in get_hosvd_model_specs(cfg)}
    rks_specs_map = {s.config_id: s for s in get_rks_model_specs(cfg)}
    markov_specs_map = {s.config_id: s for s in get_markov_model_specs(cfg)}

    locked_results: list[dict[str, Any]] = []
    per_participant_test_rows: list[dict[str, Any]] = []
    learned_markov_transitions: dict[str, Any] = {}

    def _record_locked_row(row_dict: dict[str, Any]) -> None:
        part_records = row_dict.pop("_per_participant_test_records", [])
        per_participant_test_rows.extend(part_records)
        locked_results.append(row_dict)

    # Lock 1: Baseline LinearSVC
    _record_locked_row(
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
            subject_ids_test=subject_ids_test,
        )
    )

    # Lock 2: Baseline RBF SVC
    _record_locked_row(
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
            subject_ids_test=subject_ids_test,
        )
    )

    # Lock 3: HOSVD + LinearSVC
    _record_locked_row(
        _evaluate_locked_hosvd_on_test(
            spec=hosvd_specs_map[str(best_hosvd_row["config_id"])],
            dev_row=best_hosvd_row,
            X_dev=X_dev,
            y_dev=y_dev,
            X_test=X_test,
            y_test=y_test,
            subject_ids_test=subject_ids_test,
        )
    )

    # Lock 4: RKS + LinearSVC (primary predetermined seed = cfg.random_seed)
    best_rks_spec = rks_specs_map[str(best_rks_row["config_id"])]
    _record_locked_row(
        _evaluate_locked_rks_on_test(
            spec=best_rks_spec,
            dev_row=best_rks_row,
            X_dev=X_dev,
            y_dev=y_dev,
            X_test=X_test,
            y_test=y_test,
            sfreq=dataset.dev.sfreq,
            mu_band_hz=(mu_band[0], mu_band[1]),
            beta_band_hz=(beta_band[0], beta_band[1]),
            log_epsilon=log_eps,
            subject_ids_test=subject_ids_test,
        )
    )

    # Predetermined multi-seed variability on test for locked RKS model (never selecting seed on test)
    rks_cfg = cfg.raw.get("rks", {})
    predetermined_seeds = [int(s) for s in rks_cfg.get("random_seeds", [cfg.random_seed])]
    X_dev_bp_once = extract_mu_beta_log_bandpower(
        X_dev,
        sfreq=dataset.dev.sfreq,
        mu_band_hz=(mu_band[0], mu_band[1]),
        beta_band_hz=(beta_band[0], beta_band[1]),
        log_epsilon=log_eps,
    )
    X_test_bp_once = extract_mu_beta_log_bandpower(
        X_test,
        sfreq=dataset.dev.sfreq,
        mu_band_hz=(mu_band[0], mu_band[1]),
        beta_band_hz=(beta_band[0], beta_band[1]),
        log_epsilon=log_eps,
    )
    rks_test_seed_accs: list[float] = []
    rks_test_seed_f1s: list[float] = []
    for seed_val in predetermined_seeds:
        seed_mapper = RKSFeatureMapper(
            n_components=best_rks_spec.n_components,
            gamma=best_rks_spec.gamma,
            random_state=int(seed_val),
        )
        z_dev_seed = seed_mapper.fit_transform(X_dev_bp_once)
        z_test_seed = seed_mapper.transform(X_test_bp_once)
        seed_clf = build_rks_classifier_(
            RKSModelSpec(
                model_family="rks_linear_svc",
                config_id=best_rks_spec.config_id,
                n_components=best_rks_spec.n_components,
                gamma=best_rks_spec.gamma,
                c_val=best_rks_spec.c_val,
                params=best_rks_spec.params,
                random_seed=int(seed_val),
                max_iter=best_rks_spec.max_iter,
            )
        )
        seed_clf.fit(z_dev_seed, y_dev)
        yp_seed = seed_clf.predict(z_test_seed)
        m_seed = compute_classification_metrics(y_test, yp_seed)
        rks_test_seed_accs.append(float(m_seed["accuracy"]))
        rks_test_seed_f1s.append(float(m_seed["f1_macro"]))

    # Lock 5, 6, 7: Standalone Markov, HOSVD+Markov, RKS+Markov
    for fam in ("markov_linear_svc", "hosvd_markov_linear_svc", "rks_markov_linear_svc"):
        best_m_row = markov_best_by_fam[fam]
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
            subject_ids_test=subject_ids_test,
        )
        _record_locked_row(res_row)
        learned_markov_transitions[fam] = m_transitions

    locked_df = pd.DataFrame(locked_results)
    rank_curve_df = compute_m5_hosvd_rank_curve_on_test(X_dev, X_test, cfg)

    if cfg.data.is_multi_subject:
        disclosures = [
            f"1. Multi-participant cross-recording evaluation (Option A): Models are trained on Runs 4 and 8 across N={len(dataset.subject_ids)} eligible participants ({len(set(dataset.dev.recording_ids))} recordings, {len(y_dev)} trials) and tested once on held-out Run 12 across those same {len(dataset.subject_ids)} participants ({len(set(dataset.test.recording_ids))} recordings, {len(y_test)} trials). Evaluates generalization to new recordings of known participants, not unseen participants.",
            f"2. Pre-evaluation eligibility audit & pilot separation: Pilot Subject 1 (S001) was excluded a priori because its Run 12 was already inspected in the single-subject pilot. {len(dataset.eligibility_records) - len(dataset.subject_ids)} of {len(dataset.eligibility_records)} total subjects were excluded before model evaluation ({[r.subject_id for r in dataset.eligibility_records if not r.eligible]}); no participant was excluded due to model accuracy.",
            f"3. Participant-clustered uncertainty vs. validation fold SD: 2-fold validation SD measures inter-run split spread (Fold 1 Run 4->8 vs. Fold 2 Run 8->4) and is NOT a confidence interval. Uncertainty is reported via 95% Student-t confidence intervals across the N={len(dataset.subject_ids)} independent participants (df={max(len(dataset.subject_ids) - 1, 1)}).",
            "4. End-to-end representation comparison & RKS note: Baselines and RKS transform 128-D mu/beta log10-bandpower features (with RBFSampler acting as explicit kernel feature expansion when D > 128, not dimensionality reduction), whereas HOSVD projects 3D band-pass filtered epoch tensors (64x320) using train-only bases (U_c, U_t) and Markov models intra-trial sub-window spatial-energy state transitions without crossing trial, recording, or participant boundaries.",
            "5. Offline non-causal filtering & exploratory rank curve disclosure: Zero-phase 8-30 Hz FIR filtering is used offline (no real-time, clinical, or implanted BCI claims). The HOSVD rank curve on Run 12 is strictly an exploratory post-hoc diagnostic using bases fitted on Runs 4+8; locked HOSVD ranks were chosen solely on development Runs 4 and 8.",
        ]
    else:
        disclosures = [
            "1. Single-subject, cross-run evaluation: Models are trained on Subject 1 Runs 4+8 (30 trials) and tested on Subject 1 Run 12 (15 trials); results do not claim cross-subject generalization.",
            "2. High validation variance: With only 2 development runs (15 trials/run) and 15 test trials, each trial corresponds to 6.67% accuracy.",
            "3. End-to-end pipeline comparison: Baselines and RKS transform 128-D mu/beta log-bandpower features, whereas HOSVD projects 3D band-pass filtered epoch tensors (64x320) and Markov models intra-trial sub-window spatial-energy state transitions.",
            "4. Offline non-causal filtering: Zero-phase 8-30 Hz FIR filtering is used offline; no real-time streaming, device control, clinical, or implanted BCI claims are made.",
            "5. RKS dimension note: RBFSampler is kernel approximation (feature expansion when D > 128), not inherently dimensionality reduction.",
        ]

    report_dict: dict[str, Any] = {
        "milestone": (
            "E3 / M5 (Confirmatory Cohort Locked Final Evaluation on Untouched Held-Out Run 12)"
            if cfg.data.is_multi_subject
            else "M5 (Locked Comparison on Untouched Held-Out Run 12 Across All 8 Faculty Measures)"
        ),
        "experiment_mode": "multi_subject_option_a" if cfg.data.is_multi_subject else "single_subject_pilot",
        "subject_id": dataset.subject_id,
        "n_eligible_participants": len(dataset.subject_ids),
        "eligible_subject_ids": list(dataset.subject_ids),
        "excluded_subjects": [
            {
                "subject_id": r.subject_id,
                "reason_code": r.reason_code,
                "reason_detail": r.reason_detail,
            }
            for r in dataset.eligibility_records
            if not r.eligible
        ],
        "random_seed": cfg.random_seed,
        "selection_protocol": {
            "primary_metric": "mean_f1_macro",
            "tie_breaking_rule": SELECTION_TIE_BREAKING_RULE_DOC,
            "predetermined_primary_seed": int(cfg.random_seed),
        },
        "development_runs_refit": list(dataset.dev.run_ids),
        "n_development_recordings": len(set(dataset.dev.recording_ids)),
        "held_out_test_run_evaluated_once": int(cfg.data.held_out_test_run),
        "n_test_recordings": len(set(dataset.test.recording_ids)),
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
        "methodological_disclosures_and_limitations": disclosures,
        "rks_predetermined_seed_variability": {
            "locked_config_id": best_rks_spec.config_id,
            "predetermined_seeds": predetermined_seeds,
            "primary_locked_seed": int(cfg.random_seed),
            "test_per_seed_accuracy": rks_test_seed_accs,
            "test_per_seed_f1_macro": rks_test_seed_f1s,
            "test_seed_mean_accuracy": float(np.mean(rks_test_seed_accs)),
            "test_seed_std_accuracy": (
                float(np.std(rks_test_seed_accs, ddof=1)) if len(rks_test_seed_accs) > 1 else 0.0
            ),
            "test_seed_mean_f1_macro": float(np.mean(rks_test_seed_f1s)),
            "test_seed_std_f1_macro": (
                float(np.std(rks_test_seed_f1s, ddof=1)) if len(rks_test_seed_f1s) > 1 else 0.0
            ),
            "dev_seed_variability": (
                e2_report.get("rks_predetermined_seed_variability_on_dev")
                if e2_report is not None
                else None
            ),
        },
        "locked_test_results": locked_results,
        "per_participant_test_results": per_participant_test_rows,
        "hosvd_test_rank_curve": rank_curve_df.to_dict(orient="records"),
        "learned_markov_models_on_runs_4_8": learned_markov_transitions,
    }

    return locked_df, rank_curve_df, report_dict


SELECTION_TIE_BREAKING_RULE_DOC = (
    "Primary selection metric: pooled development validation mean_f1_macro across Fold 1 (Run 4->8) "
    "and Fold 2 (Run 8->4). Deterministic tie-breaking order: "
    "(1) higher mean_f1_macro, (2) higher mean_balanced_accuracy, (3) higher mean_accuracy, "
    "(4) higher participant_mean_f1_macro, (5) lower fold_std_f1_macro (cross-fold stability), "
    "(6) smaller n_features (higher compression ratio), (7) smaller SVM regularization parameter C."
)


def select_best_dev_configuration(family_df: pd.DataFrame) -> dict[str, Any]:
    """Select the single best hyperparameter configuration for a model family using development CV only.

    Follows SELECTION_TIE_BREAKING_RULE_DOC deterministically without ever inspecting Run 12.
    """
    if family_df.empty:
        raise ValueError("Cannot select best configuration from an empty DataFrame.")

    df = family_df.copy()
    if "C" not in df.columns:
        df["C"] = [
            float(json.loads(p).get("C", 1.0)) for p in df["params_json"]
        ]
    if "mean_balanced_accuracy" not in df.columns:
        df["mean_balanced_accuracy"] = df["mean_accuracy"]
    if "participant_mean_f1_macro" not in df.columns:
        df["participant_mean_f1_macro"] = df["mean_f1_macro"]
    if "std_f1_macro" not in df.columns:
        df["std_f1_macro"] = 0.0

    sorted_df = df.sort_values(
        by=[
            "mean_f1_macro",
            "mean_balanced_accuracy",
            "mean_accuracy",
            "participant_mean_f1_macro",
            "std_f1_macro",
            "n_features",
            "C",
        ],
        ascending=[False, False, False, False, True, True, True],
        kind="mergesort",
    )
    return sorted_df.iloc[0].to_dict()


def run_e2_cohort_development_validation(
    dataset: PreparedEEGDataset,
    cfg: ProjectConfig,
    predetermined_seeds: tuple[int, ...] = (42, 1337, 2026),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run Stage 2 (E2) multi-participant development validation on Runs 4 and 8 ONLY.

    Enforces all Option A cohort rules:
    - Uses ONLY development Runs 4 and 8 across the eligible cohort participants.
    - Held-out Test Run 12 is never accessed or evaluated in Stage 2.
    - Fold 0 trains on Run 4 across all eligible participants and validates on Run 8.
    - Fold 1 trains on Run 8 across all eligible participants and validates on Run 4.
    - Fits all learned transforms (StandardScaler, HOSVD bases, RKS weights, Markov quantizer
      and class-conditional transition models, and SVM classifiers) strictly on each training fold.
    - Computes both pooled metrics and participant-grouped metrics (with 95% Student-t CIs across
      participants) and distinguishes 2-fold validation standard deviation from participant CIs.
    - Evaluates predetermined multi-seed variability on development folds for randomized models.
    """
    assert dataset.dev.subject_ids is not None
    splits = build_run_swapped_splits(
        groups=dataset.dev.groups,
        development_runs=cfg.data.development_runs,
        forbidden_test_run=cfg.data.held_out_test_run,
        recording_ids=dataset.dev.recording_ids,
        subject_ids=dataset.dev.subject_ids,
    )

    X_raw_dev = dataset.dev.X
    y_dev = dataset.dev.y
    subj_dev = dataset.dev.subject_ids
    n_channels = int(X_raw_dev.shape[1])
    n_times = int(X_raw_dev.shape[2])
    raw_elements_per_trial = n_channels * n_times

    feat_cfg = cfg.raw.get("features", {})
    mu_band = tuple(float(x) for x in feat_cfg.get("mu_band_hz", [8.0, 13.0]))
    beta_band = tuple(float(x) for x in feat_cfg.get("beta_band_hz", [13.0, 30.0]))
    log_eps = float(feat_cfg.get("log_epsilon", 1e-12))

    hosvd_cfg = cfg.raw.get("hosvd", {})
    r_c_grid = sorted({int(r) for r in hosvd_cfg.get("channel_ranks", [4, 8, 16])})
    r_t_grid = sorted({int(r) for r in hosvd_cfg.get("time_ranks", [10, 20, 40])})

    m_cfg = cfg.raw.get("markov", {})
    k_grid = sorted({int(k) for k in m_cfg.get("n_states_grid", [3, 4, 6])})
    sub_samples = int(m_cfg.get("subwindow_samples", 40))
    sub_stride = int(m_cfg.get("subwindow_stride", 20))
    laplace_alpha = float(m_cfg.get("laplace_alpha", 0.5))

    # Pre-extract fold-level shared representations strictly on each fold's training split
    fold_cache: dict[int, dict[str, Any]] = {}
    for fold in splits:
        X_tr = X_raw_dev[fold.train_indices]
        y_tr = y_dev[fold.train_indices]
        X_va = X_raw_dev[fold.val_indices]

        # 1. Deterministic mu/beta log-bandpower per fold
        t0 = time.perf_counter()
        bp_tr = extract_mu_beta_log_bandpower(
            X_tr, sfreq=dataset.dev.sfreq, mu_band_hz=(mu_band[0], mu_band[1]),
            beta_band_hz=(beta_band[0], beta_band[1]), log_epsilon=log_eps,
        )
        bp_tr_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        bp_va = extract_mu_beta_log_bandpower(
            X_va, sfreq=dataset.dev.sfreq, mu_band_hz=(mu_band[0], mu_band[1]),
            beta_band_hz=(beta_band[0], beta_band[1]), log_epsilon=log_eps,
        )
        bp_va_time = time.perf_counter() - t1

        # 2. Train-only HOSVD singular spaces per fold
        t_svd = time.perf_counter()
        U_c_full, s_c_full, U_t_full, s_t_full = fit_full_hosvd_singular_spaces(X_tr)
        svd_fit_time = time.perf_counter() - t_svd

        hosvd_rank_pairs = {
            (r_c, r_t) for r_c in r_c_grid for r_t in r_t_grid
        }
        for m_spec in get_markov_model_specs(cfg):
            if m_spec.r_c is not None and m_spec.r_t is not None:
                hosvd_rank_pairs.add((int(m_spec.r_c), int(m_spec.r_t)))

        hosvd_by_rank: dict[tuple[int, int], dict[str, Any]] = {}
        for r_c, r_t in sorted(hosvd_rank_pairs):
            t_p0 = time.perf_counter()
            bases = slice_hosvd_bases(U_c_full, s_c_full, U_t_full, s_t_full, r_c=r_c, r_t=r_t)
            h_tr = project_hosvd_trials(X_tr, bases.U_c, bases.U_t, flatten=True)
            proj_tr_time = svd_fit_time + (time.perf_counter() - t_p0)

            t_p1 = time.perf_counter()
            h_va = project_hosvd_trials(X_va, bases.U_c, bases.U_t, flatten=True)
            proj_va_time = time.perf_counter() - t_p1

            tr_err = compute_relative_frobenius_error(X_tr, bases.U_c, bases.U_t)
            va_err = compute_relative_frobenius_error(X_va, bases.U_c, bases.U_t)
            hosvd_by_rank[(r_c, r_t)] = {
                "train_feat": h_tr,
                "val_feat": h_va,
                "train_time": proj_tr_time,
                "val_time": proj_va_time,
                "train_recon_err": tr_err,
                "val_recon_err": va_err,
            }

        # 3. Train-only Intra-Trial Markov features per (fold, K)
        markov_by_k: dict[int, dict[str, Any]] = {}
        for k_val in k_grid:
            m_ext = IntraTrialMarkovFeatureExtractor(
                n_states=k_val,
                subwindow_samples=sub_samples,
                subwindow_stride=sub_stride,
                spatial_rank_sub=4,
                laplace_alpha=laplace_alpha,
                random_state=cfg.random_seed,
            )
            t_m0 = time.perf_counter()
            m_tr = m_ext.fit_transform(X_tr, y_tr)
            m_tr_time = time.perf_counter() - t_m0

            t_m1 = time.perf_counter()
            m_va = m_ext.transform(X_va)
            m_va_time = time.perf_counter() - t_m1

            markov_by_k[k_val] = {
                "train_feat": m_tr,
                "val_feat": m_va,
                "train_time": m_tr_time,
                "val_time": m_va_time,
            }

        fold_cache[fold.fold_index] = {
            "bp_tr": bp_tr,
            "bp_va": bp_va,
            "bp_tr_time": bp_tr_time,
            "bp_va_time": bp_va_time,
            "hosvd_by_rank": hosvd_by_rank,
            "markov_by_k": markov_by_k,
        }

    fold_records: list[dict[str, Any]] = []
    oof_predictions: dict[str, np.ndarray] = {}

    def _init_oof(config_id: str) -> np.ndarray:
        if config_id not in oof_predictions:
            oof_predictions[config_id] = np.zeros_like(y_dev)
        return oof_predictions[config_id]

    # A. Evaluate Baselines (M2)
    for spec in get_baseline_model_specs(cfg):
        oof = _init_oof(spec.config_id)
        for fold in splits:
            fc = fold_cache[fold.fold_index]
            y_tr = y_dev[fold.train_indices]
            y_va = y_dev[fold.val_indices]
            s_va = subj_dev[fold.val_indices]

            pipeline = build_baseline_pipeline(spec)
            t_fit = time.perf_counter()
            pipeline.fit(fc["bp_tr"], y_tr)
            svm_train_time = time.perf_counter() - t_fit

            t_pred = time.perf_counter()
            y_pred = pipeline.predict(fc["bp_va"])
            svm_pred_time = time.perf_counter() - t_pred
            oof[fold.val_indices] = y_pred

            metrics = compute_classification_metrics(y_va, y_pred)
            part_summary, _ = compute_participant_grouped_metrics(y_va, y_pred, s_va)
            n_feat = int(fc["bp_tr"].shape[1])
            model_bytes = estimate_fitted_pipeline_storage_bytes(pipeline)

            fold_records.append({
                "model_family": spec.model_family,
                "config_id": spec.config_id,
                "params_json": json.dumps(spec.params, sort_keys=True),
                "C": float(spec.params["C"]),
                "fold_index": fold.fold_index,
                "train_run": fold.train_run_id,
                "val_run": fold.val_run_id,
                "n_train_trials": int(len(y_tr)),
                "n_val_trials": int(len(y_va)),
                "n_features": n_feat,
                "compression_ratio_vs_raw": float(raw_elements_per_trial / n_feat),
                "hosvd_train_reconstruction_error": None,
                "hosvd_reconstruction_error": None,
                "feature_bytes_per_trial": n_feat * 8,
                "model_storage_bytes": model_bytes,
                "train_feature_time_sec": float(fc["bp_tr_time"]),
                "svm_train_time_sec": float(svm_train_time),
                "val_feature_time_sec": float(fc["bp_va_time"]),
                "svm_predict_time_sec": float(svm_pred_time),
                "total_prediction_time_sec": float(fc["bp_va_time"] + svm_pred_time),
                **metrics,
                **part_summary,
            })

    # B. Evaluate HOSVD (M3)
    for spec in get_hosvd_model_specs(cfg):
        oof = _init_oof(spec.config_id)
        for fold in splits:
            fc = fold_cache[fold.fold_index]
            h_entry = fc["hosvd_by_rank"][(spec.r_c, spec.r_t)]
            y_tr = y_dev[fold.train_indices]
            y_va = y_dev[fold.val_indices]
            s_va = subj_dev[fold.val_indices]

            pipeline = build_hosvd_classifier_pipeline(spec)
            t_fit = time.perf_counter()
            pipeline.fit(h_entry["train_feat"], y_tr)
            svm_train_time = time.perf_counter() - t_fit

            t_pred = time.perf_counter()
            y_pred = pipeline.predict(h_entry["val_feat"])
            svm_pred_time = time.perf_counter() - t_pred
            oof[fold.val_indices] = y_pred

            metrics = compute_classification_metrics(y_va, y_pred)
            part_summary, _ = compute_participant_grouped_metrics(y_va, y_pred, s_va)
            storage = compute_hosvd_storage_accounting(
                n_trials=len(y_tr), n_channels=n_channels, n_times=n_times,
                r_c=spec.r_c, r_t=spec.r_t,
            )
            scaler_svm_bytes = estimate_fitted_pipeline_storage_bytes(pipeline)

            fold_records.append({
                "model_family": spec.model_family,
                "config_id": spec.config_id,
                "params_json": json.dumps(spec.params, sort_keys=True),
                "r_c": spec.r_c,
                "r_t": spec.r_t,
                "C": float(spec.c_val),
                "fold_index": fold.fold_index,
                "train_run": fold.train_run_id,
                "val_run": fold.val_run_id,
                "n_train_trials": int(len(y_tr)),
                "n_val_trials": int(len(y_va)),
                "n_features": int(storage["n_features"]),
                "compression_ratio_vs_raw": float(storage["compression_ratio_vs_raw"]),
                "tensor_decomposition_compression_ratio": float(
                    storage["tensor_decomposition_compression_ratio"]
                ),
                "hosvd_train_reconstruction_error": float(h_entry["train_recon_err"]),
                "hosvd_reconstruction_error": float(h_entry["val_recon_err"]),
                "feature_bytes_per_trial": int(storage["feature_bytes_per_trial"]),
                "model_storage_bytes": int(storage["reusable_bases_bytes"] + scaler_svm_bytes),
                "train_feature_time_sec": float(h_entry["train_time"]),
                "svm_train_time_sec": float(svm_train_time),
                "val_feature_time_sec": float(h_entry["val_time"]),
                "svm_predict_time_sec": float(svm_pred_time),
                "total_prediction_time_sec": float(h_entry["val_time"] + svm_pred_time),
                **metrics,
                **part_summary,
            })

    # C. Evaluate RKS (M4)
    rks_fold_cache: dict[tuple[int, int, float], dict[str, Any]] = {}
    for spec in get_rks_model_specs(cfg):
        oof = _init_oof(spec.config_id)
        for fold in splits:
            fc = fold_cache[fold.fold_index]
            y_tr = y_dev[fold.train_indices]
            y_va = y_dev[fold.val_indices]
            s_va = subj_dev[fold.val_indices]

            rks_key = (fold.fold_index, spec.n_components, spec.gamma)
            if rks_key not in rks_fold_cache:
                mapper = RKSFeatureMapper(
                    n_components=spec.n_components, gamma=spec.gamma, random_state=spec.random_seed
                )
                t_r0 = time.perf_counter()
                z_tr = mapper.fit_transform(fc["bp_tr"])
                rks_tr_time = fc["bp_tr_time"] + (time.perf_counter() - t_r0)

                t_r1 = time.perf_counter()
                z_va = mapper.transform(fc["bp_va"])
                rks_va_time = fc["bp_va_time"] + (time.perf_counter() - t_r1)
                rks_fold_cache[rks_key] = {
                    "train_feat": z_tr,
                    "val_feat": z_va,
                    "train_time": rks_tr_time,
                    "val_time": rks_va_time,
                    "mapper_bytes": mapper.storage_bytes(),
                }
            rc = rks_fold_cache[rks_key]

            clf = build_rks_classifier_(spec)
            t_fit = time.perf_counter()
            clf.fit(rc["train_feat"], y_tr)
            svm_train_time = time.perf_counter() - t_fit

            t_pred = time.perf_counter()
            y_pred = clf.predict(rc["val_feat"])
            svm_pred_time = time.perf_counter() - t_pred
            oof[fold.val_indices] = y_pred

            metrics = compute_classification_metrics(y_va, y_pred)
            part_summary, _ = compute_participant_grouped_metrics(y_va, y_pred, s_va)
            storage = compute_rks_storage_accounting(
                input_dim=int(fc["bp_tr"].shape[1]),
                n_components=spec.n_components,
                raw_trial_elements=raw_elements_per_trial,
            )
            svm_bytes = estimate_linear_svc_storage_bytes(clf)

            fold_records.append({
                "model_family": spec.model_family,
                "config_id": spec.config_id,
                "params_json": json.dumps(spec.params, sort_keys=True),
                "n_components": spec.n_components,
                "gamma": spec.gamma,
                "C": float(spec.c_val),
                "fold_index": fold.fold_index,
                "train_run": fold.train_run_id,
                "val_run": fold.val_run_id,
                "n_train_trials": int(len(y_tr)),
                "n_val_trials": int(len(y_va)),
                "n_features": int(storage["n_features"]),
                "expansion_ratio_vs_bandpower": float(storage["expansion_ratio_vs_bandpower"]),
                "compression_ratio_vs_raw": float(storage["compression_ratio_vs_raw"]),
                "hosvd_train_reconstruction_error": None,
                "hosvd_reconstruction_error": None,
                "feature_bytes_per_trial": int(storage["feature_bytes_per_trial"]),
                "model_storage_bytes": int(rc["mapper_bytes"] + svm_bytes),
                "train_feature_time_sec": float(rc["train_time"]),
                "svm_train_time_sec": float(svm_train_time),
                "val_feature_time_sec": float(rc["val_time"]),
                "svm_predict_time_sec": float(svm_pred_time),
                "total_prediction_time_sec": float(rc["val_time"] + svm_pred_time),
                **metrics,
                **part_summary,
            })

    # D. Evaluate Intra-Trial Markov Chain & Hybrid Fusions (M4b)
    for spec in get_markov_model_specs(cfg):
        oof = _init_oof(spec.config_id)
        for fold in splits:
            fc = fold_cache[fold.fold_index]
            mc = fc["markov_by_k"][spec.n_states]
            y_tr = y_dev[fold.train_indices]
            y_va = y_dev[fold.val_indices]
            s_va = subj_dev[fold.val_indices]

            markov_storage = compute_markov_storage_accounting(
                n_states=spec.n_states,
                n_channels=n_channels,
                spatial_rank_sub=4,
                raw_trial_elements=raw_elements_per_trial,
            )
            extra_bytes = int(markov_storage["markov_extractor_bytes"])
            tr_recon_err: float | None = None
            va_recon_err: float | None = None

            if spec.fusion_mode == "standalone":
                f_tr = mc["train_feat"]
                f_va = mc["val_feat"]
                tr_time = mc["train_time"]
                va_time = mc["val_time"]
            elif spec.fusion_mode == "hosvd_fusion":
                assert spec.r_c is not None and spec.r_t is not None
                hc = fc["hosvd_by_rank"][(spec.r_c, spec.r_t)]
                f_tr = np.concatenate([hc["train_feat"], mc["train_feat"]], axis=1)
                f_va = np.concatenate([hc["val_feat"], mc["val_feat"]], axis=1)
                tr_time = hc["train_time"] + mc["train_time"]
                va_time = hc["val_time"] + mc["val_time"]
                tr_recon_err = float(hc["train_recon_err"])
                va_recon_err = float(hc["val_recon_err"])
                h_storage = compute_hosvd_storage_accounting(
                    n_trials=len(y_tr), n_channels=n_channels, n_times=n_times,
                    r_c=spec.r_c, r_t=spec.r_t,
                )
                extra_bytes += int(h_storage["reusable_bases_bytes"])
            elif spec.fusion_mode == "rks_fusion":
                assert spec.rks_dim is not None and spec.rks_gamma is not None
                rks_key = (fold.fold_index, spec.rks_dim, spec.rks_gamma)
                if rks_key not in rks_fold_cache:
                    mapper = RKSFeatureMapper(
                        n_components=spec.rks_dim,
                        gamma=spec.rks_gamma,
                        random_state=spec.random_seed,
                    )
                    t_r0 = time.perf_counter()
                    z_tr = mapper.fit_transform(fc["bp_tr"])
                    rks_tr_time = fc["bp_tr_time"] + (time.perf_counter() - t_r0)
                    t_r1 = time.perf_counter()
                    z_va = mapper.transform(fc["bp_va"])
                    rks_va_time = fc["bp_va_time"] + (time.perf_counter() - t_r1)
                    rks_fold_cache[rks_key] = {
                        "train_feat": z_tr,
                        "val_feat": z_va,
                        "train_time": rks_tr_time,
                        "val_time": rks_va_time,
                        "mapper_bytes": mapper.storage_bytes(),
                    }
                rc = rks_fold_cache[rks_key]
                f_tr = np.concatenate([rc["train_feat"], mc["train_feat"]], axis=1)
                f_va = np.concatenate([rc["val_feat"], mc["val_feat"]], axis=1)
                tr_time = rc["train_time"] + mc["train_time"]
                va_time = rc["val_time"] + mc["val_time"]
                extra_bytes += int(rc["mapper_bytes"])
            else:
                raise ValueError(f"Unknown fusion_mode: {spec.fusion_mode}")

            pipeline = build_markov_classifier_pipeline(spec)
            t_fit = time.perf_counter()
            pipeline.fit(f_tr, y_tr)
            svm_train_time = time.perf_counter() - t_fit

            t_pred = time.perf_counter()
            y_pred = pipeline.predict(f_va)
            svm_pred_time = time.perf_counter() - t_pred
            oof[fold.val_indices] = y_pred

            metrics = compute_classification_metrics(y_va, y_pred)
            part_summary, _ = compute_participant_grouped_metrics(y_va, y_pred, s_va)
            n_feat = int(f_tr.shape[1])
            scaler_svm_bytes = estimate_fitted_pipeline_storage_bytes(pipeline)

            fold_records.append({
                "model_family": spec.model_family,
                "config_id": spec.config_id,
                "fusion_mode": spec.fusion_mode,
                "n_states": spec.n_states,
                "C": float(spec.c_val),
                "params_json": json.dumps(spec.params, sort_keys=True),
                "fold_index": fold.fold_index,
                "train_run": fold.train_run_id,
                "val_run": fold.val_run_id,
                "n_train_trials": int(len(y_tr)),
                "n_val_trials": int(len(y_va)),
                "n_features": n_feat,
                "compression_ratio_vs_raw": float(raw_elements_per_trial / n_feat),
                "hosvd_train_reconstruction_error": tr_recon_err,
                "hosvd_reconstruction_error": va_recon_err,
                "feature_bytes_per_trial": n_feat * 8,
                "model_storage_bytes": int(extra_bytes + scaler_svm_bytes),
                "train_feature_time_sec": float(tr_time),
                "svm_train_time_sec": float(svm_train_time),
                "val_feature_time_sec": float(va_time),
                "svm_predict_time_sec": float(svm_pred_time),
                "total_prediction_time_sec": float(va_time + svm_pred_time),
                **metrics,
                **part_summary,
            })

    fold_df = pd.DataFrame(fold_records)

    # Aggregate across Fold 0 and Fold 1 + compute participant-grouped OOF statistics across Runs 4 & 8
    summary_records: list[dict[str, Any]] = []
    per_subject_oof_map: dict[str, list[dict[str, Any]]] = {}

    for config_id, grp in fold_df.groupby("config_id", sort=False):
        first = grp.iloc[0]
        oof_pred = oof_predictions[str(config_id)]
        oof_part_summary, oof_part_rows = compute_participant_grouped_metrics(
            y_dev, oof_pred, subj_dev
        )
        per_subject_oof_map[str(config_id)] = oof_part_rows
        oof_cm = confusion_matrix(y_dev, oof_pred, labels=[0, 1]).tolist()

        recon_tr_vals = grp["hosvd_train_reconstruction_error"].dropna()
        recon_va_vals = grp["hosvd_reconstruction_error"].dropna()
        mean_val_trials = float(grp["n_val_trials"].mean())
        mean_tot_pred_sec = float(grp["total_prediction_time_sec"].mean())

        summary_records.append({
            "model_family": str(first["model_family"]),
            "config_id": str(config_id),
            "params_json": str(first["params_json"]),
            "C": float(first["C"]),
            "n_features": int(first["n_features"]),
            "compression_ratio_vs_raw": float(first["compression_ratio_vs_raw"]),
            "mean_hosvd_train_reconstruction_error": (
                float(recon_tr_vals.mean()) if len(recon_tr_vals) > 0 else None
            ),
            "mean_hosvd_val_reconstruction_error": (
                float(recon_va_vals.mean()) if len(recon_va_vals) > 0 else None
            ),
            "feature_bytes_per_trial": int(first["feature_bytes_per_trial"]),
            "model_storage_bytes_mean": float(grp["model_storage_bytes"].mean()),
            "mean_accuracy": float(grp["accuracy"].mean()),
            "std_accuracy": float(grp["accuracy"].std(ddof=0)),
            "fold1_run4to8_accuracy": float(grp.loc[grp["fold_index"] == 0, "accuracy"].iloc[0]),
            "fold2_run8to4_accuracy": float(grp.loc[grp["fold_index"] == 1, "accuracy"].iloc[0]),
            "mean_balanced_accuracy": float(grp["balanced_accuracy"].mean()),
            "std_balanced_accuracy": float(grp["balanced_accuracy"].std(ddof=0)),
            "mean_precision_macro": float(grp["precision_macro"].mean()),
            "mean_recall_macro": float(grp["recall_macro"].mean()),
            "mean_f1_macro": float(grp["f1_macro"].mean()),
            "std_f1_macro": float(grp["f1_macro"].std(ddof=0)),
            "pooled_oof_confusion_matrix": oof_cm,
            **oof_part_summary,
            "mean_train_feature_time_sec": float(grp["train_feature_time_sec"].mean()),
            "mean_svm_train_time_sec": float(grp["svm_train_time_sec"].mean()),
            "mean_val_feature_time_sec": float(grp["val_feature_time_sec"].mean()),
            "mean_svm_predict_time_sec": float(grp["svm_predict_time_sec"].mean()),
            "mean_total_prediction_time_sec": mean_tot_pred_sec,
            "mean_prediction_time_per_trial_ms": float(
                mean_tot_pred_sec * 1000.0 / max(mean_val_trials, 1.0)
            ),
        })

    summary_df = pd.DataFrame(summary_records)

    # Select best configuration per model family using documented deterministic tie-breaking rule
    families_order = (
        "baseline_linear_svc",
        "baseline_rbf_svc",
        "hosvd_linear_svc",
        "rks_linear_svc",
        "markov_linear_svc",
        "hosvd_markov_linear_svc",
        "rks_markov_linear_svc",
    )
    best_by_family_rows: list[dict[str, Any]] = []
    per_participant_best_rows: list[dict[str, Any]] = []

    for fam in families_order:
        fam_df = summary_df[summary_df["model_family"] == fam]
        if fam_df.empty:
            continue
        best_row = select_best_dev_configuration(fam_df)
        best_by_family_rows.append(best_row)
        best_cid = str(best_row["config_id"])
        for s_row in per_subject_oof_map[best_cid]:
            per_participant_best_rows.append({
                "model_family": fam,
                "locked_config_id": best_cid,
                **s_row,
            })

    best_by_family_df = pd.DataFrame(best_by_family_rows)
    per_participant_df = pd.DataFrame(per_participant_best_rows)

    # HOSVD rank-pair summary table on development folds
    hosvd_only_df = summary_df[summary_df["model_family"] == "hosvd_linear_svc"]
    rank_records: list[dict[str, Any]] = []
    for r_c in r_c_grid:
        for r_t in r_t_grid:
            sub = hosvd_only_df[
                hosvd_only_df["config_id"].str.startswith(f"HOSVD_rc={r_c}_rt={r_t}_")
            ]
            if sub.empty:
                continue
            best_r = select_best_dev_configuration(sub)
            rank_records.append({
                "r_c": r_c,
                "r_t": r_t,
                "n_features": int(best_r["n_features"]),
                "compression_ratio_vs_raw": float(best_r["compression_ratio_vs_raw"]),
                "mean_hosvd_train_reconstruction_error": best_r["mean_hosvd_train_reconstruction_error"],
                "mean_hosvd_val_reconstruction_error": best_r["mean_hosvd_val_reconstruction_error"],
                "feature_bytes_per_trial": int(best_r["feature_bytes_per_trial"]),
                "best_C": float(best_r["C"]),
                "best_mean_accuracy": float(best_r["mean_accuracy"]),
                "best_mean_balanced_accuracy": float(best_r["mean_balanced_accuracy"]),
                "best_mean_f1_macro": float(best_r["mean_f1_macro"]),
                "participant_mean_accuracy": float(best_r["participant_mean_accuracy"]),
                "participant_ci95_accuracy": best_r["participant_ci95_accuracy"],
            })
    rank_df = pd.DataFrame(rank_records)

    # Evaluate multi-seed variability on development folds for selected best RKS configuration
    best_rks_row = next(r for r in best_by_family_rows if r["model_family"] == "rks_linear_svc")
    best_rks_params = json.loads(str(best_rks_row["params_json"]))
    rks_seed_f1s: list[float] = []
    rks_seed_accs: list[float] = []
    for seed_val in predetermined_seeds:
        fold_f1s = []
        fold_accs = []
        for fold in splits:
            fc = fold_cache[fold.fold_index]
            y_tr = y_dev[fold.train_indices]
            y_va = y_dev[fold.val_indices]
            mapper = RKSFeatureMapper(
                n_components=int(best_rks_params["n_components"]),
                gamma=float(best_rks_params["gamma"]),
                random_state=int(seed_val),
            )
            z_tr = mapper.fit_transform(fc["bp_tr"])
            z_va = mapper.transform(fc["bp_va"])
            clf = build_rks_classifier_(
                RKSModelSpec(
                    model_family="rks_linear_svc",
                    config_id="seed_check",
                    n_components=int(best_rks_params["n_components"]),
                    gamma=float(best_rks_params["gamma"]),
                    c_val=float(best_rks_params["C"]),
                    params=best_rks_params,
                    random_seed=int(seed_val),
                    max_iter=int(cfg.raw.get("baselines", {}).get("max_iter", 5000)),
                )
            )
            clf.fit(z_tr, y_tr)
            yp = clf.predict(z_va)
            m = compute_classification_metrics(y_va, yp)
            fold_f1s.append(float(m["f1_macro"]))
            fold_accs.append(float(m["accuracy"]))
        rks_seed_f1s.append(float(np.mean(fold_f1s)))
        rks_seed_accs.append(float(np.mean(fold_accs)))

    report_dict: dict[str, Any] = {
        "milestone": "E2 (Confirmatory Cohort Multi-Participant Development Validation — Runs 4 & 8 Only)",
        "experiment_mode": "multi_subject_option_a" if cfg.data.is_multi_subject else "single_subject_pilot",
        "n_eligible_participants": len(dataset.subject_ids),
        "eligible_subject_ids": list(dataset.subject_ids),
        "development_runs_used": list(dataset.dev.run_ids),
        "n_development_recordings": len(set(dataset.dev.recording_ids)),
        "n_development_trials": int(len(y_dev)),
        "development_class_counts": {
            "T1_left_0": int(np.sum(y_dev == 0)),
            "T2_right_1": int(np.sum(y_dev == 1)),
        },
        "held_out_test_run_untouched": True,
        "held_out_test_run_id": int(cfg.data.held_out_test_run),
        "selection_protocol": {
            "primary_metric": "mean_f1_macro",
            "tie_breaking_rule": SELECTION_TIE_BREAKING_RULE_DOC,
            "predetermined_primary_seed": int(cfg.random_seed),
        },
        "uncertainty_accounting_note": (
            "Distinguishes (1) 2-fold validation standard deviation (std_accuracy, std_f1_macro across "
            "Fold 1 Run 4->8 and Fold 2 Run 8->4, which measures inter-run split spread and is NOT a "
            "confidence interval) from (2) participant-clustered 95% confidence intervals "
            "(participant_ci95_accuracy, participant_ci95_f1_macro computed across the N=105 independent "
            "participants using the Student-t distribution with df=N-1, accounting for within-subject "
            "trial correlation)."
        ),
        "rks_predetermined_seed_variability_on_dev": {
            "config_id": str(best_rks_row["config_id"]),
            "predetermined_seeds": list(predetermined_seeds),
            "per_seed_mean_accuracy": rks_seed_accs,
            "per_seed_mean_f1_macro": rks_seed_f1s,
            "seed_mean_accuracy": float(np.mean(rks_seed_accs)),
            "seed_std_accuracy": float(np.std(rks_seed_accs, ddof=1)) if len(rks_seed_accs) > 1 else 0.0,
            "seed_mean_f1_macro": float(np.mean(rks_seed_f1s)),
            "seed_std_f1_macro": float(np.std(rks_seed_f1s, ddof=1)) if len(rks_seed_f1s) > 1 else 0.0,
        },
        "locked_best_by_family_on_dev": best_by_family_rows,
        "hosvd_rank_sweep_summary_on_dev": rank_records,
    }

    return fold_df, summary_df, best_by_family_df, per_participant_df, report_dict

