"""Publication-grade plotting utilities for EEG waveforms and model comparisons (M1 & M6)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from eeg_p3.preprocessing import RunEpochsResult, filter_raw_run_independently


def plot_raw_vs_filtered_trace(
    raw: mne.io.BaseRaw,
    output_path: str | Path,
    channel_name: str = "C3",
    duration_sec: float = 10.0,
    low_hz: float = 8.0,
    high_hz: float = 30.0,
) -> Path:
    """Plot raw vs. 8-30 Hz band-pass filtered continuous EEG trace with annotations.

    Converts Volts to microvolts (1 V = 1e6 uV) explicitly for visual display.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw_filtered = filter_raw_run_independently(raw, low_hz=low_hz, high_hz=high_hz)

    ch_idx = raw.ch_names.index(channel_name) if channel_name in raw.ch_names else 0
    selected_ch = raw.ch_names[ch_idx]
    sfreq = float(raw.info["sfreq"])
    n_samples = min(int(round(duration_sec * sfreq)), raw.n_times)
    times = np.arange(n_samples, dtype=np.float64) / sfreq

    raw_trace_uv = raw.get_data(picks=[ch_idx], stop=n_samples, units="V")[0] * 1e6
    filt_trace_uv = raw_filtered.get_data(picks=[ch_idx], stop=n_samples, units="V")[0] * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)

    axes[0].plot(times, raw_trace_uv, color="#1f4e78", linewidth=0.9, label=f"Raw EEG ({selected_ch})")
    axes[0].set_ylabel("Amplitude ($\\mu$V)\n[Converted from V]")
    axes[0].set_title(
        f"PhysioNet EEGBCI Subject 1 — Raw vs. {low_hz:.0f}–{high_hz:.0f} Hz Band-Pass Filtered Trace "
        f"(Channel {selected_ch})"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        times,
        filt_trace_uv,
        color="#d95f02",
        linewidth=0.9,
        label=f"{low_hz:.0f}–{high_hz:.0f} Hz FIR Filtered ({selected_ch})",
    )
    axes[1].set_ylabel("Amplitude ($\\mu$V)\n[Converted from V]")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].grid(True, alpha=0.3)

    # Overlay cue annotations within the displayed window
    ann_colors = {"T0": "#7f7f7f", "T1": "#1b9e77", "T2": "#7570b3"}
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        if float(onset) <= duration_sec:
            label_str = str(desc).strip()
            color = ann_colors.get(label_str, "#333333")
            for ax in axes:
                ax.axvline(float(onset), color=color, linestyle="--", linewidth=1.2, alpha=0.85)
            axes[0].text(
                float(onset) + 0.08,
                float(np.max(raw_trace_uv)) * 0.85,
                f"Cue {label_str}",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_labeled_motor_imagery_trials(
    run_result: RunEpochsResult,
    output_path: str | Path,
    channels_to_show: tuple[str, ...] = ("C3", "Cz", "C4"),
) -> Path:
    """Plot one example Left Imagery (T1->0) trial and one Right Imagery (T2->1) trial.

    Displays sensorimotor channels (C3, Cz, C4) across the [0.5, 2.5) s post-cue window
    in microvolts (explicitly scaled from Volts by 1e6).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    left_indices = np.where(run_result.y == 0)[0]
    right_indices = np.where(run_result.y == 1)[0]
    if len(left_indices) == 0 or len(right_indices) == 0:
        raise ValueError("Need at least one T1 (left) and one T2 (right) trial to plot comparison.")

    idx_left = int(left_indices[0])
    idx_right = int(right_indices[0])

    picked_indices: list[int] = []
    picked_names: list[str] = []
    for ch in channels_to_show:
        if ch in run_result.channel_names:
            picked_indices.append(run_result.channel_names.index(ch))
            picked_names.append(ch)
    if not picked_indices:
        picked_indices = list(range(min(3, len(run_result.channel_names))))
        picked_names = [run_result.channel_names[i] for i in picked_indices]

    fig, axes = plt.subplots(len(picked_indices), 2, figsize=(11, 6.2), sharex=True, sharey=True)
    if len(picked_indices) == 1:
        axes = np.asarray([axes])

    meta_left = run_result.trials_metadata[idx_left]
    meta_right = run_result.trials_metadata[idx_right]

    for row_idx, (ch_idx, ch_name) in enumerate(zip(picked_indices, picked_names)):
        trace_left_uv = run_result.X[idx_left, ch_idx, :] * 1e6
        trace_right_uv = run_result.X[idx_right, ch_idx, :] * 1e6

        axes[row_idx, 0].plot(run_result.times_sec, trace_left_uv, color="#1b9e77", linewidth=1.0)
        axes[row_idx, 0].set_ylabel(f"{ch_name} ($\\mu$V)")
        axes[row_idx, 0].grid(True, alpha=0.3)

        axes[row_idx, 1].plot(run_result.times_sec, trace_right_uv, color="#7570b3", linewidth=1.0)
        axes[row_idx, 1].grid(True, alpha=0.3)

    axes[0, 0].set_title(
        f"Run {run_result.run_id} Trial #{meta_left.trial_index_in_run}: "
        f"{meta_left.annotation_label} → Label 0 (Left Fist Imagery)"
    )
    axes[0, 1].set_title(
        f"Run {run_result.run_id} Trial #{meta_right.trial_index_in_run}: "
        f"{meta_right.annotation_label} → Label 1 (Right Fist Imagery)"
    )
    axes[-1, 0].set_xlabel("Post-Cue Time (s) [320 samples @ 160 Hz]")
    axes[-1, 1].set_xlabel("Post-Cue Time (s) [320 samples @ 160 Hz]")

    fig.suptitle(
        f"Subject {run_result.subject_id} Run {run_result.run_id} — "
        f"8–30 Hz Filtered Motor-Imagery Epochs [0.5, 2.5) s",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_m6_hosvd_rank_curves(
    rank_curve_records: list[dict],
    output_path: str | Path,
) -> Path:
    """Plot HOSVD rank sweep reconstruction error (Runs 4+8 vs Run 12) and compression ratios."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    labels = [f"({r['r_c']},{r['r_t']})\n{r['n_features']}D" for r in rank_curve_records]
    x = np.arange(len(labels))
    train_err = [float(r["train_runs_4_8_reconstruction_error"]) for r in rank_curve_records]
    test_err = [float(r["test_run_12_reconstruction_error"]) for r in rank_curve_records]
    cr_trial = [float(r["compression_ratio_vs_raw"]) for r in rank_curve_records]
    cr_tensor = [float(r["tensor_decomposition_compression_ratio"]) for r in rank_curve_records]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    axes[0].plot(
        x, train_err, marker="o", linewidth=2.0, color="#1f4e78", label="Train Runs 4+8 Recon Error"
    )
    axes[0].plot(
        x, test_err, marker="s", linewidth=2.0, color="#d95f02", label="Held-Out Test Run 12 Recon Error"
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8.5)
    axes[0].set_ylabel("Relative Frobenius Error ($\\|X - \\hat{X}\\|_F / \\|X\\|_F$)")
    axes[0].set_xlabel("HOSVD Mode Ranks $(r_c, r_t)$ and Feature Count")
    axes[0].set_title("A. HOSVD Reconstruction Error vs. Mode Ranks")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    width = 0.36
    axes[1].bar(
        x - width / 2,
        cr_trial,
        width=width,
        color="#2b8cbe",
        label="Per-Trial Feature Compression ($20,480 / (r_c r_t)$)",
    )
    axes[1].bar(
        x + width / 2,
        cr_tensor,
        width=width,
        color="#7bccc4",
        label="Tucker-2 Tensor Compression ($N=30$)",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_ylabel("Compression Ratio ($\\times$)")
    axes[1].set_xlabel("HOSVD Mode Ranks $(r_c, r_t)$ and Feature Count")
    axes[1].set_title("B. HOSVD Per-Trial & Tensor Compression Ratios")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        "Milestone M6 — HOSVD Rank Trade-off: Reconstruction Error vs. Compression Ratio (Subject 1)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_m6_rks_sweep(
    rks_summary_records: list[dict],
    output_path: str | Path,
) -> Path:
    """Plot RKS random-feature dimension D and bandwidth gamma vs. validation accuracy & storage."""
    import json

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    normalized_records: list[dict] = []
    for r in rks_summary_records:
        rec = dict(r)
        if "n_components" not in rec or "gamma" not in rec:
            p = json.loads(str(rec.get("params_json", "{}")))
            rec.setdefault("n_components", int(p["n_components"]))
            rec.setdefault("gamma", float(p["gamma"]))
        normalized_records.append(rec)

    d_values = sorted({int(r["n_components"]) for r in normalized_records})
    gammas = sorted({float(r["gamma"]) for r in normalized_records})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    markers = ["o", "s", "^"]

    all_plotted_accs: list[float] = []
    for idx, g_val in enumerate(gammas):
        best_accs: list[float] = []
        for d_val in d_values:
            matching = [
                float(r["mean_accuracy"]) * 100.0
                for r in normalized_records
                if int(r["n_components"]) == d_val and np.isclose(float(r["gamma"]), g_val)
            ]
            val = max(matching) if matching else 0.0
            best_accs.append(val)
            all_plotted_accs.append(val)
        axes[0].plot(
            d_values,
            best_accs,
            marker=markers[idx % len(markers)],
            linewidth=2.0,
            color=colors[idx % len(colors)],
            label=f"$\\gamma = {g_val:g}$ (Best $C$)",
        )

    axes[0].set_xticks(d_values)
    axes[0].set_xticklabels([f"D={d}\n({d/128:.0f}x bandpower)" for d in d_values])
    axes[0].set_ylabel("Development CV Mean Accuracy (%) [Runs 4↔8]")
    axes[0].set_xlabel("RKS Random Fourier Components $D$ (Expansion Ratio)")
    axes[0].set_title("A. RKS Bandwidth $\\gamma$ & Dimension $D$ vs. Dev Accuracy")
    min_y = max(35.0, min(all_plotted_accs) - 6.0) if all_plotted_accs else 45.0
    max_y = min(100.0, max(all_plotted_accs) + 8.0) if all_plotted_accs else 90.0
    axes[0].set_ylim(min_y, max_y)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    storage_kb: list[float] = []
    feat_kb: list[float] = []
    for d_val in d_values:
        matching_rows = [r for r in normalized_records if int(r["n_components"]) == d_val]
        storage_kb.append(float(matching_rows[0]["model_storage_bytes_mean"]) / 1024.0)
        feat_kb.append(float(matching_rows[0]["feature_bytes_per_trial"]) / 1024.0)

    x = np.arange(len(d_values))
    axes[1].bar(x, storage_kb, width=0.45, color="#3182bd", label="Deployed Model Storage (KB)")
    for i, val in enumerate(storage_kb):
        axes[1].text(
            i,
            val + 8.0,
            f"{val:.1f} KB\n({feat_kb[i]:.1f} KB/trial)",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"D={d} ({d/128:.0f}x expansion)" for d in d_values])
    axes[1].set_ylabel("Memory / Storage Footprint (KB)")
    axes[1].set_xlabel("RKS Output Dimension $D$")
    axes[1].set_title("B. RKS Storage Scaling with Random Components $D$")
    axes[1].set_ylim(0, max(storage_kb) * 1.25)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper left")

    fig.suptitle(
        "Milestone M6 — Random Kitchen Sinks (RBFSampler) Kernel Approximation Analysis",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_m6_markov_transition_heatmaps(
    markov_model_dict: dict,
    output_path: str | Path,
) -> Path:
    """Plot Class 0 (Left) vs. Class 1 (Right) learned Markov transition matrices P^(0), P^(1), and Delta P."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    P0 = np.asarray(markov_model_dict["class_0_left_transition_matrix"], dtype=np.float64)
    P1 = np.asarray(markov_model_dict["class_1_right_transition_matrix"], dtype=np.float64)
    dP = P1 - P0
    K = int(markov_model_dict["n_states"])
    state_labels = [f"S{i}" for i in range(K)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    matrices = [
        (P0, "Class 0 (T1: Left Imagery)\n$P^{(0)}(S_{m+1} \\mid S_m)$", "Blues", 0.0, 0.8),
        (P1, "Class 1 (T2: Right Imagery)\n$P^{(1)}(S_{m+1} \\mid S_m)$", "Purples", 0.0, 0.8),
        (dP, "Difference $\\Delta P = P^{(1)} - P^{(0)}$\n(Right minus Left)", "coolwarm", -0.2, 0.2),
    ]

    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, matrices):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(K))
        ax.set_yticks(np.arange(K))
        ax.set_xticklabels(state_labels)
        ax.set_yticklabels(state_labels)
        ax.set_xlabel("Next State $S_{m+1}$")
        ax.set_ylabel("Current State $S_m$")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        for i in range(K):
            for j in range(K):
                val = mat[i, j]
                txt_color = "white" if abs(val - vmin) > 0.65 * (vmax - vmin) and cmap != "coolwarm" else "black"
                ax.text(j, i, f"{val:+.2f}" if cmap == "coolwarm" else f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=9.5, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Milestone M6 — Learned Intra-Trial First-Order Markov Transition Matrices ($K={K}$ States, Runs 4+8)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_m6_locked_comparison_dashboard(
    locked_results: list[dict],
    output_path: str | Path,
) -> Path:
    """Generate a 4-panel master dashboard comparing all 7 locked models across the 8 faculty measures."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    short_names = {
        "baseline_linear_svc": "1. Base\nLinearSVC",
        "baseline_rbf_svc": "2. Base\nRBF SVC",
        "hosvd_linear_svc": "3. HOSVD\n(16,40)",
        "rks_linear_svc": "4. RKS\n(D=128)",
        "markov_linear_svc": "5. Markov\n(K=3)",
        "hosvd_markov_linear_svc": "6. HOSVD\n+Markov",
        "rks_markov_linear_svc": "7. RKS\n+Markov",
    }
    labels = [short_names.get(r["model_family"], r["model_family"]) for r in locked_results]
    x = np.arange(len(labels))

    dev_acc = [float(r["dev_cv_mean_accuracy"]) * 100.0 for r in locked_results]
    test_acc = [float(r["test_accuracy"]) * 100.0 for r in locked_results]
    test_f1 = [float(r["test_f1_macro"]) * 100.0 for r in locked_results]
    n_feats = [int(r["n_features"]) for r in locked_results]
    cr_raw = [float(r["compression_ratio_vs_raw"]) for r in locked_results]
    svm_train_ms = [float(r["svm_train_time_sec"]) * 1000.0 for r in locked_results]
    pred_per_trial_ms = [float(r["prediction_time_per_trial_ms"]) for r in locked_results]
    storage_kb = [float(r["model_storage_bytes"]) / 1024.0 for r in locked_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # Panel A: Dev CV Accuracy vs Held-Out Run 12 Test Accuracy & Macro F1
    w = 0.27
    axes[0, 0].bar(x - w, dev_acc, width=w, color="#6baed6", label="Dev CV Mean Acc (Runs 4↔8)")
    axes[0, 0].bar(x, test_acc, width=w, color="#08519c", label="Locked Test Acc (Run 12)")
    axes[0, 0].bar(x + w, test_f1, width=w, color="#fd8d3c", label="Locked Test Macro F1 (×100)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, fontsize=9)
    axes[0, 0].set_ylabel("Score (%)")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_title("A. Classification Accuracy & Macro F1 (Dev CV vs. Held-Out Run 12)")
    axes[0, 0].grid(True, axis="y", alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8.5)

    # Panel B: Feature Count & Compression Ratio (Log Scale)
    ax_b2 = axes[0, 1].twinx()
    b1 = axes[0, 1].bar(x - 0.18, n_feats, width=0.35, color="#2ca25f", label="Number of Features (p)")
    b2 = ax_b2.bar(x + 0.18, cr_raw, width=0.35, color="#8856a7", label="Compression Ratio vs. Raw (×)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, fontsize=9)
    axes[0, 1].set_ylabel("Number of Features ($p$)", color="#2ca25f", fontweight="bold")
    ax_b2.set_ylabel("Compression Ratio vs. Raw ($20,480 / p$, log scale)", color="#8856a7", fontweight="bold")
    ax_b2.set_yscale("log")
    axes[0, 1].set_title("B. Feature Dimension ($p$) & Compression Ratio vs. Raw EEG")
    axes[0, 1].grid(True, axis="y", alpha=0.3)
    lines_b = [b1, b2]
    axes[0, 1].legend(lines_b, ["Number of Features (p)", "Compression Ratio vs. Raw (×)"], loc="upper left", fontsize=8.5)

    # Panel C: Runtime Breakdown (SVM Training Time vs Per-Trial Prediction Time)
    axes[1, 0].bar(x - 0.18, svm_train_ms, width=0.35, color="#e6550d", label="SVM Fit Time on Runs 4+8 (ms)")
    axes[1, 0].bar(x + 0.18, pred_per_trial_ms, width=0.35, color="#31a354", label="End-to-End Prediction Time / Trial (ms)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, fontsize=9)
    axes[1, 0].set_ylabel("Wall-Clock Time (milliseconds)")
    axes[1, 0].set_title("C. Compute Cost: SVM Training Time vs. Per-Trial Prediction Time")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    axes[1, 0].legend(loc="upper right", fontsize=8.5)

    # Panel D: Total Deployed Model Memory / Storage Requirement (KB)
    axes[1, 1].bar(x, storage_kb, width=0.45, color="#756bb1")
    axes[1, 1].set_yscale("log")
    for i, val in enumerate(storage_kb):
        axes[1, 1].text(i, val * 1.15, f"{val:.1f} KB", ha="center", fontsize=8.5, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, fontsize=9)
    axes[1, 1].set_ylabel("Deployed Model Storage (KB, log scale)")
    axes[1, 1].set_ylim(1.0, max(storage_kb) * 3.0)
    axes[1, 1].set_title("D. Memory / Storage Requirement (Bases / Random Weights + Scaler + SVM)")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Milestone M6 — Master Comparison Across All 8 Faculty Evaluation Measures (Subject 1)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_m6_confusion_matrices(
    locked_results: list[dict],
    output_path: str | Path,
) -> Path:
    """Plot 2x2 held-out Run 12 confusion matrices for all locked model families."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(locked_results)
    cols = 4
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows))
    axes_flat = np.asarray(axes).reshape(-1)

    for idx, res in enumerate(locked_results):
        ax = axes_flat[idx]
        cm = np.asarray(res["test_confusion_matrix"], dtype=np.int64)
        cell_max = max(int(np.max(cm)), 8)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cell_max)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Left (0)", "Pred Right (1)"], fontsize=8.5)
        ax.set_yticklabels(["True Left (0)", "True Right (1)"], fontsize=8.5)
        ax.set_title(
            f"{res['locked_config_id']}\nTest Acc: {float(res['test_accuracy'])*100:.1f}% | F1: {float(res['test_f1_macro']):.2f}",
            fontsize=8.5,
            fontweight="bold",
        )
        for i in range(2):
            for j in range(2):
                val = int(cm[i, j])
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color="white" if val >= 0.55 * cell_max else "black",
                    fontsize=11,
                    fontweight="bold",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")

    if locked_results:
        n_left = int(locked_results[0].get("test_support_class_0_left", 0))
        n_right = int(locked_results[0].get("test_support_class_1_right", 0))
        n_tot = n_left + n_right
        subtitle = f"N={n_tot} Trials: {n_left} Left T1, {n_right} Right T2"
    else:
        subtitle = "Held-Out Run 12"

    fig.suptitle(
        f"Milestone M6 — Held-Out Test Run 12 Confusion Matrices ({subtitle})",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out

