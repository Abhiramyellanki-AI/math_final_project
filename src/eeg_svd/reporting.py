"""Generate tables and figures from completed artifacts only; never fit a model."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import file_hash, read_json
from .models import FAMILIES


def build_report(experiment: Path, out: Path) -> None:
    """Render saved evidence, refusing incomplete or modified result files."""
    status = read_json(experiment / "status.json")
    if status["status"] != "complete":
        raise ValueError("Evaluation is not complete; report generation cannot evaluate models")
    for name, expected in status["sha256"].items():
        if file_hash(experiment / name) != expected:
            raise ValueError(f"Result file changed: {name}")
    result = read_json(experiment / "results.json")
    names = ["RBF without SVD", "SVD + RBF", "Training-majority"]
    families = [*FAMILIES, "majority_baseline"]
    summary = result["aggregate"]
    fig, ax = plt.subplots(figsize=(7, 4))
    means = [summary[f]["participant_metrics"]["accuracy"]["mean"] for f in families]
    intervals = [summary[f]["participant_metrics"]["accuracy"]["ci95"] for f in families]
    errors = [0 if ci is None else (ci[1] - ci[0]) / 2 for ci in intervals]
    ax.bar(names, np.array(means) * 100, yerr=np.array(errors) * 100, capsize=5)
    ax.set(ylabel="Participant mean accuracy (%)", ylim=(0, 100),
           title="Exploratory run 12 evaluation; 95% participant t intervals")
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, family, title in zip(axes, FAMILIES, names):
        cm = np.array(summary[family]["pooled_metrics"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Left", "Right"],
               yticklabels=["Left", "Right"], xlabel="Prediction", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for spectrum in result["spectra"]:
        s = np.array(spectrum["singular_values"])
        axes[0].plot(np.arange(1, len(s) + 1), np.cumsum(s ** 2) / np.sum(s ** 2), alpha=.25)
    axes[0].set(xlabel="Retained rank", ylabel="Training spectral energy fraction",
                title="Every participant; no best-subject selection")
    rows = [r for r in result["rows"] if r["family"] == "svd_rbf"]
    axes[1].scatter([r["train_reconstruction_error"] for r in rows],
                    [r["test_reconstruction_error"] for r in rows])
    axes[1].set(xlabel="Development reconstruction error", ylabel="Run 12 error",
                title="Standardized bandpower, not raw EEG")
    fig.tight_layout()
    fig.savefig(out / "svd_diagnostics.png", dpi=180)
    plt.close(fig)
    lines = ["# Measured exploratory evaluation", "", result["disclosure"], "",
             f"Participants: {result['n_participants']}", "",
             "| Method | Participant mean accuracy | Macro F1 | Total pipeline bytes |",
             "|---|---:|---:|---:|"]
    latex = []
    for family, name in zip(families, names):
        m = summary[family]["participant_metrics"]
        acc, f1 = m["accuracy"]["mean"], m["macro_f1"]["mean"]
        size = summary[family]["total_serialized_model_bytes"]
        lines.append(f"| {name} | {100*acc:.2f}% | {f1:.4f} | {size} |")
        latex.append(f"{name} & {100*acc:.2f} & {f1:.4f} \\\\")
    lines.extend(["", "Paired SVD minus baseline differences (participant unit):", ""])
    for key, value in summary["paired_svd_minus_baseline"].items():
        lines.append(f"- {key}: mean {value['mean']:.4f}; 95% CI {value['ci95']}")
    lines.extend(["", result["timing_scope"], "", result["storage_scope"], "",
                  "No statistical equivalence or 75% achievement is assumed. See participant_metrics.csv for all measures."])
    (out / "measured_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table = ("\\begin{table}[ht]\n\\centering\n"
             "\\caption{Measured exploratory follow-up; participant-averaged metrics.}\n"
             "\\begin{tabular}{lrr}\\hline Method & Accuracy (\\%) & Macro F1 \\\\ \\hline\n"
             + "\n".join(latex) + "\n\\hline\\end{tabular}\n\\end{table}\n")
    (out / "results.tex").write_text(table, encoding="utf-8")


def plot_eeg(cfg: dict, subject: int, out: Path) -> None:
    """Produce a genuine development-run EEG plot, without any model evaluation."""
    import mne
    from .data import epochs_from_raw, resolve_edf
    raw = mne.io.read_raw_edf(resolve_edf(Path(cfg["data_dir"]), subject, 4),
                              preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    raw.pick("eeg")
    filtered = raw.copy().filter(*cfg["bandpass"], method="fir", phase="zero",
                                 fir_design="firwin", verbose="ERROR")
    X, y, _ = epochs_from_raw(raw, cfg, subject, 4)
    channel = sorted(raw.ch_names).index("C3")
    fig, axes = plt.subplots(2, 1, figsize=(9, 5))
    n = min(int(20 * cfg["sfreq"]), raw.n_times)
    for signal, label in ((raw, "Raw"), (filtered, "8–30 Hz")):
        axes[0].plot(signal.times[:n], signal.get_data(picks=["C3"])[0, :n] * 1e6,
                     label=label, alpha=.7)
    axes[0].set(xlabel="Time (s)", ylabel="C3 (µV)", title=f"S{subject:03d} run 4")
    axes[0].legend()
    for label, name in ((0, "Left"), (1, "Right")):
        i = int(np.flatnonzero(y == label)[0])
        axes[1].plot(.5 + np.arange(320) / cfg["sfreq"], X[i, channel] * 1e6, label=name)
    axes[1].set(xlabel="Seconds after cue", ylabel="C3 (µV)",
                title="First eligible trial of each class; illustrative only")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "eeg_example.png", dpi=180)
    plt.close(fig)
