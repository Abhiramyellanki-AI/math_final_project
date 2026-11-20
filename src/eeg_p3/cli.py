"""Command-line interface entry point for the P3 EEG Motor-Imagery project (M0-M6)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

from eeg_p3 import __version__
from eeg_p3.config import load_config


def get_dependency_versions() -> dict[str, str | None]:
    """Return installed versions of core scientific and testing libraries."""
    packages = ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "yaml", "mne", "pytest")
    versions: dict[str, str | None] = {}
    for pkg in packages:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            versions[pkg] = None
        else:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "installed")
    return versions


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="eeg-p3",
        description=(
            "P3 EEG Motor-Imagery Research Pipeline: HOSVD + SVM vs RKS + Linear SVM "
            "with Intra-Trial Markov Chains."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # M0 command: info
    info_parser = subparsers.add_parser(
        "info",
        help="Display validated project configuration, split guardrails, and package versions (M0).",
    )
    info_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/subject01.yaml"),
        help="Path to experiment configuration YAML file.",
    )

    # M1 command: inspect-data
    inspect_parser = subparsers.add_parser(
        "inspect-data",
        help="Ingest Subject 1 Runs 4/8/12, filter 8-30 Hz, extract [0.5, 2.5) s epochs, and print summary (M1).",
    )
    inspect_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/subject01.yaml"),
        help="Path to experiment configuration YAML file.",
    )
    inspect_parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/metrics/m1_data_summary.json"),
        help="Path to write JSON data inspection summary.",
    )

    # M1 command: plot-eeg
    plot_parser = subparsers.add_parser(
        "plot-eeg",
        help="Generate raw vs. filtered EEG trace and labeled trial demonstration figures (M1).",
    )
    plot_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/subject01.yaml"),
        help="Path to experiment configuration YAML file.",
    )
    plot_parser.add_argument(
        "--run-id",
        type=int,
        default=4,
        help="Run ID to visualize for faculty demonstration (default: 4).",
    )
    plot_parser.add_argument(
        "--channel",
        type=str,
        default="C3",
        help="Channel name for raw vs. filtered trace comparison (default: C3).",
    )

    # M2 command: run-baselines
    baselines_parser = subparsers.add_parser(
        "run-baselines",
        help="Evaluate deterministic mu/beta log-bandpower + LinearSVC and RBF SVC baselines on Runs 4/8 (M2).",
    )
    baselines_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/subject01.yaml"),
        help="Path to experiment configuration YAML file.",
    )
    baselines_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory to export M2 CSV and JSON validation metrics.",
    )

    # M3 command: run-hosvd
    hosvd_parser = subparsers.add_parser(
        "run-hosvd",
        help="Run train-only HOSVD subspace projection, rank sweep, reconstruction error, and LinearSVC CV (M3).",
    )
    hosvd_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/subject01.yaml"),
        help="Path to experiment configuration YAML file.",
    )
    hosvd_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory to export M3 CSV and JSON validation metrics.",
    )

    # M4 command: run-rks
    rks_parser = subparsers.add_parser(
        "run-rks",
        help="Run Random Kitchen Sinks (StandardScaler -> RBFSampler -> LinearSVC) sweep on Runs 4/8 (M4).",
    )
    rks_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/subject01.yaml"),
        help="Path to experiment configuration YAML file.",
    )
    rks_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory to export M4 CSV and JSON validation metrics.",
    )

    # M4b command: run-markov
    markov_parser = subparsers.add_parser(
        "run-markov",
        help="Run Intra-Trial Observable Markov Chain standalone and HOSVD/RKS hybrid evaluations on Runs 4/8 (M4b).",
    )
    markov_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/subject01.yaml"),
        help="Path to experiment configuration YAML file.",
    )
    markov_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory to export M4b CSV and JSON validation metrics.",
    )

    # M5 command: evaluate-locked
    locked_parser = subparsers.add_parser(
        "evaluate-locked",
        help="Lock best configs from Runs 4/8 CV, refit on Runs 4+8, evaluate once on held-out Run 12 (M5/E3).",
    )
    locked_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/subject01.yaml"),
        help="Path to experiment configuration YAML file.",
    )
    locked_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to export M5/E3 locked test CSV and JSON metrics.",
    )

    # M6 command: build-report
    report_parser = subparsers.add_parser(
        "build-report",
        help="Generate publication-style figures and research report manifest from M5/E3 outputs (M6).",
    )
    report_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/subject01.yaml"),
        help="Path to experiment configuration YAML file.",
    )
    report_parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Directory containing M4/M5 or E2/E3 JSON and CSV outputs.",
    )
    report_parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory to save generated M6 publication figures.",
    )

    # E1 command: audit-cohort
    audit_parser = subparsers.add_parser(
        "audit-cohort",
        help="Audit local dataset eligibility across candidate subjects and export split manifest (E1).",
    )
    audit_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cohort_expanded.yaml"),
        help="Path to cohort configuration YAML file (default: configs/cohort_expanded.yaml).",
    )
    audit_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save cohort_manifest.json and cohort_eligibility.csv.",
    )

    # E2 command: validate-cohort
    val_cohort_parser = subparsers.add_parser(
        "validate-cohort",
        help="Run multi-participant development validation on Runs 4 and 8 across all model families (E2).",
    )
    val_cohort_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cohort_expanded.yaml"),
        help="Path to cohort configuration YAML file (default: configs/cohort_expanded.yaml).",
    )
    val_cohort_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save E2 development validation CSV and JSON reports.",
    )

    return parser


def run_info(config_path: Path) -> int:
    """Execute the M0 'info' subcommand without loading or reading any EEG data."""
    cfg = load_config(config_path)
    summary = {
        "milestone": "M0 (Repository Scaffold)",
        "package_version": __version__,
        "python_version": platform.python_version(),
        "project_name": cfg.project_name,
        "random_seed": cfg.random_seed,
        "subject_id": cfg.data.subject_id,
        "is_multi_subject": cfg.data.is_multi_subject,
        "candidate_subject_ids_count": len(cfg.data.candidate_subject_ids),
        "excluded_subject_ids": list(cfg.data.excluded_subject_ids),
        "development_runs": list(cfg.data.development_runs),
        "held_out_test_run": cfg.data.held_out_test_run,
        "local_data_dir": str(cfg.data.local_data_dir),
        "bandpass_hz": [cfg.preprocessing.bandpass_low_hz, cfg.preprocessing.bandpass_high_hz],
        "epoch_window_sec": [cfg.preprocessing.epoch_tmin_sec, cfg.preprocessing.epoch_tmax_sec],
        "expected_shape_per_trial": [
            cfg.data.expected_channels,
            cfg.preprocessing.expected_n_times,
        ],
        "label_mapping": cfg.data.label_mapping,
        "excluded_annotations": list(cfg.data.excluded_annotations),
        "dependencies": get_dependency_versions(),
    }
    print(json.dumps(summary, indent=2))
    return 0


def run_inspect_data(config_path: Path, output_json: Path) -> int:
    """Execute the M1 'inspect-data' subcommand on real Subject 1 EEG recordings."""
    from eeg_p3.data import prepare_subject_dataset, summarize_prepared_dataset

    cfg = load_config(config_path)
    dataset = prepare_subject_dataset(cfg)
    summary = summarize_prepared_dataset(dataset)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


def run_plot_eeg(config_path: Path, run_id: int, channel: str) -> int:
    """Execute the M1 'plot-eeg' subcommand to generate waveform and trial demonstration figures."""
    from eeg_p3.data import load_single_raw_run, process_single_run, resolve_run_edf_path
    from eeg_p3.plotting import plot_labeled_motor_imagery_trials, plot_raw_vs_filtered_trace

    cfg = load_config(config_path)
    edf_path = resolve_run_edf_path(
        subject_id=cfg.data.subject_id,
        run_id=run_id,
        local_data_dir=cfg.data.local_data_dir,
        allow_download_fallback=cfg.data.allow_download_fallback,
    )
    raw = load_single_raw_run(
        edf_path=edf_path,
        expected_channels=cfg.data.expected_channels,
        expected_sfreq=cfg.data.expected_sfreq,
    )
    run_result = process_single_run(
        raw=raw,
        subject_id=cfg.data.subject_id,
        run_id=run_id,
        cfg=cfg,
    )

    figures_dir = Path(cfg.raw.get("outputs", {}).get("figures_dir", "outputs/figures"))
    trace_path = plot_raw_vs_filtered_trace(
        raw=raw,
        output_path=figures_dir / f"m1_sub{cfg.data.subject_id:02d}_run{run_id:02d}_raw_vs_filtered_{channel}.png",
        channel_name=channel,
        duration_sec=12.0,
        low_hz=cfg.preprocessing.bandpass_low_hz,
        high_hz=cfg.preprocessing.bandpass_high_hz,
    )
    trials_path = plot_labeled_motor_imagery_trials(
        run_result=run_result,
        output_path=figures_dir / f"m1_sub{cfg.data.subject_id:02d}_run{run_id:02d}_labeled_trials.png",
    )

    result_payload = {
        "milestone": "M1 (EEG Waveform & Trial Demonstration Plots)",
        "subject_id": cfg.data.subject_id,
        "run_id": run_id,
        "edf_path": str(edf_path),
        "raw_vs_filtered_plot": str(trace_path),
        "labeled_trials_plot": str(trials_path),
    }
    print(json.dumps(result_payload, indent=2))
    return 0


def run_baselines(config_path: Path, output_dir: Path) -> int:
    """Execute the M2 'run-baselines' subcommand using run-swapped validation on Runs 4 and 8."""
    from eeg_p3.data import prepare_subject_dataset
    from eeg_p3.evaluation import run_m2_baselines_validation

    cfg = load_config(config_path)
    dataset = prepare_subject_dataset(cfg)
    fold_df, summary_df, report_dict = run_m2_baselines_validation(dataset, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    folds_csv = output_dir / "m2_baselines_folds.csv"
    summary_csv = output_dir / "m2_baselines_summary.csv"
    report_json = output_dir / "m2_baselines_report.json"

    fold_df.to_csv(folds_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report_dict, handle, indent=2)

    output_payload = {
        "milestone": report_dict["milestone"],
        "subject_id": report_dict["subject_id"],
        "development_runs_used": report_dict["development_runs_used"],
        "held_out_test_run_untouched": report_dict["held_out_test_run_untouched"],
        "feature_representation": report_dict["feature_representation"],
        "best_linear_svc_on_dev": report_dict["best_linear_svc_on_dev"],
        "best_rbf_svc_on_dev": report_dict["best_rbf_svc_on_dev"],
        "exported_files": {
            "folds_csv": str(folds_csv),
            "summary_csv": str(summary_csv),
            "report_json": str(report_json),
        },
        "summary_table": report_dict["summary_table"],
    }
    print(json.dumps(output_payload, indent=2))
    return 0


def run_hosvd(config_path: Path, output_dir: Path) -> int:
    """Execute the M3 'run-hosvd' subcommand using run-swapped validation on Runs 4 and 8."""
    from eeg_p3.data import prepare_subject_dataset
    from eeg_p3.evaluation import run_m3_hosvd_validation

    cfg = load_config(config_path)
    dataset = prepare_subject_dataset(cfg)
    fold_df, summary_df, rank_df, report_dict = run_m3_hosvd_validation(dataset, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    folds_csv = output_dir / "m3_hosvd_folds.csv"
    summary_csv = output_dir / "m3_hosvd_summary.csv"
    rank_csv = output_dir / "m3_hosvd_rank_sweep.csv"
    report_json = output_dir / "m3_hosvd_report.json"

    fold_df.to_csv(folds_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    rank_df.to_csv(rank_csv, index=False)
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report_dict, handle, indent=2)

    output_payload = {
        "milestone": report_dict["milestone"],
        "subject_id": report_dict["subject_id"],
        "development_runs_used": report_dict["development_runs_used"],
        "held_out_test_run_untouched": report_dict["held_out_test_run_untouched"],
        "best_hosvd_on_dev": report_dict["best_hosvd_on_dev"],
        "rank_sweep_summary": report_dict["rank_sweep_summary"],
        "exported_files": {
            "folds_csv": str(folds_csv),
            "summary_csv": str(summary_csv),
            "rank_sweep_csv": str(rank_csv),
            "report_json": str(report_json),
        },
    }
    print(json.dumps(output_payload, indent=2))
    return 0


def run_rks(config_path: Path, output_dir: Path) -> int:
    """Execute the M4 'run-rks' subcommand using run-swapped validation on Runs 4 and 8."""
    from eeg_p3.data import prepare_subject_dataset
    from eeg_p3.evaluation import run_m4_rks_validation

    cfg = load_config(config_path)
    dataset = prepare_subject_dataset(cfg)
    fold_df, summary_df, report_dict = run_m4_rks_validation(dataset, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    folds_csv = output_dir / "m4_rks_folds.csv"
    summary_csv = output_dir / "m4_rks_summary.csv"
    report_json = output_dir / "m4_rks_report.json"

    fold_df.to_csv(folds_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report_dict, handle, indent=2)

    output_payload = {
        "milestone": report_dict["milestone"],
        "subject_id": report_dict["subject_id"],
        "development_runs_used": report_dict["development_runs_used"],
        "held_out_test_run_untouched": report_dict["held_out_test_run_untouched"],
        "rks_note": report_dict["rks_note"],
        "best_rks_on_dev": report_dict["best_rks_on_dev"],
        "exported_files": {
            "folds_csv": str(folds_csv),
            "summary_csv": str(summary_csv),
            "report_json": str(report_json),
        },
        "summary_table": report_dict["summary_table"],
    }
    print(json.dumps(output_payload, indent=2))
    return 0


def run_markov(config_path: Path, output_dir: Path) -> int:
    """Execute the M4b 'run-markov' subcommand using run-swapped validation on Runs 4 and 8."""
    from eeg_p3.data import prepare_subject_dataset
    from eeg_p3.evaluation import run_m4b_markov_validation

    cfg = load_config(config_path)
    dataset = prepare_subject_dataset(cfg)
    fold_df, summary_df, report_dict = run_m4b_markov_validation(dataset, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    folds_csv = output_dir / "m4b_markov_folds.csv"
    summary_csv = output_dir / "m4b_markov_summary.csv"
    report_json = output_dir / "m4b_markov_report.json"

    fold_df.to_csv(folds_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report_dict, handle, indent=2)

    output_payload = {
        "milestone": report_dict["milestone"],
        "subject_id": report_dict["subject_id"],
        "development_runs_used": report_dict["development_runs_used"],
        "held_out_test_run_untouched": report_dict["held_out_test_run_untouched"],
        "markov_design": report_dict["markov_design"],
        "best_by_family_on_dev": report_dict["best_by_family_on_dev"],
        "exported_files": {
            "folds_csv": str(folds_csv),
            "summary_csv": str(summary_csv),
            "report_json": str(report_json),
        },
        "summary_table": report_dict["summary_table"],
    }
    print(json.dumps(output_payload, indent=2))
    return 0


def run_evaluate_locked(config_path: Path, output_dir: Path | None) -> int:
    """Execute the M5/E3 'evaluate-locked' subcommand: lock on Runs 4/8, refit on 4+8, test Run 12 once."""
    import pandas as pd
    from eeg_p3.data import prepare_subject_dataset
    from eeg_p3.evaluation import run_m5_locked_evaluation

    cfg = load_config(config_path)
    resolved_out_dir = (
        output_dir
        if output_dir is not None
        else Path(cfg.raw.get("outputs", {}).get("metrics_dir", "outputs/metrics"))
    )
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    precomputed_e2: dict | None = None
    e2_json_path = resolved_out_dir / "e2_cohort_dev_report.json"
    if cfg.data.is_multi_subject and e2_json_path.is_file():
        with e2_json_path.open("r", encoding="utf-8") as handle:
            precomputed_e2 = json.load(handle)

    dataset = prepare_subject_dataset(cfg)
    dep_versions = {
        "python": platform.python_version(),
        "eeg_p3": __version__,
        **get_dependency_versions(),
    }
    locked_df, rank_curve_df, report_dict = run_m5_locked_evaluation(
        dataset=dataset,
        cfg=cfg,
        dependency_versions=dep_versions,
        precomputed_e2_report=precomputed_e2,
    )

    locked_csv = resolved_out_dir / "m5_locked_test_results.csv"
    rank_csv = resolved_out_dir / "m5_hosvd_test_rank_curve.csv"
    participant_test_csv = resolved_out_dir / "m5_per_participant_test_results.csv"
    report_json = resolved_out_dir / "m5_locked_test_report.json"

    locked_df.to_csv(locked_csv, index=False)
    rank_curve_df.to_csv(rank_csv, index=False)
    if report_dict.get("per_participant_test_results"):
        pd.DataFrame(report_dict["per_participant_test_results"]).to_csv(
            participant_test_csv, index=False
        )
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report_dict, handle, indent=2)

    output_payload = {
        "milestone": report_dict["milestone"],
        "experiment_mode": report_dict.get("experiment_mode", "single_subject_pilot"),
        "subject_id": report_dict["subject_id"],
        "n_eligible_participants": report_dict.get("n_eligible_participants", 1),
        "random_seed": report_dict["random_seed"],
        "development_runs_refit": report_dict["development_runs_refit"],
        "held_out_test_run_evaluated_once": report_dict["held_out_test_run_evaluated_once"],
        "class_balance": report_dict["class_balance"],
        "rks_predetermined_seed_variability": report_dict.get("rks_predetermined_seed_variability"),
        "software_versions": report_dict["software_versions"],
        "methodological_disclosures_and_limitations": report_dict[
            "methodological_disclosures_and_limitations"
        ],
        "exported_files": {
            "locked_test_csv": str(locked_csv),
            "per_participant_test_csv": str(participant_test_csv),
            "hosvd_test_rank_curve_csv": str(rank_csv),
            "locked_test_report_json": str(report_json),
        },
        "locked_test_results": report_dict["locked_test_results"],
    }
    print(json.dumps(output_payload, indent=2))
    return 0


def run_build_report(
    config_path: Path,
    metrics_dir: Path | None,
    figures_dir: Path | None,
) -> int:
    """Execute the M6 'build-report' subcommand using actual M4/M5 or E2/E3 measured outputs."""
    import pandas as pd
    from eeg_p3.plotting import (
        plot_m6_confusion_matrices,
        plot_m6_hosvd_rank_curves,
        plot_m6_locked_comparison_dashboard,
        plot_m6_markov_transition_heatmaps,
        plot_m6_rks_sweep,
    )

    cfg = load_config(config_path)
    resolved_metrics_dir = (
        metrics_dir
        if metrics_dir is not None
        else Path(cfg.raw.get("outputs", {}).get("metrics_dir", "outputs/metrics"))
    )
    resolved_figures_dir = (
        figures_dir
        if figures_dir is not None
        else Path(cfg.raw.get("outputs", {}).get("figures_dir", "outputs/figures"))
    )

    m5_json_path = resolved_metrics_dir / "m5_locked_test_report.json"
    m4_json_path = resolved_metrics_dir / "m4_rks_report.json"
    e2_summary_csv = resolved_metrics_dir / "e2_cohort_dev_summary.csv"

    if not m5_json_path.is_file():
        raise FileNotFoundError(
            f"Final test report '{m5_json_path}' not found. "
            "Run 'evaluate-locked' explicitly first; 'build-report' never silently triggers final-test evaluation."
        )

    with m5_json_path.open("r", encoding="utf-8") as handle:
        m5_report = json.load(handle)

    rks_summary_records: list[dict]
    source_rks_path: str
    if m4_json_path.is_file():
        with m4_json_path.open("r", encoding="utf-8") as handle:
            m4_report = json.load(handle)
        rks_summary_records = m4_report["summary_table"]
        source_rks_path = str(m4_json_path)
    elif e2_summary_csv.is_file():
        e2_df = pd.read_csv(e2_summary_csv)
        rks_df = e2_df[e2_df["model_family"] == "rks_linear_svc"]
        rks_summary_records = rks_df.to_dict(orient="records")
        source_rks_path = str(e2_summary_csv)
    else:
        raise FileNotFoundError(
            f"Neither '{m4_json_path}' nor '{e2_summary_csv}' found. "
            "Run 'run-rks' or 'validate-cohort' first."
        )

    resolved_figures_dir.mkdir(parents=True, exist_ok=True)

    fig_hosvd = plot_m6_hosvd_rank_curves(
        rank_curve_records=m5_report["hosvd_test_rank_curve"],
        output_path=resolved_figures_dir / "m6_hosvd_rank_error_and_compression.png",
    )
    fig_rks = plot_m6_rks_sweep(
        rks_summary_records=rks_summary_records,
        output_path=resolved_figures_dir / "m6_rks_sweep_quality_and_storage.png",
    )
    fig_markov = plot_m6_markov_transition_heatmaps(
        markov_model_dict=m5_report["learned_markov_models_on_runs_4_8"]["markov_linear_svc"],
        output_path=resolved_figures_dir / "m6_markov_transition_heatmaps.png",
    )
    fig_dashboard = plot_m6_locked_comparison_dashboard(
        locked_results=m5_report["locked_test_results"],
        output_path=resolved_figures_dir / "m6_locked_comparison_dashboard.png",
    )
    fig_cm = plot_m6_confusion_matrices(
        locked_results=m5_report["locked_test_results"],
        output_path=resolved_figures_dir / "m6_confusion_matrices_run12.png",
    )

    manifest = {
        "milestone": "M6 (Publication-Style Figures, Methodology Diagram & Viva Report)",
        "source_m5_report": str(m5_json_path),
        "source_rks_report": source_rks_path,
        "generated_figures": {
            "hosvd_rank_error_and_compression": str(fig_hosvd),
            "rks_sweep_quality_and_storage": str(fig_rks),
            "markov_transition_heatmaps": str(fig_markov),
            "locked_comparison_dashboard": str(fig_dashboard),
            "confusion_matrices_run12": str(fig_cm),
        },
        "viva_report_markdown": "REPORT_AND_VIVA_GUIDE.md",
        "educational_notebook": "notebooks/01_explore_eeg_and_results.ipynb",
    }

    manifest_path = resolved_metrics_dir / "m6_report_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        return run_info(args.config)
    if args.command == "inspect-data":
        return run_inspect_data(args.config, args.output_json)
    if args.command == "plot-eeg":
        return run_plot_eeg(args.config, args.run_id, args.channel)
    if args.command == "run-baselines":
        return run_baselines(args.config, args.output_dir)
    if args.command == "run-hosvd":
        return run_hosvd(args.config, args.output_dir)
    if args.command == "run-rks":
        return run_rks(args.config, args.output_dir)
    if args.command == "run-markov":
        return run_markov(args.config, args.output_dir)
    if args.command == "evaluate-locked":
        return run_evaluate_locked(args.config, args.output_dir)
    if args.command == "build-report":
        return run_build_report(args.config, args.metrics_dir, args.figures_dir)
    if args.command == "audit-cohort":
        return run_audit_cohort(args.config, args.output_dir)
    if args.command == "validate-cohort":
        return run_validate_cohort(args.config, args.output_dir)

    print(f"Unknown command: '{args.command}'", file=sys.stderr)
    return 2


def run_validate_cohort(config_path: Path, output_dir: Path | None) -> int:
    """Execute the E2 'validate-cohort' subcommand on Runs 4 and 8 across all eligible subjects."""
    from eeg_p3.data import prepare_subject_dataset
    from eeg_p3.evaluation import run_e2_cohort_development_validation

    cfg = load_config(config_path)
    resolved_out_dir = (
        output_dir
        if output_dir is not None
        else Path(cfg.raw.get("outputs", {}).get("metrics_dir", "outputs/cohort_expanded/metrics"))
    )
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_subject_dataset(cfg)
    fold_df, summary_df, best_df, per_participant_df, report_dict = (
        run_e2_cohort_development_validation(dataset=dataset, cfg=cfg)
    )

    folds_csv = resolved_out_dir / "e2_cohort_dev_folds.csv"
    summary_csv = resolved_out_dir / "e2_cohort_dev_summary.csv"
    best_csv = resolved_out_dir / "e2_cohort_best_by_family_dev.csv"
    participant_csv = resolved_out_dir / "e2_cohort_per_participant_dev.csv"
    report_json = resolved_out_dir / "e2_cohort_dev_report.json"

    fold_df.to_csv(folds_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    best_df.to_csv(best_csv, index=False)
    per_participant_df.to_csv(participant_csv, index=False)
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report_dict, handle, indent=2)

    output_payload = {
        "milestone": report_dict["milestone"],
        "experiment_mode": report_dict["experiment_mode"],
        "n_eligible_participants": report_dict["n_eligible_participants"],
        "n_development_recordings": report_dict["n_development_recordings"],
        "n_development_trials": report_dict["n_development_trials"],
        "development_class_counts": report_dict["development_class_counts"],
        "held_out_test_run_untouched": report_dict["held_out_test_run_untouched"],
        "selection_protocol": report_dict["selection_protocol"],
        "uncertainty_accounting_note": report_dict["uncertainty_accounting_note"],
        "rks_predetermined_seed_variability_on_dev": report_dict[
            "rks_predetermined_seed_variability_on_dev"
        ],
        "locked_best_by_family_on_dev": report_dict["locked_best_by_family_on_dev"],
        "hosvd_rank_sweep_summary_on_dev": report_dict["hosvd_rank_sweep_summary_on_dev"],
        "exported_files": {
            "folds_csv": str(folds_csv),
            "summary_csv": str(summary_csv),
            "best_by_family_csv": str(best_csv),
            "per_participant_csv": str(participant_csv),
            "report_json": str(report_json),
        },
    }
    print(json.dumps(output_payload, indent=2))
    return 0


def run_audit_cohort(config_path: Path, output_dir: Path | None) -> int:
    """Execute the E1 'audit-cohort' subcommand without training models or evaluating Run 12."""
    import pandas as pd
    from eeg_p3.data import audit_cohort_eligibility, build_cohort_manifest_report

    cfg = load_config(config_path)
    resolved_out_dir = (
        output_dir
        if output_dir is not None
        else Path(cfg.raw.get("outputs", {}).get("metrics_dir", "outputs/cohort_expanded/metrics"))
    )
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    records = audit_cohort_eligibility(cfg)
    manifest = build_cohort_manifest_report(cfg, records)

    manifest_json_path = resolved_out_dir / "cohort_manifest.json"
    eligibility_csv_path = resolved_out_dir / "cohort_eligibility.csv"

    with manifest_json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    rows = []
    for r in records:
        r04 = r.per_run_class_counts.get(4, {})
        r08 = r.per_run_class_counts.get(8, {})
        r12 = r.per_run_class_counts.get(12, {})
        rows.append(
            {
                "subject_id": r.subject_id,
                "eligible": r.eligible,
                "status": r.status,
                "reason_code": r.reason_code,
                "reason_detail": r.reason_detail,
                "run04_T1_left": r04.get("T1_left_0", 0),
                "run04_T2_right": r04.get("T2_right_1", 0),
                "run08_T1_left": r08.get("T1_left_0", 0),
                "run08_T2_right": r08.get("T2_right_1", 0),
                "run12_T1_left": r12.get("T1_left_0", 0),
                "run12_T2_right": r12.get("T2_right_1", 0),
            }
        )
    pd.DataFrame(rows).to_csv(eligibility_csv_path, index=False)

    summary_output = {
        "milestone": manifest["milestone"],
        "experiment_mode": manifest["experiment_mode"],
        "local_data_dir": manifest["local_data_dir"],
        "local_data_dir_accessible": manifest["local_data_dir_accessible"],
        "local_data_dir_access_note": manifest["local_data_dir_access_note"],
        "cohort_audit_summary": manifest["cohort_audit_summary"],
        "split_manifest_summary": {
            "development_total_trials": manifest["split_manifest"]["development_total_runs_4_8"]["n_trials"],
            "development_total_class_counts": manifest["split_manifest"]["development_total_runs_4_8"]["class_counts"],
            "fold_1_train_run4_trials": manifest["split_manifest"]["development_fold_1_train_run4_val_run8"]["train"]["n_trials"],
            "fold_1_val_run8_trials": manifest["split_manifest"]["development_fold_1_train_run4_val_run8"]["val"]["n_trials"],
            "held_out_test_run12_trials": manifest["split_manifest"]["held_out_test_run_12"]["n_trials"],
            "held_out_test_run12_class_counts": manifest["split_manifest"]["held_out_test_run_12"]["class_counts"],
            "zero_recording_overlap_verified": manifest["split_manifest"]["development_fold_1_train_run4_val_run8"]["zero_recording_overlap_verified"],
        },
        "exported_files": {
            "cohort_manifest_json": str(manifest_json_path),
            "cohort_eligibility_csv": str(eligibility_csv_path),
        },
    }
    print(json.dumps(summary_output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
