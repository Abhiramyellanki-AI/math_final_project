"""EEG data ingestion, annotation verification, and leakage-safe split preparation (M1).

Enforces PRD Section 4 and Section 5.1-5.3:
- Loads each EDF run independently using MNE-Python.
- Prioritizes pre-downloaded local EDF files (workspace data/ mirror or configured local_data_dir).
- Standardizes 64-channel EEGBCI channel naming and verifies 160 Hz sampling rate.
- Filters each run independently before extracting [0.5, 2.5) s post-cue epochs (320 samples).
- Separates development runs (4, 8) from untouched held-out test run (12).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np

from eeg_p3.config import ProjectConfig
from eeg_p3.preprocessing import (
    RunEpochsResult,
    TrialMetadata,
    count_run_annotations,
    extract_epochs_from_filtered_run,
    filter_raw_run_independently,
)


@dataclass(frozen=True)
class RunSplitBundle:
    """Bundled trial tensor, labels, run groups, and metadata for a split (dev or test)."""

    split_name: str
    run_ids: tuple[int, ...]
    X: np.ndarray  # (n_trials, n_channels, n_times) in Volts
    y: np.ndarray  # (n_trials,) in {0, 1}
    groups: np.ndarray  # (n_trials,) run IDs
    times_sec: np.ndarray  # (n_times,)
    channel_names: tuple[str, ...]
    sfreq: float
    signal_unit: str
    trials_metadata: tuple[TrialMetadata, ...]
    per_run_results: dict[int, RunEpochsResult]


@dataclass(frozen=True)
class PreparedEEGDataset:
    """Complete Subject dataset separated into development (Runs 4, 8) and test (Run 12)."""

    subject_id: int
    dev: RunSplitBundle
    test: RunSplitBundle
    resolved_edf_paths: dict[int, Path]


def _safe_path_is_file(candidate: Path) -> bool:
    """Return True if candidate exists and is a regular file, catching PermissionError."""
    try:
        return candidate.is_file()
    except OSError:
        return False


def resolve_run_edf_path(
    subject_id: int,
    run_id: int,
    local_data_dir: Path,
    workspace_data_dir: Path | None = None,
    allow_download_fallback: bool = False,
) -> Path:
    """Resolve the local EDF file path for a given subject and run.

    Search order:
    1. Workspace-local cache `data/S001/S001R04.edf` or `data/S001R04.edf` (if present).
    2. Configured `local_data_dir/S001/S001R04.edf` or `local_data_dir/S001R04.edf`.
    3. Optional MNE download fallback only if `allow_download_fallback=True`.

    Raises:
        FileNotFoundError: If the EDF file cannot be found locally and download is disabled.
    """
    subj_tag = f"S{subject_id:03d}"
    file_name = f"{subj_tag}R{run_id:02d}.edf"

    candidates: list[Path] = []
    if workspace_data_dir is not None:
        candidates.append(workspace_data_dir / subj_tag / file_name)
        candidates.append(workspace_data_dir / file_name)

    default_ws_data = Path("data")
    candidates.append(default_ws_data / subj_tag / file_name)
    candidates.append(default_ws_data / file_name)

    candidates.append(local_data_dir / subj_tag / file_name)
    candidates.append(local_data_dir / file_name)

    for candidate in candidates:
        if _safe_path_is_file(candidate):
            return candidate.resolve()

    if allow_download_fallback:
        downloaded = mne.datasets.eegbci.load_data(
            subjects=subject_id,
            runs=[run_id],
            update_path=False,
            verbose="ERROR",
        )
        return Path(downloaded[0]).resolve()

    searched_str = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate {file_name} for subject {subject_id}, run {run_id}. "
        f"Searched paths: [{searched_str}]."
    )


def load_single_raw_run(
    edf_path: str | Path,
    expected_channels: int = 64,
    expected_sfreq: float = 160.0,
) -> mne.io.BaseRaw:
    """Load a single EDF recording run with MNE, standardize channel names, and verify contract.

    Args:
        edf_path: Path to the `.edf` file.
        expected_channels: Expected count of EEG channels (64 for EEGBCI).
        expected_sfreq: Expected sampling rate in Hz (160.0 for EEGBCI).

    Returns:
        Preloaded MNE Raw object with standardized EEG channel names in Volts.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    raw.pick("eeg")

    n_channels = len(raw.ch_names)
    sfreq = float(raw.info["sfreq"])

    if n_channels != expected_channels:
        raise ValueError(
            f"Channel count mismatch in {edf_path}: got {n_channels}, expected {expected_channels}."
        )
    if not np.isclose(sfreq, expected_sfreq, atol=1e-3):
        raise ValueError(
            f"Sampling rate mismatch in {edf_path}: got {sfreq} Hz, expected {expected_sfreq} Hz."
        )

    return raw


def process_single_run(
    raw: mne.io.BaseRaw,
    subject_id: int,
    run_id: int,
    cfg: ProjectConfig,
) -> RunEpochsResult:
    """Independently filter a single raw run and extract its post-cue motor-imagery epochs."""
    raw_filtered = filter_raw_run_independently(
        raw=raw,
        low_hz=cfg.preprocessing.bandpass_low_hz,
        high_hz=cfg.preprocessing.bandpass_high_hz,
        filter_method=cfg.preprocessing.filter_method,
        fir_design=cfg.preprocessing.filter_design,
    )
    filter_params = {
        "bandpass_low_hz": cfg.preprocessing.bandpass_low_hz,
        "bandpass_high_hz": cfg.preprocessing.bandpass_high_hz,
        "method": cfg.preprocessing.filter_method,
        "fir_design": cfg.preprocessing.filter_design,
        "phase": "zero (offline non-causal)",
    }
    return extract_epochs_from_filtered_run(
        raw_filtered=raw_filtered,
        subject_id=subject_id,
        run_id=run_id,
        tmin_sec=cfg.preprocessing.epoch_tmin_sec,
        tmax_sec=cfg.preprocessing.epoch_tmax_sec,
        expected_n_times=cfg.preprocessing.expected_n_times,
        label_mapping=cfg.data.label_mapping,
        excluded_annotations=cfg.data.excluded_annotations,
        filter_params=filter_params,
    )


def combine_run_results(
    split_name: str,
    run_ids: tuple[int, ...],
    per_run_results: dict[int, RunEpochsResult],
) -> RunSplitBundle:
    """Combine already-filtered and epoched run results into a split bundle.

    Note: Individual runs are ALWAYS filtered and epoched independently prior to
    calling this function. No time-domain raw concatenation ever occurs.
    """
    if not run_ids:
        raise ValueError(f"Split '{split_name}' has no run_ids.")

    first = per_run_results[run_ids[0]]
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    groups_list: list[np.ndarray] = []
    metadata_all: list[TrialMetadata] = []

    for rid in run_ids:
        res = per_run_results[rid]
        if res.channel_names != first.channel_names:
            raise ValueError(f"Channel name mismatch across runs in split '{split_name}'.")
        if not np.isclose(res.sfreq, first.sfreq):
            raise ValueError(f"Sampling frequency mismatch across runs in split '{split_name}'.")
        X_list.append(res.X)
        y_list.append(res.y)
        groups_list.append(res.groups)
        metadata_all.extend(res.trials_metadata)

    return RunSplitBundle(
        split_name=split_name,
        run_ids=run_ids,
        X=np.concatenate(X_list, axis=0),
        y=np.concatenate(y_list, axis=0),
        groups=np.concatenate(groups_list, axis=0),
        times_sec=first.times_sec,
        channel_names=first.channel_names,
        sfreq=first.sfreq,
        signal_unit=first.signal_unit,
        trials_metadata=tuple(metadata_all),
        per_run_results={rid: per_run_results[rid] for rid in run_ids},
    )


def prepare_subject_dataset(
    cfg: ProjectConfig,
    workspace_data_dir: Path | None = None,
) -> PreparedEEGDataset:
    """Load, filter, epoch, and split Subject EEG runs into development (4, 8) and test (12)."""
    subject_id = cfg.data.subject_id
    dev_runs = cfg.data.development_runs
    test_run = cfg.data.held_out_test_run

    if test_run in dev_runs:
        raise ValueError(f"Leakage violation: test_run {test_run} in dev_runs {dev_runs}.")

    all_runs = (*dev_runs, test_run)
    resolved_paths: dict[int, Path] = {}
    per_run_results: dict[int, RunEpochsResult] = {}

    for run_id in all_runs:
        edf_path = resolve_run_edf_path(
            subject_id=subject_id,
            run_id=run_id,
            local_data_dir=cfg.data.local_data_dir,
            workspace_data_dir=workspace_data_dir,
            allow_download_fallback=cfg.data.allow_download_fallback,
        )
        resolved_paths[run_id] = edf_path
        raw = load_single_raw_run(
            edf_path=edf_path,
            expected_channels=cfg.data.expected_channels,
            expected_sfreq=cfg.data.expected_sfreq,
        )
        per_run_results[run_id] = process_single_run(
            raw=raw,
            subject_id=subject_id,
            run_id=run_id,
            cfg=cfg,
        )

    dev_bundle = combine_run_results("development", dev_runs, per_run_results)
    test_bundle = combine_run_results("held_out_test", (test_run,), per_run_results)

    # Non-negotiable leakage check
    if not set(dev_bundle.groups.tolist()).isdisjoint(set(test_bundle.groups.tolist())):
        raise AssertionError("Leakage detected between development and held-out test group IDs.")

    return PreparedEEGDataset(
        subject_id=subject_id,
        dev=dev_bundle,
        test=test_bundle,
        resolved_edf_paths=resolved_paths,
    )


def summarize_prepared_dataset(dataset: PreparedEEGDataset) -> dict[str, Any]:
    """Generate a JSON-serializable inspection summary of the prepared dataset."""
    per_run_summary: dict[str, Any] = {}
    all_run_results = {**dataset.dev.per_run_results, **dataset.test.per_run_results}

    for run_id in sorted(all_run_results.keys()):
        res = all_run_results[run_id]
        n_left = int(np.sum(res.y == 0))
        n_right = int(np.sum(res.y == 1))
        per_run_summary[f"run_{run_id:02d}"] = {
            "run_id": run_id,
            "role": "development" if run_id in dataset.dev.run_ids else "held_out_test",
            "edf_path": str(dataset.resolved_edf_paths[run_id]),
            "X_shape": list(res.X.shape),
            "y_shape": list(res.y.shape),
            "raw_annotation_counts": res.annotation_counts,
            "class_counts": {"T1_left_0": n_left, "T2_right_1": n_right, "total_trials": int(len(res.y))},
            "signal_unit": res.signal_unit,
            "amplitude_volts_min": float(np.min(res.X)),
            "amplitude_volts_max": float(np.max(res.X)),
            "amplitude_microvolts_std": float(np.std(res.X) * 1e6),
            "first_trial_metadata": asdict(res.trials_metadata[0]),
        }

    return {
        "milestone": "M1 (Real EEG Ingestion & Preprocessing)",
        "subject_id": dataset.subject_id,
        "sfreq_hz": dataset.dev.sfreq,
        "n_channels": len(dataset.dev.channel_names),
        "channel_names_sample": list(dataset.dev.channel_names[:10]),
        "all_channel_names": list(dataset.dev.channel_names),
        "epoch_window_sec": [
            float(dataset.dev.times_sec[0]),
            float(dataset.dev.times_sec[-1] + 1.0 / dataset.dev.sfreq),
        ],
        "n_times_per_epoch": int(len(dataset.dev.times_sec)),
        "signal_unit": dataset.dev.signal_unit,
        "filter_params": dataset.dev.per_run_results[dataset.dev.run_ids[0]].filter_params,
        "splits": {
            "development_runs_4_8": {
                "run_ids": list(dataset.dev.run_ids),
                "X_shape": list(dataset.dev.X.shape),
                "y_shape": list(dataset.dev.y.shape),
                "groups_unique": sorted(int(g) for g in np.unique(dataset.dev.groups)),
                "class_counts": {
                    "T1_left_0": int(np.sum(dataset.dev.y == 0)),
                    "T2_right_1": int(np.sum(dataset.dev.y == 1)),
                },
            },
            "held_out_test_run_12": {
                "run_ids": list(dataset.test.run_ids),
                "X_shape": list(dataset.test.X.shape),
                "y_shape": list(dataset.test.y.shape),
                "groups_unique": sorted(int(g) for g in np.unique(dataset.test.groups)),
                "class_counts": {
                    "T1_left_0": int(np.sum(dataset.test.y == 0)),
                    "T2_right_1": int(np.sum(dataset.test.y == 1)),
                },
            },
        },
        "per_run_details": per_run_summary,
    }
