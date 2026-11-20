"""EEG data ingestion, cohort eligibility auditing, and leakage-safe split preparation (M1 & E1).

Enforces PRD Section 4, Section 5.1-5.3, and Section 5.8 (Confirmatory Cohort Amendment):
- Loads each EDF run independently using MNE-Python without automatic downloading.
- Audits candidate participants (e.g., Subjects 2-109, excluding pilot Subject 1 by default)
  before model evaluation for: file existence, MNE readability, 64 EEG channels in canonical
  order, 160.0 Hz sampling rate, valid {T0, T1, T2} annotations with both T1 and T2 present,
  and valid [0.5, 2.5) s (320-sample) finite epochs.
- Logs every exclusion with an objective reason code and detail; never excludes by accuracy.
- Preserves subject_id, run_id, unique recording_id ('S002R04'), global_trial_id ('S002R04_T00'),
  and strict chronological order for every trial.
- Uses the exact same eligible participants in development (Runs 4, 8) and final test (Run 12).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np

from eeg_p3.config import ProjectConfig
from eeg_p3.preprocessing import (
    RunEpochsResult,
    TrialMetadata,
    extract_epochs_from_filtered_run,
    filter_raw_run_independently,
    format_recording_id,
)


# Canonical 64-channel standardized EEGBCI montage order
CANONICAL_EEGBCI_64_CHANNELS: tuple[str, ...] = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8",
    "T7", "T8", "T9", "T10",
    "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
    "Iz",
)


@dataclass(frozen=True)
class SubjectEligibilityRecord:
    """Audit outcome for a single candidate or excluded participant across required runs."""

    subject_id: int
    eligible: bool
    status: str  # 'eligible' or 'excluded'
    reason_code: str
    reason_detail: str
    checked_runs: tuple[int, ...]
    resolved_edf_paths: dict[int, str] = field(default_factory=dict)
    sfreq_by_run: dict[int, float] = field(default_factory=dict)
    n_channels_by_run: dict[int, int] = field(default_factory=dict)
    per_run_raw_annotations: dict[int, dict[str, int]] = field(default_factory=dict)
    per_run_class_counts: dict[int, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSplitBundle:
    """Bundled trial tensor, labels, run groups, subject/recording IDs, and metadata for a split."""

    split_name: str
    run_ids: tuple[int, ...]
    X: np.ndarray  # (n_trials, n_channels, n_times) in Volts
    y: np.ndarray  # (n_trials,) in {0, 1}
    groups: np.ndarray  # (n_trials,) run IDs (4, 8, or 12)
    times_sec: np.ndarray  # (n_times,)
    channel_names: tuple[str, ...]
    sfreq: float
    signal_unit: str
    trials_metadata: tuple[TrialMetadata, ...]
    per_run_results: dict[int, RunEpochsResult]  # Retained for single-subject backward compatibility
    subject_ids: np.ndarray | None = None  # (n_trials,) participant IDs
    recording_ids: tuple[str, ...] = ()  # (n_trials,) e.g. 'S002R04'
    trial_ids: tuple[str, ...] = ()  # (n_trials,) e.g. 'S002R04_T00'
    recording_results: dict[str, RunEpochsResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.subject_ids is None:
            sids = np.array([m.subject_id for m in self.trials_metadata], dtype=np.int64)
            object.__setattr__(self, "subject_ids", sids)
        if not self.recording_ids:
            rids = tuple(m.recording_id for m in self.trials_metadata)
            object.__setattr__(self, "recording_ids", rids)
        if not self.trial_ids:
            tids = tuple(m.global_trial_id for m in self.trials_metadata)
            object.__setattr__(self, "trial_ids", tids)


@dataclass(frozen=True)
class PreparedEEGDataset:
    """Complete dataset (single-subject or multi-subject cohort) separated into dev and test."""

    subject_id: int  # Primary subject ID (for single-subject backward compatibility)
    dev: RunSplitBundle
    test: RunSplitBundle
    resolved_edf_paths: dict[int, Path]  # Primary subject's paths (backward compatibility)
    subject_ids: tuple[int, ...] = ()
    resolved_recording_paths: dict[str, Path] = field(default_factory=dict)
    eligibility_records: tuple[SubjectEligibilityRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.subject_ids:
            object.__setattr__(self, "subject_ids", (int(self.subject_id),))


def _safe_path_is_file(candidate: Path) -> bool:
    """Return True if candidate exists and is a regular file, catching PermissionError."""
    try:
        return candidate.is_file()
    except OSError:
        return False


def _safe_dir_is_accessible(directory: Path) -> tuple[bool, str | None]:
    """Check whether a directory exists and can be listed without PermissionError."""
    try:
        if not directory.exists():
            return False, f"Directory does not exist: {directory}"
        _ = next(directory.iterdir(), None)
        return True, None
    except PermissionError as exc:
        return False, f"PermissionError accessing {directory}: {exc}"
    except OSError as exc:
        return False, f"OSError accessing {directory}: {exc}"


def resolve_run_edf_path(
    subject_id: int,
    run_id: int,
    local_data_dir: Path,
    workspace_data_dir: Path | None = None,
    allow_download_fallback: bool = False,
) -> Path:
    """Resolve the local EDF file path for a given subject and run.

    Search order:
    1. Configured `local_data_dir/S001/S001R04.edf` or `local_data_dir/S001R04.edf`.
    2. Explicit `workspace_data_dir/S001/S001R04.edf` or `workspace_data_dir/S001R04.edf`.
    3. Default workspace cache `data/S001/S001R04.edf` or `data/S001R04.edf`.

    Raises:
        FileNotFoundError: If the EDF file cannot be found locally.
    """
    subj_tag = f"S{subject_id:03d}"
    file_name = f"{subj_tag}R{run_id:02d}.edf"

    candidates: list[Path] = []
    candidates.append(local_data_dir / subj_tag / file_name)
    candidates.append(local_data_dir / file_name)

    if workspace_data_dir is not None:
        candidates.append(workspace_data_dir / subj_tag / file_name)
        candidates.append(workspace_data_dir / file_name)

    default_ws_data = Path("data")
    candidates.append(default_ws_data / subj_tag / file_name)
    candidates.append(default_ws_data / file_name)

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
    """Load a single EDF recording run with MNE, standardize channel names, and verify contract."""
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


def audit_single_subject_eligibility(
    subject_id: int,
    cfg: ProjectConfig,
    workspace_data_dir: Path | None = None,
    reference_channels: tuple[str, ...] | None = CANONICAL_EEGBCI_64_CHANNELS,
    return_processed_runs: bool = False,
) -> tuple[SubjectEligibilityRecord, dict[int, RunEpochsResult] | None]:
    """Audit a single participant across all required runs (4, 8, 12) against eligibility checks.

    Quality and eligibility criteria checked BEFORE any model evaluation:
    1. Not in excluded_subject_ids (e.g., pilot Subject 1 excluded by default).
    2. Required EDF files for development_runs (4, 8) and held_out_test_run (12) exist locally.
    3. Each EDF file is readable by MNE-Python.
    4. Standardized EEG channel count matches expected_channels (64) and channel names/order
       match the canonical 64-channel reference montage.
    5. Measured sampling frequency matches expected_sfreq (160.0 Hz).
    6. Annotations are in chronological order, contain no unexpected codes outside {T0, T1, T2},
       and every run contains valid [0.5, 2.5) s post-cue epochs of shape (n_trials, 64, 320)
       with finite signal values and both T1 (left) and T2 (right) classes present.
    """
    required_runs = (*cfg.data.development_runs, cfg.data.held_out_test_run)

    if subject_id in cfg.data.excluded_subject_ids:
        return (
            SubjectEligibilityRecord(
                subject_id=int(subject_id),
                eligible=False,
                status="excluded",
                reason_code="excluded_pilot_subject",
                reason_detail=(
                    f"Subject {subject_id} is in excluded_subject_ids {list(cfg.data.excluded_subject_ids)} "
                    "(pilot subject whose held-out Run 12 was already inspected)."
                ),
                checked_runs=required_runs,
            ),
            None,
        )

    resolved_paths: dict[int, str] = {}
    sfreq_by_run: dict[int, float] = {}
    n_channels_by_run: dict[int, int] = {}
    raw_ann_by_run: dict[int, dict[str, int]] = {}
    class_counts_by_run: dict[int, dict[str, int]] = {}
    processed_runs: dict[int, RunEpochsResult] = {}

    # Step 1: Verify all required EDF files exist locally
    for run_id in required_runs:
        try:
            p = resolve_run_edf_path(
                subject_id=subject_id,
                run_id=run_id,
                local_data_dir=cfg.data.local_data_dir,
                workspace_data_dir=workspace_data_dir,
                allow_download_fallback=False,
            )
            resolved_paths[run_id] = str(p)
        except FileNotFoundError as exc:
            return (
                SubjectEligibilityRecord(
                    subject_id=int(subject_id),
                    eligible=False,
                    status="excluded",
                    reason_code="missing_edf_file",
                    reason_detail=f"Missing required EDF for run {run_id}: {exc}",
                    checked_runs=required_runs,
                    resolved_edf_paths=resolved_paths,
                ),
                None,
            )

    # Step 2: Load each run independently, verify header/channels/sfreq, and extract epochs
    for run_id in required_runs:
        edf_path = resolved_paths[run_id]
        rec_id = format_recording_id(subject_id, run_id)
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
            mne.datasets.eegbci.standardize(raw)
            raw.pick("eeg")
        except Exception as exc:
            return (
                SubjectEligibilityRecord(
                    subject_id=int(subject_id),
                    eligible=False,
                    status="excluded",
                    reason_code="edf_read_error",
                    reason_detail=f"Failed to read/standardize {rec_id} ({edf_path}): {exc}",
                    checked_runs=required_runs,
                    resolved_edf_paths=resolved_paths,
                ),
                None,
            )

        ch_names = tuple(str(ch) for ch in raw.ch_names)
        n_ch = len(ch_names)
        sfreq = float(raw.info["sfreq"])
        n_channels_by_run[run_id] = n_ch
        sfreq_by_run[run_id] = sfreq

        if n_ch != cfg.data.expected_channels:
            return (
                SubjectEligibilityRecord(
                    subject_id=int(subject_id),
                    eligible=False,
                    status="excluded",
                    reason_code="channel_count_mismatch",
                    reason_detail=(
                        f"{rec_id} has {n_ch} EEG channels; expected {cfg.data.expected_channels}."
                    ),
                    checked_runs=required_runs,
                    resolved_edf_paths=resolved_paths,
                    sfreq_by_run=sfreq_by_run,
                    n_channels_by_run=n_channels_by_run,
                ),
                None,
            )

        if reference_channels is not None and len(reference_channels) == cfg.data.expected_channels:
            if ch_names != reference_channels:
                return (
                    SubjectEligibilityRecord(
                        subject_id=int(subject_id),
                        eligible=False,
                        status="excluded",
                        reason_code="channel_names_order_mismatch",
                        reason_detail=(
                            f"{rec_id} channel names/order do not match canonical 64-channel EEGBCI montage."
                        ),
                        checked_runs=required_runs,
                        resolved_edf_paths=resolved_paths,
                        sfreq_by_run=sfreq_by_run,
                        n_channels_by_run=n_channels_by_run,
                    ),
                    None,
                )

        if not np.isclose(sfreq, cfg.data.expected_sfreq, atol=1e-3):
            return (
                SubjectEligibilityRecord(
                    subject_id=int(subject_id),
                    eligible=False,
                    status="excluded",
                    reason_code="sfreq_mismatch",
                    reason_detail=(
                        f"{rec_id} has sampling rate {sfreq} Hz; expected {cfg.data.expected_sfreq} Hz."
                    ),
                    checked_runs=required_runs,
                    resolved_edf_paths=resolved_paths,
                    sfreq_by_run=sfreq_by_run,
                    n_channels_by_run=n_channels_by_run,
                ),
                None,
            )

        # Extract epochs (using filtered signal when return_processed_runs=True, or raw for fast audit)
        try:
            if return_processed_runs:
                run_res = process_single_run(raw=raw, subject_id=subject_id, run_id=run_id, cfg=cfg)
            else:
                run_res = extract_epochs_from_filtered_run(
                    raw_filtered=raw,
                    subject_id=subject_id,
                    run_id=run_id,
                    tmin_sec=cfg.preprocessing.epoch_tmin_sec,
                    tmax_sec=cfg.preprocessing.epoch_tmax_sec,
                    expected_n_times=cfg.preprocessing.expected_n_times,
                    label_mapping=cfg.data.label_mapping,
                    excluded_annotations=cfg.data.excluded_annotations,
                )
        except ValueError as exc:
            msg = str(exc)
            code = "unexpected_annotation" if "Unexpected annotation" in msg else "invalid_epoch_bounds_or_shape"
            return (
                SubjectEligibilityRecord(
                    subject_id=int(subject_id),
                    eligible=False,
                    status="excluded",
                    reason_code=code,
                    reason_detail=f"{rec_id} failed epoch extraction: {msg}",
                    checked_runs=required_runs,
                    resolved_edf_paths=resolved_paths,
                    sfreq_by_run=sfreq_by_run,
                    n_channels_by_run=n_channels_by_run,
                ),
                None,
            )
        except Exception as exc:
            return (
                SubjectEligibilityRecord(
                    subject_id=int(subject_id),
                    eligible=False,
                    status="excluded",
                    reason_code="invalid_epoch_bounds_or_shape",
                    reason_detail=f"{rec_id} failed epoch validation: {exc}",
                    checked_runs=required_runs,
                    resolved_edf_paths=resolved_paths,
                    sfreq_by_run=sfreq_by_run,
                    n_channels_by_run=n_channels_by_run,
                ),
                None,
            )

        n_left = int(np.sum(run_res.y == 0))
        n_right = int(np.sum(run_res.y == 1))
        n_total = int(len(run_res.y))
        raw_ann_by_run[run_id] = run_res.annotation_counts
        class_counts_by_run[run_id] = {
            "T1_left_0": n_left,
            "T2_right_1": n_right,
            "total_trials": n_total,
        }

        if cfg.data.require_both_classes_per_run and (n_left == 0 or n_right == 0):
            return (
                SubjectEligibilityRecord(
                    subject_id=int(subject_id),
                    eligible=False,
                    status="excluded",
                    reason_code="missing_class_annotations",
                    reason_detail=(
                        f"{rec_id} is missing one or both classes (T1_left={n_left}, T2_right={n_right})."
                    ),
                    checked_runs=required_runs,
                    resolved_edf_paths=resolved_paths,
                    sfreq_by_run=sfreq_by_run,
                    n_channels_by_run=n_channels_by_run,
                    per_run_raw_annotations=raw_ann_by_run,
                    per_run_class_counts=class_counts_by_run,
                ),
                None,
            )

        if n_total < cfg.data.min_trials_per_run:
            return (
                SubjectEligibilityRecord(
                    subject_id=int(subject_id),
                    eligible=False,
                    status="excluded",
                    reason_code="insufficient_trials_in_run",
                    reason_detail=(
                        f"{rec_id} has {n_total} trials; minimum required is {cfg.data.min_trials_per_run}."
                    ),
                    checked_runs=required_runs,
                    resolved_edf_paths=resolved_paths,
                    sfreq_by_run=sfreq_by_run,
                    n_channels_by_run=n_channels_by_run,
                    per_run_raw_annotations=raw_ann_by_run,
                    per_run_class_counts=class_counts_by_run,
                ),
                None,
            )

        if return_processed_runs:
            processed_runs[run_id] = run_res

    record = SubjectEligibilityRecord(
        subject_id=int(subject_id),
        eligible=True,
        status="eligible",
        reason_code="ok",
        reason_detail="Passed all file, channel, sampling rate, annotation, and epoch boundary checks.",
        checked_runs=required_runs,
        resolved_edf_paths=resolved_paths,
        sfreq_by_run=sfreq_by_run,
        n_channels_by_run=n_channels_by_run,
        per_run_raw_annotations=raw_ann_by_run,
        per_run_class_counts=class_counts_by_run,
    )
    return record, (processed_runs if return_processed_runs else None)


def audit_cohort_eligibility(
    cfg: ProjectConfig,
    workspace_data_dir: Path | None = None,
    reference_channels: tuple[str, ...] | None = CANONICAL_EEGBCI_64_CHANNELS,
) -> tuple[SubjectEligibilityRecord, ...]:
    """Audit all configured candidate and excluded subjects and return their eligibility records."""
    all_subjects_to_audit = sorted(
        set(cfg.data.excluded_subject_ids) | set(cfg.data.candidate_subject_ids)
    )
    records: list[SubjectEligibilityRecord] = []
    for sid in all_subjects_to_audit:
        rec, _ = audit_single_subject_eligibility(
            subject_id=sid,
            cfg=cfg,
            workspace_data_dir=workspace_data_dir,
            reference_channels=reference_channels,
            return_processed_runs=False,
        )
        records.append(rec)
    return tuple(records)


def combine_run_results(
    split_name: str,
    run_ids: tuple[int, ...],
    per_run_results: dict[int, RunEpochsResult],
) -> RunSplitBundle:
    """Combine already-filtered and epoched single-subject run results into a split bundle."""
    ordered = [per_run_results[rid] for rid in run_ids]
    return combine_recording_results(split_name=split_name, run_ids=run_ids, ordered_results=ordered)


def combine_recording_results(
    split_name: str,
    run_ids: tuple[int, ...],
    ordered_results: list[RunEpochsResult],
) -> RunSplitBundle:
    """Combine already-filtered and epoched RunEpochsResult objects across subjects and runs.

    Individual recordings are ALWAYS filtered and epoched independently prior to
    calling this function. No time-domain raw concatenation ever occurs.
    """
    if not ordered_results:
        raise ValueError(f"Split '{split_name}' has no recording results to combine.")

    first = ordered_results[0]
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    groups_list: list[np.ndarray] = []
    subject_ids_list: list[np.ndarray] = []
    metadata_all: list[TrialMetadata] = []
    per_run_compat: dict[int, RunEpochsResult] = {}
    recording_map: dict[str, RunEpochsResult] = {}

    for res in ordered_results:
        if res.channel_names != first.channel_names:
            raise ValueError(
                f"Channel name/order mismatch in {res.recording_id} inside split '{split_name}'."
            )
        if not np.isclose(res.sfreq, first.sfreq):
            raise ValueError(
                f"Sampling frequency mismatch in {res.recording_id} inside split '{split_name}'."
            )
        if res.recording_id in recording_map:
            raise ValueError(
                f"Duplicate recording_id '{res.recording_id}' in split '{split_name}'."
            )

        X_list.append(res.X)
        y_list.append(res.y)
        groups_list.append(res.groups)
        assert res.subject_ids is not None
        subject_ids_list.append(res.subject_ids)
        metadata_all.extend(res.trials_metadata)
        recording_map[res.recording_id] = res
        if res.run_id not in per_run_compat:
            per_run_compat[res.run_id] = res

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
        per_run_results=per_run_compat,
        subject_ids=np.concatenate(subject_ids_list, axis=0),
        recording_ids=tuple(m.recording_id for m in metadata_all),
        trial_ids=tuple(m.global_trial_id for m in metadata_all),
        recording_results=recording_map,
    )


def prepare_subject_dataset(
    cfg: ProjectConfig,
    workspace_data_dir: Path | None = None,
    reference_channels: tuple[str, ...] | None = None,
) -> PreparedEEGDataset:
    """Load, filter, epoch, and split single-subject or multi-subject cohort EEG runs.

    Maintains 100% backward compatibility with single-subject pilot configs (subject01.yaml)
    while supporting multi-participant Option A cohort configs (cohort_expanded.yaml).
    """
    dev_runs = cfg.data.development_runs
    test_run = cfg.data.held_out_test_run

    if test_run in dev_runs:
        raise ValueError(f"Leakage violation: test_run {test_run} in dev_runs {dev_runs}.")

    if not cfg.data.is_multi_subject:
        subject_id = cfg.data.subject_id
        all_runs = (*dev_runs, test_run)
        resolved_paths: dict[int, Path] = {}
        per_run_results: dict[int, RunEpochsResult] = {}
        rec_paths: dict[str, Path] = {}

        for run_id in all_runs:
            edf_path = resolve_run_edf_path(
                subject_id=subject_id,
                run_id=run_id,
                local_data_dir=cfg.data.local_data_dir,
                workspace_data_dir=workspace_data_dir,
                allow_download_fallback=cfg.data.allow_download_fallback,
            )
            resolved_paths[run_id] = edf_path
            rec_paths[format_recording_id(subject_id, run_id)] = edf_path
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

        if not set(dev_bundle.groups.tolist()).isdisjoint(set(test_bundle.groups.tolist())):
            raise AssertionError("Leakage detected between development and held-out test group IDs.")

        return PreparedEEGDataset(
            subject_id=subject_id,
            dev=dev_bundle,
            test=test_bundle,
            resolved_edf_paths=resolved_paths,
            subject_ids=(subject_id,),
            resolved_recording_paths=rec_paths,
        )

    return prepare_cohort_dataset(
        cfg=cfg,
        workspace_data_dir=workspace_data_dir,
        reference_channels=(
            CANONICAL_EEGBCI_64_CHANNELS
            if reference_channels is None and cfg.data.expected_channels == 64
            else reference_channels
        ),
    )


def prepare_cohort_dataset(
    cfg: ProjectConfig,
    workspace_data_dir: Path | None = None,
    reference_channels: tuple[str, ...] | None = CANONICAL_EEGBCI_64_CHANNELS,
) -> PreparedEEGDataset:
    """Audit candidate participants, load all eligible participants, and build dev/test splits."""
    dev_runs = cfg.data.development_runs
    test_run = cfg.data.held_out_test_run
    if test_run in dev_runs:
        raise ValueError(f"Leakage violation: test_run {test_run} in dev_runs {dev_runs}.")

    all_subjects = sorted(
        set(cfg.data.excluded_subject_ids) | set(cfg.data.candidate_subject_ids)
    )

    eligibility_records: list[SubjectEligibilityRecord] = []
    eligible_sids: list[int] = []
    dev_ordered_results: list[RunEpochsResult] = []
    test_ordered_results: list[RunEpochsResult] = []
    resolved_recording_paths: dict[str, Path] = {}
    first_eligible_paths: dict[int, Path] = {}

    for sid in all_subjects:
        rec, proc_runs = audit_single_subject_eligibility(
            subject_id=sid,
            cfg=cfg,
            workspace_data_dir=workspace_data_dir,
            reference_channels=reference_channels,
            return_processed_runs=True,
        )
        eligibility_records.append(rec)
        if rec.eligible and proc_runs is not None:
            eligible_sids.append(sid)
            for rid in dev_runs:
                res = proc_runs[rid]
                dev_ordered_results.append(res)
                resolved_recording_paths[res.recording_id] = Path(rec.resolved_edf_paths[rid])
            test_res = proc_runs[test_run]
            test_ordered_results.append(test_res)
            resolved_recording_paths[test_res.recording_id] = Path(
                rec.resolved_edf_paths[test_run]
            )
            if not first_eligible_paths:
                first_eligible_paths = {
                    rid: Path(rec.resolved_edf_paths[rid]) for rid in (*dev_runs, test_run)
                }

    if not eligible_sids:
        raise FileNotFoundError(
            "No eligible participants found in local dataset directory "
            f"'{cfg.data.local_data_dir}' (or workspace data/ directory)."
        )

    dev_bundle = combine_recording_results("development", dev_runs, dev_ordered_results)
    test_bundle = combine_recording_results("held_out_test", (test_run,), test_ordered_results)

    # Non-negotiable Option A split integrity checks
    assert dev_bundle.subject_ids is not None and test_bundle.subject_ids is not None
    dev_subjects_set = set(dev_bundle.subject_ids.tolist())
    test_subjects_set = set(test_bundle.subject_ids.tolist())
    if dev_subjects_set != test_subjects_set or dev_subjects_set != set(eligible_sids):
        raise AssertionError(
            "Participant mismatch between development and held-out test splits in Option A."
        )
    if not set(dev_bundle.groups.tolist()).isdisjoint(set(test_bundle.groups.tolist())):
        raise AssertionError("Run leakage detected between development and held-out test splits.")
    if not set(dev_bundle.recording_ids).isdisjoint(set(test_bundle.recording_ids)):
        raise AssertionError("Recording ID overlap detected between development and test splits.")

    return PreparedEEGDataset(
        subject_id=eligible_sids[0],
        dev=dev_bundle,
        test=test_bundle,
        resolved_edf_paths=first_eligible_paths,
        subject_ids=tuple(eligible_sids),
        resolved_recording_paths=resolved_recording_paths,
        eligibility_records=tuple(eligibility_records),
    )


def build_cohort_manifest_report(
    cfg: ProjectConfig,
    eligibility_records: tuple[SubjectEligibilityRecord, ...],
    workspace_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a reproducible JSON manifest of participant eligibility, exclusions, and splits."""
    local_accessible, local_access_err = _safe_dir_is_accessible(cfg.data.local_data_dir)
    ws_dir = workspace_data_dir if workspace_data_dir is not None else Path("data")
    ws_accessible, _ = _safe_dir_is_accessible(ws_dir)

    eligible_records = [r for r in eligibility_records if r.eligible]
    excluded_records = [r for r in eligibility_records if not r.eligible]

    exclusion_counts_by_reason: dict[str, int] = {}
    for r in excluded_records:
        exclusion_counts_by_reason[r.reason_code] = (
            exclusion_counts_by_reason.get(r.reason_code, 0) + 1
        )

    dev_runs = cfg.data.development_runs
    test_run = cfg.data.held_out_test_run
    run_a, run_b = dev_runs[0], dev_runs[1]

    def _aggregate_run_stats(target_runs: tuple[int, ...]) -> dict[str, Any]:
        rec_ids: list[str] = []
        n_left = 0
        n_right = 0
        for r in eligible_records:
            for rid in target_runs:
                rec_ids.append(format_recording_id(r.subject_id, rid))
                cc = r.per_run_class_counts.get(rid, {})
                n_left += int(cc.get("T1_left_0", 0))
                n_right += int(cc.get("T2_right_1", 0))
        return {
            "run_ids": list(target_runs),
            "n_subjects": len(eligible_records),
            "n_recordings": len(rec_ids),
            "recording_ids": rec_ids,
            "n_trials": n_left + n_right,
            "class_counts": {
                "T1_left_0": n_left,
                "T2_right_1": n_right,
            },
        }

    run_a_stats = _aggregate_run_stats((run_a,))
    run_b_stats = _aggregate_run_stats((run_b,))
    dev_total_stats = _aggregate_run_stats(dev_runs)
    test_stats = _aggregate_run_stats((test_run,))

    fold1_zero_overlap = set(run_a_stats["recording_ids"]).isdisjoint(
        set(run_b_stats["recording_ids"])
    )

    per_subject_summary: list[dict[str, Any]] = []
    for r in eligible_records:
        per_subject_summary.append(
            {
                "subject_id": r.subject_id,
                "status": r.status,
                "sfreq_hz": r.sfreq_by_run.get(run_a, cfg.data.expected_sfreq),
                "n_channels": r.n_channels_by_run.get(run_a, cfg.data.expected_channels),
                "run_04": r.per_run_class_counts.get(run_a, {}),
                "run_08": r.per_run_class_counts.get(run_b, {}),
                "run_12": r.per_run_class_counts.get(test_run, {}),
                "resolved_edf_paths": r.resolved_edf_paths,
            }
        )

    excluded_list: list[dict[str, Any]] = [
        {
            "subject_id": r.subject_id,
            "status": r.status,
            "reason_code": r.reason_code,
            "reason_detail": r.reason_detail,
        }
        for r in excluded_records
    ]

    return {
        "milestone": "E1 (Expanded Cohort Configuration, Eligibility Audit & Split Manifest — Option A)",
        "project_name": cfg.project_name,
        "experiment_mode": "multi_subject_option_a" if cfg.data.is_multi_subject else "single_subject_pilot",
        "local_data_dir": str(cfg.data.local_data_dir),
        "local_data_dir_accessible": local_accessible,
        "local_data_dir_access_note": local_access_err,
        "workspace_data_dir": str(ws_dir),
        "workspace_data_dir_accessible": ws_accessible,
        "allow_download_fallback": cfg.data.allow_download_fallback,
        "task_specification": {
            "task": "Imagined Left Fist (T1 -> 0) vs. Imagined Right Fist (T2 -> 1); Exclude Rest (T0)",
            "development_runs": list(dev_runs),
            "held_out_test_run": test_run,
            "expected_channels": cfg.data.expected_channels,
            "expected_sfreq_hz": cfg.data.expected_sfreq,
            "epoch_window_sec": [
                cfg.preprocessing.epoch_tmin_sec,
                cfg.preprocessing.epoch_tmax_sec,
            ],
            "expected_n_times": cfg.preprocessing.expected_n_times,
        },
        "cohort_audit_summary": {
            "total_subjects_audited": len(eligibility_records),
            "candidate_subject_ids_considered": list(cfg.data.candidate_subject_ids),
            "excluded_pilot_subject_ids": list(cfg.data.excluded_subject_ids),
            "n_eligible_subjects": len(eligible_records),
            "eligible_subject_ids": [r.subject_id for r in eligible_records],
            "n_excluded_subjects": len(excluded_records),
            "exclusion_counts_by_reason": exclusion_counts_by_reason,
        },
        "split_manifest": {
            "option_a_protocol": (
                "Same eligible participants appear in development (Runs 4, 8) and final test (Run 12). "
                "Evaluates generalization to new recordings of known participants."
            ),
            "development_total_runs_4_8": dev_total_stats,
            "development_fold_1_train_run4_val_run8": {
                "fold_index": 0,
                "train": run_a_stats,
                "val": run_b_stats,
                "zero_recording_overlap_verified": fold1_zero_overlap,
            },
            "development_fold_2_train_run8_val_run4": {
                "fold_index": 1,
                "train": run_b_stats,
                "val": run_a_stats,
                "zero_recording_overlap_verified": fold1_zero_overlap,
            },
            "held_out_test_run_12": {
                **test_stats,
                "same_subjects_as_development_verified": True,
                "zero_recording_overlap_with_development_verified": set(
                    dev_total_stats["recording_ids"]
                ).isdisjoint(set(test_stats["recording_ids"])),
            },
        },
        "eligible_participants_detail": per_subject_summary,
        "excluded_participants_detail": excluded_list,
    }


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
            "class_counts": {
                "T1_left_0": n_left,
                "T2_right_1": n_right,
                "total_trials": int(len(res.y)),
            },
            "signal_unit": res.signal_unit,
            "amplitude_volts_min": float(np.min(res.X)),
            "amplitude_volts_max": float(np.max(res.X)),
            "amplitude_microvolts_std": float(np.std(res.X) * 1e6),
            "first_trial_metadata": asdict(res.trials_metadata[0]),
        }

    return {
        "milestone": "M1 (Real EEG Ingestion & Preprocessing)",
        "subject_id": dataset.subject_id,
        "subject_ids": list(dataset.subject_ids),
        "n_subjects": len(dataset.subject_ids),
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
                "n_recordings": len(set(dataset.dev.recording_ids)),
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
                "n_recordings": len(set(dataset.test.recording_ids)),
                "groups_unique": sorted(int(g) for g in np.unique(dataset.test.groups)),
                "class_counts": {
                    "T1_left_0": int(np.sum(dataset.test.y == 0)),
                    "T2_right_1": int(np.sum(dataset.test.y == 1)),
                },
            },
        },
        "per_run_details": per_run_summary,
    }
