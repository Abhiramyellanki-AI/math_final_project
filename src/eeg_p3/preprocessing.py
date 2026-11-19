"""Per-run band-pass filtering and fixed post-cue epoch extraction (Milestone M1).

Non-negotiable rules enforced:
- Every EDF recording run is filtered independently before epoch extraction.
- Zero-phase offline FIR filtering (8-30 Hz) is documented as offline/non-causal.
- No global scaler or data-dependent learned transform is applied during preprocessing.
- Fixed half-open post-cue window [0.5, 2.5) s yields exactly 320 samples at 160 Hz.
- Rest annotations (T0) are excluded; T1 -> 0 (left imagery), T2 -> 1 (right imagery).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne
import numpy as np


@dataclass(frozen=True)
class TrialMetadata:
    """Per-trial provenance metadata preserving subject, run, label, and cue timing."""

    subject_id: int
    run_id: int
    trial_index_in_run: int
    annotation_label: str
    numeric_label: int
    onset_sec: float
    onset_sample: int
    epoch_start_sample: int
    epoch_stop_sample: int


@dataclass(frozen=True)
class RunEpochsResult:
    """Processed epochs and metadata extracted from a single independent recording run."""

    subject_id: int
    run_id: int
    X: np.ndarray  # Shape: (n_trials, n_channels, n_times), float64 in Volts
    y: np.ndarray  # Shape: (n_trials,), int64 in {0, 1}
    groups: np.ndarray  # Shape: (n_trials,), int64 equal to run_id
    times_sec: np.ndarray  # Shape: (n_times,), relative to cue onset [0.5, 2.5)
    channel_names: tuple[str, ...]
    sfreq: float
    signal_unit: str
    annotation_counts: dict[str, int]
    trials_metadata: tuple[TrialMetadata, ...]
    filter_params: dict[str, Any]


def filter_raw_run_independently(
    raw: mne.io.BaseRaw,
    low_hz: float = 8.0,
    high_hz: float = 30.0,
    filter_method: str = "fir",
    fir_design: str = "firwin",
) -> mne.io.BaseRaw:
    """Apply an offline band-pass filter to a copy of a single independent raw run.

    Args:
        raw: Preloaded MNE Raw object for a single recording run.
        low_hz: Lower passband edge in Hz (default 8.0 Hz for mu band).
        high_hz: Upper passband edge in Hz (default 30.0 Hz for beta band).
        filter_method: MNE filter method ('fir').
        fir_design: FIR window design ('firwin').

    Returns:
        Filtered copy of the MNE Raw recording (units preserved in Volts).
    """
    if not (0.0 < low_hz < high_hz):
        raise ValueError(f"Invalid bandpass cutoff frequencies: [{low_hz}, {high_hz}] Hz.")

    raw_filtered = raw.copy()
    raw_filtered.filter(
        l_freq=low_hz,
        h_freq=high_hz,
        method=filter_method,
        fir_design=fir_design,
        phase="zero",
        verbose="ERROR",
    )
    return raw_filtered


def count_run_annotations(raw: mne.io.BaseRaw) -> dict[str, int]:
    """Count occurrences of each annotation label in a raw recording."""
    counts: dict[str, int] = {}
    for desc in raw.annotations.description:
        key = str(desc).strip()
        counts[key] = counts.get(key, 0) + 1
    return counts


def extract_epochs_from_filtered_run(
    raw_filtered: mne.io.BaseRaw,
    subject_id: int,
    run_id: int,
    tmin_sec: float = 0.5,
    tmax_sec: float = 2.5,
    expected_n_times: int = 320,
    label_mapping: dict[str, int] | None = None,
    excluded_annotations: tuple[str, ...] = ("T0",),
    filter_params: dict[str, Any] | None = None,
) -> RunEpochsResult:
    """Extract fixed [tmin_sec, tmax_sec) post-cue motor-imagery epochs from a single run.

    Uses exact sample indexing based on the recording's measured sampling frequency
    so that a 2.0-second half-open window [0.5, 2.5) s at 160 Hz contains exactly
    320 time samples, excluding T0 rest annotations and mapping T1 -> 0, T2 -> 1.

    Args:
        raw_filtered: Independently band-pass filtered MNE Raw instance.
        subject_id: Integer subject identifier (e.g., 1).
        run_id: Integer recording run identifier (e.g., 4, 8, or 12).
        tmin_sec: Post-cue window start in seconds (default 0.5).
        tmax_sec: Post-cue window stop (exclusive) in seconds (default 2.5).
        expected_n_times: Expected sample count per epoch (default 320 at 160 Hz).
        label_mapping: Mapping from annotation string to class label {'T1': 0, 'T2': 1}.
        excluded_annotations: Annotations to skip (default ('T0',)).
        filter_params: Metadata describing the applied band-pass filter.

    Returns:
        RunEpochsResult containing (n_trials, n_channels, n_times) array X in Volts,
        labels y in {0, 1}, run group IDs, and trial-level provenance metadata.
    """
    if label_mapping is None:
        label_mapping = {"T1": 0, "T2": 1}
    if label_mapping.get("T1") != 0 or label_mapping.get("T2") != 1:
        raise ValueError(f"Invalid label_mapping {label_mapping}; must map T1->0 and T2->1.")

    sfreq = float(raw_filtered.info["sfreq"])
    offset_start = int(round(tmin_sec * sfreq))
    offset_stop = int(round(tmax_sec * sfreq))
    n_times = offset_stop - offset_start
    if n_times != expected_n_times:
        raise ValueError(
            f"Computed sample count {n_times} for [{tmin_sec}, {tmax_sec}) at {sfreq} Hz "
            f"does not match expected_n_times={expected_n_times}."
        )

    data_volts = raw_filtered.get_data(units="V")  # (n_channels, total_samples)
    n_channels, total_samples = data_volts.shape
    ch_names = tuple(str(ch) for ch in raw_filtered.ch_names)
    annotation_counts = count_run_annotations(raw_filtered)

    epochs_list: list[np.ndarray] = []
    labels_list: list[int] = []
    metadata_list: list[TrialMetadata] = []

    trial_idx = 0
    for onset_sec, desc_raw in zip(raw_filtered.annotations.onset, raw_filtered.annotations.description):
        desc = str(desc_raw).strip()
        if desc in excluded_annotations:
            continue
        if desc not in label_mapping:
            raise ValueError(
                f"Unexpected annotation '{desc}' in subject {subject_id} run {run_id}. "
                f"Allowed: {list(label_mapping.keys())}, excluded: {list(excluded_annotations)}."
            )

        onset_sample = raw_filtered.time_as_index(float(onset_sec))[0]
        start_sample = int(onset_sample + offset_start)
        stop_sample = int(onset_sample + offset_stop)

        if start_sample < 0 or stop_sample > total_samples:
            raise ValueError(
                f"Epoch [{start_sample}, {stop_sample}) out of bounds for run {run_id} "
                f"with {total_samples} total samples."
            )

        epoch_slice = data_volts[:, start_sample:stop_sample].astype(np.float64, copy=True)
        if epoch_slice.shape != (n_channels, expected_n_times):
            raise AssertionError(
                f"Epoch shape mismatch: got {epoch_slice.shape}, "
                f"expected {(n_channels, expected_n_times)}."
            )

        numeric_label = int(label_mapping[desc])
        epochs_list.append(epoch_slice)
        labels_list.append(numeric_label)
        metadata_list.append(
            TrialMetadata(
                subject_id=subject_id,
                run_id=run_id,
                trial_index_in_run=trial_idx,
                annotation_label=desc,
                numeric_label=numeric_label,
                onset_sec=float(onset_sec),
                onset_sample=int(onset_sample),
                epoch_start_sample=start_sample,
                epoch_stop_sample=stop_sample,
            )
        )
        trial_idx += 1

    if not epochs_list:
        raise ValueError(f"No valid T1/T2 trials extracted from subject {subject_id} run {run_id}.")

    X_run = np.stack(epochs_list, axis=0)
    y_run = np.asarray(labels_list, dtype=np.int64)
    groups_run = np.full(shape=(len(labels_list),), fill_value=int(run_id), dtype=np.int64)
    times_sec = np.arange(offset_start, offset_stop, dtype=np.float64) / sfreq

    return RunEpochsResult(
        subject_id=subject_id,
        run_id=run_id,
        X=X_run,
        y=y_run,
        groups=groups_run,
        times_sec=times_sec,
        channel_names=ch_names,
        sfreq=sfreq,
        signal_unit="V",
        annotation_counts=annotation_counts,
        trials_metadata=tuple(metadata_list),
        filter_params=filter_params
        or {
            "bandpass_low_hz": 8.0,
            "bandpass_high_hz": 30.0,
            "method": "fir",
            "fir_design": "firwin",
            "phase": "zero (offline non-causal)",
        },
    )
