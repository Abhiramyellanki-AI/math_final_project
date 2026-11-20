"""Per-run band-pass filtering and fixed post-cue epoch extraction (M1 & Cohort Expansion).

Non-negotiable rules enforced:
- Every EDF recording run is filtered independently before epoch extraction.
- Zero-phase offline FIR filtering (8-30 Hz) is documented as offline/non-causal.
- No global scaler or data-dependent learned transform is applied during preprocessing.
- Fixed half-open post-cue window [0.5, 2.5) s yields exactly 320 samples at 160 Hz.
- Rest annotations (T0) are excluded; T1 -> 0 (left imagery), T2 -> 1 (right imagery).
- Preserves subject_id, run_id, unique recording_id ('S002R04'), global_trial_id ('S002R04_T00'),
  and strict chronological cue order for every trial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne
import numpy as np


def format_recording_id(subject_id: int, run_id: int) -> str:
    """Format a participant-and-run unique recording identifier (e.g., 'S002R04')."""
    return f"S{int(subject_id):03d}R{int(run_id):02d}"


def format_global_trial_id(subject_id: int, run_id: int, trial_index_in_run: int) -> str:
    """Format a globally unique trial identifier across participants and runs (e.g., 'S002R04_T00')."""
    return f"{format_recording_id(subject_id, run_id)}_T{int(trial_index_in_run):02d}"


@dataclass(frozen=True)
class TrialMetadata:
    """Per-trial provenance metadata preserving subject, run, recording ID, label, and cue timing."""

    subject_id: int
    run_id: int
    trial_index_in_run: int
    annotation_label: str
    numeric_label: int
    onset_sec: float
    onset_sample: int
    epoch_start_sample: int
    epoch_stop_sample: int
    recording_id: str = ""
    global_trial_id: str = ""

    def __post_init__(self) -> None:
        if not self.recording_id:
            object.__setattr__(
                self, "recording_id", format_recording_id(self.subject_id, self.run_id)
            )
        if not self.global_trial_id:
            object.__setattr__(
                self,
                "global_trial_id",
                format_global_trial_id(self.subject_id, self.run_id, self.trial_index_in_run),
            )


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
    recording_id: str = ""
    subject_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.recording_id:
            object.__setattr__(
                self, "recording_id", format_recording_id(self.subject_id, self.run_id)
            )
        if self.subject_ids is None:
            object.__setattr__(
                self,
                "subject_ids",
                np.full(shape=(len(self.y),), fill_value=int(self.subject_id), dtype=np.int64),
            )


def filter_raw_run_independently(
    raw: mne.io.BaseRaw,
    low_hz: float = 8.0,
    high_hz: float = 30.0,
    filter_method: str = "fir",
    fir_design: str = "firwin",
) -> mne.io.BaseRaw:
    """Apply an offline band-pass filter to a copy of a single independent raw run."""
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
    """Extract fixed [tmin_sec, tmax_sec) post-cue motor-imagery epochs from a single run."""
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
    prev_onset_sec = -1.0
    rec_id = format_recording_id(subject_id, run_id)

    for onset_sec, desc_raw in zip(
        raw_filtered.annotations.onset, raw_filtered.annotations.description
    ):
        onset_float = float(onset_sec)
        if onset_float < prev_onset_sec - 1e-9:
            raise ValueError(
                f"Non-chronological annotation order in {rec_id}: "
                f"onset {onset_float} < previous {prev_onset_sec}."
            )
        prev_onset_sec = onset_float

        desc = str(desc_raw).strip()
        if desc in excluded_annotations:
            continue
        if desc not in label_mapping:
            raise ValueError(
                f"Unexpected annotation '{desc}' in subject {subject_id} run {run_id}. "
                f"Allowed: {list(label_mapping.keys())}, excluded: {list(excluded_annotations)}."
            )

        onset_sample = int(raw_filtered.time_as_index(onset_float)[0])
        start_sample = int(onset_sample + offset_start)
        stop_sample = int(onset_sample + offset_stop)

        if start_sample < 0 or stop_sample > total_samples:
            raise ValueError(
                f"Epoch [{start_sample}, {stop_sample}) out of bounds for {rec_id} "
                f"with {total_samples} total samples."
            )

        epoch_slice = data_volts[:, start_sample:stop_sample].astype(np.float64, copy=True)
        if epoch_slice.shape != (n_channels, expected_n_times):
            raise AssertionError(
                f"Epoch shape mismatch: got {epoch_slice.shape}, "
                f"expected {(n_channels, expected_n_times)}."
            )
        if not np.all(np.isfinite(epoch_slice)):
            raise ValueError(f"Non-finite values in epoch {trial_idx} of {rec_id}.")

        numeric_label = int(label_mapping[desc])
        epochs_list.append(epoch_slice)
        labels_list.append(numeric_label)
        metadata_list.append(
            TrialMetadata(
                subject_id=int(subject_id),
                run_id=int(run_id),
                trial_index_in_run=trial_idx,
                annotation_label=desc,
                numeric_label=numeric_label,
                onset_sec=onset_float,
                onset_sample=onset_sample,
                epoch_start_sample=start_sample,
                epoch_stop_sample=stop_sample,
                recording_id=rec_id,
                global_trial_id=format_global_trial_id(subject_id, run_id, trial_idx),
            )
        )
        trial_idx += 1

    if not epochs_list:
        raise ValueError(f"No valid T1/T2 trials extracted from subject {subject_id} run {run_id}.")

    X_run = np.stack(epochs_list, axis=0)
    y_run = np.asarray(labels_list, dtype=np.int64)
    groups_run = np.full(shape=(len(labels_list),), fill_value=int(run_id), dtype=np.int64)
    subject_ids_run = np.full(
        shape=(len(labels_list),), fill_value=int(subject_id), dtype=np.int64
    )
    times_sec = np.arange(offset_start, offset_stop, dtype=np.float64) / sfreq

    return RunEpochsResult(
        subject_id=int(subject_id),
        run_id=int(run_id),
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
        recording_id=rec_id,
        subject_ids=subject_ids_run,
    )
