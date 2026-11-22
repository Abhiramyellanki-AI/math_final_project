"""Local EDF ingestion, mechanical eligibility audit, and trial provenance."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import mne
import numpy as np

from .config import file_hash

RUNS = (4, 8, 12)
LABELS = {"T1": 0, "T2": 1}


def resolve_edf(root: Path, subject: int, run: int) -> Path:
    """Resolve only explicitly supported local layouts; never download."""
    if run not in RUNS:
        raise ValueError("Only left/right imagery runs 4, 8, 12 are allowed")
    name = f"S{subject:03d}R{run:02d}.edf"
    for path in (root / f"S{subject:03d}" / name, root / name):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Missing local recording {name} under {root}")


def epochs_from_raw(raw: mne.io.BaseRaw, cfg: dict, subject: int, run: int,
                    filtered: bool = True) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Extract [0.5,2.5) with rounded sample indexing and one vector per cue."""
    if run not in RUNS:
        raise ValueError("Wrong task run")
    raw = raw.copy().load_data()
    mne.datasets.eegbci.standardize(raw)
    raw.pick("eeg")
    if len(raw.ch_names) != cfg["channels"] or len(set(raw.ch_names)) != len(raw.ch_names):
        raise ValueError("channel_count_or_duplicate_mismatch")
    known = set(mne.channels.make_standard_montage(
        "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages() else "standard_1005").ch_names)
    if not set(raw.ch_names) <= known:
        raise ValueError("unrecognized_channel_names")
    raw.reorder_channels(sorted(raw.ch_names))
    if not np.isclose(raw.info["sfreq"], cfg["sfreq"], rtol=0, atol=1e-6):
        raise ValueError(f"sfreq_mismatch: {raw.info['sfreq']}")
    descriptions = set(raw.annotations.description)
    if not descriptions <= {"T0", "T1", "T2"} or not set(LABELS) <= descriptions:
        raise ValueError(f"invalid_annotations: {sorted(descriptions)}")
    # EDFs start at sample zero. Make other inputs explicit rather than misalign cues.
    if raw.first_samp != 0:
        raise ValueError("Nonzero first_samp is unsupported for this EDF protocol")
    if filtered:
        raw.filter(*cfg["bandpass"], method="fir", phase="zero", fir_design="firwin",
                   picks="eeg", verbose="ERROR")
    values = raw.get_data()
    if not np.isfinite(values).all():
        raise ValueError("nonfinite_signal")
    sf = cfg["sfreq"]
    offset = int(round(cfg["epoch"][0] * sf))
    length = int(round((cfg["epoch"][1] - cfg["epoch"][0]) * sf))
    epochs, labels, rows = [], [], []
    previous_stop = -1
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        if description == "T0":
            continue
        start = int(raw.time_as_index(onset, use_rounding=True)[0]) + offset
        stop = start + length
        if start < 0 or stop > values.shape[1]:
            raise ValueError("invalid_epoch_bounds")
        if start < previous_stop:
            raise ValueError("overlapping_or_unordered_task_epochs")
        previous_stop = stop
        trial = len(epochs)
        recording = f"S{subject:03d}R{run:02d}"
        epochs.append(values[:, start:stop])
        labels.append(LABELS[description])
        rows.append(dict(subject_id=subject, run_id=run, recording_id=recording,
                         trial_id=f"{recording}T{trial:03d}", trial_index=trial,
                         annotation=description, label=LABELS[description],
                         onset_sec=float(onset), start_sample=start, stop_sample=stop))
    X = np.asarray(epochs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if X.shape[1:] != (64, 320) or set(y) != {0, 1}:
        raise ValueError("invalid_epoch_shape_or_classes")
    return X, y, rows


def load_run(path: Path, cfg: dict, subject: int, run: int,
             filtered: bool = True) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Read one independent run and return measured loading/preprocessing times."""
    start = perf_counter()
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    read_sec = perf_counter() - start
    start = perf_counter()
    X, y, rows = epochs_from_raw(raw, cfg, subject, run, filtered=filtered)
    process_sec = perf_counter() - start
    mne.datasets.eegbci.standardize(raw)
    channels = sorted(raw.copy().pick("eeg").ch_names)
    return X, y, rows, dict(read_sec=read_sec, preprocessing_sec=process_sec,
                            channels=channels)


def audit(cfg: dict) -> dict:
    """Audit all candidates with the same objective checks, before model fitting."""
    root = Path(cfg["data_dir"])
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory is not accessible: {root}")
    eligible, excluded, records = [], [], []
    reference = None
    for subject in range(cfg["subject_start"], cfg["subject_stop"] + 1):
        if subject in cfg["excluded_subjects"]:
            excluded.append(dict(subject_id=subject, reason="prespecified_exclusion"))
            continue
        current = []
        try:
            for run in RUNS:
                path = resolve_edf(root, subject, run)
                _, y, rows, info = load_run(path, cfg, subject, run, filtered=False)
                if current and info["channels"] != current[0]["channels"]:
                    raise ValueError("channel_set_mismatch_between_runs")
                if reference is not None and info["channels"] != reference:
                    raise ValueError("channel_set_mismatch_across_participants")
                current.append(dict(subject_id=subject, run_id=run, path=str(path),
                                    sha256=file_hash(path), n_trials=len(y),
                                    class_counts=np.bincount(y, minlength=2).tolist(),
                                    channels=info["channels"], trials=rows))
        except (OSError, ValueError, RuntimeError) as exc:
            excluded.append(dict(subject_id=subject, reason=str(exc)))
            continue
        reference = current[0]["channels"]
        eligible.append(subject)
        records.extend(current)
    return dict(schema_version=1, eligible_subjects=eligible, exclusions=excluded,
                records=records, config=cfg,
                protocol="Personalized; development runs 4/8; exploratory follow-up run 12",
                audit_scope="Mechanical quality checks; no model selection on run 12")


def verify_manifest(manifest: dict) -> None:
    """Check file content provenance before any fit or evaluation."""
    if not manifest["eligible_subjects"]:
        raise ValueError("No eligible participants; inspect audit exclusions")
    for record in manifest["records"]:
        path = Path(record["path"])
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Dataset changed since audit: {path}")


def subject_records(manifest: dict, subject: int) -> dict[int, dict]:
    return {r["run_id"]: r for r in manifest["records"] if r["subject_id"] == subject}
