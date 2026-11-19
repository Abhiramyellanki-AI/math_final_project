"""Configuration loading and leakage-guardrail validation for eeg_p3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Dataset path, subject, split, and annotation contract settings."""

    dataset_name: str
    local_data_dir: Path
    allow_download_fallback: bool
    subject_id: int
    development_runs: tuple[int, ...]
    held_out_test_run: int
    expected_channels: int
    expected_sfreq: float
    signal_unit: str
    label_mapping: dict[str, int]
    excluded_annotations: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Fixed offline signal filtering and epoching parameters."""

    bandpass_low_hz: float
    bandpass_high_hz: float
    filter_method: str
    filter_design: str
    zero_phase_offline: bool
    epoch_tmin_sec: float
    epoch_tmax_sec: float
    expected_n_times: int


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level validated experiment configuration."""

    project_name: str
    version: str
    random_seed: int
    data: DataConfig
    preprocessing: PreprocessingConfig
    raw: dict[str, Any]


def validate_config_dict(raw: dict[str, Any]) -> None:
    """Validate raw configuration dictionary against PRD and AGENTS.md rules.

    Raises:
        ValueError: If any non-negotiable leakage or data contract constraint is violated.
    """
    required_sections = ("project", "data", "preprocessing")
    for section in required_sections:
        if section not in raw or not isinstance(raw[section], dict):
            raise ValueError(f"Missing required config section: '{section}'")

    data_cfg = raw["data"]
    dev_runs = tuple(int(r) for r in data_cfg.get("development_runs", []))
    test_run = int(data_cfg.get("held_out_test_run", -1))

    if not dev_runs:
        raise ValueError("development_runs must not be empty.")
    if test_run in dev_runs:
        raise ValueError(
            f"Data leakage violation: held_out_test_run ({test_run}) appears in "
            f"development_runs ({dev_runs})."
        )
    if len(set(dev_runs)) != len(dev_runs):
        raise ValueError(f"Duplicate runs found in development_runs: {dev_runs}")

    label_map = data_cfg.get("label_mapping", {})
    if label_map.get("T1") != 0 or label_map.get("T2") != 1:
        raise ValueError(
            f"Invalid label_mapping {label_map}: PRD requires T1 -> 0 (left) and T2 -> 1 (right)."
        )

    excluded = tuple(data_cfg.get("excluded_annotations", []))
    if "T0" not in excluded:
        raise ValueError("PRD requires rest annotation 'T0' to be in excluded_annotations.")

    prep_cfg = raw["preprocessing"]
    low_hz = float(prep_cfg.get("bandpass_low_hz", 0.0))
    high_hz = float(prep_cfg.get("bandpass_high_hz", 0.0))
    if not (0.0 < low_hz < high_hz):
        raise ValueError(f"Invalid bandpass range: [{low_hz}, {high_hz}] Hz.")

    tmin = float(prep_cfg.get("epoch_tmin_sec", 0.0))
    tmax = float(prep_cfg.get("epoch_tmax_sec", 0.0))
    sfreq = float(data_cfg.get("expected_sfreq", 160.0))
    expected_n_times = int(prep_cfg.get("expected_n_times", 0))
    computed_samples = int(round((tmax - tmin) * sfreq))
    if expected_n_times != computed_samples:
        raise ValueError(
            f"expected_n_times ({expected_n_times}) does not match "
            f"(tmax - tmin) * sfreq = ({tmax} - {tmin}) * {sfreq} = {computed_samples}."
        )


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated ProjectConfig instance.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    validate_config_dict(raw)

    proj = raw["project"]
    data = raw["data"]
    prep = raw["preprocessing"]

    data_config = DataConfig(
        dataset_name=str(data["dataset_name"]),
        local_data_dir=Path(str(data["local_data_dir"])),
        allow_download_fallback=bool(data.get("allow_download_fallback", False)),
        subject_id=int(data["subject_id"]),
        development_runs=tuple(int(r) for r in data["development_runs"]),
        held_out_test_run=int(data["held_out_test_run"]),
        expected_channels=int(data["expected_channels"]),
        expected_sfreq=float(data["expected_sfreq"]),
        signal_unit=str(data["signal_unit"]),
        label_mapping={str(k): int(v) for k, v in data["label_mapping"].items()},
        excluded_annotations=tuple(str(a) for a in data["excluded_annotations"]),
    )

    prep_config = PreprocessingConfig(
        bandpass_low_hz=float(prep["bandpass_low_hz"]),
        bandpass_high_hz=float(prep["bandpass_high_hz"]),
        filter_method=str(prep["filter_method"]),
        filter_design=str(prep["filter_design"]),
        zero_phase_offline=bool(prep["zero_phase_offline"]),
        epoch_tmin_sec=float(prep["epoch_tmin_sec"]),
        epoch_tmax_sec=float(prep["epoch_tmax_sec"]),
        expected_n_times=int(prep["expected_n_times"]),
    )

    return ProjectConfig(
        project_name=str(proj["name"]),
        version=str(proj["version"]),
        random_seed=int(proj["random_seed"]),
        data=data_config,
        preprocessing=prep_config,
        raw=raw,
    )
