"""Strict configuration and reproducible JSON artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULTS = dict(subject_start=2, subject_stop=109, excluded_subjects=[1], sfreq=160.0,
                channels=64, bandpass=[8.0, 30.0], epoch=[0.5, 2.5],
                bands=[[8.0, 13.0], [13.0, 30.0]], welch_nperseg=160,
                power_floor=1e-20, ranks=[2, 4, 6, 8, 10, 12],
                c_values=[0.1, 1., 10., 100.], gammas=["scale", .001, .01, .1],
                timing_repeats=5, seed=42, prior_test_exposure=True)


def check_config(cfg: dict) -> dict:
    """Validate the fixed task contract and bounded search settings."""
    if set(cfg) - (set(DEFAULTS) | {"data_dir"}):
        raise ValueError("Unknown configuration keys")
    if not cfg.get("data_dir"):
        raise ValueError("data_dir is required; downloads are never automatic")
    if not 1 <= cfg["subject_start"] <= cfg["subject_stop"] <= 109:
        raise ValueError("Subject range must be within 1..109")
    if cfg["sfreq"] != 160 or cfg["channels"] != 64:
        raise ValueError("This protocol requires 64 channels at 160 Hz")
    if cfg["epoch"] != [.5, 2.5] or cfg["bandpass"] != [8., 30.]:
        raise ValueError("Protocol fixes epoch [0.5,2.5) and filter [8,30] Hz")
    if cfg["bands"] != [[8., 13.], [13., 30.]]:
        raise ValueError("Protocol fixes mu/beta bands")
    if not cfg["prior_test_exposure"]:
        raise ValueError("Previous test inspection must remain disclosed")
    if not 2 <= cfg["welch_nperseg"] <= 320 or cfg["power_floor"] <= 0:
        raise ValueError("Invalid PSD settings")
    if not cfg["ranks"] or any(type(r) is not int or not 1 <= r <= 128 for r in cfg["ranks"]):
        raise ValueError("Ranks must be integers in 1..128")
    if not cfg["c_values"] or any(not np.isfinite(c) or c <= 0 for c in cfg["c_values"]):
        raise ValueError("C values must be positive and finite")
    if not cfg["gammas"] or any(g != "scale" and
                                (not isinstance(g, (float, int)) or not np.isfinite(g) or g <= 0)
                                for g in cfg["gammas"]):
        raise ValueError("Gamma must be scale or a positive finite number")
    if type(cfg["timing_repeats"]) is not int or cfg["timing_repeats"] < 2:
        raise ValueError("At least two timing repetitions required")
    return cfg


def load_config(path: str | Path) -> dict:
    """Read YAML and apply documented defaults."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    return check_config({**DEFAULTS, **raw})


def json_text(obj: Any) -> str:
    """Serialize strict JSON, rejecting NaN and infinity."""
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"


def digest(obj: Any) -> str:
    return hashlib.sha256(json_text(obj).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json_text(obj), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def new_directory(path: Path) -> Path:
    """Refuse to overwrite even a partially completed experiment."""
    path.mkdir(parents=True, exist_ok=False)
    return path
