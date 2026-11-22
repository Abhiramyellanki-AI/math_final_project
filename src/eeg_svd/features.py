"""Deterministic trial-level spectral features shared by both pipelines."""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


def bandpower(X: np.ndarray, cfg: dict) -> np.ndarray:
    """Integrate Welch density (V²/Hz) and return log10 power by band then channel."""
    if X.ndim != 3 or X.shape[1:] != (64, 320) or not np.isfinite(X).all():
        raise ValueError("Expected finite (trials,64,320) EEG tensor")
    f, psd = welch(X, fs=cfg["sfreq"], window="hann", nperseg=cfg["welch_nperseg"],
                   noverlap=cfg["welch_nperseg"] // 2, detrend="constant", scaling="density",
                   axis=-1)
    features = []
    for low, high in cfg["bands"]:
        mask = (f >= low) & (f <= high)
        if mask.sum() < 2:
            raise ValueError("Welch frequency resolution is insufficient")
        power = trapezoid(psd[..., mask], x=f[mask], axis=-1)
        features.append(np.log10(np.maximum(power, cfg["power_floor"])))
    return np.concatenate(features, axis=1)
