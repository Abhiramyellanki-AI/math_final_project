"""Deterministic mu/beta log-bandpower feature extraction (Milestone M2).

Extracts 2 * n_channels features per trial (128 features for 64 channels):
- Features [0..63]: log10 bandpower in the mu band (8-13 Hz) for each channel.
- Features [64..127]: log10 bandpower in the beta band (13-30 Hz) for each channel.

This transformation is purely per-trial and deterministic (no parameters learned
across trials), so it cannot leak information across trials or runs. Any feature
standardization (StandardScaler) is applied downstream strictly inside each training fold.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin


def extract_mu_beta_log_bandpower(
    X: np.ndarray,
    sfreq: float = 160.0,
    mu_band_hz: tuple[float, float] = (8.0, 13.0),
    beta_band_hz: tuple[float, float] = (13.0, 30.0),
    log_epsilon: float = 1e-12,
    nperseg: int | None = None,
) -> np.ndarray:
    """Compute per-channel mu (8-13 Hz) and beta (13-30 Hz) log10-bandpower features.

    Args:
        X: EEG trial tensor of shape (n_trials, n_channels, n_times) in Volts.
        sfreq: Sampling frequency in Hz (default 160.0).
        mu_band_hz: (low_hz, high_hz) for sensorimotor mu rhythm (default (8.0, 13.0)).
        beta_band_hz: (low_hz, high_hz) for sensorimotor beta rhythm (default (13.0, 30.0)).
        log_epsilon: Small positive constant added before log10 to avoid log(0).
        nperseg: Welch segment length in samples (default min(160, n_times) -> 1 Hz resolution).

    Returns:
        Feature matrix of shape (n_trials, 2 * n_channels) of dtype float64.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D trial tensor (n_trials, n_channels, n_times), got shape {X.shape}.")
    if log_epsilon <= 0.0:
        raise ValueError(f"log_epsilon must be positive, got {log_epsilon}.")

    n_trials, n_channels, n_times = X.shape
    seg_len = min(int(round(sfreq)), n_times) if nperseg is None else min(int(nperseg), n_times)

    freqs, psd = welch(
        X,
        fs=sfreq,
        window="hann",
        nperseg=seg_len,
        noverlap=seg_len // 2,
        axis=-1,
    )  # psd shape: (n_trials, n_channels, n_freqs)

    mu_mask = (freqs >= mu_band_hz[0]) & (freqs <= mu_band_hz[1])
    beta_mask = (freqs >= beta_band_hz[0]) & (freqs <= beta_band_hz[1])

    if not np.any(mu_mask) or not np.any(beta_mask):
        raise ValueError(
            f"Frequency grid (max={freqs[-1]} Hz) does not cover mu={mu_band_hz} or beta={beta_band_hz}."
        )

    mu_power = trapezoid(psd[:, :, mu_mask], x=freqs[mu_mask], axis=-1)
    beta_power = trapezoid(psd[:, :, beta_mask], x=freqs[beta_mask], axis=-1)

    mu_log = np.log10(np.maximum(mu_power, 0.0) + log_epsilon)
    beta_log = np.log10(np.maximum(beta_power, 0.0) + log_epsilon)

    features = np.concatenate([mu_log, beta_log], axis=1).astype(np.float64, copy=False)
    if features.shape != (n_trials, 2 * n_channels):
        raise AssertionError(f"Unexpected feature shape {features.shape}.")
    if not np.all(np.isfinite(features)):
        raise ValueError("Non-finite values encountered in mu/beta log-bandpower features.")

    return features


class MuBetaLogBandpowerTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible stateless transformer for mu/beta log-bandpower extraction."""

    def __init__(
        self,
        sfreq: float = 160.0,
        mu_band_hz: tuple[float, float] = (8.0, 13.0),
        beta_band_hz: tuple[float, float] = (13.0, 30.0),
        log_epsilon: float = 1e-12,
    ) -> None:
        self.sfreq = sfreq
        self.mu_band_hz = mu_band_hz
        self.beta_band_hz = beta_band_hz
        self.log_epsilon = log_epsilon

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> MuBetaLogBandpowerTransformer:
        """Stateless fit; validates 3D input shape (n_trials, n_channels, n_times)."""
        if X.ndim != 3:
            raise ValueError(f"Expected 3D tensor (n_trials, n_channels, n_times), got {X.shape}.")
        self.n_channels_in_ = int(X.shape[1])
        self.n_times_in_ = int(X.shape[2])
        self.n_features_out_ = 2 * self.n_channels_in_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract (n_trials, 2 * n_channels) mu/beta log-bandpower feature matrix."""
        return extract_mu_beta_log_bandpower(
            X=X,
            sfreq=self.sfreq,
            mu_band_hz=self.mu_band_hz,
            beta_band_hz=self.beta_band_hz,
            log_epsilon=self.log_epsilon,
        )
