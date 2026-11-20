"""Train-only HOSVD channel/time subspace learning, projection, and storage accounting (M3).

Non-negotiable rules enforced (PRD Section 5.5 & AGENTS.md Rules 2, 8, 9):
- Channel (U_c) and time (U_t) singular subspaces are estimated strictly from the
  training tensor X_train of shape (N_train, C, T) via mode unfoldings.
- Never computes or exposes a trial-mode factor matrix as a feature representation.
- Any trial X_i of shape (C, T) is projected via Z_i = U_c.T @ X_i @ U_t of shape (r_c, r_t)
  and flattened to a 1D feature vector of length (r_c * r_t,).
- Deterministic singular vector sign convention ensures identical projections across runs.
- Distinguishes per-trial feature + deployed basis storage from full tensor Tucker-2 storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class HOSVDBases:
    """Orthonormal channel and time projection bases learned from training trials only."""

    U_c: np.ndarray  # Shape: (C, r_c), orthonormal columns
    U_t: np.ndarray  # Shape: (T, r_t), orthonormal columns
    singular_values_c: np.ndarray  # Shape: (r_c,)
    singular_values_t: np.ndarray  # Shape: (r_t,)
    n_channels: int
    n_times: int
    r_c: int
    r_t: int


def _fix_left_singular_vector_signs(U: np.ndarray) -> np.ndarray:
    """Enforce deterministic sign on each column of U (largest-magnitude element positive)."""
    max_abs_rows = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_rows, np.arange(U.shape[1])])
    signs[signs == 0.0] = 1.0
    return U * signs[np.newaxis, :]


def unfold_channel_mode(X: np.ndarray) -> np.ndarray:
    """Unfold 3D trial tensor (N, C, T) along the channel mode into (C, N * T)."""
    if X.ndim != 3:
        raise ValueError(f"Expected 3D tensor (N, C, T), got shape {X.shape}.")
    n_trials, n_channels, n_times = X.shape
    return np.transpose(X, (1, 0, 2)).reshape(n_channels, n_trials * n_times)


def unfold_time_mode(X: np.ndarray) -> np.ndarray:
    """Unfold 3D trial tensor (N, C, T) along the time mode into (T, N * C)."""
    if X.ndim != 3:
        raise ValueError(f"Expected 3D tensor (N, C, T), got shape {X.shape}.")
    n_trials, n_channels, n_times = X.shape
    return np.transpose(X, (2, 0, 1)).reshape(n_times, n_trials * n_channels)


def _left_svd_wide_matrix(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute deterministic left singular vectors U and singular values s of matrix A (m, n).

    For wide multi-participant unfoldings (n > 20,000 with m << n), uses Householder QR
    reduction A^T = Q R (mode='r') followed by SVD of R^T (m, m), yielding the exact
    left singular vectors and singular values without allocating an (m, n) right singular
    matrix V^T.
    """
    m, n = A.shape
    if n > 20000 and m < n:
        R = np.linalg.qr(A.T, mode="r")
        U_raw, s, _ = np.linalg.svd(R.T, full_matrices=False)
    else:
        U_raw, s, _ = np.linalg.svd(A, full_matrices=False)
    return _fix_left_singular_vector_signs(U_raw), s


def fit_channel_hosvd_basis(X_train: np.ndarray, r_c: int) -> np.ndarray:
    """Compute top-r_c deterministic orthonormal channel singular vectors U_c from X_train."""
    if X_train.ndim != 3:
        raise ValueError(f"Expected 3D training tensor (N_train, C, T), got {X_train.shape}.")
    X_mode_c = unfold_channel_mode(X_train)
    U_c_full, _ = _left_svd_wide_matrix(X_mode_c)
    if not (1 <= r_c <= U_c_full.shape[1]):
        raise ValueError(f"Invalid channel rank r_c={r_c}; must be in [1, {U_c_full.shape[1]}].")
    return U_c_full[:, :r_c].astype(np.float64, copy=True)


def fit_full_hosvd_singular_spaces(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute full deterministic channel and time singular bases from training tensor X_train.

    Note: Does NOT compute a trial-mode unfolding or trial-mode factor matrix.

    Args:
        X_train: Training EEG tensor of shape (N_train, C, T).

    Returns:
        Tuple (U_c_full, s_c_full, U_t_full, s_t_full) where U_c_full has shape (C, min(C, N*T))
        and U_t_full has shape (T, min(T, N*C)).
    """
    if X_train.ndim != 3:
        raise ValueError(f"Expected 3D training tensor (N_train, C, T), got {X_train.shape}.")

    X_mode_c = unfold_channel_mode(X_train)
    U_c_full, s_c = _left_svd_wide_matrix(X_mode_c)

    X_mode_t = unfold_time_mode(X_train)
    U_t_full, s_t = _left_svd_wide_matrix(X_mode_t)

    return U_c_full, s_c, U_t_full, s_t


def slice_hosvd_bases(
    U_c_full: np.ndarray,
    s_c_full: np.ndarray,
    U_t_full: np.ndarray,
    s_t_full: np.ndarray,
    r_c: int,
    r_t: int,
) -> HOSVDBases:
    """Truncate full training singular matrices to mode ranks (r_c, r_t)."""
    n_channels = int(U_c_full.shape[0])
    n_times = int(U_t_full.shape[0])

    if not (1 <= r_c <= U_c_full.shape[1]):
        raise ValueError(f"Invalid channel rank r_c={r_c}; must be in [1, {U_c_full.shape[1]}].")
    if not (1 <= r_t <= U_t_full.shape[1]):
        raise ValueError(f"Invalid time rank r_t={r_t}; must be in [1, {U_t_full.shape[1]}].")

    return HOSVDBases(
        U_c=U_c_full[:, :r_c].astype(np.float64, copy=True),
        U_t=U_t_full[:, :r_t].astype(np.float64, copy=True),
        singular_values_c=s_c_full[:r_c].astype(np.float64, copy=True),
        singular_values_t=s_t_full[:r_t].astype(np.float64, copy=True),
        n_channels=n_channels,
        n_times=n_times,
        r_c=int(r_c),
        r_t=int(r_t),
    )


def fit_hosvd_bases(X_train: np.ndarray, r_c: int, r_t: int) -> HOSVDBases:
    """Learn rank-(r_c, r_t) channel and time HOSVD subspaces from training trials only."""
    U_c_full, s_c_full, U_t_full, s_t_full = fit_full_hosvd_singular_spaces(X_train)
    return slice_hosvd_bases(U_c_full, s_c_full, U_t_full, s_t_full, r_c=r_c, r_t=r_t)


def project_hosvd_trials(
    X: np.ndarray,
    U_c: np.ndarray,
    U_t: np.ndarray,
    flatten: bool = True,
) -> np.ndarray:
    """Project each trial X_i (C, T) via Z_i = U_c.T @ X_i @ U_t.

    Accepts either a 2D single trial (C, T) or a 3D batch of trials (N, C, T).

    Args:
        X: Single trial of shape (C, T) or trial tensor of shape (N, C, T).
        U_c: Orthonormal channel basis of shape (C, r_c).
        U_t: Orthonormal time basis of shape (T, r_t).
        flatten: If True, flattens each (r_c, r_t) core matrix to (r_c * r_t,).

    Returns:
        For 3D input: array of shape (N, r_c * r_t) if flatten=True else (N, r_c, r_t).
        For 2D input: array of shape (r_c * r_t,) if flatten=True else (r_c, r_t).
    """
    single_trial = X.ndim == 2
    if single_trial:
        X_batch = X[np.newaxis, :, :]
    elif X.ndim == 3:
        X_batch = X
    else:
        raise ValueError(f"Expected 2D (C, T) or 3D (N, C, T) array, got shape {X.shape}.")

    n_trials, n_channels, n_times = X_batch.shape
    if U_c.shape[0] != n_channels:
        raise ValueError(f"U_c row count {U_c.shape[0]} does not match n_channels {n_channels}.")
    if U_t.shape[0] != n_times:
        raise ValueError(f"U_t row count {U_t.shape[0]} does not match n_times {n_times}.")

    r_c = int(U_c.shape[1])
    r_t = int(U_t.shape[1])

    # Z_i = U_c.T @ X_i @ U_t for every trial i
    cores = np.einsum("cr,nct,ts->nrs", U_c, X_batch, U_t, optimize=True)
    if flatten:
        out = cores.reshape(n_trials, r_c * r_t)
    else:
        out = cores

    return out[0] if single_trial else out


def reconstruct_hosvd_trials(
    X: np.ndarray,
    U_c: np.ndarray,
    U_t: np.ndarray,
) -> np.ndarray:
    """Reconstruct trials from their rank-(r_c, r_t) projection: X_hat_i = U_c @ Z_i @ U_t.T."""
    cores = project_hosvd_trials(X, U_c, U_t, flatten=False)
    if X.ndim == 2:
        return U_c @ cores @ U_t.T
    return np.einsum("cr,nrs,ts->nct", U_c, cores, U_t, optimize=True)


def compute_relative_frobenius_error(
    X: np.ndarray,
    U_c: np.ndarray,
    U_t: np.ndarray,
) -> float:
    """Compute relative Frobenius reconstruction error ||X - X_hat||_F / ||X||_F."""
    denom = float(np.linalg.norm(X))
    if denom == 0.0:
        return 0.0
    X_hat = reconstruct_hosvd_trials(X, U_c, U_t)
    num = float(np.linalg.norm(X - X_hat))
    return float(num / denom)


def compute_hosvd_storage_accounting(
    n_trials: int,
    n_channels: int,
    n_times: int,
    r_c: int,
    r_t: int,
    bytes_per_float: int = 8,
) -> dict[str, Any]:
    """Compute both per-trial deployed extractor storage and full tensor Tucker-2 storage.

    Distinguishes:
    1. Per-trial projected feature vector size vs. raw trial size, plus reusable bases (U_c, U_t).
    2. Full tensor decomposition storage (N projected cores + U_c + U_t) vs. raw N*C*T tensor.
    """
    raw_trial_elements = int(n_channels * n_times)
    projected_trial_features = int(r_c * r_t)
    basis_uc_elements = int(n_channels * r_c)
    basis_ut_elements = int(n_times * r_t)
    reusable_bases_elements = basis_uc_elements + basis_ut_elements

    raw_tensor_elements = int(n_trials * raw_trial_elements)
    decomposed_tensor_elements = int(n_trials * projected_trial_features + reusable_bases_elements)

    return {
        "n_features": projected_trial_features,
        "raw_trial_elements": raw_trial_elements,
        "compression_ratio_vs_raw": float(raw_trial_elements / projected_trial_features),
        "tensor_decomposition_compression_ratio": float(
            raw_tensor_elements / decomposed_tensor_elements
        ),
        "feature_bytes_per_trial": int(projected_trial_features * bytes_per_float),
        "raw_bytes_per_trial": int(raw_trial_elements * bytes_per_float),
        "reusable_bases_elements": reusable_bases_elements,
        "reusable_bases_bytes": int(reusable_bases_elements * bytes_per_float),
        "decomposed_tensor_total_bytes": int(decomposed_tensor_elements * bytes_per_float),
        "raw_tensor_total_bytes": int(raw_tensor_elements * bytes_per_float),
    }


class HOSVDFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer fitting train-only (U_c, U_t) bases and projecting trials."""

    def __init__(self, r_c: int = 8, r_t: int = 20) -> None:
        self.r_c = r_c
        self.r_t = r_t

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> HOSVDFeatureExtractor:
        """Fit channel and time bases U_c and U_t strictly on training trials X."""
        bases = fit_hosvd_bases(X, r_c=int(self.r_c), r_t=int(self.r_t))
        self.bases_ = bases
        self.U_c_ = bases.U_c
        self.U_t_ = bases.U_t
        self.n_features_out_ = int(bases.r_c * bases.r_t)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project trials X of shape (N, C, T) into flattened cores (N, r_c * r_t)."""
        if not hasattr(self, "U_c_") or not hasattr(self, "U_t_"):
            raise RuntimeError("HOSVDFeatureExtractor must be fitted before calling transform().")
        return project_hosvd_trials(X, self.U_c_, self.U_t_, flatten=True)
