"""Random Kitchen Sinks / Random Fourier Features (RBFSampler + LinearSVC) (Milestone M4).

Non-negotiable rules enforced (PRD Section 5.6 & AGENTS.md Rules 2, 5, 8, 10):
- Input is the shared deterministic 128-D mu/beta log-bandpower representation.
- StandardScaler and RBFSampler are fit strictly on the training fold.
- Fixed deterministic random seeds ensure reproducible Fourier weights W and offsets b.
- RKS is kernel approximation (approximating an RBF kernel via explicit D-dimensional
  cosine features z(x) = sqrt(2/D) cos(W x + b)), NOT guaranteed dimensionality reduction.
- Reports both feature expansion ratio relative to 128-D bandpower (D / 128) and
  compression ratio relative to the 20,480-element raw trial tensor (20,480 / D).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler


def compute_rks_storage_accounting(
    input_dim: int,
    n_components: int,
    raw_trial_elements: int = 20480,
    bytes_per_float: int = 8,
) -> dict[str, Any]:
    """Compute dimension, expansion/compression ratios, and byte footprint for RKS features.

    Explicitly distinguishes:
    - expansion_ratio_vs_bandpower = D / input_dim (kernel feature expansion, >= 1.0 when D >= 128).
    - compression_ratio_vs_raw = raw_trial_elements / D (relative to raw C*T trial tensor).
    - rks_weights_bytes = 8 * (input_dim * D + D) bytes for random_weights_ (W) and random_offset_ (b).
    """
    if input_dim <= 0 or n_components <= 0:
        raise ValueError(f"input_dim ({input_dim}) and n_components ({n_components}) must be positive.")

    rks_weights_elements = int(input_dim * n_components + n_components)
    input_scaler_elements = int(2 * input_dim)  # mean_ and scale_

    return {
        "input_bandpower_features": int(input_dim),
        "n_features": int(n_components),
        "expansion_ratio_vs_bandpower": float(n_components / input_dim),
        "is_dimensionality_expansion_vs_bandpower": bool(n_components > input_dim),
        "compression_ratio_vs_raw": float(raw_trial_elements / n_components),
        "feature_bytes_per_trial": int(n_components * bytes_per_float),
        "rks_weights_elements": rks_weights_elements,
        "rks_weights_bytes": int(rks_weights_elements * bytes_per_float),
        "input_scaler_bytes": int(input_scaler_elements * bytes_per_float),
    }


class RKSFeatureMapper(BaseEstimator, TransformerMixin):
    """Train-only StandardScaler followed by RBFSampler (Random Kitchen Sinks).

    Fits StandardScaler on the training fold's 128-D bandpower vectors, then applies
    RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state).
    """

    def __init__(
        self,
        n_components: int = 256,
        gamma: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> RKSFeatureMapper:
        """Fit StandardScaler and RBFSampler random projection weights on training data X."""
        if X.ndim != 2:
            raise ValueError(f"Expected 2D bandpower feature matrix (N, d), got shape {X.shape}.")

        self.input_dim_ = int(X.shape[1])
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.rbf_sampler_ = RBFSampler(
            gamma=float(self.gamma),
            n_components=int(self.n_components),
            random_state=int(self.random_state),
        )
        self.rbf_sampler_.fit(X_scaled)
        self.n_features_out_ = int(self.n_components)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform 2D bandpower features (N, d) into D-dimensional RKS cosine features (N, D)."""
        if not hasattr(self, "scaler_") or not hasattr(self, "rbf_sampler_"):
            raise RuntimeError("RKSFeatureMapper must be fitted before calling transform().")
        if X.ndim != 2 or X.shape[1] != self.input_dim_:
            raise ValueError(
                f"Expected 2D feature matrix with {self.input_dim_} columns, got shape {X.shape}."
            )
        X_scaled = self.scaler_.transform(X)
        Z = self.rbf_sampler_.transform(X_scaled)
        return np.asarray(Z, dtype=np.float64)

    def storage_bytes(self) -> int:
        """Return total byte footprint of fitted input StandardScaler + RBFSampler parameters."""
        if not hasattr(self, "scaler_") or not hasattr(self, "rbf_sampler_"):
            return 0
        total = 0
        for attr in ("mean_", "scale_", "var_"):
            arr = getattr(self.scaler_, attr, None)
            if isinstance(arr, np.ndarray):
                total += int(arr.nbytes)
        for attr in ("random_weights_", "random_offset_"):
            arr = getattr(self.rbf_sampler_, attr, None)
            if isinstance(arr, np.ndarray):
                total += int(arr.nbytes)
        return total
