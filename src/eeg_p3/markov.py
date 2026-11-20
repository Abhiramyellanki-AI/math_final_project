"""Intra-trial first-order discrete-time observable Markov chain modeling (Milestone M4b).

Implements Reviewed Option A from PRD Section 5.6b:
- True observable first-order discrete-time Markov chain (NOT a Hidden Markov Model).
- Strict intra-trial sequence boundaries: each 2.0 s (320-sample) trial is divided into
  M sequential sub-windows (default: 40-sample / 250 ms window, 20-sample / 125 ms stride ->
  M = 15 sub-windows and 14 intra-trial transitions per trial).
- Never computes transitions across separate trials or recording runs, and never uses
  the experimenter's cue sequence (T1/T2 order).
- Observation-to-state mapping (fitted strictly on training trials):
  1. Learns top r_sub (default 4) orthonormal spatial channel basis vectors U_c_sub from X_train.
  2. In each sub-window m of trial i, computes the log-variance (instantaneous log band-energy)
     of the spatially projected sub-window plus sensorimotor C3/C4/Cz channels.
  3. Standardizes sub-window observation vectors on the training fold and clusters them into
     K discrete observable states {0, ..., K-1} via deterministic KMeans(n_clusters=K, random_state=seed).
- Transition probability estimation:
  1. Per-trial Laplace-smoothed empirical transition matrix P_hat_i of shape (K, K) and
     stationary state occupancy vector pi_i of shape (K,).
  2. Class-conditional training Markov models M_0 = (pi^(0), P^(0)) and M_1 = (pi^(1), P^(1))
     fitted strictly on training trials with Laplace smoothing alpha > 0.
- Unseen trial processing:
  Assigns sub-windows to states {0, ..., K-1} using frozen training quantizer, evaluates
  flattened P_hat_i (K^2), pi_i (K), and class log-likelihood scores [L_0, L_1, L_1 - L_0] (3),
  yielding exactly ONE fixed-length feature vector of dimension (K^2 + K + 3) per trial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from eeg_p3.hosvd import fit_channel_hosvd_basis


@dataclass(frozen=True)
class ClassConditionalMarkovModel:
    """First-order discrete-time Markov chain parameters (pi, P) for one class."""

    n_states: int
    laplace_alpha: float
    initial_probs: np.ndarray  # Shape: (K,), sums to 1.0
    transition_matrix: np.ndarray  # Shape: (K, K), each row sums to 1.0


def slice_trial_subwindows(
    X: np.ndarray,
    subwindow_samples: int = 40,
    subwindow_stride: int = 20,
) -> np.ndarray:
    """Slice 3D trial tensor (N, C, T) into intra-trial sub-windows (N, M, C, L).

    Sequence boundaries strictly reset per trial: no sub-window or transition ever
    crosses from trial i to trial i+1.

    Args:
        X: Trial tensor of shape (N, C, T).
        subwindow_samples: Length L of each sub-window in samples (default 40 = 250 ms @ 160 Hz).
        subwindow_stride: Step size between consecutive sub-windows (default 20 = 125 ms).

    Returns:
        4D array of shape (N, M, C, L) where M = 1 + (T - L) // stride.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D trial tensor (N, C, T), got shape {X.shape}.")
    n_trials, n_channels, n_times = X.shape
    if subwindow_samples <= 0 or subwindow_samples > n_times:
        raise ValueError(
            f"Invalid subwindow_samples={subwindow_samples} for n_times={n_times}."
        )
    if subwindow_stride <= 0:
        raise ValueError(f"subwindow_stride must be positive, got {subwindow_stride}.")

    starts = list(range(0, n_times - subwindow_samples + 1, subwindow_stride))
    if not starts:
        raise ValueError("No sub-windows could be extracted with given window/stride settings.")

    subwindows = np.stack(
        [X[:, :, s : s + subwindow_samples] for s in starts],
        axis=1,
    )  # Shape: (N, M, C, L)
    return subwindows


def extract_subwindow_observations(
    X: np.ndarray,
    U_c_sub: np.ndarray,
    subwindow_samples: int = 40,
    subwindow_stride: int = 20,
    log_epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute sub-window spatial-spectral log-energy observations of shape (N, M, d_obs).

    Projects each (C, L) sub-window onto the training spatial basis U_c_sub (C, r_sub)
    and computes log10 variance across the L samples in the sub-window.
    """
    subwindows = slice_trial_subwindows(
        X, subwindow_samples=subwindow_samples, subwindow_stride=subwindow_stride
    )  # (N, M, C, L)
    # Project onto training spatial basis: (N, M, r_sub, L)
    projected = np.einsum("cr,nmcl->nmrl", U_c_sub, subwindows, optimize=True)
    var_sub = np.var(projected, axis=-1)  # (N, M, r_sub)
    log_energy = np.log10(np.maximum(var_sub, 0.0) + log_epsilon)
    return log_energy


def estimate_empirical_transition_matrix(
    state_sequence: np.ndarray,
    n_states: int,
    laplace_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Laplace-smoothed (K, K) transition matrix and (K,) occupancy for one sequence.

    Args:
        state_sequence: 1D integer array of length M with values in {0, ..., K-1}.
        n_states: Number of discrete states K.
        laplace_alpha: Positive Laplace (Dirichlet) smoothing constant (> 0).

    Returns:
        Tuple (P_hat, pi_hat) where P_hat has shape (K, K) with row sums equal to 1.0,
        and pi_hat has shape (K,) with sum equal to 1.0.
    """
    if laplace_alpha <= 0.0:
        raise ValueError(f"laplace_alpha must be strictly positive, got {laplace_alpha}.")
    seq = np.asarray(state_sequence, dtype=np.int64)
    if seq.ndim != 1 or len(seq) < 2:
        raise ValueError(f"Expected 1D state sequence of length >= 2, got shape {seq.shape}.")
    if np.any(seq < 0) or np.any(seq >= n_states):
        raise ValueError(f"State sequence contains out-of-range states for K={n_states}.")

    counts = np.zeros((n_states, n_states), dtype=np.float64)
    np.add.at(counts, (seq[:-1], seq[1:]), 1.0)

    smoothed_counts = counts + float(laplace_alpha)
    row_sums = smoothed_counts.sum(axis=1, keepdims=True)
    P_hat = smoothed_counts / row_sums

    occ_counts = np.bincount(seq, minlength=n_states).astype(np.float64)
    pi_hat = (occ_counts + float(laplace_alpha)) / (len(seq) + n_states * float(laplace_alpha))

    return P_hat, pi_hat


def fit_class_conditional_markov_model(
    sequences: np.ndarray,
    n_states: int,
    laplace_alpha: float = 0.5,
) -> ClassConditionalMarkovModel:
    """Estimate a pooled first-order Markov chain (pi, P) from training sequences of one class.

    Transitions are summed ONLY within each trial sequence (never across trials).
    """
    if sequences.ndim != 2 or sequences.shape[1] < 2:
        raise ValueError(f"Expected 2D sequences array (N_class, M>=2), got {sequences.shape}.")
    if laplace_alpha <= 0.0:
        raise ValueError(f"laplace_alpha must be positive, got {laplace_alpha}.")

    init_counts = np.bincount(sequences[:, 0], minlength=n_states).astype(np.float64)
    initial_probs = (init_counts + float(laplace_alpha)) / (
        sequences.shape[0] + n_states * float(laplace_alpha)
    )

    trans_counts = np.zeros((n_states, n_states), dtype=np.float64)
    for seq in sequences:
        np.add.at(trans_counts, (seq[:-1], seq[1:]), 1.0)

    smoothed = trans_counts + float(laplace_alpha)
    transition_matrix = smoothed / smoothed.sum(axis=1, keepdims=True)

    return ClassConditionalMarkovModel(
        n_states=int(n_states),
        laplace_alpha=float(laplace_alpha),
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
    )


def compute_sequence_log_likelihood(
    state_sequence: np.ndarray,
    model: ClassConditionalMarkovModel,
) -> float:
    """Evaluate log P(s_1, ..., s_M | model) = log pi(s_1) + sum_{m=1}^{M-1} log P(s_{m+1} | s_m)."""
    seq = np.asarray(state_sequence, dtype=np.int64)
    log_init = float(np.log(model.initial_probs[seq[0]]))
    log_trans = float(np.sum(np.log(model.transition_matrix[seq[:-1], seq[1:]])))
    return log_init + log_trans


def compute_markov_storage_accounting(
    n_states: int,
    n_channels: int = 64,
    spatial_rank_sub: int = 4,
    raw_trial_elements: int = 20480,
    bytes_per_float: int = 8,
) -> dict[str, Any]:
    """Compute feature count, compression ratio, and byte storage for the Markov extractor."""
    n_features = int(n_states * n_states + n_states + 3)
    # Deployed Markov extractor stores:
    # U_c_sub (C * r_sub) + subwindow scaler (2 * r_sub) + KMeans centroids (K * r_sub)
    # + 2 class transition matrices (2 * K * K) + 2 initial prob vectors (2 * K)
    extractor_elements = int(
        n_channels * spatial_rank_sub
        + 2 * spatial_rank_sub
        + n_states * spatial_rank_sub
        + 2 * (n_states * n_states + n_states)
    )
    return {
        "n_states": int(n_states),
        "n_features": n_features,
        "compression_ratio_vs_raw": float(raw_trial_elements / n_features),
        "feature_bytes_per_trial": int(n_features * bytes_per_float),
        "markov_extractor_elements": extractor_elements,
        "markov_extractor_bytes": int(extractor_elements * bytes_per_float),
    }


class IntraTrialMarkovFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer extracting intra-trial first-order Markov chain features.

    For each trial i, outputs a fixed-length vector of dimension (K^2 + K + 3):
    - K^2 flattened Laplace-smoothed intra-trial transition probabilities P_hat_i(b | a)
    - K smoothed intra-trial state occupancy probabilities pi_hat_i(a)
    - 3 class-conditional generative scores: [log P(s | M_0), log P(s | M_1), LLR = L_1 - L_0]
    """

    def __init__(
        self,
        n_states: int = 4,
        subwindow_samples: int = 40,
        subwindow_stride: int = 20,
        spatial_rank_sub: int = 4,
        laplace_alpha: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.subwindow_samples = subwindow_samples
        self.subwindow_stride = subwindow_stride
        self.spatial_rank_sub = spatial_rank_sub
        self.laplace_alpha = laplace_alpha
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> IntraTrialMarkovFeatureExtractor:
        """Fit spatial sub-basis, sub-window quantizer, and class-conditional Markov chains on X, y."""
        if X.ndim != 3:
            raise ValueError(f"Expected 3D training tensor (N_train, C, T), got {X.shape}.")
        y_arr = np.asarray(y, dtype=np.int64)
        if len(y_arr) != X.shape[0]:
            raise ValueError(f"Length of y ({len(y_arr)}) does not match X.shape[0] ({X.shape[0]}).")

        # 1. Learn spatial projection basis strictly from training tensor X
        self.U_c_sub_ = fit_channel_hosvd_basis(X, r_c=int(self.spatial_rank_sub))  # (C, r_sub)

        # 2. Extract training sub-window observations (N_train, M, r_sub)
        obs = extract_subwindow_observations(
            X,
            U_c_sub=self.U_c_sub_,
            subwindow_samples=int(self.subwindow_samples),
            subwindow_stride=int(self.subwindow_stride),
        )
        n_train, n_steps, d_obs = obs.shape
        self.n_steps_per_trial_ = int(n_steps)

        # 3. Fit sub-window StandardScaler and KMeans state quantizer strictly on training sub-windows
        obs_flat = obs.reshape(n_train * n_steps, d_obs)
        self.subwindow_scaler_ = StandardScaler()
        obs_scaled = self.subwindow_scaler_.fit_transform(obs_flat)

        self.kmeans_ = KMeans(
            n_clusters=int(self.n_states),
            random_state=int(self.random_state),
            n_init=10,
        )
        train_states_flat = self.kmeans_.fit_predict(obs_scaled)
        train_sequences = train_states_flat.reshape(n_train, n_steps)

        # 4. Estimate class-conditional Markov chains M_0 (left) and M_1 (right) on training fold
        seq_class0 = train_sequences[y_arr == 0]
        seq_class1 = train_sequences[y_arr == 1]
        if len(seq_class0) == 0 or len(seq_class1) == 0:
            raise ValueError("Training fold must contain both Class 0 and Class 1 trials.")

        self.model_class0_ = fit_class_conditional_markov_model(
            seq_class0, n_states=int(self.n_states), laplace_alpha=float(self.laplace_alpha)
        )
        self.model_class1_ = fit_class_conditional_markov_model(
            seq_class1, n_states=int(self.n_states), laplace_alpha=float(self.laplace_alpha)
        )

        K = int(self.n_states)
        self.n_features_out_ = K * K + K + 3
        return self

    def transform_to_state_sequences(self, X: np.ndarray) -> np.ndarray:
        """Map each trial in X (N, C, T) to its observed intra-trial state sequence (N, M)."""
        if not hasattr(self, "kmeans_"):
            raise RuntimeError("IntraTrialMarkovFeatureExtractor must be fitted before transform.")
        obs = extract_subwindow_observations(
            X,
            U_c_sub=self.U_c_sub_,
            subwindow_samples=int(self.subwindow_samples),
            subwindow_stride=int(self.subwindow_stride),
        )
        n_trials, n_steps, d_obs = obs.shape
        obs_scaled = self.subwindow_scaler_.transform(obs.reshape(n_trials * n_steps, d_obs))
        states_flat = self.kmeans_.predict(obs_scaled)
        return states_flat.reshape(n_trials, n_steps)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Convert each trial in X (N, C, T) into a fixed-length (K^2 + K + 3) Markov feature vector."""
        sequences = self.transform_to_state_sequences(X)
        n_trials = sequences.shape[0]
        K = int(self.n_states)
        alpha = float(self.laplace_alpha)

        features = np.zeros((n_trials, self.n_features_out_), dtype=np.float64)
        for i in range(n_trials):
            seq = sequences[i]
            P_hat, pi_hat = estimate_empirical_transition_matrix(
                seq, n_states=K, laplace_alpha=alpha
            )
            ll0 = compute_sequence_log_likelihood(seq, self.model_class0_)
            ll1 = compute_sequence_log_likelihood(seq, self.model_class1_)
            llr = ll1 - ll0

            features[i, : K * K] = P_hat.reshape(-1)
            features[i, K * K : K * K + K] = pi_hat
            features[i, K * K + K :] = [ll0, ll1, llr]

        return features
