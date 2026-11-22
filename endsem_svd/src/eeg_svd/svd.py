"""Explicit training-fitted right-singular-vector feature projection."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class SVDProjector(TransformerMixin, BaseEstimator):
    """Project standardized features onto a training-only right singular subspace.

    Centering is performed by the preceding StandardScaler, not repeated here.
    Singular vector signs may vary across LAPACK implementations; predictions and
    reconstruction are invariant to simultaneous coordinate sign changes.
    """

    def __init__(self, rank: int = 4):
        self.rank = rank

    def fit(self, X: np.ndarray, y=None) -> SVDProjector:
        X = check_array(X, dtype=np.float64)
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        tol = s[0] * max(X.shape) * np.finfo(float).eps if len(s) else 0.
        self.numerical_rank_ = int(np.sum(s > tol))
        if type(self.rank) is not int or not 1 <= self.rank <= self.numerical_rank_:
            raise ValueError(f"Rank {self.rank} exceeds numerical rank {self.numerical_rank_}")
        self.n_features_in_ = X.shape[1]
        self.components_ = vt[:self.rank].copy()
        self.singular_values_ = s.copy()
        self.retained_energy_ = float(np.sum(s[:self.rank] ** 2) / np.sum(s ** 2))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "components_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count changed")
        return X @ self.components_.T

    def reconstruction_error(self, X: np.ndarray) -> float:
        Z = self.transform(X)
        norm = np.linalg.norm(X)
        return float(np.linalg.norm(X - Z @ self.components_) / norm) if norm else 0.
