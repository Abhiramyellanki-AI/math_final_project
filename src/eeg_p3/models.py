"""Unified classifier pipeline builders and hyperparameter grids (Milestones M2-M4b)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from eeg_p3.config import ProjectConfig


@dataclass(frozen=True)
class BaselineModelSpec:
    """Specification for a single baseline classifier configuration (M2)."""

    model_family: str  # 'baseline_linear_svc' or 'baseline_rbf_svc'
    config_id: str
    params: dict[str, Any]
    random_seed: int
    max_iter: int


@dataclass(frozen=True)
class HOSVDModelSpec:
    """Specification for a single HOSVD + LinearSVC configuration (M3)."""

    model_family: str  # 'hosvd_linear_svc'
    config_id: str
    r_c: int
    r_t: int
    c_val: float
    params: dict[str, Any]
    random_seed: int
    max_iter: int


@dataclass(frozen=True)
class RKSModelSpec:
    """Specification for a single Random Kitchen Sinks (RBFSampler + LinearSVC) configuration (M4)."""

    model_family: str  # 'rks_linear_svc'
    config_id: str
    n_components: int
    gamma: float
    c_val: float
    params: dict[str, Any]
    random_seed: int
    max_iter: int


@dataclass(frozen=True)
class MarkovModelSpec:
    """Specification for a standalone or hybrid Intra-Trial Markov Chain + LinearSVC config (M4b)."""

    model_family: str  # 'markov_linear_svc', 'hosvd_markov_linear_svc', or 'rks_markov_linear_svc'
    config_id: str
    n_states: int
    subwindow_samples: int
    subwindow_stride: int
    laplace_alpha: float
    c_val: float
    fusion_mode: str  # 'standalone', 'hosvd_fusion', or 'rks_fusion'
    r_c: int | None
    r_t: int | None
    rks_dim: int | None
    rks_gamma: float | None
    params: dict[str, Any]
    random_seed: int
    max_iter: int


def build_baseline_pipeline(spec: BaselineModelSpec) -> Pipeline:
    """Build a leakage-safe (StandardScaler -> SVM) pipeline from a BaselineModelSpec."""
    if spec.model_family == "baseline_linear_svc":
        clf = LinearSVC(
            C=float(spec.params["C"]),
            max_iter=spec.max_iter,
            random_state=spec.random_seed,
            dual="auto",
        )
    elif spec.model_family == "baseline_rbf_svc":
        gamma_val = spec.params["gamma"]
        gamma_arg: str | float = str(gamma_val) if isinstance(gamma_val, str) else float(gamma_val)
        clf = SVC(
            kernel="rbf",
            C=float(spec.params["C"]),
            gamma=gamma_arg,
            max_iter=spec.max_iter,
            random_state=spec.random_seed,
        )
    else:
        raise ValueError(f"Unknown baseline model_family: '{spec.model_family}'")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def build_hosvd_classifier_pipeline(spec: HOSVDModelSpec) -> Pipeline:
    """Build a leakage-safe (StandardScaler -> LinearSVC) pipeline for projected HOSVD features."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    C=float(spec.c_val),
                    max_iter=spec.max_iter,
                    random_state=spec.random_seed,
                    dual="auto",
                ),
            ),
        ]
    )


def build_rks_classifier_(spec: RKSModelSpec) -> LinearSVC:
    """Build the LinearSVC classifier applied after train-fitted RKSFeatureMapper."""
    return LinearSVC(
        C=float(spec.c_val),
        max_iter=spec.max_iter,
        random_state=spec.random_seed,
        dual="auto",
    )


def build_markov_classifier_pipeline(spec: MarkovModelSpec) -> Pipeline:
    """Build a leakage-safe (StandardScaler -> LinearSVC) pipeline for Markov/hybrid features."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    C=float(spec.c_val),
                    max_iter=spec.max_iter,
                    random_state=spec.random_seed,
                    dual="auto",
                ),
            ),
        ]
    )


def get_baseline_model_specs(cfg: ProjectConfig) -> list[BaselineModelSpec]:
    """Enumerate all configured LinearSVC and RBF SVC baseline hyperparameter specifications."""
    base_cfg = cfg.raw.get("baselines", {})
    linear_c_grid = base_cfg.get("linear_svc_c_grid", [0.1, 1.0, 10.0])
    rbf_c_grid = base_cfg.get("rbf_svc_c_grid", [0.1, 1.0, 10.0])
    rbf_gamma_grid = base_cfg.get("rbf_svc_gamma_grid", ["scale", 0.01, 0.1])
    max_iter = int(base_cfg.get("max_iter", 5000))
    seed = cfg.random_seed

    specs: list[BaselineModelSpec] = []

    for c_val in linear_c_grid:
        c_float = float(c_val)
        specs.append(
            BaselineModelSpec(
                model_family="baseline_linear_svc",
                config_id=f"LinearSVC_C={c_float:g}",
                params={"C": c_float},
                random_seed=seed,
                max_iter=max_iter,
            )
        )

    for c_val in rbf_c_grid:
        c_float = float(c_val)
        for g_val in rbf_gamma_grid:
            g_parsed: str | float = str(g_val) if str(g_val) in ("scale", "auto") else float(g_val)
            specs.append(
                BaselineModelSpec(
                    model_family="baseline_rbf_svc",
                    config_id=f"RBF_SVC_C={c_float:g}_gamma={g_parsed}",
                    params={"C": c_float, "gamma": g_parsed},
                    random_seed=seed,
                    max_iter=max_iter,
                )
            )

    return specs


def get_hosvd_model_specs(cfg: ProjectConfig) -> list[HOSVDModelSpec]:
    """Enumerate all configured HOSVD rank pairs (r_c, r_t) and LinearSVC C grid values."""
    hosvd_cfg = cfg.raw.get("hosvd", {})
    base_cfg = cfg.raw.get("baselines", {})
    r_c_grid = [int(r) for r in hosvd_cfg.get("channel_ranks", [4, 8, 16])]
    r_t_grid = [int(r) for r in hosvd_cfg.get("time_ranks", [10, 20, 40])]
    c_grid = [float(c) for c in hosvd_cfg.get("linear_svc_c_grid", [0.1, 1.0, 10.0])]
    max_iter = int(base_cfg.get("max_iter", 5000))
    seed = cfg.random_seed

    specs: list[HOSVDModelSpec] = []
    for r_c in r_c_grid:
        for r_t in r_t_grid:
            for c_val in c_grid:
                specs.append(
                    HOSVDModelSpec(
                        model_family="hosvd_linear_svc",
                        config_id=f"HOSVD_rc={r_c}_rt={r_t}_C={c_val:g}",
                        r_c=r_c,
                        r_t=r_t,
                        c_val=c_val,
                        params={"r_c": r_c, "r_t": r_t, "C": c_val},
                        random_seed=seed,
                        max_iter=max_iter,
                    )
                )
    return specs


def get_rks_model_specs(cfg: ProjectConfig) -> list[RKSModelSpec]:
    """Enumerate all configured RKS (D, gamma, C) hyperparameter specifications (M4)."""
    rks_cfg = cfg.raw.get("rks", {})
    base_cfg = cfg.raw.get("baselines", {})
    d_grid = [int(d) for d in rks_cfg.get("n_components_grid", [128, 256, 512])]
    gamma_grid = [float(g) for g in rks_cfg.get("gamma_grid", [0.001, 0.01, 0.1])]
    c_grid = [float(c) for c in rks_cfg.get("linear_svc_c_grid", [0.1, 1.0, 10.0])]
    max_iter = int(base_cfg.get("max_iter", 5000))
    seed = cfg.random_seed

    specs: list[RKSModelSpec] = []
    for d_val in d_grid:
        for gamma_val in gamma_grid:
            for c_val in c_grid:
                specs.append(
                    RKSModelSpec(
                        model_family="rks_linear_svc",
                        config_id=f"RKS_D={d_val}_gamma={gamma_val:g}_C={c_val:g}",
                        n_components=d_val,
                        gamma=gamma_val,
                        c_val=c_val,
                        params={"n_components": d_val, "gamma": gamma_val, "C": c_val},
                        random_seed=seed,
                        max_iter=max_iter,
                    )
                )
    return specs


def get_markov_model_specs(cfg: ProjectConfig) -> list[MarkovModelSpec]:
    """Enumerate standalone and hybrid Markov chain model configurations (M4b)."""
    m_cfg = cfg.raw.get("markov", {})
    base_cfg = cfg.raw.get("baselines", {})
    k_grid = [int(k) for k in m_cfg.get("n_states_grid", [3, 4, 6])]
    sub_samples = int(m_cfg.get("subwindow_samples", 40))
    sub_stride = int(m_cfg.get("subwindow_stride", 20))
    alpha = float(m_cfg.get("laplace_alpha", 0.5))
    c_grid = [float(c) for c in m_cfg.get("linear_svc_c_grid", [0.1, 1.0, 10.0])]
    max_iter = int(base_cfg.get("max_iter", 5000))
    seed = cfg.random_seed

    specs: list[MarkovModelSpec] = []

    # 1. Standalone Intra-Trial Markov Chain + LinearSVC
    for k_val in k_grid:
        for c_val in c_grid:
            specs.append(
                MarkovModelSpec(
                    model_family="markov_linear_svc",
                    config_id=f"Markov_K={k_val}_C={c_val:g}",
                    n_states=k_val,
                    subwindow_samples=sub_samples,
                    subwindow_stride=sub_stride,
                    laplace_alpha=alpha,
                    c_val=c_val,
                    fusion_mode="standalone",
                    r_c=None,
                    r_t=None,
                    rks_dim=None,
                    rks_gamma=None,
                    params={
                        "n_states": k_val,
                        "subwindow_samples": sub_samples,
                        "subwindow_stride": sub_stride,
                        "laplace_alpha": alpha,
                        "C": c_val,
                        "fusion_mode": "standalone",
                    },
                    random_seed=seed,
                    max_iter=max_iter,
                )
            )

    # 2. HOSVD + Intra-Trial Markov Hybrid Fusion + LinearSVC
    for k_val in k_grid:
        for c_val in c_grid:
            specs.append(
                MarkovModelSpec(
                    model_family="hosvd_markov_linear_svc",
                    config_id=f"HOSVD_rc=16_rt=40+Markov_K={k_val}_C={c_val:g}",
                    n_states=k_val,
                    subwindow_samples=sub_samples,
                    subwindow_stride=sub_stride,
                    laplace_alpha=alpha,
                    c_val=c_val,
                    fusion_mode="hosvd_fusion",
                    r_c=16,
                    r_t=40,
                    rks_dim=None,
                    rks_gamma=None,
                    params={
                        "r_c": 16,
                        "r_t": 40,
                        "n_states": k_val,
                        "laplace_alpha": alpha,
                        "C": c_val,
                        "fusion_mode": "hosvd_fusion",
                    },
                    random_seed=seed,
                    max_iter=max_iter,
                )
            )

    # 3. RKS + Intra-Trial Markov Hybrid Fusion + LinearSVC
    for k_val in k_grid:
        for c_val in c_grid:
            specs.append(
                MarkovModelSpec(
                    model_family="rks_markov_linear_svc",
                    config_id=f"RKS_D=256_gamma=0.01+Markov_K={k_val}_C={c_val:g}",
                    n_states=k_val,
                    subwindow_samples=sub_samples,
                    subwindow_stride=sub_stride,
                    laplace_alpha=alpha,
                    c_val=c_val,
                    fusion_mode="rks_fusion",
                    r_c=None,
                    r_t=None,
                    rks_dim=256,
                    rks_gamma=0.01,
                    params={
                        "rks_dim": 256,
                        "rks_gamma": 0.01,
                        "n_states": k_val,
                        "laplace_alpha": alpha,
                        "C": c_val,
                        "fusion_mode": "rks_fusion",
                    },
                    random_seed=seed,
                    max_iter=max_iter,
                )
            )

    return specs
