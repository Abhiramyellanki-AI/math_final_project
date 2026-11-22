"""Participant-specific model search using complete recording-separated folds."""
from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .svd import SVDProjector

FAMILIES = ("rbf_baseline", "svd_rbf")


def make_pipeline(params: dict) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if params["family"] == "svd_rbf":
        steps.append(("svd", SVDProjector(params["rank"])))
    elif params["family"] != "rbf_baseline":
        raise ValueError("Unknown model family")
    steps.append(("svm", SVC(C=params["C"], gamma=params["gamma"], kernel="rbf",
                              cache_size=128)))
    return Pipeline(steps)


def valid_ranks(features: dict[int, np.ndarray], cfg: dict) -> list[int]:
    """Bound candidate ranks using each training fold only; no run 12 argument."""
    maximum = min(np.linalg.matrix_rank(StandardScaler().fit_transform(features[r]))
                  for r in (4, 8))
    return sorted({r for r in cfg["ranks"] if r <= maximum})


def select_models(features: dict[int, np.ndarray], labels: dict[int, np.ndarray],
                  cfg: dict) -> tuple[dict, list[dict]]:
    """Select both families on 4→8 and 8→4; reject any supplied test recording."""
    if set(features) != {4, 8} or set(labels) != {4, 8}:
        raise ValueError("Selection accepts exactly development runs 4 and 8")
    ranks = valid_ranks(features, cfg)
    if not ranks:
        raise ValueError("No valid SVD rank for this participant; do not drop silently")
    all_rows, best = [], {}
    for family in FAMILIES:
        candidates = []
        for rank, c, gamma in product(ranks if family == "svd_rbf" else [None],
                                      cfg["c_values"], cfg["gammas"]):
            params = dict(family=family, rank=rank, C=float(c), gamma=gamma)
            scores = []
            for train, val in ((4, 8), (8, 4)):
                model = make_pipeline(params)
                model.fit(features[train], labels[train])
                pred = model.predict(features[val])
                scores.append(dict(train_run=train, validation_run=val,
                                   accuracy=float(np.mean(pred == labels[val])),
                                   macro_f1=float(f1_score(labels[val], pred, average="macro",
                                                          labels=[0, 1], zero_division=0)),
                                   balanced_accuracy=float(balanced_accuracy_score(labels[val], pred))))
            row = dict(params=params, folds=scores,
                       mean_macro_f1=float(np.mean([s["macro_f1"] for s in scores])),
                       mean_balanced_accuracy=float(np.mean([s["balanced_accuracy"] for s in scores])),
                       mean_accuracy=float(np.mean([s["accuracy"] for s in scores])),
                       accuracy_fold_population_sd=float(np.std([s["accuracy"] for s in scores])),
                       invalid_ranks=sorted(set(cfg["ranks"]) - set(ranks)))
            candidates.append(row)
        # Exact ties only. Gamma ordering is the fixed config order.
        candidates.sort(key=lambda r: (-r["mean_macro_f1"], -r["mean_balanced_accuracy"],
                                      r["params"]["rank"] or 0, r["params"]["C"],
                                      cfg["gammas"].index(r["params"]["gamma"])))
        best[family] = candidates[0]
        all_rows.extend(candidates)
    return best, all_rows
