"""Auditable development locks, exploratory follow-up evaluation, and metrics."""
from __future__ import annotations

import importlib.metadata
import os
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from threadpoolctl import threadpool_info

from .config import digest, file_hash, read_json, write_json
from .data import load_run, subject_records, verify_manifest
from .features import bandpower
from .models import FAMILIES, make_pipeline, select_models

DISCLOSURE = ("Exploratory follow-up: run 12 results were inspected in earlier work. "
              "Personalized models assess new recordings of known participants, not unseen users.")


def environment() -> dict:
    """Record actual runtime information without inventing hardware details."""
    packages = ["numpy", "scipy", "scikit-learn", "mne", "pandas", "joblib"]
    return dict(python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor() or "not reported by OS", cpu_count=os.cpu_count(),
                versions={p: importlib.metadata.version(p) for p in packages},
                threadpools=threadpool_info())


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Binary metrics with explicit left/right ordering and zero-division policy."""
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_precision=float(precision_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
                macro_recall=float(recall_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
                macro_f1=float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                per_class=classification_report(y, pred, labels=[0, 1],
                                               target_names=["left", "right"],
                                               output_dict=True, zero_division=0))


def mean_ci(values: list[float]) -> dict:
    """Student-t interval across independent participant units, not trials."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        raise ValueError("No participant metrics")
    mean = float(arr.mean())
    if len(arr) == 1:
        return dict(n=1, mean=mean, sd=None, ci95=None)
    sd = float(arr.std(ddof=1))
    half = float(t.ppf(.975, len(arr) - 1) * sd / np.sqrt(len(arr)))
    return dict(n=len(arr), mean=mean, sd=sd, ci95=[mean - half, mean + half])


def load_features(record: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray, list[dict], dict, np.ndarray]:
    X, y, trials, timing = load_run(Path(record["path"]), cfg, record["subject_id"], record["run_id"])
    start = perf_counter()
    F = bandpower(X, cfg)
    timing["bandpower_sec"] = perf_counter() - start
    return F, y, trials, timing, X


def develop(manifest: dict, out: Path) -> dict:
    """Fit/select only development data, then persist a tamper-evident selection record."""
    verify_manifest(manifest)
    cfg = manifest["config"]
    write_json(out / "manifest.json", manifest)
    selected, search = {}, []
    for subject in manifest["eligible_subjects"]:
        print(f"Development S{subject:03d}", flush=True)
        records = subject_records(manifest, subject)
        loaded = {run: load_features(records[run], cfg) for run in (4, 8)}
        best, rows = select_models({r: loaded[r][0] for r in loaded},
                                   {r: loaded[r][1] for r in loaded}, cfg)
        selected[str(subject)] = best
        search.extend(dict(subject_id=subject, **row) for row in rows)
    lock = dict(schema_version=1, config=cfg, manifest_sha256=digest(manifest),
                selections=selected, disclosure=DISCLOSURE, environment=environment(),
                selection_rule="mean macro F1; balanced accuracy; smaller rank; smaller C; configured gamma order")
    lock["lock_sha256"] = digest(lock)
    write_json(out / "search.json", {"rows": search})
    write_json(out / "lock.json", lock)
    return lock


def verify_lock(lock: dict, manifest: dict) -> None:
    payload = {k: v for k, v in lock.items() if k != "lock_sha256"}
    if digest(payload) != lock["lock_sha256"] or digest(manifest) != lock["manifest_sha256"]:
        raise ValueError("Selection lock or manifest was changed; refusing evaluation")
    if lock["config"] != manifest["config"]:
        raise ValueError("Configuration mismatch")
    if set(lock["selections"]) != {str(s) for s in manifest["eligible_subjects"]}:
        raise ValueError("Participant selections do not match manifest")


def repeated_fit(params: dict, F: np.ndarray, y: np.ndarray, repeats: int) -> tuple:
    """Warm up then time representation and SVM fitting independently."""
    make_pipeline(params).fit(F, y)
    representation, classifier = [], []
    for _ in range(repeats):
        model = make_pipeline(params)
        start = perf_counter()
        Z = model[:-1].fit_transform(F, y)
        representation.append(perf_counter() - start)
        start = perf_counter()
        model.named_steps["svm"].fit(Z, y)
        classifier.append(perf_counter() - start)
    return model, representation, classifier


def timing_summary(samples: list[float]) -> dict:
    return dict(median_sec=float(np.median(samples)), min_sec=float(np.min(samples)),
                max_sec=float(np.max(samples)), repeats=len(samples))


def aggregate(rows: list[dict], predictions: list[dict]) -> dict:
    """Pool predictions and compare paired participant scores, retaining every subject."""
    result = {}
    for family in (*FAMILIES, "majority_baseline"):
        subset = [r for r in rows if r["family"] == family]
        pred = [r for r in predictions if r["family"] == family]
        result[family] = dict(
            participant_metrics={key: mean_ci([r[key] for r in subset])
                                 for key in ("accuracy", "balanced_accuracy", "macro_f1",
                                             "macro_precision", "macro_recall")},
            pooled_metrics=metrics(np.array([r["label"] for r in pred]),
                                   np.array([r["prediction"] for r in pred])),
            total_serialized_model_bytes=sum(r.get("serialized_bytes", 0) for r in subset))
    by_key = {(r["subject_id"], r["family"]): r for r in rows}
    subjects = sorted({r["subject_id"] for r in rows})
    result["paired_svd_minus_baseline"] = {
        key: mean_ci([by_key[s, "svd_rbf"][key] - by_key[s, "rbf_baseline"][key]
                      for s in subjects]) for key in ("accuracy", "balanced_accuracy", "macro_f1")}
    return result


def evaluate(validation_dir: Path, out: Path) -> dict:
    """Explicit one-attempt follow-up evaluation; never called by report generation."""
    lock = read_json(validation_dir / "lock.json")
    manifest = read_json(validation_dir / "manifest.json")
    verify_lock(lock, manifest)
    verify_manifest(manifest)
    # This marker is deliberately not silently removed after failure. It is an audit
    # guard, not a cryptographic guarantee that humans cannot rerun experiments.
    with (validation_dir / "EVALUATION_STARTED.json").open("x", encoding="utf-8") as stream:
        stream.write('{"output_directory": ' + __import__("json").dumps(str(out.resolve())) + '}')
    cfg = lock["config"]
    (out / "models").mkdir()
    write_json(out / "lock.json", lock)
    write_json(out / "manifest.json", manifest)
    write_json(out / "status.json", {"status": "running", "disclosure": DISCLOSURE})
    rows, predictions, spectra = [], [], []
    try:
        for subject in manifest["eligible_subjects"]:
            print(f"Evaluation S{subject:03d}", flush=True)
            records = subject_records(manifest, subject)
            loaded = {r: load_features(records[r], cfg) for r in (4, 8, 12)}
            F = np.concatenate([loaded[r][0] for r in (4, 8)])
            y = np.concatenate([loaded[r][1] for r in (4, 8)])
            Ft, yt, provenance, load_time, Xt = loaded[12]
            io_timings = {str(r): {k: v for k, v in loaded[r][3].items() if k != "channels"}
                          for r in loaded}
            majority = int(np.argmax(np.bincount(y, minlength=2)))
            dummy = np.full(len(yt), majority)
            rows.append(dict(subject_id=subject, family="majority_baseline", **metrics(yt, dummy)))
            predictions.extend(dict(**p, family="majority_baseline", prediction=int(v))
                               for p, v in zip(provenance, dummy))
            for family in FAMILIES:
                selected = lock["selections"][str(subject)][family]
                model, fit_repr, fit_svm = repeated_fit(selected["params"], F, y, cfg["timing_repeats"])
                pred = model.predict(Ft)
                bp_times, transform_times, svm_times, totals = [], [], [], []
                # Warm up the full feature + classifier path before measurement.
                model.predict(bandpower(Xt, cfg))
                for _ in range(cfg["timing_repeats"]):
                    start = perf_counter()
                    bp = bandpower(Xt, cfg)
                    middle = perf_counter()
                    Z = model[:-1].transform(bp)
                    end_repr = perf_counter()
                    model.named_steps["svm"].predict(Z)
                    end = perf_counter()
                    bp_times.append(middle - start)
                    transform_times.append(end_repr - middle)
                    svm_times.append(end - end_repr)
                    totals.append(end - start)
                path = out / "models" / f"S{subject:03d}_{family}.joblib"
                joblib.dump(model, path, compress=0)
                arrays = sum(v.nbytes for step in model.named_steps.values()
                             for v in vars(step).values() if isinstance(v, np.ndarray))
                rank = selected["params"]["rank"] or F.shape[1]
                row = dict(subject_id=subject, family=family, params=selected["params"],
                           n_train=len(y), n_test=len(yt), features=rank,
                           feature_reduction_ratio=128 / rank, feature_bytes=rank * 8,
                           parameter_array_bytes=int(arrays), serialized_bytes=path.stat().st_size,
                           dev_macro_f1=selected["mean_macro_f1"],
                           timing=dict(io_and_preprocessing=io_timings,
                                       representation_fit=timing_summary(fit_repr),
                                       svm_fit=timing_summary(fit_svm),
                                       bandpower=timing_summary(bp_times),
                                       representation_transform=timing_summary(transform_times),
                                       svm_predict=timing_summary(svm_times),
                                       feature_plus_classifier_batch=timing_summary(totals),
                                       amortized_ms_per_trial=float(np.median(totals) * 1000 / len(yt))),
                           **metrics(yt, pred))
                if family == "svd_rbf":
                    svd = model.named_steps["svd"]
                    scale = model.named_steps["scale"]
                    row.update(train_reconstruction_error=svd.reconstruction_error(scale.transform(F)),
                               test_reconstruction_error=svd.reconstruction_error(scale.transform(Ft)),
                               retained_training_energy=svd.retained_energy_)
                    spectra.append(dict(subject_id=subject, rank=rank,
                                        singular_values=svd.singular_values_.tolist()))
                rows.append(row)
                predictions.extend(dict(**p, family=family, prediction=int(v))
                                   for p, v in zip(provenance, pred))
            # Save progress without presenting partial values as a complete cohort result.
            write_json(out / "progress.json", {"completed_subject": subject, "rows": rows})
        result = dict(schema_version=1, status="complete", disclosure=DISCLOSURE,
                      lock_sha256=lock["lock_sha256"], environment=environment(),
                      n_participants=len(manifest["eligible_subjects"]), rows=rows,
                      aggregate=aggregate(rows, predictions), spectra=spectra,
                      timing_scope="Repeated warm feature+classifier batches; excludes fixed filtering/loading. Amortized time is not measured single-trial latency.",
                      storage_scope="Parameter-array estimate and uncompressed joblib pipeline files, not process RAM.")
        pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
        pd.json_normalize(rows).to_csv(out / "participant_metrics.csv", index=False)
        write_json(out / "results.json", result)
        hashes = {p.name: file_hash(p) for p in (out / "predictions.csv", out / "results.json",
                                                out / "participant_metrics.csv")}
        write_json(out / "status.json", {"status": "complete", "sha256": hashes})
        return result
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "error": str(exc),
                                          "disclosure": DISCLOSURE})
        raise
