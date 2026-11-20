# Product Requirements Document (PRD)

## Project
**EEG Motor-Imagery Classification: HOSVD + SVM versus Random Kitchen Sinks + Linear SVM**

**Version:** 1.0 — initial student research prototype  
**Status:** Proposed; no experimental results have been obtained  
**Faculty methodology:** P3 — compare HOSVD + SVM and RKS + SVM  
**RKS:** Random Kitchen Sinks / Random Fourier Features (RFF)

## 1. Problem and goal
Build a reproducible offline research pipeline that predicts **left-hand vs right-hand imagined movement** from labeled EEG recordings and compares two feature-representation approaches:

1. **HOSVD-derived EEG features → SVM**
2. **Random Fourier Features (RKS) → Linear SVM**
3. **Intra-Trial Observable Markov Chain Transition Features → Linear SVM (standalone and fused with HOSVD / RKS)**

Measure the 8 required faculty evaluation metrics: **classification accuracy**, **precision / recall / F1**, **SVM training time**, **prediction time**, **number of features**, **compression ratio**, **HOSVD reconstruction error**, and **memory / storage requirement**. Include simple SVM baselines. **Do not claim BCI device control, Neuralink equivalence, clinical capability, superior accuracy, or real-time performance.**

### Research questions
- Does a low-rank, tensor-aware representation preserve motor-imagery discrimination?
- How do HOSVD mode ranks affect accuracy, reconstruction error, and feature size?
- How do RKS dimensionality and RBF bandwidth affect classification and compute?
- Which approach provides the better quality–compute trade-off under the **same evaluation split**?

## 2. Scope

### In scope (MVP)
- Python application / command-line research project with a clean notebook or report for plots.
- PhysioNet EEG Motor Movement/Imagery dataset through MNE (`mne.datasets.eegbci`).
- Initial **Subject 1**, **imagined left/right fist** runs **4, 8, 12** only; the dataset's `T1` and `T2` label meanings are interpreted specifically for these runs. `T0` is rest and excluded.
- 64-channel scalp EEG as available in selected files; verify channel names, sample rate, annotations, and event counts in code.
- Initial 8–30 Hz band-pass filtering and fixed 0.5–2.5 s post-cue epochs (2.0 s window; inspect/verify exact sample counts at the measured sampling frequency).
- Train on runs 4 and 8; reserve run 12 for final testing. Validate candidate hyperparameters using **run-aware, training-only** methods; disclose that only two training runs makes estimates unstable.
- Baseline: linear SVM on simple EEG features, and RBF-kernel SVM on the same baseline features.
- HOSVD: train-only channel/time factor learning on EEG trial tensor, train/test projections into equal-size trial features, optional StandardScaler, Linear SVM initially. Sweep valid mode ranks.
- RKS: same defined baseline features → fit StandardScaler on train only → RBFSampler → LinearSVC. Sweep random-feature dimension, gamma, and C.
- Precision, recall, F1 (macro and per class), accuracy, confusion matrix, class balance, feature dimensions, fitting and inference runtime, and approximate representation storage.
- Reproducible deterministic random seeds where applicable, saved configuration, software versions, and CSV/JSON metrics.
- Automated tests for dimensions, labels, feature alignment, train-only fitting, and reproducibility of deterministic components.

### Out of scope (initial version)
- Live EEG streaming or controlling an external device.
- Medical diagnosis, therapy, intention decoding beyond the prescribed experiment labels.
- Deep neural networks, online adaptive algorithms, cloud deployment, graphical dashboard, subject-independent BCI claims.
- Any invented or synthetic numbers in final experimental tables; synthetic data permitted **only in clearly labeled unit tests**.

## 3. Users and outputs
**Primary user:** student researcher preparing a faculty review and viva.  
**Deliverables:** Python repository, `README.md`, environment dependency file, CLI scripts/config, one or more educational notebooks, plots, CSV/JSON metrics, limitations section, and presentation-ready pipeline diagram.

## 4. Data contract
- `X`: floating-point array of shape `(n_trials, n_channels, n_times)`.
- `y`: integer labels `(n_trials,)`, `0=left imagery`, `1=right imagery`.
- `groups`: recording run IDs `(n_trials,)` for leakage-safe evaluation.
- Preserve subject ID and metadata per trial.
- Keep training runs separate before fitting transforms. Do not concatenate time-domain raw runs before filtering.
- Annotation extraction from each run: `T1→0`, `T2→1`; ignore `T0`; verify via tests, counts, and dataset documentation.
- Event timing and units are documented; MNE's `get_data()` yields volts by default, so any conversion to microvolts must be explicit and consistent.
- For a 160 Hz recording, a 2-second window should contain 320 time samples; assert the actual shape.
- Cache paths and artifacts are configurable; **do not commit large downloaded EEG files**.

## 5. Processing and evaluation design

### 5.1 Data ingestion
- Prefer pre-downloaded local EDF files from `C:\Users\Abhiram\Downloads\eegmmidb-1.0.0\files\` when available; fall back to `eegbci.load_data` only if local files are missing and download is explicitly enabled. Read each run independently.
- Standardize channel naming and identify actual EEG channels.
- Inspect `sfreq`, channel names, duration, annotation classes, and per-run trial counts.
- Plot an example raw and filtered channel trace, plus one labeled trial.

### 5.2 Fixed signal processing
- Apply a documented 8–30 Hz band-pass filter to each independent raw run.
- Extract a fixed post-cue window, initially `[0.5, 2.5)` seconds with exact sample-count handling; skip rest events.
- Use no fitted global scaler before splitting. Record filter type and parameters.
- Do not tune window/filter based on held-out run 12.
- Explicitly document that zero-phase offline filtering is **not a causal real-time processing claim**.

### 5.3 Train/validation/test
- **Final holdout:** train-development runs 4, 8; untouched test run 12.
- For exploratory tuning use grouped validation swapping run 4 and run 8 (train on one, validate on the other, then reverse). Compute mean validation score; note high variance due to only two runs. Alternatively use run-aware nested methods when expanding to additional data.
- Fit all learned transforms separately **within each validation fold**: feature scaling, HOSVD factors, RKS feature generator, and classifier.
- Select settings using only development runs, refit on both 4 and 8, evaluate on run 12 **once per locked comparison**.
- Do not repeatedly revise hyperparameters in response to test-run accuracy.

### 5.4 Baseline feature representation
- Use one shared, clearly documented deterministic input feature representation for baseline and RKS, e.g. log bandpower in mu (8–13 Hz) and beta (13–30 Hz) for each channel, resulting in 2×channels features.
- Compute per-epoch spectral features using fixed settings. Treat any fitted normalization as training-only.
- Add baseline LinearSVC and RBF `SVC` to contextualize results.

### 5.5 HOSVD + SVM
- Input: training tensor `(N_train,C,T)`.
- Estimate mode-specific **channel** and **time** singular subspaces from **training data alone** using matrix unfoldings; do not use test trials in factor calculation.
- Select valid ranks `r_c ≤ C` and `r_t ≤ T`; use a modest prespecified grid (e.g. `r_c∈{4,8,16}`, `r_t∈{10,20,40}`), subject to available data and runtime.
- For any trial `X_i` with shape `(C,T)`, project with `Z_i = U_c.T @ X_i @ U_t`, shape `(r_c,r_t)`, then flatten to `(r_c*r_t,)` and train a scaler + SVM.
- The full training tensor's **trial-mode factor is not a reusable representation for unseen trials**. Do not build classifier features from a trial-mode factor indexed by test observations.
- Optionally calculate relative Frobenius reconstruction error under channel/time projection, using training or held-out data only for evaluation—not for fitting factors.
- Report storage accounting correctly (core plus factor matrices for decomposition, or per-trial projected features plus reusable bases for the deployed feature extractor). Distinguish these two measures.

### 5.6 RKS + Linear SVM
- Input: the baseline feature vectors, with train-fitted scaling.
- `RBFSampler(gamma=..., n_components=D, random_state=...)` followed by `LinearSVC(C=..., max_iter=...)`.
- Initial grid: `D∈{128,256,512}`, gamma derived from a validated range such as `{0.001,0.01,0.1}` on standardized feature inputs, `C∈{0.1,1,10}`. Narrow or adapt grid if training folds are small.
- RKS is **kernel approximation, not guaranteed compression**; report output dimensions explicitly.

### 5.6b Intra-Trial Markov Chain + Linear SVM (Reviewed Amendment — Option A)
- Divide each 2.0-second (`320`-sample) trial into $M$ sequential sub-windows inside the trial; sequence boundaries reset strictly at the start and end of every trial (never across trials or runs, and never using experimental cue order).
- Extract sub-window spatial-spectral features and discretize into $K$ observable states $\{0, \dots, K-1\}$ using a quantizer fitted **strictly on the training fold**. Do not substitute a Hidden Markov Model (HMM).
- Estimate per-trial Laplace-smoothed ($\alpha > 0$) empirical transition probabilities $\hat{P}_i \in \mathbb{R}^{K \times K}$, state occupancy $\pi_i \in \mathbb{R}^K$, and class-conditional log-likelihood features $[\mathcal{L}_0, \mathcal{L}_1, \mathcal{L}_1 - \mathcal{L}_0]$ using class-conditional transition matrices $P^{(0)}, P^{(1)}$ fitted on training folds only.
- Output exactly one fixed-length feature vector of dimension $K^2 + K + 3$ per trial for `LinearSVC`. Evaluate both standalone (`Markov + LinearSVC`) and fused (`HOSVD + Markov + LinearSVC`, `RKS + Markov + LinearSVC`) while keeping the original HOSVD and RKS pipelines intact.

### 5.7 Comparability and limitations
- All models share the same labels, recording splits, evaluation metrics, and tracked machine/runtime environment.
- Baseline/RKS use bandpower; HOSVD may use raw epochs. Call this a **pipeline comparison**, not a controlled isolated comparison of mathematical transformations. For stronger claims, optionally add matched-input ablations.
- Fixed train/test split is within-subject, across runs, **not across subjects** (or Option A across multiple known participants: train on Runs 4 and 8 of eligible participants, test on Run 12 of those same eligible participants).
- Compare preprocessing/representation time and classifier-fit time **separately** and as end-to-end totals.
- Repeated stochastic seeds only where relevant; report variability if rerunning.

### 5.8 Confirmatory Multi-Participant Cohort Amendment (Option A)
- **Pilot Preservation:** Preserve `configs/subject01.yaml` and Subject 1 results as the original single-subject pilot experiment. Because Subject 1's Run 12 has already been inspected, exclude Subject 1 from the confirmatory cohort by default.
- **Candidate Cohort & Task Purity:** Consider pre-downloaded subjects `2–109` from `local_data_dir` (`allow_download_fallback: false`). Use only left-vs-right imagined fist movement runs (`Runs 4 and 8` for development, `Run 12` for final held-out testing; `T1 -> 0`, `T2 -> 1`, exclude `T0`). Never mix executed movement runs or fists-vs-feet imagery runs.
- **Pre-Evaluation Eligibility Audit:** Before any model evaluation, verify for each candidate subject across Runs 4, 8, and 12: file existence, MNE readability, 64 EEG channels in canonical standardized order, `160.0 Hz` sampling rate, valid `{T0, T1, T2}` annotations with both classes present, and valid `[0.5, 2.5)` s (`320`-sample) finite epochs. Log every exclusion with its objective quality reason; never exclude participants based on classifier accuracy.
- **Unique Provenance & Splits:** Preserve `subject_id`, `run_id`, `recording_id` (`S{sid:03d}R{rid:02d}`), `global_trial_id`, and chronological cue order for every trial. Use the same eligible participants in development (`Runs 4, 8`) and final test (`Run 12`), with development Fold 1 (`Train Run 4 -> Val Run 8` across participants) and Fold 2 (`Train Run 8 -> Val Run 4` across participants). Export a reproducible cohort manifest.

## 6. Acceptance criteria
1. One command prepares Subject 1 / runs 4, 8, 12 and prints data shapes, class counts, and channel/sample metadata.
2. Verified labels and run IDs are maintained for every trial; test run 12 is never used to fit or tune any transformation.
3. HOSVD projection unit test confirms `(C,T) → (r_c*r_t,)`, identical features for repeated deterministic inputs, and no dependence on held-out samples.
4. RKS unit test confirms output feature count `D` and fixed-seed reproducibility.
5. A baseline, HOSVD+SVM and RKS+LinearSVM successfully train and predict on a small real-data subset.
6. Final evaluation exports per-model accuracy, macro precision/recall/F1, confusion matrix, train/predict times, feature dimensions, and model/config metadata.
7. HOSVD rank sweep exports reconstruction error and retained dimensions; curves are based on measured data.
8. README explicitly documents dataset source, setup, precise runs/task, evaluation protocol, caveats, and commands.
9. `pytest` tests pass; no data files or secrets are committed.
10. Report contains no fabricated experimental results and does not present an EEG classifier as a Neuralink-like implanted interface.

## 7. Suggested repository structure
```text
p3-eeg-project/
├── PRD.md
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   └── subject01.yaml
├── src/eeg_p3/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── hosvd.py
│   ├── models.py
│   ├── evaluation.py
│   └── cli.py
├── notebooks/
│   └── 01_explore_eeg.ipynb
├── tests/
│   ├── test_data_contract.py
│   ├── test_hosvd.py
│   └── test_no_leakage.py
├── outputs/           # ignored generated outputs
└── data/              # ignored downloaded EDF files
```

## 8. Implementation milestones and definition of done
- **M0 — Scaffold:** package installs, minimal CLI, lint/test workflow, README and config; smoke test runs.
- **M1 — Data:** EDF ingest (local path priority), plots, events, epoch tensor and run IDs; labels and sample-count tests pass.
- **M2 — Baselines:** shared deterministic signal features; train-fold-only scaler; LinearSVC/RBF SVC; CSV output.
- **M3 — HOSVD:** train-only mode-factor fit, projection, rank sweep, reconstruction/error; synthetic and real tests.
- **M4 — RKS:** RBFSampler pipeline and feature/gamma/C sweep; tests.
- **M4b — Markov Chain:** train-only intra-trial sub-window quantization, Laplace-smoothed first-order Markov transition features and class log-likelihoods, standalone and hybrid HOSVD/RKS evaluations on Runs 4/8; tests.
- **M5 — Fair evaluation:** run-aware tuning on runs 4/8, lock model configurations, evaluate run 12 once; export reproducible metrics covering all 8 required faculty evaluation measures.
- **M6 — Research communication:** plots, limitations, reproducibility instructions, presentation-ready summary.

## 9. Risks and safeguards
- **Tiny number of runs:** validation variance and limited generalization. Clearly mark exploratory results and expand subjects/runs later.
- **Confounding from task cues:** choose epochs thoughtfully and discuss possible visual/cue artifacts.
- **Wrong event interpretation:** verify run-specific T1/T2 semantics against the source.
- **Leakage:** never fit scaling, decomposition, or model selection using test data.
- **Overclaiming:** avoid calling offline EEG classification mind-reading or implanted BCI.
- **Unfair comparison:** disclose differing representations and report matched-input baselines when feasible.

## 10. Official reference links
- PhysioNet dataset: https://physionet.org/content/eegmmidb/1.0.0/
- MNE EEGBCI loader: https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html
- scikit-learn RBFSampler: https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html
- scikit-learn pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
