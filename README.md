# P3 EEG Motor-Imagery Classification

**HOSVD + SVM vs. Random Kitchen Sinks (RKS) + Linear SVM with Intra-Trial Observable Markov Chains**

> **Status:** All Milestones **M0 – M6 Complete**. Full experimental report and Viva Defense Q&A available in [REPORT_AND_VIVA_GUIDE.md](file:///c:/Users/Abhiram/OneDrive/Documents/sem3/MIS-3/END%20SEM%20PROJECT/REPORT_AND_VIVA_GUIDE.md).

---

## 1. Project Overview

This repository implements an offline, reproducible EEG research pipeline on the **PhysioNet EEG Motor Movement/Imagery Dataset (`eegmmidb` v1.0.0)** to classify **imagined left-fist (`T1 -> 0`) vs. imagined right-fist (`T2 -> 1`) movement** on **Subject 1, Runs 4, 8, and 12** (excluding rest `T0`).

We compare four feature-representation paradigms paired with Support Vector Machines under an identical, strictly leakage-free cross-run evaluation protocol:
1. **Deterministic Spectral Baselines (`M2`):** Per-channel $\mu$ (`8–13 Hz`) and $\beta$ (`13–30 Hz`) Welch log-bandpower (`128` features/trial) $\rightarrow$ train-only `StandardScaler` $\rightarrow$ `LinearSVC` and exact RBF `SVC`.
2. **HOSVD + Linear SVM (`M3`):** Multilinear spatial ($U_c \in \mathbb{R}^{64 \times r_c}$) and temporal ($U_t \in \mathbb{R}^{320 \times r_t}$) singular subspaces fitted **on training trials only**, projecting each trial $X_i \in \mathbb{R}^{64 \times 320}$ into $Z_i = U_c^T X_i U_t \in \mathbb{R}^{r_c \times r_t}$ (flattened to $r_c \cdot r_t$ features) $\rightarrow$ `StandardScaler` $\rightarrow$ `LinearSVC`.
3. **Random Kitchen Sinks / Random Fourier Features + Linear SVM (`M4`):** Shared `128`-D log-bandpower features $\rightarrow$ train-only `StandardScaler` $\rightarrow$ `RBFSampler(n_components=D, gamma=gamma, random_state=42)` $\rightarrow$ `LinearSVC`.
4. **Intra-Trial Observable Markov Chain + Linear SVM (`M4b`):** First-order discrete-time Markov chain modeling state transitions across $M = 15$ sequential sub-windows (`40` samples = `250 ms` window, `20` samples = `125 ms` stride) *within* each 2.0-second trial (fitted strictly on training folds with Laplace smoothing $\alpha = 0.5$), evaluated standalone (`K^2 + K + 3` features/trial) and fused with HOSVD and RKS.

---

## 2. Non-Negotiable Data-Leakage & Scientific Guardrails

- **Held-Out Split:** Runs **4 and 8** (`15` trials/run, `30` total) are development runs evaluated via run-swapped cross-validation (Fold 0: Train Run 4 $\to$ Validate Run 8; Fold 1: Train Run 8 $\to$ Validate Run 4). Run **12** (`15` trials: `7` Left, `8` Right) is strictly held out and evaluated **once** in Milestone M5 after locking configurations.
- **Train-Fold-Only Fitting:** Every learned transformation (`StandardScaler`, HOSVD bases $U_c, U_t$, RKS random Fourier weights, Markov sub-window quantizer and class-conditional transition matrices $P^{(0)}, P^{(1)}$, and SVM classifiers) is fitted **strictly within the training fold**.
- **Out-of-Sample HOSVD:** Every trial produces one fixed-length feature vector. The trial-mode factor of a full tensor is never used as an out-of-sample feature extractor.
- **Independent Run Filtering:** Each EDF recording is filtered independently at `8–30 Hz` (offline zero-phase FIR `firwin`) before epoching `[0.5, 2.5)` seconds post-cue (`320` samples at `160 Hz`).

---

## 3. Summary of Locked Results Across All 8 Faculty Evaluation Measures (`M5`)

| Model Family | Locked Config (from Runs 4↔8 CV) | Dev CV Acc | **Held-Out Test Acc (Run 12)** | **Test Macro F1** | Features ($p$) | Compression vs. Raw | HOSVD Test Recon Error | **SVM Train Time (ms)** | **Per-Trial Predict Time (ms)** | **Deployed Storage (Bytes)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline LinearSVC** | `LinearSVC (C=0.1)` | `50.00%` | **`53.33%`** | `0.5333` | `128` | `160.0x` | N/A | `4.85` | `1.083` | `4,104` |
| **2. Baseline RBF SVC** | `RBF SVC (C=10, γ=0.01)` | `70.00%` | **`73.33%`** | **`0.7222`** | `128` | `160.0x` | N/A | `3.52` | `1.340` | `33,008` |
| **3. HOSVD + LinearSVC** | `HOSVD (rc=16, rt=40, C=0.1)` | `73.33%` | **`53.33%`** | `0.5249` | `640` | `32.0x` | `0.6777` | `3.79` | **`0.084`** | `131,080` |
| **4. RKS + LinearSVC** | `RKS (D=128, γ=0.1, C=10)` | `80.00%` | **`53.33%`** | `0.5249` | `128` | `160.0x` | N/A | **`1.56`** | `0.944` | `136,200` |
| **5. Standalone Markov** | `Markov (K=3, C=1.0)` | `66.67%` | **`53.33%`** | `0.5333` | **`15`** | **`1,365.3x`** | N/A | `1.93` | `0.534` | **`2,888`** |
| **6. HOSVD + Markov** | `HOSVD(16,40) + Markov(K=4), C=0.1` | `76.67%` | **`53.33%`** | `0.5333` | `663` | `30.89x` | `0.6777` | `4.20` | `0.721` | `134,376` |
| **7. RKS + Markov** | `RKS(D=256,γ=0.01) + Markov(K=3), C=0.1` | **`83.33%`** | **`53.33%`** | `0.5249` | `271` | `75.57x` | N/A | `3.65` | `1.635` | `278,344` |

See [REPORT_AND_VIVA_GUIDE.md](file:///c:/Users/Abhiram/OneDrive/Documents/sem3/MIS-3/END%20SEM%20PROJECT/REPORT_AND_VIVA_GUIDE.md) for the complete analysis, methodology diagram, limitations, and Top-10 Viva Q&A guide.

---

## 4. Repository Structure

```text
.
├── PRD.md                                  # Product Requirements Document (with Markov amendment)
├── AGENTS.md                               # Non-negotiable engineering & leakage guardrails
├── CODEX_WORKFLOW.md                       # Milestone-by-milestone workflow guide
├── REPORT_AND_VIVA_GUIDE.md                # M6 Faculty report, pipeline diagram & Viva Q&A guide
├── README.md                               # Setup and execution documentation
├── pyproject.toml                          # Package metadata, dependencies, pytest & ruff config
├── configs/
│   └── subject01.yaml                      # Subject 1 experiment configuration & dataset path
├── src/eeg_p3/
│   ├── __init__.py
│   ├── config.py                           # YAML config loader and leakage-rule validator
│   ├── cli.py                              # CLI entry point (M0-M6 subcommands)
│   ├── data.py                             # M1: Local EDF loader & annotation verifier
│   ├── preprocessing.py                    # M1: Per-run 8-30 Hz band-pass filtering & epoching
│   ├── features.py                         # M2: Deterministic mu/beta log-bandpower extractor
│   ├── hosvd.py                            # M3: Train-only HOSVD factor learning & projection
│   ├── rks.py                              # M4: Random Kitchen Sinks (RBFSampler) mapper
│   ├── markov.py                           # M4b: Intra-trial Markov chain transition extractor
│   ├── models.py                           # M2-M5: Classifier pipelines and parameter grids
│   ├── evaluation.py                       # M2-M5: Run-swapped CV, locked Run 12 eval, 8 metrics
│   └── plotting.py                         # M1 & M6: EEG trace and publication figures
├── notebooks/
│   └── 01_explore_eeg_and_results.ipynb    # Interactive exploration and visual summary notebook
└── tests/
    ├── test_smoke.py                       # M0-M6 package, config, and CLI smoke tests
    ├── test_data_contract.py               # M1 shapes (N, 64, 320), labels {0,1}, T0 exclusion
    ├── test_features.py                    # M2 deterministic bandpower shapes & spectral selectivity
    ├── test_hosvd.py                       # M3 (C,T)->(r_c*r_t,), recon error, out-of-sample tests
    ├── test_rks.py                         # M4 output dimension D & fixed-seed determinism
    ├── test_markov.py                      # M4b row-stochastic sum=1, Laplace smoothing, boundaries
    └── test_no_leakage.py                  # M2-M5 strict invariance to held-out Run 12 corruption
```

---

## 5. Reproducible CLI Execution Commands (M0 – M6)

```powershell
$env:PYTHONPATH="src"

# M0: Inspect configuration & environment versions
python -m eeg_p3.cli info --config configs/subject01.yaml

# M1: Ingest Subject 1 Runs 4, 8, 12 and generate EEG waveform & trial plots
python -m eeg_p3.cli inspect-data --config configs/subject01.yaml
python -m eeg_p3.cli plot-eeg --config configs/subject01.yaml --run-id 4 --channel C3

# M2: Run deterministic mu/beta log-bandpower + LinearSVC and RBF SVC baselines on Runs 4 & 8
python -m eeg_p3.cli run-baselines --config configs/subject01.yaml

# M3: Run train-only HOSVD subspace projection & rank sweep on Runs 4 & 8
python -m eeg_p3.cli run-hosvd --config configs/subject01.yaml

# M4: Run Random Kitchen Sinks (RBFSampler + LinearSVC) sweep on Runs 4 & 8
python -m eeg_p3.cli run-rks --config configs/subject01.yaml

# M4b: Run Intra-Trial Observable Markov Chain & hybrid fusions on Runs 4 & 8
python -m eeg_p3.cli run-markov --config configs/subject01.yaml

# M5: Lock best configs from Runs 4 & 8, refit on Runs 4+8, evaluate once on held-out Run 12
python -m eeg_p3.cli evaluate-locked --config configs/subject01.yaml

# M6: Generate all publication-grade comparison figures and report manifest
python -m eeg_p3.cli build-report --config configs/subject01.yaml

# Run Full Automated Test Suite (31 tests)
python -m pytest -v
```
