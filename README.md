# P3 EEG Motor-Imagery Classification

**HOSVD + SVM vs. Random Kitchen Sinks (RKS) + Linear SVM with Intra-Trial Observable Markov Chains**

> **Status:** Single-Subject Pilot (`M0 – M6`) Complete & Preserved; Confirmatory Multi-Participant Cohort Expansion (**Option A**, Stages `E1 – E3`: `105` Eligible Participants, `315` Recordings, `4,723` Trials) Complete. Full pilot + cohort comparison report and Viva Q&A available in [REPORT_AND_VIVA_GUIDE.md](file:///c:/Users/Abhiram/OneDrive/Documents/sem3/MIS-3/END%20SEM%20PROJECT/REPORT_AND_VIVA_GUIDE.md).

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

# M6: Generate all publication-grade comparison figures and report manifest (requires M5 output)
python -m eeg_p3.cli build-report --config configs/subject01.yaml

# E1 (Stage 1 Cohort Expansion): Discover local dataset (Subjects 2-109, excluding pilot Subject 1),
# audit eligibility/exclusions, and export reproducible split manifest (no model training or test eval)
python -m eeg_p3.cli audit-cohort --config configs/cohort_expanded.yaml

# E2 (Stage 2 Cohort Expansion): Run multi-participant development validation on Runs 4 and 8 across
# all 105 eligible participants (pooled & participant-grouped metrics, 95% CIs, seed variability; Run 12 untouched)
python -m eeg_p3.cli validate-cohort --config configs/cohort_expanded.yaml

# E3 (Stage 3 Cohort Expansion): Explicitly lock best configs from E2 (Runs 4 & 8), refit on combined
# Runs 4+8 (3,148 trials across 105 participants), evaluate ONCE on held-out Run 12 (1,575 trials),
# and generate cohort publication figures
python -m eeg_p3.cli evaluate-locked --config configs/cohort_expanded.yaml
python -m eeg_p3.cli build-report --config configs/cohort_expanded.yaml

# Run Full Automated Test Suite (39 unit & integration tests)
python -m pytest -v
```

---

## 6. Confirmatory Multi-Participant Cohort Expansion (Option A — Stages `E1 – E3`)

To evaluate generalization to new recordings of known participants (**Option A**) without reusing pilot Subject 1 (whose Run 12 was already inspected in the pilot):
- **Configuration:** [configs/cohort_expanded.yaml](file:///c:/Users/Abhiram/OneDrive/Documents/sem3/MIS-3/END%20SEM%20PROJECT/configs/cohort_expanded.yaml) points to `C:/Users/Abhiram/Downloads/eegmmidb-1.0.0/files` with `allow_download_fallback: false`, `candidate_subject_range: [2, 109]`, and `excluded_subject_ids: [1]`.
- **Pre-Evaluation Eligibility Audit (`audit-cohort`):** Out of `109` total subjects (`1` excluded pilot + `108` candidates `2..109`):
  - **`105` Eligible Participants:** All subjects in `2..109` except `88, 92, 100`.
  - **`4` Excluded Participants (Logged Before Model Evaluation):**
    1. `Subject 1 (S001)`: `excluded_pilot_subject` (Run 12 already inspected in single-subject pilot).
    2. `Subject 88 (S088)`: `sfreq_mismatch` (`128.0 Hz` instead of required `160.0 Hz`).
    3. `Subject 92 (S092)`: `sfreq_mismatch` (`128.0 Hz` instead of required `160.0 Hz`).
    4. `Subject 100 (S100)`: `sfreq_mismatch` (`128.0 Hz` instead of required `160.0 Hz`).
- **Cohort Trial & Class Counts (`315` recordings, `4,723` total trials):**
  - **Development (`Runs 4 + 8`, `210` recordings):** **`3,148` trials** (`1,589` Left `T1=0`, `1,559` Right `T2=1`).
    - Fold 1: Train `Run 4` (`105` recordings, `1,575` trials: `801` Left, `774` Right) $\rightarrow$ Validate `Run 8` (`105` recordings, `1,573` trials: `788` Left, `785` Right).
    - Fold 2: Train `Run 8` (`1,573` trials) $\rightarrow$ Validate `Run 4` (`1,575` trials).
  - **Held-Out Final Test (`Run 12`, `105` recordings):** **`1,575` trials** (`794` Left `T1=0`, `781` Right `T2=1`).

### Confirmatory Cohort Locked Final Test Results (`N = 105` Participants, Train Runs 4+8 = `3,148` Trials, Held-Out Test Run 12 = `1,575` Trials)

| Model Family | Locked Config (from Runs 4↔8 CV) | Dev CV Pooled Acc (±2-Fold SD) | **Test Run 12 Pooled Acc** | **Test Balanced Acc** | **Test Macro F1** | **Test Participant Mean Acc (95% CI, $N=105$)** | Features ($p$) | Compression vs. Raw | **SVM Train Time (ms)** | **SVM Predict Batch Time (ms)** | **Deployed Storage (Bytes)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BaselineLinearSVC** | `LinearSVC_C=0.1` | `57.02% ± 0.12%` | **`57.84%`** | `57.85%` | `0.5784` | **`57.84%`** (`[55.22%, 60.46%]`) | `128` | `160.0x` | `147.78` | `1.63` | `4,104` |
| **2. Baseline RBF SVC** | `RBF_SVC_C=10_gamma=scale` | `55.59% ± 0.04%` | **`58.16%`** | **`58.12%`** | **`0.5803`** | **`58.16%`** (`[55.38%, 60.94%]`) | `128` | `160.0x` | `847.75` | `716.12` | `2,907,128` |
| **3. HOSVD + LinearSVC** | `HOSVD_rc=8_rt=10_C=0.1` | `50.54% ± 1.78%` | **`47.94%`** | `47.92%` | `0.4790` | **`47.94%`** (`[45.50%, 50.38%]`) | `80` | `256.0x` | `34.68` | `1.48` (`0.016 ms/trial` total) | `32,264` |
| **4. RKS + LinearSVC** | `RKS_D=256_gamma=0.001_C=10` | **`57.56% ± 0.10%`** | **`57.59%`** | `57.58%` | `0.5758` | **`57.59%`** (`[54.86%, 60.32%]`) | `256` | `80.0x` (`2.0x` vs BP) | `365.83` | **`0.82` (`870x` faster than RBF)** | `269,320` |
| **5. Standalone Markov** | `Markov_K=3_C=0.1` | `50.76% ± 0.09%` | **`50.79%`** | `50.83%` | `0.5070` | **`50.79%`** (`[48.53%, 53.05%]`) | **`15`** | **`1,365.3x`** | **`10.49`** | **`0.62`** | **`2,888`** |
| **6. HOSVD + Markov** | `HOSVD(16,40)+Markov(K=6), C=0.1` | `48.79% ± 0.29%` | **`48.19%`** | `48.18%` | `0.4818` | **`48.19%`** (`[45.86%, 50.52%]`) | `685` | `29.90x` | `1,464.43` | `8.83` | `135,496` |
| **7. RKS + Markov** | `RKS(256,0.01)+Markov(K=4), C=0.1` | `55.21% ± 0.29%` | **`57.08%`** | `57.05%` | `0.5701` | **`57.08%`** (`[54.54%, 59.62%]`) | `279` | `73.41x` | `736.91` | `3.58` | `278,760` |

See [REPORT_AND_VIVA_GUIDE.md](file:///c:/Users/Abhiram/OneDrive/Documents/sem3/MIS-3/END%20SEM%20PROJECT/REPORT_AND_VIVA_GUIDE.md) and [outputs/cohort_expanded/metrics/m5_locked_test_report.json](file:///c:/Users/Abhiram/OneDrive/Documents/sem3/MIS-3/END%20SEM%20PROJECT/outputs/cohort_expanded/metrics/m5_locked_test_report.json) for full per-participant distributions, confusion matrices, HOSVD exploratory rank curves, and predetermined multi-seed RKS variability.
