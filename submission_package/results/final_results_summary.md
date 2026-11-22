# Final Results Summary — SVD + RBF Kernel SVM on PhysioNet eegmmidb v1.0.0

**Protocol disclosure**: Run 12 outcomes on this cohort were inspected in earlier work.
This evaluation is an **exploratory follow-up**, not a fresh untouched confirmatory test.
Personalized models assess new recordings of known participants, not unseen participants.

## 1. Cohort Audit & Trial Counts
- **Candidate range**: S002–S109 (108 subjects; Subject 1 prespecified pilot exclusion).
- **Eligible participants**: **105** (100% accounted for in validation, evaluation, and reporting).
- **Objective exclusions (3)**: S088, S092, S100 — all `sfreq_mismatch: 128.0` (non-160 Hz).
- **Recordings & trials**: 315 recordings, **4,723** trials total (2,383 left / T1; 2,340 right / T2):
  - Run 4: 1,575 trials (797 left, 778 right)
  - Run 8: 1,573 trials (792 left, 781 right)
  - Run 12: 1,575 trials (794 left, 781 right)

## 2. Development Validation (Runs 4 ↔ 8)
- **Mean development macro F1**: SVD + RBF = **0.6267** (96 candidates/subject); RBF without SVD = **0.5406** (16 candidates/subject).
- **Selected SVD ranks**: `r=2` (37), `r=4` (17), `r=6` (13), `r=8` (14), `r=10` (10), `r=12` (14); mean rank **5.71** (median 4.0), mean dimension reduction **35.15×** (10.67×–64.00×).

## 3. Exploratory Run 12 Classification Performance (N = 105, 1,575 trials)

| Method | Part. Accuracy (95% CI) | Part. Bal. Acc. (95% CI) | Part. Macro F1 (95% CI) | Pooled Acc. | Pooled Macro F1 | Pooled CM (`[[LL,LR],[RL,RR]]`) |
|---|---:|---:|---:|---:|---:|---|
| RBF without SVD | **56.44%** [53.67%, 59.22%] | 56.47% [53.71%, 59.23%] | **0.5376** [0.5075, 0.5676] | 56.44% | 0.5636 | `[[480, 314], [372, 409]]` |
| SVD + RBF | **52.63%** [49.94%, 55.33%] | 52.69% [50.01%, 55.36%] | **0.5038** [0.4752, 0.5324] | 52.63% | 0.5257 | `[[444, 350], [396, 385]]` |
| Training-majority | 49.84% [49.19%, 50.49%] | 50.00% [50.00%, 50.00%] | 0.3323 [0.3294, 0.3352] | 49.84% | 0.4473 | `[[632, 162], [628, 153]]` |

- **Macro precision / recall (participant-averaged)**:
  - RBF without SVD: precision 0.5711 [0.5344, 0.6078], recall 0.5647 [0.5371, 0.5923].
  - SVD + RBF: precision 0.5240 [0.4906, 0.5574], recall 0.5269 [0.5001, 0.5536].
- **Paired SVD minus baseline (N = 105)**:
  - Accuracy: **−3.81 pp** (SD 13.77 pp, 95% CI **[−6.47, −1.14] pp**) — 27 wins / 28 ties / 50 losses.
  - Balanced accuracy: **−3.78 pp** (SD 13.58 pp, 95% CI **[−6.41, −1.16] pp**).
  - Macro F1: **−0.0337** (SD 0.1468, 95% CI **[−0.0621, −0.0053]**) — 32 wins / 19 ties / 54 losses.
- **75% target**: **Not achieved** at cohort level (only 5/105 baseline and 6/105 SVD participants reached ≥75%; baseline range 13.33%–86.67%, median 60.00%; SVD range 20.00%–86.67%, median 53.33%).

## 4. SVD Diagnostics (Refitted on Runs 4+8)
- **Retained training spectral energy**: mean **78.27%** (SD 12.47%, range 46.94%–95.40%).
- **Standardized bandpower reconstruction error (relative Frobenius)**:
  - Development (runs 4+8): mean **0.4462** (SD 0.1357, range 0.2145–0.7284).
  - Unseen run 12: mean **0.5634** (SD 0.1094, range 0.3202–0.8229).

## 5. Timing & Storage
- **Fixed Run 12 I/O & preprocessing** (excluded from warm feature+classifier timing):
  - EDF read: median **22.48 ms** (mean 23.48 ms).
  - 8–30 Hz FIR filtering + epoching: median **112.98 ms** (mean 116.19 ms).
- **Warm stage timings** (median of per-participant medians across 5 repeats, 15-trial Run 12 batch):
  - Representation fit: 1.21 ms (baseline) vs 2.17 ms (SVD).
  - SVM fit: 1.37 ms (baseline) vs 1.44 ms (SVD).
  - Welch bandpower: 16.00 ms (baseline) vs 15.80 ms (SVD) — **>93% of batch inference time**.
  - Representation transform: 0.62 ms (baseline) vs 0.81 ms (SVD).
  - Classifier-only `svm.predict`: 0.39 ms (baseline) vs 0.34 ms (SVD).
  - Feature + classifier batch total: **16.99 ms** (baseline) vs **16.92 ms** (SVD); amortized throughput **1.13 ms/trial** for both (not single-trial real-time latency).
- **Storage (105 personalized models)**:
  - Per-trial feature vector (`float64`): 1,024 B (baseline) vs 45.7 B mean (SVD).
  - Parameter arrays: mean 32,328 B / total **3,394,464 B (3.39 MB)** (baseline) vs mean 10,908 B / total **1,145,336 B (1.15 MB)** (SVD).
  - Serialized `.joblib` pipelines: mean 34,403 B (range 28,173–36,541 B) / total **3,612,293 B (3.61 MB)** (baseline) vs mean 13,240 B (range 8,397–21,485 B) / total **1,390,149 B (1.39 MB)** (SVD, **−61.5%**).

## 6. Runtime Environment
Windows 11 (`Windows-11-10.0.26200-SP0`), AMD64 (16 logical CPUs), Python 3.13.6,
NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, MNE 1.13.2, pandas 3.1.1, joblib 1.6.0.
