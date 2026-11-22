# Study and viva guide

## The project in one minute
We classify imagined left-versus-right fist movement from 64-channel scalp EEG
(PhysioNet eegmmidb v1.0.0, runs 4, 8, and 12). Each trial has 64 channels × 320
samples (2 s at 160 Hz), converted into 128 Welch log-bandpower features (mu 8–13 Hz
and beta 13–30 Hz). Inside each participant's training fold, a `StandardScaler`
standardizes features, `SVDProjector` projects them onto the top-$r$ right singular
vectors, and an RBF `SVC` learns the decision boundary. We compare this against an
identical RBF SVM **without** SVD on the exact same features and splits.

## Actual measured findings (105 eligible participants, 1,575 run 12 trials)
- **Eligibility audit**: Out of 108 candidates (S002–S109; S001 excluded as pilot),
  **105 participants were eligible** and **3 were excluded** objectively for 128 Hz
  sampling (`sfreq_mismatch: 128.0`: S088, S092, S100). Total trials: **4,723**
  (Run 4: 1,575; Run 8: 1,573; Run 12: 1,575).
- **Exploratory run 12 classification**:
  - **RBF without SVD (baseline)**: **56.44%** participant-mean accuracy
    (95% CI [53.67%, 59.22%]), **56.47%** balanced accuracy, **0.5376** macro F1
    (pooled macro F1 0.5636; pooled CM `[[480, 314], [372, 409]]`).
  - **SVD + RBF**: **52.63%** participant-mean accuracy
    (95% CI [49.94%, 55.33%]), **52.69%** balanced accuracy, **0.5038** macro F1
    (pooled macro F1 0.5257; pooled CM `[[444, 350], [396, 385]]`).
  - **Training-majority reference**: **49.84%** accuracy (95% CI [49.19%, 50.49%]),
    **0.3323** macro F1.
- **Did SVD improve the baseline? No.** Paired SVD minus baseline across all 105
  participants was **−3.81 percentage points** in accuracy (95% CI [−6.47, −1.14] pp)
  and **−0.0337** in macro F1 (95% CI [−0.0621, −0.0053]); SVD beat baseline on 27
  participants, tied on 28, and lost on 50.
- **Was the 75% target achieved? No.** Neither method reached 75% cohort accuracy
  (only 5/105 baseline and 6/105 SVD participants scored ≥ 75% on their 15-trial
  run 12 test set).
- **Why did SVD score higher in development (0.6267 vs 0.5406 macro F1) but lower
  on run 12?** Two reasons: (1) SVD searched 6× more hyperparameter combinations
  (96 vs 16) on tiny ~15-trial folds, creating selection optimism; (2) unsupervised
  SVD keeps high-variance directions rather than class-separating directions, and
  cross-run distribution shift increased reconstruction error from 0.446 (runs 4+8)
  to 0.563 (run 12).
- **Resource trade-off**: SVD reduced total serialized model storage by **61.5%**
  (1.39 MB vs 3.61 MB across 105 pipelines), and slightly sped up classifier-only
  `svm.predict` (0.34 ms vs 0.39 ms), but Welch bandpower (~16 ms/batch) dominated
  total feature+classifier inference (~16.92 vs ~16.99 ms/batch; 1.13 ms/trial
  amortized), so total inference time was essentially unchanged.

## Dataset: what to show the faculty
Show the PhysioNet source page, the local subject directories, `outputs/audit/manifest.json`,
and the real Subject 2 EEG plot (`outputs/eeg_example/eeg_example.png`).
An EDF file is a recording container: channel metadata, sampling frequency, signal
samples in volts, and event annotations. MNE reads it directly. A subject is a
participant; a run is one recording; a trial is the `[0.5, 2.5)` s post-cue interval.

For runs 4, 8, and 12 ONLY, `T1` means imagined left fist (label 0) and `T2` means
imagined right fist (label 1); `T0` is rest and is excluded. The same annotation
names mean different tasks in other runs.

## Signal processing
160 Hz means 160 samples/s per channel. Two seconds gives 320 samples;
64 × 320 = 20,480 values per trial. EEG is stored in volts (multiply by $10^6$ only
when plotting in µV).

Each run is filtered independently at 8–30 Hz with zero-phase FIR `firwin` (offline
only, since zero-phase is non-causal). Mu is 8–13 Hz and beta is 13–30 Hz. Welch
PSD uses a 160-sample Hann window with 80-sample overlap, constant detrend, and
density scaling. Integrating each band with the trapezoidal rule and taking
$\log_{10}(\max(\text{power}, 10^{-20}))$ gives 64 × 2 = 128 features.

## Standardization and SVD
The feature matrix $F \in \mathbb{R}^{N \times 128}$ is standardized using training
means $\mu_j$ and standard deviations $s_j$: $A_{ij} = (F_{ij} - \mu_j)/s_j$.
Validation and test trials use those **exact same** training statistics.

$A = U\Sigma V^T$. Columns of $V_r \in \mathbb{R}^{128 \times r}$ are orthonormal
directions in feature space. Any unseen standardized row $a$ projects via
$z = a V_r$ (never via $U$). Reconstruction is $\widehat{A} = A V_r V_r^T$ (of
standardized bandpower, not raw EEG waveforms). Relative Frobenius error is
$E_r = \|A - \widehat{A}\|_F / \|A\|_F = \sqrt{\sum_{j>r}\sigma_j^2 / \sum_j\sigma_j^2}$.

Across our 105 participants (refitted on runs 4+8), selected ranks averaged 5.71
(median 4; $r=2$ chosen 37 times), retaining 78.27% of training spectral energy
(range 46.94%–95.40%), with mean reconstruction error 0.4462 on runs 4+8 rising to
0.5634 on run 12.

## SVM and RBF kernel
Soft-margin SVM minimizes $\frac{1}{2}\|w\|^2 + C\sum_i \xi_i$ subject to
$y_i(w^T\phi(z_i) + b) \ge 1 - \xi_i$, $\xi_i \ge 0$ (with labels $y_i \in \{-1, +1\}$).
The RBF kernel is $K(z, z') = \exp(-\gamma \|z - z'\|^2)$. Support vectors are
training points with $\alpha_i > 0$. Large $C$ penalizes training margin errors more
strongly; large $\gamma$ makes the kernel more local; `'scale'` sets
$\gamma = 1 / (d \cdot \operatorname{Var}(Z_{\text{train}}))$.

## Evaluation and integrity
- **Recording separation**: Development swaps complete runs 4→8 and 8→4. Never split
  samples from one trial or one recording across train and validation.
- **No leakage**: `StandardScaler` and `SVDProjector` are refitted inside each fold.
- **Prior test exposure**: Because run 12 outcomes on this cohort were inspected in
  earlier work, this evaluation is explicitly an **exploratory follow-up**, not a
  fresh untouched confirmatory test.
- **Why keep all 105 eligible participants?** Dropping low-scoring participants
  would cherry-pick and invalidate the cohort estimate.
- **Personalization**: Models generalize to new recordings of known participants,
  not to unseen people.

## Timing and storage breakdown (measured medians across 105 participants)
- **Fixed run 12 I/O & preprocessing** (excluded from warm feature+classifier time):
  EDF read = 22.48 ms; 8–30 Hz FIR filter + epoching = 112.98 ms.
- **Warm 15-trial run 12 batch** (5 repeats after warm-up):
  - Welch bandpower: 16.00 ms (baseline) vs 15.80 ms (SVD) — **>93% of batch time**.
  - Representation transform: 0.62 ms (baseline) vs 0.81 ms (SVD).
  - Classifier-only `svm.predict`: 0.39 ms (baseline) vs 0.34 ms (SVD).
  - Total feature + classifier batch: 16.99 ms (baseline) vs 16.92 ms (SVD);
    amortized throughput = **1.13 ms/trial** for both (not single-trial latency).
- **Storage** (105 personalized models):
  - Mean serialized `.joblib`: 34,403 B (baseline) vs 13,240 B (SVD).
  - Total serialized storage: **3,612,293 B (3.61 MB)** vs **1,390,149 B (1.39 MB)**
    (**61.5% reduction**).

## Code map
- `src/eeg_svd/config.py`: fixed protocol, validation, JSON, SHA-256 hashes.
- `src/eeg_svd/data.py`: local EDF reading, mechanical eligibility audit, provenance.
- `src/eeg_svd/features.py`: deterministic 128-D Welch log-bandpower.
- `src/eeg_svd/svd.py`: `SVDProjector` (`np.linalg.svd`, projection, reconstruction).
- `src/eeg_svd/models.py`: pipelines and recording-swapped development selection.
- `src/eeg_svd/evaluation.py`: tamper-evident locks, run 12 evaluation, timing, storage.
- `src/eeg_svd/reporting.py`: plots and summary tables from saved artifacts only.
- `src/eeg_svd/cli.py`: CLI entry points (`audit-data`, `validate`, `evaluate`, `build-report`, `plot-eeg`).
- `tests/test_project.py`: 24 synthetic tests + 1 optional real S001 EDF test.

## Viva questions and defensible answers
1. **What is new?** Not the algorithms (SVD and RBF SVM are established), but a
   strictly controlled, leakage-guarded 105-participant ablation quantifying the
   accuracy, reconstruction, stage-timing, and storage trade-off under recording
   separation.
2. **Why did SVD reduce accuracy from 56.44% to 52.63%?** Two concrete reasons:
   unsupervised SVD preserves variance rather than class separation, and searching
   96 SVD+SVM combinations (vs 16 baseline) on ~15-trial folds caused selection
   optimism (dev macro F1 0.6267 vs 0.5406) that did not transfer to run 12.
3. **Did you reach 75%?** No. Baseline averaged 56.44% and SVD+RBF averaged 52.63%
   across all 105 participants (only 5 and 6 participants respectively reached ≥75%).
   We report this honestly rather than cherry-picking or tuning on run 12.
4. **Why was total inference time ~17 ms for both methods even though SVD had fewer
   features?** Because Welch bandpower extraction took ~16 ms (~94% of batch time),
   dwarfing the tiny `svm.predict` difference (0.34 ms vs 0.39 ms).
5. **Where did SVD actually help?** Model compactness: total serialized storage
   across 105 personalized pipelines dropped by 61.5% (from 3.61 MB to 1.39 MB).
6. **Why were 3 subjects excluded?** Purely mechanical quality audit: S088, S092,
   and S100 were recorded at 128 Hz instead of the protocol's 160 Hz. Subject 1 was
   excluded upfront as a previously explored pilot.
7. **Is this clinical or real-time?** No. It is offline research using non-causal
   zero-phase FIR filtering on a single public dataset.
8. **Why is this an exploratory follow-up instead of a confirmatory test?** Because
   run 12 outcomes on this cohort were inspected in earlier work; rebuilding the
   codebase cannot erase historical exposure.
9. **Does higher retained SVD energy guarantee higher accuracy?** No. Energy
   measures unsupervised Frobenius reconstruction of bandpower, not class separation.
10. **What would you try next?** Supervised spatial/spectral methods like CSP/FBCSP
    on genuinely untouched datasets with more training trials per participant.

## Individual contribution log (complete truthfully)
| Student / USN | Work actually performed | Evidence / files | Can explain in viva? |
|---|---|---|---|
| [Name / USN] | [Fill after reviewing and executing] | [Commit/output] | [Yes/No] |
