# Presentation outline

Status: completed from measured 105-participant exploratory evaluation. Main slides: 15. Scheduled speaking time: 20.0 minutes.

## 1. Title (1 min)

- Rank-Controlled SVD Features for EEG Motor Imagery
- SVD + RBF Kernel SVM vs. Matched No-SVD Baseline
- [Names / USNs] · [Department / institution]
- Faculty: Dr Geetha KN · [Academic year]

Speaker notes: Introduce the one-combination scope: bandpower -> StandardScaler -> SVD -> RBF SVM compared against the identical pipeline without SVD across all 105 eligible PhysioNet eegmmidb participants.

## 2. Problem statement (1 min)

- Predict imagined left versus right fist from 64-channel scalp EEG.
- Each trial has 64 channels × 320 samples (2 s at 160 Hz) → 128 spectral features.
- Question: does low-rank SVD improve the accuracy–resource trade-off when recordings stay separate?
- Fairness requirement: matched no-SVD RBF SVM baseline on identical features and splits.

Speaker notes: Explain motor imagery, EEG variability, and why testing SVD against an identical no-SVD baseline is essential to isolate the projection effect.

## 3. Literature survey (1.5 min)

- Physiology & signal processing: Schalk et al. [1], Gramfort [2], Pfurtscheller [3], Welch [13].
- Supervised spatial EEG features: CSP [4, 5] and Filter-Bank CSP [6].
- SVM, kernels & low-rank theory: Cortes & Vapnik [7], Burges [8], Kernel PCA [9], PCA/SVD [10–12].
- EEG deep learning & reproducible benchmarks: Dose et al. [14], EEGNet [15], MOABB [16].
- Full 16-paper DOI-verified comparison table is in literature/survey.md and backups.

Speaker notes: Emphasize that SVD + SVM is established; our contribution is an auditable full-cohort ablation of accuracy, reconstruction, timing, and storage under recording separation.

## 4. Objectives and scope (0.75 min)

- Main pipeline: 128-D Welch log-bandpower → StandardScaler → SVD → RBF SVM.
- Ablation baseline: identical pipeline without SVD (+ training-majority reference).
- Measure accuracy, balanced accuracy, macro F1, reconstruction, stage timings, and storage.
- Personalized models across all eligible subjects; 75% accuracy is a target, not guaranteed.

Speaker notes: Clarify scope: offline research only, no HOSVD/RKS/Markov/neural networks, no clinical or real-time claim.

## 5. Theory: SVD and Kernel SVM (3 min)

- Standardized training matrix A = UΣVᵀ; unseen trial projects via z = aVᵣ (not U).
- Reconstruction Â = AVᵣVᵣᵀ; Eckart–Young minimizes Frobenius error Eᵣ = ‖A−Â‖_F / ‖A‖_F.
- Centered numerical rank ≤ min(N_train − 1, 128): ≤ 14 per dev fold, ≤ 29 on runs 4+8.
- RBF kernel K(z,z′) = exp(−γ‖z−z′‖²); soft-margin SVM balances ½‖w‖² + CΣξᵢ.
- Key caveat: SVD maximizes variance reconstruction, not class separability.

Speaker notes: Walk through the equations clearly. Highlight the theoretical reason SVD can hurt: variance-preserving projection is unsupervised.

## 6. Proposed methodology (1.5 min)

- 1. Mechanical EDF audit (S002–S109; S001 pilot excluded) + SHA-256 provenance.
- 2. Independent 8–30 Hz zero-phase FIR filter → [0.5, 2.5) s epochs → 128-D Welch log-bandpower.
- 3. Training-fold-only StandardScaler + SVD + RBF SVM on runs 4→8 and 8→4.
- 4. Lock best (r, C, γ) by mean dev macro F1 → refit on runs 4+8 → evaluate run 12 once.

Speaker notes: Trace one trial from EDF to binary prediction and explain how fitting scalers and SVD inside each training fold prevents data leakage.

## 7. Dataset and preprocessing (Audit: 105 / 108 eligible) (1.5 min)

- PhysioNet eegmmidb v1.0.0; runs 4, 8, 12 ONLY (T1=left, T2=right; T0 excluded).
- Audit result: 105 eligible participants; 3 excluded (S088, S092, S100: 128 Hz ≠ 160 Hz).
- 4,723 total trials (2,383 left, 2,340 right): Run 4 = 1,575; Run 8 = 1,573; Run 12 = 1,575.
- Right: real C3 raw vs. 8–30 Hz filtered trace & first trials from eligible Subject 2.

Speaker notes: Point out that all exclusions were purely mechanical (128 Hz sampling rate), never accuracy-based, and Subject 2 was chosen independently of accuracy for the EEG plot.

## 8. Algorithm implementation and verification (1.5 min)

- Modular package (src/eeg_svd): config, data, features, svd, models, evaluation, reporting, cli.
- Tamper-evident lock.json binds selected hyperparameters to manifest SHA-256.
- EVALUATION_STARTED.json guard prevents accidental test reruns; reporting never retrains.
- Verified on Windows 11 / Python 3.13.6: 24 synthetic tests + 1 real S001 EDF test passed (25/25).

Speaker notes: Mention ruff and compileall also passed. Environment frozen in outputs/environment.txt.

## 9. Experimental setup (1.5 min)

- Personalized models: development folds 4→8 and 8→4; refit 4+8; evaluate run 12.
- Grid: r ∈ {2, 4, 6, 8, 10, 12}, C ∈ {0.1, 1, 10, 100}, γ ∈ {scale, 0.001, 0.01, 0.1}.
- 96 SVD+RBF candidates vs. 16 no-SVD candidates per participant.
- Prior exposure disclosure: run 12 was inspected earlier → exploratory follow-up, not fresh confirmatory test.
- Hardware: Windows 11, AMD64 (16 logical CPUs), NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, MNE 1.13.2.

Speaker notes: Be transparent about prior test exposure and the 6x larger hyperparameter search space for SVD+RBF.

## 10. Measured results (Run 12, N = 105 participants, 1,575 trials) (2.5 min)

- RBF without SVD: 56.44% acc (95% CI [53.67, 59.22]%), 56.47% bal acc, 0.5376 macro F1.
- SVD + RBF: 52.63% acc (95% CI [49.94, 55.33]%), 52.69% bal acc, 0.5038 macro F1.
- Training-majority: 49.84% acc (95% CI [49.19, 50.49]%), 0.3323 macro F1.
- 75% target NOT achieved (only 5/105 baseline and 6/105 SVD subjects reached ≥ 75%).

Speaker notes: Show both the participant-mean accuracy chart with 95% t-intervals and the pooled confusion matrices (baseline [[480,314],[372,409]] vs SVD [[444,350],[396,385]]).

## 11. Comparative analysis: accuracy, reconstruction, time & storage (1.5 min)

- Paired SVD − baseline: −3.81 pp acc (95% CI [−6.47, −1.14] pp), −0.0337 macro F1; 27W / 28T / 50L.
- Dev macro F1 was higher for SVD (0.6267 vs 0.5406) → selection optimism on 15-trial folds.
- SVD mean rank 5.71 (35.15× dim cut): 78.3% train energy; recon error 0.446 (dev) → 0.563 (run 12).
- Storage: −61.5% total serialized size (1.39 MB vs 3.61 MB across 105 models).
- Timing: Welch bandpower (~16 ms) dominates ~17 ms batch (1.13 ms/trial); SVM predict 0.34 vs 0.39 ms.

Speaker notes: Highlight the central trade-off: SVD saved 61.5% model storage, had negligible end-to-end runtime impact because Welch PSD dominates, and reduced accuracy by 3.81 pp.

## 12. Applications and significance (0.5 min)

- Demonstrates rigorous offline BCI ablation methodology on a full 105-subject cohort.
- Clarifies when linear dimensionality reduction helps (storage) vs. hurts (small-sample accuracy).
- Shows why end-to-end profiling matters: classifier speedups are masked by feature extraction.
- No clinical, diagnostic, or real-time device-control claim.

Speaker notes: Emphasize that a well-controlled negative accuracy finding is valuable scientific evidence.

## 13. Limitations and future scope (1 min)

- Only ~15 trials/run and 2 dev folds → high selection variance and optimism (96 vs 16 grid).
- Unsupervised SVD discards low-variance discriminative directions; inter-run shift raises recon error.
- Personalized models do not generalize to unseen participants; prior run 12 exposure limits confirmatory claims.
- Non-causal zero-phase FIR filtering precludes online real-time claims.
- Future: genuinely untouched datasets, supervised spatial/spectral reduction (e.g., CSP), more trials.

Speaker notes: Own every limitation directly in the main presentation.

## 14. Conclusion (0.75 min)

- Completed auditable 105-participant evaluation of SVD+RBF vs. matched RBF without SVD.
- SVD did NOT improve accuracy: 52.63% vs. 56.44% (−3.81 pp, 95% CI [−6.47, −1.14] pp).
- 75% accuracy target was not achieved by either method on run 12.
- SVD cut serialized storage by 61.5% (1.39 vs 3.61 MB) with negligible total runtime change (~17 ms/batch).
- Retaining full standardized bandpower is preferable unless compact storage is prioritized.

Speaker notes: Close with the exact measured numbers and honest conclusion.

## 15. References (0.5 min)

- 16 DOI-verified references in literature/survey.md and paper/references.tex.
- Core theory: Cortes & Vapnik [7]; Burges [8]; Eckart & Young [11]; Golub & Reinsch [12]; Welch [13].
- EEG & BCI: Schalk et al. [1]; Gramfort [2]; Pfurtscheller [3]; Ramoser [4]; Blankertz [5]; Ang [6].
- Context & benchmarks: Schölkopf [9]; Jolliffe [10]; Dose [14]; Lawhern [15]; Jayaram [16].

Speaker notes: 15 main slides total 20.0 minutes. Full citations and mathematical derivations follow in backups.

## 16. Backup: SVM mathematics (0 min)

- Primal: min ½‖w‖² + CΣξᵢ subject to yᵢ(wᵀφ(zᵢ)+b) ≥ 1−ξᵢ, ξᵢ ≥ 0.
- Dual: max Σαᵢ − ½ΣαᵢαⱼyᵢyⱼK(zᵢ,zⱼ), 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0.
- RBF kernel: K(z,z′) = exp(−γ‖z−z′‖²); decision sign(ΣαᵢyᵢK(zᵢ,z)+b).
- Default γ = 'scale' sets γ = 1 / (d · Var(Z_train)) inside the SVM step.
- Mathematical labels −1/+1; scikit-learn SVC accepts 0/1 internally.

Speaker notes: Explain soft-margin regularization C and RBF locality γ without pretending to form φ(z) explicitly.

## 17. Backup: SVD reconstruction & measured diagnostics (0 min)

- Training Frobenius error² = Σ_{j>r} σⱼ² / Σ σⱼ²; retained energy = 1 − Eᵣ².
- Measured across 105 subjects (runs 4+8 refit): mean rank 5.71 (r=2:37, 4:17, 6:13, 8:14, 10:10, 12:14).
- Retained training energy: mean 78.27% (SD 12.47%, range 46.94%–95.40%).
- Bandpower recon error: mean 0.4462 (dev 4+8) → 0.5634 (run 12) due to cross-run shift.
- Toy math example: singular values (3, 1) → rank 1 retains 90% energy, error √0.1 ≈ 0.316.

Speaker notes: Emphasize that reconstruction is of standardized 128-D bandpower, not raw EEG waveforms.

## 18. Backup: implementation, timing breakdown & storage (0 min)

- Fixed Run 12 I/O: EDF read median 22.48 ms; 8–30 Hz FIR filter + epoch median 112.98 ms.
- Warm 15-trial batch (SVD vs Base): bandpower 15.80 vs 16.00 ms; transform 0.81 vs 0.62 ms; SVM 0.34 vs 0.39 ms.
- Total warm feature+classifier batch: 16.92 vs 16.99 ms (amortized 1.13 ms/trial; not single-trial latency).
- Serialized joblib size: SVD mean 13,240 B (total 1.39 MB) vs Base mean 34,403 B (total 3.61 MB).
- Parameter arrays: SVD total 1.15 MB vs Base total 3.39 MB; 210 trained models saved in outputs/evaluation/models/.

Speaker notes: Use this slide if faculty ask for exact stage-by-stage timing or memory accounting.

## 19. Backup: references 1–4 (0 min)

- [1] Schalk et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. DOI: 10.1109/TBME.2004.827072
- [2] Gramfort (2013). MEG and EEG data analysis with MNE-Python. DOI: 10.3389/fnins.2013.00267
- [3] Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. DOI: 10.1016/S1388-2457(99)00141-8
- [4] Ramoser et al. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. DOI: 10.1109/86.895946

Speaker notes: See literature/survey.md for full publication venues, method roles, and scope limitations.

## 20. Backup: references 5–8 (0 min)

- [5] Blankertz et al. (2008). Optimizing Spatial filters for Robust EEG Single-Trial Analysis. DOI: 10.1109/MSP.2008.4408441
- [6] Ang et al. (2012). Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b. DOI: 10.3389/fnins.2012.00039
- [7] Cortes & Vapnik (1995). Support-vector networks. DOI: 10.1007/BF00994018
- [8] Burges (1998). A Tutorial on Support Vector Machines for Pattern Recognition. DOI: 10.1023/A:1009715923555

Speaker notes: See literature/survey.md for full publication venues, method roles, and scope limitations.

## 21. Backup: references 9–12 (0 min)

- [9] Schölkopf et al. (1998). Nonlinear Component Analysis as a Kernel Eigenvalue Problem. DOI: 10.1162/089976698300017467
- [10] Jolliffe & Cadima (2016). Principal component analysis: a review and recent developments. DOI: 10.1098/rsta.2015.0202
- [11] Eckart & Young (1936). The Approximation of One Matrix by Another of Lower Rank. DOI: 10.1007/BF02288367
- [12] Golub & Reinsch (1970). Singular value decomposition and least squares solutions. DOI: 10.1007/BF02163027

Speaker notes: See literature/survey.md for full publication venues, method roles, and scope limitations.

## 22. Backup: references 13–16 (0 min)

- [13] Welch (1967). The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. DOI: 10.1109/TAU.1967.1161901
- [14] Dose et al. (2018). An end-to-end deep learning approach to MI-EEG signal classification for BCIs. DOI: 10.1016/j.eswa.2018.08.031
- [15] Lawhern et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. DOI: 10.1088/1741-2552/aace8c
- [16] Jayaram & Barachant (2018). MOABB: trustworthy algorithm benchmarking for BCIs. DOI: 10.1088/1741-2552/aadea0

Speaker notes: See literature/survey.md for full publication venues, method roles, and scope limitations.
