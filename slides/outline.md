# Presentation outline

Status: draft; measured EEG results pending. Main slides: 15. Scheduled speaking time: 20.0 minutes.

## 1. Title (1 min)

- Rank-Controlled SVD Features for EEG Motor Imagery
- SVD + RBF Kernel SVM
- [Names / USNs] · [Department / institution]
- Faculty: Dr Geetha KN · [Academic year]

Speaker notes: Introduce the one-combination scope. This draft has no measured EEG results yet.

## 2. Problem statement (1 min)

- Predict imagined left versus right fist from EEG.
- 64 channels × 320 samples per trial.
- Can lower-dimensional spectral features preserve discrimination?

Speaker notes: Explain motor imagery, EEG variability and why a fair baseline matters.

## 3. Literature survey (1.5 min)

- Physiology and signal processing: refs 1–3, 13.
- Spatial EEG features: refs 4–6.
- SVM, kernels and low-rank theory: refs 7–12.
- EEG learning and reproducible benchmarks: refs 14–16.
- Full 16-paper table and references are in the backups.

Speaker notes: Discuss the supported project question, not a claim that SVD-SVM is unprecedented. Refer to literature/survey.md.

## 4. Objectives and scope (0.75 min)

- Measure accuracy, macro F1, reconstruction, time and storage.
- Main: standardized bandpower → SVD → RBF SVM.
- Ablation: the identical pipeline without SVD.
- Personalized models; no clinical or real-time claim.

Speaker notes: The 75% accuracy target is an aspiration. No result is guaranteed.

## 5. Theory: SVD and Kernel SVM (3 min)

- A = UΣVᵀ; use training-fitted Vᵣ for every unseen trial.
- z = aVᵣ; reconstruction = aVᵣVᵣᵀ.
- Centered rank ≤ min(Ntrain − 1, 128).
- K(z,z′) = exp(−γ‖z−z′‖²).
- C penalizes margin violations; γ controls locality.

Speaker notes: Explain variance versus discrimination, then margin, support vectors and the RBF kernel. Derivations are in backups.

## 6. Proposed methodology (1.5 min)

- EDF → audit → independent filtering → epochs
- Welch mu/beta power → log10 → training-only scaling
- Training-only SVD → RBF SVM → left/right label
- Repeat using the no-SVD baseline on identical splits.

Speaker notes: Describe one trial through the full chain and justify the role of each transformation.

## 7. Dataset and preprocessing (1.5 min)

- PhysioNet eegmmidb v1.0.0; candidates S002–S109.
- T1=left and T2=right ONLY for runs 4, 8, 12.
- 64 EEG channels, 160 Hz, filter 8–30 Hz.
- Epoch [0.5,2.5): 320 samples; 128 bandpower features.
- Actual eligible counts come from the new audit.

Speaker notes: Explain EDF storage. Do not copy old cohort counts. All exclusions are mechanical and precede modeling.

## 8. Algorithm implementation (1.5 min)

- Fit a new scaler and SVD within each training fold.
- Search valid rank, C and γ; select by mean macro F1.
- Save model selections and recording hashes.
- Refit on runs 4+8; explicitly evaluate run 12.
- Python, MNE, NumPy/SciPy, scikit-learn, pytest.

Speaker notes: Show the CLI or a short pseudocode excerpt. The saved lock binds parameters to the dataset manifest.

## 9. Experimental setup (1.5 min)

- Per participant: 4→8 and 8→4 validation.
- r={2,4,6,8,10,12}, restricted to numerical rank.
- C={0.1,1,10,100}; γ={scale,0.001,0.01,0.1}.
- Report actual runtime hardware and software.
- Earlier run 12 inspection makes this exploratory follow-up.

Speaker notes: Explain that rebuilding code does not create an untouched test. No favorable participant selection.

## 10. Results and visualization (2.5 min)

- PENDING: run the new local EDF experiment.
- Insert actual accuracy, balanced accuracy, precision, recall and F1.
- Show confusion matrices and singular-value/reconstruction plots.
- Synthetic test outputs are never EEG evidence.

Speaker notes: Replace pending content only after checking results.json and cohort completeness. Show measured figure if supplied.

## 11. Comparative analysis (1.5 min)

- Compare SVD + RBF with the same RBF model without SVD.
- Use paired participant differences and confidence intervals.
- Report complete feature-plus-classifier timing and model storage.
- Similar means do not establish statistical equivalence.

Speaker notes: State whether SVD helped only after measurement. Preserve unfavorable or negative results.

## 12. Applications and significance (0.5 min)

- Educational offline BCI research.
- Understanding EEG representation and computational trade-offs.
- No device control, diagnosis or clinical deployment claim.

Speaker notes: Explain potential future relevance without claiming demonstrated real-world reliability.

## 13. Limitations and future scope (1 min)

- Few trials per participant; only two development folds.
- Variance preservation may lose class information.
- Known-participant evaluation is not unseen-person generalization.
- Prior test exposure; cue artifacts and nonstationarity.
- Future: genuinely unused data and prespecified extensions.

Speaker notes: These limitations belong in the main presentation, not only in a disclaimer.

## 14. Conclusion (0.75 min)

- Implemented: recording-separated SVD + Kernel SVM comparison.
- Measured finding: PENDING LOCAL EXPERIMENT.
- Report the observed trade-off, including negative outcomes.
- Do not claim 75% or superiority without evidence.

Speaker notes: After execution, replace this with the actual primary result and its uncertainty.

## 15. References (0.5 min)

- 16 verified DOI records in literature/survey.md.
- Core: Cortes & Vapnik; Eckart & Young; Welch.
- EEG: Pfurtscheller & Lopes da Silva; Schalk et al.; Gramfort et al.
- Full citations follow in backup slides.

Speaker notes: The 15 main slides total 20 minutes. Be ready to identify the directly relevant EEG literature.

## 16. Backup: SVM mathematics (0 min)

- min ½‖w‖² + CΣξᵢ; ξᵢ≥0
- yᵢ(wᵀφ(zᵢ)+b)≥1−ξᵢ
- Dual: max Σαᵢ − ½ΣαᵢαⱼyᵢyⱼK(zᵢ,zⱼ)
- 0≤αᵢ≤C; Σαᵢyᵢ=0
- Mathematical labels −1/+1; implementation accepts 0/1.

Speaker notes: Explain soft-margin trade-off and kernel decision without pretending to explicitly construct φ.

## 17. Backup: SVD reconstruction (0 min)

- Training error² = Σdiscarded σ² / Σall σ².
- Optimal Frobenius reconstruction ≠ optimal classification.
- Features per trial: r; basis entries: 128r.
- Example singular values (3,1): rank 1 retains 90% energy.
- Example is mathematical, not experimental EEG evidence.

Speaker notes: Use this example to explain reconstruction versus classification.

## 18. Backup: implementation and integrity (0 min)

- data.py: EDF checks, provenance, independent filtering.
- features.py / svd.py: spectral features and projection.
- models.py: development-only selection.
- evaluation.py: locks, paired metrics, timing, storage.
- reporting.py never trains or evaluates a model.

Speaker notes: Show actual code if asked; the study guide maps every step to a source file.

## 19. Backup: references 1–4 (0 min)

- [1] Schalk et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. DOI: 10.1109/TBME.2004.827072
- [2] Gramfort et al. (2013). MEG and EEG data analysis with MNE-Python. DOI: 10.3389/fnins.2013.00267
- [3] Pfurtscheller et al. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. DOI: 10.1016/S1388-2457(99)00141-8
- [4] Ramoser et al. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. DOI: 10.1109/86.895946

Speaker notes: See the accompanying literature table for publication venues, method and scope limitations.

## 20. Backup: references 5–8 (0 min)

- [5] Blankertz et al. (2008). Optimizing Spatial filters for Robust EEG Single-Trial Analysis. DOI: 10.1109/MSP.2008.4408441
- [6] Ang et al. (2012). Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b. DOI: 10.3389/fnins.2012.00039
- [7] Cortes et al. (1995). Support-vector networks. DOI: 10.1007/BF00994018
- [8] Burges et al. (1998). A Tutorial on Support Vector Machines for Pattern Recognition. DOI: 10.1023/A:1009715923555

Speaker notes: See the accompanying literature table for publication venues, method and scope limitations.

## 21. Backup: references 9–12 (0 min)

- [9] Schölkopf et al. (1998). Nonlinear Component Analysis as a Kernel Eigenvalue Problem. DOI: 10.1162/089976698300017467
- [10] Jolliffe et al. (2016). Principal component analysis: a review and recent developments. DOI: 10.1098/rsta.2015.0202
- [11] Eckart et al. (1936). The Approximation of One Matrix by Another of Lower Rank. DOI: 10.1007/BF02288367
- [12] Golub et al. (1970). Singular value decomposition and least squares solutions. DOI: 10.1007/BF02163027

Speaker notes: See the accompanying literature table for publication venues, method and scope limitations.

## 22. Backup: references 13–16 (0 min)

- [13] Welch et al. (1967). The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. DOI: 10.1109/TAU.1967.1161901
- [14] Dose et al. (2018). An end-to-end deep learning approach to MI-EEG signal classification for BCIs. DOI: 10.1016/j.eswa.2018.08.031
- [15] Lawhern et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. DOI: 10.1088/1741-2552/aace8c
- [16] Jayaram et al. (2018). MOABB: trustworthy algorithm benchmarking for BCIs. DOI: 10.1088/1741-2552/aadea0

Speaker notes: See the accompanying literature table for publication venues, method and scope limitations.
