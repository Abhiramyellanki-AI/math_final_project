# PRD — personalized EEG motor imagery with SVD + Kernel SVM

## Goal and authority
Implement a new, independent end-semester project using SVD + RBF SVM only, with
RBF SVM without SVD as the controlled baseline. This replaces the older repository's
HOSVD/RKS/Markov scope for this subdirectory. Preserve that original project.
The user authorizes implementation, tests, documentation, and publication on the
`endsem-svd-svm` branch. Never fabricate experimental results or promise 75% accuracy.

## Faculty deliverables
IEEE conference-format paper: Abstract; Introduction (problem/gap/contribution);
Mathematical background; Existing methods; Proposed method; Mathematical analysis;
Experimental design; Results; Comparison; Limitations; Conclusion; References.
Approximately 15 slides for 20 minutes, plus derivation/code/parameter backups.
At least 14–15 verified relevant papers, a literature comparison table, and a viva guide.
Rubric: literature 5; implementation 10; individual contribution/viva 10; novelty 5.
Faculty: Dr Geetha KN, Department of Mathematics, ASE Bengaluru.
Deadline: 23 November 2026, 08:30 Asia/Kolkata; presentations 25–27 November.
Student names, USNs and institution details require confirmation; use placeholders.
Do not submit to Turnitin, email faculty, or claim acceptance automatically.

## Dataset and access
Existing Windows EDF root: C:/Users/Abhiram/Downloads/eegmmidb-1.0.0/files.
Desired local workspace: C:/Users/Abhiram/OneDrive/Desktop/EEG_EndSem_SVD_SVM.
Cloud access does not imply access to either Windows directory. Do not download
without authorization. Use PhysioNet eegmmidb v1.0.0; candidate subjects 2–109;
exclude Subject 1 as a previously explored pilot. Audit before modeling, with no
accuracy-based exclusions. Require readable runs 4/8/12, 64 recognized consistently
ordered EEG channels, 160 Hz, valid T0/T1/T2 annotations, both classes and finite
in-bounds epochs. Log exact exclusions, counts and SHA-256 file provenance.

## Task and fixed representation
Runs 4/8/12 mean imagined left/right fist: T1→0, T2→1; exclude T0.
Filter each recording independently at 8–30 Hz with zero-phase FIR firwin.
Epoch [0.5,2.5) post cue: 64×320 values in volts, one trial per cue.
Welch PSD: Hann window, 160 samples, 80 overlap, constant detrend, density scaling.
Integrate bands [8,13] and [13,30] Hz; shared 13 Hz trapezoid endpoint has no overlap
of finite-width intervals. Apply log10 with 1e-20 V² floor. Produce 128 features.

## Protocol and integrity
Use one model per participant. Swap complete runs 4 and 8 for two development folds.
Fit StandardScaler and SVD only on training fold; never fit using run 12. Select by
mean macro F1, then balanced accuracy, smaller rank, smaller C, configured gamma order.
Ranks {2,4,6,8,10,12}, filtered to training-fold numerical rank. C {.1,1,10,100};
gamma {scale,.001,.01,.1}. Baseline uses the same C/gamma grid and features.
Freeze configuration and selections, refit on runs 4+8, explicitly evaluate run 12.
Earlier cohort run 12 results were inspected. Disclose exploratory follow-up status;
never call this a fresh untouched confirmatory test. No favorable subset/test tuning.
Record participant/run/trial IDs and preserve chronological order.

## Mathematics
A=standardized training bandpower, A=UΣVᵀ. Each unseen row projects as z=aV_r.
Reconstruct standardized features with A_hat=AV_rV_rᵀ; relative Frobenius error
||A−A_hat||/||A||. Centered numerical rank ≤ min(N_train−1,128).
Kernel exp(−gamma||z_i−z_j||²), soft-margin SVM with regularization C.
SVD compresses bandpower, not raw EEG reconstruction. Variance need not discriminate.
SVD+SVM is established; justify the evaluation contribution through the literature.

## Required measures
Accuracy, balanced accuracy, macro/per-class precision/recall/F1, confusion matrices,
participant and pooled scores, participant confidence intervals and paired differences.
Include development-trained majority prediction baseline. Report all eligible subjects.
Record dimensions, 128/r reduction, reconstruction and singular spectra, float dtype,
parameter-array storage, serialized pipeline sizes and total personalized-model storage.
Separate fixed processing, feature extraction, learned representation fitting/transform,
SVM fitting/prediction. Repeat timings after warm-up; distinguish amortized batch
throughput from single-trial latency. Record actual environment/hardware availability.

## Software and acceptance
Python package, config, CLI, deterministic tests, separate audit/validate/evaluate/report
commands, no silent downloads/evaluation/overwrite. Hash manifests and saved selections.
Write predictions, search history, parameters, results and environment. Report failure
status and do not present partial experiments as complete. Synthetic fixtures are for
tests only. Test real ingestion when accessible; otherwise explain the restriction.
Generate faculty material only from real saved results, with pending labels otherwise.
Provide Windows instructions and study material. Code completion does not imply completed
EEG experiments or a submission-ready empirical paper without data execution.
