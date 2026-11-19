# How to build P3 with Codex — practical workflow

## 1. Set up the workspace
1. Make a new empty local folder/repository named `p3-eeg-project`.
2. Copy `PRD.md` and `AGENTS.md` into its root.
3. Open that folder in Codex (IDE extension or CLI) and keep the PRD visible.
4. Ask Codex for **M0 only**; review diffs and tests before proceeding.

## 2. First prompt to paste into Codex
```text
Read PRD.md and AGENTS.md completely. We are implementing milestone M0 only (repository scaffold). First summarize requirements, data-leakage constraints, and acceptance criteria in 10 bullets or fewer. Then create the minimum Python package structure, dependency configuration, CLI entry point, README setup instructions, and a smoke test. Do not download EEG data, implement classifiers, or claim measured results in M0. Run available tests and show exact outputs. Stop after M0 and tell me how to verify it locally.
```

## 3. Prompts for subsequent milestones (one at a time)

### M1 — Real EEG ingestion
```text
Implement M1 from PRD.md only. Use MNE to load PhysioNet subject 1, runs 4/8/12 separately, confirm metadata and T1/T2 annotations, filter each run at 8–30 Hz, extract fixed 0.5–2.5 s post-cue epochs, map T1->0 and T2->1, preserve group/run IDs, and prepare train-development runs 4+8 and untouched test run 12. Provide a CLI data inspection command, plots, and unit tests for shape/labels/held-out split. Do not fit any ML models. Execute on real data if downloads work and report actual counts; otherwise state why you cannot. Stop and request review.
```

### M2 — Baselines
```text
Implement M2 only. Extract deterministic per-channel mu/beta log-bandpower trial features with documented settings. Build LinearSVC and RBF SVC baselines with training-fold-only StandardScaler. Use run-aware validation on runs 4 and 8; keep run 12 untouched. Add tests and export measured training-only validation metrics, without claiming final-test performance. Stop for review.
```

### M3 — HOSVD
```text
Implement M3 only. Fit channel/time HOSVD subspaces from training trials (no test data); project each individual trial using U_c.T @ X_i @ U_t, flatten to rank-channel × rank-time features, then train a linear SVM with training-only scaling. Add rank sweep, reconstruction error, feature and storage accounting, and tests proving no out-of-sample leakage. Do not compute trial-mode factors on the full dataset. Stop for review.
```

### M4 — RKS
```text
Implement M4 only. Apply standardized shared log-bandpower features -> RBFSampler -> LinearSVC, with fixed seeds. Expose n_components, gamma, C; tune using training-only run-aware validation. Add shape and reproducibility tests, timing and dimensions. Keep test run 12 unused. Stop for review.
```

### M5 — Locked comparison
```text
Implement M5 only. Choose final model settings from run-aware validation on runs 4 and 8, refit each pipeline on runs 4+8, and evaluate each once on untouched run 12. Export accuracy, macro precision/recall/F1, confusion matrices, per-class results, fit/transform/predict times, feature dimensions, measured rank/error statistics, software versions, random seeds, and configurations. Document the limitations of a single-subject, within-subject cross-run evaluation and differing model inputs. Never invent results. Stop for review.
```

### M6 — Faculty-ready report
```text
Implement M6 only. Use actual outputs from M5 to generate publication-style figures, a concise results summary, limitations, a research-methodology diagram, and a viva explanation of HOSVD vs Random Kitchen Sinks. Include citations/links to the dataset and core software documentation. If results are unavailable, leave clear TODOs; never fabricate numbers. Stop for review.
```

## 4. Human checkpoints after every phase
- Read the generated code, not just the summary.
- Run `pytest` or the project's documented test command.
- Inspect array shapes, labels, run IDs, and any sample plots.
- Check any fit/transform code to confirm held-out data never influences learning.
- Commit milestone only after it runs on your machine.
- Record run logs, configuration, and software versions for reproducibility.

## 5. Two caution flags
**Validation is weak with only two development runs.** Treat the first experiments as a prototype; strengthen later with more subjects/runs or additional prespecified evaluation.

**HOSVD and RKS transform different representations.** Report them as end-to-end pipeline comparisons unless implementing a controlled matched-input ablation.
