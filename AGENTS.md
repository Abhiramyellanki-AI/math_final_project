# Codex instructions — P3 EEG research project

Read `PRD.md` before writing or changing code. Implement **one milestone at a time** and stop after each milestone with: summary, file changes, commands run, test results, assumptions, and next suggested milestone. Do not implement M0–M6 in one pass.

## Non-negotiable rules
1. Never fabricate EEG accuracy, time, compression, plots, benchmark numbers, or experiment completion.
2. Do not fit scalers, HOSVD projections, hyperparameters, or signal-derived learned preprocessing using held-out run 12. Tune using only runs 4 and 8.
3. Preserve run IDs, subject IDs and labels. Verify T1/T2 mappings for runs 4, 8, 12; exclude rest (T0).
4. Use offline data from the PhysioNet EEG Motor Movement/Imagery dataset; do not claim real-time EEG, Neuralink, clinical use, or mind-reading.
5. All model pipelines must share the same held-out split and report full preprocessing + classifier time separately.
6. Keep large EEG downloads, generated figures, caches and secrets out of Git.
7. Include type hints/docstrings and small executable tests for every important transformation.
8. A trial must produce exactly one feature vector for SVM. Never use full-dataset trial-mode HOSVD factors as if they were an out-of-sample feature extractor.
9. For HOSVD, fit channel/time bases using training trials only; use fixed bases to project held-out trials.
10. RKS (`RBFSampler`) may increase dimensions; never label it inherently dimensionality reduction.

## Development loop
- Before coding: state the milestone goal and files to edit.
- Write minimal code that satisfies the milestone, not extra infrastructure.
- Add or update tests. Run them; report exact commands and results.
- Verify real data output only after actual execution; if dataset download unavailable, state limitation and test with labeled synthetic fixtures.
- End with a short human checkpoint and await the next instruction.

## Definition of done per milestone
A runnable command, clear README instructions, applicable tests passing, and no claim unsupported by output.
