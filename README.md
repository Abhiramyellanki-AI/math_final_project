# EEG end-semester: SVD + RBF Kernel SVM

**Status: full 105-participant local EEG experiment completed and verified (22 November 2026).**
This repository contains only the standalone SVD + RBF Kernel SVM implementation and its
matched no-SVD RBF SVM baseline. The [earlier project is preserved on an archive branch](https://github.com/Abhiramyellanki-AI/math_end_sem/tree/archive/hosvd-rks-original).

## Measured results at a glance (Exploratory follow-up, Run 12, N = 105)

- **Audit (S002–S109; S001 pilot excluded)**: **105 eligible** participants (4,723 total
  trials across 315 recordings: Run 4 = 1,575; Run 8 = 1,573; Run 12 = 1,575); **3 excluded**
  objectively for `sfreq_mismatch: 128.0` (S088, S092, S100).
- **RBF without SVD (baseline)**: **56.44%** participant-mean accuracy (95% CI [53.67%, 59.22%]),
  56.47% balanced accuracy, **0.5376** macro F1, **3.61 MB** total serialized storage (105 models).
- **SVD + RBF**: **52.63%** participant-mean accuracy (95% CI [49.94%, 55.33%]),
  52.69% balanced accuracy, **0.5038** macro F1, **1.39 MB** total serialized storage (105 models;
  **−61.5%** storage reduction).
- **Training-majority baseline**: **49.84%** accuracy (95% CI [49.19%, 50.49%]), **0.3323** macro F1.
- **Paired SVD minus baseline (N = 105)**: **−3.81 pp** accuracy (95% CI [−6.47, −1.14] pp),
  **−3.78 pp** balanced accuracy, **−0.0337** macro F1 (95% CI [−0.0621, −0.0053]).
- **Key conclusion**: Unsupervised SVD did **not** improve accuracy over the matched no-SVD
  baseline, and **neither method achieved the 75% target**. SVD reduced serialized model storage
  by 61.5% while leaving end-to-end warm feature+classifier batch time essentially unchanged
  (~16.92 vs ~16.99 ms/15-trial batch; Welch bandpower accounts for >93%).

**Run 12 has been inspected in earlier work. This is exploratory follow-up evaluation,
not a fresh untouched confirmatory test.** Personalized models assess new recordings of
known participants, not unseen participants.

## Windows quick start

Install Python 3.11 or newer (verified with Python 3.13.6). In PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,slides]"
.\.venv\Scripts\python.exe -m pytest -q
```

Edit `configs/experiment.yaml` if your dataset moved. Default:
`C:/Users/Abhiram/Downloads/eegmmidb-1.0.0/files` (expected layout `files/S002/S002R04.edf`, R08, R12).
No downloads occur automatically.

## Execute the experiment

Every `--out` must be a NEW directory (overwrites are refused):

```powershell
.\.venv\Scripts\python.exe -m eeg_svd.cli audit-data --out outputs/audit
.\.venv\Scripts\python.exe -m eeg_svd.cli validate --manifest outputs/audit/manifest.json --out outputs/validation
.\.venv\Scripts\python.exe -m eeg_svd.cli evaluate --validation outputs/validation --out outputs/evaluation
.\.venv\Scripts\python.exe -m eeg_svd.cli build-report --experiment outputs/evaluation --out outputs/report
.\.venv\Scripts\python.exe -m eeg_svd.cli plot-eeg --subject 2 --out outputs/eeg_example
```

## Outputs

- `outputs/audit/manifest.json`: participant eligibility, reasons, class/trial counts, SHA-256 provenance.
- `outputs/validation/search.json` & `lock.json`: all development candidates/folds and tamper-evident selections.
- `outputs/evaluation/results.json`: classification, uncertainty, timings, storage, reconstruction, environment, paired differences.
- `outputs/evaluation/predictions.csv` & `participant_metrics.csv`: all 4,725 predictions and 315 participant-metric rows.
- `outputs/evaluation/models/`: 210 trained participant-specific `.joblib` pipelines (105 SVD+RBF, 105 baseline).
- `outputs/report/`: measured figures (`accuracy.png`, `confusion_matrices.png`, `svd_diagnostics.png`), `measured_summary.md`, `final_results_summary.md`, and `results.tex`.
- `outputs/eeg_example/eeg_example.png`: illustrative Subject 2 run 4 EEG trace.
- `outputs/environment.txt`: frozen local Python environment (`pip freeze`).

## Faculty deliverables

- **IEEE paper source**: [paper/main.tex](paper/main.tex) and [paper/references.tex](paper/references.tex) (updated with all measured results; compile with Overleaf IEEEtran or local `pdflatex main.tex` twice; fill student names/USNs placeholders).
- **Measured PowerPoint**: [deliverables/presentation_measured.pptx](deliverables/presentation_measured.pptx) (15 main slides = 20 min + 7 viva backups, with embedded measured EEG, accuracy, confusion-matrix, and SVD diagnostic plots).
- **Study & viva guide**: [study/guide.md](study/guide.md) (updated with measured numbers and viva Q&A).
- **Literature survey**: [literature/survey.md](literature/survey.md) (16 DOI-verified references).
- **Validation record**: [VALIDATION.md](VALIDATION.md).
- **Final submission folder**: `submission_package/` (ready-to-submit bundle excluding raw EDFs and heavy `.joblib` weights).

Regenerate slides if needed:

```powershell
.\.venv\Scripts\python.exe scripts/build_slides.py --report outputs/report --out deliverables/presentation_measured.pptx
```
