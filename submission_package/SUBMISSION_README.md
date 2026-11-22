# Final Submission Package — SVD + RBF Kernel SVM for EEG Motor Imagery

This folder contains the complete, self-contained faculty submission artifacts generated from the local 105-participant PhysioNet eegmmidb v1.0.0 experiment. Raw EEG `.edf` recordings and serialized `.joblib` model weights (210 files in `../outputs/evaluation/models/`) are intentionally excluded to keep the package lightweight.

## Contents

- `paper/`
  - `IEEE_Paper_SVD_RBF_SVM.docx` — Ready-to-submit IEEE conference paper in Microsoft Word (`.docx`) format with all 10 sections, 4 embedded plots, 9 comprehensive insight tables, and 16 DOI-verified references (fill `[Student Names and USNs]` before final export).
  - `main.tex` — Complete IEEE conference LaTeX source with all measured results, tables, and figure references.
  - `references.tex` — 16 DOI-verified IEEE bibliography entries.
  - `results.tex` — Auto-generated LaTeX summary table from `build-report`.
- `presentation/`
  - `presentation_measured.pptx` — 22-slide editable PowerPoint (15 main slides = 20 min + 7 viva backups) with embedded EEG, accuracy, confusion-matrix, and SVD diagnostic plots plus speaker notes.
  - `outline.md` — Matching slide-by-slide outline and speaker notes.
- `study/`
  - `guide.md` — Beginner-friendly study guide, code map, and viva Q&A grounded in the measured 105-participant results.
- `literature/`
  - `survey.md` — 16-reference comparison table with roles and scope limitations.
  - `verified_metadata.json` — Crossref DOI metadata snapshot.
- `figures/`
  - `eeg_example.png` — Subject 2 run 4 C3 raw vs. 8–30 Hz filtered trace and first left/right trials.
  - `accuracy.png` — Participant-mean run 12 accuracy with 95% participant $t$-intervals ($N=105$).
  - `confusion_matrices.png` — Pooled run 12 confusion matrices (1,575 trials).
  - `svd_diagnostics.png` — Cumulative training spectral energy and development vs. run 12 reconstruction error.
- `results/`
  - `final_results_summary.md`, `measured_summary.md`, `VALIDATION.md`
  - `participant_metrics.csv`, `predictions.csv`, `results.json`, `status.json`, `lock.json`, `environment.txt`

## How to Compile the IEEE Paper to PDF

Local LaTeX (`pdflatex`) was not installed on the execution machine. To compile `paper/main.tex`:

1. **Overleaf (easiest)**: Upload `paper/main.tex`, `paper/references.tex`, and the `figures/` folder (keeping `figures/` as a sibling or placing PNGs in the project root and updating paths if flattened), select `pdfLaTeX`, fill in your names/USNs, and compile.
2. **Local MiKTeX / TeX Live** (from inside `submission_package/paper/`):
   ```powershell
   pdflatex -interaction=nonstopmode -halt-on-error main.tex
   pdflatex -interaction=nonstopmode -halt-on-error main.tex
   ```

## How to Reproduce the Experiment

From the repository root (with PhysioNet eegmmidb v1.0.0 at `C:\Users\Abhiram\Downloads\eegmmidb-1.0.0\files`):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,slides]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m eeg_svd.cli audit-data --out outputs/audit
.\.venv\Scripts\python.exe -m eeg_svd.cli validate --manifest outputs/audit/manifest.json --out outputs/validation
.\.venv\Scripts\python.exe -m eeg_svd.cli evaluate --validation outputs/validation --out outputs/evaluation
.\.venv\Scripts\python.exe -m eeg_svd.cli build-report --experiment outputs/evaluation --out outputs/report
.\.venv\Scripts\python.exe -m eeg_svd.cli plot-eeg --subject 2 --out outputs/eeg_example
.\.venv\Scripts\python.exe scripts/build_slides.py --report outputs/report --out deliverables/presentation_measured.pptx
```
