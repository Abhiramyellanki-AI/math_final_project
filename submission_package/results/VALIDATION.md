# Verification record — 22 November 2026

## Local Windows verification & full-cohort execution

Environment: Windows 11 (`Windows-11-10.0.26200-SP0`), Python 3.13.6, AMD64 (16 logical CPUs),
NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, MNE 1.13.2, pandas 3.1.1, joblib 1.6.0,
pytest 9.2.0, ruff 0.17.6, python-pptx 1.0.2 (full freeze in `outputs/environment.txt`).

### Code & test checks
- `python -m compileall -q src`: **passed**.
- `python -m ruff check src tests scripts`: **passed**.
- Synthetic unit test suite (`pytest -q`, `EEG_TEST_DATA_DIR` unset): **24 passed, 1 skipped** (explicitly skipped optional real-EDF test when env var unset).
- Optional real-EDF integration test (`EEG_TEST_DATA_DIR=C:\Users\Abhiram\Downloads\eegmmidb-1.0.0\files`): **1 passed** (verified Subject 1 runs 4/8/12 ingestion, 15 epochs × 64 × 320, finite bandpower, fit on run 4, predict shape on run 8; no run 12 accuracy reported).
- Total tests verified: **25/25 passed**.
- Minor pre-run fixes applied: tightened boolean rejection in `config.py` (`c_values`/`gammas`) with matching tests in `tests/test_project.py`, added header `\hline` in `reporting.py` LaTeX table, and fixed author diacritics/punctuation in `paper/references.tex`.

### Full-cohort EEG workflow executed
1. `audit-data --out outputs/audit`: **completed** — 105 eligible participants (S002–S109 minus S088, S092, S100 excluded for `sfreq_mismatch: 128.0`; S001 prespecified pilot exclusion); 315 recordings, 4,723 trials (Run 4: 1,575; Run 8: 1,573; Run 12: 1,575).
2. `validate --manifest outputs/audit/manifest.json --out outputs/validation`: **completed** — recording-swapped 4↔8 development selection locked for all 105 participants in `lock.json` (dev mean macro F1: SVD+RBF 0.6267, baseline 0.5406).
3. `evaluate --validation outputs/validation --out outputs/evaluation`: **completed** (`status.json` = `complete`, SHA-256 hashes verified) — 210 trained `.joblib` pipelines saved, 4,725 predictions (`predictions.csv`), 315 participant-metric rows (`participant_metrics.csv`), complete `results.json`.
4. `build-report --experiment outputs/evaluation --out outputs/report`: **completed** — generated `accuracy.png`, `confusion_matrices.png`, `svd_diagnostics.png`, `measured_summary.md`, `results.tex`.
5. `plot-eeg --subject 2 --out outputs/eeg_example`: **completed** — generated real Subject 2 run 4 plot (`eeg_example.png`).
6. `scripts/build_slides.py --report outputs/report --out deliverables/presentation_measured.pptx`: **completed** — 22 slides (15 main = 20.0 min + 7 backups) with embedded measured figures.

### Measured primary outcomes (Exploratory follow-up, Run 12, N = 105)
- **RBF without SVD**: 56.44% accuracy (95% CI [53.67%, 59.22%]), 0.5376 macro F1, 3,612,293 B total storage.
- **SVD + RBF**: 52.63% accuracy (95% CI [49.94%, 55.33%]), 0.5038 macro F1, 1,390,149 B total storage (−61.5%).
- **Training-majority**: 49.84% accuracy, 0.3323 macro F1.
- **Paired SVD − baseline**: −3.81 pp accuracy (95% CI [−6.47, −1.14] pp), −0.0337 macro F1 (95% CI [−0.0621, −0.0053]).
- **75% target**: Not achieved by either method.

### Remaining environment note
- Local LaTeX (`pdflatex`/`xelatex`/`lualatex`) is not installed on this machine; `paper/main.tex` and `paper/references.tex` are finalized and ready to compile via Overleaf (IEEEtran) or local MiKTeX/TeX Live. Unknown student names/USNs remain explicit placeholders.
