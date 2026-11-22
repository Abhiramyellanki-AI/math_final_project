# Verification record — 22 November 2026

## Executed checks

- Core suite before adding optional integration test: **22 passed**.
- Full suite with an existing local Subject 1 EDF cache: **23 passed**.
- Ruff checks of `src`, `tests`, and `scripts`: **passed**.
- Python syntax compilation: **passed**.
- IEEE paper compilation with IEEEtran: **passed**, 3-page PDF; references resolved.
  First page visually inspected; LaTeX reported no overfull boxes on final compilation.
- PowerPoint generation: **passed**, 22 slides = 15 main + 7 backup; scheduled main
  speaking time totals 20 minutes. Text/notes verified structurally; no PowerPoint
  rendering application was available for a complete visual inspection.
- 16 bibliography identities verified through Crossref DOI metadata. The literature
  guide distinguishes identity verification from a complete critical reading.

## What was actually tested

Synthetic tests cover spectral behavior, numerical rank and reconstruction,
training-only transformations, run-12 rejection during model selection, lock
integrity, objective exclusions, no downloads, no overwrites, metrics, and the
complete audit/selection/evaluation/report workflow. Synthetic scores are not
reported as EEG experiments.

The optional real integration check used previously cached S001R04/R08/R12 EDFs.
It verified 15 epochs of shape 64×320 per recording, valid labels and finite
bandpower, then fitted both methods on run 4 and checked prediction shape on run 8.
It did not report classification accuracy or classify run 12. Subject 1 remains
excluded from the proposed primary cohort.

## Not executed / not claimed

The complete S002–S109 dataset is on the user's Windows machine and was not available
in this environment. No full-cohort audit, model selection, final evaluation, 75%
accuracy result, or statistical superiority claim has been generated. The draft
paper and slides cannot be treated as a completed experimental submission.

## Commands

```text
python -m compileall -q src
python -m pytest -q
python -m ruff check src tests scripts
# Set EEG_TEST_DATA_DIR to the existing EDF root to enable the optional real test.
python scripts/build_slides.py
# In paper/, with IEEEtran installed:
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The optional real test skips with an explicit reason when EEG_TEST_DATA_DIR is unset.
The Python package versions used for verification are recorded in
`verification_environment.json`. Local experiment runtime versions are saved
independently by the evaluation code.
