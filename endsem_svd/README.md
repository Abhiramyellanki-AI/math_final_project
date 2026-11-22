# EEG end-semester: SVD + RBF Kernel SVM

**Status: new implementation tested; full-cohort EEG experiments NOT executed here.**
No 75% accuracy claim or legacy experimental result is included. Paper/slides are
explicit drafts until the local experiment is completed and the results interpreted.
This standalone package does not import the original HOSVD/RKS project.

## Windows quick start

Install Python 3.11 or newer. In PowerShell:

```powershell
git clone --branch endsem-svd-svm https://github.com/Abhiramyellanki-AI/math_end_sem.git
cd math_end_sem\endsem_svd
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,slides]"
.\.venv\Scripts\python.exe -m pytest -q
```

If `py -3.12` is unavailable, use your installed supported Python version.
Activation is unnecessary, avoiding PowerShell execution-policy problems.
You can instead extract the GitHub branch ZIP and open `endsem_svd` in Antigravity.
Copy this directory to `C:\Users\Abhiram\OneDrive\Desktop\EEG_EndSem_SVD_SVM`
if you want that location; the program does not depend on its workspace path.

Edit `configs/experiment.yaml` if your dataset moved. Its default is:
`C:/Users/Abhiram/Downloads/eegmmidb-1.0.0/files`.
Expected layout: `files/S002/S002R04.edf` (also R08 and R12).
No downloads occur automatically.

## Execute the experiment

Run from this directory. Every `--out` must be a NEW directory.

```powershell
.\.venv\Scripts\python.exe -m eeg_svd.cli audit-data --out outputs/audit
.\.venv\Scripts\python.exe -m eeg_svd.cli validate --manifest outputs/audit/manifest.json --out outputs/validation
.\.venv\Scripts\python.exe -m eeg_svd.cli evaluate --validation outputs/validation --out outputs/evaluation
.\.venv\Scripts\python.exe -m eeg_svd.cli build-report --experiment outputs/evaluation --out outputs/report
.\.venv\Scripts\python.exe -m eeg_svd.cli plot-eeg --subject 2 --out outputs/eeg_example
```

Read the audit exclusions before training. Start with a separate smoke configuration
containing one or two participants and reduced grids if checking local installation.
Do not select that smoke subset for favorable accuracy. Keep smoke outputs distinct
from the full experiment; record any additional test exposure.

The audit performs mechanical quality checks on all required runs. Development loads
only runs 4 and 8. SVD rank, C, and gamma are chosen independently for each participant
using recording-swapped validation. Both pipelines share the spectral features and
splits. Evaluation refits using both development runs and uses run 12 once per saved
selection record. The `EVALUATION_STARTED.json` guard prevents accidental repetition.
After an interrupted evaluation, preserve the failed output/marker and investigate;
do not silently delete evidence or claim an untouched rerun.

**Run 12 has been inspected in earlier work. This is exploratory follow-up evaluation,
not a new confirmatory test.** Personalized models generalize across recordings of
known participants, not unseen participants. A 75% target is not guaranteed.

## Outputs

- `audit/manifest.json`: participant eligibility, reasons, class/trial counts,
  recording hashes and trial provenance.
- `validation/search.json`: all development candidates/folds and invalid ranks.
- `validation/lock.json`: selected parameters, configuration and manifest digest.
- `evaluation/results.json`: measured classification, uncertainty, timings, storage,
  reconstruction, environment, and paired participant differences.
- `evaluation/predictions.csv`: all predictions with labels, participant and trial IDs.
- `evaluation/participant_metrics.csv`: flattened per-participant metrics.
- `evaluation/models/`: complete participant-specific joblib pipelines. Only load
  joblib files from a trusted source; pickle-based formats can execute code.
- `report/`: saved-result figures, measured summary, and `results.tex` for the paper.

Timing separates loading/filtering, bandpower, representation and classifier costs.
Warm repeated feature-plus-classifier timings exclude fixed preprocessing. Dividing
batch time by trial count is amortized throughput, not single-trial latency.
Storage distinguishes parameter-array estimates, pipeline file sizes and total
storage across participant-specific models. Reconstruction is of standardized
bandpower, not of the original waveform.

## Faculty material

Ready-to-open drafts: [IEEE PDF](deliverables/paper_draft.pdf) and [PowerPoint](deliverables/presentation_draft.pptx). Both clearly mark empirical results as pending.


- [PRD](PRD.md): scope and acceptance requirements.
- [IEEE paper](paper/main.tex): complete methodological draft with explicit pending-results sections.
- [Literature](literature/survey.md): 16 sources, roles, limitations, and verified DOI metadata.
- [Slide content](slides/outline.md): 15 main slides plus viva backups.
- [Study guide](study/guide.md): explanations, equations, code map and viva answers.
- [Validation record](VALIDATION.md): checks actually executed in the cloud.

Generate an editable PowerPoint with speaker notes:

```powershell
.\.venv\Scripts\python.exe scripts/build_slides.py
# After completing the experiment, include measured figures:
.\.venv\Scripts\python.exe scripts/build_slides.py --report outputs/report
```

Compile the paper in Overleaf using the IEEE conference template (IEEEtran class),
or install a LaTeX distribution locally and run:

```powershell
cd paper
pdflatex main.tex
pdflatex main.tex
```

The draft includes `../outputs/report/results.tex` automatically when present. Read
and rewrite the abstract, results interpretation, comparison and conclusion after
measurement; numerical inclusion does not turn an unfinished paper into a finished one.
Fill student names/USNs and institutional details. Inspect the PDF and slides before
submission. Export PowerPoint to PDF locally if needed.

## Reproducibility and limitations

The initial grid is 6 ranks × 4 C values × 4 gamma choices for SVD and 16 baseline
candidates per participant, each evaluated on two folds. Invalid ranks are logged.
Centered training data have rank at most N_train−1, often around 14 for these files.
Training-set normalization and SVD prevent leakage but do not remove selection
variance from the small number of trials. SVD maximizes retained variance, not class
separation; it may hurt accuracy. Do not discard low-scoring participants or tune
using run 12. No clinical, online, or novel-algorithm claim is made.

Generated outputs and EDFs are excluded from Git. Share small measured CSV/JSON files
and figures in the final submission package after your local run. Freeze the actual
local environment with `python -m pip freeze > outputs/environment.txt`.
