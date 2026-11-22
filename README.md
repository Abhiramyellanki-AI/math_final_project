# EEG End-Semester Project — SVD + RBF Kernel SVM

**Start with the new standalone project on the `endsem-svd-svm` branch.**
The default `main` branch preserves the original project; it does not contain the new package.

[Open the new project](https://github.com/Abhiramyellanki-AI/math_end_sem/tree/endsem-svd-svm/endsem_svd) · [Download the project ZIP](https://github.com/Abhiramyellanki-AI/math_end_sem/archive/refs/heads/endsem-svd-svm.zip) · [Setup and execution instructions](https://github.com/Abhiramyellanki-AI/math_end_sem/blob/endsem-svd-svm/endsem_svd/README.md)

## Current project

Classify imagined left-fist versus right-fist movement from PhysioNet EEG recordings:

**Mu/beta bandpower → training-fitted standardization → SVD → RBF Kernel SVM**

The comparison baseline uses the same features and RBF SVM without SVD. The new
implementation uses participant-specific models, selects parameters on runs 4 and 8,
and explicitly evaluates run 12 as exploratory follow-up because earlier results
from those recordings have already been inspected.

### What is ready

- Python package, configuration, CLI, local EDF audit, model selection and evaluation.
- Reporting, saved predictions, timing/storage accounting and reproducibility checks.
- **23 local tests passed**, including a cached real-EDF smoke test.
- [GitHub verification passed on Python 3.11 and 3.12](https://github.com/Abhiramyellanki-AI/math_end_sem/actions/runs/45032366528).
- IEEE paper draft, editable presentation, 16-reference literature survey and study guide.

### What is pending

**The new full-cohort experiment has not been run on the user's local dataset.**
There is no new measured accuracy result or demonstrated 75% accuracy. Paper and
slide results remain explicitly pending; they are not ready for empirical submission
until the experiment runs and its conclusions are reviewed.

## Open in VS Code or Antigravity

1. Download and extract the ZIP linked above.
2. Open the extracted **`endsem_svd`** folder, not the legacy repository root.
3. Follow its README to create a Python environment and install dependencies.
4. Check the dataset path in `configs/experiment.yaml`:

```text
C:\Users\Abhiram\Downloads\eegmmidb-1.0.0\files
```

No automatic dataset download occurs. The existing local EDF recordings are used.

Alternatively:

```powershell
git clone --branch endsem-svd-svm https://github.com/Abhiramyellanki-AI/math_end_sem.git
cd math_end_sem\endsem_svd
```

## Documents

| Resource | Link |
|---|---|
| New project README and commands | [Setup guide](https://github.com/Abhiramyellanki-AI/math_end_sem/blob/endsem-svd-svm/endsem_svd/README.md) |
| Requirements | [PRD](https://github.com/Abhiramyellanki-AI/math_end_sem/blob/endsem-svd-svm/endsem_svd/PRD.md) |
| IEEE paper draft | [PDF](https://github.com/Abhiramyellanki-AI/math_end_sem/blob/endsem-svd-svm/endsem_svd/deliverables/paper_draft.pdf) |
| Presentation draft | [PowerPoint](https://github.com/Abhiramyellanki-AI/math_end_sem/blob/endsem-svd-svm/endsem_svd/deliverables/presentation_draft.pptx) |
| Literature survey | [16 references](https://github.com/Abhiramyellanki-AI/math_end_sem/blob/endsem-svd-svm/endsem_svd/literature/survey.md) |
| Study and viva guide | [Guide](https://github.com/Abhiramyellanki-AI/math_end_sem/blob/endsem-svd-svm/endsem_svd/study/guide.md) |
| Verification details and limits | [Validation record](https://github.com/Abhiramyellanki-AI/math_end_sem/blob/endsem-svd-svm/endsem_svd/VALIDATION.md) |

## Preserved original project

The earlier HOSVD/RKS/Markov implementation remains intact. Its original description
and historical results are preserved in [LEGACY_README.md](LEGACY_README.md).
Those results are not measurements of the new SVD + Kernel SVM implementation.
