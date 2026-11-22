# Measured exploratory evaluation

Exploratory follow-up: run 12 results were inspected in earlier work. Personalized models assess new recordings of known participants, not unseen users.

Participants: 105

| Method | Participant mean accuracy | Macro F1 | Total pipeline bytes |
|---|---:|---:|---:|
| RBF without SVD | 56.44% | 0.5376 | 3612293 |
| SVD + RBF | 52.63% | 0.5038 | 1390149 |
| Training-majority | 49.84% | 0.3323 | 0 |

Paired SVD minus baseline differences (participant unit):

- accuracy: mean -0.0381; 95% CI [-0.06474418553164975, -0.011446290658826418]
- balanced_accuracy: mean -0.0378; 95% CI [-0.06412543977378912, -0.011554832335054416]
- macro_f1: mean -0.0337; 95% CI [-0.062146582365488555, -0.005325994954597547]

Repeated warm feature+classifier batches; excludes fixed filtering/loading. Amortized time is not measured single-trial latency.

Parameter-array estimate and uncompressed joblib pipeline files, not process RAM.

No statistical equivalence or 75% achievement is assumed. See participant_metrics.csv for all measures.
