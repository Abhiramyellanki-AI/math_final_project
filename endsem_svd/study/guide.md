# Study and viva guide

## The project in one minute
We classify the experiment's left/right motor-imagery labels from scalp EEG. Each
trial has 64 channels and 320 time samples. We turn its signal into 128 bandpower
features. A scaler learns training means and standard deviations. SVD learns a
smaller coordinate system from standardized training features. An RBF SVM learns
the decision boundary in those coordinates. We compare the result against the
same SVM without SVD, using identical recordings. All new accuracy results are
pending until the local experiment runs.

## Dataset: what to show the faculty
Show the PhysioNet source page, the local subject directories and the audit manifest.
An EDF file is a recording container: channel metadata, sampling frequency, signal
samples and annotations. It is not a spreadsheet and does not need conversion to
Excel. MNE reads it directly. A subject is a participant; a run is one recording;
a trial is the post-cue interval extracted for one imagined movement.

For runs 4, 8 and 12 ONLY, T1 means imagined left fist and T2 means imagined right
fist. T0 is rest. The same annotation names have different meanings in other runs.
Show a real trace with the plot-eeg command. A waveform is not a visible picture
of a thought; the classifier uses statistical patterns.

## Signal processing
160 Hz means 160 samples per second per channel. Two seconds gives 320 samples;
64×320=20,480 values per trial. EEG is stored in volts. Multiply by one million
only when plotting in microvolts.

Filtering keeps 8–30 Hz. Mu is 8–13 Hz and beta is 13–30 Hz. Imagined movement can
change sensorimotor rhythmic power, but this relationship varies across people.
The fixed [0.5,2.5) interval excludes the first half-second after the cue. It does
not prove that all visual-cue artifacts have been removed. Zero-phase filtering
uses future samples and is appropriate here only for offline processing.

Welch estimation divides the trial into overlapping Hann-windowed segments and
averages their periodograms. Integrating PSD in a band gives bandpower. We apply
log10 to reduce dynamic range. Example: doubling signal amplitude multiplies power
by four, so its log10 power rises by log10(4), approximately 0.602. Each channel
contributes two bandpower values: 64×2=128 features.

## Standardization and SVD
The feature matrix F has rows=trials and columns=128 spectral features. Training
mean μ_j and standard deviation s_j produce A_ij=(F_ij−μ_j)/s_j. Test rows use
those SAME means/scales, not newly fitted values.

A=UΣVᵀ. Columns of V are orthogonal directions in feature space. Singular values
in Σ measure energy along those directions. Choose r directions and form Z=AV_r.
An unseen trial can use V_r. It cannot use an arbitrary training row of U.

Tiny example: A=[[3,0],[0,1]] has singular values 3 and 1. Keeping rank 1 preserves
9/(9+1)=90% of its squared Frobenius energy. Its relative reconstruction error is
sqrt(1/10), approximately 0.316. This is a mathematical example, not EEG evidence.

For centered data with 15 training trials, numerical rank is at most 14, regardless
of the 128 input features. The code restricts candidates using both training-fold
rank limits. Retaining high variance does not ensure good class separation: a
small-variance direction could be highly informative about left versus right.

Reconstruction is A_hat=AV_rV_rᵀ. This reconstructs standardized bandpower, not
raw EEG waveforms. The dimension reduction ratio is 128/r. Deployed storage must
also include V_r, the scaler and the SVM, so smaller features do not necessarily
mean a smaller complete model.

## SVM and kernel
A linear decision function is wᵀz+b. SVM balances a wide margin against violations:
min 0.5||w||²+CΣξ_i, subject to y_i(wᵀφ(z_i)+b)≥1−ξ_i and ξ_i≥0.
For this equation, convert binary class labels to −1/+1; scikit-learn accepts 0/1.
Support vectors are training examples with nonzero coefficients in the solution.

The kernel K(z,z')=exp(−gamma||z−z'||²) computes similarity without explicitly
constructing an infinite-dimensional feature vector. Large gamma makes similarities
more local; small gamma produces broader influence. Large C penalizes training
errors more strongly. Neither parameter is universally better when increased.
The default 'scale' gamma is computed from the training inputs at the SVM step.

The kernel is fixed to RBF for a manageable faculty-approved scope. We compare
SVD versus no SVD, not a large competition between kernel types.

## Evaluation: the important viva answer
Each participant has their own model. First train on run 4 and validate on run 8;
then reverse. Select hyperparameters from the mean validation macro F1. Refit using
both development runs and evaluate run 12. Never split samples from one trial
across training and validation. Every scaler/SVD is refitted within each fold.

Why not claim an untouched test? Earlier work already inspected run 12 outcomes.
This redesigned experiment is explicitly exploratory follow-up. Rewriting code
cannot undo exposure. Why exclude Subject 1? It was the original small pilot.
Why keep all other eligible participants? Removing low scorers would bias evaluation.
Why personalized models? Their EEG patterns can differ. The resulting claim is
new recordings of known participants, not generalization to unseen participants.

## Metrics
Accuracy is correct predictions divided by all predictions. Precision measures
how often predicted members of a class are correct. Recall measures how many
actual members were found. F1 is their harmonic mean. Macro averaging gives each
class equal weight. Balanced accuracy is the mean recall across classes.

A confusion matrix has true labels on rows and predictions on columns. With labels
[left,right], [[6,2],[1,6]] means 12/15 correct, or 80%. This is a worked example,
not an experimental result. With only 15 test trials, one prediction changes one
participant's accuracy by 6.67 percentage points.

Participant mean gives participants equal weight. Pooled accuracy weights by trial
count. A 95% participant t interval describes uncertainty across participant units;
it is not the standard deviation of the two validation folds. With one participant,
the software reports no participant confidence interval. The paired comparison
subtracts baseline from SVD for each same participant before estimating its interval.
Overlapping intervals do not prove equivalence. A 75% goal is not a measured result.

## Timing and memory
Report fixed filtering/loading separately from feature extraction and classification.
A 10× faster classifier may yield only a small total improvement if feature extraction
dominates. Warm repeated timings are summarized by median and range. Batch time
per trial is an amortized figure, not directly measured real-time single-trial latency.
Actual serialized pipeline size differs from summing selected numerical arrays and
from whole-process RAM. Personalized models require storing one model per participant.

## Code map
- config.py: fixed protocol, JSON and hashes.
- data.py: local paths, independent recording reads, eligibility and provenance.
- features.py: Welch log-bandpower.
- svd.py: training basis, projection and reconstruction.
- models.py: fold fitting and deterministic selection.
- evaluation.py: saved selections, explicit evaluation, timing and statistics.
- reporting.py: figures from saved evidence; no training.
- cli.py: user commands and error reporting.
- tests/test_project.py: mathematical and synthetic workflow checks.

## Viva questions and defensible answers
1. What is new? Our specific rank/trade-off investigation under a documented EEG
   protocol; SVD and SVM themselves are not new algorithms.
2. Why SVD? It offers an explicit orthogonal low-rank approximation with interpretable
   singular values and reconstruction error.
3. Why could it fail? Variance preservation is unsupervised and may remove discriminative
   information. Few development trials also make rank selection uncertain.
4. Why a baseline? Without the same RBF SVM without SVD, we cannot attribute changes
   to the projection.
5. Did you reach 75%? Answer using the measured all-participant results only.
6. Is this clinical or real-time? No; it is offline research with non-causal filtering.
7. Why not report the best participant? It would not represent the whole cohort.
8. Can you prove no leakage? Explain the design and tests; do not claim that tests
   prove every conceivable source of bias is absent.
9. Does more retained energy imply higher accuracy? No, energy and class information
   are different objectives.
10. What if accuracy falls? That is evidence against this particular representation
    under this protocol; report the trade-offs and limitations honestly.

## Before submission
Run the actual experiment, inspect all exclusions and warnings, check numbers against
results.json, replace pending prose, fill identities, read the primary references,
rehearse the 20-minute talk and practice explaining one trial through the full pipeline.
Do not claim independent work you did not perform; follow institutional AI-use rules.

## Individual contribution log (complete truthfully)
| Student / USN | Work actually performed | Evidence / files | Can explain in viva? |
|---|---|---|---|
| [Name / USN] | [Fill after reviewing and executing] | [Commit/output] | [Yes/No] |
