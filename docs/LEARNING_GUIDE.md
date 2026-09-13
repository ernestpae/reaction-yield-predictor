# Learning guide and interview preparation

This is a study route through the current implementation, not a claim that every topic is already mastered. Read the code, change a small example, and explain the result in your own words.

## 1. Understand the data before the model

Start with `data/README.md` and `examples/regenerate_data.py`. Derive what happens to the expected yield when time increases by one hour at fixed temperature and concentration. Why does the generator favor linear regression? Why does a yield column not make this a chemically validated dataset?

## 2. Trace the evaluation boundaries

Read `data.py`, then follow `train()` in `experiment.py`:

1. Validate the table and split rows into training and holdout sets.
2. Search model candidates using five folds drawn only from training rows.
3. Refit the winning pipeline on the full training partition.
4. Evaluate that pipeline and a training-mean baseline on the holdout.
5. Save diagnostics without changing the selected model.

The training CV score used to choose a candidate is not an unbiased estimate of its future performance. The holdout provides a separate estimate, but only for this dataset and split. Repeatedly revising a project after looking at that holdout weakens its independence. These reports are educational benchmarks, not a prospective validation study.

**Exercise:** Explain why computing a scaler's mean on all rows before cross-validation leaks information. Inspect `test_holdout_labels_cannot_change_selected_model`: changing only held-out labels changes the measured error but must not change training CV results.

## 3. Explain the models without jargon

- The dummy model predicts the training mean. A more complex method must justify itself against this baseline.
- Linear regression estimates additive feature effects by minimizing squared residuals.
- Ridge adds a penalty on coefficient magnitudes. Scaling makes that penalty less dependent on units. Alpha controls its strength.
- A random forest averages decision trees. Depth and minimum leaf size control flexibility. Forest predictions generally cannot extrapolate beyond the target values represented in their leaves.

A more complicated model is not automatically a better fit for the scientific question or sample size. Inspect all nine candidate configurations in `metrics.json`; do not choose a model merely for its name.

## 4. Interpret errors correctly

MAE is the mean absolute error, in yield percentage points. RMSE is also in percentage points and weighs large errors more strongly. R² compares squared error with variation around the evaluation target mean; it can be negative and is undefined for a constant target. It is not a probability, an accuracy percentage, or a measure of chemical understanding.

Residuals here are target minus prediction. Look for curvature, changing spread, or systematic bias. Twenty held-out points are too few to establish reliability across chemistry.

## 5. Ask what changes outside the training domain

Run both split strategies. The high-temperature experiment holds the upper temperature range out of training. Its internal CV is still shuffled training CV, so it does not directly optimize extrapolation. This is a simple condition-domain stress test, not a molecular scaffold split.

Inference flags inputs outside each training feature's min/max. Being inside all three ranges is only a coarse check: a novel combination can still be unfamiliar. A flag is not a calibrated uncertainty interval. Predictions are not clipped; implausible values remain visible.

## 6. Treat feature importance as a diagnostic

Permutation importance shuffles one input column and measures the increase in test MAE. This asks how much the fitted predictor relies on that column for this evaluation sample. It does not prove causality, mechanism, or universal chemical importance. Correlated inputs can complicate interpretation. The reported shuffle standard deviation is not a confidence interval for generalization error.

## 7. Relate the project to future chemistry work

Yield estimation could eventually support synthetic route development, including preparation of candidate compounds. That is different from predicting potency, toxicity, binding affinity, or drug efficacy. Before making drug-discovery claims, this project needs suitable experimental reaction data, molecular representations, and evaluations across unseen reaction families or relevant chemical groups.

Potential next steps, in order:

- Obtain an appropriately licensed experimental dataset and write its provenance/data card.
- Add reaction identities and group-aware splitting to prevent near-duplicate leakage.
- Study molecular descriptors or reaction fingerprints; justify what information they represent.
- Compare them with strong simple baselines before considering deep learning.
- Investigate uncertainty and independent external validation.

## Development transparency

Ernest created the original data-generation and regression exercise. The v0.2 refactor, tests, and documentation were developed with AI assistance. Learning-note comments explain design choices; they are not a fabricated timeline or evidence of independent mastery. A useful next contribution is to reproduce the results and implement one well-understood improvement with a clear explanation.
