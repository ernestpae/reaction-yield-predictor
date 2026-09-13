# Experiment record

Reports were generated with Python 3.12.14 and the versions in `constraints.txt`. The original CSV is unchanged. Row IDs are zero-based positions in that CSV.

Reproduce from the repository root:

```bash
python -m yield_predictor.cli train --output reports/random
python -m yield_predictor.cli train --split high_temperature --output reports/high_temperature
```

Each run saves a local model; `.joblib` files are intentionally excluded from version control. Small numeric differences can occur across environments.

## random

| Candidate | Parameters | Training CV MAE mean ± fold SD (pp) | Selected |
| --- | --- | ---: | --- |
| DummyRegressor | {} | 11.507 ± 2.319 |  |
| LinearRegression | {} | 3.694 ± 0.338 | Yes |
| Ridge | {"model__alpha": 0.1} | 3.694 ± 0.336 |  |
| Ridge | {"model__alpha": 1.0} | 3.704 ± 0.318 |  |
| Ridge | {"model__alpha": 10.0} | 4.191 ± 0.559 |  |
| RandomForestRegressor | {"model__max_depth": 3, "model__min_samples_leaf": 1} | 5.751 ± 1.645 |  |
| RandomForestRegressor | {"model__max_depth": 3, "model__min_samples_leaf": 3} | 5.835 ± 1.653 |  |
| RandomForestRegressor | {"model__max_depth": null, "model__min_samples_leaf": 1} | 5.032 ± 1.349 |  |
| RandomForestRegressor | {"model__max_depth": null, "model__min_samples_leaf": 3} | 5.346 ± 1.460 |  |

The table contains training CV results; it does not rank models using test labels. Standard deviations summarize fold variation, not confidence intervals.

## high_temperature

| Candidate | Parameters | Training CV MAE mean ± fold SD (pp) | Selected |
| --- | --- | ---: | --- |
| DummyRegressor | {} | 12.389 ± 2.086 |  |
| LinearRegression | {} | 3.528 ± 0.499 |  |
| Ridge | {"model__alpha": 0.1} | 3.527 ± 0.497 |  |
| Ridge | {"model__alpha": 1.0} | 3.524 ± 0.484 | Yes |
| Ridge | {"model__alpha": 10.0} | 3.809 ± 0.475 |  |
| RandomForestRegressor | {"model__max_depth": 3, "model__min_samples_leaf": 1} | 6.094 ± 0.673 |  |
| RandomForestRegressor | {"model__max_depth": 3, "model__min_samples_leaf": 3} | 6.178 ± 0.703 |  |
| RandomForestRegressor | {"model__max_depth": null, "model__min_samples_leaf": 1} | 5.600 ± 0.404 |  |
| RandomForestRegressor | {"model__max_depth": null, "model__min_samples_leaf": 3} | 5.856 ± 0.565 |  |

The table contains training CV results; it does not rank models using test labels. Standard deviations summarize fold variation, not confidence intervals.

## Interpretation limits

The two experiments reuse the same 100-row synthetic dataset with different partitions. Their results are not independent replications. This is a workflow demonstration: the original version already explored this dataset, and no new external blind validation set is claimed.

The high-temperature split tests a narrow form of condition extrapolation. It does not establish generalization to new chemical structures or reaction families. Permutation importance is computed after model selection on each holdout and should not be used to tune the model while reusing that holdout as independent evidence.
