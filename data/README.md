# Data card: synthetic reaction conditions

The original project's CSV contains **100 synthetic observations**, not laboratory measurements. It is retained unchanged in this revision.

| Column | Meaning | Generating distribution |
| --- | --- | --- |
| temperature | Temperature in °C | Uniform(20, 100) |
| time | Duration in hours | Uniform(1, 10) |
| concentration | Concentration in mol/L | Uniform(0.1, 1.0) |
| yield | Artificial yield in percent | Formula below |

```text
yield = 0.3 × temperature + 5 × time + 20 × concentration + noise
noise ~ Normal(mean=0, standard deviation=5 percentage points)
```

The features and noise were sampled sequentially using NumPy's legacy random generator with seed 0. `examples/regenerate_data.py` reproduces these values and writes to an ignored artifacts directory. The original generator does not clip yields; all 100 bundled targets happen to fall inside [0, 100]. The training validator enforces that interval for inputs.

## What is missing

There are no reactant/product SMILES, catalysts, solvents, reaction identities, laboratory batches, assay conditions, or experimental provenance. No specific reaction family is represented. Temperature, time, and concentration alone cannot identify an arbitrary chemical transformation.

## Appropriate use

Practice regression, splitting, model selection, reproducibility, and failure analysis. A linear model should do well because the target-generating equation is linear. High performance demonstrates recovery of this artificial relationship, not reaction-mechanism discovery or validated synthesis planning.

The schema rejects missing values and repeated condition triples rather than silently cleaning them. Real data would require an explicit policy for missingness, replicates, group identifiers, measurement uncertainty, and chemical validity. The 30-row minimum is a software guardrail, not a statement of statistical sufficiency.
