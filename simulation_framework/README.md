# Simulation framework

Generates correlated normal, lognormal, binary, and ordinal features with linear or right-censored survival outcomes.


## Example

`generate_data()` returns covariates, outcome array, true feature names, and diagnostics. Features are standardized within each dataset.
The runner derives a separate data seed for each replicate; it does not change a selector's `random_state`.

```python
from simulation_framework.DGP import generate_data
from simulation_framework.SimulationRunner import create_config, SimulationRunner
from selection_methods.cpss import select_cpss

config = create_config(
    name="correlation_03_05",
    n_samples=1000,
    n_features=200,
    within_cluster_correlation=(0.3, 0.5),
    outcome_type="survival",
    random_state=42
)

X, y, true_features, diagnostics = generate_data(config)

def cpss_method(X, y):
    return select_cpss(
        X, y, outcome_type="survival",
        fdr_levels=(0.05, 0.10, 0.15, 0.20),
        n_jobs=1, random_state=42
    )

runner = SimulationRunner(
    methods=[cpss_method], n_repeats=5,
    save_path="results/raw_results.csv"
)
results = runner.run([config])
results.summary().to_csv("results/summary.csv", index=False)
```
