# Knockoff sampling and selection

SCIP knockoff sampler for mixed data, followed by single-realization or derandomized knockoff.


## Setup

The sampler is adapted from the original R implementation by [kormama1/seqknockoff](https://github.com/kormama1/seqknockoff).
Install R and `glmnet`, `doParallel`, `foreach` packages.
`Rscript` must be on `PATH`, or supplied through `SeqKnockoffSampler(rscript="/path/to/Rscript")`.


## Example

`categorical` contains the column names of categorical features.

```python
from dataclasses import replace
from selection_methods.knockoff import (
    SeqKnockoffSampler,
    select_knockoff_single,
    select_knockoff_derandomized
)

sampler = SeqKnockoffSampler(n_jobs=1, random_state=42)
Xk = sampler.sample(X, categorical_features=categorical)
single = select_knockoff_single(X, Xk, y, random_state=42)

Xks = [
    replace(sampler, random_state=seed).sample(
        X, categorical_features=categorical
    )
    for seed in range(10)
]
derandomized = select_knockoff_derandomized(X, Xks, y, random_state=42)
```

Use `outcome_type="linear"` in the selectors for linear outcomes.
