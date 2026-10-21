# Feature selection methods


- `sample_split.py`: `select_single_split()` and `select_multi_split()`. Penalized screening followed by inference on separate observations, with BH/BY corrections.
- `cpss.py`: `select_cpss()`, complementary pairs stability selection.
- `ipss.py`: `select_ipss()`, integrated path stability selection. Our implementation follows [omelikechi/ipss](https://github.com/omelikechi/ipss) for linear outcomes and uses Coxnet for survival outcomes.
- `two_stage.py`: `select_lasso_uni()` and `select_uni_lasso()`, combining penalized selection and univariate tests in either order. Two-stage methods reuse observations for selection and testing and do not provide formal FDR control.
- [knockoff](knockoff/README.md): single-realization (`select_knockoff_single()`) and derandomized knockoff ((`select_knockoff_derandomized()`)).


## Usage

`X` is a numeric DataFrame or an array of shape `(n_samples, n_features)`.

- `outcome_type="linear"`: `y` is a one-dimensional numeric array (linear outcomes).
- `outcome_type="survival"`: `y` is a structured NumPy array whose first two fields are the Boolean event indicator and numeric follow-up time. For example, use `sksurv.util.Surv.from_arrays(event, time)`.

Preprocess and scale features before selection. For the linear CPSS/IPSS paths, center both `X` and `y`.
All selectors return a dictionary mapping method labels to selected feature names.


## Example


```python
from selection_methods.cpss import select_cpss

selected = select_cpss(
    X, y, outcome_type="survival",
    fdr_levels=(0.05, 0.10, 0.15, 0.20),
    l1_ratio=1.0, n_pairs=50,
    n_jobs=1, random_state=3
)
```

`l1_ratio=1.0` uses LASSO; values between 0 and 1 use elastic net. 
