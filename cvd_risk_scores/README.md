# CVD risk scores

`cvd_risk_scores.py` computes 10-year CVD risks. `cvd_score_evaluation.py` applies eligibility filters and evaluates discrimination with bootstrap confidence intervals.


## Example

Prepare `cohort.csv` with the required predictors. The `COLS` dictionary in `cvd_risk_scores.py` maps predictors to dataset columns.
`time` contains follow-up durations in month, and an `event` column (coded 0/1) contains event indicators for equation's endpoints.
Implemented risk scores are `framingham`, `pce_ascvd`, `score2`, `who_lab`, and `who_nonlab`.

```python
import pandas as pd
from cvd_score_evaluation import evaluate_score

df = pd.read_csv("cohort.csv")
result = evaluate_score(
    df,
    time=df["time"].to_numpy(),
    event=df["event"].to_numpy(),
    score="framingham",
    n_bootstrap=2000,
    random_state=42
)
print(result)
```
