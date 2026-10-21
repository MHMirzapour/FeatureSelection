"""
Evaluation of the established CVD risk scores on the ICS cohort.

Discrimination metrics:

1- Harrell's C-index
2- Uno's IPCW C-index (robust to censoring)
3- Time-dependent cumulative AUC
4- Mean AUC over specified time points
"""

import numpy as np
import pandas as pd

from cvd_risk_scores import (
    COLS, framingham_risk, pce_ascvd_risk, score2_risk, who_cvd_risk_2019
)

from sksurv.util import Surv
from sksurv.metrics import (
    cumulative_dynamic_auc, concordance_index_censored, concordance_index_ipcw
)


# Applying eligibility criteria
SCORES = {
    "framingham": {"fn": lambda df: framingham_risk(df),
                   "age": (30, 74),
                   "exclude_diabetes": False,
                   "needs": ["sex", "age", "tot_chol", "hdl", "sbp", "bp_treated",
                             "smoker", "diabetes"]},
    "pce_ascvd":  {"fn": lambda df: pce_ascvd_risk(df),
                   "age": (40, 79),
                   "exclude_diabetes": False,
                   "needs": ["sex", "age", "tot_chol", "hdl", "sbp", "bp_treated",
                             "smoker", "diabetes"]},
    "score2":     {"fn": lambda df: score2_risk(df, region="high", calibration="asia_pacific"),
                   "age": (40, 69),
                   "exclude_diabetes": True,
                   "needs": ["sex", "age", "tot_chol", "hdl", "sbp", "smoker"]},
    "who_lab":    {"fn": lambda df: who_cvd_risk_2019(df, model="lab", calibration="north_africa_middle_east"),
                   "age": (40, 80),
                   "exclude_diabetes": False,
                   "needs": ["sex", "age", "tot_chol", "sbp", "smoker", "diabetes"]},
    "who_nonlab": {"fn": lambda df: who_cvd_risk_2019(df, model="nonlab", calibration="north_africa_middle_east"),
                   "age": (40, 80),
                   "exclude_diabetes": False,
                   "needs": ["sex", "age", "bmi", "sbp", "smoker"]}
}


def eligibility_mask(df, score):
    spec = SCORES[score]
    lo, hi = spec["age"]
    age = pd.to_numeric(df[COLS["age"]]).to_numpy()
    mask = (age >= lo) & (age <= hi)
    if spec["exclude_diabetes"]:
        mask &= pd.to_numeric(df[COLS["diabetes"]]).to_numpy() == 0
    for key in spec["needs"]:
        mask &= ~pd.to_numeric(df[COLS[key]]).isna().to_numpy()
    return mask


def _metrics(t, e, risk, harrell_time, uno_tau, auc_times):
    # Harrell's C-index is not time-dependent, but since risk
    # scores are designed to estimate 10-year risks, computing
    # Harrell's C using unrestricted outcomes can affect estimation.
    harrell_c = concordance_index_censored(
        e & (t <= harrell_time), np.minimum(t, harrell_time), risk
    )[0]
    y = Surv.from_arrays(e, t)
    uno_c = concordance_index_ipcw(y, y, risk, tau=uno_tau)[0]
    aucs, mean_auc = cumulative_dynamic_auc(y, y, risk, times=auc_times)
    values = {
        "harrell_c": harrell_c,
        "uno_c": uno_c,
        **{f"auc_t{tt:g}": a for tt, a in zip(auc_times, aucs)},
        "mean_auc": mean_auc
    }
    return values


def evaluate_score(
    df, time, event, score,
    harrell_time=120,
    uno_tau=120,
    auc_times=[120.0],
    n_bootstrap=0,
    confidence_level=0.95,
    random_state=42
):
    time = np.asarray(time)
    event = np.asarray(event)

    mask = eligibility_mask(df, score)
    t, e = time[mask], np.bool(event[mask])
    risk = np.asarray(SCORES[score]["fn"](df.loc[mask]))

    values = _metrics(t, e, risk, harrell_time, uno_tau, auc_times)
    result = {
        "score": score,
        "n": int(len(t)),
        "n_events": int((e & (t <= harrell_time)).sum()),
        "n_events_max_followup": int(e.sum()),
        **values
    }

    if n_bootstrap:
        rng = np.random.default_rng(random_state)
        boot = []
        for b in range(n_bootstrap):
            index = rng.integers(0, len(t), size=len(t))
            boot.append(_metrics(t[index], e[index], risk[index],
                                 harrell_time, uno_tau, auc_times))

        alpha = (1 - confidence_level) / 2
        bounds = pd.DataFrame(boot).quantile([alpha, 1 - alpha])
        for metric in values:
            result[f"{metric}_ci_lower"], result[f"{metric}_ci_upper"] = bounds[metric].to_numpy()
        result["n_bootstrap"] = n_bootstrap
        result["confidence_level"] = confidence_level
    return result


def evaluate_all_scores(
    df, time, event, scores=tuple(SCORES),
    harrell_time=120, uno_tau=120, auc_times=[120.0],
    n_bootstrap=2000, confidence_level=0.95, random_state=42
):
    # Outcome definitions differ among risk equations, an appropriate survival outcome should be supplied for each risk score.
    rows = [evaluate_score(
        df, time, event, score, harrell_time, uno_tau, auc_times,
        n_bootstrap=n_bootstrap, confidence_level=confidence_level,
        random_state=random_state
    ) for score in scores]
    return pd.DataFrame(rows).set_index("score")