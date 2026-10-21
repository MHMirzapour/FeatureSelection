"""
Sample-splitting for feature selection

Two implementations:

1- select_split_single:
    Single-split. LASSO/EN with CV-1SE on the first set. Then,
    computes p-values per feature on the second set. BH/BY corrections
    are applied over p-values at target FDR.

2- select_split_multi:
Repeat single-split M times. Within each split:
    1- LASSO/EN on the first train
    2- Computes raw p-values for candidate features
    3- Bonferroni within split: p_adj = min(1, p_raw × |S_b|)
    4- Aggregate across splits via quantile combination, then
       calibrate for FDR control.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests

from sklearn.linear_model import LassoCV, ElasticNetCV, Lasso, ElasticNet
from sklearn.model_selection import KFold

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored


# Penalized regression with CV
def _cox_cv(
    X, y,
    l1_ratio, n_alphas, alpha_min_ratio,
    n_cv, random_state, use_one_se
):
    event_field, time_field = y.dtype.names[:2]

    pilot = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio, n_alphas=n_alphas,
        alpha_min_ratio=alpha_min_ratio, fit_baseline_model=False
    )
    pilot.fit(X, y)
    alphas = pilot.alphas_

    kf = KFold(n_splits=n_cv, shuffle=True, random_state=random_state)
    cv_scores = np.full((len(alphas), n_cv), np.nan)
    for fold_idx, (tr, va) in enumerate(kf.split(X, y)):
        model = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio, alphas=alphas, fit_baseline_model=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[tr], y[tr])
        for a_idx, alpha in enumerate(model.alphas_):
            try:
                risk = model.predict(X[va], alpha=alpha)
                c = concordance_index_censored(
                    y[va][event_field], y[va][time_field], risk
                )[0]
                cv_scores[a_idx, fold_idx] = c
            except Exception:
                cv_scores[a_idx, fold_idx] = np.nan

    mean_scores = np.nanmean(cv_scores, axis=1)
    se_scores = np.nanstd(cv_scores, axis=1, ddof=1) / np.sqrt(
        np.maximum(np.sum(~np.isnan(cv_scores), axis=1), 1)
    )

    if np.all(np.isnan(mean_scores)):
        chosen_idx = 0
    else:
        best_idx = int(np.nanargmax(mean_scores))
        if use_one_se:
            threshold = mean_scores[best_idx] - se_scores[best_idx]
            cands = np.where(mean_scores >= threshold)[0]
            chosen_idx = int(cands[0]) if len(cands) > 0 else best_idx
        else:
            chosen_idx = best_idx

    chosen_alpha = alphas[chosen_idx]
    final = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio, alphas=[chosen_alpha], fit_baseline_model=False
    )
    final.fit(X, y)
    return final.coef_


def _linear_cv(
    X, y,
    l1_ratio, n_alphas, alpha_min_ratio,
    n_cv, random_state, use_one_se
):
    if l1_ratio == 1.0:
        cv_model = LassoCV(
            alphas=n_alphas, eps=alpha_min_ratio,
            cv=KFold(n_splits=n_cv, shuffle=True, random_state=random_state),
            random_state=random_state, n_jobs=1
        )
    else:
        cv_model = ElasticNetCV(
            l1_ratio=l1_ratio, alphas=n_alphas, eps=alpha_min_ratio,
            cv=KFold(n_splits=n_cv, shuffle=True, random_state=random_state),
            random_state=random_state, n_jobs=1
        )

    cv_model.fit(X, y)

    alphas = cv_model.alphas_
    mse = cv_model.mse_path_
    mean_mse = mse.mean(axis=1)
    se_mse = mse.std(axis=1, ddof=1) / np.sqrt(n_cv)

    if not use_one_se:
        chosen_alpha = cv_model.alpha_
    else:
        best_idx = int(np.argmin(mean_mse))
        threshold = mean_mse[best_idx] + se_mse[best_idx]
        cands = np.where(mean_mse <= threshold)[0]
        chosen_idx = int(cands[0]) if len(cands) > 0 else best_idx
        chosen_alpha = alphas[chosen_idx]

    if l1_ratio == 1.0:
        final = Lasso(alpha=chosen_alpha)
    else:
        final = ElasticNet(alpha=chosen_alpha, l1_ratio=l1_ratio)
    final.fit(X, y)
    return final.coef_


# Compute p-values
def _cox_pvalues(X, y, ties="efron"):
    """
    Joint Cox refit on X (restricted to candidate set). Returns Wald
    p-values for each feature.
    """
    event_field, time_field = y.dtype.names[:2]
    t = y[time_field]
    e = y[event_field].astype(int)
    p = X.shape[1]
    try:
        m = PHReg(t, X, status=e, ties=ties)
        r = m.fit(disp=False)
        pvals = r.pvalues
        if pvals.shape[0] != p:
            return np.ones(p)
        return np.where(np.isfinite(pvals), pvals, 1.0)
    except Exception:
        return np.ones(p)


def _gaussian_pvalues(X, y):
    """
    Joint OLS refit on X (restricted to candidate set). Returns Wald
    p-values for each feature.
    """
    n, p = X.shape
    if p >= n:
        return np.ones(p)
    try:
        Xc = sm.add_constant(X, has_constant="add")
        r = sm.OLS(y, Xc).fit()
        pvals = r.pvalues
        # Drop intercept p-value
        if pvals.shape[0] == p + 1:
            pvals = pvals[1:]
        elif pvals.shape[0] != p:
            return np.ones(p)
        return np.where(np.isfinite(pvals), pvals, 1.0)
    except Exception:
        return np.ones(p)


# Multiplicity correction
def _apply_correction(pvals, alpha, method):
    sm_method = {"bh": "fdr_bh", "by": "fdr_by"}[method]
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method=sm_method)
    return reject


def _apply_correction_multi_split(pvals, alpha, method):
    """
    p_values for multi-split are already adjusted for multipicity, we just need to
    apply FDR correction.
    """
    order = np.argsort(pvals)
    ranks = np.arange(1, pvals.size + 1)
    factor = {"bh": 1.0, "by": np.sum(1.0 / ranks)}[method]
    # Exclude values clipped to 1, as in hdi::fdr.adjust.
    passed = (pvals[order] < 1.0) & (pvals[order] <= alpha * ranks / factor)
    reject = np.zeros(pvals.size, dtype=bool)
    if passed.any():
        reject[order[:np.flatnonzero(passed)[-1] + 1]] = True
    return reject


def _cv_select(X, y, outcome_type, l1_ratio, n_alphas, alpha_min_ratio,
                  n_cv, random_state, use_one_se):
    if outcome_type == "survival":
        return _cox_cv(
            X, y, l1_ratio=l1_ratio, n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio, n_cv=n_cv,
            random_state=random_state, use_one_se=use_one_se
        )
    if outcome_type == "linear":
        return _linear_cv(
            X, y, l1_ratio=l1_ratio, n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio, n_cv=n_cv,
            random_state=random_state, use_one_se=use_one_se
        )
    raise ValueError("Unknown outcome_type.")


def _compute_p(X, y, outcome_type):
    if outcome_type == "survival":
        return _cox_pvalues(X, y)
    if outcome_type == "linear":
        return _gaussian_pvalues(X, y)
    raise ValueError("Unknown outcome_type.")


# Sample splitting
def _half_split(y, rng):
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    mid = n // 2
    train_idx, test_idx = idx[:mid], idx[mid:]
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _multi_split_aggregate(pvals_by_split, gamma_min, n_gamma):
    """
    Meinshausen, Meier, and Bühlmann (2009) quantile combination across splits.
    """
    M, p = pvals_by_split.shape
    if M == 1:
        return pvals_by_split[0]
    gammas = np.linspace(gamma_min, 1.0, n_gamma)
    Q = np.empty((len(gammas), p))
    for k, g in enumerate(gammas):
        scaled = pvals_by_split / g
        Q[k] = np.minimum(1.0, np.quantile(scaled, g, axis=0, method="higher"))
    inner = Q.min(axis=0)
    return np.minimum(1.0, (1.0 - np.log(gamma_min)) * inner)


# Single-split 
def select_single_split(
    X,
    y,
    outcome_type = "survival",  # or "linear"
    fdr_levels = (0.05, 0.10, 0.15, 0.20),
    correction = "both",  # "bh", "by", or "both"
    
    l1_ratio = 1.0,
    n_alphas = 100,
    alpha_min_ratio = 0.01,
    n_cv = 5,
    use_one_se = True,
    
    random_state = 42
):
    if isinstance(X, pd.DataFrame):
        feat_names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        feat_names = [f"X{i}" for i in range(X_arr.shape[1])]

    if correction == "both":
        corr_methods = ["bh", "by"]
    elif correction in ("bh", "by"):
        corr_methods = [correction]
    else:
        raise ValueError("Correction must be 'bh', 'by', or 'both'.")

    reg = "lasso" if l1_ratio == 1.0 else f"en{l1_ratio:.2f}"
    
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = _half_split(y, rng)

    coefs = _cv_select(
        X_arr[train_idx], y[train_idx], outcome_type, l1_ratio,
        n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio,
        n_cv=n_cv, random_state=random_state, use_one_se=use_one_se
    )
    cand = np.where(coefs != 0)[0]
    
    out = {}
    if len(cand) == 0:
        for corr in corr_methods:
            for q in fdr_levels:
                out[f"split_single_{outcome_type}_{reg}_{corr}_q{q:.2f}"] = []
        return out

    pvals_cand = _compute_p(
        X_arr[np.ix_(test_idx, cand)], y[test_idx], outcome_type
    )
    for corr in corr_methods:
        for q in fdr_levels:
            reject = _apply_correction(pvals_cand, alpha=q, method=corr)
            sel = cand[reject]
            tag = f"split_single_{outcome_type}_{reg}_{corr}_q{q:.2f}"
            out[tag] = [feat_names[j] for j in sel]
    return out


# Multi-split
def select_multi_split(
    X,
    y,
    outcome_type = "survival",  # or "linear"
    fdr_levels = (0.05, 0.10, 0.15, 0.20),
    correction = "both", # "bh", "by", or "both"
    
    l1_ratio = 1.0,
    n_alphas = 100,
    alpha_min_ratio = 0.01,
    n_cv = 5,
    use_one_se = True,
    
    # MMB aggregation parameters:
    n_data_splits = 50,
    gamma_min = 0.05,
    n_gamma = 50,

    random_state = 42
):
    if isinstance(X, pd.DataFrame):
        feat_names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        feat_names = [f"X{i}" for i in range(X_arr.shape[1])]

    if correction == "both":
        corr_methods = ["bh", "by"]
    elif correction in ("bh", "by"):
        corr_methods = [correction]
    else:
        raise ValueError("Correction must be 'bh', 'by', or 'both'.")

    reg = "lasso" if l1_ratio == 1.0 else f"en{l1_ratio:.2f}"
    
    seeds = np.random.SeedSequence(random_state).spawn(n_data_splits)
    seed_ints = [int(s.generate_state(1)[0]) for s in seeds]
    
    p = X.shape[1]
    pvals_by_split = np.ones((n_data_splits, p))

    for b in range(n_data_splits):
        rng = np.random.default_rng(seed_ints[b])
        train_idx, test_idx = _half_split(y, rng)
        coefs = _cv_select(
            X_arr[train_idx], y[train_idx], outcome_type, l1_ratio,
            n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio,
            n_cv=n_cv, random_state=seed_ints[b],
            use_one_se=use_one_se
        )
        cand = np.where(coefs != 0)[0]
        if len(cand) == 0:
            continue
        pvals_cand = _compute_p(
            X_arr[np.ix_(test_idx, cand)], y[test_idx], outcome_type
        )

        # Within-split Bonferroni correction
        pvals_cand_bonf = np.minimum(1.0, pvals_cand * len(cand))
        pvals_by_split[b, cand] = pvals_cand_bonf

    p_combined = _multi_split_aggregate(
        pvals_by_split, gamma_min=gamma_min, n_gamma=n_gamma
    )

    out = {}
    for corr in corr_methods:
        for q in fdr_levels:
            reject = _apply_correction_multi_split(p_combined, alpha=q, method=corr)
            sel = np.where(reject)[0]
            tag = f"split_M_{outcome_type}_{reg}_{corr}_q{q:.2f}"
            out[tag] = [feat_names[j] for j in sel]
    return out
