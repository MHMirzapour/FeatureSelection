"""
Two-stage feature selection.

Two implementations:

1- select_lasso_uni:
    Stage 1: Selects candidate support S using LASSO/EN with CV-1SE
    Stage 2: Univariate test on S, BH/BY at target FDR


2- select_uni_lasso:
    Stage 1: Selects candidate support S using univariate test on all
    features with BH/BY at target FDR
    Stage 2: Final selection on s using LASSO/EN.

Note that these methods provide no formal FDR control, since
selection and inference steps are performed on the same data.
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


# Compute univariate p-values
def _univariate_cox_pvalues(X, y, test="lrt", ties="efron"):
    event_field, time_field = y.dtype.names[:2]
    t = y[time_field]
    e = y[event_field].astype(int)
    p = X.shape[1]
    pvals = np.ones(p)
    for j in range(p):
        xj = X[:, j:j + 1]
        try:
            m = PHReg(t, xj, status=e, ties=ties)
            r = m.fit(disp=False)
            if test == "wald":
                pvals[j] = r.pvalues[0]
            else:
                ll1 = r.llf
                ll0 = m.loglike(np.array([0.0]))
                stat = max(2.0 * (ll1 - ll0), 0.0)
                pvals[j] = stats.chi2.sf(stat, df=1)
        except Exception:
            pvals[j] = 1.0
    return pvals


def _univariate_linear_pvalues(X, y):
    n, p = X.shape
    pvals = np.ones(p)
    y_centered = y - y.mean()
    sy = np.sqrt((y_centered ** 2).sum())
    for j in range(p):
        xj = X[:, j]
        xc = xj - xj.mean()
        sx = np.sqrt((xc ** 2).sum())
        if sx == 0 or sy == 0:
            continue
        r = (xc * y_centered).sum() / (sx * sy)
        r = np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12)
        t_stat = r * np.sqrt((n - 2) / (1 - r ** 2)) if n > 2 else 0.0
        pvals[j] = 2.0 * stats.t.sf(abs(t_stat), df=n - 2)
    return pvals


# Multiplicity correction
def _apply_correction(pvals, alpha, method):
    sm_method = {"bh": "fdr_bh", "by": "fdr_by"}[method]
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method=sm_method)
    return reject


# Dispatch helpers
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


def _compute_univariate_p(X, y, outcome_type, cox_test="lrt"):
    if outcome_type == "survival":
        return _univariate_cox_pvalues(X, y, test=cox_test)
    if outcome_type == "linear":
        return _univariate_linear_pvalues(X, y)
    raise ValueError("Unknown outcome_type.")


# Variant 1: lasso-then-univariate
def select_lasso_uni(
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
    
    cox_test = "lrt",
    
    random_state = 42
):
    if isinstance(X, pd.DataFrame):
        feat_names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        feat_names = [f"X{i}" for i in range(X_arr.shape[1])]
        
    reg = "lasso" if l1_ratio == 1.0 else f"en{l1_ratio:.2f}"

    if correction == "both":
        corr_methods = ["bh", "by"]
    elif correction in ("bh", "by"):
        corr_methods = [correction]
    else:
        raise ValueError("Correction must be 'bh', 'by', or 'both'.")

    coefs = _cv_select(
        X_arr, y, outcome_type, l1_ratio,
        n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio,
        n_cv=n_cv, random_state=random_state, use_one_se=use_one_se
    )
    cand = np.where(coefs != 0)[0]
    out = {}
    if len(cand) == 0:
        for corr in corr_methods:
            for q in fdr_levels:
                out[f"two_stage_lasso_uni_{outcome_type}_{reg}_{corr}_q{q:.2f}"] = []
        return out

    pvals = _compute_univariate_p(X_arr[:, cand], y, outcome_type, cox_test=cox_test)
    for corr in corr_methods:
        for q in fdr_levels:
            reject = _apply_correction(pvals, alpha=q, method=corr)
            sel = cand[reject]
            tag = f"two_stage_lasso_uni_{outcome_type}_{reg}_{corr}_q{q:.2f}"
            out[tag] = [feat_names[j] for j in sel]
    return out


# Variant 2: univariate-then-lasso
def select_uni_lasso(
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
    
    cox_test = "lrt",
    
    random_state = 42
):
    if isinstance(X, pd.DataFrame):
        feat_names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        feat_names = [f"X{i}" for i in range(X_arr.shape[1])]

    reg = "lasso" if l1_ratio == 1.0 else f"en{l1_ratio:.2f}"

    if correction == "both":
        corr_methods = ["bh", "by"]
    elif correction in ("bh", "by"):
        corr_methods = [correction]
    else:
        raise ValueError("Correction must be 'bh', 'by', or 'both'.")

    pvals_all = _compute_univariate_p(X_arr, y, outcome_type, cox_test=cox_test)
    out = {}
    for corr in corr_methods:
        for q in fdr_levels:
            reject_s1 = _apply_correction(pvals_all, alpha=q, method=corr)
            cand = np.where(reject_s1)[0]
            tag = f"two_stage_uni_lasso_{outcome_type}_{reg}_{corr}_q{q:.2f}"
            if len(cand) == 0:
                out[tag] = []
                continue
            if len(cand) < 2:
                out[tag] = [feat_names[j] for j in cand]
                continue
            try:
                coefs = _cv_select(
                    X_arr[:, cand], y, outcome_type, l1_ratio,
                    n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio,
                    n_cv=n_cv, random_state=random_state,
                    use_one_se=use_one_se
                )
                sel = cand[np.where(coefs != 0)[0]]
            except Exception:
                sel = cand
            out[tag] = [feat_names[j] for j in sel]
    return out