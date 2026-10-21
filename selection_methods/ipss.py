"""
Integrated Path Stability Selection (IPSS)

This code follows the omelikechi/ipss implementation
(https://github.com/omelikechi/ipss) for linear outcomes.
Survival outcomes are handled using scikit-survival
CoxnetSurvivalAnalysis as the base learner.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso, lasso_path, enet_path
from sksurv.linear_model import CoxnetSurvivalAnalysis


def _compute_correlation(X):
    corr = np.corrcoef(X, rowvar=False)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0)
    avg_corr = np.mean(np.mean(abs_corr, axis=1))
    avg_max_corr = np.mean(np.max(abs_corr, axis=1))
    return avg_corr, avg_max_corr


def _compute_delta(X):
    avg_corr, avg_max = _compute_correlation(X)
    if avg_corr <= 1.0 / 20.0:
        return 1.0
    if avg_corr >= 1.0 / 4.0:
        return 0.0
    M = np.array([
        [1.0, 1.0 / 20.0, 1.0 / 2.0],
        [1.0, 1.0 / 4.0, 1.0 / 2.0],
        [1.0, 1.0 / 20.0, 1.0]
    ])
    rhs = np.array([1.0, 0.0, 0.0])
    coefs = np.linalg.inv(M) @ rhs
    m = np.array([1.0, avg_corr, avg_max])
    delta = coefs @ m
    return max(0.0, min(1.0, delta))


# Penalty grid for linear and survival outcomes
def _compute_alphas_linear(
    X, y, n_alphas, max_features, l1_ratio
):
    n = X.shape[0]
    alpha_max = 2 * np.max(np.abs(X.T @ y)) / n
    
    if not np.isfinite(alpha_max) or alpha_max == 0.0:
        alpha_max = 100.0
    
    if l1_ratio < 1.0:
        alpha_max = alpha_max / l1_ratio

    alpha_min = alpha_max * 1e-10
    test_alphas = np.logspace(
        np.log10(alpha_max / 2), np.log10(alpha_min), 100
    )

    try:
        if l1_ratio == 1.0:
            alphas, coefs, _ = lasso_path(X, y, alphas=test_alphas)
        else:
            alphas, coefs, _ = enet_path(X, y, l1_ratio=l1_ratio, alphas=test_alphas)
    except Exception:
        return np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)

    for a_idx, a in enumerate(alphas):
        if int(np.sum(coefs.T[a_idx] != 0)) >= max_features:
            alpha_min = a
            break
    
    return np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)


def _compute_alphas_cox(
    X, y, n_alphas, max_features, l1_ratio
):
    pilot = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        n_alphas=100,
        fit_baseline_model=False
    )
    
    try:
        pilot.fit(X, y)
        alpha_max = pilot.alphas_[0]
    except Exception:
        alpha_max = 100

    if not np.isfinite(alpha_max) or alpha_max == 0.0:
        alpha_max = 100
    
    alpha_min = alpha_max * 1e-10
    test_alphas = np.logspace(
        np.log10(alpha_max), np.log10(alpha_min), 100
    )

    selector = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        alphas=test_alphas,
        fit_baseline_model=False
    )
    
    try:
        selector.fit(X, y)
        for a_idx, a in enumerate(selector.alphas_):
            if int(np.sum(selector.coef_.T[a_idx] != 0)) >= max_features:
                alpha_min = a
                break
    except Exception:
        return np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)
        
    return np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)


# Integration
def _integrate(values, alphas, delta=1.0, cutoff=None):
    n_alphas = len(alphas)
    a = np.min(alphas)
    b = np.max(alphas)
    if delta == 1.0:
        normalization = (1.0 - (a / b) ** (1.0 / n_alphas)) / np.log(b / a)
    else:
        normalization = (
            (1.0 - delta) * (1.0 - (a / b) ** (1.0 / n_alphas))
            / (b ** (1.0 - delta) - a ** (1.0 - delta))
        )
    output = 0.0
    stop_index = n_alphas
    if cutoff is None:
        for i in range(1, n_alphas):
            weight = 1.0 if delta == 1.0 else alphas[i] ** (1.0 - delta)
            output += normalization * weight * values[i - 1]
    else:
        for i in range(1, n_alphas):
            weight = 1.0 if delta == 1.0 else alphas[i] ** (1.0 - delta)
            cand = output + normalization * weight * values[i - 1]
            if cand > cutoff:
                stop_index = i
                break
            output = cand
    return output, stop_index


# Subsample fits
def _cox_indicators(X, y, alphas, l1_ratio):
    p, L = X.shape[1], len(alphas)
    model = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio, alphas=alphas, fit_baseline_model=False
    )
    model.fit(X, y)
    coefs = model.coef_
    L_act = coefs.shape[1]
    ind = np.zeros((L, p))
    ind[:L_act] = (coefs != 0).T.astype(float)
    return ind


def _linear_indicators(X, y, alphas, l1_ratio):
    if l1_ratio == 1.0:
        _, coefs, _ = lasso_path(X, y, alphas=alphas)
    else:
        _, coefs, _ = enet_path(X, y, alphas=alphas, l1_ratio=l1_ratio)
    return (coefs != 0).T.astype(float)


def _one_pair(seed, X, y, alphas, l1_ratio, indicator_fn):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    sh = rng.permutation(n)
    n_split = n // 2
    A, B = sh[:n_split], sh[n_split:]
    return (
        indicator_fn(X[A], y[A], alphas=alphas, l1_ratio=l1_ratio),
        indicator_fn(X[B], y[B], alphas=alphas, l1_ratio=l1_ratio)
    )


# q-value computation
def _compute_qvalues(efp_scores):
    """
    For each unique threshold t in efp_scores:
      S_t = features with efp <= t
      FP_t = max(efp in S_t) = t (when t is achieved)
      fdr(t) = FP_t / |S_t|
    For each feature j with efp = e_j:
      q_j = min{fdr(t): t in thresholds, t >= e_j}
    Non-finite efp values are replaced by p+1 (effectively never selected).
    """
    p = len(efp_scores)
    
    # Replace non-finite efp values
    efp_scores = np.where(np.isfinite(efp_scores), efp_scores, p + 1.0)
    fdrs = []
    for t in efp_scores:
        if not np.isfinite(t):
            continue
        mask = efp_scores <= t
        if not mask.any():
            continue
        FP_t = np.max(efp_scores[mask])
        S_t = int(mask.sum())
        fdrs.append((t, FP_t / S_t if S_t > 0 else 1.0))
    q = np.empty(p)
    for j, score in enumerate(efp_scores):
        candidates = [fdr for t, fdr in fdrs if score <= t]
        q[j] = min(candidates) if candidates else 1.0
    return np.minimum(q, 1.0)


# Stability path truncation
def _stop_index(stability_paths):
    L, p = stability_paths.shape
    for i in range(2, L):
        if np.allclose(stability_paths[i, :], 0.0):
            continue
        diff = stability_paths[i, :] - stability_paths[i - 2, :]
        mean_i = stability_paths[i, :].mean()
        if np.allclose(diff, 0.0) and mean_i > 0.01:
            return i
    return L


# Theorem 4.2 of the paper
def _h2_integrand(q_path, p, B):
    """E(FP) bound for h2 (Eq. 4.4)."""
    return q_path ** 2 / (B * p) + (B - 1) * q_path ** 4 / (B * p ** 3)


def _h3_integrand(q_path, p, B):
    """E(FP) bound for h3 (Eq. 4.5)."""
    return (
        q_path ** 2 / (B ** 2 * p)
        + 3.0 * (B - 1) * q_path ** 4 / (B ** 2 * p ** 3)
        + (B - 1) * (B - 2) * q_path ** 6 / (B ** 2 * p ** 5)
    )


def _hm(x, m):
    return np.where(x >= 0.5, (2.0 * x - 1.0) ** m, 0.0)



# Main function
def select_ipss(
    X,
    y,
    outcome_type = "survival", # "survival" or "linear"
    fdr_levels = (0.05, 0.10, 0.15, 0.20),
    l1_ratio = 1.0,
    n_pairs = 50,
    n_alphas = 25,                    
    f_power = 2,                       
    cutoff = 0.05,
    delta = None,          
    max_features_frac = 0.75, # from IPSS original implementation
    n_jobs = -1,
    random_state = None
):
    
    if isinstance(X, pd.DataFrame):
        feat_names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        feat_names = [f"X{i}" for i in range(X_arr.shape[1])]
    n, p = X_arr.shape

    reg = "lasso" if l1_ratio == 1.0 else f"en{l1_ratio}"
    tag = f"ipss_h{f_power}_{outcome_type}_{reg}"

    if delta is None:
        delta = _compute_delta(X_arr)

    max_features = max(int(np.floor(max_features_frac * p)), 1)

    if outcome_type == "survival":
        indicator_fn = _cox_indicators
        alphas = _compute_alphas_cox(
            X_arr, y,
            n_alphas=n_alphas, max_features=max_features,
            l1_ratio=l1_ratio
        )

    elif outcome_type == "linear":
        indicator_fn = _linear_indicators
        alphas = _compute_alphas_linear(
            X_arr, y,
            n_alphas=n_alphas, max_features=max_features,
            l1_ratio=l1_ratio
        )
        
    else:
        raise ValueError("Unknown outcome_type.")

    if len(alphas) < 2:
        return {f"{tag}_q{q:.2f}": [] for q in fdr_levels}

    seeds = np.random.SeedSequence(random_state).spawn(n_pairs)
    seed_ints = [int(s.generate_state(1)[0]) for s in seeds]

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_pair)(
            seed_ints[b], X_arr, y, alphas, l1_ratio, indicator_fn
        )
        for b in range(n_pairs)
    )

    L = len(alphas)
    pi_sum = np.zeros((L, p))
    q_sum = np.zeros(L)
    for indA, indB in results:
        pi_sum += indA + indB
        q_sum += indA.sum(axis=1) + indB.sum(axis=1)
    pi_path = pi_sum / (2 * n_pairs)
    q_path = q_sum / (2 * n_pairs)

    # Path truncation
    stop = _stop_index(pi_path)
    if stop < L:
        pi_path = pi_path[:stop]
        q_path = q_path[:stop]
        alphas = alphas[:stop]
        L = stop

    if not np.any(q_path > 0):
        return {f"{tag}_q{q:.2f}": [] for q in fdr_levels}

    if f_power == 2:
        integrand = _h2_integrand(q_path, p, B=n_pairs)
    elif f_power == 3:
        integrand = _h3_integrand(q_path, p, B=n_pairs)
    else:
        raise ValueError("Only m = 2 or 3 is supported.")

    integral, stop_idx_cut = _integrate(integrand, alphas, delta=delta, cutoff=cutoff)

    if stop_idx_cut < 2 or integral <= 0.0 or not np.isfinite(integral):
        return {f"{tag}_q{q:.2f}": [] for q in fdr_levels}

    alphas_stop = alphas[:stop_idx_cut]
    pi_stop = pi_path[:stop_idx_cut]
    h_vals = _hm(pi_stop, f_power) # (stop_idx_cut, p)
    scores = np.zeros(p)
    for j in range(p):
        s_j, _ = _integrate(h_vals[:, j], alphas_stop, delta=delta, cutoff=None)
        scores[j] = s_j

    # efp = integral / max(scores, integral / p)
    floor = integral / p
    denom = np.maximum(scores, floor)
    denom = np.where((denom > 0) & np.isfinite(denom), denom, max(floor, 1e-300))
    efp = integral / denom
    efp = np.where(np.isfinite(efp), efp, p + 1.0)
    efp = np.minimum(efp, p + 1)

    qvals = _compute_qvalues(efp)

    out = {}
    for q_target in fdr_levels:
        sel_idx = np.where(qvals <= q_target)[0]
        out[f"{tag}_q{q_target:.2f}"] = [feat_names[j] for j in sel_idx]
    return out
    