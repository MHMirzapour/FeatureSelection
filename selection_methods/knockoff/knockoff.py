"""
Model-X knockoff using Kormaksson et al. sequential knockoff sampler for mixed data

Two implementations:

1- Single-realization knockoff
2- Derandomized Knockoff
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import KFold


def _prepare_X(X):
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.to_numpy()
    else:
        X_arr = np.asarray(X)
        feature_names = [f"X{i}" for i in range(X_arr.shape[1])]
    return X_arr, feature_names


def _spawn_two_seeds(random_state):
    ss = np.random.SeedSequence(random_state)
    a, b = ss.spawn(2)
    sa = int(a.generate_state(1)[0])
    sb = int(b.generate_state(1)[0])
    return sa, sb


# Test statistics
def _lcd_cox(
    X_aug, y,
    l1_ratio, n_alphas, alpha_min_ratio,
    n_cv, random_state
):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    p_aug = X_aug.shape[1]
    p = p_aug // 2
    event_field, time_field = y.dtype.names[:2]

    pilot = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        n_alphas=n_alphas,
        alpha_min_ratio=alpha_min_ratio,
        fit_baseline_model=False
    )
    pilot.fit(X_aug, y)
    alphas = pilot.alphas_

    kf = KFold(n_splits=n_cv, shuffle=True, random_state=random_state)
    cv_c = np.full((len(alphas), n_cv), np.nan)
    for k, (tr, va) in enumerate(kf.split(X_aug, y)):
        m = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio, alphas=alphas, fit_baseline_model=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_aug[tr], y[tr])
        for ai, a in enumerate(m.alphas_):
            try:
                risk = m.predict(X_aug[va], alpha=a)
                cv_c[ai, k] = concordance_index_censored(
                    y[va][event_field], y[va][time_field], risk
                )[0]
            except Exception:
                cv_c[ai, k] = np.nan

    mean_c = np.nanmean(cv_c, axis=1)
    if np.all(np.isnan(mean_c)):
        return np.zeros(p)
    chosen_alpha = alphas[int(np.nanargmax(mean_c))]

    final = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio, alphas=[chosen_alpha], fit_baseline_model=False
    )
    final.fit(X_aug, y)
    coefs = np.abs(final.coef_.ravel())
    return coefs[:p] - coefs[p:]


def _lcd_linear(
    X_aug, y,
    l1_ratio, n_alphas, alpha_min_ratio,
    n_cv, random_state
):
    p_aug = X_aug.shape[1]
    p = p_aug // 2

    if l1_ratio == 1.0:
        model = LassoCV(
            cv=n_cv,alphas=n_alphas, eps=alpha_min_ratio,
            random_state=random_state, n_jobs=1
        )
    else:
        model = ElasticNetCV(
            cv=n_cv, alphas=n_alphas, eps=alpha_min_ratio,
            l1_ratio=l1_ratio, random_state=random_state, n_jobs=1
        )
    model.fit(X_aug, y)
    coefs = np.abs(model.coef_)
    return coefs[:p] - coefs[p:]


def _compute_W(
    X_aug, y,
    outcome_type, l1_ratio, n_alphas,
    alpha_min_ratio, n_cv, random_state
):
    if outcome_type == "survival":
        return _lcd_cox(
            X_aug, y,
            l1_ratio=l1_ratio, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio,
            n_cv=n_cv, random_state=random_state
        )
    if outcome_type == "linear":
        return _lcd_linear(
            X_aug, y,
            l1_ratio=l1_ratio, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio,
            n_cv=n_cv, random_state=random_state
        )
    raise ValueError("Unknown outcome_type.")


# Knockoff filters
def _knockoff_plus_threshold(W, fdr):
    ts = np.sort(np.concatenate(([0.0], np.abs(W))))
    ratios = np.array([
        (1 + np.sum(W <= -t)) / max(1, np.sum(W >= t))
        for t in ts
    ])
    ok = np.flatnonzero(ratios <= fdr)
    if ok.size > 0:
        return ts[ok[0]]
    return float("inf")


def _knockoff_plus_threshold_improved(W, alpha_kn):
    """Improved stoping time for the derandomized knockoff."""
    tau = _knockoff_plus_threshold(W, fdr=alpha_kn)

    # Sorting in decreasing order
    order = np.argsort(-np.abs(W), kind="stable")
    sorted_W = W[order]

    if np.sum(W > 0) >= 1.0 / alpha_kn:
        pos_ind = np.flatnonzero(sorted_W > 0)

        # Original R implementation:
        # pos_ind[ceiling(1/gamma) - 1]
        r_idx = int(np.ceil(1.0 / alpha_kn)) - 1
        py_idx = r_idx - 1
        tau1 = sorted_W[pos_ind[py_idx]]
    else:
        tau1 = 0.0  
    return min(tau, tau1)


def select_knockoff_single(
    X,
    Xk,
    y,    
    outcome_type="survival", # or "linear"
    fdr_levels = (0.05, 0.10, 0.15, 0.20),
    l1_ratio=1.0,
    n_alphas=100,
    alpha_min_ratio=0.01,
    n_cv=5,
    random_state=None
):
    _, stat_seed = _spawn_two_seeds(random_state)
    
    X_arr, feat_names = _prepare_X(X)
    X_tilde = np.asarray(Xk)
    X_aug = np.hstack([X_arr, X_tilde])
    
    reg = "lasso" if l1_ratio == 1.0 else f"en{l1_ratio:.2f}"
    tag = f"single_knockoff_{outcome_type}_{reg}"

    W = _compute_W(
        X_aug, y,
        outcome_type=outcome_type, l1_ratio=l1_ratio, n_alphas=n_alphas,
        alpha_min_ratio=alpha_min_ratio, n_cv=n_cv,
        random_state=stat_seed
    )
    
    out = {}
    for q in fdr_levels:
        threshold = _knockoff_plus_threshold(W, q)
        selected = np.where(
            W >= threshold)[0] if np.isfinite(threshold) else np.array([], dtype=int)
        out[f"{tag}_q{q:.2f}"] = [feat_names[j] for j in selected]

    return out


def select_knockoff_derandomized(
    X,
    Xks,
    y,
    outcome_type="survival", # or "linear"
    fdr_levels = (0.05, 0.10, 0.15, 0.20),
    q_kn_ratio=0.5,
    l1_ratio=1.0,
    n_alphas=100,
    alpha_min_ratio=0.01,
    n_cv=5,
    random_state=None
):
    X_arr, feat_names = _prepare_X(X)
    _, p = X_arr.shape
    n_realizations = len(Xks)

    master = np.random.SeedSequence(random_state)
    realization_ss = master.spawn(n_realizations)
    realization_seeds = [int(s.generate_state(1)[0]) for s in realization_ss]

    def _compute_one(seed, Xk, l1):
        _, stat_seed = _spawn_two_seeds(seed)
        X_tilde = np.asarray(Xk)
        X_aug = np.hstack([X_arr, X_tilde])
        W = _compute_W(
            X_aug, y,
            outcome_type=outcome_type, l1_ratio=l1, n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio, n_cv=n_cv,
            random_state=stat_seed
        )
        return W

    reg = "lasso" if l1_ratio == 1.0 else f"en{l1_ratio:.2f}"
    tag = f"derand_knockoff_{outcome_type}_{reg}"
    
    Ws = []
    for n, seed in enumerate(realization_seeds):
        W = _compute_one(seed, Xks[n], l1_ratio)
        Ws.append(W)
    W_stack = np.stack(Ws, axis=0)
    
    out = {}
    for q_ebh in fdr_levels:
        q_kn = q_ebh * q_kn_ratio
        e_stack = np.zeros((n_realizations, p))
        for m, W_m in enumerate(W_stack):
            threshold = _knockoff_plus_threshold_improved(W_m, q_kn)
            if not np.isfinite(threshold):
                continue
            n_neg = int((W_m <= -threshold).sum())
            e_stack[m] = p * (W_m >= threshold) / (1.0 + n_neg)
        e_bar = e_stack.mean(axis=0)

        order = np.argsort(-e_bar)
        e_sorted = e_bar[order]
        ks = np.arange(1, p + 1)
        feasible = e_sorted >= p / (q_ebh * ks)
        if feasible.any():
            k_hat = int(np.where(feasible)[0].max() + 1)
            selected = order[:k_hat]
        else:
            selected = np.array([], dtype=int)
        out[f"{tag}_q_kn_{q_kn_ratio}_q{q_ebh:.2f}"] = [feat_names[j] for j in selected]

    return out
