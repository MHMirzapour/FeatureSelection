"""
Complementary Pairs Stability Selection (CPSS) with approximate FDR control
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import lasso_path, enet_path
from sksurv.linear_model import CoxnetSurvivalAnalysis


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


def _complementary_split(n, rng):
    sh = rng.permutation(n)
    mid = n // 2
    return sh[:mid], sh[mid:2 * mid]


def _one_pair(seed, X, y, alphas, l1_ratio, indicator_fn):
    rng = np.random.default_rng(seed)
    A, B = _complementary_split(X.shape[0], rng)
    return (
        indicator_fn(X[A], y[A], alphas=alphas, l1_ratio=l1_ratio),
        indicator_fn(X[B], y[B], alphas=alphas, l1_ratio=l1_ratio)
    )


def _compute_stability_paths(
    X, y, alphas, l1_ratio, n_pairs, indicator_fn,
    n_jobs=-1, random_state=None
):
    seeds = np.random.SeedSequence(random_state).spawn(n_pairs)
    seed_ints = [int(s.generate_state(1)[0]) for s in seeds]

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_pair)(
            seed_ints[b], X, y, alphas, l1_ratio, indicator_fn
        )
        for b in range(n_pairs)
    )

    L, p = len(alphas), X.shape[1]
    pi_sum = np.zeros((L, p))
    q_sum = np.zeros(L)
    for indA, indB in results:
        pi_sum += indA + indB
        q_sum += (
            np.logical_or.accumulate(indA, axis=0).sum(axis=1)
            + np.logical_or.accumulate(indB, axis=0).sum(axis=1)
        )
    pi_path = pi_sum / (2 * n_pairs)
    q_path = q_sum / (2 * n_pairs)
    return pi_path, q_path


# Shah-Samworth unimodal bound and selection
def _check_unimodal_domain(tau, B):
    """
    Validate (tau, B) against the domain of the Shah-Samworth (2013)
    unimodal bound (Theorem 5 and 6).

    The lower-tau branch of the closed form requires the denominator
    2*(2*tau - 1 - 1/(2B)) to be positive, i.e., tau > 1/2 + 1/(4B).
    """
    if not (0.5 < tau <= 1.0):
        raise ValueError("tau must be in (0.5, 1]")
    if B < 1:
        raise ValueError("n_pairs must be >= 1")
    lower_limit = 0.5 + 1.0 / (4.0 * B)
    if tau <= lower_limit:
        raise ValueError(
            f"tau={tau} with n_pairs={B} is outside the Shah-Samworth "
            f"unimodal-bound domain (need tau > 1/2 + 1/(4B) = "
            f"{lower_limit:.4f}). Increase tau or n_pairs."
        )


def _unimodal_constant(tau, B):
    """
    Shah-Samworth (2013) unimodal-bound constant C(tau, B).

    For tau in (1/2, 3/4]:
        C(tau, B) = 1 / [2 * (2*tau - 1 - 1/(2B))]
    For tau in (3/4, 1]:
        C(tau, B) = 4 * (1 - tau + 1/(2B)) / (1 + 1/B)
    """
    if tau <= 0.5:
        raise ValueError("tau must be > 0.5.")
    if tau <= 0.75:
        denom = 2.0 * (2.0 * tau - 1.0 - 1.0 / (2.0 * B))
        if denom <= 0:
            return float("inf")
        return 1.0 / denom
    else:
        return 4.0 * (1.0 - tau + 1.0 / (2.0 * B)) / (1.0 + 1.0 / B)


def _select_pfer(
    pi_path, q_path, tau, target_pfer, B
):
    """
    Select features whose max stability across the operating window
    exceeds tau, where the window is the largest path-prefix with
    Shah-Samworth PFER bound <= target_pfer.
    
    The bound:
        E(FP) <= C(tau, B) * q^2 / p
    """
    p = pi_path.shape[1]
    C = _unimodal_constant(tau, B)
    if not np.isfinite(C):
        # Bound is vacuous (tau, B) outside unimodal domain; reject nothing.
        return np.zeros(p)
    bounds = C * q_path ** 2 / p
    feasible = bounds <= target_pfer
    if not feasible.any():
        return np.zeros(p)
    last = int(np.where(feasible)[0].max())
    pi_window = pi_path[:last + 1]
    return pi_window.max(axis=0) >= tau


def _select_fdr(
    pi_path, q_path, tau, q_target, B
):
    """
    The PFER bound delivers E[V] <= C(tau, B) * q_Lambda^2 / p. To
    convert that into an approximate target FDR (E[V] / |S| <= q_target),
    we need a size s and a selection set S(s) (built by thresholding
    pi_path over the operating window induced by s) such that
        |S(s)| >= s and PFER bound at that window <= q_target * s.

    f(s) := |S(s)| is monotone non-decreasing in s (a larger PFER
    budget admits a wider operating window, which can only keep or
    increase the number of features whose max stability exceeds tau).
    The map f typically has multiple fixed points; we want the
    largest s with f(s) >= s, which is the largest self-consistent
    selection set. We find it by scanning s = 1, 2, ..., p.
    """
    p = pi_path.shape[1]
    best_mask = np.zeros(p)
    for s in range(1, p + 1):
        mask = _select_pfer(
            pi_path, q_path,
            tau=tau, target_pfer=q_target * s, B=B
        )
        if int(mask.sum()) >= s:
            best_mask = mask
    return best_mask


# Main function
def select_cpss(
    X,
    y,
    outcome_type = "survival", # or "linear"
    fdr_levels = (0.05, 0.10, 0.15, 0.20),
    tau = 0.75, # must be in (0.5, 1]
    n_pairs = 50,
    l1_ratio = 1.0,
    n_alphas = 50,
    alpha_min_ratio = 0.01,
    n_jobs = -1,
    random_state = None
):
    
    _check_unimodal_domain(tau, n_pairs)

    if isinstance(X, pd.DataFrame):
        feat_names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        feat_names = [f"X{i}" for i in range(X_arr.shape[1])]
    n, p = X_arr.shape

    reg = "lasso" if l1_ratio == 1.0 else f"en{l1_ratio:.2f}"
    tag = f"cpss_{outcome_type}_{reg}"

    if outcome_type == "survival":
        indicator_fn = _cox_indicators
        pilot = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio, n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio, fit_baseline_model=False
        )
        pilot.fit(X_arr, y)
        alphas = pilot.alphas_

    elif outcome_type == "linear":
        indicator_fn = _linear_indicators
        if l1_ratio == 1.0:
            alphas, _, _ = lasso_path(
                X_arr, y,
                n_alphas=n_alphas, eps=alpha_min_ratio
            )
        else:
            alphas, _, _ = enet_path(
                X_arr, y,
                n_alphas=n_alphas, eps=alpha_min_ratio,
                l1_ratio=l1_ratio
            )
    else:
        raise ValueError("Unknown outcome_type.")

    pi_path, q_path = _compute_stability_paths(
        X_arr, y,
        alphas=alphas, l1_ratio=l1_ratio,
        n_pairs=n_pairs, indicator_fn=indicator_fn,
        n_jobs=n_jobs, random_state=random_state
    )

    out = {}
    for q_target in fdr_levels:
        mask = _select_fdr(
            pi_path, q_path,
            tau=tau, q_target=q_target, B=n_pairs
        )
        out[f"{tag}_q{q_target:.2f}"] = [feat_names[j] for j in np.where(mask)[0]]

    return out
