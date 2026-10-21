"""Generate synthetic data"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky
from itertools import combinations

from .data_classes import SimulationConfig
from .outcome_models import _generate_linear_outcome, _generate_survival_outcome


def _sample_covariate_specs(
    n_features, covariate_schema, covariate_params, rng
):
    # Sample covariate types
    keys = list(covariate_schema)
    w = np.array([covariate_schema[k] for k in keys])
    w = np.clip(w, 0.0, None)

    w = w / w.sum()
    covariate_types = rng.choice(keys, size=n_features, replace=True, p=w).tolist()

    # Apply features marginal parameters
    specs = []
    for t in covariate_types:
        spec = {"type": t}

        if t == "binary":
            prev = covariate_params["binary_prevalence"]
            if isinstance(prev, tuple):
                prev = rng.uniform(*prev)
            spec["prevalence"] = np.clip(prev, 0.01, 0.99)

        elif t == "ordinal":
            k = covariate_params["ordinal_levels"]
            if isinstance(k, tuple):
                k = int(rng.integers(int(k[0]), int(k[1]) + 1))
            spec["levels"] = int(np.clip(k, 2, 10))

            # Sample per-feature probability vector from a Dirichlet.
            # Higher concentration -> closer to uniform; lower -> more peaked.
            concentration = covariate_params["ordinal_concentration"]
            if isinstance(concentration, tuple):
                concentration = rng.uniform(*concentration)
            concentration = max(concentration, 1e-3)
            spec["probs"] = rng.dirichlet([concentration] * k)

        elif t == "lognormal":
            sigma = covariate_params["lognormal_sigma"]
            if isinstance(sigma, tuple):
                sigma = rng.uniform(*sigma)
            spec["sigma"] = max(sigma, 1e-3)

        specs.append(spec)

    return specs


def _inverse_cdf(spec, u):
    t = spec["type"]
    if t == "normal":
        return stats.norm.ppf(u)
    if t == "lognormal":
        return stats.lognorm(s=spec["sigma"], scale=1.0).ppf(u)
    if t == "binary":
        return (u > 1 - spec["prevalence"]).astype(float)
    if t == "ordinal":
        probs = np.asarray(spec["probs"])
        thresholds = np.cumsum(probs)[:-1]
        return np.searchsorted(thresholds, u, side="right").astype(float)


def _apply_marginals(U, specs, standardize=True):
    n, p = U.shape
    U = np.clip(U, 1e-9, 1 - 1e-9)
    X = np.empty((n, p))

    for j, spec in enumerate(specs):
        xj = _inverse_cdf(spec, U[:, j])
        if standardize:
            mu = np.mean(xj)
            sd = np.std(xj, ddof=0)
            if sd < 1e-12:
                xj = xj - mu
            else:
                xj = (xj - mu) / sd
        X[:, j] = xj
    return X


def _sample_copula_uniforms(
    n_samples, latent_corr, copula_family, copula_df=None, rng=None
):
    """
    Generates uniform marginals with dependence structure determined
    by latent_corr and copula parameters.
    """
    p = latent_corr.shape[0]
    L = cholesky(latent_corr, lower=True)


    Z = rng.standard_normal((n_samples, p)) @ L.T
    if copula_family == "gaussian":
        return stats.norm.cdf(Z)
    if copula_family == "t":
        W = rng.chisquare(copula_df, size=n_samples)
        scale = np.sqrt(copula_df / W)[:, None]
        return stats.t.cdf(Z * scale, copula_df)


def _pd_correction(corr):
    """Projects to a near positive-definite matrix"""
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    out = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Rescale
    d = np.sqrt(np.diag(out))
    out = out / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out


def _spearman_to_latent(rho_spearman):
    """Spearman correlation matrix to the latent correlation matrix"""
    rho = np.clip(rho_spearman, -1.0, 1.0)
    sigma = 2.0 * np.sin(np.pi * rho / 6.0)
    np.fill_diagonal(sigma, 1.0)
    return sigma


def _generate_correlation_matrix(
    n_features,
    n_clusters,
    cluster_size,
    isolated_feature_fraction,
    within_cluster_correlation,
    positive_feature_rate,
    rng
):
    p = int(n_features)
    n_isolated = int(round(isolated_feature_fraction * p))
    n_to_cluster = p - n_isolated

    if isinstance(cluster_size, tuple):
        min_size, max_size = int(cluster_size[0]), int(cluster_size[1])
    else:
        min_size = max_size = int(cluster_size)

    if n_clusters * min_size > n_to_cluster:
        raise ValueError(
            f"Infeasible: n_clusters * min_size = {n_clusters * min_size} > "
            f"non-isolated features = {n_to_cluster}. Reduce n_clusters, "
            f"reduce min cluster_size, or reduce isolated_feature_fraction."
        )
    if n_clusters * max_size < n_to_cluster:
        raise ValueError(
            f"Infeasible: n_clusters * max_size = {n_clusters * max_size} < "
            f"non-isolated features = {n_to_cluster}. Increase n_clusters, "
            f"increase max cluster_size, or increase isolated_feature_fraction."
        )

    if isinstance(within_cluster_correlation, tuple):
        wc_lo, wc_hi = within_cluster_correlation
    else:
        wc_lo = wc_hi = within_cluster_correlation

    # Picking isolated features
    all_indices = np.arange(p)
    rng.shuffle(all_indices)
    isolated_indices = sorted(all_indices[:n_isolated].tolist())
    non_isolated = all_indices[n_isolated:]
    rng.shuffle(non_isolated)

    # Sampling cluster sizes (summing to n_to_cluster)
    sizes = np.full(n_clusters, min_size, dtype=int)
    remaining = np.full(n_clusters, max_size - min_size, dtype=int)
    leftover = n_to_cluster - n_clusters * min_size
    for _ in range(leftover):
        eligible = np.where(remaining > 0)[0]
        if len(eligible) == 0:
            break
        picked = int(rng.choice(eligible))
        sizes[picked] += 1
        remaining[picked] -= 1

    # Assigning features to clusters
    clusters = []
    pos = 0
    for s in sizes:
        clusters.append(sorted(non_isolated[pos:pos + s].tolist()))
        pos += s
    
    # Sampling Spearman correlation for each cluster
    if wc_lo == wc_hi:
        cluster_levels = np.full(n_clusters, wc_lo)
    else:
        cluster_levels = rng.uniform(wc_lo, wc_hi, size=n_clusters)

    # Sampling feature signs within each cluster
    feature_signs = np.ones(p)
    if positive_feature_rate < 1.0:
        for members in clusters:
            for member_idx in members:
                if rng.random() >= positive_feature_rate:
                    feature_signs[member_idx] = -1.0

    # Build the target correlation matrix
    spearman_target = np.eye(p)
    p_pairs = 0
    n_pairs = 0
    for c_idx, members in enumerate(clusters):
        corr_magnitude = abs(cluster_levels[c_idx])
        for i, j in combinations(members, 2):
            pair_sign = feature_signs[i] * feature_signs[j]
            val = pair_sign * corr_magnitude
            spearman_target[i, j] = spearman_target[j, i] = val
            if pair_sign > 0:
                p_pairs += 1
            else:
                n_pairs += 1

    latent_corr = _spearman_to_latent(spearman_target)

    # Ensure PDness
    pre_pd_min_eig = np.linalg.eigvalsh(latent_corr).min()
    needed_projection = pre_pd_min_eig < 1e-8
    if needed_projection:
        latent_corr_pre = latent_corr.copy()
        latent_corr = _pd_correction(latent_corr)
        max_pd_shift = np.max(np.abs(latent_corr - latent_corr_pre))
    else:
        max_pd_shift = 0.0
    post_pd_min_eig = np.linalg.eigvalsh(latent_corr).min()

    cluster_summaries = [
        {
            "cluster_idx": c_idx,
            "size": sizes[c_idx],
            "members": members,
            "spearman_magnitude": cluster_levels[c_idx],
            "n_positive_signs": np.sum(feature_signs[members] > 0),
            "n_negative_signs": np.sum(feature_signs[members] < 0)
        }
        for c_idx, members in enumerate(clusters)
    ]

    diagnostics = {
        "n_isolated": n_isolated,
        "isolated_indices": isolated_indices,
        "n_positive_pairs": p_pairs,
        "n_negative_pairs": n_pairs,
        "feature_signs": feature_signs.tolist(),
        "clusters": cluster_summaries,
        "pd_projection": {
            "was_needed": needed_projection,
            "pre_min_eigenvalue": pre_pd_min_eig,
            "post_min_eigenvalue": post_pd_min_eig,
            "max_shift": max_pd_shift
        }
    }

    return latent_corr, spearman_target, diagnostics


def _generate_coefficients(
    n_features,
    n_strong, n_moderate, n_weak,
    strong_range, moderate_range, weak_range,
    feature_signs=None, isolated_indices=None,
    align_signs=True, rng=None
):
    n_true = n_strong + n_moderate + n_weak

    isolated_set = set(int(i) for i in isolated_indices) if isolated_indices is not None else set()

    # Sampling true features uniformly without replacement
    if n_true > 0:
        true_indices = rng.choice(n_features, size=n_true, replace=False)
    else:
        true_indices = np.array([], dtype=int)

    coefs = np.zeros(n_features)
    labels = np.full(n_features, "N", dtype=object)

    def _draw_coef(n, lo, hi):
        if n == 0:
            return np.array([])
        return rng.uniform(lo, hi, size=n)

    tier_labels = ["S"] * n_strong + ["M"] * n_moderate + ["W"] * n_weak
    tier_magnitudes = np.concatenate([
        _draw_coef(n_strong, *strong_range),
        _draw_coef(n_moderate, *moderate_range),
        _draw_coef(n_weak, *weak_range)
    ])
    
    for idx, label, mag in zip(true_indices, tier_labels, tier_magnitudes):
        idx = int(idx)
        
        # Aligning coefficient and cluster signs
        if align_signs and idx not in isolated_set:
            sign = feature_signs[idx]
        else:
            sign = rng.choice([-1.0, 1.0])
            
        coefs[idx] = sign * mag
        labels[idx] = label

    return coefs, labels


def generate_data(config: SimulationConfig):
    rng = np.random.default_rng(config.random_state)
    n, p = config.n_samples, config.n_features

    cov_specs = _sample_covariate_specs(
        p, config.covariate_schema, config.covariate_params, rng
    )

    latent_corr, spearman_target, corr_diag = _generate_correlation_matrix(
        n_features=p, **config.corr_params, rng=rng
    )

    coefs, coef_labels = _generate_coefficients(
        n_features=p,
        **config.coef_params,
        feature_signs=corr_diag["feature_signs"],
        isolated_indices=corr_diag["isolated_indices"],
        rng=rng
    )

    U = _sample_copula_uniforms(
        n_samples=n,
        latent_corr=latent_corr,
        **config.copula_params,
        rng=rng
    )
    X = _apply_marginals(U, cov_specs, standardize=True)

    outcome_kwargs = config.outcome_params.copy()
    outcome_kwargs.setdefault("rng", rng)
    if config.outcome_type == "survival":
        y, y_diag = _generate_survival_outcome(
            X=X, beta=coefs, **outcome_kwargs
        )
    elif config.outcome_type == "linear":
        y, y_diag = _generate_linear_outcome(
            X=X, beta=coefs, **outcome_kwargs
        )
    else:
        raise ValueError("Unknown outcome_type.")

    # Realized correlations
    pair_i = []
    pair_j = []
    for cluster in corr_diag["clusters"]:
        for i, j in combinations(sorted(int(x) for x in cluster["members"]), 2):
            pair_i.append(i)
            pair_j.append(j)

    realized_correlations = None
    if pair_i:
        pearson_mat = np.corrcoef(X, rowvar=False)
        sp = stats.spearmanr(X).statistic
        ii = np.asarray(pair_i)
        jj = np.asarray(pair_j)
        target_vals = spearman_target[ii, jj]
        spearman_vals = sp[ii, jj]
        pearson_vals = pearson_mat[ii, jj]
        realized_correlations = {
            "spearman_mean_abs_error": np.mean(np.abs(spearman_vals - target_vals)),
            "spearman_max_abs_error": np.max(np.abs(spearman_vals - target_vals)),
            "spearman_realized_range": [np.min(spearman_vals), np.max(spearman_vals)],
            "pearson_realized_range": [np.min(pearson_vals), np.max(pearson_vals)],
            "spearman_mean_abs": np.mean(np.abs(spearman_vals)),
            "pearson_mean_abs": np.mean(np.abs(pearson_vals))
        }

    # Final outputs
    dist_tag_map = {
        "normal": "NORM",
        "binary": "BIN",
        "ordinal": "ORD",
        "lognormal": "LOGN"
    }
    dist_tags = [dist_tag_map.get(spec["type"]) for spec in cov_specs]
    feature_names = [f"X{i}_{coef_labels[i]}_{dist_tags[i]}" for i in range(p)]
    true_features = [f for idx, f in enumerate(feature_names) if coefs[idx] != 0]

    X_df = pd.DataFrame(X, columns=feature_names)

    diagnostics = {
        "correlation": corr_diag,
        "realized_correlations": realized_correlations,
        "coefficients": coefs.tolist(),
        "copula": config.copula_params,
        "outcome": y_diag
    }

    return X_df, y, true_features, diagnostics
