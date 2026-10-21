"""Linear and right-censored survival outcomes"""

import warnings
import numpy as np
from scipy.optimize import brentq


def _generate_linear_outcome(X, beta, target_r_squared, rng=None):
    """
    Linear outcome with calibrated gaussian noise using the empirical variance of X @ beta
    """
    n = X.shape[0]
    eta = X @ beta

    eta_var = eta.var()
    if eta_var < 1e-12:
        sigma_e = 1.0
    else:
        sigma_e = np.sqrt(eta_var * (1.0 - target_r_squared) / target_r_squared)

    y = eta + rng.standard_normal(n) * sigma_e

    y_var = y.var()
    realized_r2 = eta_var / y_var if y_var > 1e-12 else 0.0

    diagnostics = {
        "outcome_type": "linear",
        "calibrated_sigma_e": sigma_e,
        "realized_r_squared": realized_r2
    }

    return y, diagnostics


def _generate_survival_outcome(
    X, beta,
    target_event_rate,
    max_followup,
    hazard_shape,
    p_loss_to_followup,
    time_resolution=None,
    calibration_tol=0.02,
    rng=None
):
    """
    Weibull proportional hazards model with administrative censoring and random loss to follow-up.
    Note that if hazard_shape = 1, the Weibull model reduces to the exponential model with constant hazard.
    """

    n = X.shape[0]
    eta = X @ beta
    exp_eta = np.exp(eta)

    admin_times = np.full(n, max_followup)

    # Loss to follow-up
    is_lt_subject = rng.random(n) < p_loss_to_followup
    lt_times = rng.uniform(0.0, max_followup, size=n)

    censor_times = np.where(is_lt_subject, lt_times, admin_times)

    # Keep the random draws fixed throughout calibration.
    U = rng.uniform(low=1e-12, high=1.0, size=n)
    log_U = np.log(U)

    def _survival_times(lam):
        # T_i^* = (-log(U_i) / exp(eta_i))^(1/rho) / lam
        return (-log_U / exp_eta) ** (1.0 / hazard_shape) / lam

    def _event_rate(lam):
        return (_survival_times(lam) <= censor_times).mean()

    # Lambda calibration via 1-D root finding
    bracket_lo, bracket_hi = 1e-6, 1e3

    rate_at_lo = _event_rate(bracket_lo)
    rate_at_hi = _event_rate(bracket_hi)

    if rate_at_lo > target_event_rate:
        warnings.warn(
            f"target_event_rate={target_event_rate} is below the minimum achievable rate "
            f"({rate_at_lo:.4f}) at lambda={bracket_lo}. Returning closest "
            f"achievable rate. Consider increasing max_followup or "
            f"p_loss_to_followup, or target_event_rate.",
            stacklevel=2
        )
        calibrated_lambda = bracket_lo
    elif rate_at_hi < target_event_rate:
        warnings.warn(
            f"target_event_rate={target_event_rate} is above the maximum achievable rate "
            f"({rate_at_hi:.4f}) at lambda={bracket_hi}. Returning closest "
            f"achievable rate. Consider decreasing p_loss_to_followup or target_event_rate.",
            stacklevel=2
        )
        calibrated_lambda = bracket_hi
    else:
        calibrated_lambda = brentq(
            lambda lam: _event_rate(lam) - target_event_rate,
            bracket_lo, bracket_hi, xtol=1e-6, rtol=1e-6, maxiter=100
        )

    T_star = _survival_times(calibrated_lambda)

    observed_times = np.minimum(T_star, censor_times)
    events = (T_star <= censor_times)

    realized_rate = events.mean()
    calibration_succeeded = abs(realized_rate - target_event_rate) <= calibration_tol

    if not calibration_succeeded and bracket_lo < calibrated_lambda < bracket_hi:
        warnings.warn(
            f"Calibration did not achieve target_event_rate={target_event_rate} within "
            f"tolerance {calibration_tol}: realized rate is {realized_rate:.4f}.",
            stacklevel=2
        )

    if time_resolution is not None:
        # Round up to the next visit.
        observed_times = np.ceil(observed_times / time_resolution) * time_resolution

    y_struct = np.empty(n, dtype=[("event", "?"), ("time", "<f8")])
    y_struct["event"] = events
    y_struct["time"] = observed_times

    diagnostics = {
        "outcome_type": "survival",
        "calibration_succeeded": calibration_succeeded,
        "calibrated_lambda": calibrated_lambda
    }

    return y_struct, diagnostics
