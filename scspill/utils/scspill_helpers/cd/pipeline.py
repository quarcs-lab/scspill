"""The ``cd`` model's driver: SCM equations, effects, and inference."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd

from .estimation import (
    OMEGA_RIDGE,
    effect_operator,
    residual_covariance,
    weighted_effect_operator,
)
from .inference import kappa_tests, pointwise_intervals, pointwise_p_values, reference_residuals
from .scm_core import fit_all_unit_scm
from .sensitivity import pure_donor_sensitivity
from .structures import CDEfficientFit, CDFit, CDInputs


def run_cd(
    inputs: CDInputs,
    *,
    reference_mode: Literal["full", "leave_one_period_out"] = "full",
    quantile_method: Literal["empirical", "linear"] = "empirical",
    ci_method: Literal["inversion", "published"] = "inversion",
    ci: float = 0.95,
    weighting: Literal["identity", "efficient"] = "identity",
    scm_maxiter: int = 3000,
    scm_ftol: float = 1e-12,
) -> CDFit:
    """Fit the Cao--Dowd model and its P-test inference.

    1. Fit an intercept-adjusted simplex SCM equation for every unit on the
       pre-treatment outcomes (v3 Section 3).
    2. Estimate the effect parameters and unit effects in every post period
       from the residual equations and ``A`` (v3 Section 3.2).
    3. Build the pre-treatment reference statistics and the pointwise
       intervals, P values, and ``kappa_A`` tests (v3 Sections 4 and 5.1.2).
    4. Compute the pure-donor sensitivity weights (v3 Section 5.2.2) and,
       with ``weighting="efficient"``, the efficient estimator (v3 Section
       S.1.1).

    Parameters
    ----------
    inputs : CDInputs
        Prepared panel and exposure matrix.
    reference_mode, quantile_method, ci_method, ci, weighting, scm_maxiter, scm_ftol
        See the ``(method='cd')`` fields of ``SCSPILLConfig``.

    Returns
    -------
    CDFit
    """
    labels = inputs.unit_labels
    times = inputs.time_labels
    t0 = inputs.T0
    y = inputs.Y_units
    a_matrix = inputs.A.to_numpy(dtype=float)
    scm = partial(fit_all_unit_scm, maxiter=scm_maxiter, ftol=scm_ftol)

    intercepts, b = scm(y[:, :t0])
    op = effect_operator(b, a_matrix)
    eye = np.eye(len(labels))
    residuals = (eye - b) @ y[:, :t0] - intercepts[:, None]
    post_target = (eye - b) @ y[:, t0:] - intercepts[:, None]
    gamma = op.gamma_map @ post_target
    alpha = op.unit_map @ post_target
    placebo, kappa_pre = reference_residuals(
        y, a_matrix, t0=t0, intercepts=intercepts, b=b, mode=reference_mode, scm=scm
    )

    post_times = times[t0:]
    pointwise, published = pointwise_intervals(
        alpha, placebo, labels=labels, periods=post_times, level=ci, quantile_method=quantile_method
    )
    n_treated = inputs.n_treated
    scm_paths = intercepts[:n_treated, None] + b[:n_treated] @ y
    att_by_unit = {labels[i]: float(alpha[i].mean()) for i in range(n_treated)}
    att_scm_by_unit = {
        labels[i]: float((y[i, t0:] - scm_paths[i, t0:]).mean()) for i in range(n_treated)
    }
    efficient = None
    if weighting == "efficient":
        efficient = _efficient_fit(inputs, b, a_matrix, residuals, post_target)
    return CDFit(
        inputs=inputs,
        intercepts=pd.Series(intercepts, index=list(labels)),
        scm_matrix=pd.DataFrame(b, index=list(labels), columns=list(labels)),
        gamma=pd.DataFrame(gamma.T, index=post_times, columns=inputs.A.columns),
        effects_panel=pd.DataFrame(alpha.T, index=post_times, columns=list(labels)),
        pre_residuals=pd.DataFrame(residuals.T, index=times[:t0], columns=list(labels)),
        placebo_effects=pd.DataFrame(placebo.T, index=times[:t0], columns=list(labels)),
        pointwise_p_values=pointwise_p_values(alpha, placebo, labels=labels, periods=post_times),
        pointwise_ci=pointwise,
        published_ci=published,
        kappa_test=kappa_tests(
            kappa_pre,
            (eye - op.projector) @ post_target,
            periods=post_times,
            level=ci,
            quantile_method=quantile_method,
        ),
        rank=op.rank,
        condition_number=op.condition_number,
        scm_paths=pd.DataFrame(scm_paths.T, index=times, columns=list(labels[:n_treated])),
        att_by_unit=att_by_unit,
        att_scm_by_unit=att_scm_by_unit,
        reference_mode=reference_mode,
        quantile_method=quantile_method,
        ci_method=ci_method,
        ci_level=ci,
        weighting=weighting,
        efficient=efficient,
        sensitivity=pure_donor_sensitivity(inputs, b, op, maxiter=scm_maxiter, ftol=scm_ftol),
    )


def _efficient_fit(
    inputs: CDInputs,
    b: np.ndarray,
    a_matrix: np.ndarray,
    residuals: np.ndarray,
    post_target: np.ndarray,
) -> CDEfficientFit:
    """Fit the ``Omega^{-1}``-weighted estimator of v3 Section S.1.1."""
    labels = list(inputs.unit_labels)
    if len(labels) >= inputs.T0:
        warnings.warn(
            f"weighting='efficient' estimates a {len(labels)}x{len(labels)} residual "
            f"covariance from {inputs.T0} pre-treatment periods, so it is singular "
            "without its ridge and the efficient estimates are unreliable.",
            RuntimeWarning,
            stacklevel=3,
        )
    omega = residual_covariance(residuals)
    weight = np.linalg.inv(omega)
    op = weighted_effect_operator(b, a_matrix, weight)
    alpha = op.unit_map @ post_target
    post_times = inputs.time_post
    return CDEfficientFit(
        omega_hat=pd.DataFrame(omega, index=labels, columns=labels),
        weight_matrix=pd.DataFrame(weight, index=labels, columns=labels),
        ridge=OMEGA_RIDGE,
        gamma=pd.DataFrame(
            (op.gamma_map @ post_target).T, index=post_times, columns=inputs.A.columns
        ),
        effects_panel=pd.DataFrame(alpha.T, index=post_times, columns=labels),
        condition_number=op.condition_number,
        att_by_unit={labels[i]: float(alpha[i].mean()) for i in range(inputs.n_treated)},
    )
