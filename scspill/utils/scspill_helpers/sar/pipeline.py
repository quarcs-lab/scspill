"""Two-step estimation driver for the SCSPILL estimator.

Runs Step 1 (horseshoe ``alpha``), Step 2 (SAR ``rho`` and nuisances given
``alpha_hat``), sweeps the posterior draws through the identification
formulas, fits the classical simplex-SCM comparator, and precomputes the
default MCMC summary table.

Reproducibility note: sampling runs under ``threadpoolctl``'s single-threaded
BLAS limit, which makes same-seed runs bit-identical on OpenBLAS/MKL builds
(and is faster for the small matrices involved). Apple's Accelerate BLAS is
outside threadpoolctl's control and selects kernels by memory alignment, so
on macOS same-seed runs are statistically identical but may differ at
floating-point rounding level.
"""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from ...diagnostics import mcmc_summary
from ...scm_baseline import classical_scm_weights
from ..structures import SCSPILLFit, SCSPILLInputs
from ._kernels import resolve_backend
from .effects import RhoSolver, posterior_effects
from .sampler_alpha import hs_alpha_gibbs
from .sampler_sar import sar_step2_sampler


def run_scspill(
    inputs: SCSPILLInputs,
    *,
    p_factors: int = 1,
    m_iter: int = 2000,
    burn: int = 1000,
    step_rho: float = 0.05,
    adapt_rho: bool = True,
    target_accept_rho: float = 0.44,
    beta_prior: str = "horseshoe",
    propagate_alpha: bool = True,
    a0: float = 1.0,
    b0: float = 1.0,
    ci: float = 0.95,
    max_effect_draws: int | None = None,
    backend: str = "auto",
    seed: int | None = None,
    verbose: bool = False,
) -> SCSPILLFit:
    """Fit the ``sar`` model: two-step Bayesian inference for the SAR spillover SCM.

    Parameters mirror :class:`scspill.config_models.SCSPILLConfig`; see its
    docstring for the R argument mapping and the paper-correct defaults.

    Parameters
    ----------
    inputs : SCSPILLInputs
        Prepared estimation inputs from
        :func:`~scspill.utils.scspill_helpers.setup.prepare_scspill_inputs`.
    p_factors, m_iter, burn, step_rho, adapt_rho, target_accept_rho : see config
    beta_prior, propagate_alpha, a0, b0, ci, max_effect_draws : see config
    backend : {"auto", "numpy", "numba"}, default "auto"
        Sampler kernel backend.
    seed : int, optional
        Seed for ``numpy.random.default_rng``.
    verbose : bool, default False
        Print stage progress.

    Returns
    -------
    SCSPILLFit
        The posterior blocks, effect summaries, comparator weights, and the
        default diagnostics table.
    """
    kernels = resolve_backend(backend)
    rng = np.random.default_rng(seed)
    T0 = inputs.T0

    with threadpool_limits(limits=1, user_api="blas"):
        if verbose:  # pragma: no cover - cosmetic
            print(f"SCSPILL: Step 1 (horseshoe alpha), {m_iter} iterations [{kernels.name}] ...")
        alpha_post = hs_alpha_gibbs(
            rng, inputs.Y0[:T0], inputs.Yc[:T0], m_iter, burn, kernels=kernels
        )

        if verbose:  # pragma: no cover - cosmetic
            print(f"SCSPILL: Step 2 (SAR rho), {m_iter} iterations [{kernels.name}] ...")
        X_pre = inputs.X[:T0] if inputs.X is not None else None
        sar_post = sar_step2_sampler(
            rng,
            inputs.Yc[:T0],
            alpha_post.alpha_hat,
            inputs.wn,
            inputs.Wn,
            m_iter,
            burn,
            X=X_pre,
            p=p_factors,
            step_rho=step_rho,
            adapt_rho=adapt_rho,
            target_accept=target_accept_rho,
            beta_prior=beta_prior,
            a0=a0,
            b0=b0,
            kernels=kernels,
        )

        if verbose:  # pragma: no cover - cosmetic
            print("SCSPILL: computing posterior effects ...")
        solver = RhoSolver(inputs.Wn, inputs.wn)
        effects = posterior_effects(
            inputs,
            alpha_post.draws,
            sar_post.rho,
            alpha_post.alpha_hat,
            sar_post.rho_hat,
            ci=ci,
            propagate_alpha=propagate_alpha,
            max_draws=max_effect_draws,
            solver=solver,
        )

    scm_w = classical_scm_weights(inputs.Y0[:T0], inputs.Yc[:T0])
    scm_weights = {str(lab): float(w) for lab, w in zip(inputs.control_labels, scm_w, strict=True)}

    # Default diagnostics table: rho, sigma2, and the top donors by |alpha|.
    chains = {"rho": sar_post.rho, "sigma2": sar_post.sigma2}
    labels = list(inputs.control_labels)
    top = np.argsort(-np.abs(alpha_post.alpha_hat))[:6]
    for j in top:
        chains[f"alpha[{labels[j]}]"] = alpha_post.draws[:, j]
    summary = mcmc_summary(chains)

    return SCSPILLFit(
        inputs=inputs,
        alpha_posterior=alpha_post,
        sar_posterior=sar_post,
        effects=effects,
        scm_weights=scm_weights,
        mcmc_summary_table=summary,
    )
