"""Sampler validation and prior checking for the ``sar`` model.

The machinery of that model's article appendix, as user-facing functions:

* :func:`geweke_test` -- the Geweke (2004) joint distribution test of the
  Step-2 Gibbs/Metropolis sampler (marginal-conditional vs.
  successive-conditional simulators), against either the appendix's
  simplified kernel (``kernel="simple"``, comparable to the R replication
  package) or the production sampler users actually run
  (``kernel="production"``).
* :func:`prior_sensitivity` -- re-run the Step-2 posterior across a grid of
  prior settings ``(a0, b0, rho support, step)`` and compare the ``rho``
  posteriors.
* :func:`prior_predictive` / :func:`ppc_stats` -- prior predictive checks of
  the SAR model against nine summary statistics of the pre-treatment donor
  panel.

Unlike their R counterparts, all functions here are pure: they read only
their explicit arguments (the R versions accidentally captured global
``W``/``w`` bindings).
"""

from .geweke import batch_means_variance, default_g_fn, geweke_test
from .kernels import ProductionKernel, SimpleKernel, SimpleState, simulate_yc_forward
from .plotter import plot_geweke, plot_prior_predictive
from .robustness import ppc_stats, prior_predictive, prior_sensitivity, run_posterior_mcmc
from .structures import (
    GewekeReport,
    PosteriorSummary,
    PriorPredictiveResult,
    PriorSensitivityResult,
)

__all__ = [
    "GewekeReport",
    "PosteriorSummary",
    "PriorPredictiveResult",
    "PriorSensitivityResult",
    "ProductionKernel",
    "SimpleKernel",
    "SimpleState",
    "batch_means_variance",
    "default_g_fn",
    "geweke_test",
    "plot_geweke",
    "plot_prior_predictive",
    "ppc_stats",
    "prior_predictive",
    "prior_sensitivity",
    "run_posterior_mcmc",
    "simulate_yc_forward",
]
