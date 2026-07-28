"""scspill: Bayesian spatial-spillover synthetic control.

Implements:

    Sakaguchi, S., & Tagawa, H. "Identification and Bayesian Inference for
    Synthetic Control Methods with Spillover Effects." The Econometrics
    Journal.

A synthetic control method that relaxes SUTVA: spillovers from the treated
unit to the donor pool are modeled through a spatial-autoregressive (SAR)
structure with user-supplied spatial weights, and both the treatment effect on
the treated and the spillover effect received by every control unit are
identified from the synthetic-control weights ``alpha``, the spillover
intensity ``rho``, and the weights ``(w, W)``. Estimation is a two-step
Bayesian sampler: a horseshoe Gibbs sampler for ``alpha`` on the
pre-treatment fit, then a SAR block (latent AR(1) factors, covariates, and a
random-walk Metropolis step for ``rho``) conditional on the posterior mean of
``alpha``.

Public API::

    from scspill import SCSPILL, SCSPILLConfig

    result = SCSPILL(config).fit()   # -> SCSPILLResults

plus the companion subpackages :mod:`scspill.validation` (Geweke joint
distribution test, prior sensitivity, prior predictive checks),
:mod:`scspill.simulate` (the paper's Monte Carlo engine), and
:mod:`scspill.data` (the bundled California Proposition 99 and Sudan
secession case studies).
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - fallback exercised only in odd install states
    __version__ = version("scspill")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

from scspill.exceptions import (
    ScspillConfigError,
    ScspillDataError,
    ScspillError,
    ScspillEstimationError,
    ScspillPlottingError,
)

__all__ = [
    "SCSPILL",
    "SCSPILLConfig",
    "SCSPILLResults",
    "ScspillConfigError",
    "ScspillDataError",
    "ScspillError",
    "ScspillEstimationError",
    "ScspillPlottingError",
    "__version__",
]

# Lazy exports (PEP 562): the estimator stack imports pandas/pydantic/scipy,
# so defer those imports until first attribute access for fast cold starts.
_LAZY_EXPORTS = {
    "SCSPILL": ("scspill.estimators.scspill", "SCSPILL"),
    "SCSPILLConfig": ("scspill.utils.scspill_helpers.config", "SCSPILLConfig"),
    "SCSPILLResults": ("scspill.utils.scspill_helpers.structures", "SCSPILLResults"),
}


def __getattr__(name: str):
    """Resolve lazily exported public names (PEP 562)."""
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        import importlib

        module_path, attr = target
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazy exports in ``dir(scspill)``."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
