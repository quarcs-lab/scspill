"""scspill: synthetic control models with spillover effects.

Synthetic control estimators that relax SUTVA -- the treatment is allowed to
reach the donor pool -- and that report two estimands: the effect on the
treated unit, purged of the contamination, and the spillover effect received
by every control unit.

The package is organized around a model layer, selected by ``method``.
Exactly one model ships today:

``"sar"``
    The Bayesian spatial-autoregressive model of Sakaguchi & Tagawa (2026),
    https://doi.org/10.1093/ectj/utag006. Spillovers from the treated unit to
    the donor pool travel through a spatial-autoregressive structure with
    user-supplied weights, and both estimands are identified from the
    synthetic-control weights ``alpha``, the spillover intensity ``rho``, and
    the weights ``(w, W)`` alone. Estimation is a two-step Bayesian sampler: a
    horseshoe Gibbs sampler for ``alpha`` on the pre-treatment fit, then a SAR
    block (latent AR(1) factors, covariates, and a random-walk Metropolis step
    for ``rho``) conditional on the posterior mean of ``alpha``.

Further spillover-aware models are planned behind the same interface; the
catalogue at https://quarcs-lab.github.io/scspill/models/ tracks them and
states plainly which are implemented.

Public API::

    from scspill import SCSPILL, SCSPILLConfig

    result = SCSPILL(config).fit()   # -> SCSPILLResults  (method="sar")

plus the companion subpackages :mod:`scspill.validation` (the ``sar`` model's
Geweke joint distribution test, prior sensitivity, and prior predictive
checks), :mod:`scspill.simulate` (that model's Monte Carlo engine), and
:mod:`scspill.data` (the bundled California Proposition 99 and Sudan
secession spillover panels).
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
