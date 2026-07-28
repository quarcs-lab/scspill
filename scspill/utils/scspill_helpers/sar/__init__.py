"""The ``method="sar"`` model: Sakaguchi & Tagawa's Bayesian spillover SCM.

Implements:

    Sakaguchi, S., & Tagawa, H. (2026). "Identification and Bayesian
    Inference for Synthetic Control Methods with Spillover Effects." The
    Econometrics Journal. https://doi.org/10.1093/ectj/utag006

Spillovers travel through a spatial-autoregressive structure on the donor
outcomes, with user-supplied weights ``(w, W)`` and a single intensity
``rho``. Both the effect on the treated unit and the effect received by each
donor are identified from ``(alpha, rho, w, W)`` alone. At ``rho = 0`` the
model collapses exactly to the Bayesian horseshoe synthetic control.

Module layout:

* :mod:`.pipeline` -- :func:`run_scspill`, the two-step driver.
* :mod:`.sampler_alpha` -- Step 1, the horseshoe Gibbs sampler for ``alpha``.
* :mod:`.sampler_sar` -- Step 2, the SAR block conditional on ``alpha_hat``.
* :mod:`.effects` -- the identification formulas and the ``RhoSolver``.
* :mod:`.inference` -- posterior draws to the standardized inference models.
* :mod:`._kernels` / :mod:`.numba_kernels` -- the single-source sampler loops
  and their optional JIT compilation.

Everything above this subpackage is model-agnostic: panel ingestion, the
spatial-weight primitives, the result containers, diagnostics, and plotting
are shared by any spillover model added alongside this one.
"""

from .effects import RhoSolver, posterior_effects
from .inference import build_inference, prediction_interval_spec
from .pipeline import run_scspill
from .sampler_alpha import hs_alpha_gibbs
from .sampler_sar import sar_step2_sampler

__all__ = [
    "RhoSolver",
    "build_inference",
    "hs_alpha_gibbs",
    "posterior_effects",
    "prediction_interval_spec",
    "run_scspill",
    "sar_step2_sampler",
]
