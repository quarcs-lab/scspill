"""Typed containers of the ``sar`` model: its inputs and posterior blocks.

:class:`SARInputs` extends the shared :class:`PanelInputs` with the
normalized spatial weights; :class:`AlphaPosterior` and :class:`SARPosterior`
hold the two steps' draws, and :class:`SARFit` bundles everything the
pipeline produced. The public result exposes a fit through
``SCSPILLResults.sar``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..containers import PanelInputs, SCSPILLEffects, _freeze_array


@dataclass(frozen=True, kw_only=True)
class SARInputs(PanelInputs):
    """Prepared estimation inputs for the ``sar`` model.

    Adds the spatial structure to :class:`PanelInputs`.

    Attributes
    ----------
    Wn : np.ndarray
        Row-normalized control-to-control spatial weight matrix, ``(N, N)``.
    wn : np.ndarray
        Sum-normalized treated-to-control exposure vector, ``(N,)``.
    W_raw, w_raw : np.ndarray
        The user-supplied spatial weights before normalization (label-aligned).
    """

    Wn: np.ndarray
    wn: np.ndarray
    W_raw: np.ndarray
    w_raw: np.ndarray

    def __post_init__(self) -> None:
        """Validate the spatial shapes on top of the panel checks."""
        super().__post_init__()
        N = len(self.control_labels)
        assert self.Wn.shape == (N, N), "Wn shape does not match (N, N)"
        assert self.wn.shape == (N,), "wn shape does not match (N,)"
        for arr in (self.Wn, self.wn, self.W_raw, self.w_raw):
            _freeze_array(arr)


@dataclass(frozen=True)
class AlphaPosterior:
    """Step-1 posterior: horseshoe draws of the synthetic weights ``alpha``.

    Attributes
    ----------
    draws : np.ndarray
        Post-burn ``alpha`` draws in original outcome units, shape ``(M, N)``.
    sigma2 : np.ndarray
        Post-burn draws of the Step-1 error variance (on the standardized
        scale), shape ``(M,)``.
    alpha_hat : np.ndarray
        Posterior mean of ``alpha``, shape ``(N,)`` -- the plug-in weights
        used by Step 2.
    iters, burn : int
        Total iterations and burn-in of the sampler.
    """

    draws: np.ndarray
    sigma2: np.ndarray
    alpha_hat: np.ndarray
    iters: int
    burn: int

    def __post_init__(self) -> None:
        """Freeze the draw arrays."""
        for arr in (self.draws, self.sigma2, self.alpha_hat):
            _freeze_array(arr)


@dataclass(frozen=True)
class SARPosterior:
    """Step-2 posterior: the SAR block conditional on ``alpha_hat``.

    Attributes
    ----------
    rho : np.ndarray
        Post-burn draws of the spillover intensity, shape ``(M,)``.
    sigma2 : np.ndarray
        Post-burn draws of the innovation variance, shape ``(M,)``.
    beta : np.ndarray or None
        Post-burn draws of the covariate coefficients, shape ``(M, K)``, or
        ``None`` when there are no covariates.
    rho_hat : float
        Posterior mean of ``rho``.
    acc_rho : float
        Post-burn acceptance rate of the random-walk Metropolis step.
    step_rho_final : float
        Random-walk step size after burn-in adaptation (equals the initial
        ``step_rho`` when adaptation is off).
    rho_bound : float
        Half-width of the admissible ``rho`` support,
        ``0.95 / max(1, max |eig(Wn)|)``.
    beta_prior : str
        Prior used for ``beta``: ``"horseshoe"`` (paper) or ``"ridge"``
        (R-code compatibility).
    p_factors : int
        Number of AR(1) latent factors in the error model.
    iters, burn : int
        Total iterations and burn-in of the sampler.
    """

    rho: np.ndarray
    sigma2: np.ndarray
    beta: np.ndarray | None
    rho_hat: float
    acc_rho: float
    step_rho_final: float
    rho_bound: float
    beta_prior: str
    p_factors: int
    iters: int
    burn: int

    def __post_init__(self) -> None:
        """Freeze the draw arrays."""
        for arr in (self.rho, self.sigma2, self.beta):
            _freeze_array(arr)


@dataclass(frozen=True)
class SARFit:
    """Bundle of everything the ``sar`` pipeline produced."""

    inputs: SARInputs
    alpha_posterior: AlphaPosterior
    sar_posterior: SARPosterior
    effects: SCSPILLEffects
    scm_weights: dict[str, float] = field(default_factory=dict)
    mcmc_summary_table: pd.DataFrame | None = None


#: Pre-0.3 names, kept so existing imports keep working.
SCSPILLInputs = SARInputs
SCSPILLFit = SARFit
