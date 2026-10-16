"""Model-agnostic containers shared by every spillover model.

This is a leaf module (NumPy and pandas only): the model subpackages
(:mod:`.sar`, :mod:`.cd`) build on :class:`PanelInputs` and fill
:class:`SCSPILLEffects`, and :mod:`.structures` assembles the public result
from them, so nothing here may import from those modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _freeze_array(arr: np.ndarray | None) -> None:
    """Mark an array read-only so frozen containers stay effectively immutable."""
    if isinstance(arr, np.ndarray):
        arr.setflags(write=False)


@dataclass(frozen=True, kw_only=True)
class PanelInputs:
    """A prepared long panel, split into treated and control outcomes.

    Units are ordered treated-first (sorted labels), then the controls in
    sorted label order -- the donor order of the R package's
    ``scspill_prep_X`` and of every model in scspill.

    Attributes
    ----------
    Y0 : np.ndarray
        Outcome path of the headline treated unit (the first treated label),
        shape ``(T,)``.
    Yc : np.ndarray
        Control-unit outcome matrix, shape ``(T, N)`` in ``control_labels``
        column order.
    T0 : int
        Number of pre-treatment periods (shared by every treated unit).
    X : np.ndarray or None
        Covariate cube ``(T, N, K)`` for the control units, or ``None``.
    treated_label : Any
        Label of the headline treated unit.
    control_labels : tuple
        Control labels in column order of ``Yc``.
    time_labels : np.ndarray
        Sorted time labels, length ``T``.
    covariate_names : tuple of str
        Names of the covariate columns stacked into ``X``.
    treated_labels : tuple
        Every treated unit, sorted; defaults to ``(treated_label,)``.
    Yt : np.ndarray
        Outcome paths of every treated unit, shape ``(T, n_treated)`` in
        ``treated_labels`` column order; defaults to ``Y0[:, None]``.
    """

    Y0: np.ndarray
    Yc: np.ndarray
    T0: int
    X: np.ndarray | None = None
    treated_label: Any
    control_labels: tuple[Any, ...]
    time_labels: np.ndarray
    covariate_names: tuple[str, ...] = ()
    treated_labels: tuple[Any, ...] = ()
    Yt: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Fill the multi-treated defaults, validate shapes, and freeze arrays."""
        if not self.treated_labels:
            object.__setattr__(self, "treated_labels", (self.treated_label,))
        if self.Yt is None:
            object.__setattr__(self, "Yt", self.Y0.reshape(-1, 1).copy())
        N = len(self.control_labels)
        T = self.Y0.shape[0]
        assert self.Yc.shape == (T, N), "Yc shape does not match (T, N)"
        assert self.time_labels.shape[0] == T, "time_labels length does not match T"
        assert self.treated_labels[0] == self.treated_label, "headline must be treated_labels[0]"
        assert self.Yt is not None and self.Yt.shape == (T, len(self.treated_labels)), (
            "Yt shape does not match (T, n_treated)"
        )
        if self.X is not None:
            assert self.X.shape[:2] == (T, N), "X shape does not match (T, N, K)"
        for arr in (self.Y0, self.Yc, self.Yt, self.X, self.time_labels):
            _freeze_array(arr)

    @property
    def N(self) -> int:
        """Number of control units."""
        return len(self.control_labels)

    @property
    def n_treated(self) -> int:
        """Number of treated units."""
        return len(self.treated_labels)

    @property
    def unit_labels(self) -> tuple[Any, ...]:
        """Every unit: the treated units first, then the controls."""
        return tuple(self.treated_labels) + tuple(self.control_labels)

    @property
    def T(self) -> int:
        """Total number of time periods."""
        return int(self.Y0.shape[0])

    @property
    def T1(self) -> int:
        """Number of post-treatment periods."""
        return self.T - self.T0

    @property
    def K(self) -> int:
        """Number of covariates."""
        return 0 if self.X is None else int(self.X.shape[2])

    @property
    def time_pre(self) -> np.ndarray:
        """Pre-treatment time labels."""
        return self.time_labels[: self.T0]

    @property
    def time_post(self) -> np.ndarray:
        """Post-treatment time labels."""
        return self.time_labels[self.T0 :]


@dataclass(frozen=True)
class SCSPILLEffects:
    """Posterior treatment and spillover effects (Theorems 3.1-3.2).

    All paths cover the full sample (pre and post periods); pre-period rows of
    the spillover panel are fit residuals by construction, not causal
    spillovers.

    Attributes
    ----------
    att : float
        Posterior mean of the average treatment effect on the treated
        (mean over the ``att_draws``).
    att_ci : tuple of float
        Equal-tailed credible interval for the ATT at level ``ci_level``.
    att_draws : np.ndarray
        One ATT draw per retained posterior draw, shape ``(M,)``.
    att_plugin : float
        ATT evaluated at the posterior means ``(alpha_hat, rho_hat)``.
    att_scm : float
        ATT of the ``rho = 0`` special case (plain Bayesian horseshoe SCM at
        ``alpha_hat``) -- the no-spillover comparator.
    cf_mean, cf_lower, cf_upper : np.ndarray
        Posterior mean and pointwise credible band of the treated unit's
        untreated counterfactual path, each shape ``(T,)``.
    spill_mean, spill_lower, spill_upper : pd.DataFrame
        Posterior mean and pointwise credible band of the spillover effects,
        each ``(T, N)`` indexed by time labels with donor-label columns.
    ci_level : float
        Credible level of all intervals.
    propagate_alpha : bool
        True when effect draws paired ``(alpha^(m), rho^(m))``; False when
        ``alpha`` was held at ``alpha_hat`` (the R package's convention).
    n_draws_used : int
        Number of posterior draws swept in the effects computation.
    """

    att: float
    att_ci: tuple[float, float]
    att_draws: np.ndarray
    att_plugin: float
    att_scm: float
    cf_mean: np.ndarray
    cf_lower: np.ndarray
    cf_upper: np.ndarray
    spill_mean: pd.DataFrame
    spill_lower: pd.DataFrame
    spill_upper: pd.DataFrame
    ci_level: float
    propagate_alpha: bool
    n_draws_used: int

    def __post_init__(self) -> None:
        """Freeze the draw arrays."""
        for arr in (self.att_draws, self.cf_mean, self.cf_lower, self.cf_upper):
            _freeze_array(arr)
