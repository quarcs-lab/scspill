"""Typed containers of the ``cd`` model: its inputs, tests, and fit.

:class:`CDInputs` extends the shared :class:`PanelInputs` with the exposure
matrix ``A``; :class:`CDFit` holds everything the pipeline produced. The
public result exposes a fit through ``SCSPILLResults.cd``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..containers import PanelInputs, _freeze_array


@dataclass(frozen=True, kw_only=True)
class CDInputs(PanelInputs):
    """Prepared estimation inputs for the ``cd`` model.

    Adds the exposure structure to :class:`PanelInputs`.

    Attributes
    ----------
    A : pd.DataFrame
        Exposure matrix: one row per unit in ``unit_labels`` order (treated
        units first), one column per effect parameter. Each treated unit's
        row is its own direct-effect indicator.
    spillover_structure : str
        How ``A`` was built (``SCSPILLConfig.spillover_structure``).
    affected_labels : tuple
        Control units with a nonzero row of ``A``: those that may receive a
        spillover.
    """

    A: pd.DataFrame
    spillover_structure: str
    affected_labels: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """Check that ``A`` is aligned with the units."""
        super().__post_init__()
        assert tuple(self.A.index) == self.unit_labels, "A rows must follow unit_labels"

    @property
    def Y_units(self) -> np.ndarray:
        """Outcomes of every unit, units by periods, shape ``(n_units, T)``."""
        assert self.Yt is not None
        out = np.vstack([self.Yt.T, self.Yc.T])
        _freeze_array(out)
        return out

    @property
    def clean_labels(self) -> tuple[Any, ...]:
        """Controls assumed unexposed: the pure donors (zero rows of ``A``)."""
        affected = set(self.affected_labels)
        return tuple(u for u in self.control_labels if u not in affected)


@dataclass(frozen=True)
class CDTestResult:
    """A P test with its pre-treatment reference distribution.

    Attributes
    ----------
    statistic : float
        The post-treatment statistic.
    critical_value : float
        The ``ci``-quantile of the reference statistics.
    p_value : float
        Share of reference statistics at least as large as ``statistic``.
    reject : bool
        Whether ``statistic`` exceeds ``critical_value``.
    reference_statistics : np.ndarray
        The pre-treatment reference statistics, shape ``(T0,)``.
    """

    statistic: float
    critical_value: float
    p_value: float
    reject: bool
    reference_statistics: np.ndarray

    def __post_init__(self) -> None:
        """Freeze the reference statistics."""
        _freeze_array(self.reference_statistics)


@dataclass(frozen=True, kw_only=True)
class CDFit:
    """Everything the ``cd`` pipeline produced.

    Attributes
    ----------
    inputs : CDInputs
        The prepared panel and exposure matrix.
    intercepts : pd.Series
        Intercept ``a_i`` of every unit's SCM equation.
    scm_matrix : pd.DataFrame
        The all-unit SCM weight matrix ``B``; row ``i`` is unit ``i``'s
        simplex weights on the other units.
    gamma : pd.DataFrame
        Estimated effect parameters, post periods by columns of ``A``.
    effects_panel : pd.DataFrame
        Estimated effect on every unit, ``alpha_t = A gamma_t``: post periods
        by unit labels. The treated columns are the direct effects; the
        control columns are the spillovers.
    pre_residuals : pd.DataFrame
        Pre-treatment SCM residuals ``(I - B) Y_t - a``, periods by units.
    placebo_effects : pd.DataFrame
        Pre-treatment placebo effects ``G u_t`` used as P-test references,
        periods by units.
    pointwise_p_values : pd.DataFrame
        P value of ``alpha_{i,t} = 0``, post periods by units.
    pointwise_ci, published_ci : dict
        ``{unit: DataFrame(lower, upper)}`` intervals by test inversion and by
        the authors' signed-quantile rule.
    kappa_test : dict
        ``{post period: CDTestResult}`` of the ``kappa_A`` structure test.
    rank, condition_number : int, float
        Rank of ``(I - B) A`` and condition number of ``A'(I-B)'(I-B)A``.
    scm_paths : pd.DataFrame
        Each treated unit's no-spillover synthetic control ``a_i + B_i Y_t``,
        all periods by treated labels.
    att_by_unit, att_scm_by_unit : dict
        Average post-treatment direct effect of every treated unit, and the
        same from its no-spillover synthetic control.
    reference_mode, quantile_method, ci_method : str
        The inference settings.
    ci_level : float
        Level of every interval and test.
    """

    inputs: CDInputs
    intercepts: pd.Series
    scm_matrix: pd.DataFrame
    gamma: pd.DataFrame
    effects_panel: pd.DataFrame
    pre_residuals: pd.DataFrame
    placebo_effects: pd.DataFrame
    pointwise_p_values: pd.DataFrame
    pointwise_ci: dict[Any, pd.DataFrame]
    published_ci: dict[Any, pd.DataFrame]
    kappa_test: dict[Any, CDTestResult]
    rank: int
    condition_number: float
    scm_paths: pd.DataFrame
    att_by_unit: dict[Any, float]
    att_scm_by_unit: dict[Any, float]
    reference_mode: Literal["full", "leave_one_period_out"]
    quantile_method: Literal["empirical", "linear"]
    ci_method: Literal["inversion", "published"]
    ci_level: float

    @property
    def effect_ci(self) -> dict[Any, pd.DataFrame]:
        """Per-unit pointwise intervals selected by ``ci_method``."""
        return self.published_ci if self.ci_method == "published" else self.pointwise_ci

    def test_hypothesis(
        self,
        C: np.ndarray | pd.DataFrame,
        *,
        d: np.ndarray | float | None = None,
        period: Any | None = None,
        weight: np.ndarray | Literal["inverse_covariance"] | None = None,
        quantile_method: Literal["empirical", "linear"] | None = None,
    ) -> CDTestResult:
        """Test ``C @ alpha_t = d`` by the pre-treatment P distribution.

        Parameters
        ----------
        C : array or DataFrame
            Contrast matrix with one column per unit in ``unit_labels``
            order, or labeled columns when passed as a DataFrame. For example
            ``np.eye(n)[1:]`` tests "no spillover on any control" when there
            is one treated unit.
        d : array or float, optional
            Null value, zero by default.
        period : optional
            A post-treatment period label; the first one by default.
        weight : array or "inverse_covariance", optional
            Weighting matrix of the quadratic statistic: the identity by
            default, an explicit positive-definite matrix, or the inverse
            second moment of the projected placebo effects.
        quantile_method : {"empirical", "linear"}, optional
            Overrides the fit's critical-value rule.

        Returns
        -------
        CDTestResult
        """
        from ....exceptions import ScspillConfigError
        from .inference import parse_contrast, quadratic_test

        if period is None:
            period = self.effects_panel.index[0]
        if period not in self.effects_panel.index:
            raise ScspillConfigError(f"Unknown post-treatment period {period!r}.")
        c, null = parse_contrast(C, d, self.inputs.unit_labels)
        return quadratic_test(
            c,
            null,
            self.effects_panel.loc[period].to_numpy(dtype=float),
            self.placebo_effects.to_numpy(dtype=float).T,
            weight=weight,
            level=self.ci_level,
            quantile_method=quantile_method or self.quantile_method,
        )
