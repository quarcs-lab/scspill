"""Exposure structures for the ``cd`` model.

Cao & Dowd (arXiv:1902.07343v3, Section 2.2) take the spillover structure as
known: an exposure matrix ``A`` maps a small vector of effect parameters to
one effect per unit, ``alpha_t = A gamma_t``. This module builds ``A`` --
per unit (Example 1), homogeneous (Example 2), decaying with distance
(Example 3), or user-supplied -- and attaches it to a prepared panel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

import numpy as np
import pandas as pd

from ....exceptions import ScspillConfigError
from ..containers import PanelInputs
from .structures import CDInputs


def validate_effect_matrix(
    effect_matrix: pd.DataFrame, units: Sequence[Any], treated: Sequence[Any]
) -> pd.DataFrame:
    """Check a labeled exposure matrix ``A`` and return it in ``units`` row order.

    Rows must be exactly the unit labels and columns unique and nonempty, with
    finite entries. Each treated unit's row must be its own direct-effect
    indicator: a single nonzero entry, equal to 1, in a column that is zero on
    every other row. Anything else would mix a spillover into the treated
    unit's effect, so the headline counterfactual would no longer be the
    direct effect's.

    Raises
    ------
    ScspillConfigError
        On any violation.
    """
    a = effect_matrix
    if a.index.has_duplicates or set(a.index) != set(units):
        raise ScspillConfigError("effect_matrix needs exactly one row per unit label.")
    if a.columns.has_duplicates or a.shape[1] == 0:
        raise ScspillConfigError("effect_matrix needs unique, nonempty columns.")
    a = a.loc[list(units)].astype(float)
    values = a.to_numpy()
    if not np.isfinite(values).all():
        raise ScspillConfigError("effect_matrix must be finite.")
    rows = {label: k for k, label in enumerate(units)}
    for label in treated:
        row = values[rows[label]]
        support = np.flatnonzero(row != 0)
        if support.size != 1 or row[support[0]] != 1:
            raise ScspillConfigError(
                f"effect_matrix row {label!r} (a treated unit) must be a direct-effect "
                "indicator: a single 1 and zeros elsewhere."
            )
        if np.count_nonzero(values[:, support[0]]) != 1:
            raise ScspillConfigError(
                f"effect_matrix column {a.columns[support[0]]!r} is the direct effect of "
                f"{label!r}, so it must be zero for every other unit."
            )
    return a


#: Column of ``A`` holding the shared spillover parameter.
SPILLOVER_COLUMN = "spillover"


def _direct_columns(unit_labels: Sequence[Any], treated: Sequence[Any]) -> pd.DataFrame:
    """One direct-effect indicator column per treated unit."""
    a = pd.DataFrame(0.0, index=list(unit_labels), columns=list(treated))
    for label in treated:
        a.loc[label, label] = 1.0
    return a


def _shared_column_name(treated: Sequence[Any]) -> str:
    """``"spillover"``, unless a treated unit already uses that label."""
    name = SPILLOVER_COLUMN
    while name in set(treated):
        name += "_"
    return name


def build_A_per_unit(
    unit_labels: Sequence[Any], treated: Sequence[Any], affected: Sequence[Any]
) -> pd.DataFrame:
    """One free effect per treated unit and per affected control.

    Cao & Dowd v3 Section 2.2, Example 1 (limited range): column ``j`` is the
    indicator of the ``j``-th treated or affected unit, so ``gamma_t`` holds
    each one's own effect and every other control is a pure donor.
    """
    a = _direct_columns(unit_labels, treated)
    for label in affected:
        a[label] = 0.0
        a.loc[label, label] = 1.0
    return a


def build_A_homogeneous(
    unit_labels: Sequence[Any], treated: Sequence[Any], affected: Sequence[Any]
) -> pd.DataFrame:
    """One free effect per treated unit and one spillover shared by the affected.

    Cao & Dowd v3 Section 3.4, Example 2 (homogeneous spillovers): the
    affected controls all receive the same spillover ``b``, held in the
    ``"spillover"`` column of ``gamma_t``.
    """
    if not affected:
        raise ScspillConfigError("A homogeneous structure needs at least one affected unit.")
    a = _direct_columns(unit_labels, treated)
    column = _shared_column_name(treated)
    a[column] = 0.0
    a.loc[list(affected), column] = 1.0
    return a


def decay_weights(control_labels: Sequence[Any], unit_distances: Mapping[Any, float]) -> pd.Series:
    """Decay weights ``exp(-d_i)`` per control; zero for controls left out.

    Raises
    ------
    ScspillConfigError
        On a negative or non-finite distance, or when every weight is zero.
    """
    w = pd.Series(0.0, index=list(control_labels))
    for label, distance in unit_distances.items():
        if label not in w.index:
            raise ScspillConfigError(f"unit_distances label {label!r} is not a control unit.")
        if not np.isfinite(distance) or distance < 0:
            raise ScspillConfigError(
                f"unit_distances[{label!r}] must be finite and non-negative; got {distance!r}."
            )
        w[label] = float(np.exp(-distance))
    if not (w > 0).any():
        raise ScspillConfigError("unit_distances gives every control a zero decay weight.")
    return w


def build_A_distance_decay(
    unit_labels: Sequence[Any],
    treated: Sequence[Any],
    control_labels: Sequence[Any],
    unit_distances: Mapping[Any, float],
) -> pd.DataFrame:
    """One free effect per treated unit and a spillover decaying with distance.

    Cao & Dowd v3 Section 7.1, Example 3 (exponential decay): the spillover
    on control ``i`` is ``b * exp(-d_i)``; the ``"spillover"`` column of
    ``gamma_t`` holds ``b``, the spillover at distance zero.
    """
    a = _direct_columns(unit_labels, treated)
    column = _shared_column_name(treated)
    a[column] = 0.0
    weights = decay_weights(control_labels, unit_distances)
    a.loc[list(control_labels), column] = weights.to_numpy()
    return a


def build_exposure_matrix(
    panel: PanelInputs,
    *,
    spillover_structure: str,
    affected_units: Sequence[Any] | None = None,
    unit_distances: Mapping[Any, float] | None = None,
    effect_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build ``A`` for a panel from a structure specification.

    The rows follow ``panel.unit_labels``; the first columns are the treated
    units' direct effects.

    Raises
    ------
    ScspillConfigError
        On an incomplete or invalid specification.
    """
    units = panel.unit_labels
    treated = panel.treated_labels
    affected = list(affected_units or [])
    clash = set(affected) & set(treated)
    if clash:
        raise ScspillConfigError(f"affected_units must not include a treated unit: {clash}.")
    unknown = [u for u in affected if u not in panel.control_labels]
    if unknown:
        raise ScspillConfigError(f"affected_units are not control units: {unknown}.")
    if spillover_structure == "custom":
        if effect_matrix is None:
            raise ScspillConfigError("spillover_structure='custom' requires effect_matrix.")
        return validate_effect_matrix(effect_matrix, units, treated)
    if spillover_structure == "per_unit":
        return build_A_per_unit(units, treated, affected)
    if spillover_structure == "homogeneous":
        return build_A_homogeneous(units, treated, affected)
    if spillover_structure == "distance_decay":
        if not unit_distances:
            raise ScspillConfigError(
                "spillover_structure='distance_decay' requires unit_distances."
            )
        return build_A_distance_decay(units, treated, panel.control_labels, unit_distances)
    raise ScspillConfigError(f"Unknown spillover_structure {spillover_structure!r}.")


def prepare_cd_inputs(
    panel: PanelInputs,
    *,
    spillover_structure: str,
    affected_units: Sequence[Any] | None = None,
    unit_distances: Mapping[Any, float] | None = None,
    effect_matrix: pd.DataFrame | None = None,
) -> CDInputs:
    """Attach the exposure matrix to a prepared panel.

    Parameters
    ----------
    panel : PanelInputs
        Output of :func:`~scspill.utils.scspill_helpers.setup.prepare_panel_inputs`.
    spillover_structure : str
        ``SCSPILLConfig.spillover_structure``.
    affected_units : sequence, optional
        Controls that may receive a spillover (``"per_unit"``,
        ``"homogeneous"``).
    unit_distances : mapping, optional
        Distance of each exposed control (``"distance_decay"``).
    effect_matrix : pd.DataFrame, optional
        The labeled ``A`` (``"custom"``).

    Returns
    -------
    CDInputs

    Raises
    ------
    ScspillConfigError
        On an invalid structure.
    """
    a = build_exposure_matrix(
        panel,
        spillover_structure=spillover_structure,
        affected_units=affected_units,
        unit_distances=unit_distances,
        effect_matrix=effect_matrix,
    )
    exposed = a.loc[list(panel.control_labels)].ne(0).any(axis=1)
    base = {f.name: getattr(panel, f.name) for f in fields(PanelInputs)}
    return CDInputs(
        **base,
        A=a,
        spillover_structure=spillover_structure,
        affected_labels=tuple(exposed.index[exposed.to_numpy()]),
    )
