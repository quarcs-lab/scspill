"""Exposure structures for the ``cd`` model.

Cao & Dowd (arXiv:1902.07343v3, Section 2.2) take the spillover structure as
known: an exposure matrix ``A`` maps a small vector of effect parameters to
one effect per unit, ``alpha_t = A gamma_t``. This module builds ``A`` from
the configuration and attaches it to a prepared panel.
"""

from __future__ import annotations

from collections.abc import Sequence
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


def build_A_per_unit(
    unit_labels: Sequence[Any], treated: Sequence[Any], affected: Sequence[Any]
) -> pd.DataFrame:
    """One free effect per treated unit and per affected control.

    Cao & Dowd v3 Section 2.2, the per-unit structure: column ``j`` is the
    indicator of the ``j``-th treated or affected unit, so ``gamma_t`` holds
    each one's own effect and every other control is a pure donor.
    """
    selected = [*treated, *affected]
    a = pd.DataFrame(0.0, index=list(unit_labels), columns=selected)
    for label in selected:
        a.loc[label, label] = 1.0
    return a


def prepare_cd_inputs(
    panel: PanelInputs,
    *,
    spillover_structure: str,
    affected_units: Sequence[Any] | None = None,
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
        Controls that may receive a spillover (structures other than
        ``"custom"``).
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
    units = panel.unit_labels
    treated = panel.treated_labels
    if spillover_structure == "custom":
        if effect_matrix is None:
            raise ScspillConfigError("spillover_structure='custom' requires effect_matrix.")
        a = validate_effect_matrix(effect_matrix, units, treated)
    elif spillover_structure == "per_unit":
        a = build_A_per_unit(units, treated, list(affected_units or []))
    else:
        raise ScspillConfigError(f"Unknown spillover_structure {spillover_structure!r}.")
    exposed = a.loc[list(panel.control_labels)].ne(0).any(axis=1)
    base = {f.name: getattr(panel, f.name) for f in fields(PanelInputs)}
    return CDInputs(
        **base,
        A=a,
        spillover_structure=spillover_structure,
        affected_labels=tuple(exposed.index[exposed.to_numpy()]),
    )
