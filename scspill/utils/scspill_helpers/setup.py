"""Long-format panel ingestion shared by every spillover model.

:func:`prepare_panel_inputs` turns the user's long DataFrame into a
model-agnostic :class:`~scspill.utils.scspill_helpers.containers.PanelInputs`:
the treated/control outcome split, ``T0``, and (optionally) a time-varying
covariate cube. Each model then adds what it needs -- the ``sar`` model its
normalized spatial weights (:func:`~.sar.setup.prepare_sar_inputs`), the
``cd`` model its exposure matrix (:func:`~.cd.setup.prepare_cd_inputs`).

Unit order is the sorted treated labels, then the sorted control labels,
matching the R package's ``scspill_prep_X``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ...exceptions import ScspillDataError
from ..spatial import normalize_w, row_normalize
from .containers import PanelInputs
from .sar.setup import prepare_sar_inputs
from .sar.structures import SARInputs

__all__ = [
    "normalize_w",
    "prepare_panel_inputs",
    "prepare_scspill_inputs",
    "row_normalize",
]


def prepare_panel_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: Sequence[str] | None = None,
    *,
    allow_multiple_treated: bool = False,
    min_pre: int = 2,
    min_controls: int = 3,
    require_persistent: bool = False,
) -> PanelInputs:
    """Build :class:`PanelInputs` from a long panel.

    The treated units are those whose 0/1 ``treat`` indicator is ever 1; the
    first period at which they switch on defines ``T0``, which must be common
    to all of them. Covariate columns, if given, are assembled into a
    ``(T, N, K)`` cube over the control units -- the full period range is
    kept, and the models slice the pre-treatment rows.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel, one row per unit-period.
    outcome, treat, unitid, time : str
        Column names in ``df``.
    covariates : sequence of str, optional
        Covariate column names in ``df``.
    allow_multiple_treated : bool, default False
        Accept several treated units (they must share a start date).
    min_pre : int, default 2
        Minimum number of pre-treatment periods.
    min_controls : int, default 3
        Minimum number of control units.
    require_persistent : bool, default False
        Require treatment to stay on in every period after it starts.

    Returns
    -------
    PanelInputs
        Frozen estimation inputs.

    Raises
    ------
    ScspillDataError
        On missing columns, NaNs, duplicate unit-period rows, a non-binary
        treatment indicator, the wrong number of treated units, staggered
        adoption, too few controls, an unbalanced panel, or too few pre- or
        post-treatment periods.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise ScspillDataError(f"SCSPILL: required column {col!r} missing.")
    if df[outcome].isna().any():
        raise ScspillDataError("SCSPILL: outcome column contains NaN.")
    if df[treat].isna().any():
        # NaN != 0 evaluates True, which would silently shift the inferred T0.
        raise ScspillDataError("SCSPILL: treatment indicator column contains NaN.")
    if df.duplicated([unitid, time]).any():
        raise ScspillDataError("SCSPILL: panel has duplicate unit-period rows.")
    if not df[treat].isin([0, 1]).all():
        raise ScspillDataError("SCSPILL: treatment indicator must be 0/1.")

    time_labels = np.array(sorted(df[time].unique()))
    T = int(time_labels.size)
    units = sorted(df[unitid].unique())

    treated_mask = df.groupby(unitid)[treat].max()
    treated_units = [u for u in units if treated_mask.get(u, 0) == 1]
    if not treated_units or (not allow_multiple_treated and len(treated_units) != 1):
        raise ScspillDataError(
            f"SCSPILL: needs exactly one treated unit, found {len(treated_units)}."
        )
    controls = [u for u in units if u not in treated_units]
    if len(controls) < min_controls:
        raise ScspillDataError(f"SCSPILL: needs at least {min_controls} control units.")

    ywide = df.pivot(index=time, columns=unitid, values=outcome)
    if ywide.isna().any().any():
        raise ScspillDataError("SCSPILL: panel is unbalanced (missing cells).")
    ywide = ywide.loc[time_labels]

    # First treated period -> T0, common to every treated unit.
    twide = df.pivot(index=time, columns=unitid, values=treat).loc[time_labels]
    starts = {u: int(np.flatnonzero(twide[u].to_numpy() != 0)[0]) for u in treated_units}
    if len(set(starts.values())) > 1:
        raise ScspillDataError(
            "SCSPILL: treated units must share a common treatment start; got "
            + ", ".join(f"{u!r}: {time_labels[s]!r}" for u, s in starts.items())
            + "."
        )
    T0 = starts[treated_units[0]]
    if require_persistent and not (twide[treated_units].to_numpy()[T0:] == 1).all():
        raise ScspillDataError("SCSPILL: treatment must stay on once it starts.")
    if min_pre > T0:
        raise ScspillDataError(f"SCSPILL: needs >= {min_pre} pre-treatment periods.")
    if T - T0 < 1:
        raise ScspillDataError("SCSPILL: needs >= 1 post-treatment period.")

    X = None
    cov_names: tuple = ()
    if covariates:
        cov_names = tuple(covariates)
        cubes = []
        for c in cov_names:
            if c not in df.columns:
                raise ScspillDataError(f"SCSPILL: covariate {c!r} missing.")
            cw = df.pivot(index=time, columns=unitid, values=c).loc[time_labels, controls]
            if cw.isna().any().any():
                raise ScspillDataError(f"SCSPILL: covariate {c!r} has missing control cells.")
            cubes.append(cw.to_numpy(dtype=float))
        X = np.stack(cubes, axis=2)  # (T, N, K)

    return PanelInputs(
        Y0=ywide[treated_units[0]].to_numpy(dtype=float),
        Yc=ywide[controls].to_numpy(dtype=float),
        Yt=ywide[treated_units].to_numpy(dtype=float),
        T0=T0,
        X=X,
        treated_label=treated_units[0],
        treated_labels=tuple(treated_units),
        control_labels=tuple(controls),
        time_labels=time_labels,
        covariate_names=cov_names,
    )


def prepare_scspill_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    spatial_W,
    spatial_w,
    covariates: Sequence[str] | None = None,
) -> SARInputs:
    """Build the ``sar`` model's inputs from a long panel and spatial weights.

    :func:`prepare_panel_inputs` followed by
    :func:`~scspill.utils.scspill_helpers.sar.setup.prepare_sar_inputs`.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel, one row per unit-period.
    outcome, treat, unitid, time : str
        Column names in ``df``.
    spatial_W : DataFrame or array
        Control-to-control weights (see ``SCSPILLConfig.spatial_W``).
    spatial_w : Series, dict, DataFrame, or array
        Treated-to-control weights (see ``SCSPILLConfig.spatial_w``).
    covariates : sequence of str, optional
        Covariate column names in ``df``.

    Returns
    -------
    SARInputs
        Frozen, label-aligned estimation inputs.
    """
    panel = prepare_panel_inputs(df, outcome, treat, unitid, time, covariates=covariates)
    return prepare_sar_inputs(panel, spatial_W, spatial_w)
