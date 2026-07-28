"""Long-format panel ingestion for the SCSPILL estimator.

Turns the user's long DataFrame plus a spatial-weight specification into a
:class:`~scspill.utils.scspill_helpers.structures.SCSPILLInputs`: the
treated/control outcome split, a row-normalized control-to-control weight
matrix ``Wn``, a sum-normalized treated-to-control weight vector ``wn``, and
(optionally) a time-varying covariate cube. The spatial weights may be
supplied as labelled pandas objects (aligned by unit label) or as bare NumPy
arrays (assumed already in donor-label order).

Donor order is the sorted unit labels minus the treated unit, matching the R
package's ``scspill_prep_X``; all label alignment is enforced against that
order so a shuffled ``spatial_w`` index or ``spatial_W`` rows can never be
silently mismatched.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ...exceptions import ScspillDataError
from ..spatial import _align_W, _align_w, normalize_w, row_normalize
from .structures import SCSPILLInputs


def prepare_scspill_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    spatial_W,
    spatial_w,
    covariates: Sequence[str] | None = None,
) -> SCSPILLInputs:
    """Build :class:`SCSPILLInputs` from a long panel and a spatial-weight spec.

    The treated unit is the one whose ``treat`` indicator is ever 1; the first
    period at which it switches on defines ``T0``. ``spatial_W`` /
    ``spatial_w`` are aligned to the donor order (sorted unit labels minus the
    treated unit). Covariate columns, if given, are assembled into a
    ``(T, N, K)`` cube over the control units -- the full period range is
    kept, and the samplers slice the pre-treatment rows.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel, one row per unit-period.
    outcome, treat, unitid, time : str
        Column names in ``df``.
    spatial_W : DataFrame or array
        Control-to-control weights (see :class:`SCSPILLConfig.spatial_W`).
    spatial_w : Series, dict, DataFrame, or array
        Treated-to-control weights (see :class:`SCSPILLConfig.spatial_w`).
    covariates : sequence of str, optional
        Covariate column names in ``df``.

    Returns
    -------
    SCSPILLInputs
        Frozen, label-aligned estimation inputs.

    Raises
    ------
    ScspillDataError
        On missing columns, NaNs, an unbalanced panel, zero or multiple
        treated units, too few pre-periods or donors, or misaligned spatial
        weights.
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise ScspillDataError(f"SCSPILL: required column {col!r} missing.")
    if df[outcome].isna().any():
        raise ScspillDataError("SCSPILL: outcome column contains NaN.")
    if df[treat].isna().any():
        # NaN != 0 evaluates True, which would silently shift the inferred T0.
        raise ScspillDataError("SCSPILL: treatment indicator column contains NaN.")

    time_labels = np.array(sorted(df[time].unique()))
    T = int(time_labels.size)
    units = sorted(df[unitid].unique())

    treated_mask = df.groupby(unitid)[treat].max()
    treated_units = [u for u in units if treated_mask.get(u, 0) == 1]
    if len(treated_units) != 1:
        raise ScspillDataError(
            f"SCSPILL: needs exactly one treated unit, found {len(treated_units)}."
        )
    treated = treated_units[0]
    controls = [u for u in units if u != treated]
    N = len(controls)
    if N < 3:
        raise ScspillDataError("SCSPILL: needs at least 3 control units.")

    ywide = df.pivot(index=time, columns=unitid, values=outcome)
    if ywide.isna().any().any():
        raise ScspillDataError("SCSPILL: panel is unbalanced (missing cells).")
    ywide = ywide.loc[time_labels]
    Y0 = ywide[treated].to_numpy(dtype=float)
    Yc = ywide[controls].to_numpy(dtype=float)

    # First treated period -> T0.
    twide = df.pivot(index=time, columns=unitid, values=treat).loc[time_labels, treated]
    on = np.where(twide.to_numpy() != 0)[0]
    if on.size == 0:
        raise ScspillDataError("SCSPILL: treated unit never switches on.")
    T0 = int(on[0])
    if T0 < 2:
        raise ScspillDataError("SCSPILL: needs >= 2 pre-treatment periods.")
    if T - T0 < 1:
        raise ScspillDataError("SCSPILL: needs >= 1 post-treatment period.")

    W_raw = _align_W(spatial_W, controls)
    w_raw = _align_w(spatial_w, controls)
    Wn = row_normalize(W_raw)
    wn = normalize_w(w_raw)

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

    return SCSPILLInputs(
        Y0=Y0,
        Yc=Yc,
        Wn=Wn,
        wn=wn,
        W_raw=W_raw,
        w_raw=w_raw,
        T0=T0,
        X=X,
        treated_label=treated,
        control_labels=tuple(controls),
        time_labels=time_labels,
        covariate_names=cov_names,
    )
