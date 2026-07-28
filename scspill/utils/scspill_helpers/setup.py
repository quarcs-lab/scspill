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
from .structures import SCSPILLInputs


def row_normalize(W: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Zero the diagonal and scale each row of ``W`` to sum to one.

    Rows whose (post-diagonal-zeroing) sum is at most ``tol`` are kept
    all-zero rather than divided -- isolated units stay isolated, matching the
    R package's ``row_normalize(zero_policy = "keep")``.

    Parameters
    ----------
    W : np.ndarray
        Square spatial weight matrix, shape ``(N, N)``.
    tol : float, default 1e-12
        Row sums at or below this threshold are treated as zero.

    Returns
    -------
    np.ndarray
        The row-normalized matrix (a new array; the input is not modified).
    """
    W = np.array(W, dtype=float, copy=True)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ScspillDataError(f"row_normalize expects a square matrix, got shape {W.shape}.")
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1)
    rs[rs <= tol] = 1.0
    return W / rs[:, None]


def normalize_w(w: np.ndarray) -> np.ndarray:
    """Scale the treated-to-control exposure vector to sum to one.

    Parameters
    ----------
    w : np.ndarray
        Raw exposure weights, shape ``(N,)``.

    Returns
    -------
    np.ndarray
        ``w / w.sum()``.

    Raises
    ------
    ScspillDataError
        If ``w`` has no positive entry (the treated unit would be linked to
        no control).
    """
    w = np.asarray(w, dtype=float).ravel()
    if not np.any(w > 0):
        raise ScspillDataError(
            "spatial_w has no positive entries (treated unit is linked to no controls)."
        )
    return w / w.sum()


def _align_W(spatial_W, control_labels) -> np.ndarray:
    """Return an ``(N, N)`` control-to-control matrix in ``control_labels`` order."""
    labels = list(control_labels)
    N = len(labels)
    if isinstance(spatial_W, pd.DataFrame):
        cols = list(spatial_W.columns)
        # Drop a leading label column if present (e.g. an exported "state" column).
        if spatial_W.shape[1] == N + 1 and spatial_W.index.name is None:
            first = cols[0]
            spatial_W = spatial_W.set_index(first)
        missing = [u for u in labels if u not in spatial_W.index or u not in spatial_W.columns]
        if missing:
            raise ScspillDataError(f"SCSPILL: spatial_W missing rows/cols for units {missing[:5]}.")
        W = spatial_W.loc[labels, labels].to_numpy(dtype=float)
    else:
        W = np.asarray(spatial_W, dtype=float)
        if W.shape != (N, N):
            raise ScspillDataError(
                f"SCSPILL: spatial_W has shape {W.shape}, expected ({N}, {N}). "
                "Pass a labelled DataFrame to align by unit, or an N x N array "
                "in donor-label order."
            )
    return W


def _align_w(spatial_w, control_labels) -> np.ndarray:
    """Return an ``(N,)`` treated-to-control weight vector in label order."""
    labels = list(control_labels)
    N = len(labels)
    if isinstance(spatial_w, pd.DataFrame):
        # Use the last numeric column as the weight, indexed by the first column.
        if spatial_w.shape[1] >= 2:
            spatial_w = spatial_w.set_index(spatial_w.columns[0]).iloc[:, -1]
        else:
            spatial_w = spatial_w.iloc[:, 0]
    w: np.ndarray
    if isinstance(spatial_w, pd.Series):
        missing = [u for u in labels if u not in spatial_w.index]
        if missing:
            raise ScspillDataError(f"SCSPILL: spatial_w missing units {missing[:5]}.")
        w = spatial_w.loc[labels].to_numpy(dtype=float)
    elif isinstance(spatial_w, dict):
        w = np.array([float(spatial_w.get(u, 0.0)) for u in labels], dtype=float)
    else:
        w = np.asarray(spatial_w, dtype=float).ravel()
        if w.shape[0] != N:
            raise ScspillDataError(f"SCSPILL: spatial_w has length {w.shape[0]}, expected {N}.")
    if not np.any(w > 0):
        raise ScspillDataError(
            "SCSPILL: spatial_w has no positive entries (treated unit is linked to no controls)."
        )
    return w


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
