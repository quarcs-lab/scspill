"""Spatial-weight primitives shared by every spillover model.

Normalization and label alignment for the two weight objects a spillover
model needs: a control-to-control matrix ``W`` and a treated-to-control
exposure vector ``w``. Both may arrive as labelled pandas objects (aligned by
unit label) or as bare NumPy arrays (assumed already in donor-label order).

These are model-agnostic on purpose -- any model that lets a treatment leak
along a network consumes the same two objects, so they live above
:mod:`scspill.utils.scspill_helpers` rather than inside any one model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..exceptions import ScspillDataError


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
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ScspillDataError(
            "spatial_w must be non-negative and finite (spatial exposure weights)."
        )
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
        if spatial_W.index.has_duplicates or spatial_W.columns.has_duplicates:
            raise ScspillDataError("SCSPILL: spatial_W has duplicate row/column labels.")
        missing = [u for u in labels if u not in spatial_W.index or u not in spatial_W.columns]
        if missing:
            raise ScspillDataError(f"SCSPILL: spatial_W missing rows/cols for units {missing[:5]}.")
        W = spatial_W.loc[labels, labels].to_numpy(dtype=float)
    else:
        # Copy so freezing the inputs never write-protects the caller's array.
        W = np.array(spatial_W, dtype=float, copy=True)
        if W.shape != (N, N):
            raise ScspillDataError(
                f"SCSPILL: spatial_W has shape {W.shape}, expected ({N}, {N}). "
                "Pass a labelled DataFrame to align by unit, or an N x N array "
                "in donor-label order."
            )
    if np.any(W < 0) or not np.all(np.isfinite(W)):
        raise ScspillDataError(
            "SCSPILL: spatial_W must be non-negative and finite (spatial weights)."
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
        if spatial_w.index.has_duplicates:
            raise ScspillDataError("SCSPILL: spatial_w has duplicate labels.")
        missing = [u for u in labels if u not in spatial_w.index]
        if missing:
            raise ScspillDataError(f"SCSPILL: spatial_w missing units {missing[:5]}.")
        w = spatial_w.loc[labels].to_numpy(dtype=float)
    elif isinstance(spatial_w, dict):
        unknown = sorted(set(spatial_w) - set(labels))
        if unknown:
            raise ScspillDataError(
                f"SCSPILL: spatial_w has unknown donor keys {unknown[:5]} "
                "(typo, or a unit missing from the panel?)."
            )
        w = np.array([float(spatial_w.get(u, 0.0)) for u in labels], dtype=float)
    else:
        # Copy so freezing the inputs never write-protects the caller's array.
        w = np.array(spatial_w, dtype=float, copy=True).ravel()
        if w.shape[0] != N:
            raise ScspillDataError(f"SCSPILL: spatial_w has length {w.shape[0]}, expected {N}.")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ScspillDataError(
            "SCSPILL: spatial_w must be non-negative and finite (spatial exposure weights)."
        )
    if not np.any(w > 0):
        raise ScspillDataError(
            "SCSPILL: spatial_w has no positive entries (treated unit is linked to no controls)."
        )
    return w
