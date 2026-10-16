"""Spatial-weight preparation for the ``sar`` model.

Aligns the user's spatial weights to the donor order of a prepared
:class:`~scspill.utils.scspill_helpers.containers.PanelInputs` (sorted unit
labels minus the treated unit) and normalizes them: ``W`` by rows, ``w`` to
sum to one. Label alignment is enforced, so a shuffled ``spatial_w`` index or
``spatial_W`` rows can never be silently mismatched.
"""

from __future__ import annotations

from dataclasses import fields

from ...spatial import _align_W, _align_w, normalize_w, row_normalize
from ..containers import PanelInputs
from .structures import SARInputs


def prepare_sar_inputs(panel: PanelInputs, spatial_W, spatial_w) -> SARInputs:
    """Attach the normalized spatial weights to a prepared panel.

    Parameters
    ----------
    panel : PanelInputs
        Output of :func:`~scspill.utils.scspill_helpers.setup.prepare_panel_inputs`.
    spatial_W : DataFrame or array
        Control-to-control weights (see ``SCSPILLConfig.spatial_W``).
    spatial_w : Series, dict, DataFrame, or array
        Treated-to-control weights (see ``SCSPILLConfig.spatial_w``).

    Returns
    -------
    SARInputs

    Raises
    ------
    ScspillDataError
        On misaligned or invalid spatial weights.
    """
    controls = list(panel.control_labels)
    W_raw = _align_W(spatial_W, controls)
    w_raw = _align_w(spatial_w, controls)
    base = {f.name: getattr(panel, f.name) for f in fields(PanelInputs)}
    return SARInputs(
        **base,
        Wn=row_normalize(W_raw),
        wn=normalize_w(w_raw),
        W_raw=W_raw,
        w_raw=w_raw,
    )
