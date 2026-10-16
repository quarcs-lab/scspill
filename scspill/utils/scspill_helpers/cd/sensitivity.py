"""Sensitivity of the ``cd`` estimate to spillovers the structure missed.

Cao & Dowd (arXiv:1902.07343v3, Section 5.2.2): if ``p`` of the controls
assumed clean in fact receive spillovers of size at most ``alpha_bar``, the
asymptotic bias of the direct effect is linear in those missed spillovers,

* for the spillover-aware estimator, with weights ``w_SP``: the treated
  unit's row of ``G (I - B) - I`` on the clean controls (Section 5.1.1,
  equation 10);
* for the pure-donor synthetic control, which drops every exposed unit from
  the donor pool, with weights ``-w_PD``: its donor weights.

The worst case over which ``p`` clean controls are hit is
``+/- alpha_bar * c_p``, with ``c_p`` the sum of the ``p`` largest absolute
weights. A smaller ``c_p`` means a method is more robust to missed
spillovers; the paper's Figure 3 plots these bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estimation import EffectOperator
from .scm_core import fit_simplex_scm
from .structures import CDInputs, PureDonorSensitivity


def pure_donor_sensitivity(
    inputs: CDInputs,
    b: np.ndarray,
    op: EffectOperator,
    *,
    maxiter: int,
    ftol: float,
) -> PureDonorSensitivity | None:
    """Compute both estimators' bias weights; ``None`` without clean controls.

    Parameters
    ----------
    inputs : CDInputs
        The prepared panel and exposure matrix.
    b : np.ndarray
        The all-unit SCM weight matrix of the fit.
    op : EffectOperator
        The fit's effect maps.
    maxiter, ftol : int, float
        Solver settings for the pure-donor synthetic control.
    """
    clean = inputs.clean_labels
    if not clean:
        return None
    labels = list(inputs.unit_labels)
    columns = [labels.index(u) for u in clean]
    row = labels.index(inputs.treated_label)
    q = op.unit_map @ (np.eye(len(labels)) - b) - np.eye(len(labels))
    y_pre = inputs.Y_units[:, : inputs.T0]
    a_pd, w_pd = fit_simplex_scm(y_pre[row], y_pre[columns].T, maxiter=maxiter, ftol=ftol)
    return PureDonorSensitivity(
        treated_label=inputs.treated_label,
        clean_labels=tuple(clean),
        w_sp=pd.Series(q[row, columns], index=list(clean)),
        w_pd=pd.Series(w_pd, index=list(clean)),
        a_pd=a_pd,
    )
