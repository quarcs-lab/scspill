"""California prior predictive check against the frozen R p-value table.

The R replication computes nine prior predictive p-values for the observed
pre-treatment donor panel under the simplified SAR prior (a0=3, b0=1, rho
flat on (-0.99, 0.99), no factors), with the synthetic weights fixed at the
posterior mean of the California fit (the R argument is *named*
``alpha_hat_scaled`` but receives the unscaled ``alpha_hat``) and the raw
`retprice` covariate cube entering the forward simulations with
``beta ~ N(0, 1)`` -- the covariate contribution is what puts the simulated
panels on the observed outcome scale. The observed statistics are
deterministic (pinned to 3 decimals in the test suite); the p-values depend
mildly on the plug-in alpha, so they are accepted within a +-0.06 band of
the frozen values.
"""

from __future__ import annotations

import json
from pathlib import Path

NAME = "prior_checks_ca"

N_DRAWS = 10_000
PP_TOL = 0.06

_REF = json.loads((Path(__file__).resolve().parents[1] / "reference" / "values.json").read_text())[
    "california"
]

EXPECTED = {f"pdiff_{name}": (-PP_TOL, PP_TOL) for name in _REF["ppa_pvalues"]}
EXPECTED["observed_max_abs_diff"] = (0.0, 5e-3)


def run() -> dict:
    """Reproduce the R prior predictive table on the bundled California data."""
    import numpy as np
    from scspill import SCSPILL
    from scspill.data import load_california
    from scspill.validation import prior_predictive

    panel = load_california()
    # Plug-in alpha from a moderate California Step-1 fit (the R driver uses
    # its production fit's unscaled alpha_hat).
    fit = SCSPILL(
        {
            **panel.config_kwargs(),
            "p_factors": 0,
            "m_iter": 20_000,
            "burn": 10_000,
            "seed": 20251022,
            "display_graphs": False,
        }
    ).fit()
    inputs = fit.inputs
    Y0_pre = inputs.Y0[: inputs.T0]
    Yc_pre = inputs.Yc[: inputs.T0]

    res = prior_predictive(
        Y0_pre,
        inputs.W_raw,
        inputs.w_raw,
        fit.alpha_hat,
        Yc_obs=Yc_pre,
        X=inputs.X[: inputs.T0],  # retprice enters the forward simulations
        p=0,
        a0=3.0,
        b0=1.0,
        rho_support=(-0.99, 0.99),
        n_draws=N_DRAWS,
        seed=123,
    )

    out: dict = {}
    obs_diffs = []
    for name, frozen_p in _REF["ppa_pvalues"].items():
        out[f"pdiff_{name}"] = float(res.p_values[name] - frozen_p)
        obs_diffs.append(abs(res.observed[name] - _REF["ppa_observed"][name]))
    out["observed_max_abs_diff"] = float(np.max(obs_diffs))
    out["p_values"] = {k: round(float(v), 3) for k, v in res.p_values.items()}
    return out
