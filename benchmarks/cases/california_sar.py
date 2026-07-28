"""California Proposition 99 cross-validation against the R replication run.

The R production fit (m=100k, burn=50k, step_rho=0.01, seed 20251022)
reports rho = 0.1847 with 95% CrI [0.0957, 0.2677]. The R code, however,
passes its covariate array to the C++ sampler with a memory-layout mismatch
(t-fastest flattening read k-fastest), so `retprice` entered the R fit as
scrambled noise -- the R result is effectively the *no-covariate* model.

This case therefore validates two claims:

* **Port fidelity** -- fitted under the R-equivalent specification (no
  covariates, one latent factor), the Python rho posterior must land in the
  R credible band;
* **Paper-correct estimate** -- with `retprice` entering correctly (the
  covariate fix), the fit must still deliver the strongly-identified
  quantities: alpha[Nevada] inside its R credible interval, a negative
  paper-magnitude ATT, and Nevada as the dominant spillover recipient. The
  correctly-entered covariate shifts rho upward (reported, not gated: rho is
  weakly identified in this model and the R band is not the right target for
  a specification the R run never actually estimated).
"""

from __future__ import annotations

import numpy as np

NAME = "california_sar"

M_ITER = 50_000
BURN = 25_000
SEED = 20251022

EXPECTED = {
    # Port fidelity: rho under the R-equivalent (covariate-free) spec inside
    # the R 95% credible interval (R point estimate 0.1847).
    "rho_hat_r_spec": (0.0957, 0.2677),
    # Paper-correct fit: strongly-identified quantities.
    "att": (-25.0, -10.0),
    "nevada_is_top_spillover": (1, 1),
    "nevada_dominance_ratio": (3.0, np.inf),
    "alpha_nevada": (0.1204, 0.2681),
    "acc_rho": (0.2, 0.7),
}


def run() -> dict:
    """Fit both specifications on the bundled California panel."""
    from scspill import SCSPILL
    from scspill.data import load_california

    panel = load_california()
    base = {
        **panel.config_kwargs(),
        "p_factors": 1,
        "m_iter": M_ITER,
        "burn": BURN,
        "step_rho": 0.01,
        "seed": SEED,
        "display_graphs": False,
    }

    # R-equivalent specification: the R covariate-layout bug reduced retprice
    # to noise, so the comparable Python spec drops the covariate.
    res_r = SCSPILL({**base, "covariates": None}).fit()

    # Paper-correct specification: retprice enters correctly.
    res = SCSPILL(base).fit()

    spill_post = res.spillover_panel.iloc[res.inputs.T0 :]
    ranking = spill_post.abs().mean(axis=0).sort_values(ascending=False)
    labels = list(res.inputs.control_labels)

    return {
        "rho_hat_r_spec": float(res_r.rho_hat),
        "rho_ci_r_spec": (float(res_r.rho_ci[0]), float(res_r.rho_ci[1])),
        "att": float(res.att),
        "nevada_is_top_spillover": int(ranking.index[0] == "Nevada"),
        "nevada_dominance_ratio": float(ranking.iloc[0] / max(ranking.iloc[1], 1e-12)),
        "alpha_nevada": float(res.alpha_hat[labels.index("Nevada")]),
        "acc_rho": float(res.acc_rho),
        "rho_hat_paper_correct": float(res.rho_hat),
        "rho_ci_paper_correct": (float(res.rho_ci[0]), float(res.rho_ci[1])),
        "rho_ess": float(res.rho_ess),
        "att_scm": float(res.effects_detail.att_scm),
        "top5_spillover": ", ".join(str(u) for u in ranking.index[:5]),
    }
