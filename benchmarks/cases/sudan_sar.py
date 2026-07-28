"""Sudan secession cross-validation against the R replication run.

The R production fit (m=1e6, burn=5e5, step_rho=0.02) reports rho = 0.4274
with 95% CrI [0.2941, 0.5498], and Egypt and Kenya -- Sudan's largest trade
partners -- as the leading spillover recipients. As in the California case,
the R run's covariate-layout bug scrambled its six WDI covariates, so the
R-comparable Python specification drops them; the paper-correct fit (with
the covariates entering properly) is reported alongside.

This case runs at a tenth of the R budget, so rho is judged against the R
credible band, never the point estimate.
"""

from __future__ import annotations

NAME = "sudan_sar"

M_ITER = 100_000
BURN = 50_000
SEED = 20251022

EXPECTED = {
    # Port fidelity: rho under the R-equivalent (covariate-free) spec inside
    # the R 95% credible interval (R point estimate 0.4274).
    "rho_hat_r_spec": (0.2941, 0.5498),
    # Negative secession effect on GDP per capita (constant 2015 US$).
    "att": (-1000.0, -10.0),
    # Egypt and Kenya are the two leading spillover recipients.
    "egypt_kenya_top2": (1, 1),
    "acc_rho": (0.2, 0.7),
}


def run() -> dict:
    """Fit both specifications on the bundled Sudan panel."""
    from scspill import SCSPILL
    from scspill.data import load_sudan

    panel = load_sudan()
    base = {
        **panel.config_kwargs(),
        "p_factors": 1,
        "m_iter": M_ITER,
        "burn": BURN,
        "step_rho": 0.02,
        "seed": SEED,
        "display_graphs": False,
    }

    res_r = SCSPILL({**base, "covariates": None}).fit()
    res = SCSPILL(base).fit()

    spill_post = res.spillover_panel.iloc[res.inputs.T0 :]
    ranking = spill_post.abs().mean(axis=0).sort_values(ascending=False)
    top2 = set(ranking.index[:2])

    return {
        "rho_hat_r_spec": float(res_r.rho_hat),
        "rho_ci_r_spec": (float(res_r.rho_ci[0]), float(res_r.rho_ci[1])),
        "att": float(res.att),
        "egypt_kenya_top2": int(top2 == {"Egypt, Arab Rep.", "Kenya"}),
        "acc_rho": float(res.acc_rho),
        "rho_hat_paper_correct": float(res.rho_hat),
        "rho_ci_paper_correct": (float(res.rho_ci[0]), float(res.rho_ci[1])),
        "rho_ess": float(res.rho_ess),
        "att_percent_of_cf": float(
            100.0 * res.att / res.effects_detail.cf_mean[res.inputs.T0 :].mean()
        ),
        "top5_spillover": ", ".join(str(u) for u in ranking.index[:5]),
    }
