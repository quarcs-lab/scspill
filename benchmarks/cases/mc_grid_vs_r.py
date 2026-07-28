"""Reduced Monte Carlo grid against the frozen R Tables 1-2 results.

Reruns the paper's N=16 / T0=20 simulation cells at rho in {-0.3, 0, 0.3}
with 200 replications (the R study used 1000) and compares SCSPILL's
per-period bias, RMSE, and 95% coverage against the frozen
``mc_study_N=16T0=20T1=10.csv``. The frozen SCSPILL rows at these cells are
bias {-0.0014, -0.0005, -0.0001}, RMSE {0.017, 0.008, 0.014}, coverage
{0.946, 0.958, 0.954}.
"""

from __future__ import annotations

from pathlib import Path

NAME = "mc_grid_vs_r"

RHOS = (-0.3, 0.0, 0.3)
SIMS_PER = 200

EXPECTED = {
    # SCSPILL is essentially unbiased at every cell.
    "scspill_abs_bias_max": (0.0, 0.02),
    # 95% pointwise coverage lands near nominal at every cell.
    "scspill_cover_min": (0.88, 1.0),
    "scspill_cover_max": (0.88, 0.995),
    # At |rho| = 0.3 the proposed method beats both comparators on RMSE.
    "rmse_ordering_holds": (1, 1),
    # RMSE within a factor of the frozen R values (MC noise at 200 reps).
    "rmse_ratio_vs_r_max": (0.0, 2.0),
}


def run() -> dict:
    """Run the reduced grid and score it against the frozen R results."""
    from scspill.simulate import load_r_mc_reference, mc_grid

    out = mc_grid(
        Ns=(16,),
        T0s=(20,),
        T1=10,
        rhos=RHOS,
        sims_per=SIMS_PER,
        K=1,
        beta=(1.0,),
        sigma2=0.1,
        m_iter=2000,
        burn=1000,
        step_rho=0.05,
        seed=20251030,
        progress=True,
    )
    ref = load_r_mc_reference(Path(__file__).resolve().parents[1] / "reference" / "mc_result")
    ref = ref[(ref["N"] == 16) & (ref["T0"] == 20) & (ref["rho"].isin(RHOS))]

    sp = out[out["method"] == "SCSPILL"].set_index("rho")
    sp_ref = ref[ref["method"] == "SCSPILL"].set_index("rho")

    ratios = []
    for rho in RHOS:
        got = sp.loc[rho, "rmse_point"]
        frozen = sp_ref.loc[rho, "rmse_point"]
        ratios.append(float(got / frozen))

    # RMSE ordering at |rho| = 0.3: SCSPILL < BSCM < SCM.
    ordering = True
    for rho in (-0.3, 0.3):
        cell = out[out["rho"] == rho].set_index("method")["rmse_point"]
        ordering = ordering and (cell["SCSPILL"] < cell["BSCM"] < cell["SCM"])

    return {
        "scspill_abs_bias_max": float(sp["bias_point"].abs().max()),
        "scspill_cover_min": float(sp["cover95_point"].min()),
        "scspill_cover_max": float(sp["cover95_point"].max()),
        "rmse_ordering_holds": int(ordering),
        "rmse_ratio_vs_r_max": float(max(ratios)),
        "scspill_rmse_by_rho": {r: round(float(sp.loc[r, "rmse_point"]), 4) for r in RHOS},
        "frozen_rmse_by_rho": {r: round(float(sp_ref.loc[r, "rmse_point"]), 4) for r in RHOS},
    }
