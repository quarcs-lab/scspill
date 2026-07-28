"""Long-run Geweke joint distribution test of the Step-2 sampler.

Runs the simplified (appendix) kernel with every block active (covariates
and one latent factor) at a quarter million draws with a well-mixed
Metropolis step and a batch size large enough for the slow global
statistics. The R replication's frozen full-scale run (two million draws)
reports max |Z| = 1.579; at this reduced budget residual long-memory noise
allows a slightly looser ceiling.
"""

from __future__ import annotations

import numpy as np

NAME = "geweke_full"

EXPECTED = {
    "max_abs_z": (0.0, 3.5),
    "n_flagged": (0, 1),
}


def run() -> dict:
    """Run the JDT on the full simplified model at long scale."""
    from scspill.validation import geweke_test

    rep = geweke_test(
        kernel="simple",
        T0=8,
        N=6,
        K=1,
        p=1,
        m_iid=250_000,
        m_mcmc=250_000,
        burn=30_000,
        a0=3.0,
        b0=1.0,
        step_rho=0.5,
        batch_size=8_000,
        seed=20251031,
    )
    return {
        "max_abs_z": float(np.abs(rep.table["z"]).max()),
        "n_flagged": int(rep.n_flagged),
        "passed": bool(rep.passed),
        "z_by_stat": {g: round(float(z), 2) for g, z in zip(rep.table["g"], rep.table["z"], strict=True)},
    }
