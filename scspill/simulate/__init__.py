"""Monte Carlo engine for the ``sar`` model (its article's Section 5 design).

Simulate spillover panels from a known SAR data-generating process (rook
lattice weights, planted synthetic weights and spillover intensity), fit the
classical simplex SCM, the Bayesian horseshoe SCM, and SCSPILL on each
replication, and summarize bias / RMSE / credible-interval coverage --
reproducing the paper's Tables 1-2 workflow.

Public API::

    from scspill.simulate import (
        rook_W, make_w, scspill_sim_dgp, run_one_sim, run_many_sim,
        summarize_many, mc_grid, load_r_mc_reference,
    )
"""

from .dgp import SimDGP, SimTruth, make_w, rook_W, scspill_sim_dgp
from .runner import SimRunResult, run_many_sim, run_one_sim
from .summary import load_r_mc_reference, mc_grid, summarize_many

__all__ = [
    "SimDGP",
    "SimRunResult",
    "SimTruth",
    "load_r_mc_reference",
    "make_w",
    "mc_grid",
    "rook_W",
    "run_many_sim",
    "run_one_sim",
    "scspill_sim_dgp",
    "summarize_many",
]
