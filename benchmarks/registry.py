"""Benchmark case registry for scspill.

Each case is a module in ``benchmarks/cases/`` exposing:

* ``NAME`` -- the registry key;
* ``run() -> dict`` -- computes the metrics (may take minutes);
* ``EXPECTED: dict[str, tuple[float, float]]`` -- inclusive ``(lo, hi)``
  acceptance interval per metric (booleans use ``(1, 1)``).

The cases validate the Python port against the R replication package's
frozen results (Path: cross-validation against an authoritative reference
implementation). Run them with ``python benchmarks/run_benchmarks.py --all``.
"""

CASES = {
    # California Prop 99: rho within the R 95% CrI, Nevada the top spillover.
    "california_sar": "cases.california_sar",
    # Sudan secession: rho within the R 95% CrI, Egypt/Kenya top spillovers.
    "sudan_sar": "cases.sudan_sar",
    # Geweke joint distribution test of the Step-2 sampler at long scale.
    "geweke_full": "cases.geweke_full",
    # Reduced Monte Carlo grid vs the frozen R Tables 1-2 results.
    "mc_grid_vs_r": "cases.mc_grid_vs_r",
    # California prior predictive check vs the frozen R p-value table.
    "prior_checks_ca": "cases.prior_checks_ca",
}
