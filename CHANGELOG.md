# Changelog

## v0.1.0 (2026-07-28)

Initial release: a Python implementation of the Bayesian spatial-spillover
synthetic control of Sakaguchi & Tagawa, architecturally aligned with
mlsynth.

- `SCSPILL(config).fit() -> SCSPILLResults`: two-step sampler (horseshoe
  synthetic weights via Makalic–Schmidt Gibbs; SAR block with AR(1) latent
  factors, horseshoe or ridge covariate priors, and adaptive random-walk
  Metropolis for the spillover intensity), effects via the identification
  formulas with an eigendecomposition + Sherman–Morrison fast path, credible
  bands, a time-by-donor spillover panel, MCMC diagnostics, and R-parity
  plot kinds.
- Paper-correct defaults with R-compatibility escape hatches: proper
  covariate arrays (the R replication package's fits received scrambled
  covariates), horseshoe on the covariate coefficients (`beta_prior`),
  paired posterior draws in the effect intervals (`propagate_alpha`), and
  burn-in step adaptation for the Metropolis step (`adapt_rho`).
- A coherent latent-factor block: the Geweke joint distribution test caught
  two mutually inconsistent conditionals in the reference implementation
  (the omega parametrization and the FFBS initialization); both follow the
  paper's parametrization here, and every isolated sampler block passes the
  test.
- `scspill.validation`: the Geweke joint distribution test (simplified and
  production kernels), prior-sensitivity grids, and prior predictive checks
  whose nine statistics are pinned to the R package's frozen California
  table to three decimals.
- `scspill.simulate`: the paper's Monte Carlo engine (rook-lattice SAR DGP,
  SCM/BSCM/SCSPILL comparison, the Tables 1-2 grid) with a loader for the
  frozen R results.
- `scspill.data`: the bundled California Proposition 99 and Sudan secession
  case studies with label-aligned spatial weights.
- Optional numba backend compiling the same kernel source (~10x faster
  sampling); single-threaded BLAS inside the samplers for reproducibility.
- Cross-validation benchmarks against the frozen R results
  (`benchmarks/run_benchmarks.py`).
