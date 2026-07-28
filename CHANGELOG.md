# Changelog

## v0.2.1 (2026-07-28)

Authorship metadata only; no functional change.

- **Carlos Mendez is the sole author of the software**, in `CITATION.cff` and
  in the package metadata. Shosei Sakaguchi and Hayato Tagawa authored the
  method and the R/C++ implementation the `sar` model ports, and they are
  credited where that belongs: as the authors of the article and of the
  replication package, both listed under `references` in `CITATION.cff`, and
  in the acknowledgements. Cite their article whenever you fit `sar`.
- `LICENSE` is unchanged and still carries their copyright notice. That is a
  condition of reusing their MIT-licensed code, and is independent of who
  authored this package.

## v0.2.0 (2026-07-28)

scspill is now presented as what it is becoming: a package of **synthetic
control models with spillover effects**, of which Sakaguchi & Tagawa's is the
first and, today, the only one. Estimation is unchanged — same samplers, same
numbers, same public API.

- **`SCSPILLConfig` gains `method`**, a `Literal["sar"]` defaulting to the only
  implemented model, so existing code is unaffected. Results carry the value
  as `SCSPILLResults.method`, and `method_details.method_name` is now
  `"SCSPILL/sar"`. Models added later become a new value plus a subpackage
  rather than a new class. The literal deliberately lists only what exists:
  the roadmap names are rejected, not silently accepted.
- **Internal layout separates the shared spillover layer from the model.** The
  `sar` model's samplers, kernels, identification formulas, inference assembly
  and pipeline moved to `scspill/utils/scspill_helpers/sar/`; the MCMC
  diagnostics and classical-SCM baseline moved up to `scspill/utils/`, and the
  spatial-weight primitives to a new `scspill/utils/spatial.py`. The shared
  root now mirrors mlsynth's equivalent level exactly. Public imports are
  unaffected; code importing private helper paths must follow the move.
- **Documentation gains a Models section** — a catalogue with a clearly
  labelled roadmap (Cao & Dowd, the inclusive SCM of Di Stefano & Mellace, and
  Grossi et al.'s partial-interference SCG, none implemented) and a page per
  model. The method article is now the `sar` model's page; its published URL
  still resolves.
- **Fixes a latent crash**: the default plot filename is derived from
  `method_name`, so a name containing `/` made `save=True` write into a
  directory that does not exist. The slug is now sanitized.

## v0.1.1 (2026-07-28)

Attribution and citation metadata only — no functional change to the
estimator, and results are bit-for-bit identical to v0.1.0.

- The software's citation title is now *Synthetic Control Models with
  Spillovers in Python*, and Shosei Sakaguchi and Hayato Tagawa are credited
  as co-authors alongside Carlos Mendez, in `CITATION.cff` and in the package
  metadata. They authored the method and the R/C++ implementation this
  package ports; the Python implementation is Carlos Mendez's.
- `LICENSE` now retains the original implementation's MIT copyright notice
  (Shosei Sakaguchi and Hayato Tagawa) alongside this package's.
- Citations throughout the README, documentation, and module docstrings now
  carry the article's year (2026) and DOI (`10.1093/ectj/utag006`), and
  `CITATION.cff` additionally references the replication package
  (`10.5281/zenodo.19066186`).
- The documentation site gained a "Citing" section, which it previously
  lacked entirely.

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
