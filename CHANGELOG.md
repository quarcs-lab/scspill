# Changelog

## Unreleased

### New model: `cd` (Cao & Dowd)

Contributed by Hayato Tagawa, and restructured to the package's architecture.

- **`method="cd"` implements Cao and Dowd, "Estimation and Inference for
  Synthetic Control Methods with Spillover Effects" (arXiv:1902.07343v3).**
  It fits an intercept-adjusted simplex SCM for every unit and takes a known
  exposure structure: `affected_units` (one effect per exposed control), or a
  labeled `effect_matrix` with `spillover_structure="custom"`. It reports
  period-specific direct and spillover effects (`effects_panel`), pointwise
  intervals and P values (`effect_ci`, `pointwise_p_values`), Andrews-type P
  tests of any `C α_t = d` (`test_hypothesis`), and the `κ_A` specification
  test of the structure (`kappa_test`).
- **The defaults are the paper's Section 4 definitions.** Critical values are
  the left inverse of the empirical CDF, intervals invert the squared P test,
  and the references come from the full-sample fit. This is the same rule
  `sar` follows: paper-correct defaults, with the reference implementation's
  behaviour behind switches. The authors' Proposition 99 application code
  differs in two places. `quantile_method="linear"` and
  `ci_method="published"` reproduce it, and `reference_mode="leave_one_period_out"`
  refits every SCM equation without each pre-treatment period.
- **One configuration, one result type for every model.** `cd` is selected
  with `SCSPILLConfig(method="cd")`, and its fields carry a `(method='cd')`
  prefix as `sar`'s carry `(method='sar')`. Setting a field of the model you
  did not select is an error that names the field. A fit returns
  `SCSPILLResults` with the model's own fit in a typed slot (`result.sar` or
  `result.cd`). The shared surface — `att`, `counterfactual`, `gap`,
  `donor_weights`, `effects_detail`, `plot()` — works for both. Model-only
  accessors raise `AttributeError` on the other model.
- **Exposure structures from the paper.** `spillover_structure` takes:
  - `"per_unit"`: one effect per exposed control (Example 1);
  - `"homogeneous"`: one spillover shared by the `affected_units`
    (Example 2);
  - `"distance_decay"`: a spillover of `b·exp(-d_i)`, with distances in
    `unit_distances` (Example 3);
  - `"custom"`: a labeled `effect_matrix`.
- **Several treated units with a common start date** (v3 Section S.1.2).
  Each gets its own direct effect. The headline surface (`att`,
  `counterfactual`, plots) follows the first treated unit in label order;
  `att_by_unit` and `effects_panel` cover all of them.
- **`weighting="efficient"`** adds the lower-variance estimator of v3
  Section S.1.1, which weights the residual equations by the inverse of their
  estimated covariance. It is reported alongside the headline fit as
  `result.cd.efficient`. A warning flags when there are too few
  pre-treatment periods to estimate that covariance.
- **Pure-donor sensitivity** (v3 Section 5.2.2). `result.cd.sensitivity`
  holds the worst-case bias of the direct effect when `p` assumed-clean
  controls in fact receive spillovers, for the spillover-aware estimator and
  for the pure-donor synthetic control. The new `sensitivity` plot kind draws
  it, and the `weights` plot compares against the pure-donor weights.
- **Structure selection by `κ_A`** (v3 Sections 5.1.2 and S.1.3).
  `result.select_structure_by_kappa(candidates)` compares candidate
  structures on the same SCM fit. The default statistic is the paper's
  multi-period `κ_A`, the norm of the summed residual vectors over `√T1`.
  `statistic="mean_norm"` uses mlsynth's mean of the per-period norms
  instead.
- **Cross-checked against mlsynth.** `scspill/tests/test_cd_mlsynth_parity.py`
  compares every feature with mlsynth's `SPILLSYNTH(method="cd")` on frozen
  fixtures (`benchmarks/reference/mlsynth_cd/`, written by
  `tools/freeze_mlsynth_cd_reference.py`). SCM weights agree to the
  difference between the two solvers, and P values agree exactly.
- **Each treated unit's row of a custom `effect_matrix` must be its own
  direct-effect indicator**, so a spillover column cannot leak into the
  headline effect. `affected_units` may not list the treated unit.
- **Plots.** `panel`, `full`, `effect`, `spill_top` and `weights` work for
  `cd`. New `cd` kinds: `spillover` (every exposed unit's effect with its
  interval) and `kappa` (the structure test by period).
- **Data.** `load_california_cd(51)` and `load_california_cd(41)` bundle the
  authors' 50-states-plus-DC panel, treated from 1989, as a `SpillPanel`
  with the paper's exposed states. `scspill/tests/test_cd.py` checks both
  fits against the authors' frozen output
  (`jcao0/synthetic-control-spillover`, commit `86004020`), and their
  simulation code's statistics, stored in `benchmarks/reference/cao_dowd/`.

### Changed

- **Panel preparation is shared by every model**
  (`scspill_helpers.setup.prepare_panel_inputs`). It now also rejects, for
  `sar` too, duplicate unit-period rows and a treatment indicator that is not
  0/1.
- **The containers moved next to the code that fills them.**
  - `PanelInputs` and `SCSPILLEffects` are in `scspill_helpers.containers`.
  - The `sar` inputs, posterior blocks and fit are in
    `scspill_helpers.sar.structures` (`SARInputs`, `AlphaPosterior`,
    `SARPosterior`, `SARFit`).
  - The old import paths still work, and `SCSPILLInputs` and `SCSPILLFit`
    remain as aliases.
- **`SCSPILLEffects` is keyword-only.** Its `sar`-only fields (`att_draws`,
  `att_plugin`, `propagate_alpha`, `n_draws_used`) are optional, and
  `interval_kind` records how the bands were built.
- **`SpillPanel.spatial_w` and `spatial_W` may be `None`.**
  `SpillPanel.config_kwargs(method=None)` configures `sar` when the panel has
  spatial weights and `cd` otherwise.

### Authorship and citation

- **The software has three authors and one citation.** Shosei Sakaguchi and
  Hayato Tagawa have joined the project as co-authors, so `CITATION.cff`, the
  package metadata and `LICENSE` now list Carlos Mendez, Shosei Sakaguchi and
  Hayato Tagawa. This supersedes the v0.2.1 decision below, which made Carlos
  Mendez the sole author.
- **The "Citing" sections ask for a single reference**, the software itself:
  Mendez, C., Sakaguchi, S., & Tagawa, H. (2026). *Synthetic Control Models
  with Spillovers in Python*. The article and replication-package entries have
  been removed from `references` in `CITATION.cff`, which now describes only
  the software.
- **The methodological article is still cited where it is a reference** — the
  models table, the bibliographies under `docs/models/`, the estimator
  docstrings, and the `Paper (sar model)` project URL. `docs/models/sar.qmd`
  now says outright that `sar` implements Sakaguchi and Tagawa's method.

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
