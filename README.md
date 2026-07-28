<p align="center">
  <img src="https://raw.githubusercontent.com/quarcs-lab/scspill/main/docs/images/hero-v5.png" alt="scspill — synthetic control models with spillover effects and key result plots" width="85%">
</p>

# scspill

[![CI](https://github.com/quarcs-lab/scspill/actions/workflows/ci.yml/badge.svg)](https://github.com/quarcs-lab/scspill/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-quarcs--lab.github.io%2Fscspill-blue)](https://quarcs-lab.github.io/scspill/)
[![PyPI](https://img.shields.io/pypi/v/scspill.svg)](https://pypi.org/project/scspill/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/scspill/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quarcs-lab/scspill/blob/main/notebooks/california.ipynb)

Synthetic control when the treatment leaks. **scspill** is a Python
implementation of the Bayesian spatial-spillover synthetic control of
Sakaguchi & Tagawa (*Identification and Bayesian Inference for Synthetic
Control Methods with Spillover Effects*, The Econometrics Journal): it
relaxes SUTVA by letting the treatment spill over to the donor pool through
a spatial-autoregressive channel with user-supplied weights, and estimates
both the treatment effect on the treated unit and the spillover effect
received by every donor — with full Bayesian uncertainty.

The estimator follows the
[mlsynth](https://github.com/jgreathouse9/mlsynth) architecture (a pydantic
config in, a standardized results object out) so the two libraries compose
naturally; the documentation site follows
[geometrics](https://github.com/quarcs-lab/geometrics).

## Installation

```bash
pip install scspill              # NumPy/SciPy sampler backend
pip install "scspill[numba]"     # + JIT-compiled samplers (~10x faster)
pip install "scspill @ git+https://github.com/quarcs-lab/scspill.git"   # latest
```

Python 3.10+.

## At a glance

```python
from scspill import SCSPILL
from scspill.data import load_california

panel = load_california()        # Prop 99 panel + rook-contiguity weights
result = SCSPILL(
    {**panel.config_kwargs(), "m_iter": 20_000, "burn": 10_000, "seed": 42}
).fit()

result.att, result.att_ci          # treatment effect on California + 95% CrI
result.rho_hat, result.rho_ci      # spillover intensity posterior
result.spillover_panel["Nevada"]   # the effect received by Nevada, per year
result.diagnostics()               # ESS / R-hat / MCSE per chain
result.plot(kind="panel")          # counterfactual | effect | top spillovers
```

## What's inside

| Subpackage | What it does | Docs |
|---|---|---|
| `scspill` | `SCSPILL(config).fit()` — the two-step Bayesian sampler (horseshoe synthetic weights, SAR spillover block, adaptive Metropolis for the spillover intensity) and the identification formulas | [Get started](https://quarcs-lab.github.io/scspill/get-started.html) |
| `scspill.validation` | The Geweke (2004) joint distribution test of the sampler, prior-sensitivity grids, prior predictive checks | [Validation](https://quarcs-lab.github.io/scspill/articles/validation.html) |
| `scspill.simulate` | The paper's Monte Carlo engine: rook-lattice SAR DGP, SCM/BSCM/SCSPILL comparison, the Tables 1–2 grid | [Simulation study](https://quarcs-lab.github.io/scspill/articles/simulation-study.html) |
| `scspill.data` | The bundled California Prop 99 and Sudan secession case studies | [Datasets](https://quarcs-lab.github.io/scspill/articles/datasets.html) |

## Validated against the R replication package

The Python port is cross-validated against the authors' R replication
package (`python benchmarks/run_benchmarks.py --all --report`): California
and Sudan posteriors against the frozen R credible intervals, the Monte
Carlo grid against the paper's frozen Tables 1–2, prior predictive
statistics to three decimals, and the samplers against the Geweke joint
distribution test. The defaults are *paper-correct*: several documented bugs
of the reference implementation (a covariate memory-layout mismatch, a
missing horseshoe prior, alpha-frozen credible intervals, two incoherent
factor-block conditionals) are fixed here, each with an escape hatch or a
benchmark quantifying the difference — see the
[method article](https://quarcs-lab.github.io/scspill/articles/method.html).

## Documentation

Full documentation, executed tutorials, and the API reference live at
**<https://quarcs-lab.github.io/scspill/>**. Machine-readable entry points
for AI agents: [`llms.txt`](https://quarcs-lab.github.io/scspill/llms.txt)
and [`llms-full.txt`](https://quarcs-lab.github.io/scspill/llms-full.txt).

## Development

```bash
git clone https://github.com/quarcs-lab/scspill && cd scspill
uv sync --all-extras --group dev --group docs
make test      # pytest (fast tier; `make test-slow` for the long tier)
make lint      # ruff check + format
make typecheck # mypy
make docs      # quartodoc build -> quarto render -> llms.txt
```

## Citing

If you use scspill, please cite both the software and the methodological
article (machine-readable metadata lives in `CITATION.cff`):

> Mendez, C., Sakaguchi, S., & Tagawa, H. (2026). *Synthetic Control Models
> with Spillovers in Python* (version 0.1.1).
> <https://github.com/quarcs-lab/scspill>

> Sakaguchi, S., & Tagawa, H. (2026). Identification and Bayesian Inference
> for Synthetic Control Methods with Spillover Effects. *The Econometrics
> Journal*. <https://doi.org/10.1093/ectj/utag006>

## Acknowledgments

The method and the original R/C++ implementation are the work of Shosei
Sakaguchi and Hayato Tagawa, who are credited as co-authors of this library
on that basis; the Python implementation is by Carlos Mendez and any bug in
it is his. Their [replication
package](https://doi.org/10.5281/zenodo.19066186) is MIT-licensed, and every
release of scspill is cross-validated against its frozen results. The
estimator architecture follows Jared Greathouse's mlsynth; the documentation
stack follows the QuaRCS-lab geometrics package.

## License

MIT
