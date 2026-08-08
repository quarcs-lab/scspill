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

Synthetic control when the treatment leaks. **scspill** is a Python package of
synthetic control models that drop SUTVA on the donor pool: the treatment is
allowed to reach the controls, and every model reports two estimands — the
effect on the treated unit, purged of the contamination, *and* the spillover
effect received by each donor, which classical synthetic control cannot
express at all.

### Models

| Model | `method` | Reference | Class | Docs |
|---|---|---|---|---|
| Bayesian spatial-autoregressive spillover SCM | `"sar"` | [Sakaguchi & Tagawa (2026)](https://doi.org/10.1093/ectj/utag006), *The Econometrics Journal* | `SCSPILL` | [`sar`](https://quarcs-lab.github.io/scspill/models/sar.html) |

One model ships today. `sar` routes spillovers through spatial weights you
supply, scaled by a single intensity ρ that is estimated rather than assumed,
fits unconstrained horseshoe-shrunk synthetic weights, and returns full
Bayesian uncertainty for both estimands. At ρ = 0 it collapses exactly to the
Bayesian horseshoe synthetic control.

Three further spillover-aware models — Cao & Dowd, the inclusive SCM of
Di Stefano & Mellace, and the partial-interference SCG of Grossi et al. — are
on the [roadmap](https://quarcs-lab.github.io/scspill/models/#planned). They
are **not implemented**, and `SCSPILLConfig` rejects their names rather than
falling back silently.

The estimator architecture follows
[mlsynth](https://github.com/jgreathouse9/mlsynth) (a pydantic config in, a
standardized results object out), and the `method` names match its
`SPILLSYNTH` dispatcher, so the two libraries compose naturally; the
documentation site follows
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

## Package layout

| Subpackage | What it does | Docs |
|---|---|---|
| `scspill` | The model layer. `SCSPILL(config).fit()` — today the `sar` model's two-step Bayesian sampler (horseshoe synthetic weights, SAR spillover block, adaptive Metropolis for the spillover intensity) and its identification formulas | [Get started](https://quarcs-lab.github.io/scspill/get-started.html) |
| `scspill.validation` | `sar`'s sampler validation: the Geweke (2004) joint distribution test, prior-sensitivity grids, prior predictive checks | [Validation](https://quarcs-lab.github.io/scspill/articles/validation.html) |
| `scspill.simulate` | `sar`'s Monte Carlo engine: rook-lattice SAR DGP, SCM/BSCM/SCSPILL comparison, the Tables 1–2 grid | [Simulation study](https://quarcs-lab.github.io/scspill/articles/simulation-study.html) |
| `scspill.data` | The bundled California Prop 99 and Sudan secession spillover panels — model-agnostic, usable by any model added later | [Datasets](https://quarcs-lab.github.io/scspill/articles/datasets.html) |

## `sar` is validated, not just implemented

The `sar` model is cross-validated against the authors' R replication
package (`python benchmarks/run_benchmarks.py --all --report`): California
and Sudan posteriors against the frozen R credible intervals, the Monte
Carlo grid against the paper's frozen Tables 1–2, prior predictive
statistics to three decimals, and the samplers against the Geweke joint
distribution test. The defaults are *paper-correct*: several documented bugs
of the reference implementation (a covariate memory-layout mismatch, a
missing horseshoe prior, alpha-frozen credible intervals, two incoherent
factor-block conditionals) are fixed here, each with an escape hatch or a
benchmark quantifying the difference — see the
[`sar` model page](https://quarcs-lab.github.io/scspill/models/sar.html).

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

If you use scspill, please cite it as (machine-readable metadata lives in
`CITATION.cff`):

> Mendez, C., Sakaguchi, S., & Tagawa, H. (2026). *Synthetic Control Models
> with Spillovers in Python* (version 0.2.1).
> <https://github.com/quarcs-lab/scspill>

Authors: [Carlos Mendez](https://carlos-mendez.org) ·
[Shosei Sakaguchi](https://sites.google.com/view/shoseisakaguchi/home) ·
[Hayato Tagawa](https://hayataga.github.io/HayaTaga/)

## Acknowledgments

The library's design, syntax, and infrastructure are based on Jared
Greathouse's [mlsynth](https://github.com/jgreathouse9/mlsynth), a larger and
more comprehensive package of synthetic control methods. The documentation
stack follows [geometrics](https://github.com/quarcs-lab/geometrics), from the
[QuaRCS lab](https://quarcs.netlify.app/), where scspill is developed. The
`sar` model is cross-validated, every release, against the MIT-licensed
[replication package](https://doi.org/10.5281/zenodo.19066186) for the article
it implements; that package's copyright notice is retained in `LICENSE`.

## License

MIT
