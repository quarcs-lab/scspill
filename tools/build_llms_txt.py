"""Build the machine-readable docs indexes for scspill.

Writes the curated, committed ``docs/llms.txt`` (llmstxt.org convention),
and -- when ``docs/_site`` exists -- copies it there plus a full-corpus
``docs/_site/llms-full.txt`` concatenating every docs page's source and the
signature/docstring of every public callable.

Usage::

    python tools/build_llms_txt.py                 # write docs/llms.txt (+ _site files if present)
    python tools/build_llms_txt.py --canonical-only  # only docs/llms.txt (CI drift check)
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BASE = "https://quarcs-lab.github.io/scspill"

PAGES = [
    ("Get started (California Prop 99)", "get-started.qmd", "get-started.html"),
    ("Sudan case study", "sudan.qmd", "sudan.html"),
    ("The method", "articles/method.qmd", "articles/method.html"),
    ("Validating the sampler", "articles/validation.qmd", "articles/validation.html"),
    (
        "The Monte Carlo simulation study",
        "articles/simulation-study.qmd",
        "articles/simulation-study.html",
    ),
    ("The bundled datasets", "articles/datasets.qmd", "articles/datasets.html"),
    ("For AI / LLMs", "use-with-llms.qmd", "use-with-llms.html"),
    ("Changelog", "changelog.qmd", "changelog.html"),
]

API_GROUPS = [
    ("Top level", "scspill", ["SCSPILL", "SCSPILLConfig", "SCSPILLResults"]),
    ("scspill.validation", "scspill.validation", None),
    ("scspill.simulate", "scspill.simulate", None),
    ("scspill.data", "scspill.data", None),
]


def canonical() -> str:
    """Build the curated llms.txt content."""
    lines = [
        "# scspill",
        "",
        "> Bayesian spatial-spillover synthetic control (Sakaguchi & Tagawa, The",
        "> Econometrics Journal) in Python: estimate a policy's effect on the treated",
        "> unit AND the spillover received by every donor, from a long panel plus",
        "> user-supplied spatial weights. Two-step MCMC (horseshoe synthetic weights,",
        "> SAR spillover block); pydantic config in, standardized results out",
        "> (mlsynth-style architecture).",
        "",
        'Install: `pip install scspill` (or `pip install "scspill[numba]"` for',
        "JIT-compiled samplers).",
        "",
        "Minimal contract: SCSPILL({df, outcome, treat, unitid, time, spatial_w,",
        "spatial_W, covariates?, m_iter, burn, seed}).fit() -> result with .att,",
        ".att_ci, .rho_hat, .rho_ci, .counterfactual, .spillover_panel,",
        ".diagnostics(), .plot(). The treated unit and date are inferred from the",
        "0/1 `treat` column; spatial weights align by donor label.",
        "",
        "## Docs",
        "",
    ]
    for title, _, html in PAGES:
        lines.append(f"- [{title}]({BASE}/{html})")
    lines += ["", f"- [API reference]({BASE}/reference/index.html)", "", "## API", ""]
    for title, module_name, names in API_GROUPS:
        module = importlib.import_module(module_name)
        if names is None:
            names = sorted(getattr(module, "__all__", []))
        lines.append(f"### {title}")
        lines.append("")
        lines.append(", ".join(f"`{n}`" for n in names))
        lines.append("")
    lines += [
        "## Full corpus",
        "",
        f"- [llms-full.txt]({BASE}/llms-full.txt) — every docs page's source plus",
        "  every public signature and docstring.",
        "",
    ]
    return "\n".join(lines)


def build_full() -> str:
    """Concatenate every docs page source plus API signatures/docstrings."""
    parts = [canonical(), "\n\n---\n\n# Full documentation corpus\n"]
    for title, qmd, _ in PAGES:
        source = (DOCS / qmd).read_text(encoding="utf-8")
        parts.append(f"\n\n---\n\n## PAGE: {title}\n\n{source}")
    parts.append("\n\n---\n\n## API signatures and docstrings\n")
    for title, module_name, names in API_GROUPS:
        module = importlib.import_module(module_name)
        if names is None:
            names = sorted(getattr(module, "__all__", []))
        parts.append(f"\n### {title}\n")
        for name in names:
            obj = getattr(module, name, None)
            if obj is None:
                continue
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = ""
            doc = inspect.getdoc(obj) or ""
            parts.append(f"\n#### `{name}{sig}`\n\n{doc}\n")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-only", action="store_true")
    args = parser.parse_args()

    text = canonical()
    (DOCS / "llms.txt").write_text(text + "\n", encoding="utf-8")
    print(f"wrote {DOCS / 'llms.txt'}")
    if args.canonical_only:
        return 0

    site = DOCS / "_site"
    if site.exists():
        (site / "llms.txt").write_text(text + "\n", encoding="utf-8")
        (site / "llms-full.txt").write_text(build_full() + "\n", encoding="utf-8")
        print(f"wrote {site / 'llms.txt'} and {site / 'llms-full.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
