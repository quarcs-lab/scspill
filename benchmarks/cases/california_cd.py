"""Cao-Dowd Proposition 99 against the authors' output and the paper (v3).

Two sources, two settings:

* **The authors' application code.** With its settings
  (``quantile_method="linear"``, ``ci_method="published"``), the 51- and
  41-unit fits must reproduce the frozen output of jcao0/
  synthetic-control-spillover at commit 86004020: the effects, the
  counterfactual, every P value, and the joint "no spillover" rejections.
* **The paper.** With the defaults (v3 Section 4), the fit must reproduce
  what v3 states in Section 6.2: California's 1989 and 1990 effects are not
  significant at 5%, and the break-even points of Figure 3's sensitivity
  analysis are 17.07 (one missed spillover) and 9.66 (two).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NAME = "california_cd"

REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "cao_dowd"
APPLICATION_CODE = {"quantile_method": "linear", "ci_method": "published"}

EXPECTED = {
    # The authors' application output (two-decimal agreement or better).
    "max_abs_effect_diff_51": (0.0, 0.03),
    "max_abs_counterfactual_diff_51": (0.0, 0.03),
    "p_value_mismatches_51": (0, 0),
    "spillover_test_mismatches_51": (0, 0),
    "max_abs_effect_diff_41": (0.0, 0.03),
    "p_value_mismatches_41": (0, 0),
    "rank_matches_k_51": (1, 1),
    # The paper (v3), with the default settings.
    "ca_1989_1990_not_significant": (1, 1),
    "fig3_breakeven_p1": (17.065, 17.075),
    "fig3_breakeven_p2": (9.655, 9.665),
}


def _against_output(spec: int) -> dict:
    from scspill import SCSPILL
    from scspill.data import load_california_cd

    expected = json.loads((REFERENCE / f"results_{spec}.json").read_text())
    fit = SCSPILL(
        {**load_california_cd(spec).config_kwargs(), **APPLICATION_CODE, "display_graphs": False}
    ).fit()
    effects = [fit.effects_panel["CA"].to_numpy() - np.asarray(expected["te"])]
    p_mismatch = int(np.sum(fit.pointwise_p_values["CA"].to_numpy() != expected["te_pvalue"]))
    for state in ("AZ", "NV", "OR"):
        effects.append(fit.effects_panel[state].to_numpy() - np.asarray(expected["u" + state][19:]))
        p_mismatch += int(np.sum(fit.pointwise_p_values[state].to_numpy() != expected["p" + state]))
    out = {
        f"max_abs_effect_diff_{spec}": float(np.max(np.abs(np.concatenate(effects)))),
        f"max_abs_counterfactual_diff_{spec}": float(
            np.max(np.abs(fit.counterfactual - np.asarray(expected["sp"])))
        ),
        f"p_value_mismatches_{spec}": p_mismatch,
        f"rank_matches_k_{spec}": int(fit.cd.rank == expected["k"]),
        f"condition_number_{spec}": round(fit.cd.condition_number, 3),
    }
    if spec == 51:
        no_spillover = np.eye(len(fit.unit_labels))[1:]
        rejects = [
            int(fit.test_hypothesis(no_spillover, period=t).reject) for t in fit.effects_panel.index
        ]
        out["spillover_test_mismatches_51"] = int(
            np.sum(np.asarray(rejects) != np.asarray(expected["spillover_test"]))
        )
    return out


def _against_paper() -> dict:
    from scspill import SCSPILL
    from scspill.data import load_california_cd

    fit = SCSPILL({**load_california_cd(51).config_kwargs(), "display_graphs": False}).fit()
    ci = fit.effect_ci["CA"]
    early = ci.loc[[1989, 1990]]
    sens = fit.cd.sensitivity
    largest = float(fit.effects_panel["CA"].max())
    return {
        "ca_1989_1990_not_significant": int(
            bool(((early["lower"] <= 0) & (early["upper"] >= 0)).all())
        ),
        "fig3_breakeven_p1": largest / sens.coefficients(1)[0],
        "fig3_breakeven_p2": largest / sens.coefficients(2)[0],
    }


def run() -> dict:
    """Fit both specifications under both settings."""
    return {**_against_output(51), **_against_output(41), **_against_paper()}
