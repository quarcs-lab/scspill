"""scspill's ``cd`` against mlsynth's ``SPILLSYNTH(method="cd")``.

Reads the fixtures frozen by ``tools/freeze_mlsynth_cd_reference.py``
(``reference/mlsynth_cd/reference.json``, mlsynth 1.0.0 with cvxpy and
CLARABEL) and refits every case with scspill, under mlsynth's inference
settings (those of the authors' application code). The two use different
simplex solvers, so the SCM weights agree to solver tolerance; the P values
must agree exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NAME = "cd_vs_mlsynth"

REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "mlsynth_cd" / "reference.json"
SETTINGS = {"quantile_method": "linear", "ci_method": "published", "display_graphs": False}

EXPECTED = {
    "max_abs_B_diff": (0.0, 1e-4),
    "max_rel_effect_diff": (0.0, 1e-4),
    "p_value_mismatches": (0, 0),
    "max_rel_kappa_diff": (0.0, 1e-4),
    "max_abs_sensitivity_diff": (0.0, 1e-4),
    "max_rel_efficient_diff": (0.0, 1e-4),
    "selection_agrees": (1, 1),
}


def _config(case: dict, ref: dict) -> dict:
    from scspill.data import load_california_cd

    options = case["options"]
    name = case["name"]
    if name.startswith("ca"):
        base = {**load_california_cd(int(name[2:4])).config_kwargs(), "affected_units": None}
    else:
        base = {
            "method": "cd",
            "df": pd.DataFrame(ref["panels"]["synth_two_treated"]),
            "outcome": "y",
            "treat": "treated",
            "unitid": "unit",
            "time": "time",
        }
    config = {**base, **SETTINGS}
    config["spillover_structure"] = options.get("spillover_structure", "per_unit")
    for key in ("affected_units", "unit_distances"):
        if options.get(key) is not None:
            config[key] = options[key]
    if options.get("weighting", "identity") != "identity":
        config["weighting"] = options["weighting"]
    return config


def run() -> dict:
    """Refit every frozen case and record the largest disagreements."""
    from scspill import SCSPILL

    ref = json.loads(REFERENCE.read_text())
    b_diff = effect_diff = kappa_diff = sens_diff = eff_diff = 0.0
    p_mismatch = 0
    for name, case in ref["cases"].items():
        fit = SCSPILL(_config({**case, "name": name}, ref)).fit()
        units = case["units"]
        scale = float(np.abs(fit.inputs.Y0).max())
        b = fit.scm_matrix.loc[units, units].to_numpy()
        b_diff = max(
            b_diff, float(np.max(np.abs(b - [[case["B"][u][v] for v in units] for u in units])))
        )
        for unit in units:
            diff = np.abs(fit.effects_panel[unit].to_numpy() - case["alpha"][unit]) / scale
            effect_diff = max(effect_diff, float(diff.max()))
        for unit, test in {**case["treatment_tests"], **case["spillover_tests"]}.items():
            p_mismatch += int(np.sum(fit.pointwise_p_values[unit].to_numpy() != test["p_value"]))
        stats = [fit.kappa_test[t].statistic for t in fit.effects_panel.index]
        kappa_diff = max(
            kappa_diff,
            float(np.max(np.abs(np.subtract(stats, case["kappa"]["statistic"]))) / scale),
        )
        if "sensitivity" in case:
            sens = fit.cd.sensitivity
            for mine, theirs in (
                (sens.w_sp, case["sensitivity"]["w_sp_sorted_abs"]),
                (sens.w_pd, case["sensitivity"]["w_pd_sorted_abs"]),
            ):
                sorted_abs = np.sort(np.abs(mine.to_numpy()))[::-1]
                sens_diff = max(sens_diff, float(np.max(np.abs(sorted_abs - theirs))))
        if "efficient" in case:
            for unit in units:
                diff = (
                    np.abs(
                        fit.cd.efficient.effects_panel[unit].to_numpy()
                        - case["efficient"]["alpha"][unit]
                    )
                    / scale
                )
                eff_diff = max(eff_diff, float(diff.max()))
    selection = ref["selection"]
    fit = SCSPILL(
        {
            "method": "cd",
            "df": pd.DataFrame(ref["panels"]["synth_homogeneous"]),
            "outcome": "y",
            "treat": "treated",
            "unitid": "unit",
            "time": "time",
            "affected_units": selection["candidates"]["per_unit_3"]["affected_units"],
            "display_graphs": False,
        }
    ).fit()
    choice = fit.select_structure_by_kappa(selection["candidates"], statistic="mean_norm")
    return {
        "max_abs_B_diff": b_diff,
        "max_rel_effect_diff": effect_diff,
        "p_value_mismatches": p_mismatch,
        "max_rel_kappa_diff": kappa_diff,
        "max_abs_sensitivity_diff": sens_diff,
        "max_rel_efficient_diff": eff_diff,
        "selection_agrees": int(choice.best == selection["best"]),
        "cases": len(ref["cases"]),
        "mlsynth_version": ref["provenance"]["mlsynth_version"],
    }
