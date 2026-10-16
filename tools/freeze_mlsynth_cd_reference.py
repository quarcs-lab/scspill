"""Freeze mlsynth's Cao-Dowd results as cross-check fixtures for scspill's ``cd``.

Run in an environment where mlsynth (with cvxpy and CLARABEL) is installed,
from the repository root; scspill itself is not imported::

    python tools/freeze_mlsynth_cd_reference.py path/to/mlsynth-main

It fits mlsynth's ``SPILLSYNTH(method="cd")`` building blocks on the bundled
Cao-Dowd Proposition 99 panel and on synthetic panels generated here (and
stored alongside the results), and writes
``benchmarks/reference/mlsynth_cd/reference.json``. Every array is keyed by
unit label, because mlsynth orders units treated / affected / clean while
scspill orders them treated / controls sorted.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scspill" / "data" / "files" / "cao_dowd"
OUT = ROOT / "benchmarks" / "reference" / "mlsynth_cd"


def _prop99(spec: int) -> tuple[pd.DataFrame, list[str]]:
    """The Cao-Dowd panel, as scspill's load_california_cd builds it."""
    df = pd.read_csv(DATA / "cigs_consumption.csv")
    flags = pd.read_csv(DATA / "state_list.csv", encoding="utf-8-sig")
    policy = set(flags.loc[flags["missing12"] == 1, "state"])
    neighbors = set(flags.loc[flags["neighbor"] == 1, "state"])
    if spec == 41:
        df = df.loc[~df["state"].isin(policy - neighbors)].reset_index(drop=True)
        affected = sorted(neighbors)
    else:
        affected = sorted(policy | neighbors)
    df["treated"] = ((df["state"] == "CA") & (df["year"] >= 1989)).astype(int)
    return df, affected


def _synthetic(
    *, n_units: int, t0: int, t1: int, treated: list[str], spill: dict[str, float], seed: int
) -> pd.DataFrame:
    """Two-factor panel with level shifts on treated and exposed units."""
    rng = np.random.default_rng(seed)
    labels = [f"U{i:02d}" for i in range(n_units)]
    t = t0 + t1
    factors = np.cumsum(rng.normal(size=(t, 2)), axis=0) * 0.3 + rng.normal(size=(t, 2))
    loadings = rng.uniform(0.2, 1.5, size=(n_units, 2))
    y = 10 + loadings @ factors.T + rng.normal(scale=0.5, size=(n_units, t))
    for label in treated:
        y[labels.index(label), t0:] += -3.0
    for label, size in spill.items():
        y[labels.index(label), t0:] += size
    rows = [
        {"unit": u, "time": s, "y": float(y[i, s]), "treated": int(u in treated and s >= t0)}
        for i, u in enumerate(labels)
        for s in range(t)
    ]
    return pd.DataFrame(rows)


def _labels(inputs) -> list:
    return [*inputs.treated_labels, *inputs.affected_labels, *inputs.clean_labels]


def _keyed(labels: list, values: np.ndarray) -> dict:
    return {
        str(k): np.asarray(v, dtype=float).tolist() for k, v in zip(labels, values, strict=True)
    }


def _test(res) -> dict:
    return {
        "statistic": np.asarray(res.P_post, dtype=float).tolist(),
        "reference": np.asarray(res.P_pre, dtype=float).tolist(),
        "p_value": np.asarray(res.p_value, dtype=float).tolist(),
        "cutoff_05": float(res.cutoff_05),
    }


def _run_case(name: str, df: pd.DataFrame, columns: dict, **options) -> dict:
    from mlsynth.utils.spillsynth_helpers.cd.pipeline import run_cd
    from mlsynth.utils.spillsynth_helpers.setup import prepare_spillsynth_inputs

    weighting = options.pop("weighting", "identity")
    inputs = prepare_spillsynth_inputs(df, **columns, **options)
    fit = run_cd(inputs, weighting=weighting)
    labels = _labels(inputs)
    case = {
        "options": {**options, "weighting": weighting},
        "units": [str(u) for u in labels],
        "a": _keyed(labels, fit.a),
        "B": {str(u): _keyed(labels, row) for u, row in zip(labels, fit.B, strict=True)},
        "alpha": _keyed(labels, fit.alpha),
        "gamma": np.asarray(fit.gamma, dtype=float).tolist(),
        "cond_AMA": float(fit.cond_AMA),
        "att_sp_by_unit": {str(k): float(v) for k, v in fit.atts_sp_by_unit.items()},
        "att_scm_by_unit": {str(k): float(v) for k, v in fit.atts_scm_by_unit.items()},
        "treatment_tests": {str(k): _test(v) for k, v in fit.treatment_tests.items()},
        "treatment_ci_95": {
            str(k): np.asarray(v, dtype=float).tolist() for k, v in fit.treatment_cis_95.items()
        },
        "spillover_tests": {str(k): _test(v) for k, v in fit.spillover_tests.items()},
        "joint_spillover_test": (
            None if fit.joint_spillover_test is None else _test(fit.joint_spillover_test)
        ),
        "kappa": {
            "statistic": np.asarray(fit.kappa_A_test.kappa_A, dtype=float).tolist(),
            "reference": np.asarray(fit.kappa_A_test.kappa_pre, dtype=float).tolist(),
            "p_value": np.asarray(fit.kappa_A_test.p_value, dtype=float).tolist(),
            "cutoff_05": float(fit.kappa_A_test.cutoff_05),
        },
    }
    sens = fit.pure_donor_sensitivity
    if sens is not None:
        case["sensitivity"] = {
            "w_sp_sorted_abs": np.asarray(sens.w_sp, dtype=float).tolist(),
            "w_pd_sorted_abs": np.asarray(sens.w_pd, dtype=float).tolist(),
            "a_pd": float(sens.a_pd),
        }
    if fit.efficient_fit is not None:
        eff = fit.efficient_fit
        case["efficient"] = {
            "alpha": _keyed(labels, eff["alpha_W"]),
            "gamma": np.asarray(eff["gamma_W"], dtype=float).tolist(),
            "att": float(eff["att_sp_W"]),
        }
    print(f"froze {name}: {len(labels)} units")
    return case


def _selection(df: pd.DataFrame, columns: dict, candidates: dict) -> dict:
    """mlsynth's select_A_by_kappa over candidate structures on one fit."""
    from mlsynth.utils.spillsynth_helpers.cd.inference import select_A_by_kappa
    from mlsynth.utils.spillsynth_helpers.cd.scm_core import fit_leave_one_out_sc
    from mlsynth.utils.spillsynth_helpers.setup import prepare_spillsynth_inputs

    # One unit order for every candidate: the base per-unit structure's.
    base = prepare_spillsynth_inputs(df, **columns, **next(iter(candidates.values())))
    labels = _labels(base)
    a, B = fit_leave_one_out_sc(base.Y_pre)
    matrices = []
    for spec in candidates.values():
        inputs = prepare_spillsynth_inputs(df, **columns, **spec)
        order = [_labels(inputs).index(u) for u in labels]
        matrices.append(inputs.A[order])
    best, means = select_A_by_kappa(
        Y_post=base.Y_post, Y_pre=base.Y_pre, a=a, B=B, candidates=matrices
    )
    names = list(candidates)
    return {
        "candidates": candidates,
        "mean_kappa": dict(zip(names, map(float, means), strict=True)),
        "best": names[best],
    }


def main(mlsynth_root: str) -> None:
    """Fit every case and write the reference JSON."""
    prop99 = {"outcome": "cigs", "treat": "treated", "unitid": "state", "time": "year"}
    synth = {"outcome": "y", "treat": "treated", "unitid": "unit", "time": "time"}
    ca51, aff51 = _prop99(51)
    ca41, aff41 = _prop99(41)
    two_treated = _synthetic(
        n_units=10, t0=40, t1=5, treated=["U00", "U01"], spill={"U02": 2.0}, seed=7
    )
    homogeneous = _synthetic(
        n_units=12,
        t0=40,
        t1=8,
        treated=["U00"],
        spill={"U01": 2.5, "U02": 2.5, "U03": 2.5},
        seed=11,
    )
    distances = {"AZ": 1.0, "NV": 0.5, "OR": 1.5, "UT": 2.0, "ID": 2.5}
    cases = {
        "ca51_per_unit": _run_case("ca51_per_unit", ca51, prop99, affected_units=aff51),
        "ca41_per_unit": _run_case("ca41_per_unit", ca41, prop99, affected_units=aff41),
        "ca41_homogeneous": _run_case(
            "ca41_homogeneous",
            ca41,
            prop99,
            affected_units=aff41,
            spillover_structure="homogeneous",
        ),
        "ca41_distance_decay": _run_case(
            "ca41_distance_decay",
            ca41,
            prop99,
            spillover_structure="distance_decay",
            unit_distances=distances,
        ),
        "synth_two_treated": _run_case(
            "synth_two_treated", two_treated, synth, affected_units=["U02"]
        ),
        "synth_efficient": _run_case(
            "synth_efficient",
            two_treated,
            synth,
            affected_units=["U02"],
            weighting="efficient",
        ),
    }
    selection = _selection(
        homogeneous,
        synth,
        {
            "per_unit_3": {"affected_units": ["U01", "U02", "U03"]},
            "homogeneous_3": {
                "affected_units": ["U01", "U02", "U03"],
                "spillover_structure": "homogeneous",
            },
            "per_unit_1": {"affected_units": ["U01"]},
            "homogeneous_5": {
                "affected_units": ["U01", "U02", "U03", "U04", "U05"],
                "spillover_structure": "homogeneous",
            },
        },
    )
    helpers = Path(mlsynth_root) / "mlsynth" / "utils" / "spillsynth_helpers"
    sources = sorted([helpers / "setup.py", *(helpers / "cd").glob("*.py")])
    provenance = {
        "mlsynth_version": version("mlsynth"),
        "cvxpy_version": version("cvxpy"),
        "clarabel_version": version("clarabel"),
        "numpy_version": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "source_sha256": {
            str(p.relative_to(helpers)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": provenance,
        "unit_distances": distances,
        "panels": {
            "synth_two_treated": two_treated.to_dict(orient="list"),
            "synth_homogeneous": homogeneous.to_dict(orient="list"),
        },
        "cases": cases,
        "selection": selection,
    }
    (OUT / "reference.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT / 'reference.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mlsynth-main")
