# mlsynth cross-check fixtures for the `cd` model

[mlsynth](https://github.com/jgreathouse9/mlsynth)'s `SPILLSYNTH(method="cd")`
implements the same estimator as scspill's `cd` model: Cao and Dowd,
[arXiv:1902.07343v3](https://arxiv.org/abs/1902.07343v3). It uses a different
simplex solver, cvxpy with CLARABEL instead of SciPy's SLSQP, so the two
agree to solver tolerance rather than bit for bit.
`scspill/tests/test_cd_mlsynth_parity.py` compares them on every case
below. mlsynth's inference follows the authors' application code, so the
tests pin `quantile_method="linear"` and `ci_method="published"`.

`reference.json` was written by `tools/freeze_mlsynth_cd_reference.py`, run in a
separate environment with mlsynth installed. scspill does not depend on
mlsynth, so its test suite reads only this file. A live comparison runs as
well when mlsynth happens to be importable.

| Case | Panel | Structure |
|---|---|---|
| `ca51_per_unit` | Cao-Dowd Prop 99, 51 units | per unit, the paper's 13 exposed states |
| `ca41_per_unit` | Cao-Dowd Prop 99, 41 units | per unit, AZ, NV, OR |
| `ca41_homogeneous` | Cao-Dowd Prop 99, 41 units | one shared spillover on AZ, NV, OR |
| `ca41_distance_decay` | Cao-Dowd Prop 99, 41 units | `b * exp(-d)` with the fixed distances in `unit_distances` |
| `synth_two_treated` | synthetic, 10 units, `T0 = 40` | two treated units, one exposed |
| `synth_efficient` | the same | plus `weighting="efficient"` |

Selection reference: candidate structures on a synthetic panel whose three
exposed units share one spillover, compared with mlsynth's
`select_A_by_kappa` (the mean of the per-period `kappa_A` norms). scspill's
default statistic is v3 Section S.1.3's norm of the summed residual vectors.
mlsynth's mean of norms is available as `statistic="mean_norm"`, and the
test uses it.

Every array is keyed by unit label: mlsynth orders units treated, affected,
clean, while scspill orders them treated, then the controls sorted.

The provenance block of `reference.json` records the versions of mlsynth,
cvxpy, CLARABEL, NumPy and Python, the platform, and the SHA-256 of every
mlsynth source file the cases exercise (`spillsynth_helpers/setup.py` and
`spillsynth_helpers/cd/*.py`). The fixtures were frozen with mlsynth 1.0.0.
