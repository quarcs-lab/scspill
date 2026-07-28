# scspill validation report

Cross-validation of the Python port against the frozen results of the
R replication package (see `reference/values.json` for provenance).

| Case | Status | Time |
| --- | --- | --- |
| california_sar | PASS | 46.2s |
| sudan_sar | PASS | 42.7s |
| geweke_full | PASS | 229.3s |
| mc_grid_vs_r | PASS | 55.4s |
| prior_checks_ca | PASS | 29.5s |

## california_sar

| Metric | Got | Accepted range | OK |
| --- | --- | --- | --- |
| rho_hat_r_spec | 0.1866 | [0.0957, 0.2677] | yes |
| att | -17.03 | [-25, -10] | yes |
| nevada_is_top_spillover | 1 | [1, 1] | yes |
| nevada_dominance_ratio | 10.1 | [3, inf] | yes |
| alpha_nevada | 0.2003 | [0.1204, 0.2681] | yes |
| acc_rho | 0.4346 | [0.2, 0.7] | yes |

Additional metrics:

- `rho_ci_r_spec`: (0.10218085314640935, 0.2619063109979668)
- `rho_hat_paper_correct`: 0.34618354508476856
- `rho_ci_paper_correct`: (0.2758056811120312, 0.45420641636247616)
- `rho_ess`: 13.444464412911055
- `att_scm`: -15.762839322881167
- `top5_spillover`: Nevada, Idaho, Utah, Wyoming, Montana

## sudan_sar

| Metric | Got | Accepted range | OK |
| --- | --- | --- | --- |
| rho_hat_r_spec | 0.3946 | [0.2941, 0.5498] | yes |
| att | -30.69 | [-1000, -10] | yes |
| egypt_kenya_top2 | 1 | [1, 1] | yes |
| acc_rho | 0.4547 | [0.2, 0.7] | yes |

Additional metrics:

- `rho_ci_r_spec`: (0.24201531358561107, 0.507523609928553)
- `rho_hat_paper_correct`: 0.4002117839213539
- `rho_ci_paper_correct`: (0.2613004725135062, 0.5564329416933556)
- `rho_ess`: 31.298999375371388
- `att_percent_of_cf`: -2.233202475350229
- `top5_spillover`: Egypt, Arab Rep., Kenya, Uganda, Algeria, Tunisia

## geweke_full

| Metric | Got | Accepted range | OK |
| --- | --- | --- | --- |
| max_abs_z | 2.501 | [0, 3.5] | yes |
| n_flagged | 0 | [0, 1] | yes |

Additional metrics:

- `passed`: True
- `z_by_stat`: {'rho': 0.17, 'log_sigma2': 0.65, 'yc_mean': 2.4, 'log_yc_var': -2.08, 'spatial_quadratic': 1.23, 'corr_y0_wyc': -2.37, 'beta_mean': -2.5, 'Eta_mean': -0.65, 'Gamma_mean': -0.28}

## mc_grid_vs_r

| Metric | Got | Accepted range | OK |
| --- | --- | --- | --- |
| scspill_abs_bias_max | 0.001213 | [0, 0.02] | yes |
| scspill_cover_min | 0.929 | [0.88, 1] | yes |
| scspill_cover_max | 0.9725 | [0.88, 0.995] | yes |
| rmse_ordering_holds | 1 | [1, 1] | yes |
| rmse_ratio_vs_r_max | 0.4603 | [0, 2] | yes |

Additional metrics:

- `scspill_rmse_by_rho`: {-0.3: 0.0064, 0.0: 0.0118, 0.3: 0.0162}
- `frozen_rmse_by_rho`: {-0.3: 0.0173, 0.0: 0.0257, 0.3: 0.0398}

## prior_checks_ca

| Metric | Got | Accepted range | OK |
| --- | --- | --- | --- |
| pdiff_yc_mean | 0.0077 | [-0.06, 0.06] | yes |
| pdiff_log_yc_var | 0.0185 | [-0.06, 0.06] | yes |
| pdiff_spatial_quadratic | 0.0082 | [-0.06, 0.06] | yes |
| pdiff_corr_y0_wyc | -0.0175 | [-0.06, 0.06] | yes |
| pdiff_ac1 | 0.0169 | [-0.06, 0.06] | yes |
| pdiff_ac2 | 0.0185 | [-0.06, 0.06] | yes |
| pdiff_pve_pc1 | -0.0139 | [-0.06, 0.06] | yes |
| pdiff_avg_skewness | 0.0039 | [-0.06, 0.06] | yes |
| pdiff_avg_kurtosis | 0.0177 | [-0.06, 0.06] | yes |
| observed_max_abs_diff | 0.000459 | [0, 0.005] | yes |

Additional metrics:

- `p_values`: {'yc_mean': 0.914, 'log_yc_var': 0.7, 'spatial_quadratic': 0.81, 'corr_y0_wyc': 0.614, 'ac1': 0.993, 'ac2': 0.999, 'pve_pc1': 0.035, 'avg_skewness': 0.484, 'avg_kurtosis': 0.733}
