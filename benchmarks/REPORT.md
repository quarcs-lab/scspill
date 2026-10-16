# scspill validation report

Cross-validation of the `sar` model against the frozen results of the
R replication package (see `reference/values.json` for provenance), and
of the `cd` model against the Cao-Dowd authors' frozen output, the paper,
and mlsynth (see `reference/cao_dowd/` and `reference/mlsynth_cd/`).

| Case | Status | Time |
| --- | --- | --- |
| california_sar | PASS | 258.2s |
| sudan_sar | PASS | 360.0s |
| geweke_full | PASS | 227.3s |
| mc_grid_vs_r | PASS | 386.9s |
| prior_checks_ca | PASS | 40.7s |
| california_cd | PASS | 1.5s |
| cd_vs_mlsynth | PASS | 1.4s |

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

- `rho_ci_r_spec`: (0.10218085314798951, 0.26190631099971656)
- `rho_hat_paper_correct`: 0.3461835450853728
- `rho_ci_paper_correct`: (0.2758056811124836, 0.4542064163633116)
- `rho_ess`: 13.444464412910865
- `att_scm`: -15.762839405338472
- `top5_spillover`: Nevada, Idaho, Utah, Wyoming, Montana

## sudan_sar

| Metric | Got | Accepted range | OK |
| --- | --- | --- | --- |
| rho_hat_r_spec | 0.3946 | [0.2941, 0.5498] | yes |
| att | -30.91 | [-1000, -10] | yes |
| egypt_kenya_top2 | 1 | [1, 1] | yes |
| acc_rho | 0.4547 | [0.2, 0.7] | yes |

Additional metrics:

- `rho_ci_r_spec`: (0.24201579017066752, 0.5075240655609687)
- `rho_hat_paper_correct`: 0.40021299192490767
- `rho_ci_paper_correct`: (0.2613017084438055, 0.5564341182901895)
- `rho_ess`: 31.298999375371185
- `att_percent_of_cf`: -2.2485831294060277
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

## california_cd

| Metric | Got | Accepted range | OK |
| --- | --- | --- | --- |
| max_abs_effect_diff_51 | 0.0084 | [0, 0.03] | yes |
| max_abs_counterfactual_diff_51 | 0.002502 | [0, 0.03] | yes |
| p_value_mismatches_51 | 0 | [0, 0] | yes |
| spillover_test_mismatches_51 | 0 | [0, 0] | yes |
| max_abs_effect_diff_41 | 0.004121 | [0, 0.03] | yes |
| p_value_mismatches_41 | 0 | [0, 0] | yes |
| rank_matches_k_51 | 1 | [1, 1] | yes |
| ca_1989_1990_not_significant | 1 | [1, 1] | yes |
| fig3_breakeven_p1 | 17.07 | [17.07, 17.07] | yes |
| fig3_breakeven_p2 | 9.664 | [9.655, 9.665] | yes |

Additional metrics:

- `condition_number_51`: 12.485
- `max_abs_counterfactual_diff_41`: 0.0007847614174920636
- `rank_matches_k_41`: 1
- `condition_number_41`: 4.341

## cd_vs_mlsynth

| Metric | Got | Accepted range | OK |
| --- | --- | --- | --- |
| max_abs_B_diff | 6.081e-06 | [0, 0.0001] | yes |
| max_rel_effect_diff | 1.495e-07 | [0, 0.0001] | yes |
| p_value_mismatches | 0 | [0, 0] | yes |
| max_rel_kappa_diff | 2.081e-07 | [0, 0.0001] | yes |
| max_abs_sensitivity_diff | 4.51e-07 | [0, 0.0001] | yes |
| max_rel_efficient_diff | 3.36e-12 | [0, 0.0001] | yes |
| selection_agrees | 1 | [1, 1] | yes |

Additional metrics:

- `cases`: 6
- `mlsynth_version`: 1.0.0
