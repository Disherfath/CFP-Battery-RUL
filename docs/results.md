# Reproducing the reported tables and figures

The `results/` directory contains the derived results used in the six-page camera-ready revision of *Cell-Level Feature-Preserving Imputation for Battery Remaining Useful Life Prediction* (27 October 2026). All reported dataset-pattern combinations are retained, including cases in which CFP does not improve the baseline.

These files support table and figure reconstruction without downloading the raw battery curves or retraining neural models. The training and evaluation programs provide the separate path from the original datasets to model predictions. Original measured curves are distributed by their dataset owners; this repository includes derived errors and summaries.

## Generate the paper tables and Figure 2

From the repository root in an environment with NumPy, pandas, and Matplotlib:

```bash
python reproduction/reproduce_paper.py --output outputs/paper --bootstrap
```

Choose a new output directory for each invocation. The program:

1. Verifies the SHA256 hashes of all 18 frozen result CSVs.
2. Recomputes every Table II mean and sample standard deviation from physical-cell errors for seven methods and two datasets.
3. Generates LaTeX fragments for Tables I-V, along with machine-readable combined Tables III and V.
4. Checks all 24 paired comparisons and, with `--bootstrap`, replays their 20,000-resample physical-cell intervals.
5. Regenerates Figure 2 as a PDF and a 350-dpi PNG and writes `verification.json`.

LaTeX table fragments use `booktabs`; the wide tables use the `table*` environment. Layout may differ from the paper's final float placement. Figure labels and values follow the reported Figure 2. Font/library versions can change rendering slightly without changing the data.

Use `--tables-only` to skip the figure. The command does not fit an imputer or a regression head. Replaying the bootstrap resamples saved errors; it is separate from repeating neural training.

## Experimental interpretation

- HUST has 47/8/22 training/validation/test physical cells; XJTU has 29/7/19.
- All tasks use the first 50 usable cycles, 128 positions per cycle, and voltage/current/capacity channels.
- `mcar_time40`, `block40`, and `window40` hide 51 of 128 positions per cycle (39.84%). Channels share the position mask. Window masks retain a contiguous 77-position interval.
- APE is averaged over the repeated missing masks within each physical cell and condition. The three missing conditions then receive equal weight. The reported repeated-run figures average errors over initializations, rather than averaging predictions into an ensemble.
- Table I uses the original single initialization and reports both Ridge and RBF outputs. Table II uses three initializations and the Ridge head, with the jointly trained MLP reported separately. Values from those two tables should be interpreted with their respective replication levels.
- The main SAITS arms and SAITS-64 use seeds 20261005, 20261018, and 20261019. T1-within uses 20261008, 20261018, and 20261019, preserving its original reference run.
- Mean MAPE is in percent; MAE and RMSE are in cycles. RMSE is the square root of mean squared error. It is not the mean of individual cell RMSEs. Sample standard deviations use `ddof=1` across three initialization-specific dataset metrics.
- In paired comparisons, `delta_pp = CFP - comparator`, in MAPE percentage points. A negative value favors CFP.
- Bootstrap intervals resample 22 HUST or 19 XJTU physical cells after averaging their errors over the observed initializations. They assess cell-sampling uncertainty under the fixed split/protocol. The three initialization deltas assess a different source of variation.
- All 24 dataset/scope/comparator combinations are exported. The supplemental sign and signed-rank statistics are descriptive, without a multiple-testing adjustment.

## File map and data dictionary

| File | Rows represent | Main columns and units |
| --- | --- | --- |
| `table1_single_initialization.csv` | Each Table I configuration | `configuration` is the internal method ID; `label` is the display label. Four dataset/regressor columns contain MAPE in percent. The Joint MLP has one output per dataset, stored in the Ridge-position column; blank RBF fields indicate that it is not an RBF model. |
| `table2_three_initializations.csv` | Seven methods in Table II | `method`, `arm`, `family`; each dataset has `mape`, `mae`, and `rmse` fields suffixed `_mean` and `_sd`. |
| `table3_conditions_core.csv` | Each dataset and missingness pattern | `physical_cells` is the number of test cells. Base, CFP, and Joint MLP have `_mean`/`_sd`; `li` and `observed_only` are deterministic MAPE references. |
| `table3_conditions_references.csv` | The same six dataset-pattern pairs | `saits64_*` and `t1_within_*` contain MAPE means and sample SDs. This file joins the preceding file on `dataset,condition`. |
| `table4_hust_features.csv` | Four Table IV methods | `point_*` is hidden-point MAE in training-channel-standardized space; `feature_*` is descriptor MSE on the shared frozen training-LI feature scale; `mape_*` is Ridge RUL MAPE. LI has `initializations=0` and blank SDs to indicate a deterministic calculation. |
| `validation_weight_sensitivity.csv` | Dataset, feature weight, head family, and selection rule | Includes all four CFP weights, both `ridge`/`rbf` heads, and `fixed_head`/`same_grid_selected` results. `mape_pct`, `mae_cycles`, and `rmse_cycles` are validation metrics. Table V uses `same_grid_selected` with `ridge`; its main feature weight remains 0.1. |
| `validation_reference_grid_hust.csv`, `validation_reference_grid_xjtu.csv` | Selected original validation configurations | Include the Base row used at weight zero in Table V. `missing_*` metrics average the three missing conditions; `four_*` also includes clean; `clean_*` reports clean only. `alpha`, `gamma`, and `grid_index` describe the selected head. Blank `gamma` is expected for Ridge. |
| `feature_diagnostics_summary.csv` | Dataset and matched SAITS arm | `point_*_z` is point reconstruction in channel-standardized space; `feature_*_z` is on the frozen descriptor scale. `feature_mse_without_Qmin_z_*` averages the remaining 17 coordinates after excluding capacity minimum. `_sample_sd` is across three initializations. |
| `feature_diagnostics_by_initialization.csv` | Dataset, initialization, and SAITS arm | Includes each of the 18 `*_square_delta_z` coordinates. `V`, `I`, and `Q` mean voltage, current, and capacity. `last5_minus_first5` and `cycle_mean_slope_t01` are the cross-cycle descriptors. `view` is an averaged historical bookkeeping index, not an independent observation count. |
| `paired_cells.csv` | A physical cell within a dataset, scope, and comparator | Mean cell APE in percent for CFP and comparator, and `delta_pp`. `missing3` is the equal-three-condition aggregate. `OO` means observed-only descriptors, distinct from `C0`. |
| `paired_summary.csv` | All 24 dataset/scope/comparator groups | Means, medians, improved/worsened/tied-cell counts, physical-cell bootstrap bounds, descriptive tests, and initialization delta mean/SD. Columns ending `_pp` are percentage points. |
| `paired_by_initialization.csv` | Each paired group and initialization | Dataset MAPE for CFP/comparator and their difference. Three rows per group. |
| `paired_source_errors.csv` | Physical cell, initialization, method, and condition | Base and CFP's `ape`, `ae`, `mse`, already averaged over repeated masks within a condition. `ape` is a fraction here, not percent. |
| `cell_errors_saits.csv` | Matched SAITS arm, head family, initialization, and cell | Three-condition-averaged `ape` (fraction), `ae` (cycles), and `mse` (cycles squared); `stage=test`. |
| `cell_errors_joint.csv` | Joint model output/head, initialization, and cell | Same error definitions. `family=joint` is the learned MLP; `ridge`/`rbf` refer to refitted heads. The supplied cell `rmse` column is not averaged when reconstructing a dataset RMSE. |
| `cell_errors_references.csv` | SAITS-64 or T1-within, head, initialization, and cell | `kind` distinguishes the two references. Same error definitions; includes both head families. |
| `validation_selection_stability.csv` | Dataset, initialization, arm, and head family | 36 groups of 20,000 validation-cell bootstraps. `original_choice_retention_fraction` is the frequency of reselecting the original head. `physical_validation_cells` is eight or seven. Losses and gaps ending `_pp` or `_percent` are on the MAPE percentage scale. |

`arm=base` denotes the native SAITS reconstruction objective; `uniform` denotes CFP; `task` denotes the directional feature-loss ablation. These historical IDs are retained for traceability. `C0` is same-cell cross-cycle borrowing. It is separate from `OO` and `observed_only18`.

The SHA256 manifest records the original export identity for each frozen CSV. Hashes apply to the supplied file bytes, including their line endings; preserve the repository's result-file attributes when checking out the release.

## Data attribution and reuse

The derived result tables in `results/` are distributed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Cite the paper and acknowledge the source datasets when reusing them. The predictions and errors were generated by the present study, not by the dataset authors.

- **HUST:** Ye Yuan, Guijun Ma, and Songpei Xu, dataset version 2, [DOI 10.17632/nsc7hnsg4s.2](https://doi.org/10.17632/nsc7hnsg4s.2), CC BY 4.0. Related paper: Ma et al., *Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning*, Energy & Environmental Science, 2022, [DOI 10.1039/D2EE01676A](https://doi.org/10.1039/D2EE01676A).
- **XJTU:** Fujin Wang, [Zenodo record 10963339](https://doi.org/10.5281/zenodo.10963339), CC BY 4.0. Related paper: Wang et al., *Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis*, Nature Communications, 2024, [DOI 10.1038/s41467-024-48779-z](https://doi.org/10.1038/s41467-024-48779-z).

Transformations represented here include fixed cell splitting, early-history resampling, synthetic missingness, model completion, lifetime regression, error aggregation, and resampling of physical-cell errors. Raw measured curves and raw dataset archives are not bundled.
