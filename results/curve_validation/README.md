# Curve-grouped reference results

Computed on 12 November 2026 using the workbook and protocol recorded in `reference_environment.json`. Code and commands are in [../../code/curve_validation](../../code/curve_validation/README.md).

| File | Contents |
| --- | --- |
| `repeated_results.csv` | Ten-repeat ExtraTrees scores for all four datasets |
| `repeated_preds.json` | Out-of-fold predictions in workbook row order, keyed by `seed|dataset` |
| `per_curve_summary.csv` | Mean per-curve RMSE across repeats, with/without resistivity; `wins` counts repeat-level improvements |
| `paired_summary.json` | Aggregate RMSE comparison and two-sided Wilcoxon result |
| `all_models_verified.csv` | Seven regressors and a mean baseline, one curve-grouped fold assignment |
| `ablation.csv` | Fixed-setting ExtraTrees scores after removing inputs |
| `learning.csv` | Fixed-setting learning curves, ten training/held-out curve assignments |
| `reference_environment.json` | Versions, data checksum and reference protocol |

Paired hardness mean RMSE: **61.6667 → 58.7142 HV**, a **4.7879%** reduction. Six of ten repeats improved. Mean error improved in **21 of 27 curves**; the two-sided Wilcoxon test on the 27 paired curve averages gives **p = 0.028018**. These are additional curve-grouped results, distinct from the original row-split results.

Resistivity scores use µΩ·m; hardness scores use HV. R² is dimensionless. The seven-model table uses one fold assignment, and the ablation/learning experiments use fixed settings rather than nested tuning.
