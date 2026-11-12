# Curve-grouped validation

These analyses evaluate unseen aging curves in the alloy systems represented by the dataset. They supplement the original row-split analysis.

Use Python 3.12 in a separate environment from the original notebook:

```bash
python -m venv .venv-curve
# Activate the environment using the command for your operating system.
python -m pip install -r code/curve_validation/requirements.txt
python code/curve_validation/run.py check
python code/curve_validation/run.py repeated
python code/curve_validation/run.py summary
python code/curve_validation/run.py models
python code/curve_validation/run.py ablation
python code/curve_validation/run.py learning
```

Run these commands from the repository root. Results are written to `curve_validation_output/`; reference results under `results/curve_validation/` are preserved. Use `--output PATH` to choose another directory. The complete nested analyses can take several hours on a laptop. The `summary` task needs the predictions and scores produced by `repeated` in the same output directory.

## Protocol

- Inputs: eight composition values in wt.%, aging temperature in K, log10 of aging time in ks, and solution-treatment temperature in K. The paired model adds measured resistivity in µΩ·m. Missing composition entries are treated as design zeros, following the dataset convention.
- Grouping: `curve_id` for the two main sheets and `curve_id_hv` for paired hardness. Both paired models use identical rows and outer folds.
- Repeated analysis: five outer GroupKFold folds with shuffle enabled, seeds 1–10. The ExtraTrees seed also changes with the repeat. Each outer training set uses three inner GroupKFold folds without shuffle.
- Tuning: 200 trees; `max_depth = [None, 6, 12]`, `min_samples_leaf = [1, 2, 4]`, `max_features = [0.7, 1.0]`. The 18 combinations are selected by mean inner-fold RMSE, without using outer-fold scores.
- The seven-regressor comparison and mean baseline use one unshuffled five-fold assignment, with model seed 42. Ridge, KNN and SVR scaling is fitted within the training folds through pipelines. Model-specific grids are defined in `all_models`.
- Ablation: fixed ExtraTrees settings (200 trees, `max_features=0.7`, seed 42), five unshuffled curve-grouped folds. Inputs are removed without retuning.
- Learning curves: ten GroupShuffleSplit assignments, seeds 0–9, holding out 20% of curves. Each assignment uses nested subsets of the remaining training curves at fractions 0.2–1.0. ExtraTrees settings are fixed as in ablation. The reference file retains its original four columns; rows within each dataset occur in repeat order, with five fractions per repeat.

## Paired statistics

For each of the 27 curves, RMSE is calculated separately in each of the ten repeats and then averaged. A two-sided Wilcoxon signed-rank test compares those 27 paired averages, with SciPy `method="auto"`. This gives p = 0.028018 in the recorded environment. It is not a test of the count 21/27, and the ten appearances of each curve are not treated as independent observations. The test is an exploratory curve-level summary; fitted training sets overlap and the curves share literature sources.

Mean paired RMSE is the arithmetic mean of ten pooled out-of-fold RMSE values. The reported 4.8% reduction is calculated from those two means. It is distinct from the median percentage improvement across curves.

The workbook checksum and installed package versions are recorded by the runner. Reference computations were completed locally; hosted Colab execution of this additional runner has not been tested. These additional results use scikit-learn 1.8.0. Earlier runs with different environments/settings gave different magnitudes, so use the included reference files for this protocol.
