# Beta titanium alloy aging: resistivity and hardness prediction

Code, curated Excel data and selected plots for **“Interpretable Machine Learning for Predicting Room-Temperature Electrical Resistivity and Vickers Hardness in Aged Beta Titanium Alloys”**

The literature-based datasets contain **836** room-temperature resistivity rows and **568** Vickers hardness rows. **349** conditions have both measurements. The model comparison uses a fixed 80:20 row split, with five-fold cross-validation on the training rows for model selection. The paired comparison evaluates hardness with and without *measured* resistivity on the same rows.

## Start here

1. Open `data/Titanium_Simple_ML_Dataset.xlsx` for the three main sheets: `Resistivity`, `Hardness`, and `Paired_Rho_Hardness`.
2. Open `data/Titanium_Aging_Dataset_v1.2.xlsx` for the full dataset, its source register, scope decisions and data dictionary.
3. Read `results/Titanium_ML_Results.xlsx` or view the plots under `results/` for the reported findings.
4. To rerun the models, download this repository as a ZIP, open `Titanium_ML_Google_Colab.ipynb` in Google Colab, run the first cell, upload the downloaded ZIP, then run the remaining cells in order. The final cell displays the new scores and prediction plots, checks scores against the included reference results, and downloads `Titanium_ML_New_Results.zip`.

The notebook runs `code/main_ml/train.py`, `code/main_ml/complete_xgb_shap.py`, and `code/paired_ml/run.py`. The pinned Python packages are in `code/requirements.txt`. A Colab runtime may need to restart after installation if package versions already loaded in that session conflict.

## Included results

`results/resistivity/` and `results/hardness/` contain model comparison tables, predictions, SHAP importance tables and figures. `results/paired_hardness/` contains the identical-row hardness comparisons and plots. The `.joblib` model binaries were intentionally omitted; running the notebook regenerates fitted models.

## Sources and reuse

Each row in the full workbook retains source identifiers. The nine cited source publications are listed in the manuscript; the publishers' PDFs are **not** part of this repository. The data were transcribed or digitized from the cited literature, including with WebPlotDigitizer (automeris.io). Refer to the source register and cite the original studies when reusing their measurements.

Code is released under the MIT License (see LICENSE). The authors' dataset compilation, annotations and original figures are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), to the extent the authors hold the relevant rights. This license does not cover third-party material or grant rights owned by the cited publishers or researchers. Cite the original studies and this repository when reusing the data.

## Reproducing the results

Use the pinned packages and run the three scripts in notebook order. The workflow was tested locally with Python 3.12 and in hosted Google Colab. Local results match the reference tables. In Colab, selected-model scores and all non-KNN comparisons match; the KNN baseline scores differ. The notebook reports those differences in `reference_comparison.csv` and checks all other models against the reference tables. New runs export `Titanium_ML_New_Results.zip` separately from the reference `results/` folder.

Near-equal distances at the neighbor cutoff occur in these data; numerical differences in how those ties are resolved are a possible reason for the KNN variation.

XGBoost was added after the initial seven-model comparison, using the same split and training folds. The paired analysis is exploratory. SHAP values describe model associations.

Source labels were corrected on 23 October 2026: `Niinomi2002` → `Ikeda2002` and `IkedaUeda2017` → `IkedaUeda2016`. Numerical data, row order and splits were unchanged. Version 1.2 simplifies the documentation and corrects the remaining Ikeda/Ueda year label; it does not change numerical data.

## Selected results

Model selection uses training CV RMSE. ExtraTrees test scores are R² = 0.9640 and RMSE = 0.0664 µΩ·m for resistivity, and R² = 0.9703 and RMSE = 17.19 HV for hardness. On the paired subset, adding measured resistivity reduces test RMSE from 24.38 to 19.13 HV.
