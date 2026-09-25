# Titanium alloy aging: resistivity and hardness prediction

Code, curated Excel data and selected plots for **“Interpretable Machine Learning of Room-Temperature Electrical Resistivity and Vickers Hardness in Aged Titanium Alloys.”**

The literature-based datasets contain **836** room-temperature resistivity rows and **568** Vickers hardness rows. **349** conditions have both measurements. The model comparison uses a fixed 80:20 row split, with five-fold cross-validation on the training rows for model selection. The paired comparison evaluates hardness with and without *measured* resistivity on the same rows. Reported row-split performance does not represent a test on entirely new alloy families.

## Start here

1. Open `data/Titanium_Simple_ML_Dataset.xlsx` for the three main sheets: `Resistivity`, `Hardness`, and `Paired_Rho_Hardness`.
2. Open `data/Titanium_Aging_RT_Resistivity_Dataset_Frozen_v1.1_2026-09-20.xlsx` for the frozen dataset, its source register, scope decisions and data dictionary.
3. Read `results/Titanium_ML_Results.xlsx` or view the plots under `results/` for the reported findings.
4. To rerun the models, download this repository as a ZIP, open `Titanium_ML_Google_Colab.ipynb` in Google Colab, run the first cell, upload the downloaded ZIP, then run the remaining cells in order. The final cell downloads a results ZIP.

The notebook runs `code/main_ml/train.py`, `code/main_ml/complete_xgb_shap.py`, and `code/paired_ml/run.py`. The pinned Python packages are in `code/requirements.txt`. A Colab runtime may need to restart after installation if package versions already loaded in that session conflict.

## Included results

`results/resistivity/` and `results/hardness/` contain model comparison tables, predictions, SHAP importance tables and figures. `results/paired_hardness/` contains the identical-row hardness comparisons and plots. The `.joblib` model binaries were intentionally omitted; running the notebook regenerates fitted models.

## Sources and reuse

Each row in the frozen workbook retains source identifiers. The nine cited source publications are listed in the manuscript; the publishers' PDFs are **not** part of this repository. The data were transcribed or digitized from the cited literature, including with WebPlotDigitizer (automeris.io). Refer to the source register and cite the original studies when reusing their measurements.

The journal manuscript is under coauthor review and is not included here. A license for repository reuse has not yet been selected by the authors; agree on one before making this repository public.
