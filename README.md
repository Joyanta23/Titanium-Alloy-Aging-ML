# Titanium alloy aging: resistivity and hardness prediction

Code, curated Excel data and selected plots for **“Interpretable Machine Learning of Room-Temperature Electrical Resistivity and Vickers Hardness in Aged Titanium Alloys.”**

The literature-based datasets contain **836** room-temperature resistivity rows and **568** Vickers hardness rows. **349** conditions have both measurements. The model comparison uses a fixed 80:20 row split, with five-fold cross-validation on the training rows for model selection. The paired comparison evaluates hardness with and without *measured* resistivity on the same rows. Reported row-split performance does not represent a test on entirely new alloy families.

## Start here

1. Open `data/Titanium_Simple_ML_Dataset.xlsx` for the three main sheets: `Resistivity`, `Hardness`, and `Paired_Rho_Hardness`.
2. Open `data/Titanium_Aging_RT_Resistivity_Dataset_Frozen_v1.1_2026-09-20.xlsx` for the full dataset, its source register, scope decisions and data dictionary.
3. Read `results/Titanium_ML_Results.xlsx` or view the plots under `results/` for the reported findings.
4. To rerun the models, download this repository as a ZIP, open `Titanium_ML_Google_Colab.ipynb` in Google Colab, run the first cell, upload the downloaded ZIP, then run the remaining cells in order. The final cell displays the new scores and prediction plots, checks scores against the included reference results, and downloads `Titanium_ML_New_Results.zip`.

The notebook runs `code/main_ml/train.py`, `code/main_ml/complete_xgb_shap.py`, and `code/paired_ml/run.py`. The pinned Python packages are in `code/requirements.txt`. A Colab runtime may need to restart after installation if package versions already loaded in that session conflict.

## Included results

`results/resistivity/` and `results/hardness/` contain model comparison tables, predictions, SHAP importance tables and figures. `results/paired_hardness/` contains the identical-row hardness comparisons and plots. The `.joblib` model binaries were intentionally omitted; running the notebook regenerates fitted models.

## Sources and reuse

Each row in the full workbook retains source identifiers. The nine cited source publications are listed in the manuscript; the publishers' PDFs are **not** part of this repository. The data were transcribed or digitized from the cited literature, including with WebPlotDigitizer (automeris.io). Refer to the source register and cite the original studies when reusing their measurements.

The journal manuscript is under coauthor review and is not included here. A license for repository reuse has not yet been selected by the authors; agree on one before making this repository public.

## Reproducibility check

On 25 September 2026, all three ML scripts were run from a GitHub-verified copy using Python 3.12 and the pinned scientific packages. The model comparison tables, predictions and included SHAP importance tables reproduced the reference CSVs exactly at their saved precision. The new Colab output cell was also checked locally. This was a local Python execution of the notebook workflow, not a test of Google's hosted Colab service.

The `results/` folder and its Excel workbook contain the reference results. Each notebook run separately exports `Titanium_ML_New_Results.zip`, containing `Summary.csv` and `resistivity/`, `hardness/`, and `paired_hardness/`. These are the fresh outputs; the reference workbook is not silently overwritten.
