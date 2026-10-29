"""Compare hardness models with and without measured resistivity on matched conditions."""

import os

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
from pathlib import Path
import json, hashlib, shutil, sys
import numpy as np
import pandas as pd
import joblib, shap
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else ROOT.parents[1] / "data" / "Titanium_Aging_Dataset_v1.2.xlsx"
)


def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2, default=str))


def metric(y, p):
    return dict(
        R2=float(r2_score(y, p)),
        RMSE=float(mean_squared_error(y, p) ** 0.5),
        MAE=float(mean_absolute_error(y, p)),
    )


expected = "8e385adac25aa74bf8d464577f36a3a55828d0d15dd885af3979307b671dbb77"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == expected
r = pd.read_excel(SOURCE, "RT_Resistivity")
h = pd.read_excel(SOURCE, "Vickers_Hardness")
keys = [
    "source_id",
    "alloy_system",
    "alloy_code",
    "Ti_wt_pct",
    "Al_wt_pct",
    "Cr_wt_pct",
    "Mn_wt_pct",
    "Fe_wt_pct",
    "Mo_wt_pct",
    "Nb_wt_pct",
    "Ta_wt_pct",
    "Zr_wt_pct",
    "solution_temperature_K",
    "solution_time_ks",
    "aging_temperature_K",
    "aging_time_ks",
]
d = r.merge(h, on=keys, suffixes=("_rho", "_hv"), validate="one_to_one", sort=False)
assert len(d) == 349
for c in [
    "quench",
    "prior_processing",
    "initial_phase_or_condition",
    "O_wt_pct",
    "N_wt_pct",
    "H_wt_pct",
]:
    assert d[c + "_rho"].fillna("NA").equals(d[c + "_hv"].fillna("NA")), c
d.to_csv(ROOT / "paired_source_records.csv", index=False)
features = [e + "_wt_pct" for e in ["Al", "Cr", "Mn", "Fe", "Mo", "Nb", "Ta", "Zr"]]
X = d[features].copy()
logs = []
for c in features:
    missing = X[c].isna()
    element = c.split("_")[0]
    assert not (
        missing & d.alloy_system.str.split("-").apply(lambda a: element in a)
    ).any()
    for i in d.index[missing]:
        logs.append(
            dict(
                record_id=d.at[i, "record_id_rho"],
                feature=c,
                reason="Unlisted alloy-system element; design-zero, not measured purity",
            )
        )
    X[c] = X[c].fillna(0)
pd.DataFrame(logs).to_csv(ROOT / "design_zero_log.csv", index=False)
X["aging_temperature_K"] = d.aging_temperature_K
X["log10_aging_time_ks"] = np.log10(d.aging_time_ks)
X["solution_temperature_K"] = d.solution_temperature_K
features = X.columns.tolist()
X["resistivity_uOhm_m"] = d.resistivity_uOhm_m
y = d.hardness_HV
assert np.isfinite(X.to_numpy()).all()
tr, te = train_test_split(np.arange(len(d)), test_size=0.2, random_state=42)
cv = list(KFold(5, shuffle=True, random_state=42).split(tr))
fold = np.full(len(d), -1)
for i, (_, v) in enumerate(cv):
    fold[tr[v]] = i
ready = X.assign(
    record_id_rho=d.record_id_rho,
    record_id_hv=d.record_id_hv,
    hardness_HV=y,
    split=np.where(np.isin(np.arange(len(d)), tr), "train", "test"),
    fold=fold,
)
ready.to_csv(ROOT / "model_ready.csv", index=False)
etgrid = dict(
    max_depth=[None, 6, 12], min_samples_leaf=[1, 2, 4], max_features=[0.7, 1.0]
)
xgrid = dict(
    n_estimators=[150, 300, 500],
    max_depth=[2, 3, 5],
    learning_rate=[0.03, 0.08, 0.15],
    min_child_weight=[1, 3, 6],
    reg_lambda=[1, 5, 15],
    subsample=[0.8, 1.0],
    colsample_bytree=[0.8, 1.0],
)
save(
    "protocol.json",
    dict(
        seed=42,
        rows=349,
        train=len(tr),
        test=len(te),
        pair_keys=keys,
        source_hash=expected,
        primary="Fixed ExtraTrees: 200 trees, max_features=0.7; compare with and without measured resistivity",
        secondary="12-trial searches for ET, RF and XGB for each feature set; select by training CV RMSE",
        search_spaces=dict(ExtraTrees=etgrid, RandomForest=etgrid, XGBoost=xgrid),
        caveats=[
            "Exploratory paired analysis; pilot test results were already available",
            "Exact matching on stored conditions, including standardized nominal aging times",
            "Pairs match reported conditions; specimen identity is not established",
            "Row split: related curves can occur in both train and test sets",
            "The with-resistivity model requires measured resistivity",
        ],
    ),
)
models = {}
rows = []
for variant, cols in [
    ("without_rho", features),
    ("with_rho", features + ["resistivity_uOhm_m"]),
]:
    for label, m, grid in [
        (
            "FixedExtraTrees",
            ExtraTreesRegressor(
                n_estimators=200, max_features=0.7, random_state=42, n_jobs=1
            ),
            None,
        ),
        (
            "ExtraTrees",
            ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1),
            etgrid,
        ),
        (
            "RandomForest",
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1),
            etgrid,
        ),
        (
            "XGBoost",
            XGBRegressor(
                random_state=42,
                n_jobs=1,
                tree_method="hist",
                objective="reg:squarederror",
            ),
            xgrid,
        ),
    ]:
        if grid is None:
            mse = -cross_val_score(
                m, X.iloc[tr][cols], y.iloc[tr], cv=cv, scoring="neg_mean_squared_error"
            )
            m.fit(X.iloc[tr][cols], y.iloc[tr])
            params = m.get_params()
            score = float(np.sqrt(mse.mean()))
            meanrmse = float(np.sqrt(mse).mean())
        else:
            search = RandomizedSearchCV(
                m,
                grid,
                n_iter=12,
                scoring="neg_mean_squared_error",
                cv=cv,
                random_state=42,
                n_jobs=2,
                error_score="raise",
            ).fit(X.iloc[tr][cols], y.iloc[tr])
            m = search.best_estimator_
            params = search.best_params_
            score = float(np.sqrt(-search.best_score_))
            meanrmse = None
            pd.DataFrame(search.cv_results_).to_csv(
                ROOT / f"{variant}_{label}_search.csv", index=False
            )
        models[(variant, label)] = (m, cols)
        rows.append(
            dict(
                variant=variant,
                model=label,
                CV_RMSE=score,
                CV_mean_fold_RMSE=meanrmse,
                params=params,
            )
        )
        print("FIT", variant, label, score, flush=True)
# Select tuned models by training CV RMSE; retain the fixed-model comparison.
chosen = {
    v: min(
        [q for q in rows if q["variant"] == v and q["model"] != "FixedExtraTrees"],
        key=lambda q: q["CV_RMSE"],
    )["model"]
    for v in ["without_rho", "with_rho"]
}
save("selection_before_test.json", chosen)
preds = ready[["record_id_rho", "record_id_hv", "split", "hardness_HV"]].copy()
for row in rows:
    v, label = row["variant"], row["model"]
    m, cols = models[(v, label)]
    p = m.predict(X[cols])
    row.update({"train_" + k: a for k, a in metric(y.iloc[tr], p[tr]).items()})
    row.update({"test_" + k: a for k, a in metric(y.iloc[te], p[te]).items()})
    preds[v + "_" + label] = p
    joblib.dump(dict(model=m, features=cols), ROOT / f"{v}_{label}.joblib")
    assert np.allclose(
        joblib.load(ROOT / f"{v}_{label}.joblib")["model"].predict(X[cols]), p
    )
preds.to_csv(ROOT / "predictions.csv", index=False)
pd.DataFrame(rows).to_csv(ROOT / "comparison.csv", index=False)
save("comparison.json", rows)
shap_checks = {}
for v in ["without_rho", "with_rho"]:
    label = chosen[v]
    m, cols = models[(v, label)]
    bg = np.random.default_rng(42).choice(tr, 64, replace=False)
    E = shap.TreeExplainer(
        m, data=X.iloc[bg][cols], feature_perturbation="interventional"
    )(X.iloc[te][cols])
    p = m.predict(X.iloc[te][cols])
    err = float(np.max(np.abs(E.base_values + E.values.sum(1) - p)))
    assert np.allclose(E.base_values + E.values.sum(1), p, rtol=1e-5, atol=1e-4)
    shap_checks[v] = dict(
        max_error=err, background_ids=d.record_id_rho.iloc[bg].tolist(), model=label
    )
    tab = pd.DataFrame(E.values, columns=cols)
    tab.insert(0, "base_value", E.base_values)
    tab.insert(0, "record_id_rho", d.record_id_rho.iloc[te].to_numpy())
    tab.to_csv(ROOT / f"{v}_shap_values.csv", index=False)
    imp = pd.DataFrame(
        dict(feature=cols, mean_absolute_SHAP_HV=np.abs(E.values).mean(0))
    ).sort_values("mean_absolute_SHAP_HV", ascending=False)
    imp.to_csv(ROOT / f"{v}_shap_importance.csv", index=False)
    E.feature_names = [
        c.replace("_wt_pct", " (wt.%)")
        .replace("resistivity_uOhm_m", "Measured resistivity (µΩ·m)")
        .replace("aging_temperature_K", "Aging temperature (K)")
        .replace("log10_aging_time_ks", "log10(aging time / ks)")
        .replace("solution_temperature_K", "Solution temperature (K)")
        for c in cols
    ]
    shap.plots.beeswarm(E, max_display=len(cols), show=False, plot_size=(10, 7))
    plt.title(f"{label}: " + v.replace("_", " "))
    plt.xlabel("SHAP contribution to hardness (HV)")
    plt.tight_layout()
    for ext in ["png", "svg"]:
        plt.savefig(ROOT / f"{v}_shap.{ext}", bbox_inches="tight")
    plt.close("all")
    if v == "with_rho":
        i = cols.index("resistivity_uOhm_m")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X.resistivity_uOhm_m.iloc[te], E.values[:, i], c="#007f86")
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(
            xlabel="Measured resistivity (µΩ·m)",
            ylabel="Resistivity SHAP contribution (HV)",
        )
        fig.tight_layout()
        fig.savefig(ROOT / "resistivity_SHAP_dependence.png", dpi=200)
        plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, v in zip(axes, ["without_rho", "with_rho"]):
    label = chosen[v]
    m, cols = models[(v, label)]
    p = m.predict(X.iloc[te][cols])
    mt = metric(y.iloc[te], p)
    ax.scatter(y.iloc[te], p, c="#007f86", s=25)
    limits = [min(y.iloc[te].min(), p.min()), max(y.iloc[te].max(), p.max())]
    ax.plot(limits, limits, "k--", lw=1)
    ax.set(
        xlabel="Measured hardness (HV)",
        ylabel="Predicted hardness (HV)",
        title=f"{label}: "
        + (
            "with measured resistivity"
            if v == "with_rho"
            else "without measured resistivity"
        )
        + f'\nTest R²={mt["R2"]:.4f}; RMSE={mt["RMSE"]:.2f} HV',
    )
fig.tight_layout()
for ext in ["png", "svg"]:
    fig.savefig(ROOT / f"paired_prediction.{ext}", dpi=200)
plt.close(fig)
primary = [q for q in rows if q["model"] == "FixedExtraTrees"]
secondary = [q for q in rows if q["model"] == chosen[q["variant"]]]
save(
    "summary.json",
    dict(
        primary=primary,
        selected_tuned=secondary,
        primary_test_RMSE_reduction_pct=100
        * (1 - primary[1]["test_RMSE"] / primary[0]["test_RMSE"]),
        sources=d.groupby("source_id").size().to_dict(),
        alloy_source_identities=len(d[["source_id", "alloy_code"]].drop_duplicates()),
    ),
)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == expected
save(
    "verification.json",
    dict(
        pairs_unique=True,
        processing_fields_match=True,
        input_unchanged=True,
        models_reloaded_predictions_match=True,
        SHAP=shap_checks,
    ),
)
if SOURCE.resolve() != (ROOT / SOURCE.name).resolve():
    shutil.copy2(SOURCE, ROOT / SOURCE.name)
print(json.dumps(dict(primary=primary, selected=secondary), indent=2), flush=True)
