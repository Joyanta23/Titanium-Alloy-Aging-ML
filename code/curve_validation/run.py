"""Curve-grouped validation of the titanium aging datasets."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
DATA_HASH = "05c7e9f95ec602dd4bea681bee304c356cfeb26cac53c1c33acb6affac2fb6c7"
ELEMENTS = [f"{e}_wt_pct" for e in ["Al", "Cr", "Mn", "Fe", "Mo", "Nb", "Ta", "Zr"]]
FEATURES = ELEMENTS + ["aging_temperature_K", "lt", "solution_temperature_K"]
GRID = {
    "max_depth": [None, 6, 12],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [0.7, 1.0],
}


def load_sets(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_HASH:
        raise ValueError("Dataset checksum differs from the reference workbook")
    book = pd.ExcelFile(path)
    frames = {}
    for sheet in ["Resistivity", "Hardness", "Paired_Rho_Hardness"]:
        frame = book.parse(sheet)
        frame[ELEMENTS] = frame[ELEMENTS].fillna(0.0)
        if not (frame.aging_time_ks > 0).all():
            raise ValueError("Aging times must be positive")
        frame["lt"] = np.log10(frame.aging_time_ks)
        frames[sheet] = frame
    paired = frames["Paired_Rho_Hardness"]
    return {
        "Resistivity": (frames["Resistivity"], "resistivity_uOhm_m", [], "curve_id"),
        "Hardness": (frames["Hardness"], "hardness_HV", [], "curve_id"),
        "Paired without rho": (paired, "hardness_HV", [], "curve_id_hv"),
        "Paired with rho": (paired, "hardness_HV", ["resistivity_uOhm_m"], "curve_id_hv"),
    }


def metrics(y, prediction):
    return {
        "R2": r2_score(y, prediction),
        "RMSE": mean_squared_error(y, prediction) ** 0.5,
        "MAE": mean_absolute_error(y, prediction),
    }


def grouped_prediction(frame, target, features, group, model, grid, outer):
    x = frame[features].to_numpy()
    y = frame[target].to_numpy()
    groups = frame[group].to_numpy()
    prediction = np.full(len(y), np.nan)
    for train, test in outer.split(x, y, groups):
        assert not set(groups[train]) & set(groups[test])
        if grid:
            inner = GroupKFold(3)
            for a, b in inner.split(x[train], y[train], groups[train]):
                assert not set(groups[train[a]]) & set(groups[train[b]])
            estimator = GridSearchCV(
                model, grid, scoring="neg_root_mean_squared_error", cv=inner, n_jobs=1
            )
            estimator.fit(x[train], y[train], groups=groups[train])
        else:
            estimator = model.fit(x[train], y[train])
        prediction[test] = estimator.predict(x[test])
    assert np.isfinite(prediction).all()
    return prediction


def repeated(sets, output, seeds):
    rows, predictions = [], {}
    for seed in seeds:
        for name, (frame, target, extra, group) in sets.items():
            model = ExtraTreesRegressor(200, random_state=seed, n_jobs=1)
            outer = GroupKFold(5, shuffle=True, random_state=seed)
            prediction = grouped_prediction(
                frame, target, FEATURES + extra, group, model, GRID, outer
            )
            rows.append(dict(seed=seed, dataset=name, **metrics(frame[target], prediction)))
            predictions[f"{seed}|{name}"] = prediction.tolist()
            pd.DataFrame(rows).to_csv(output / "repeated_results.csv", index=False)
            (output / "repeated_preds.json").write_text(json.dumps(predictions))
            print(rows[-1], flush=True)


def summarize(sets, output):
    results = pd.read_csv(output / "repeated_results.csv")
    predictions = json.loads((output / "repeated_preds.json").read_text())
    frame, target, _, group = sets["Paired without rho"]
    seeds = sorted(results.seed.unique())
    y = frame[target].to_numpy()
    rows = []
    for curve, subset in frame.groupby(group):
        ids = subset.index.to_numpy()
        errors = {}
        for label in ["Paired without rho", "Paired with rho"]:
            errors[label] = [
                np.mean((y[ids] - np.asarray(predictions[f"{seed}|{label}"])[ids]) ** 2) ** 0.5
                for seed in seeds
            ]
        a, b = errors["Paired without rho"], errors["Paired with rho"]
        rows.append(dict(curve=curve, family=subset.alloy_system.iloc[0],
                         without=np.mean(a), with_rho=np.mean(b),
                         wins=sum(v < u for u, v in zip(a, b))))
    curves = pd.DataFrame(rows)
    curves.to_csv(output / "per_curve_summary.csv", index=False)
    paired = results.pivot(index="seed", columns="dataset", values="RMSE")
    a, b = paired["Paired without rho"], paired["Paired with rho"]
    test = wilcoxon(curves.without, curves.with_rho, alternative="two-sided", method="auto")
    summary = {
        "repeats": len(seeds), "seeds": [int(seed) for seed in seeds],
        "mean_rmse_without_rho_HV": a.mean(), "mean_rmse_with_rho_HV": b.mean(),
        "reduction_of_mean_rmse_pct": 100 * (1 - b.mean() / a.mean()),
        "repeats_improved": int((b < a).sum()),
        "curves_improved": int((curves.with_rho < curves.without).sum()),
        "curves": len(curves), "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_p_two_sided": float(test.pvalue),
        "test_unit": "Per-curve RMSE averaged over repeats, paired with/without resistivity",
    }
    (output / "paired_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def ablation(sets, output):
    drops = {"All inputs": [], "No solution T": ["solution_temperature_K"],
             "No aging T": ["aging_temperature_K"], "No aging time": ["lt"],
             "No aging T and time": ["aging_temperature_K", "lt"], "No composition": ELEMENTS}
    rows = []
    for name, (frame, target, extra, group) in sets.items():
        for label, removed in drops.items():
            features = [f for f in FEATURES + extra if f not in removed]
            model = ExtraTreesRegressor(200, max_features=0.7, random_state=42, n_jobs=1)
            prediction = grouped_prediction(frame, target, features, group, model, {}, GroupKFold(5))
            score = metrics(frame[target], prediction)
            rows.append(dict(dataset=name, inputs=label, R2=round(score["R2"], 3),
                             RMSE=round(score["RMSE"], 4)))
    pd.DataFrame(rows).to_csv(output / "ablation.csv", index=False)


def learning(sets, output):
    rows = []
    for name, (frame, target, extra, group) in sets.items():
        for repeat in range(10):
            train, test = next(GroupShuffleSplit(1, test_size=0.2, random_state=repeat).split(frame, groups=frame[group]))
            curves = frame.iloc[train][group].unique()
            np.random.default_rng(repeat).shuffle(curves)
            for fraction in [0.2, 0.4, 0.6, 0.8, 1.0]:
                keep = set(curves[:max(2, int(round(fraction * len(curves))))])
                subset = frame.iloc[train][frame.iloc[train][group].isin(keep)]
                assert not set(subset[group]) & set(frame.iloc[test][group])
                model = ExtraTreesRegressor(200, max_features=0.7, random_state=42, n_jobs=1)
                model.fit(subset[FEATURES + extra], subset[target])
                prediction = model.predict(frame.iloc[test][FEATURES + extra])
                rows.append(dict(dataset=name, frac=fraction, n_train=len(subset),
                                 RMSE=metrics(frame.iloc[test][target], prediction)["RMSE"]))
    pd.DataFrame(rows).to_csv(output / "learning.csv", index=False)


def all_models(sets, output):
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    models = {
        "Baseline (mean)": (DummyRegressor(), {}),
        "Ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1, 10, 100]}),
        "KNN": (make_pipeline(StandardScaler(), KNeighborsRegressor()),
                {"kneighborsregressor__n_neighbors": [3, 5, 9, 15], "kneighborsregressor__weights": ["uniform", "distance"]}),
        "SVR": (make_pipeline(StandardScaler(), SVR()), {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 0.1]}),
        "Random Forest": (RandomForestRegressor(200, random_state=42, n_jobs=1),
                          {"max_depth": [None, 6, 12], "min_samples_leaf": [1, 2, 4]}),
        "Extra Trees": (ExtraTreesRegressor(200, random_state=42, n_jobs=1), GRID),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=42),
                              {"n_estimators": [200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3, 4]}),
        "XGBoost": (XGBRegressor(n_estimators=300, random_state=42, n_jobs=1, verbosity=0),
                    {"learning_rate": [0.05, 0.1], "max_depth": [3, 5], "subsample": [0.8, 1.0]}),
    }
    rows = []
    labels = {"Paired without rho": "Paired-noRho", "Paired with rho": "Paired-Rho"}
    for name, (frame, target, extra, group) in sets.items():
        for label, (model, grid) in models.items():
            prediction = grouped_prediction(frame, target, FEATURES + extra, group, model, grid, GroupKFold(5))
            score = metrics(frame[target], prediction)
            rows.append(dict(ds=labels.get(name, name), scheme="Curve", model=label,
                             R2=score["R2"], RMSE=score["RMSE"]))
            pd.DataFrame(rows).to_csv(output / "all_models_verified.csv", index=False)
            print(rows[-1], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["check", "repeated", "summary", "models", "ablation", "learning"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/Titanium_Simple_ML_Dataset.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "curve_validation_output")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be distinct")
    sets = load_sets(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "joblib", "xgboost"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    metadata = dict(python=platform.python_version(), packages=versions,
                    input_sha256=DATA_HASH, task=args.task, seeds=args.seeds)
    (args.output / f"environment_{args.task}.json").write_text(json.dumps(metadata, indent=2))
    if args.task == "check":
        print({name: {"rows": len(d), "curves": d[g].nunique()} for name, (d, _, _, g) in sets.items()})
    elif args.task == "repeated":
        repeated(sets, args.output, args.seeds)
    elif args.task == "summary":
        summarize(sets, args.output)
    else:
        {"models": all_models, "ablation": ablation, "learning": learning}[args.task](sets, args.output)


if __name__ == "__main__":
    main()
