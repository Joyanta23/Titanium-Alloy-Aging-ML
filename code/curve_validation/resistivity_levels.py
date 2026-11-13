"""Resistivity under composition-grouped and alloy-system-grouped validation.

Uses the same nested ExtraTrees procedure as run.py. Composition groups use
GroupKFold(5) shuffled with seeds 1-10. Alloy-system groups use
leave-one-group-out, repeated with ExtraTrees seeds 1-10.
"""
import argparse
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from run import ELEMENTS, FEATURES, GRID, grouped_prediction, load_sets, metrics


def one(frame, scheme, seed):
    group = "composition" if scheme == "Composition" else "alloy_system"
    outer = GroupKFold(5, shuffle=True, random_state=seed) if scheme == "Composition" else LeaveOneGroupOut()
    model = ExtraTreesRegressor(200, random_state=seed, n_jobs=1)
    prediction = grouped_prediction(frame, "resistivity_uOhm_m", FEATURES, group, model, GRID, outer)
    return {"scheme": scheme, "seed": seed, "groups": frame[group].nunique(),
            **metrics(frame.resistivity_uOhm_m, prediction)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "Titanium_Simple_ML_Dataset.xlsx")
    parser.add_argument("--output", type=Path, default=Path("curve_validation_output"))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    frame = load_sets(args.data)["Resistivity"][0].copy()
    frame["composition"] = frame[ELEMENTS].round(3).astype(str).agg("|".join, axis=1)
    rows = Parallel(n_jobs=args.jobs)(delayed(one)(frame, s, seed)
                                       for s in ["Composition", "Alloy system"] for seed in range(1, 11))
    table = pd.DataFrame(rows).sort_values(["scheme", "seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "resistivity_levels.csv", index=False)
    print(table.groupby("scheme")[["groups", "R2", "RMSE"]].agg(["mean", "std", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
