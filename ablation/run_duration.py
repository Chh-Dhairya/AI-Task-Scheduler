"""
Ablation 1: duration model. Which inputs, which model type, which loss?

    python -m ablation.run_duration                # seeds 0 1 2

Split (B9): fit on Jan–Apr, early-stop on May, test on ALL June rows.
Every variant changes ONE thing from the production model (D3 = M4 = L1):
  1a features (XGBoost): D1 activity | D2 +hour | D3 +day-of-week (current) | D4 +person
  1b model (features D3): M1 per-activity median | M2 linear regression (one-hot activity)
                          | M3 random forest (100 trees) | M4 XGBoost (current)
  1c loss (XGBoost, D3):  L1 squared + p99 cap (current) | L2 squared, no cap | L3 absolute error
Random models use seeds 0,1,2 (XGBoost subsample, RF bootstrap); M1/M2 are deterministic.

Outputs: ablation/results/duration.csv            variant, seed, metric, value
         ablation/results/duration_per_activity.csv variant, seed, activity, mae
         ablation/figures/duration_mae_per_activity.png
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from ablation.common import set_seed, time_split
from data_loader import load_dataset
from duration_model import XGB_PARAMS, TARGET
from preprocess import DataPreprocessor

DATA_PATH = "activities_of_daily_living_fixed.csv"
RESULTS   = os.path.join("ablation", "results")
FIGURES   = os.path.join("ablation", "figures")

FEATS = {"D1": ["Activity_Encoded"],
         "D2": ["Activity_Encoded", "Hour"],
         "D3": ["Activity_Encoded", "Hour", "DayOfWeek"],
         "D4": ["Activity_Encoded", "Hour", "DayOfWeek", "Person_Encoded"]}
# variant → (model, features, cap, objective); production = ("xgb", "D3", True, squared)
VARIANTS = {
    "D1": ("xgb", "D1", True, "reg:squarederror"),
    "D2": ("xgb", "D2", True, "reg:squarederror"),
    "D3": ("xgb", "D3", True, "reg:squarederror"),          # current
    "D4": ("xgb", "D4", True, "reg:squarederror"),
    "M1": ("median", "D3", True, None),
    "M2": ("linear", "D3", True, None),
    "M3": ("rf", "D3", True, None),
    "M4": ("xgb", "D3", True, "reg:squarederror"),          # current (= D3)
    "L1": ("xgb", "D3", True, "reg:squarederror"),          # current (= D3)
    "L2": ("xgb", "D3", False, "reg:squarederror"),
    "L3": ("xgb", "D3", True, "reg:absoluteerror"),
}
DETERMINISTIC = {"M1", "M2"}


def fit_predict(variant, seed, train, val, test):
    model, fkey, use_cap, objective = VARIANTS[variant]
    F = FEATS[fkey]
    cap = float(train[TARGET].quantile(0.99)) if use_cap else None
    tr = train[train[TARGET] <= cap] if use_cap else train
    va = val[val[TARGET] <= cap] if use_cap else val
    hi = cap if use_cap else float(tr[TARGET].max())          # prediction upper clip

    if model == "median":
        med = tr.groupby("Activity_Encoded")[TARGET].median()
        pred = test["Activity_Encoded"].map(med).fillna(tr[TARGET].median()).to_numpy()
    elif model == "linear":
        def design(d):
            oh = pd.get_dummies(d["Activity_Encoded"].astype("category").cat.set_categories(
                sorted(train["Activity_Encoded"].unique())), prefix="act", dtype=float)
            return pd.concat([oh, d[["Hour", "DayOfWeek"]].astype(float)], axis=1)
        pred = LinearRegression().fit(design(tr), tr[TARGET]).predict(design(test))
    elif model == "rf":
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=2, random_state=seed)
        pred = rf.fit(tr[F], tr[TARGET]).predict(test[F])
    else:
        params = {**XGB_PARAMS, "random_state": seed, "objective": objective}
        xgb = XGBRegressor(**params)
        xgb.fit(tr[F], tr[TARGET], eval_set=[(va[F], va[TARGET])], verbose=False)
        pred = xgb.predict(test[F])
    return np.clip(pred, 1.0, hi)


def metrics(y, p):
    err = np.abs(y - p)
    return {"mae": float(err.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "within_15": float((err <= 15).mean())}


def main(seeds):
    set_seed(seeds[0])
    df = DataPreprocessor().fit_transform(load_dataset(DATA_PATH))
    train, val, test = time_split(df)
    act_name =(df.drop_duplicates("Activity_Encoded").set_index("Activity_Encoded")["Activity Type"].to_dict())
    y = test[TARGET].to_numpy(dtype=float)
    rows, per_act = [], []
    for v in VARIANTS:
        for seed in ([0] if v in DETERMINISTIC else seeds):
            set_seed(seed)
            p = fit_predict(v, seed, train, val, test)
            m = metrics(y, p)
            rows += [{"variant": v, "seed": seed, "metric": k, "value": x} for k, x in m.items()]
            err = pd.Series(np.abs(y - p), index=test.index)
            for a, e in err.groupby(test["Activity_Encoded"]).mean().items():
                per_act.append({"variant": v, "seed": seed, "activity": act_name[a], "mae": float(e)})
            print(f"  {v} seed {seed}: MAE {m['mae']:6.2f} | RMSE {m['rmse']:6.2f} | ±15 min {m['within_15']:.1%}", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    _write(os.path.join(RESULTS, "duration.csv"), rows, ["variant", "seed", "metric", "value"])
    _write(os.path.join(RESULTS, "duration_per_activity.csv"), per_act, ["variant", "seed", "activity", "mae"])
    summary(rows)
    plot(per_act)


def summary(rows):
    print(f"\n{'variant':8} {'n':>2} {'MAE':>15} {'RMSE':>15} {'within ±15':>12}")
    for v in VARIANTS:
        g = lambda m: [r["value"] for r in rows if r["variant"] == v and r["metric"] == m]
        mae, rmse, w = g("mae"), g("rmse"), g("within_15")
        print(f"{v:8} {len(mae):>2} {np.mean(mae):7.2f} ± {np.std(mae):4.2f} {np.mean(rmse):7.2f} ± {np.std(rmse):4.2f} {np.mean(w):11.1%}")


def plot(per_act):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.DataFrame(per_act).groupby(["activity", "variant"])["mae"].mean().unstack()
    show = ["M1", "M2", "M3", "M4"]
    acts = d.sort_values("M4").index
    x = np.arange(len(acts)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = {"M1": "Per-activity median", "M2": "Linear regression", "M3": "Random forest", "M4": "XGBoost (current)"}
    for i, v in enumerate(show):
        ax.bar(x + (i - 1.5) * w, d.loc[acts, v], w, label=labels[v])
    ax.set_xticks(x); ax.set_xticklabels(acts, rotation=30, ha="right")
    ax.set_ylabel("MAE on June test (minutes)"); ax.set_title("Ablation 1: duration MAE per activity")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, "duration_mae_per_activity.png"); fig.savefig(path, dpi=150)
    print(f"\nFigure → {path}")


def _write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    main(ap.parse_args().seeds)
