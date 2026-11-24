"""
Ablation 2: habit model. Does personalisation help, does the 3-level fallback
help, and are 5 time buckets too coarse?

    python -m ablation.run_habit

Fit on Jan–May, test on June (B9). 50 persons are held out of fitting entirely
(held_out_persons, seed 7) for the cold-start test.
  H1 fallback levels (5-bucket output):
     H1-a  level 2 only (per activity, no personalisation)
     H1-b  level 1 + level 3 (skip the activity level)
     H1-c  level 1 + 2 + 3 (current)
  H2 time resolution (levels 1+2+3):
     H2-a  5 buckets → fixed hour 7/11/15/19/22 (current)
     H2-b  most common hour (24 bins)
     H2-c  median start time in minutes (circular, so Sleeping around midnight works)
  H3 recency (current model, optional):
     H3-a  all history (Jan–May)      H3-b  last 30 days only
Groups: "normal" = June rows of persons seen in fitting, "cold" = held-out persons.
Metrics: bucket_acc, start_err (minutes, circular), within_30, within_60.
The habit model is deterministic, so there is one run (seed 0).

Outputs: ablation/results/habit.csv              variant, seed, metric, value
         ablation/results/habit_per_activity.csv variant, activity, start_err
         ablation/figures/habit_start_error.png
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd

from ablation.common import set_seed, time_split, held_out_persons
from data_loader import load_dataset
from preprocess import DataPreprocessor, BUCKET_TO_HOUR, get_time_bucket

DATA_PATH = "activities_of_daily_living_fixed.csv"
RESULTS   = os.path.join("ablation", "results")
FIGURES   = os.path.join("ablation", "figures")
DAY = 1440
KEY = ["Person_Encoded", "Activity_Encoded"]


def minute_of_day(df):
    return (df["Start"].dt.hour * 60 + df["Start"].dt.minute).to_numpy(dtype=float)


def circ_err(pred, true):
    """|pred - true| in minutes on a 24 h clock (23:30 vs 00:30 = 60 min)."""
    d = np.abs(np.asarray(pred, float) - np.asarray(true, float)) % DAY
    return np.minimum(d, DAY - d)


def circ_median(m):
    """Median of times of day: rotate so the circular mean sits at 12:00, take the median, rotate back."""
    a = np.asarray(m, float) / DAY * 2 * np.pi
    centre = (np.arctan2(np.sin(a).sum(), np.cos(a).sum()) / (2 * np.pi) * DAY) % DAY
    shifted = (np.asarray(m, float) - centre + DAY / 2) % DAY
    return (np.median(shifted) + centre - DAY / 2) % DAY


class LevelTable:
    """Per-(person, activity), per-activity and global values of one statistic (3 levels)."""
    def __init__(self, fit, value_fn):
        self.l1 = fit.groupby(KEY).apply(value_fn, include_groups=False).to_dict()
        self.l2 = fit.groupby("Activity_Encoded").apply(value_fn, include_groups=False).to_dict()
        self.l3 = value_fn(fit)

    def predict(self, rows, levels=(1, 2, 3)):
        out = []
        for p, a in zip(rows["Person_Encoded"], rows["Activity_Encoded"]):
            if 1 in levels and (p, a) in self.l1:
                out.append(self.l1[(p, a)])
            elif 2 in levels and a in self.l2:
                out.append(self.l2[a])
            else:
                out.append(self.l3)
        return np.array(out, dtype=float)


mode_bucket = lambda g: int(g["TimeBucket"].value_counts().idxmax())   # same rule as HabitModel
mode_hour   = lambda g: int(g["Hour"].value_counts().idxmax())
med_minute  = lambda g: float(circ_median(minute_of_day(g)))


def score(pred_minutes, rows, pred_bucket=None):
    true_m = minute_of_day(rows)
    if pred_bucket is None:
        pred_bucket = np.array([get_time_bucket(int(m // 60) % 24) for m in pred_minutes])
    e = circ_err(pred_minutes, true_m)
    return {"bucket_acc": float((pred_bucket == rows["TimeBucket"].to_numpy()).mean()),
            "start_err": float(e.mean()), "within_30": float((e <= 30).mean()),
            "within_60": float((e <= 60).mean()), "n": int(len(rows))}


def main(seed):
    set_seed(seed)
    df = DataPreprocessor().fit_transform(load_dataset(DATA_PATH))
    train, val, test = time_split(df)
    fit_all = pd.concat([train, val])
    held = held_out_persons(df)
    fit = fit_all[~fit_all["Person ID"].isin(held)]
    groups = {"normal": test[~test["Person ID"].isin(held)], "cold": test[test["Person ID"].isin(held)]}
    print(f"fit rows {len(fit)} (Jan–May, {fit['Person ID'].nunique()} persons) | June test: "
          f"normal {len(groups['normal'])}, cold start {len(groups['cold'])} ({len(held)} persons)")

    buckets = LevelTable(fit, mode_bucket)
    hours   = LevelTable(fit, mode_hour)
    medians = LevelTable(fit, med_minute)
    b2m = lambda b: np.array([BUCKET_TO_HOUR[int(x)] * 60 for x in b], float)

    results, per_act = {}, []
    def add(variant, group, rows, pm, pb=None):
        results[(variant, group)] = score(pm, rows, pb)
        e = pd.Series(circ_err(pm, minute_of_day(rows)), index=rows.index)
        if group == "normal":
            name = rows.drop_duplicates("Activity_Encoded").set_index("Activity_Encoded")["Activity Type"]
            for a, x in e.groupby(rows["Activity_Encoded"]).mean().items():
                per_act.append({"variant": variant, "activity": name[a], "start_err": float(x)})

    for g, rows in groups.items():
        for v, lv in [("H1-a", (2,)), ("H1-b", (1, 3)), ("H1-c", (1, 2, 3))]:
            b = buckets.predict(rows, lv).astype(int); add(v, g, rows, b2m(b), b)
        b = buckets.predict(rows).astype(int); add("H2-a", g, rows, b2m(b), b)
        add("H2-b", g, rows, hours.predict(rows) * 60)
        add("H2-c", g, rows, medians.predict(rows))

    # H3 recency: current model (buckets, 3 levels) on all history vs the last 30 days
    recent = fit[fit["Start"] >= pd.Timestamp("2023-06-01") - pd.Timedelta(days=30)]
    rb = LevelTable(recent, mode_bucket)
    for g, rows in groups.items():
        b = buckets.predict(rows).astype(int); add("H3-a", g, rows, b2m(b), b)
        b = rb.predict(rows).astype(int);      add("H3-b", g, rows, b2m(b), b)
    print(f"H3-b uses {len(recent)} rows from {recent['Start'].min():%Y-%m-%d} to {recent['Start'].max():%Y-%m-%d}")

    rows = [{"variant": v, "seed": seed, "metric": f"{g}_{k}", "value": x}
            for (v, g), m in results.items() for k, x in m.items()]
    os.makedirs(RESULTS, exist_ok=True)
    _write(os.path.join(RESULTS, "habit.csv"), rows, ["variant", "seed", "metric", "value"])
    _write(os.path.join(RESULTS, "habit_per_activity.csv"), per_act, ["variant", "activity", "start_err"])

    print(f"\n{'variant':7} {'group':7} {'bucket acc':>10} {'start err':>10} {'±30 min':>8} {'±60 min':>8}")
    for (v, g), m in results.items():
        print(f"{v:7} {g:7} {m['bucket_acc']:10.1%} {m['start_err']:8.1f} m {m['within_30']:8.1%} {m['within_60']:8.1%}")
    plot(per_act)


def plot(per_act):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.DataFrame(per_act)
    d = d[d["variant"].isin(["H2-a", "H2-b", "H2-c"])].pivot(index="activity", columns="variant", values="start_err")
    d = d.sort_values("H2-a")
    x = np.arange(len(d)); w = 0.27
    labels = {"H2-a": "5 buckets → fixed hour (current)", "H2-b": "Most common hour", "H2-c": "Median start time"}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, v in enumerate(["H2-a", "H2-b", "H2-c"]):
        ax.bar(x + (i - 1) * w, d[v], w, label=labels[v])
    ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=30, ha="right")
    ax.set_ylabel("Start-time error, June (minutes)"); ax.set_title("Ablation 2: habit start-time error per activity (normal users)")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, "habit_start_error.png"); fig.savefig(path, dpi=150)
    print(f"\nFigure → {path}")


def _write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)   # deterministic model; kept per repo rule
    main(ap.parse_args().seed)
