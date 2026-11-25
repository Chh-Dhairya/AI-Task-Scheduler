"""
Ablation 5: fusion engine. How should ML and API be weighted, and does the
adaptive weighting help?

    python -m ablation.run_fusion [--api A3]

Uses the SAVED ML schedules and API schedules from Ablation 4 (run_llm.py), so
every variant fuses exactly the same inputs (first 100 e2e scenarios). No API calls.
  5a weight sweep (adaptive off, weighted rule): w_ml ∈ {1.0, 0.8, 0.5, 0.2, 0.0}
  5b F1 adaptive off, w_ml 0.8 | F2 adaptive on (thresholds 0.5 / 0.8, app default)
  5c D0 old rule (higher raw score picks the time) | D1 weighted rule (A6)
Metrics: habit_dev (vs TRUE start), health violations, deadline accuracy (TRUE
durations), overlaps (must be 0), % of tasks whose time came from the API
(pct_from_api; includes API times identical to the ML time) and % of tasks whose
start differs from the ML schedule (pct_time_changed).
final_score is NOT used to compare variants (it is the weighted quantity itself).

Outputs: ablation/results/fusion.csv   variant, seed, metric, value
         ablation/figures/fusion_tradeoff.png
"""
import argparse
import copy
import csv
import logging
import os

import numpy as np

from ablation.common import set_seed, load_json
from ablation import e2e
from ablation.run_llm import ML_PATH, API_PATH

RESULTS = os.path.join("ablation", "results")
FIGURES = os.path.join("ablation", "figures")
SWEEP = [1.0, 0.8, 0.5, 0.2, 0.0]


def variants():
    v = {f"W{w:.1f}": dict(model_weight=w, api_weight=1 - w, adaptive=False) for w in SWEEP}
    v["F1"] = dict(model_weight=0.8, api_weight=0.2, adaptive=False)
    v["F2"] = dict(model_weight=0.8, api_weight=0.2, adaptive=True)
    v["D0"] = dict(model_weight=0.8, api_weight=0.2, adaptive=False, decision_rule="raw")
    v["D1"] = dict(model_weight=0.8, api_weight=0.2, adaptive=False, decision_rule="weighted")
    return v


def main(api_variant, seed):
    set_seed(seed)
    logging.getLogger().setLevel(logging.ERROR)
    from fusion_engine import ScheduleFusionEngine
    bank = load_json(e2e.E2E_BANK)
    saved = load_json(ML_PATH)
    api_all = load_json(API_PATH)
    if api_variant not in api_all:
        raise SystemExit(f"{api_variant} not in {API_PATH} (have {sorted(api_all)})")
    n = len(api_all[api_variant])
    ML  = [e2e.load_schedule(s["ml"]) for s in saved[:n]]
    API = [e2e.load_schedule(s) for s in api_all[api_variant]]
    conf = [float(np.mean([t["Confidence"] for t in s["tasks"]])) for s in saved[:n]]
    print(f"Fusing {n} scenarios: ML (PPO) + API schedules from Ablation 4 variant {api_variant}")

    rows, table = [], {}
    for name, kw in variants().items():
        eng = ScheduleFusionEngine(**kw)
        res, from_api, changed, total = [], 0, 0, 0
        for i in range(n):
            fused = eng.combine_schedules(copy.deepcopy(ML[i]), copy.deepcopy(API[i]), habit_confidence=conf[i])
            res.append(e2e.evaluate(fused, bank[i]))
            ml_start = {t["task"]: t["start_time"] for t in ML[i]}
            from_api += sum(f["source"] == "API" for f in fused); total += len(fused)
            changed += sum(f["start_time"] != ml_start.get(f["task"]) for f in fused)
        m = e2e.summarise(res)
        m["pct_from_api"] = from_api / max(total, 1)
        m["pct_time_changed"] = changed / max(total, 1)
        table[name] = m
        rows += [{"variant": name, "seed": seed, "metric": k, "value": x} for k, x in m.items()]

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "fusion.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "seed", "metric", "value"]); w.writeheader(); w.writerows(rows)
    print(f"\n{'variant':7} {'from API':>9} {'changed':>8} {'habit_dev':>10} {'violations':>10} {'deadline':>9} {'overlaps':>8}")
    for k, m in table.items():
        print(f"{k:7} {m['pct_from_api']:9.1%} {m['pct_time_changed']:8.1%} {m['habit_dev']:8.1f} m {m['violations']:10d} {m['deadline_acc']:9.1%} {m['overlaps']:8d}")
    plot(table, api_variant)


def plot(table, api_variant):
    """5a: each metric against the ML weight (the adaptive F2 point drawn at its base weight 0.8)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("habit_dev", "habit_dev (min, lower = closer to habits)", 1),
              ("deadline_acc", "Deadline accuracy (%)", 100),
              ("violations", "Health violations (100 days)", 1),
              ("pct_time_changed", "Tasks whose time changed vs ML (%)", 100)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (m, label, k) in zip(axes, panels):
        ax.plot(SWEEP, [table[f"W{w:.1f}"][m] * k for w in SWEEP], "-o", color="tab:blue", label="fixed weights")
        ax.plot([0.8], [table["F2"][m] * k], "s", color="tab:orange", ms=8, mfc="none", mew=2, label="adaptive (F2)")
        ax.set_xlabel("ML weight w_ml (API weight = 1 − w_ml)"); ax.set_ylabel(label, fontsize=9)
        ax.invert_xaxis(); ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Ablation 5a: ML/API weight sweep (API schedules = Ablation 4 variant {api_variant}, 100 days)", fontsize=10)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, "fusion_tradeoff.png"); fig.savefig(path, dpi=150)
    print(f"\nFigure → {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="A3", help="Ablation 4 variant whose API schedules are fused")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.api, a.seed)
