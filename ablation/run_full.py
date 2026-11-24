"""
Final comparison: full system, one part removed at a time (main table of the
ablation chapter).

    python -m ablation.run_full            # offline: the "API" side is the rules engine
    python -m ablation.run_full --llm      # E0 with Gemini + rules (needs a paid key; cached)

All 150 scenarios of ablation/data/e2e_bank.json, every variant through the same
pipeline as the app:  tasks → ML schedule → API correction → fusion (0.8 / 0.2, adaptive).

  E0 Full system
  E1 Without PPO: every task at its habit hour (HH:00), rl_reward 0.8 hard-coded
     (what the app did before B8); overlaps are left to the fusion engine
  E2 Without habit model: activity-level preferred time for everyone
  E3 Without duration model: per-activity median duration (Jan–Apr training data)
  E4 Without LLM: rules only as the API side
  E5 ML only: PPO schedule, no API and no fusion (= w_ml 1, Ablation 5)

Offline (default) the API side of E0 is the rules engine too, so E0 = E4.
Metrics: deadline accuracy (TRUE durations), habit_dev (vs TRUE start), health
violations, overlaps, % tasks placed. Deterministic (no seeds needed: PPO inference
is greedy and the rules are fixed), checked by running twice.

Outputs: ablation/results/full.csv   variant, seed, metric, value
         ablation/results/full_table.md
"""
import argparse
import contextlib
import copy
import csv
import io
import logging
import os
from datetime import datetime, timedelta

import numpy as np

from ablation.common import set_seed, load_json, time_split
from ablation import e2e

RESULTS = os.path.join("ablation", "results")
DATA_PATH = "activities_of_daily_living_fixed.csv"
NAMES = {"E0": "Full system", "E1": "− PPO (tasks at habit hour, old app)",
         "E2": "− habit model (activity-level time)", "E3": "− duration model (per-activity median)",
         "E4": "− LLM (rules only)", "E5": "ML only (no API, no fusion)"}


def median_durations():
    """E3: per-activity median duration on the training months (Jan–Apr)."""
    from data_loader import load_dataset
    from duration_model import TARGET
    train, _, _ = time_split(load_dataset(DATA_PATH))
    return train.groupby("Activity Type")[TARGET].median().to_dict()


def habit_schedule(tasks):
    """E1: the pre-B8 ML schedule: each task at its habit hour, no PPO, no overlap check."""
    from config import ML_WEIGHTS
    out = []
    for t in tasks:
        start = datetime.combine(e2e.FIXED_DAY, datetime.min.time()) + timedelta(hours=t["Preferred_Hour"])
        end = start + timedelta(minutes=max(1, round(t["Estimated_Duration"])))
        deadline = datetime.combine(e2e.FIXED_DAY, datetime.min.time()) + timedelta(hours=t["Deadline_Hour"])
        if t["Task"] == "Sleeping" and deadline < start:      # overnight, as in scheduler.py
            deadline += timedelta(days=1)
        dl_ok = 1.0 if end <= deadline else 0.0
        prio = t["Priority"] / 4.0
        score = (ML_WEIGHTS["rl_reward"] * 0.8 + ML_WEIGHTS["habit_alignment"] * t["Confidence"] +
                 ML_WEIGHTS["deadline_satisfaction"] * dl_ok + ML_WEIGHTS["priority_score"] * prio)
        out.append({"task": t["Task"], "start_time": start, "end_time": end, "ml_score": score,
                    "rl_reward": 0.8, "habit_alignment": t["Confidence"], "deadline_satisfaction": dl_ok,
                    "priority_score": prio, "deadline_hour": t["Deadline_Hour"],
                    "status": "Scheduled Successfully" if dl_ok else "Scheduled but Missed Deadline"})
    return out


def run(bank, hm, dm, med, use_llm):
    import api_scheduler as A
    from fusion_engine import ScheduleFusionEngine

    def api(at, llm):
        s = A.APIScheduler(); s._today = e2e.FIXED_DAY
        return s.get_schedule(at) if llm else s._build_fallback_schedule(at)

    def pipeline(sc, tasks, ml, llm):
        at = e2e.api_tasks(tasks, ml)
        corrected = api(at, llm)
        eng = ScheduleFusionEngine()                     # app defaults: 0.8 / 0.2, adaptive
        conf = float(np.mean([t["Confidence"] for t in tasks]))
        return eng.combine_schedules(copy.deepcopy(ml), corrected, habit_confidence=conf)

    out = {v: [] for v in NAMES}
    for sc in bank:
        full = e2e.scenario_tasks(sc, hm, dm)
        ml = e2e.ml_schedule(full)
        out["E0"].append(e2e.evaluate(pipeline(sc, full, ml, use_llm), sc))
        out["E1"].append(e2e.evaluate(pipeline(sc, full, habit_schedule(full), use_llm), sc))
        t2 = e2e.scenario_tasks(sc, hm, dm, use_habit=False)
        out["E2"].append(e2e.evaluate(pipeline(sc, t2, e2e.ml_schedule(t2), use_llm), sc))
        t3 = e2e.scenario_tasks(sc, hm, dm, use_duration=False, duration_lookup=med)
        out["E3"].append(e2e.evaluate(pipeline(sc, t3, e2e.ml_schedule(t3), use_llm), sc))
        out["E4"].append(e2e.evaluate(pipeline(sc, full, ml, False), sc))
        out["E5"].append(e2e.evaluate(ml, sc))
    return out


def main(seed, use_llm):
    set_seed(seed)
    logging.disable(logging.WARNING)                     # fusion / fallback info + warnings
    import api_scheduler as A
    if use_llm:
        if not A.API_KEY:
            raise SystemExit("--llm needs GEMINI_API_KEY (task_scheduler/.env).")
        A._client = e2e.CachingClient(A.get_client())
    else:
        A.API_KEY = ""; A._client = None                 # nothing can reach Gemini

    bank = load_json(e2e.E2E_BANK)
    hm, dm = e2e.load_models()
    med = median_durations()
    with contextlib.redirect_stdout(io.StringIO()):
        res = run(bank, hm, dm, med, use_llm)

    table, rows = {}, []
    for v, r in res.items():
        m = e2e.summarise(r)
        m["placed"] = sum(x["n_placed"] for x in r) / sum(x["n_tasks"] for x in r)
        table[v] = m
        rows += [{"variant": v, "seed": seed, "metric": k, "value": x} for k, x in m.items()]

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "full.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "seed", "metric", "value"]); w.writeheader(); w.writerows(rows)

    ref = table["E0"]
    md = [f"# Final comparison: full system, one part removed ({len(bank)} scenarios, e2e_bank.json)", "",
          ("E0 uses Gemini + rules." if use_llm else
           "**Offline run:** the API side is the rules engine for every variant (free-tier Gemini quota), "
           "so E0 and E4 are the same pipeline."),
          "In brackets: change vs. E0. Deadline acc. uses TRUE durations; habit_dev = |start − true start|.", "",
          "| Variant | Deadline acc. | habit_dev (min) | Health violations | Overlaps | Tasks placed |",
          "|---|---|---|---|---|---|"]
    for v, m in table.items():
        d = lambda k, s: "" if v == "E0" else f" ({(m[k] - ref[k]) * s:+.1f})"
        md.append(f"| {v}: {NAMES[v]} | {m['deadline_acc']:.1%}{d('deadline_acc', 100)} | "
                  f"{m['habit_dev']:.1f}{d('habit_dev', 1)} | {m['violations']}{d('violations', 1)} | "
                  f"{m['overlaps']} | {m['placed']:.1%} |")
    path = os.path.join(RESULTS, "full_table.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\n→ {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", action="store_true", help="E0 with Gemini + rules (needs a key with quota)")
    a = ap.parse_args()
    main(a.seed, a.llm)
