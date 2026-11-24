"""
Ablation 4: LLM correction (Gemini + rules). Does the LLM add anything over
simple rules, and does it need to see the ML schedule?

    python -m ablation.run_llm                 # first 100 scenarios of e2e_bank.json

The ML (PPO) schedule of every scenario is computed ONCE and saved, so every
variant corrects the same input. Gemini replies are cached in ablation/cache/
(re-runs cost nothing). Needs GEMINI_API_KEY (task_scheduler/.env).

  4a source:  A0 no correction (ML only) | A1 rules only | A2 Gemini only (no rules
              fallback; a task Gemini drops keeps its ML time) | A3 Gemini + rules (current)
  4b prompt:  P0 without ML times (old prompt) | P1 with ML times (= A3)
  4c consistency (30 scenarios × 3 repeats): C0 temperature 0 | C1 temperature 0.7

    python -m ablation.run_llm --no-llm        # offline: only A0 and A1, no Gemini calls
  Used for the report because the free-tier key allows only 20 Gemini requests per
  day (the full run needs ~380). A2/A3, 4b and 4c are then NOT evaluated.

Outputs: ablation/results/llm.csv                 variant, seed, metric, value
         ablation/results/e2e_ml_schedules.json   ML schedule per e2e scenario (all 150)
         ablation/results/llm_api_schedules.json  corrected schedules per variant (for Ablation 5)
"""
import argparse
import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ablation.common import set_seed, load_json, save_json
from ablation import e2e

RESULTS  = os.path.join("ablation", "results")
ML_PATH  = os.path.join(RESULTS, "e2e_ml_schedules.json")
API_PATH = os.path.join(RESULTS, "llm_api_schedules.json")
N_MAIN, N_CONS, REPEATS = 100, 30, 3


def ml_schedules(bank, hm, dm):
    """PPO schedule for every e2e scenario, computed once and saved."""
    if os.path.exists(ML_PATH):
        saved = load_json(ML_PATH)
        return [e2e.load_schedule(s["ml"]) for s in saved], [s["tasks"] for s in saved]
    out, tasks_all = [], []
    for sc in bank:
        tasks = e2e.scenario_tasks(sc, hm, dm)
        out.append(e2e.ml_schedule(tasks)); tasks_all.append(tasks)
    save_json([{"tasks": t, "ml": e2e.dump_schedule(m)} for t, m in zip(tasks_all, out)], ML_PATH)
    print(f"ML schedules computed for {len(out)} scenarios → {ML_PATH}")
    return out, tasks_all


def gemini_only(api, at, ml):
    """A2: Gemini's reply as is; a task it drops (or a failed call) keeps its ML time."""
    from datetime import timedelta
    res = {r["task"]: r for r in api._call_gemini(at)}
    out = []
    for t, m in zip(at, ml):
        r = res.get(t["name"])
        if r is None:
            out.append({**m, "source": "ml"})
        else:
            out.append({**r, "end_time": r["start_time"] + timedelta(minutes=t["duration"])})
    return out


def main(workers, seed, no_llm=False):
    set_seed(seed)
    logging.getLogger().setLevel(logging.ERROR)
    import api_scheduler as A
    if no_llm:
        A.API_KEY = ""; A._client = None          # make sure nothing can reach Gemini
    elif not A.API_KEY:
        raise SystemExit("GEMINI_API_KEY is not set (put it in task_scheduler/.env), or use --no-llm.")
    else:
        cc = e2e.CachingClient(A.get_client()); A._client = cc

    bank = load_json(e2e.E2E_BANK)
    hm, dm = e2e.load_models()
    ML, TASKS = ml_schedules(bank, hm, dm)

    def new_api(**kw):
        api = A.APIScheduler(**kw); api._today = e2e.FIXED_DAY
        return api

    def one(i):
        ml = ML[i]; at = e2e.api_tasks(TASKS[i], ml)
        a1 = new_api()._build_fallback_schedule(at)
        if no_llm:
            return {"A0": ml, "A1": a1}
        a3 = new_api(include_ml_times=True).get_schedule(at)          # Gemini call (P1)
        a2 = gemini_only(new_api(include_ml_times=True), at, ml)      # same reply, from cache
        p0 = new_api(include_ml_times=False).get_schedule(at)         # Gemini call (old prompt)
        return {"A0": ml, "A1": a1, "A2": a2, "A3": a3, "P0": p0, "P1": a3}

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        per = list(pool.map(one, range(N_MAIN)))
    VARS = ["A0", "A1"] if no_llm else ["A0", "A1", "A2", "A3", "P0", "P1"]
    print(f"4a/4b: {N_MAIN} scenarios in {(time.time() - t0) / 60:.1f} min" + ("" if no_llm else f" | cache {cc.stats}"))

    rows = []
    for v in VARS:
        res = [e2e.evaluate(per[i][v], bank[i], ML[i]) for i in range(N_MAIN)]
        m = e2e.summarise(res)
        items = [x for i in range(N_MAIN) for x in per[i][v]]
        if v in ("A2", "A3", "P0", "P1"):
            m["fallback_rate"] = float(np.mean([x.get("source") != "gemini" for x in items]))
            m["full_fallback_days"] = int(sum(all(x.get("source") != "gemini" for x in per[i][v]) for i in range(N_MAIN)))
        rows += [{"variant": v, "seed": seed, "metric": k, "value": x} for k, x in m.items()]
    save_json({v: [e2e.dump_schedule(per[i][v]) for i in range(N_MAIN)] for v in VARS if v != "P1"}, API_PATH)
    if no_llm:
        write_and_print(rows, VARS, [])
        print("\nOffline run (--no-llm): A2, A3, P0/P1 and C0/C1 were NOT evaluated.")
        return

    # 4c consistency: same prompt, 3 repeats, temperature 0 vs 0.7
    def cons(args):
        i, temp = args
        ml = ML[i]; at = e2e.api_tasks(TASKS[i], ml)
        reps = []
        for r in range(REPEATS):
            cc.set_repeat(r)
            reps.append(new_api(temperature=temp)._call_gemini(at))
        cc.set_repeat(0)
        return reps
    t0 = time.time()
    for name, temp in [("C0", 0.0), ("C1", 0.7)]:
        with ThreadPoolExecutor(workers) as pool:
            reps = list(pool.map(cons, [(i, temp) for i in range(N_CONS)]))
        valid = np.mean([len(r) > 0 for rr in reps for r in rr])
        stds = []
        for rr in reps:
            starts = {}
            for r in rr:
                for x in r:
                    starts.setdefault(x["task"], []).append(x["start_time"].hour * 60 + x["start_time"].minute)
            stds += [np.std(v) for v in starts.values() if len(v) == REPEATS]
        changed = np.mean([s > 0 for s in stds]) if stds else float("nan")
        for k, x in {"valid_json": float(valid), "start_std_min": float(np.mean(stds)),
                     "tasks_with_any_change": float(changed)}.items():
            rows.append({"variant": name, "seed": seed, "metric": k, "value": x})
    print(f"4c: {N_CONS} scenarios × {REPEATS} repeats × 2 temperatures in {(time.time() - t0) / 60:.1f} min | cache {cc.stats}")

    write_and_print(rows, VARS, ["C0", "C1"])


def write_and_print(rows, VARS, CONS):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "llm.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "seed", "metric", "value"]); w.writeheader(); w.writerows(rows)

    print(f"\n{'variant':7} {'deadline':>9} {'habit_dev':>10} {'violations':>10} {'shift':>8} {'overlaps':>8} {'fallback':>9}")
    for v in VARS:
        g = {r["metric"]: r["value"] for r in rows if r["variant"] == v}
        fb = f"{g['fallback_rate']:8.1%}" if "fallback_rate" in g else "       –"
        print(f"{v:7} {g['deadline_acc']:9.1%} {g['habit_dev']:8.1f} m {g['violations']:10d} {g['shift_min']:6.1f} m {g['overlaps']:8d} {fb}")
    for v in CONS:
        g = {r["metric"]: r["value"] for r in rows if r["variant"] == v}
        print(f"{v}: valid JSON {g['valid_json']:.1%} | start std across repeats {g['start_std_min']:.1f} min | "
              f"tasks whose time changed {g['tasks_with_any_change']:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-llm", action="store_true", help="offline: only A0 (ML) and A1 (rules), no Gemini calls")
    a = ap.parse_args()
    main(a.workers, a.seed, a.no_llm)
