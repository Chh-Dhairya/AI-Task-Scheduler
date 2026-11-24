"""
Ablation 3: all PPO variants × seeds, plus the rule baselines (3a).

    python -m ablation.run_ppo_all                  # 9 variants × seeds 0,1,2, 4 at a time
    python -m ablation.run_ppo_all --baselines-only # 3a only (seconds)

Runs that already have a JSON are skipped, so the batch can be resumed.
Outputs (ablation/results/):
  ppo_baselines.csv  variant, seed, metric, value   (Random / Habit-sort / EDF on rl_test)
  ppo.csv            variant, seed, metric, value   (val_* and test_* per run)
  ppo_curves.csv     variant, seed, episode, metric, value   (learning curves, val)
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ablation.common import set_seed, load_json
from ablation.run_ppo import VARIANTS, EPISODES, OUT

RESULTS = os.path.join("ablation", "results")


class RulePolicy:
    """Stand-in agent for evaluate(): picks the next task by a fixed rule."""
    def __init__(self, env, rule, seed=0):
        self.env, self.rule, self.eps = env, rule, 0.0
        self.rng = np.random.RandomState(seed)
        self.net = torch.nn.Identity()          # evaluate() calls net.eval()/train()

    def act(self, state, valid):
        T = self.env.tasks
        if self.rule == "random":
            return int(self.rng.choice(valid))
        key = "Preferred_Hour" if self.rule == "habit_sort" else "Deadline_Hour"
        return min(valid, key=lambda i: T[i][key])


def run_baselines(seeds):
    from rl_environment import SchedulingEnv
    from train_rl_scheduler import evaluate, TEST_PATH
    test, env, rows = load_json(TEST_PATH), SchedulingEnv(max_tasks=10), []
    for rule in ["random", "habit_sort", "edf"]:
        for seed in (seeds if rule == "random" else [0]):     # only Random is stochastic
            set_seed(seed)
            m = evaluate(RulePolicy(env, rule, seed), env, test)
            rows += [{"variant": rule, "seed": seed, "metric": f"test_{k}", "value": v} for k, v in m.items()]
            print(f"  {rule:10} seed {seed}: deadline {m['deadline_acc']:.2%} | habit_dev {m['habit_dev']:.1f} min")
    _write(os.path.join(RESULTS, "ppo_baselines.csv"), rows, ["variant", "seed", "metric", "value"])


def _launch(job):
    variant, seed, episodes, out = job
    tag = f"{variant}_s{seed}"
    if os.path.exists(os.path.join(out, f"{tag}.json")):
        return f"{tag}: exists, skipped"
    with open(os.path.join(out, f"{tag}.log"), "w") as log:
        rc = subprocess.call([sys.executable, "-m", "ablation.run_ppo", "--variant", variant,
                              "--seed", str(seed), "--episodes", str(episodes), "--out", out],
                             stdout=log, stderr=subprocess.STDOUT)
    return f"{tag}: {'done' if rc == 0 else f'FAILED (exit {rc}), see {tag}.log'}"


def collect(out):
    rows, curves = [], []
    for v in VARIANTS:
        for f in sorted(os.listdir(out)):
            if not (f.startswith(f"{v}_s") and f.endswith(".json")):
                continue
            r = json.load(open(os.path.join(out, f)))
            for split in ("val", "test"):
                rows += [{"variant": v, "seed": r["seed"], "metric": f"{split}_{k}", "value": x}
                         for k, x in r[split].items()]
            curves += [{"variant": v, "seed": r["seed"], "episode": h["episode"], "metric": k, "value": x}
                       for h in r["history"] for k, x in h.items() if k != "episode"]
    _write(os.path.join(RESULTS, "ppo.csv"), rows, ["variant", "seed", "metric", "value"])
    _write(os.path.join(RESULTS, "ppo_curves.csv"), curves, ["variant", "seed", "episode", "metric", "value"])
    # summary: mean ± std over seeds
    print(f"\n{'variant':8} {'n':>2} {'test deadline':>16} {'test habit_dev':>17} {'test habit_acc':>15}")
    for v in VARIANTS:
        get = lambda m: [r["value"] for r in rows if r["variant"] == v and r["metric"] == m]
        d, h, a = get("test_deadline_acc"), get("test_habit_dev"), get("test_habit_acc")
        if d:
            print(f"{v:8} {len(d):>2} {np.mean(d):8.2%} ± {np.std(d):5.2%} {np.mean(h):9.1f} ± {np.std(h):4.1f} "
                  f"{np.mean(a):8.2%} ± {np.std(a):4.2%}")


def _write(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--workers", type=int, default=4)     # 4 × 2 torch threads = 8 cores
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--baselines-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("3a. Rule baselines on rl_test.json")
    run_baselines(a.seeds)
    if a.baselines_only:
        sys.exit(0)

    jobs = [(v, s, a.episodes, a.out) for s in a.seeds for v in a.variants]
    print(f"\n{len(jobs)} PPO runs, {a.workers} at a time, {a.episodes} episodes each")
    with ThreadPoolExecutor(a.workers) as pool:
        for msg in pool.map(_launch, jobs):
            print("  " + msg, flush=True)
    collect(a.out)
