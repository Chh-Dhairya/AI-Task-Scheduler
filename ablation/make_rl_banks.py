"""
B5: build the fixed PPO scenario banks. Run ONCE, after B4 (priority 1-4):

    python -m ablation.make_rl_banks

  rl_val.json  : 500 scenarios  (seed 1001), used to pick the best checkpoint
  rl_test.json : 2000 scenarios (seed 1002), used ONLY for reported numbers

The seeds are fixed on purpose so the banks are always the same. Existing files
are never overwritten (CLAUDE.md: never regenerate ablation/data/ once results exist).
"""
import os

import numpy as np

from ablation.common import save_json
from rl_environment import SchedulingEnv

OUT = os.path.join(os.path.dirname(__file__), "data")
BANKS = [("rl_val.json", 1001, 500), ("rl_test.json", 1002, 2000)]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    env = SchedulingEnv(max_tasks=10)
    for name, seed, n in BANKS:
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            print(f"{path} already exists, not regenerating.")
            continue
        np.random.seed(seed)
        save_json([env._random_tasks() for _ in range(n)], path)
        print(f"Saved {n} scenarios → {path}  (seed {seed})")
