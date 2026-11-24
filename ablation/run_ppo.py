"""
Ablation 3: train + evaluate ONE PPO variant with one seed.

    python -m ablation.run_ppo --variant R1 --seed 0            # 80k episodes (B6)

Writes  <out>/<variant>_s<seed>.json  (val/test metrics + learning curve)
and     <out>/<variant>_s<seed>.pth   (best checkpoint, picked on rl_val.json).
"""
import argparse
import json
import os
import time

from ablation.common import set_seed

# One change per variant; everything else = the final model (B6: Table 4.1, 80k).
# R0 = S0 = T0 is the same run (full model).
VARIANTS = {
    "R0": {},                                                             # full model
    "R1": {"env": {"reward_terms": ("buffer", "priority")}},              # − habit bonus
    "R2": {"env": {"reward_terms": ("habit", "priority")}},               # − buffer bonus
    "R3": {"env": {"reward_terms": ("habit", "buffer")}},                 # − priority bonus
    "S1": {"env": {"use_mode_flag": False}},                              # − mode flag
    "S2": {"env": {"use_pref": False}},                                   # − preferred hour
    "T1": {"agent": {"buggy_gae": True}},                                 # old buggy GAE (B1)
    "T2": {"curriculum": False},                                          # no curriculum
    "T3": {"agent": {"norm_mode": "none"}},                               # no reward normalisation
}
EPISODES = 80_000
OUT = os.path.join("ablation", "results", "ppo_runs")


def run(variant, seed, episodes=EPISODES, out=OUT):
    from train_rl_scheduler import train          # import after set_seed is defined
    os.makedirs(out, exist_ok=True)
    v = VARIANTS[variant]
    tag = f"{variant}_s{seed}"
    t0 = time.time()
    agent = train(episodes=episodes, seed=seed, save_path=os.path.join(out, f"{tag}.pth"),
                  env_kwargs=v.get("env"), agent_kwargs=v.get("agent"),
                  curriculum=v.get("curriculum", True))
    res = {"variant": variant, "seed": seed, "episodes": episodes, "config": v,
           "updates": agent._updates, "minutes": round((time.time() - t0) / 60, 2),
           "val": agent.val_metrics, "test": agent.test_metrics, "history": agent.history}
    with open(os.path.join(out, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"RESULT {tag}: test deadline {res['test']['deadline_acc']:.2%} | "
          f"habit_dev {res['test']['habit_dev']:.1f} min | {res['minutes']} min")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    set_seed(a.seed)                   # train() seeds again; kept per the repo rule
    run(a.variant, a.seed, a.episodes, a.out)
