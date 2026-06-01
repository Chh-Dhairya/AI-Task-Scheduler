import itertools
import numpy as np
import torch
import csv
import random
import os
from datetime import datetime

from rl_environment import SchedulingEnv
from train_rl_scheduler import CurriculumEnv, evaluate
from rl_agent import PPOAgent

# ── GPU SETUP ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# ── GRID SEARCH SPACE ────────────────────────────────────────────────────────
# Values chosen to push deadline_acc UP (needs long-term planning)
# and keep habit_acc stable/improving
param_grid = {
    # Lower LR → stable learning for deadline patterns
    "lr":         [3e-5, 1e-4, 2e-4, 4e-4, 7e-4, 1e-3],

    # Higher gamma → agent plans further ahead (key for deadlines)
    "gamma":      [0.95, 0.97, 0.98, 0.99, 0.995, 0.999],

    # Higher gae_lambda → better credit assignment over long episodes
    "gae_lambda": [0.88, 0.92, 0.94, 0.96, 0.98, 0.99],

    # Keep clip_eps moderate → avoid too large policy updates
    "clip_eps":   [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],

    # More epochs → squeeze more learning from each batch
    "ppo_epochs": [4, 8, 12, 16, 20, 24],

    # Larger batches → more stable gradient estimates
    "batch_size": [32, 64, 128, 256, 512, 1024],

    # Lower entropy → more exploitation (agent already explores well at 75%)
    "ent_coef":   [0.002, 0.005, 0.01, 0.03, 0.06, 0.1],
}

# ── RANDOM SAMPLING (exhaustive = 6^7 = 279,936 — too slow!) ─────────────────
# We randomly sample N combos instead. 300 gives good coverage on RTX 3050
N_SAMPLES = 300
random.seed(42)

keys = list(param_grid.keys())
all_combinations = list(itertools.product(*param_grid.values()))
sampled = random.sample(all_combinations, min(N_SAMPLES, len(all_combinations)))

print(f"Total possible combinations : {len(all_combinations):,}")
print(f"Randomly sampling           : {len(sampled)}")
print(f"Estimated time (RTX 3050)   : ~{len(sampled) * 8 // 60}–{len(sampled) * 12 // 60} minutes\n")

# ── RESULTS CSV ──────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"grid_search_results_{timestamp}.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(keys + ["deadline_acc", "habit_acc", "score"])


# ── SHORT TRAINING ────────────────────────────────────────────────────────────
def train_short(agent, episodes=2000, max_steps=30):
    """Slightly longer than before to give deadline patterns time to emerge."""
    env = CurriculumEnv()

    for ep in range(episodes):
        env.set_phase(ep)
        state = env.reset()

        for _ in range(max_steps):
            valid = env.get_valid_actions()
            if not valid:
                break

            action = agent.act(state, valid)
            next_s, r, done, _ = env.step(action)

            agent.remember(state, action, r, next_s, done)
            agent.learn()

            state = next_s
            if done:
                break

    return agent


# ── MAIN GRID SEARCH LOOP ─────────────────────────────────────────────────────
best_score        = -float("inf")
best_params       = None
best_metrics      = None

for i, values in enumerate(sampled):
    params = dict(zip(keys, values))

    print(f"🔍 Combo {i+1}/{len(sampled)}")
    print(f"   {params}")

    # ── Init agent (pass device) ──
    env = CurriculumEnv()
    agent = PPOAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        hidden=512,
        lr=params["lr"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_eps=params["clip_eps"],
        ppo_epochs=params["ppo_epochs"],
        batch_size=params["batch_size"],
        vf_coef=0.5,
        ent_coef=params["ent_coef"],
        ent_coef_min=0.002,
        ent_decay=0.9995,
        buffer_size=2048,
        normalise_rewards=True,
                  # ← GPU passed here
    )

    # ── Train ──
    agent = train_short(agent)

    # ── Evaluate ──
    eval_env = SchedulingEnv(max_tasks=10)
    metrics  = evaluate(agent, eval_env)

    deadline = metrics["deadline_acc"]
    habit    = metrics["habit_acc"]

    # Weighted score — deadline weighted MORE since it's the weak point
    score = 0.65 * deadline + 0.35 * habit

    print(f"   Deadline: {deadline:.2%}  |  Habit: {habit:.2%}  |  Score: {score:.4f}")

    # ── Save to CSV ──
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(values) + [f"{deadline:.4f}", f"{habit:.4f}", f"{score:.4f}"])

    # ── Track best ──
    if score > best_score:
        best_score   = score
        best_params  = params
        best_metrics = metrics
        print(f"   ✅ NEW BEST!")

    # ── VRAM cleanup every 50 runs ──
    if device.type == "cuda" and (i + 1) % 50 == 0:
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated() / 1e6
        print(f"\n   🧹 VRAM cleared — currently using {used:.1f} MB\n")

# ── FINAL RESULTS ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🏆 BEST HYPERPARAMETERS FOUND:")
for k, v in best_params.items():
    print(f"   {k:15s} = {v}")
print(f"\n   Deadline  : {best_metrics['deadline_acc']:.2%}")
print(f"   Habit     : {best_metrics['habit_acc']:.2%}")
print(f"   Score     : {best_score:.4f}")
print(f"\n📄 Full results saved to: {csv_path}")
print("=" * 60)