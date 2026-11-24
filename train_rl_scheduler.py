from __future__ import annotations
import os
import numpy as np
from rl_environment import SchedulingEnv, WORK_END
from rl_agent       import PPOAgent
from ablation.common import set_seed, load_json

# Fixed scenario banks (B5), built once by `python -m ablation.make_rl_banks`
VAL_PATH  = os.path.join("ablation", "data", "rl_val.json")    # checkpoint selection
TEST_PATH = os.path.join("ablation", "data", "rl_test.json")   # reported numbers only

# B6: the two hyper-parameter sets that are compared (seed 0, picked on rl_val.json)
OLD_HP     = dict(lr=3e-4, gamma=0.99,  gae_lambda=0.95, clip_eps=0.2,
                  ppo_epochs=8, batch_size=128, ent_coef=0.05)   # code defaults before B6
TABLE41_HP = dict(lr=7e-4, gamma=0.999, gae_lambda=0.88, clip_eps=0.2,
                  ppo_epochs=8, batch_size=256, ent_coef=0.06)   # report Table 4.1, iteration 5

#── evaluation ──
def evaluate(agent: PPOAgent, env: SchedulingEnv, scenarios: list) -> dict:
    """
    Run agent greedily on a FIXED list of scenarios (B5) and return metrics.
    The same scenarios every call, so evaluations are comparable across
    checkpoints, runs and variants.
    """
    agent.net.eval()
    saved_eps, agent.eps = agent.eps, 0.0

    total_tasks    = 0
    deadline_met   = 0
    habit_eligible = 0
    habit_on_time  = 0
    habit_dev_sum  = 0.0   # minutes, over ALL scheduled tasks
    reward_sum     = 0.0   # episode reward under env's reward terms (full in eval envs)

    for tasks in scenarios:
        state = env.reset(tasks)
        for _ in range(env.max_tasks + 3):
            valid = env.get_valid_actions()
            if not valid:
                break
            action = agent.act(state, valid)
            state, r, done, _ = env.step(action)
            reward_sum += r
            if done:
                break

        for entry in env.schedule:
            total_tasks += 1
            # habit_dev (B5): |start - preferred| in minutes. The old habit_acc
            # only counts habit-mode tasks that met their deadline, and a random
            # policy already scores ~77 % on it, so it cannot tell good from bad.
            habit_dev_sum += abs(entry["start"] - entry["task"]["Preferred_Hour"]) * 60.0
            met = (entry["end"] <= entry["task"]["Deadline_Hour"]
                   and entry["end"] <= WORK_END)
            if met:
                deadline_met += 1
            if entry["mode"] == "habit" and met:
                habit_eligible += 1
                if abs(entry["start"] - entry["task"]["Preferred_Hour"]) <= 0.5:
                    habit_on_time += 1

    agent.eps = saved_eps
    agent.net.train()

    return {
        "deadline_acc": deadline_met  / max(total_tasks,    1),
        "habit_acc":    habit_on_time / max(habit_eligible, 1),   # old metric (Table 4.1)
        "habit_dev":    habit_dev_sum / max(total_tasks,    1),   # minutes, lower = better
        "avg_reward":   reward_sum    / max(len(scenarios), 1),   # per episode
    }

# ── curriculum env wrapper ─────────────────────────────────────────────────────
class CurriculumEnv:
    """
    Wraps SchedulingEnv to gradually increase task count.

    Phase 1 (ep 0 – 4999)   : 2–4 tasks   easy, agent learns basic ordering
    Phase 2 (ep 5000 – 9999): 2–7 tasks   medium complexity
    Phase 3 (ep 10000+)     : 2–10 tasks  full difficulty
    """

    def __init__(self, **env_kwargs):
        # env_kwargs: ablation switches (reward_terms, use_mode_flag, use_pref)
        self._envs = {
            1: SchedulingEnv(max_tasks=4,  **env_kwargs),
            2: SchedulingEnv(max_tasks=7,  **env_kwargs),
            3: SchedulingEnv(max_tasks=10, **env_kwargs),
        }
        self._phase    = 1
        self.state_size  = self._envs[3].state_size   # always largest for agent
        self.action_size = self._envs[3].action_size

    def set_phase(self, ep: int):
        if ep < 5_000:
            self._phase = 1
        elif ep < 10_000:
            self._phase = 2
        else:
            self._phase = 3

    @property
    def _env(self):
        return self._envs[self._phase]

    def reset(self):
        # Pad state to full state_size so the network input is always the same
        s = self._env.reset()
        return self._pad(s)

    def step(self, action):
        s, r, done, info = self._env.step(action)
        return self._pad(s), r, done, info

    def get_valid_actions(self):
        return self._env.get_valid_actions()

    @property
    def schedule(self):
        return self._env.schedule

    @property
    def max_tasks(self):
        return self._envs[3].max_tasks

    def _pad(self, s: np.ndarray) -> np.ndarray:
        if len(s) < self.state_size:
            return np.concatenate([s, np.zeros(self.state_size - len(s),
                                               dtype=np.float32)])
        return s


def expected_updates(episodes: int, buffer_size: int, curriculum: bool = True) -> int:
    """
    Expected number of PPO updates for a curriculum run (fix B2).
    One env step per task; tasks per episode ~ U{2..max_tasks} → mean (2+max)/2.
    Phases: ep < 5000 → max 4, ep < 10000 → max 7, else max 10.
    """
    if not curriculum:                             # always the 10-task env (Ablation T2)
        return max(1, int(episodes * (2 + 10) / 2 // buffer_size))
    p1 = min(episodes, 4_999)                      # set_phase: ep 1..4999 → phase 1
    p2 = min(max(episodes - 4_999, 0), 5_000)      # ep 5000..9999 → phase 2
    p3 = max(episodes - 9_999, 0)                  # ep 10000+     → phase 3
    steps = p1 * (2 + 4) / 2 + p2 * (2 + 7) / 2 + p3 * (2 + 10) / 2
    return max(1, int(steps // buffer_size))


# ── training loop ──────────────────────────────────────────────────────────────
def train(
    episodes:   int = 80_000,      # B6: 80k (20k = only 47 PPO updates, too few)
    max_steps:  int = 25,
    eval_every: int = 500,
    save_path:  str = "rl_scheduler.pth",
    seed:       int = 0,
    val_path:   str = VAL_PATH,
    test_path:  str = TEST_PATH,
    hparams:    dict | None = None,
    buffer_size: int = 2048,
    env_kwargs:   dict | None = None,   # Ablation 3: reward_terms / use_mode_flag / use_pref
    agent_kwargs: dict | None = None,   # Ablation 3: buggy_gae / norm_mode
    curriculum:   bool = True,          # Ablation 3, T2: False → always the 10-task env
) -> PPOAgent:
    """
    Train PPO with the curriculum. Checkpoints are picked on the fixed
    validation bank; at the end the best checkpoint is loaded and scored ONCE
    on the test bank (agent.val_metrics / agent.test_metrics).
    """
    set_seed(seed)                       # B5: same seed → same run
    val_scenarios  = load_json(val_path)
    test_scenarios = load_json(test_path)

    env_kwargs   = dict(env_kwargs or {})
    agent_kwargs = dict(agent_kwargs or {})
    cur_env  = CurriculumEnv(**env_kwargs)
    # Eval env: the agent must see the same STATE as in training, but episode
    # reward is always measured with the FULL reward, so variants are comparable.
    eval_env = SchedulingEnv(max_tasks=10, **{k: v for k, v in env_kwargs.items() if k != "reward_terms"})

    total_updates = expected_updates(episodes, buffer_size, curriculum)   # length of LR/entropy schedules
    # B6: Table 4.1 won on rl_val.json (seed 0), so it is the default; pass
    # hparams=OLD_HP to reproduce the old settings.
    hp = {**TABLE41_HP, **(hparams or {})}

    agent = PPOAgent(
        state_size   = cur_env.state_size,
        action_size  = cur_env.action_size,
        hidden       = 512,
        lr           = hp["lr"],
        gamma        = hp["gamma"],
        gae_lambda   = hp["gae_lambda"],
        clip_eps     = hp["clip_eps"],
        ppo_epochs   = hp["ppo_epochs"],
        batch_size   = hp["batch_size"],
        vf_coef      = 0.5,
        ent_coef     = hp["ent_coef"],
        ent_coef_min = 0.005,
        ent_decay    = 0.9995,
        buffer_size  = buffer_size,
        total_updates = total_updates,
        normalise_rewards = True,
        **agent_kwargs,
    )

    print(f"Training PPO scheduler — {episodes} episodes  (curriculum)  seed={seed}")
    print(f"Hyper-parameters: {hp}  buffer={buffer_size}  updates≈{total_updates}")
    if env_kwargs or agent_kwargs or not curriculum:
        print(f"Ablation switches: env={env_kwargs} agent={agent_kwargs} curriculum={curriculum}")
    print(f"Val scenarios: {len(val_scenarios)}   Test scenarios: {len(test_scenarios)}")
    print(f"State size: {cur_env.state_size}   Action size: {cur_env.action_size}")
    print(f"Device: {agent.device}")
    print(f"Phases: 1=ep0-4999(≤4 tasks)  2=ep5000-9999(≤7)  3=ep10000+(≤10)")
    print("─" * 70)

    rewards_window:    list[float] = []
    best_score:        float       = 0.0
    best_val:          dict | None = None
    history:           list[dict]  = []     # learning curve (val metrics every eval_every)

    for ep in range(1, episodes + 1):
        cur_env.set_phase(ep if curriculum else 10_000)   # T2: no curriculum → phase 3
        state     = cur_env.reset()
        ep_reward = 0.0

        for _ in range(max_steps):
            valid = cur_env.get_valid_actions()
            if not valid:
                break

            action             = agent.act(state, valid)
            next_s, r, done, _ = cur_env.step(action)
            agent.remember(state, action, r, next_s, done)
            agent.learn()

            state      = next_s
            ep_reward += r
            if done:
                break

        agent.decay_eps()   # no-op for PPO, kept for compatibility

        rewards_window.append(ep_reward)
        if len(rewards_window) > 200:
            rewards_window.pop(0)

        if ep % eval_every == 0:
            metrics = evaluate(agent, eval_env, val_scenarios)   # validation only
            avg_r   = float(np.mean(rewards_window))
            history.append({"episode": ep, "train_avg_reward": avg_r, **metrics})
            phase   = cur_env._phase
            score   = metrics["deadline_acc"] + 0.5 * metrics["habit_acc"]

            print(
                f"Ep {ep:6d} [P{phase}] | "
                f"Avg Reward: {avg_r:7.2f} | "
                f"Deadline Acc: {metrics['deadline_acc']:6.1%} | "
                f"Habit Acc: {metrics['habit_acc']:6.1%} | "
                f"Habit Dev: {metrics['habit_dev']:5.1f} min | "
                f"EntCoef: {agent.ent_coef:.4f} | "
                f"LR: {agent.optimizer.param_groups[0]['lr']:.2e}"
            )

            if score > best_score:
                best_score = score
                best_val   = metrics
                agent.save(save_path)
                print(f"  ✓  New best saved  "
                      f"(deadline: {metrics['deadline_acc']:.1%}  "
                      f"habit: {metrics['habit_acc']:.1%})")

    print(f"\nTraining complete.  Best composite score: {best_score:.3f}")
    print(f"PPO updates: actual {agent._updates}, expected {total_updates} "
          f"(LR/entropy schedule length)")

    # ── final score: best checkpoint (picked on val) scored ONCE on test ──────
    if best_val is not None:
        agent.load(save_path)
    test = evaluate(agent, eval_env, test_scenarios)
    agent.val_metrics, agent.test_metrics, agent.history = best_val, test, history
    print(f"TEST ({len(test_scenarios)} scenarios, best checkpoint): "
          f"Deadline Acc {test['deadline_acc']:.2%} | Habit Acc {test['habit_acc']:.2%} | "
          f"Habit Dev {test['habit_dev']:.1f} min")
    return agent


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--episodes", type=int, default=80_000)
    args = ap.parse_args()
    train(episodes=args.episodes, seed=args.seed)