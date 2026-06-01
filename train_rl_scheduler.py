from __future__ import annotations
import numpy as np
from rl_environment import SchedulingEnv, WORK_END
from rl_agent       import PPOAgent

#── evaluation ──
def evaluate(agent: PPOAgent, env: SchedulingEnv, n_eval: int = 1_000) -> dict:
    """Run agent greedily on *n_eval* random scenarios and return metrics."""
    agent.net.eval()
    saved_eps, agent.eps = agent.eps, 0.0

    total_tasks    = 0
    deadline_met   = 0
    habit_eligible = 0
    habit_on_time  = 0

    for _ in range(n_eval):
        state = env.reset()
        for _ in range(env.max_tasks + 3):
            valid = env.get_valid_actions()
            if not valid:
                break
            action = agent.act(state, valid)
            state, _, done, _ = env.step(action)
            if done:
                break

        for entry in env.schedule:
            total_tasks += 1
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
        "habit_acc":    habit_on_time / max(habit_eligible, 1),
    }

# ── curriculum env wrapper ─────────────────────────────────────────────────────
class CurriculumEnv:
    """
    Wraps SchedulingEnv to gradually increase task count.

    Phase 1 (ep 0 – 4999)   : 2–4 tasks   easy, agent learns basic ordering
    Phase 2 (ep 5000 – 9999): 2–7 tasks   medium complexity
    Phase 3 (ep 10000+)     : 2–10 tasks  full difficulty
    """

    def __init__(self):
        self._envs = {
            1: SchedulingEnv(max_tasks=4),
            2: SchedulingEnv(max_tasks=7),
            3: SchedulingEnv(max_tasks=10),
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


# ── training loop ──────────────────────────────────────────────────────────────
def train(
    episodes:   int = 20_000,
    max_steps:  int = 25,
    eval_every: int = 500,
    save_path:  str = "rl_scheduler.pth",
) -> PPOAgent:
    
    cur_env = CurriculumEnv()
    eval_env = SchedulingEnv(max_tasks=10)     

    agent = PPOAgent(
        state_size   = cur_env.state_size,
        action_size  = cur_env.action_size,
        hidden       = 512,
        lr           = 3e-4,
        gamma        = 0.99,
        gae_lambda   = 0.95,
        clip_eps     = 0.2,
        ppo_epochs   = 8,
        batch_size   = 128,
        vf_coef      = 0.5,
        ent_coef     = 0.05,
        ent_coef_min = 0.005,
        ent_decay    = 0.9995,
        buffer_size  = 2048,
        normalise_rewards = True,
    )

    print(f"Training PPO scheduler — {episodes} episodes  (curriculum)")
    print(f"State size: {cur_env.state_size}   Action size: {cur_env.action_size}")
    print(f"Device: {agent.device}")
    print(f"Phases: 1=ep0-4999(≤4 tasks)  2=ep5000-9999(≤7)  3=ep10000+(≤10)")
    print("─" * 70)

    rewards_window:    list[float] = []
    best_score:        float       = 0.0   

    for ep in range(1, episodes + 1):
        cur_env.set_phase(ep)
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
            metrics = evaluate(agent, eval_env)
            avg_r   = float(np.mean(rewards_window))
            phase   = cur_env._phase
            score   = metrics["deadline_acc"] + 0.5 * metrics["habit_acc"]

            print(
                f"Ep {ep:6d} [P{phase}] | "
                f"Avg Reward: {avg_r:7.2f} | "
                f"Deadline Acc: {metrics['deadline_acc']:6.1%} | "
                f"Habit Acc: {metrics['habit_acc']:6.1%} | "
                f"EntCoef: {agent.ent_coef:.4f}"
            )

            if score > best_score:
                best_score = score
                agent.save(save_path)
                print(f"  ✓  New best saved  "
                      f"(deadline: {metrics['deadline_acc']:.1%}  "
                      f"habit: {metrics['habit_acc']:.1%})")

    best_dl = best_score / 1.5   # approx — actual stored in checkpoint
    print(f"\nTraining complete.  Best composite score: {best_score:.3f}")
    return agent


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(episodes=20_000)