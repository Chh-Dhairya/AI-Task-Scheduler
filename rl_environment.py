from __future__ import annotations
import numpy as np

WORK_START = 7
WORK_END   = 22
MAX_TASKS  = 10


class SchedulingEnv:
    
    TASK_FEATURES: int = 6  # pref, deadline, priority, dur, done, available

    ALL_REWARD_TERMS = ("habit", "buffer", "priority")

    def __init__(self, max_tasks: int = MAX_TASKS,
                 reward_terms: tuple = ALL_REWARD_TERMS,
                 use_mode_flag: bool = True,
                 use_pref: bool = True) -> None:
        """
        Ablation switches (Ablation 3; defaults = full model):
          reward_terms  : bonus terms kept in _reward ("habit", "buffer", "priority")
          use_mode_flag : False → state shows 0 instead of the habit/priority mode flag
          use_pref      : False → state shows 0 instead of each task's preferred hour
        Placement and metrics are unchanged; only what the agent sees / is rewarded for.
        """
        self.max_tasks     = max_tasks
        self.state_size    = 2 + max_tasks * self.TASK_FEATURES
        self.action_size   = max_tasks
        unknown = set(reward_terms) - set(self.ALL_REWARD_TERMS)
        assert not unknown, f"unknown reward terms {unknown}"
        self.reward_terms  = tuple(reward_terms)
        self.use_mode_flag = use_mode_flag
        self.use_pref      = use_pref

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self, tasks: list | None = None) -> np.ndarray:
        """Start a new episode.  Pass real tasks or None to generate a random scenario."""
        self.tasks     = list(tasks) if tasks is not None else self._random_tasks()
        self.n         = len(self.tasks)
        self.done_mask = [False] * self.n
        self.cur_hour  = float(WORK_START)
        self.schedule  = []

        # Mode is determined once at episode start
        self.mode = (
            "habit" if not self._conflicts_exist(list(range(self.n)))
            else "priority"
        )
        return self._state()

    # ── state ──────────────────────────────────────────────────────────────────

    def _state(self) -> np.ndarray:
        s = np.zeros(self.state_size, dtype=np.float32)
        s[0] = (self.cur_hour - WORK_START) / (WORK_END - WORK_START)
        s[1] = 1.0 if (self.use_mode_flag and self.mode == "priority") else 0.0

        for i in range(self.max_tasks):
            off = 2 + i * self.TASK_FEATURES
            if i < self.n:
                t = self.tasks[i]
                if self.use_pref:
                    s[off + 0] = (t["Preferred_Hour"] - WORK_START) / (WORK_END - WORK_START)
                s[off + 1] = (t["Deadline_Hour"]       - WORK_START) / (WORK_END - WORK_START)
                s[off + 2] = t["Priority"] / 4.0   # priority 1-4 → 0.25-1.0
                s[off + 3] = t["Estimated_Duration"] / 480.0
                s[off + 4] = 1.0 if self.done_mask[i] else 0.0
                s[off + 5] = 0.0 if self.done_mask[i] else 1.0
        return s

    # ── step ───────────────────────────────────────────────────────────────────

    def step(self, action: int):
        """Schedule task at index *action*.  Returns (next_state, reward, done, info)."""
        if action >= self.n or self.done_mask[action]:
            return self._state(), -5.0, False, {"invalid": True}

        task     = self.tasks[action]
        dur_h    = task["Estimated_Duration"] / 60.0
        pref     = float(task["Preferred_Hour"])
        deadline = float(task["Deadline_Hour"])

        # ── place the task ─────────────────────────────────────────────────────
        # Habit-first: start as close to preferred time as possible
        start = max(self.cur_hour, pref)
        end   = start + dur_h

        # Pull back only when staying at habit time would bust the deadline
        if end > deadline:
            start = max(self.cur_hour, deadline - dur_h)
            end   = start + dur_h

        # ── reward ─────────────────────────────────────────────────────────────
        reward = self._reward(task, start, end)

        # ── update bookkeeping ─────────────────────────────────────────────────
        self.done_mask[action] = True

        if end > WORK_END:
            status = "Not Scheduled (Exceeds Day)"
            # don't advance clock — task didn't run
        elif end > deadline:
            status = "Scheduled but Missed Deadline"
            self.cur_hour = end
        else:
            status = "Scheduled Successfully"
            self.cur_hour = end

        self.schedule.append({
            "task":   task,
            "start":  start,
            "end":    end,
            "status": status,
            "mode":   self.mode,
        })

        done = all(self.done_mask[: self.n])
        return self._state(), reward, done, {"status": status, "start": start, "end": end}

    def get_valid_actions(self) -> list[int]:
        return [i for i in range(self.n) if not self.done_mask[i]]

    # ── reward model ───────────────────────────────────────────────────────────

    def _reward(self, task: dict, start: float, end: float) -> float:
        """
        Psychologically grounded reward.

        HABIT MODE
          • Primary driver  : habit adherence — big bonus for being at preferred time
          • Secondary driver: small priority bonus (person still cares a little)

        PRIORITY MODE
          • Primary driver  : meeting the deadline
          • Secondary driver: finishing early (buffers future tasks)
          • Tertiary driver : residual habit adherence (small but non-zero)
        """
        deadline = float(task["Deadline_Hour"])
        pref     = float(task["Preferred_Hour"])
        priority = task["Priority"]          # 1-4 (Low, Medium, High, Critical)

        # Can't fit in the working day
        if end > WORK_END:
            return -12.0

        # Missed deadline — penalty scales with priority (urgent missed = worse)
        if end > deadline:
            return -8.0 - 2.5 * (priority / 4.0)

        # ── deadline met ───────────────────────────────────────────────────────
        base      = 8.0
        deviation = abs(start - pref)       # hours from preferred start

        if self.mode == "habit":
            # Strong habit reward: +6 at pref time, fades to 0 at 2 h deviation
            habit_bonus = 6.0 * max(0.0, 1.0 - deviation / 2.0)
            # Small priority spice (person isn't ignoring importance entirely)
            prio_bonus  = 0.5 * priority
            return (base + (habit_bonus if "habit" in self.reward_terms else 0.0)
                         + (prio_bonus  if "priority" in self.reward_terms else 0.0))

        else:  # priority mode
            # Buffer bonus: finishing early frees up time for remaining tasks
            buffer       = deadline - end
            buffer_bonus = min(3.0, buffer * 0.7)
            # Priority matters more now
            prio_bonus   = 2.0 * (priority / 4.0)
            # Residual habit (people under stress still prefer comfortable timings)
            habit_bonus  = 2.0 * max(0.0, 1.0 - deviation / 4.0)
            return (base + (buffer_bonus if "buffer"   in self.reward_terms else 0.0)
                         + (prio_bonus   if "priority" in self.reward_terms else 0.0)
                         + (habit_bonus  if "habit"    in self.reward_terms else 0.0))

    # ── conflict detection ─────────────────────────────────────────────────────

    def _conflicts_exist(self, indices: list[int]) -> bool:
        """Return True iff scheduling the indexed tasks in habit order misses any deadline."""
        tasks_sorted = sorted(
            [self.tasks[i] for i in indices],
            key=lambda t: t["Preferred_Hour"],
        )
        t = self.cur_hour
        for task in tasks_sorted:
            dur   = task["Estimated_Duration"] / 60.0
            start = max(t, float(task["Preferred_Hour"]))
            end   = start + dur
            if end > task["Deadline_Hour"]:
                return True
            t = end
        return False

    # ── random scenario generation (training only) ─────────────────────────────

    def _random_tasks(self) -> list[dict]:
        """
        Generate a random scheduling scenario with realistic variety.

        Scenario mix (approximate):
          45 % habit    — generous deadlines, agent should learn habit order
          35 % priority — tight deadlines, agent should learn urgency ordering
          20 % mixed    — some tasks tight, some slack
        """
        n    = np.random.randint(2, self.max_tasks + 1)
        kind = np.random.choice(["habit", "priority", "mixed"], p=[0.45, 0.35, 0.20])
        tasks = []

        for i in range(n):
            pref    = int(np.random.uniform(WORK_START, WORK_END - 3))
            dur_min = int(np.random.choice([15, 20, 30, 45, 60, 90, 120]))
            dur_h   = dur_min / 60.0
            prio    = int(np.random.randint(1, 5))     # 1-4, same scale as the app

            if kind == "habit":
                slack = np.random.uniform(2.0, 5.0)
            elif kind == "priority":
                slack = np.random.uniform(0.0, 1.0)
            else:
                slack = np.random.uniform(0.5, 3.0)

            deadline = min(WORK_END, int(pref + dur_h + slack))

            tasks.append({
                "Task":               f"Task_{i}",
                "Preferred_Hour":     pref,
                "Deadline_Hour":      deadline,
                "Priority":           prio,
                "Estimated_Duration": dur_min,
            })

        return tasks