from __future__ import annotations
import os
from datetime import datetime, timedelta

WORK_START           = 7
WORK_END             = 22
OVERNIGHT_ACTIVITIES = {"Sleeping"}
RL_MODEL_PATH        = "rl_scheduler.pth"

# ── public entry-point (unchanged interface) ───────────────────────────────────

def schedule_tasks(tasks: list) -> list:
    today     = datetime.now().date()
    regular   = [t for t in tasks if t["Task"] not in OVERNIGHT_ACTIVITIES]
    overnight = [t for t in tasks if t["Task"] in OVERNIGHT_ACTIVITIES]
    schedule  = []

    # ── regular tasks ─────────────────────────────────────────────────────────
    if regular:
        if os.path.exists(RL_MODEL_PATH):
            ordered = _rl_order(regular)
        else:
            print(f"[WARN] RL model not found at '{RL_MODEL_PATH}'.")
            print("       Run  python train_rl_scheduler.py  to train it.")
            print("       Using greedy habit/priority fallback.\n")
            ordered = _greedy_order(regular)

        entries = _place_tasks(ordered, today)
        schedule.extend(entries)

    # ── overnight tasks (Sleeping etc.) ───────────────────────────────────────
    for task in overnight:
        duration      = timedelta(minutes=max(1, round(task["Estimated_Duration"])))
        task_start    = _make_dt(today, task["Preferred_Hour"])
        task_end      = task_start + duration
        deadline_time = _make_dt(today, task["Deadline_Hour"])

        if deadline_time < task_start:          # deadline wraps to next day
            deadline_time += timedelta(days=1)

        schedule.append({
            "Task":           task["Task"],
            "Start":          task_start,
            "End":            task_end,
            "Preferred_Hour": task["Preferred_Hour"],
            "Priority":       task["Priority"],
            "Status":         (
                "Scheduled Successfully"
                if task_end <= deadline_time
                else "Scheduled but Missed Deadline"
            ),
            "Mode":           "habit",
        })

    # ── sort: placed tasks by start time, unscheduled at the end ──────────────
    placed      = sorted([s for s in schedule if s["Start"] is not None],
                         key=lambda x: x["Start"])
    unscheduled = [s for s in schedule if s["Start"] is None]
    return placed + unscheduled


# ── RL ordering ───────────────────────────────────────────────────────────────

def _rl_order(tasks: list) -> list:
    """
    Ask the trained DQN agent to choose a task ordering.

    The agent runs in pure-exploitation mode (ε = 0).  It sees the full
    state (current time, mode flag, per-task features) and picks tasks
    one-by-one until all are scheduled.
    """
    from rl_environment import SchedulingEnv
    from rl_agent       import DQNAgent

    env   = SchedulingEnv(max_tasks=10)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    agent.load(RL_MODEL_PATH)
    agent.eps = 0.0         # greedy — no exploration at inference

    state = env.reset(tasks)
    order: list[dict] = []

    for _ in range(len(tasks) + 5):     # +5 safety margin
        valid = env.get_valid_actions()
        if not valid:
            break
        action = agent.act(state, valid)
        state, _, done, _ = env.step(action)
        order.append(tasks[action])
        if done:
            break

    # Safety net: append any task the agent skipped (shouldn't happen normally)
    scheduled_names = {t["Task"] for t in order}
    for t in tasks:
        if t["Task"] not in scheduled_names:
            order.append(t)

    return order


# ── greedy fallback ───────────────────────────────────────────────────────────

def _greedy_order(tasks: list) -> list:
    """
    Original habit / priority greedy ordering — used only when the RL
    model file is absent.
    """

    def simulate(lst, start_h):
        slot, results = start_h, []
        for t in lst:
            dur   = max(1, round(t["Estimated_Duration"])) / 60.0
            start = max(slot, float(t["Preferred_Hour"]))
            end   = start + dur
            results.append((t, start, end))
            slot = end
        return results

    def missed(sim):
        return sum(1 for t, _, end in sim if end > t["Deadline_Hour"])

    habit_order = sorted(tasks, key=lambda x: x["Preferred_Hour"])
    if missed(simulate(habit_order, float(WORK_START))) == 0:
        return habit_order

    # Under stress: high priority first, then earliest deadline
    priority_order = sorted(tasks, key=lambda x: (-x["Priority"], x["Deadline_Hour"]))

    # Try every possible front-task to minimise misses (same as original)
    best_order   = list(priority_order)
    best_missed  = missed(simulate(priority_order, float(WORK_START)))
    remaining    = list(priority_order)

    for i in range(1, len(remaining)):
        candidate = [remaining[i]] + remaining[:i] + remaining[i + 1:]
        m = missed(simulate(candidate, float(WORK_START)))
        if m < best_missed:
            best_missed = m
            best_order  = candidate
        if best_missed == 0:
            break

    return best_order


# ── task placement ────────────────────────────────────────────────────────────

def _place_tasks(ordered: list, today) -> list:
    """
    Convert an ordered task list into a list of schedule entries with
    real datetime start/end values.

    Placement rules (same in both modes — only order differs):
      1. Start at max(current_time, preferred_hour)  — honour the habit.
      2. If that would bust the deadline (accounting for same-deadline
         peers still to come), pull start back to fit.
      3. Never schedule before current_time.
      4. Never exceed WORK_END.
    """
    entries      = []
    current_h    = float(WORK_START)
    mode         = _detect_mode(ordered, current_h)

    for task in ordered:
        dur_h       = max(1, round(task["Estimated_Duration"])) / 60.0
        deadline_h  = float(task["Deadline_Hour"])
        preferred_h = float(task["Preferred_Hour"])

        # Time that same-deadline peers still to be scheduled will need
        reserved_h = sum(
            max(1, round(t["Estimated_Duration"])) / 60.0
            for t in ordered
            if t is not task
            and t["Deadline_Hour"] == task["Deadline_Hour"]
            and not any(e["Task"] == t["Task"] for e in entries)
        )
        must_finish_by = deadline_h - reserved_h

        # ── habit-first placement ─────────────────────────────────────────────
        task_start = max(current_h, preferred_h)
        task_end   = task_start + dur_h

        # Pull back only when habit time would violate the deadline window
        if task_end > must_finish_by:
            task_start = max(current_h, must_finish_by - dur_h)
            task_end   = task_start + dur_h

        # Guarantee we never go before now
        if task_start < current_h:
            task_start = current_h
            task_end   = task_start + dur_h

        # ── build output entry ────────────────────────────────────────────────
        if task_end > WORK_END:
            status   = "Not Scheduled (Exceeds Day)"
            dt_start = None
            dt_end   = None
        else:
            dt_start = _make_dt(today, task_start)
            dt_end   = _make_dt(today, task_end)
            status   = (
                "Scheduled but Missed Deadline"
                if task_end > deadline_h
                else "Scheduled Successfully"
            )
            current_h = task_end

        entries.append({
            "Task":           task["Task"],
            "Start":          dt_start,
            "End":            dt_end,
            "Preferred_Hour": task["Preferred_Hour"],
            "Priority":       task["Priority"],
            "Status":         status,
            "Mode":           mode,
        })

    return entries


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_mode(tasks: list, start_h: float) -> str:
    """Return 'habit' or 'priority' based on whether habit order is conflict-free."""
    habit_sorted = sorted(tasks, key=lambda t: t["Preferred_Hour"])
    t = start_h
    for task in habit_sorted:
        dur   = max(1, round(task["Estimated_Duration"])) / 60.0
        start = max(t, float(task["Preferred_Hour"]))
        end   = start + dur
        if end > task["Deadline_Hour"]:
            return "priority"
        t = end
    return "habit"


def _make_dt(today, hour_float: float) -> datetime:
    """Convert a fractional hour (e.g. 9.5 → 09:30) to a datetime on *today*."""
    h = int(hour_float)
    m = int(round((hour_float - h) * 60))
    if m >= 60:
        h += 1
        m -= 60
    h = min(h, 23)
    return datetime.combine(today, datetime.min.time()).replace(hour=h, minute=m)