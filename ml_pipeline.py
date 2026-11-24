"""
ML side of the hybrid pipeline, shared by app.py (UI) and main.py (CLI) so both
use exactly the same logic (B8).

    tasks ──► create_ml_schedule ──► ml_schedule ──► prepare_tasks_for_api ──► APIScheduler
              (PPO order + placement)                (ML start sent to the LLM, A3)
"""
from config import ML_WEIGHTS
from scheduler import schedule_tasks

# The PPO network has 10 task slots (SchedulingEnv max_tasks).
MAX_TASKS = 10

_RL_REWARD = {"Scheduled Successfully": 1.0, "Scheduled but Missed Deadline": 0.4}


def create_ml_schedule(tasks: list) -> list:
    """PPO chooses the order; tasks that cannot fit in the day are left out."""
    rl_out  = schedule_tasks(tasks)                     # PPO order + placement
    by_name = {t["Task"]: t for t in tasks}
    ml_schedule = []
    for e in rl_out:
        if e["Start"] is None:                          # could not fit in the day
            continue
        t         = by_name[e["Task"]]
        rl_reward = _RL_REWARD.get(e["Status"], 0.0)
        habit     = t.get("Confidence", 0.7)
        # Status already compares end vs deadline, and handles overnight tasks
        # (Sleeping ending next morning) correctly; End.hour alone would not.
        dl_ok     = 1.0 if e["Status"] == "Scheduled Successfully" else 0.0
        prio      = t["Priority"] / 4.0
        ml_score  = (ML_WEIGHTS["rl_reward"]             * rl_reward +
                     ML_WEIGHTS["habit_alignment"]       * habit +
                     ML_WEIGHTS["deadline_satisfaction"] * dl_ok +
                     ML_WEIGHTS["priority_score"]        * prio)
        ml_schedule.append({
            "task": e["Task"], "start_time": e["Start"], "end_time": e["End"],
            "ml_score": ml_score, "rl_reward": rl_reward, "habit_alignment": habit,
            "deadline_satisfaction": dl_ok, "priority_score": prio,
            "deadline_hour": t["Deadline_Hour"], "status": e["Status"],
        })
    return ml_schedule


def unscheduled_tasks(tasks: list, ml_schedule: list) -> list:
    """Names of tasks the ML schedule could not place (show as 'could not be scheduled')."""
    placed = {m["task"] for m in ml_schedule}
    return [t["Task"] for t in tasks if t["Task"] not in placed]


def prepare_tasks_for_api(tasks: list, ml_schedule: list) -> list:
    """
    One API task per PLACED task, carrying its ML start time (A3). Unscheduled
    tasks are not sent, otherwise fusion would add them back as API-only tasks.
    """
    by_name = {t["Task"]: t for t in tasks}
    return [
        {
            "name":     m["task"],
            "ml_start": m["start_time"],
            "duration": int(round(by_name[m["task"]]["Estimated_Duration"])),
            "priority": by_name[m["task"]]["Priority"],
            "deadline": m["start_time"].replace(
                hour=by_name[m["task"]]["Deadline_Hour"], minute=0, second=0, microsecond=0
            ),
        }
        for m in ml_schedule
    ]
