"""
C0: count health-rule violations in a schedule (Ablations 4, 5, full system).

Same windows as the rule engine (A4: config.UNHEALTHY_PATTERNS), plus work hours:
  • late_eating     Eating starts 22:00–05:00
  • night_exercise  Exercising starts 21:00–05:00
  • late_sleep      Sleeping starts 01:00–12:00
  • early_sleep     Sleeping starts 12:00–20:00
  • work_hours      Working starts before 07:00 or ends after 22:00

LIMITATION (write it in the report): the checker uses the same rules as the rule
engine, so the rule engine will look good on it by construction.
"""
from datetime import datetime

from config import UNHEALTHY_PATTERNS
from api_scheduler import in_window

WORK_WINDOW = (7, 22)


def _hours(t):
    """datetime → fractional hour of day; numbers are returned as is."""
    return t.hour + t.minute / 60 if isinstance(t, datetime) else float(t)


def violations(schedule):
    """
    schedule: list of dicts with a task name ("task" or "Task") and start/end as
    datetimes ("start_time"/"end_time") or hours ("start"/"end").
    Returns a list of (task, rule) for every violation.
    """
    out = []
    for item in schedule:
        name  = item.get("task", item.get("Task"))
        start = _hours(item.get("start_time", item.get("start")))
        end_t = item.get("end_time", item.get("end"))
        for rule, p in UNHEALTHY_PATTERNS.items():
            if name in p["activity"] and in_window(int(start), *p["window"]):
                out.append((name, rule))
        if name == "Working" and end_t is not None:
            end = _hours(end_t)
            # an end time on the next day (datetime) also breaks the window
            next_day = isinstance(end_t, datetime) and isinstance(item.get("start_time"), datetime) \
                and end_t.date() > item["start_time"].date()
            if start < WORK_WINDOW[0] or end > WORK_WINDOW[1] or next_day:
                out.append((name, "work_hours"))
    return out


def count_violations(schedule):
    return len(violations(schedule))
