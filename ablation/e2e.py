"""
Shared pieces for the end-to-end ablations (Ablation 4 LLM, Ablation 5 fusion,
full-system comparison), all on ablation/data/e2e_bank.json.

  • build the ML (PPO) schedule for a scenario, exactly like the app (ml_pipeline),
  • put every schedule on one FIXED day so Gemini prompts (which contain the date)
    are identical on re-runs and the reply cache stays valid,
  • a caching wrapper around the Gemini client (ablation/cache/<hash>.json),
  • metrics: deadline accuracy with TRUE durations, habit_dev vs the TRUE start,
    health violations, overlaps, time shift.
"""
import contextlib
import hashlib
import io
import json
import os
import threading
import time
from datetime import datetime, date, timedelta

import numpy as np

from ablation.health_checker import count_violations

FIXED_DAY = date(2023, 6, 15)            # a June day; every schedule is placed on it
CACHE_DIR = os.path.join("ablation", "cache")
E2E_BANK  = os.path.join("ablation", "data", "e2e_bank.json")


# ── models + scenario → tasks ────────────────────────────────────────────────

def load_models():
    from habit_model import HabitModel
    from duration_model import DurationModel
    with contextlib.redirect_stdout(io.StringIO()):
        hm = HabitModel(); hm.load("habit_model.pkl")
    dm = DurationModel(); dm.load("duration_model.pkl")
    return hm, dm


def scenario_tasks(scenario, hm, dm, use_habit=True, use_duration=True, duration_lookup=None):
    """
    Turn one e2e scenario into app-style tasks (what a user would type in).
      use_habit=False    → activity-level preferred time for everyone (no personalisation)
      use_duration=False → duration from duration_lookup[activity] (e.g. per-activity median)
    Confidence comes from the habit profile (fit on Jan–May; no June data).
    """
    from preprocess import BUCKET_TO_HOUR
    tasks = []
    for t in scenario:
        p, a = t["Person_Encoded"], t["Activity_Encoded"]
        prof = hm.get_profile(p, a) if use_habit else {**hm.activity_profile[a], "level": 2}
        hour = BUCKET_TO_HOUR[prof["preferred_bucket"]]
        dur = dm.predict(a, hour, t["DayOfWeek"]) if use_duration else duration_lookup[t["Task"]]
        tasks.append({"Task": t["Task"], "Activity_Encoded": a, "Priority": t["Priority"],
                      "Preferred_Hour": hour, "Deadline_Hour": t["Deadline_Hour"],
                      "Estimated_Duration": round(float(dur)), "Confidence": float(prof["confidence"])})
    return tasks


def on_fixed_day(dt):
    return datetime.combine(FIXED_DAY, dt.time()) + timedelta(days=(dt.date() - datetime.now().date()).days)


def ml_schedule(tasks):
    """PPO schedule exactly as the app makes it (ml_pipeline), moved to FIXED_DAY."""
    from ml_pipeline import create_ml_schedule
    with contextlib.redirect_stdout(io.StringIO()):
        ml = create_ml_schedule(tasks)
    for m in ml:
        m["start_time"], m["end_time"] = on_fixed_day(m["start_time"]), on_fixed_day(m["end_time"])
    return ml


def api_tasks(tasks, ml):
    from ml_pipeline import prepare_tasks_for_api
    return prepare_tasks_for_api(tasks, ml)          # dates follow ml (FIXED_DAY)


# ── (de)serialising schedules ────────────────────────────────────────────────

def dump_schedule(s):
    return [{**{k: v for k, v in x.items() if not isinstance(v, datetime)},
             "start_time": x["start_time"].isoformat(), "end_time": x["end_time"].isoformat()} for x in s]


def load_schedule(s):
    return [{**x, "start_time": datetime.fromisoformat(x["start_time"]),
             "end_time": datetime.fromisoformat(x["end_time"])} for x in s]


# ── Gemini reply cache ───────────────────────────────────────────────────────

class CachingClient:
    """
    Drop-in for api_scheduler's client: models.generate_content(model, contents, config).
    Replies are cached by (model, temperature, mime type, prompt, repeat). Transient errors
    (429 / 5xx / timeouts) are retried with back-off; failures are never cached, so
    APIScheduler falls back to rules exactly as in the app and the fallback is counted.
    """
    def __init__(self, real_client, retries=5, pause=4.0):
        self.real, self.retries, self.pause = real_client, retries, pause
        self.models = self
        self.local = threading.local()            # per-thread repeat index (Ablation 4c)
        self.stats = {"hits": 0, "calls": 0, "errors": 0}
        self._lock = threading.Lock()
        os.makedirs(CACHE_DIR, exist_ok=True)

    def set_repeat(self, r):
        self.local.repeat = r

    def key(self, model, contents, config):
        r = getattr(self.local, "repeat", 0)
        blob = json.dumps({"model": model, "temperature": getattr(config, "temperature", None),
                           "mime": getattr(config, "response_mime_type", None), "prompt": contents,
                           **({"repeat": r} if r else {})}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def generate_content(self, model, contents, config=None):
        path = os.path.join(CACHE_DIR, self.key(model, contents, config) + ".json")
        if os.path.exists(path):
            with self._lock: self.stats["hits"] += 1
            with open(path) as f:
                return _Reply(json.load(f)["text"])
        last = None
        for attempt in range(self.retries):
            try:
                with self._lock: self.stats["calls"] += 1
                r = self.real.models.generate_content(model=model, contents=contents, config=config)
                text = r.text or ""
                with open(path, "w") as f:
                    json.dump({"model": model, "text": text, "time": time.time()}, f)
                return _Reply(text)
            except Exception as e:                 # 429 / 503 / network → back off and retry
                last = e
                time.sleep(self.pause * (2 ** attempt))
        with self._lock: self.stats["errors"] += 1
        raise last


class _Reply:
    def __init__(self, text): self.text = text


# ── metrics ──────────────────────────────────────────────────────────────────

def _h(dt):
    return dt.hour + dt.minute / 60 + 24 * (dt.date() - FIXED_DAY).days


def evaluate(schedule, scenario, ml=None):
    """
    schedule: list of {"task", "start_time", "end_time"} (datetimes on FIXED_DAY).
    Deadline met = start + TRUE duration ≤ deadline (same day); tasks missing from the
    schedule count as missed. habit_dev = |start − TRUE start| in minutes.
    """
    by = {t["Task"]: t for t in scenario}
    placed = {s["task"]: s for s in schedule}
    met = sum(1 for name, t in by.items() if name in placed and
              _h(placed[name]["start_time"]) + t["true_duration"] / 60 <= t["Deadline_Hour"])
    dev = [abs(_h(placed[n]["start_time"]) - by[n]["true_start"]) * 60 for n in placed if n in by]
    ov = sum(1 for i, a in enumerate(schedule) for b in schedule[i + 1:]
             if a["start_time"] < b["end_time"] and b["start_time"] < a["end_time"])
    out = {"deadline_met": met, "n_tasks": len(by), "habit_dev_sum": float(np.sum(dev)), "n_placed": len(dev),
           "violations": count_violations(schedule), "overlaps": ov}
    if ml is not None:
        mlb = {m["task"]: m for m in ml}
        shift = [abs(_h(placed[n]["start_time"]) - _h(mlb[n]["start_time"])) * 60 for n in placed if n in mlb]
        out.update({"shift_sum": float(np.sum(shift)), "n_shift": len(shift)})
    return out


def summarise(results):
    """Pool per-scenario results into report metrics."""
    s = lambda k: sum(r[k] for r in results)
    out = {"deadline_acc": s("deadline_met") / max(s("n_tasks"), 1),
           "habit_dev": s("habit_dev_sum") / max(s("n_placed"), 1),
           "violations": s("violations"), "violations_per_day": s("violations") / len(results),
           "overlaps": s("overlaps")}
    if all("shift_sum" in r for r in results):
        out["shift_min"] = s("shift_sum") / max(s("n_shift"), 1)
    return out
