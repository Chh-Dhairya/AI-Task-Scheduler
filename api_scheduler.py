"""
api_scheduler.py
Gemini-based behavioral correction scheduler.

Key fixes vs original
─────────────────────
1. Model name comes from config.GEMINI_MODEL (env var GEMINI_MODEL);
   gemini-2.0-flash was shut down on 1 June 2026. Calls use temperature 0
   and JSON response mode. Every item carries "source": "gemini" | "rules".
2. API response time strings ("HH:MM") are parsed into real datetime objects
   so fusion_engine.py can do arithmetic on them without crashing.
3. Graceful fallback: if Gemini is unavailable, _build_fallback_schedule()
   returns a rule-based schedule instead of an empty list, so the fusion
   engine always has something to work with.
4. Prompt tightened: today's date is injected so Gemini returns today's
   datetimes, not some arbitrary date.
"""

from google import genai
from google.genai import types
from datetime import datetime, timedelta
import json
import logging
from config import API_KEY, GEMINI_MODEL, ACTIVITY_TIME_RULES, UNHEALTHY_PATTERNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini client is created lazily on first use, so importing this module
# never fails when GEMINI_API_KEY is unset (the rule fallback is used instead).
_client = None


def get_client():
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=API_KEY)
    return _client


def in_window(h, start, end):
    """True if hour h is inside [start, end), handling windows that cross midnight."""
    return start <= h < end if start <= end else (h >= start or h < end)


class APIScheduler:
    """
    Gemini-based API Scheduler for behavioral corrections.

    Role in the hybrid system
    ─────────────────────────
    • Does NOT replace the ML schedule.
    • Receives the ML-predicted tasks and applies health / routine rules.
    • Returns a corrected schedule with a score (0-1) and a reason string
      for each task.
    • If the Gemini API is unavailable the built-in rule engine takes over
      so the fusion engine is never starved of an API schedule.
    """

    def __init__(self, temperature: float = 0.0, include_ml_times: bool = True):
        self._today = datetime.now().date()
        self.temperature = temperature   # 0.0 = deterministic (Ablation 4c uses 0.7)
        # False = old prompt without ML start times (Ablation 4b, variant P0)
        self.include_ml_times = include_ml_times

    # ── public entry-point ────────────────────────────────────────────────────

    def get_schedule(self, tasks: list) -> list:
        """
        Return a list of schedule dicts, one per task:
            {
                'task'      : str,
                'start_time': datetime,
                'end_time'  : datetime,
                'score'     : float,   # 0-1
                'reason'    : str
            }
        """
        if not tasks:
            return []

        result = self._call_gemini(tasks)

        # If Gemini failed or returned nothing, fall back to rule engine
        if not result:
            logger.warning("Gemini unavailable — using built-in rule-based fallback.")
            return self._build_fallback_schedule(tasks)

        return self._reconcile(tasks, result)

    def _reconcile(self, tasks: list, items: list) -> list:
        """
        Make Gemini's reply match the request exactly:
        • keep only items whose name is a requested task (first one per name),
        • force the requested duration (the prompt forbids changing it),
        • tasks Gemini left out are corrected by the rule fallback instead.
        Output is in request order, one item per requested task.
        """
        wanted   = {t["name"] for t in tasks}
        by_name  = {}
        for it in items:
            by_name.setdefault(it["task"], it)
        unknown = [n for n in by_name if n not in wanted]
        if unknown:
            logger.warning(f"Gemini returned unknown task names {unknown}; ignored.")

        missing = [t for t in tasks if t["name"] not in by_name]
        if missing:
            logger.warning(f"Gemini omitted {[t['name'] for t in missing]}; "
                           f"using rule fallback for these.")
        fallback = {f["task"]: f for f in self._build_fallback_schedule(missing)}

        out = []
        for t in tasks:
            it = by_name.get(t["name"])
            if it is None:
                out.append(fallback[t["name"]])
                continue
            dur = timedelta(minutes=int(t.get("duration", 60)))
            if it["end_time"] - it["start_time"] != dur:
                logger.warning(f"Gemini changed the duration of {t['name']}; "
                               f"keeping the requested {dur}.")
                it["end_time"] = it["start_time"] + dur
            out.append(it)
        return out

    # ── Gemini API call ───────────────────────────────────────────────────────

    def _call_gemini(self, tasks: list) -> list:
        """Call Gemini and parse its JSON response.  Returns [] on any error."""
        try:
            # Build a compact task summary for the prompt
            task_lines = []
            for t in tasks:
                dl = t['deadline'].strftime('%H:%M') if isinstance(t.get('deadline'), datetime) else str(t.get('deadline', '23:00'))
                ml = f"ML start={t['ml_start'].strftime('%H:%M')}, " if self.include_ml_times else ""
                task_lines.append(
                    f"  - {t['name']}: {ml}duration={t['duration']} min, "
                    f"priority={t['priority']}, deadline={dl}"
                )
            tasks_text = "\n".join(task_lines)
            keep_rule = ("If a task's ML start time already follows the health rules, keep it unchanged.\n"
                         if self.include_ml_times else "")

            prompt = f"""You are a behavioral correction scheduling AI.
Your ONLY job is to correct unhealthy timings suggested by an ML model.
You must NOT reorder tasks or change durations.
{keep_rule}
Today's date: {self._today}

Tasks to schedule:
{tasks_text}

Health rules you must enforce:
- Eating and Cooking: 07:00-21:00. Never start Eating between 22:00 and 05:00
- Sleeping: start between 21:00 and 01:00. Starting between 01:00 and 12:00 is too late; between 12:00 and 20:00 is too early
- Exercising: best 06:00-10:00. Never start between 21:00 and 05:00
- Working: 09:00-18:00, and never outside 07:00-22:00
- Walking: 06:00-20:00. Shopping: 09:00-21:00
- Prefer morning productivity

Return ONLY a valid JSON array — no markdown, no explanation:

[
  {{
    "task": "<exact task name>",
    "start_time": "HH:MM",
    "end_time": "HH:MM",
    "score": <float 0.0-1.0>,
    "reason": "<one sentence>"
  }}
]
"""

            response = get_client().models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,           # 0.0 → same answer every time
                    response_mime_type="application/json",  # forces valid JSON
                ),
            )

            if not response or not response.text:
                logger.error("Empty response from Gemini.")
                return []

            text = response.text.strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()

            raw_list = json.loads(text)
            return self._parse_gemini_response(raw_list)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error from Gemini response: {e}")
            return []
        except Exception as e:
            logger.error(f"Gemini API failed: {e}")
            return []

    # ── response parser ───────────────────────────────────────────────────────

    def _parse_gemini_response(self, raw_list: list) -> list:
        """
        Convert Gemini's HH:MM strings into real datetime objects so the
        fusion engine can do timedelta arithmetic on them.
        """
        parsed = []
        today = self._today

        for item in raw_list:
            try:
                start_dt = self._parse_time(item.get("start_time", "09:00"), today)
                end_dt   = self._parse_time(item.get("end_time",   "10:00"), today)

                # If end is before start (midnight crossing), push end to next day
                if end_dt <= start_dt:
                    end_dt += timedelta(days=1)

                parsed.append({
                    "task":       item.get("task", "Unknown"),
                    "start_time": start_dt,
                    "end_time":   end_dt,
                    "score":      float(item.get("score", 0.7)),
                    "reason":     item.get("reason", "API suggestion"),
                    "source":     "gemini",
                })
            except Exception as e:
                logger.warning(f"Skipping malformed API item {item}: {e}")

        return parsed

    @staticmethod
    def _parse_time(time_str: str, today) -> datetime:
        """Parse 'HH:MM' string into a datetime on *today*."""
        h, m = map(int, time_str.strip().split(":"))
        return datetime.combine(today, datetime.min.time()).replace(hour=h, minute=m)

    # ── built-in rule-based fallback ──────────────────────────────────────────

    def _build_fallback_schedule(self, tasks: list) -> list:
        """
        Pure rule-based behavioral correction used when Gemini is unavailable.

        Logic
        ─────
        • For each activity, check ACTIVITY_TIME_RULES for the ideal window.
        • If the ML-predicted hour is inside the window → keep it, score = 0.9.
        • If outside → clamp to the window start, score = 0.6 + penalty note.
        • Durations are preserved from the task input.
        """
        today    = self._today
        schedule = []

        for task in tasks:
            name     = task["name"]
            dur_min  = int(task.get("duration", 60))
            ml_start = task["ml_start"]
            ml_hour  = ml_start.hour

            # Look up rules (exact activity name)
            rule = self._find_rule(name)

            if rule:
                ideal_start = rule["ideal_start"]
                ideal_end   = rule.get("ideal_end", ideal_start + 2)
                penalty     = rule.get("penalty_outside", 0.2)

                if in_window(ml_hour, ideal_start, ideal_end):
                    chosen_hour = ml_hour
                    score       = 0.9 + rule.get("bonus", 0.0)
                    reason      = f"✓ Ideal time for {name}"
                else:
                    chosen_hour = ideal_start
                    score       = max(0.4, 0.8 - penalty)
                    reason      = (
                        f"⚠ Moved to {ideal_start:02d}:00 — "
                        f"ML suggestion ({ml_hour:02d}:00) outside healthy window "
                        f"({ideal_start:02d}:00-{ideal_end:02d}:00)"
                    )
            else:
                # No specific rule — keep ML time, neutral score
                chosen_hour = ml_hour
                score       = 0.75
                reason      = f"No specific rule for {name} — keeping ML time"

            # Check unhealthy patterns and apply extra penalty
            score, reason = self._check_unhealthy(name, chosen_hour, score, reason)

            if chosen_hour == ml_hour:
                # ML time kept → keep it exactly (incl. minutes)
                start_dt = datetime.combine(today, ml_start.time()).replace(second=0, microsecond=0)
            else:
                start_dt = datetime.combine(today, datetime.min.time()).replace(hour=chosen_hour, minute=0)
            end_dt   = start_dt + timedelta(minutes=dur_min)

            schedule.append({
                "task":       name,
                "start_time": start_dt,
                "end_time":   end_dt,
                "score":      min(1.0, max(0.0, score)),
                "reason":     reason,
                "source":     "rules",
            })

        return schedule

    @staticmethod
    def _find_rule(name: str) -> dict | None:
        """Exact match on the dataset activity name."""
        return ACTIVITY_TIME_RULES.get(name)

    @staticmethod
    def _check_unhealthy(name: str, hour: int, score: float, reason: str):
        """Apply unhealthy-pattern penalties from config."""
        for pattern_name, pattern in UNHEALTHY_PATTERNS.items():
            if name not in pattern.get("activity", []):
                continue
            pen = pattern.get("penalty", 0.2)
            if in_window(hour, *pattern["window"]):
                score  -= pen
                reason += f" | ⚠ Unhealthy pattern: {pattern_name}"

        return score, reason