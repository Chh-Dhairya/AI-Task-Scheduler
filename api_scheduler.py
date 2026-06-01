"""
api_scheduler.py
Gemini-based behavioral correction scheduler.

Key fixes vs original
─────────────────────
1. Model updated to "gemini-2.0-flash"  (gemini-1.5-flash is 404 on v1beta)
2. API response time strings ("HH:MM") are parsed into real datetime objects
   so fusion_engine.py can do arithmetic on them without crashing.
3. Graceful fallback: if Gemini is unavailable, _build_fallback_schedule()
   returns a rule-based schedule instead of an empty list, so the fusion
   engine always has something to work with.
4. Prompt tightened: today's date is injected so Gemini returns today's
   datetimes, not some arbitrary date.
"""

from google import genai
from datetime import datetime, timedelta
import json
import logging
from config import API_KEY, ACTIVITY_TIME_RULES, UNHEALTHY_PATTERNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialise Gemini client once at import time
client = genai.Client(api_key=API_KEY)

# Gemini model to use — update here if you want to switch
GEMINI_MODEL = "gemini-2.0-flash"


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

    def __init__(self):
        self._today = datetime.now().date()

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
            result = self._build_fallback_schedule(tasks)

        return result

    # ── Gemini API call ───────────────────────────────────────────────────────

    def _call_gemini(self, tasks: list) -> list:
        """Call Gemini and parse its JSON response.  Returns [] on any error."""
        try:
            # Build a compact task summary for the prompt
            task_lines = []
            for t in tasks:
                dl = t['deadline'].strftime('%H:%M') if isinstance(t.get('deadline'), datetime) else str(t.get('deadline', '23:00'))
                task_lines.append(
                    f"  - {t['name']}: duration={t['duration']} min, "
                    f"priority={t['priority']}, deadline={dl}"
                )
            tasks_text = "\n".join(task_lines)

            prompt = f"""You are a behavioral correction scheduling AI.
Your ONLY job is to correct unhealthy timings suggested by an ML model.
You must NOT reorder tasks or change durations.

Today's date: {self._today}

Tasks to schedule:
{tasks_text}

Health rules you must enforce:
- No eating (Breakfast / Lunch / Dinner / Eating) after 22:00
- No Sleep after 01:00
- Exercise is best before 10:00; penalise if after 21:00
- Work / Study should stay within 07:00-22:00
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

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
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
            ml_hour  = self._infer_ml_hour(task)

            # Look up rules (case-insensitive partial match)
            rule = self._find_rule(name)

            if rule:
                ideal_start = rule["ideal_start"]
                ideal_end   = rule.get("ideal_end", ideal_start + 2)
                penalty     = rule.get("penalty_outside", 0.2)

                if ideal_start <= ml_hour < ideal_end:
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

            start_dt = datetime.combine(today, datetime.min.time()).replace(hour=chosen_hour, minute=0)
            end_dt   = start_dt + timedelta(minutes=dur_min)

            schedule.append({
                "task":       name,
                "start_time": start_dt,
                "end_time":   end_dt,
                "score":      min(1.0, max(0.0, score)),
                "reason":     reason,
            })

        return schedule

    def _infer_ml_hour(self, task: dict) -> int:
        """Extract the ML-predicted hour from a task dict (best effort)."""
        dl = task.get("deadline")
        if isinstance(dl, datetime):
            # Back-estimate: assume ML puts it 1-2 h before deadline
            return max(7, dl.hour - 1)
        return 9  # safe default

    @staticmethod
    def _find_rule(name: str) -> dict | None:
        """Case-insensitive partial key match against ACTIVITY_TIME_RULES."""
        name_lower = name.lower()
        for key, rule in ACTIVITY_TIME_RULES.items():
            if key.lower() in name_lower or name_lower in key.lower():
                return rule
        return None

    @staticmethod
    def _check_unhealthy(name: str, hour: int, score: float, reason: str):
        """Apply unhealthy-pattern penalties from config."""
        name_lower = name.lower()
        for pattern_name, pattern in UNHEALTHY_PATTERNS.items():
            activities = [a.lower() for a in pattern.get("activity", [])]
            if not any(a in name_lower for a in activities):
                continue

            after  = pattern.get("after_hour")
            before = pattern.get("before_hour")
            pen    = pattern.get("penalty", 0.2)

            if after is not None and hour >= after:
                score  -= pen
                reason += f" | ⚠ Unhealthy pattern: {pattern_name}"
            if before is not None and hour < before:
                score  -= pen
                reason += f" | ⚠ Unhealthy pattern: {pattern_name}"

        return score, reason