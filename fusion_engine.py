import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import logging
from config import (
    DEFAULT_MODEL_WEIGHT, DEFAULT_API_WEIGHT,
    ENABLE_ADAPTIVE_WEIGHTING,
    LOW_CONFIDENCE_THRESHOLD, HIGH_ADHERENCE_THRESHOLD,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScheduleFusionEngine:
    """
    Hybrid scheduling system that combines ML and API predictions
    with intelligent weighting.

    Core formula
    ────────────
        final_score = (ml_weight × ml_score) + (api_weight × api_score)

    Adaptive weighting
    ──────────────────
        habit_confidence low  → increase API weight (API knows better)
        habit_confidence high → increase ML weight  (ML knows the user well)
    """

    def __init__(self,
                 model_weight: float = DEFAULT_MODEL_WEIGHT,
                 api_weight: float = DEFAULT_API_WEIGHT,
                 adaptive: bool = ENABLE_ADAPTIVE_WEIGHTING):
        self.model_weight = model_weight
        self.api_weight   = api_weight
        self.adaptive     = adaptive
        self._normalize_weights()
        logger.info(
            f"Fusion Engine initialized: ML={self.model_weight:.2f}, "
            f"API={self.api_weight:.2f}"
        )

    # ── weight management ─────────────────────────────────────────────────────

    def _normalize_weights(self):
        """Ensure weights always sum to exactly 1.0."""
        total = self.model_weight + self.api_weight
        if total > 0 and abs(total - 1.0) > 1e-9:
            self.model_weight /= total
            self.api_weight   /= total

    def update_weights(self, new_model_weight: float, new_api_weight: float):
        """Manually set weights ."""
        self.model_weight = new_model_weight
        self.api_weight   = new_api_weight
        self._normalize_weights()
        logger.info(
            f"Weights updated: ML={self.model_weight:.2f}, "
            f"API={self.api_weight:.2f}"
        )

    # ── main entry-point ──────────────────────────────────────────────────────

    def combine_schedules(self,
                          ml_schedule: List[Dict[str, Any]],
                          api_schedule: List[Dict[str, Any]],
                          habit_confidence: Optional[float] = None
                          ) -> List[Dict[str, Any]]:
        """
        Merge ML and API schedules into a single optimised schedule.

        Parameters
        ──────────
        ml_schedule      : list of ML task dicts (must contain datetime objects)
        api_schedule     : list of API task dicts (must contain datetime objects)
        habit_confidence : average confidence from the habit model (0-1)

        Returns
        ───────
        Sorted, conflict-resolved list of fused task dicts.
        """
        # Adaptive weight adjustment
        if self.adaptive and habit_confidence is not None:
            self.model_weight, self.api_weight = self._adjust_weights(habit_confidence)

        matched_tasks  = self._match_tasks(ml_schedule, api_schedule)
        fused_schedule = [self._fuse_single_task(t) for t in matched_tasks]
        fused_schedule = self._resolve_conflicts(fused_schedule)
        fused_schedule = sorted(fused_schedule, key=lambda x: x["start_time"])
        return fused_schedule

    # ── adaptive weighting ────────────────────────────────────────────────────

    def _adjust_weights(self, habit_confidence: float) -> Tuple[float, float]:
        if habit_confidence < LOW_CONFIDENCE_THRESHOLD:
            api_w   = min(0.4, self.api_weight + 0.15)
            model_w = 1.0 - api_w
            logger.info(
                f"Low habit confidence ({habit_confidence:.2f}): "
                f"Increasing API weight to {api_w:.2f}"
            )
        elif habit_confidence > HIGH_ADHERENCE_THRESHOLD:
            model_w = min(0.9, self.model_weight + 0.1)
            api_w   = 1.0 - model_w
            logger.info(
                f"High habit confidence ({habit_confidence:.2f}): "
                f"Increasing ML weight to {model_w:.2f}"
            )
        else:
            model_w = self.model_weight
            api_w   = self.api_weight
        return model_w, api_w

    # ── task matching ─────────────────────────────────────────────────────────

    def _match_tasks(self,
                     ml_schedule: List[Dict],
                     api_schedule: List[Dict]) -> List[Dict]:
        """
        Pair ML tasks with their API counterpart by name.
        Tasks present in only one schedule are kept as singletons.
        """
        api_lookup = {task["task"]: task for task in api_schedule}
        matched    = []

        for ml_task in ml_schedule:
            name = ml_task["task"]
            matched.append({
                "task_name": name,
                "ml_data":   ml_task,
                "api_data":  api_lookup.get(name),
            })

        # API-only tasks (not predicted by ML)
        ml_names = {t["task"] for t in ml_schedule}
        for api_task in api_schedule:
            if api_task["task"] not in ml_names:
                matched.append({
                    "task_name": api_task["task"],
                    "ml_data":   None,
                    "api_data":  api_task,
                })

        return matched

    # ── single-task fusion ────────────────────────────────────────────────────

    def _fuse_single_task(self, task_data: Dict) -> Dict[str, Any]:
        """Compute weighted score and choose the best timing for one task."""
        task_name = task_data["task_name"]
        ml_data   = task_data["ml_data"]
        api_data  = task_data["api_data"]

        # ── API-only ──────────────────────────────────────────────────────────
        if ml_data is None:
            return {
                "task":         task_name,
                "start_time":   api_data["start_time"],
                "end_time":     api_data["end_time"],
                "final_score":  api_data["score"],
                "ml_score":     0.0,
                "api_score":    api_data["score"],
                "ml_influence": 0.0,
                "api_influence":100.0,
                "reason":       api_data.get("reason", "API only"),
                "source":       "API",
            }

        # ── ML-only ───────────────────────────────────────────────────────────
        if api_data is None:
            ml_score = ml_data.get("ml_score", 0.7)
            return {
                "task":         task_name,
                "start_time":   ml_data["start_time"],
                "end_time":     ml_data["end_time"],
                "final_score":  ml_score,
                "ml_score":     ml_score,
                "api_score":    0.0,
                "ml_influence": 100.0,
                "api_influence":0.0,
                "reason":       "ML prediction (no API data)",
                "source":       "ML",
            }

        # ── Both available ────────────────────────────────────────────────────
        ml_score  = float(ml_data.get("ml_score", 0.7))
        api_score = float(api_data.get("score",   0.7))

        final_score = (self.model_weight * ml_score) + (self.api_weight * api_score)

        # Choose timing from whichever source scored higher
        if ml_score >= api_score:
            chosen_start   = ml_data["start_time"]
            chosen_end     = ml_data["end_time"]
            dominant_source = "ML"
        else:
            chosen_start   = api_data["start_time"]
            chosen_end     = api_data["end_time"]
            dominant_source = "API"

        # Influence percentages (avoid div-by-zero)
        if final_score > 0:
            ml_influence  = (self.model_weight * ml_score  / final_score) * 100
        else:
            ml_influence  = 50.0
        api_influence = 100.0 - ml_influence

        reason = self._generate_fusion_reason(
            ml_score, api_score, api_data.get("reason", "")
        )

        return {
            "task":          task_name,
            "start_time":    chosen_start,
            "end_time":      chosen_end,
            "final_score":   round(final_score, 4),
            "ml_score":      ml_score,
            "api_score":     api_score,
            "ml_influence":  round(ml_influence,  1),
            "api_influence": round(api_influence, 1),
            "reason":        reason,
            "source":        dominant_source,
        }

    @staticmethod
    def _generate_fusion_reason(ml_score: float,
                                 api_score: float,
                                 api_reason: str) -> str:
        if ml_score > api_score + 0.2:
            return f"ML-dominant (habit-based) | {api_reason}"
        if api_score > ml_score + 0.2:
            return f"API-corrected (health-based) | {api_reason}"
        return f"Balanced fusion | {api_reason}"

    # ── conflict resolution ───────────────────────────────────────────────────

    def _resolve_conflicts(self, schedule: List[Dict]) -> List[Dict]:
        
        if len(schedule) <= 1:
            return schedule

        schedule = sorted(schedule, key=lambda x: x["start_time"])
        resolved = []
        current  = schedule[0]

        for nxt in schedule[1:]:
            if current["end_time"] <= nxt["start_time"]:
                # No overlap — carry on
                resolved.append(current)
                current = nxt
                continue

            # ── Overlap detected ──────────────────────────────────────────────
            logger.warning(
                f"Overlap detected: {current['task']} and {nxt['task']}"
            )

            # Preserve the original durations BEFORE we move anything
            current_dur = current["end_time"] - current["start_time"]
            nxt_dur     = nxt["end_time"]     - nxt["start_time"]

            if nxt["final_score"] > current["final_score"]:
                # Next task has higher score — shift CURRENT to end before next
                logger.info(f"Resolving: {nxt['task']} wins (higher score)")
                # Current task is pushed to start before next task
                # (it keeps its duration but is squeezed earlier if possible)
                resolved.append(current)
                # Shift next task to start right after current
                new_start          = current["end_time"]
                nxt["start_time"]  = new_start
                nxt["end_time"]    = new_start + nxt_dur
                current            = nxt
            else:
                # Current task wins — push next task to start after current ends
                logger.info(f"Resolving: {current['task']} wins (higher score)")
                new_start          = current["end_time"]
                nxt["start_time"]  = new_start
                nxt["end_time"]    = new_start + nxt_dur
                resolved.append(current)
                current = nxt

        resolved.append(current)
        return resolved


# ── quick self-test ───────────────────────────────────────────────────────────

def _test_fusion_engine():
    ml_schedule = [
        {
            "task":       "Breakfast",
            "start_time": datetime.now().replace(hour=8,  minute=30, second=0, microsecond=0),
            "end_time":   datetime.now().replace(hour=9,  minute=0,  second=0, microsecond=0),
            "ml_score": 0.85, "rl_reward": 0.9, "habit_alignment": 0.8,
            "deadline_satisfaction": 1.0, "priority_score": 0.75,
        },
        {
            "task":       "Exercise",
            "start_time": datetime.now().replace(hour=14, minute=0,  second=0, microsecond=0),
            "end_time":   datetime.now().replace(hour=15, minute=0,  second=0, microsecond=0),
            "ml_score": 0.75, "rl_reward": 0.7, "habit_alignment": 0.8,
            "deadline_satisfaction": 1.0, "priority_score": 0.5,
        },
    ]

    api_schedule = [
        {
            "task":       "Breakfast",
            "start_time": datetime.now().replace(hour=7, minute=30, second=0, microsecond=0),
            "end_time":   datetime.now().replace(hour=8, minute=0,  second=0, microsecond=0),
            "score": 0.95, "reason": "✓ Ideal time for Breakfast",
        },
        {
            "task":       "Exercise",
            "start_time": datetime.now().replace(hour=7, minute=0,  second=0, microsecond=0),
            "end_time":   datetime.now().replace(hour=8, minute=0,  second=0, microsecond=0),
            "score": 1.0,  "reason": "✓ Ideal time for Exercise",
        },
    ]

    engine = ScheduleFusionEngine(model_weight=0.7, api_weight=0.3)
    fused  = engine.combine_schedules(ml_schedule, api_schedule, habit_confidence=0.6)

    print("\n" + "=" * 70)
    print("FUSION ENGINE TEST")
    print("=" * 70)
    for task in fused:
        print(f"\nTask : {task['task']}")
        print(f"  Time        : {task['start_time'].strftime('%H:%M')} - "
              f"{task['end_time'].strftime('%H:%M')}")
        print(f"  Final Score : {task['final_score']:.2f}")
        print(f"  ML  Score   : {task['ml_score']:.2f}  ({task['ml_influence']:.1f}%)")
        print(f"  API Score   : {task['api_score']:.2f}  ({task['api_influence']:.1f}%)")
        print(f"  Reason      : {task['reason']}")


if __name__ == "__main__":
    _test_fusion_engine()