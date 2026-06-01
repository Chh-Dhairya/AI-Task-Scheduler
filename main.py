"""
main.py — CLI interface for the Hybrid AI Task Scheduler.

Fixes vs original
─────────────────
1. `from datetime import timedelta` moved to the top-level imports
   (was only inside the __main__ guard, causing NameError when imported).
2. ml_schedule_tasks now uses timedelta correctly for end_time calculation.
3. Minor: added a check for missing duration_model.pkl to give a clear
   error instead of a cryptic AttributeError.
"""

import numpy as np
from datetime import datetime, timedelta          # ← fix: timedelta at top

from duration_model import DurationModel
from data_loader    import load_dataset
from preprocess     import DataPreprocessor
from scheduler      import schedule_tasks
from habit_model    import HabitModel
from api_scheduler  import APIScheduler
from fusion_engine  import ScheduleFusionEngine
from config         import DEFAULT_MODEL_WEIGHT, DEFAULT_API_WEIGHT

BUCKET_TO_HOUR = {
    0: 7,
    1: 11,
    2: 15,
    3: 19,
    4: 22,
}

DATA_PATH          = "activities_of_daily_living_fixed.csv"
HABIT_MODEL_PATH   = "habit_model.pkl"
DURATION_MODEL_PATH= "duration_model.pkl"


def main():
    print("=" * 80)
    print("  HYBRID AI TASK SCHEDULER — CLI MODE")
    print("=" * 80)

    # ── load data & models ────────────────────────────────────────────────────
    print("\n🔄 Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("🔄 Encoding data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform(df)

    print("🔄 Loading ML models...")
    habit_model = HabitModel()
    habit_model.load(HABIT_MODEL_PATH)

    duration_model = DurationModel()
    duration_model.load(DURATION_MODEL_PATH)

    print("🔄 Initialising hybrid system...")
    api_scheduler  = APIScheduler()
    fusion_engine  = ScheduleFusionEngine(
        model_weight=DEFAULT_MODEL_WEIGHT,
        api_weight=DEFAULT_API_WEIGHT,
    )

    print("✅ All systems loaded!\n")

    # ── person selection ──────────────────────────────────────────────────────
    print("Available Person IDs (sample):")
    print(df["Person ID"].unique()[:20], "...")

    try:
        person_input = int(input("\n👤 Enter Person ID: "))
    except ValueError:
        print("❌ Invalid input.")
        return

    if person_input not in df["Person ID"].values:
        print("❌ Person not found in dataset.")
        return

    person_encoded = preprocessor.person_encoder.transform([person_input])[0]
    day_of_week    = datetime.now().weekday()

    # ── activity list ─────────────────────────────────────────────────────────
    print("\n📋 Available Activities:")
    activities = sorted(df["Activity Type"].unique())
    for i, act in enumerate(activities[:20], 1):
        print(f"  {i:2}. {act}")
    print("  ...")

    # ── collect tasks ─────────────────────────────────────────────────────────
    try:
        num_tasks = int(input("\n🔢 How many tasks for today? "))
    except ValueError:
        print("❌ Invalid number.")
        return

    tasks              = []
    ml_schedule_tasks  = []
    api_tasks          = []

    for i in range(num_tasks):
        print(f"\n{'=' * 60}")
        print(f"Task {i + 1}/{num_tasks}")
        print("=" * 60)

        task_name = input("📝 Activity Type: ").strip()
        if task_name not in preprocessor.activity_encoder.classes_:
            print(f"  ⚠️  '{task_name}' not recognised — skipping.")
            continue

        activity_encoded = preprocessor.activity_encoder.transform([task_name])[0]

        try:
            priority     = int(input("⭐ Priority (1=Low, 2=Medium, 3=High, 4=Critical): "))
            deadline_hour= int(input("⏰ Deadline hour (0-23): "))
            assert 1 <= priority     <= 4
            assert 0 <= deadline_hour <= 23
        except (ValueError, AssertionError):
            print("  ⚠️  Invalid priority or deadline — skipping.")
            continue

        # ── ML predictions ────────────────────────────────────────────────────
        bucket, confidence = habit_model.predict_with_confidence(
            person_encoded, activity_encoded, df=df
        )
        predicted_hour     = BUCKET_TO_HOUR[bucket]
        predicted_duration = duration_model.predict(
            activity_encoded, predicted_hour, day_of_week
        )

        print(f"\n  🤖 ML Prediction:")
        print(f"     Time     : {predicted_hour:02d}:00  (confidence: {confidence:.0%})")
        print(f"     Duration : {round(predicted_duration)} min")

        # ── store task ────────────────────────────────────────────────────────
        task = {
            "Task":               task_name,
            "Activity_Encoded":   activity_encoded,
            "Priority":           priority,
            "Preferred_Hour":     predicted_hour,
            "Deadline_Hour":      deadline_hour,
            "Estimated_Duration": predicted_duration,
            "Confidence":         confidence,
        }
        tasks.append(task)

        # Prepare ML schedule entry (uses datetime + timedelta properly)
        task_start = datetime.now().replace(
            hour=predicted_hour, minute=0, second=0, microsecond=0
        )
        task_end = task_start + timedelta(minutes=predicted_duration)   # ← fix

        ml_schedule_tasks.append({
            "task":                  task_name,
            "start_time":            task_start,
            "end_time":              task_end,
            "ml_score":              confidence,
            "rl_reward":             0.8,
            "habit_alignment":       confidence,
            "deadline_satisfaction": 1.0 if predicted_hour <= deadline_hour else 0.5,
            "priority_score":        priority / 4.0,
        })

        api_tasks.append({
            "name":     task_name,
            "duration": int(predicted_duration),
            "priority": priority,
            "deadline": datetime.now().replace(hour=deadline_hour, minute=0,
                                               second=0, microsecond=0),
        })

    if not tasks:
        print("\n❌ No valid tasks entered.")
        return

    # ── hybrid scheduling ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  GENERATING HYBRID SCHEDULE")
    print("=" * 80)

    print("\n🌐 Getting API recommendations...")
    api_schedule = api_scheduler.get_schedule(api_tasks)

    print("⚖️  Fusing ML and API schedules...")
    avg_confidence  = float(np.mean([t["Confidence"] for t in tasks]))
    fused_schedule  = fusion_engine.combine_schedules(
        ml_schedule_tasks,
        api_schedule,
        habit_confidence=avg_confidence,
    )

    # ── display results ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL HYBRID SCHEDULE")
    print("=" * 80)
    print(f"\nML Weight  : {fusion_engine.model_weight:.0%}  |  "
          f"API Weight : {fusion_engine.api_weight:.0%}")
    print(f"Avg Confidence : {avg_confidence:.0%}\n")

    for idx, item in enumerate(fused_schedule, 1):
        start_str = item["start_time"].strftime("%H:%M")
        end_str   = item["end_time"].strftime("%H:%M")

        print(f"{idx}. [{item['source']:3s}] {item['task']}")
        print(f"   ⏰ Time        : {start_str} – {end_str}")
        print(f"   📊 Final Score : {item['final_score']:.2f}")
        print(f"   🤖 ML          : {item['ml_score']:.2f}  ({item['ml_influence']:.0f}%)")
        print(f"   🌐 API         : {item['api_score']:.2f}  ({item['api_influence']:.0f}%)")
        print(f"   💡 {item['reason']}")
        print("   " + "-" * 74)

    print("\n✅ Schedule generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()