import joblib
import numpy as np


# Maps time buckets to representative hours
# Adjust these if your dataset uses different bucket boundaries
BUCKET_TO_HOUR = {0: 7, 1: 11, 2: 15, 3: 19}


class HabitModel:
    """
    Three-level fallback habit model.

    Level 1 — per (person, activity)  : most specific, highest confidence
    Level 2 — per activity only        : fallback when person has no history
    Level 3 — global default           : last resort

    Each profile stores:
        preferred_bucket  : mode time-bucket the person does this activity
        preferred_hour    : representative hour for that bucket
        avg_duration      : mean duration in minutes
        std_duration      : std-dev of duration (spread of habit)
        confidence        : fraction of observations that match preferred bucket
        count             : number of observations used to build this profile
    """

    def __init__(self):
        # (person_encoded, activity_encoded) -> profile dict
        self.person_activity_profile: dict = {}

        # activity_encoded -> profile dict
        self.activity_profile: dict = {}

        # Global fallbacks
        self.global_bucket: int   = 1
        self.global_avg_duration: float = 60.0
        self.global_std_duration: float = 15.0

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, df):
        """
        Build habit profiles from a DataFrame.

        Required columns : Person_Encoded, Activity_Encoded, TimeBucket
        Optional column  : Duration  (minutes)
        """
        duration_col_exists = 'Duration' in df.columns

        # ── Level 1: per person × activity ───────────────────────────────────
        for (person, activity), group in df.groupby(['Person_Encoded', 'Activity_Encoded']):
            bucket_counts    = group['TimeBucket'].value_counts()
            preferred_bucket = int(bucket_counts.idxmax())
            confidence       = float(bucket_counts.max() / len(group))

            if duration_col_exists:
                avg_dur = float(group['Duration'].mean())
                std_dur = float(group['Duration'].std())
                std_dur = 0.0 if np.isnan(std_dur) else std_dur
            else:
                avg_dur = 60.0
                std_dur = 15.0

            self.person_activity_profile[(int(person), int(activity))] = {
                'preferred_bucket': preferred_bucket,
                'preferred_hour':   BUCKET_TO_HOUR.get(preferred_bucket, 9),
                'avg_duration':     avg_dur,
                'std_duration':     std_dur,
                'confidence':       confidence,
                'count':            int(len(group)),
            }

        # ── Level 2: per activity only ────────────────────────────────────────
        for activity, group in df.groupby('Activity_Encoded'):
            bucket_counts    = group['TimeBucket'].value_counts()
            preferred_bucket = int(bucket_counts.idxmax())
            confidence       = float(bucket_counts.max() / len(group))

            if duration_col_exists:
                avg_dur = float(group['Duration'].mean())
                std_dur = float(group['Duration'].std())
                std_dur = 0.0 if np.isnan(std_dur) else std_dur
            else:
                avg_dur = 60.0
                std_dur = 15.0

            self.activity_profile[int(activity)] = {
                'preferred_bucket': preferred_bucket,
                'preferred_hour':   BUCKET_TO_HOUR.get(preferred_bucket, 9),
                'avg_duration':     avg_dur,
                'std_duration':     std_dur,
                'confidence':       confidence,
                'count':            int(len(group)),
            }

        # ── Level 3: global ───────────────────────────────────────────────────
        self.global_bucket = int(df['TimeBucket'].value_counts().idxmax())
        if duration_col_exists:
            self.global_avg_duration = float(df['Duration'].mean())
            self.global_std_duration = float(df['Duration'].std())

        print(f"Habit model trained.")
        print(f"  Person-activity profiles : {len(self.person_activity_profile)}")
        print(f"  Activity-level profiles  : {len(self.activity_profile)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Profile retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def get_profile(self, person_encoded, activity_encoded) -> dict:
        """
        Return the best available habit profile for a person-activity pair.
        Falls back gracefully through the three levels.
        """
        key = (int(person_encoded), int(activity_encoded))

        if key in self.person_activity_profile:
            return {**self.person_activity_profile[key], 'level': 1}

        act_key = int(activity_encoded)
        if act_key in self.activity_profile:
            return {**self.activity_profile[act_key], 'level': 2}

        return {
            'preferred_bucket': self.global_bucket,
            'preferred_hour':   BUCKET_TO_HOUR.get(self.global_bucket, 9),
            'avg_duration':     self.global_avg_duration,
            'std_duration':     self.global_std_duration,
            'confidence':       0.3,
            'count':            0,
            'level':            3,
        }

    def predict_bucket(self, person_encoded, activity_encoded) -> int:
        return self.get_profile(person_encoded, activity_encoded)['preferred_bucket']

    def predict_with_confidence(self, person_encoded, activity_encoded, df=None):
        """
        Returns (bucket, confidence).
        If df is passed, confidence is recalculated from raw data (more accurate).
        """
        profile    = self.get_profile(person_encoded, activity_encoded)
        bucket     = profile['preferred_bucket']
        confidence = profile['confidence']

        if df is not None:
            mask = (
                (df['Person_Encoded']   == int(person_encoded)) &
                (df['Activity_Encoded'] == int(activity_encoded))
            )
            person_act_data = df[mask]['TimeBucket']
            if len(person_act_data) > 0:
                confidence = float(
                    (person_act_data == bucket).sum() / len(person_act_data)
                )

        return bucket, confidence

    def predict_duration(self, person_encoded, activity_encoded) -> tuple:
        """Returns (avg_duration_minutes, std_duration_minutes)."""
        profile = self.get_profile(person_encoded, activity_encoded)
        return profile['avg_duration'], profile['std_duration']

    # ──────────────────────────────────────────────────────────────────────────
    # Summary / inspection
    # ──────────────────────────────────────────────────────────────────────────

    def summarize_person(self, person_encoded, activity_label_map: dict = None) -> str:
        """
        Print a readable summary of one person's full habit profile.

        activity_label_map: optional dict {encoded_int: "ActivityName"}
        """
        lines   = [f"\n── Habit Profile for Person {person_encoded} ──"]
        entries = {
            k: v for k, v in self.person_activity_profile.items()
            if k[0] == int(person_encoded)
        }

        if not entries:
            lines.append("  No personal data found — using activity-level defaults.")
            return "\n".join(lines)

        # Sort by preferred hour so it reads like a daily timeline
        for (person, activity), profile in sorted(entries.items(),
                                                   key=lambda x: x[1]['preferred_hour']):
            act_name = (
                activity_label_map[activity]
                if activity_label_map and activity in activity_label_map
                else f"Activity {activity}"
            )
            lines.append(
                f"  {act_name:<18} | "
                f"Preferred: {profile['preferred_hour']:02d}:00  | "
                f"Avg duration: {profile['avg_duration']:.0f} min  | "
                f"Confidence: {profile['confidence']:.0%}  | "
                f"Observations: {profile['count']}"
            )

        return "\n".join(lines)

    def summarize_all_persons(self, activity_label_map: dict = None) -> str:
        """Print a compact summary of every person in the dataset."""
        person_ids = sorted({k[0] for k in self.person_activity_profile})
        lines = [f"\n{'='*60}", f"  HABIT PROFILES — {len(person_ids)} persons", f"{'='*60}"]
        for pid in person_ids:
            lines.append(self.summarize_person(pid, activity_label_map))
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path="habit_model.pkl"):
        joblib.dump({
            # New full-profile keys
            'person_activity_profile': self.person_activity_profile,
            'activity_profile':        self.activity_profile,
            'global_bucket':           self.global_bucket,
            'global_avg_duration':     self.global_avg_duration,
            'global_std_duration':     self.global_std_duration,
            # Legacy keys — keeps compatibility with old code that reads these
            'person_activity_bucket': {
                k: v['preferred_bucket'] for k, v in self.person_activity_profile.items()
            },
            'activity_bucket': {
                k: v['preferred_bucket'] for k, v in self.activity_profile.items()
            },
        }, path)
        print(f"Habit model saved → {path}")

    def load(self, path="habit_model.pkl"):
        data = joblib.load(path)

        if 'person_activity_profile' in data:
            # New format
            self.person_activity_profile = data['person_activity_profile']
            self.activity_profile        = data['activity_profile']
            self.global_bucket           = data['global_bucket']
            self.global_avg_duration     = data.get('global_avg_duration', 60.0)
            self.global_std_duration     = data.get('global_std_duration', 15.0)
        else:
            # Legacy format — rebuild profiles with default durations
            self.global_bucket = data['global_bucket']
            for k, bucket in data.get('person_activity_bucket', {}).items():
                self.person_activity_profile[k] = {
                    'preferred_bucket': bucket,
                    'preferred_hour':   BUCKET_TO_HOUR.get(bucket, 9),
                    'avg_duration':     60.0,
                    'std_duration':     15.0,
                    'confidence':       0.5,
                    'count':            1,
                }
            for k, bucket in data.get('activity_bucket', {}).items():
                self.activity_profile[k] = {
                    'preferred_bucket': bucket,
                    'preferred_hour':   BUCKET_TO_HOUR.get(bucket, 9),
                    'avg_duration':     60.0,
                    'std_duration':     15.0,
                    'confidence':       0.5,
                    'count':            1,
                }
        print(f"Habit model loaded ← {path}")