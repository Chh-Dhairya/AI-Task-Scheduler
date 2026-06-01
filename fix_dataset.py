import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv('C:\\Users\\user\\OneDrive\\Desktop\\task_scheduler\\activities_of_daily_living_fixed.csv')

ACTIVITY_BASE_HOURS = {
    'Working':     9,
    'Eating':      13,
    'Sleeping':    23,
    'Exercising':  7,
    'Cooking':     18,
    'Relaxing':    20,
    'Shopping':    11,
    'Cleaning':    10,
    'Walking':     8,
    'Socializing': 19,
}

DURATION_PROFILES = {
    'Working':     (30, 240, 120, 40),
    'Eating':      (10,  60,  25,  8),
    'Sleeping':    (360, 540, 420, 60),
    'Exercising':  (15,  90,  45, 15),
    'Cooking':     (10,  90,  35, 15),
    'Relaxing':    (15, 120,  55, 25),
    'Shopping':    (15, 120,  50, 20),
    'Cleaning':    (10,  90,  40, 20),
    'Walking':     (10,  90,  35, 15),
    'Socializing': (20, 180,  75, 35),
}

persons   = df['Person ID'].unique()
activities = df['Activity Type'].unique()

# Assign each (person, activity) pair a fixed preferred hour — sampled ONCE
# This is what creates learnable per-person habits
preferred_hours = {}
for pid in persons:
    person_shift = np.random.uniform(-1.5, 1.5)  # early bird vs night owl
    for act in activities:
        base = ACTIVITY_BASE_HOURS.get(act, 12)
        # Each person has a consistent offset for this activity
        act_shift  = np.random.uniform(-0.5, 0.5)
        preferred  = int(round(base + person_shift + act_shift)) % 24
        # Keep sleeping in evening range
        if act == 'Sleeping' and preferred < 20:
            preferred = 22
        preferred_hours[(pid, act)] = preferred

def sample_duration(activity):
    min_d, max_d, mean, std = DURATION_PROFILES.get(activity, (10, 120, 45, 20))
    return int(np.clip(int(np.random.normal(mean, std)), min_d, max_d))

def make_time_str(h, m):
    return f'{h:02d}:{m:02d}'

new_rows = []
for _, row in df.iterrows():
    activity  = row['Activity Type']
    person_id = row['Person ID']

    # Use the fixed preferred hour for this person-activity pair
    # Add small noise (±30 min) so it's not perfectly rigid
    pref_hour = preferred_hours[(person_id, activity)]
    noise_min = np.random.randint(-30, 31)
    total_min = pref_hour * 60 + noise_min
    hour      = (total_min // 60) % 24
    minute    = total_min % 60
    if minute < 0:
        minute += 60
        hour   = (hour - 1) % 24

    duration  = sample_duration(activity)
    end_total = hour * 60 + minute + duration

    new_rows.append({
        'Person ID':          person_id,
        'Activity Type':      activity,
        'Start Time':         make_time_str(hour, minute),
        'End Time':           make_time_str((end_total // 60) % 24, end_total % 60),
        'Duration (minutes)': duration,
        'Date':               row['Date']
    })

df_fixed = pd.DataFrame(new_rows)
df_fixed.to_csv('activities_of_daily_living_fixed.csv', index=False)
print("Saved to activities_of_daily_living_fixed.csv")

# Verify
df_check = pd.read_csv('activities_of_daily_living_fixed.csv')
df_check['Hour'] = df_check['Start Time'].apply(lambda x: int(x.split(':')[0]))

print("\nAverage start hour per activity:")
print(df_check.groupby('Activity Type')['Hour'].mean().round(1).sort_values())

print("\nPer-person consistency (should be 0.7+):")
from preprocess import get_time_bucket
df_check['TimeBucket'] = df_check['Hour'].apply(get_time_bucket)
consistency = df_check.groupby(
    ['Person ID', 'Activity Type']
)['TimeBucket'].agg(lambda x: x.value_counts().iloc[0] / len(x))
print(f'Average: {consistency.mean():.3f}')
print(f'Median:  {consistency.median():.3f}')

print("\nSample — Person 1 preferred hours per activity:")
p1 = df_check[df_check['Person ID'] == 1]
print(p1.groupby('Activity Type')['Hour'].mean().round(1).sort_values())