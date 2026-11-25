"""
C0: build the fixed end-to-end test bank ONCE (used by Ablations 4, 5 and the
full-system comparison):

    python -m ablation.make_e2e_bank

150 "days to schedule" from JUNE data (never seen by the habit / duration models):
  • 150 random persons that have ≥5 distinct non-sleep activities in June,
  • 5 distinct activities (Sleeping excluded), one real June occurrence each,
  • true_start / true_duration = what the person actually did,
  • deadline = ceil(true_end + slack), slack ~ U(0.5, 3) h, capped at 23,
  • priority ~ randint(1, 5)  (1-4).
Note: the 5 tasks come from different June days (no person has 5 activities on
one day in this dataset), so a "day" is a set of real tasks, not a real diary.
Seed 1003 is fixed; an existing bank is never overwritten.
"""
import math
import os

import numpy as np

from ablation.common import save_json, time_split
from data_loader import load_dataset
from preprocess import DataPreprocessor

OUT = os.path.join(os.path.dirname(__file__), "data", "e2e_bank.json")
SEED, N_SCENARIOS, N_TASKS = 1003, 150, 5
DATA_PATH = "activities_of_daily_living_fixed.csv"

if __name__ == "__main__":
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} already exists, not regenerating.")
    df = DataPreprocessor().fit_transform(load_dataset(DATA_PATH))
    _, _, june = time_split(df)
    june = june[june["Activity Type"] != "Sleeping"]

    n_act = june.groupby("Person ID")["Activity Type"].nunique()
    eligible = np.sort(n_act[n_act >= N_TASKS].index.to_numpy())

    np.random.seed(SEED)
    persons = np.random.choice(eligible, N_SCENARIOS, replace=False)
    bank = []
    for pid in persons:
        rows = june[june["Person ID"] == pid]
        acts = np.random.choice(np.sort(rows["Activity Type"].unique()), N_TASKS, replace=False)
        tasks = []
        for act in acts:
            r = rows[rows["Activity Type"] == act].iloc[np.random.randint((rows["Activity Type"] == act).sum())]
            start_h = r["Start"].hour + r["Start"].minute / 60
            dur = int(r["Duration (minutes)"])
            slack = np.random.uniform(0.5, 3.0)
            tasks.append({
                "Task":             str(act),
                "Person ID":        int(pid),
                "Person_Encoded":   int(r["Person_Encoded"]),
                "Activity_Encoded": int(r["Activity_Encoded"]),
                "Date":             r["Start"].strftime("%Y-%m-%d"),
                "DayOfWeek":        int(r["DayOfWeek"]),
                "true_start":       round(float(start_h), 4),          # hours, e.g. 9.25 = 09:15
                "true_duration":    dur,                               # minutes
                "Deadline_Hour":    int(min(23, math.ceil(start_h + dur / 60 + slack))),
                "Priority":         int(np.random.randint(1, 5)),
            })
        bank.append(tasks)

    save_json(bank, OUT)
    print(f"Saved {len(bank)} scenarios × {N_TASKS} tasks → {OUT}  (seed {SEED}, "
          f"{len(eligible)} eligible persons)")
