"""
Train the habit model with an honest time split (B9):
  fit on Jan–May (train + val), measure bucket accuracy on June (test).
Also a cold-start test: 50 held-out persons are left out of a second fit, and
their June rows are predicted from the activity-level fallback.
"""
import contextlib
import io

import pandas as pd

from data_loader import load_dataset
from preprocess import DataPreprocessor
from habit_model import HabitModel
from ablation.common import time_split, held_out_persons

DATA_PATH = "activities_of_daily_living_fixed.csv"


def bucket_accuracy(model, rows):
    pred = [model.predict_bucket(p, a) for p, a in zip(rows["Person_Encoded"], rows["Activity_Encoded"])]
    return float((pd.Series(pred, index=rows.index) == rows["TimeBucket"]).mean())


df = load_dataset(DATA_PATH)
preprocessor = DataPreprocessor()
df = preprocessor.fit_transform(df)

train, val, test = time_split(df)
fit_df = pd.concat([train, val])
print(f"Rows: fit (Jan–May) {len(fit_df)} | test (June) {len(test)}")

model = HabitModel()
model.train(fit_df)
model.save("habit_model.pkl")

print(f"\nTraining accuracy (Jan–May, old metric, optimistic): {bucket_accuracy(model, fit_df):.1%}")
print(f"TEST accuracy (June):                                 {bucket_accuracy(model, test):.1%}")

# Cold start: persons never seen in training
held = held_out_persons(df)
with contextlib.redirect_stdout(io.StringIO()):
    cold = HabitModel()
    cold.train(fit_df[~fit_df["Person ID"].isin(held)])
held_test = test[test["Person ID"].isin(held)]
print(f"TEST accuracy, {len(held)} held-out persons (cold start, {len(held_test)} rows): "
      f"{bucket_accuracy(cold, held_test):.1%}")
