"""
Train the XGBoost duration model with an honest time split (B9):
  train Jan–Apr  → fit (p99 cap from train only)
  val   May      → early stopping
  test  June     → reported MAE (never used for training or tuning)
"""
from data_loader import load_dataset
from preprocess import DataPreprocessor
from duration_model import DurationModel, TARGET
from ablation.common import time_split

DATA_PATH = "activities_of_daily_living_fixed.csv"

df = load_dataset(DATA_PATH)
preprocessor = DataPreprocessor()
df = preprocessor.fit_transform(df)          # also adds DayOfWeek

train, val, test = time_split(df)
print(f"Rows: train {len(train)} | val {len(val)} | test {len(test)}")

duration_model = DurationModel()
duration_model.train(train, val)

res = duration_model.evaluate(test)
capped = duration_model.evaluate(test[test[TARGET] <= duration_model.cap])
print(f"TEST (June, all {res['n']} rows): MAE {res['mae']:.2f} min | RMSE {res['rmse']:.2f} | "
      f"within ±15 min {res['within_15']:.1%}")
print(f"TEST (June, rows ≤ cap, {capped['n']} rows; comparable to the old report number): "
      f"MAE {capped['mae']:.2f} min")

duration_model.save("duration_model.pkl")
print("Saved → duration_model.pkl")
