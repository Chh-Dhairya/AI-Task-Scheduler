import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["Activity_Encoded", "Hour", "DayOfWeek"]
TARGET   = "Duration (minutes)"

# Production XGBoost settings (shared with ablation/run_duration.py, so the
# ablation's "current model" variant is exactly this model).
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=400,          # more trees for better fit
    max_depth=4,               # slightly deeper for interaction terms
    learning_rate=0.03,        # lower LR with more trees
    colsample_bytree=0.8,
    subsample=0.8,             # row subsampling reduces overfitting
    min_child_weight=5,        # prevents fitting on tiny groups
    reg_lambda=1.5,            # L2 regularization
    random_state=42,
    early_stopping_rounds=20,
    eval_metric="mae",
)


class DurationModel:
    def __init__(self):
        self.model = None
        self.cap = None   # stores 99th percentile for clipping predictions

    def train(self, train_df, val_df):
        """
        B9: honest training.
          • p99 cap computed on TRAIN only (outliers wreck MAE),
          • early stopping on VAL (never on the test set).
        Report test error separately with evaluate(test_df).
        """
        p99 = float(train_df[TARGET].quantile(0.99))
        self.cap = p99
        tr = train_df[train_df[TARGET] <= p99]
        va = val_df[val_df[TARGET] <= p99]       # same cap, so the stopping loss matches training

        X_train, y_train = tr[FEATURES], tr[TARGET]
        X_val,   y_val   = va[FEATURES], va[TARGET]

        self.model = XGBRegressor(**XGB_PARAMS)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],           # early stopping on validation
            verbose=False
        )
        val_mae = mean_absolute_error(y_val, self.predict_batch(va))
        print(f"Duration model: cap(p99, train) = {p99:.0f} min | "
              f"best iteration {self.model.best_iteration} | val MAE {val_mae:.2f} min")

    def predict_batch(self, df) -> np.ndarray:
        """Predictions for every row of df (clipped like predict())."""
        pred = self.model.predict(df[FEATURES])
        return np.clip(pred, 1.0, self.cap if self.cap else 480.0)

    def evaluate(self, df) -> dict:
        """MAE / RMSE / % within ±15 min on ALL rows of df (no outlier removal)."""
        y, p = df[TARGET].to_numpy(dtype=float), self.predict_batch(df)
        err = np.abs(y - p)
        return {"mae": float(err.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
                "within_15": float((err <= 15).mean()), "n": int(len(df))}

    def save(self, path="duration_model.pkl"):
        joblib.dump({"model": self.model, "cap": self.cap}, path)

    def load(self, path="duration_model.pkl"):
        data = joblib.load(path)
        self.model = data["model"]
        self.cap = data["cap"]

    def predict(self, activity_encoded, hour, day_of_week):
        input_df = pd.DataFrame({
            "Activity_Encoded": [activity_encoded],
            "Hour":             [hour],
            "DayOfWeek":        [day_of_week]
        })
        pred = float(self.model.predict(input_df)[0])
        # Clip prediction to valid range
        return float(np.clip(pred, 1.0, self.cap if self.cap else 480.0))