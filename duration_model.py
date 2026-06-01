import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class DurationModel:
    def __init__(self):
        self.model = None
        self.cap = None   # stores 99th percentile for clipping predictions

    def train(self, df):
        # Clip extreme durations at 99th percentile — outliers wreck MAE
        p99 = df["Duration (minutes)"].quantile(0.99)
        self.cap = p99
        df = df[df["Duration (minutes)"] <= p99].copy()

        X = df[["Activity_Encoded", "Hour", "DayOfWeek"]]
        y = df["Duration (minutes)"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = XGBRegressor(
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
            eval_metric="mae"
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"\nDuration Model MAE: {round(mae, 2)} minutes")

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