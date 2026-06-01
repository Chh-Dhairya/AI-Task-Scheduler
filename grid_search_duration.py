# grid_search_duration.py

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.metrics import mean_absolute_error
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_dataset
from preprocess import DataPreprocessor

# ── GPU CHECK ─────────────────────────────────────────────────────────────────
def verify_gpu():
    try:
        m = XGBRegressor(device="cuda", n_estimators=10)
        m.fit(np.random.rand(50, 3), np.random.rand(50))
        print("=" * 60)
        print("  GPU STATUS : ✅ CUDA is ACTIVE — training on GPU")
        print("=" * 60)
        return "cuda"
    except Exception as e:
        print("=" * 60)
        print("  GPU STATUS : ❌ GPU NOT available — using CPU")
        print(f"  Reason     : {e}")
        print("=" * 60)
        return "cpu"

device = verify_gpu()

# ── 1. Load & preprocess ──────────────────────────────────────────────────────
df = load_dataset("C:\\Users\\user\\OneDrive\\Desktop\\activities_of_daily_living_fixed.csv")
df["DayOfWeek"] = df["Start"].dt.dayofweek

preprocessor = DataPreprocessor()
df = preprocessor.fit_transform(df)

p99 = df["Duration (minutes)"].quantile(0.99)
df  = df[df["Duration (minutes)"] <= p99].copy()

X = df[["Activity_Encoded", "Hour", "DayOfWeek"]].values
y = df["Duration (minutes)"].values

print(f"\n  Dataset size : {len(X)} rows")

# ── 2. Random search config ───────────────────────────────────────────────────
N_ITER   = 50     # ← only 50 random combos instead of 729
N_FOLDS  = 3      # ← 3 folds instead of 5 for speed
TOP_N    = 10
SEED     = 42

# Distributions to sample from (wider range than before)
param_distributions = {
    "n_estimators":     randint(200, 800),          # random int between 200–800
    "max_depth":        randint(3, 8),               # 3–7
    "learning_rate":    uniform(0.01, 0.09),         # 0.01–0.10
    "subsample":        uniform(0.6, 0.4),           # 0.6–1.0
    "colsample_bytree": uniform(0.6, 0.4),           # 0.6–1.0
    "min_child_weight": randint(1, 15),              # 1–14
    "reg_lambda":       uniform(0.5, 2.5),           # 0.5–3.0
    "reg_alpha":        uniform(0.0, 1.0),           # L1 regularization added
}

sampled_params = list(
    ParameterSampler(param_distributions, n_iter=N_ITER, random_state=SEED)
)

print(f"  Sampled combos : {N_ITER}  (out of ~thousands possible)")
print(f"  CV folds       : {N_FOLDS}")
print(f"  Device         : {device.upper()}\n")

# ── 3. Random search loop ─────────────────────────────────────────────────────
kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
results = []

for idx, params in enumerate(sampled_params, 1):

    # Round sampled values to sensible precision
    params["n_estimators"]     = int(params["n_estimators"])
    params["max_depth"]        = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])
    params["learning_rate"]    = round(params["learning_rate"], 4)
    params["subsample"]        = round(params["subsample"], 3)
    params["colsample_bytree"] = round(params["colsample_bytree"], 3)
    params["reg_lambda"]       = round(params["reg_lambda"], 3)
    params["reg_alpha"]        = round(params["reg_alpha"], 3)

    fold_maes = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(
            **params,
            objective             = "reg:squarederror",
            eval_metric           = "mae",
            early_stopping_rounds = 20,
            random_state          = SEED,
            device                = device,
        )
        model.fit(
            X_tr, y_tr,
            eval_set = [(X_val, y_val)],
            verbose  = False,
        )

        fold_maes.append(mean_absolute_error(y_val, model.predict(X_val)))

    mean_mae = round(np.mean(fold_maes), 4)
    std_mae  = round(np.std(fold_maes), 4)
    results.append((mean_mae, std_mae, params))

    # ── Live progress ──────────────────────────────────────────────────────
    best_so_far = min(results, key=lambda x: x[0])
    print(
        f"  [{idx:>2}/{N_ITER}]  "
        f"This combo = {mean_mae:.4f} min  │  "
        f"Best so far = {best_so_far[0]:.4f} min  │  "
        f"Params: depth={params['max_depth']} "
        f"lr={params['learning_rate']} "
        f"trees={params['n_estimators']} "
        f"λ={params['reg_lambda']}"
    )

# ── 4. Sort ────────────────────────────────────────────────────────────────────
results.sort(key=lambda x: x[0])

# ── 5. Top-N table ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  TOP {TOP_N} CONFIGURATIONS")
print(f"{'=' * 60}")

for rank, (mae, std, params) in enumerate(results[:TOP_N], 1):
    print(f"\n  Rank #{rank}  ──  MAE = {mae:.4f} min  (± {std:.4f})")
    print(f"  {'Parameter':<22}  Value")
    print(f"  {'-'*35}")
    for k, v in params.items():
        print(f"  {k:<22}  {v}")

# ── 6. Best config — paste-ready ──────────────────────────────────────────────
best_mae, best_std, best_params = results[0]
print(f"\n{'=' * 60}")
print(f"  🏆 BEST CONFIG  —  MAE = {best_mae:.4f} min  (± {best_std:.4f})")
print(f"{'=' * 60}")
print("\n  ✂  Paste this directly into duration_model.py:\n")
print("  self.model = XGBRegressor(")
for k, v in best_params.items():
    val = f'"{v}"' if isinstance(v, str) else v
    print(f"      {k:<22} = {val},")
print('      objective             = "reg:squarederror",')
print(f'      device                = "{device}",')
print('      early_stopping_rounds = 20,')
print('      eval_metric           = "mae",')
print('      random_state          = 42,')
print("  )")

# ── 7. Save CSV ────────────────────────────────────────────────────────────────
rows = []
for mae, std, params in results:
    row = {"mean_mae": mae, "std_mae": std}
    row.update(params)
    rows.append(row)

pd.DataFrame(rows).to_csv("grid_search_results.csv", index=False)
print(f"\n  All {N_ITER} results saved → grid_search_results.csv\n")