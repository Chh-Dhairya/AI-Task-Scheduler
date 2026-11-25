"""Shared helpers for the ablation study (seeds, JSON I/O)."""
import json
import random

import numpy as np
# XGBoost MUST be imported before torch: both ship their own libomp, and on macOS
# loading an XGBoost model after torch crashes Python (segfault). The other order
# is safe as long as torch inference runs single-threaded (see scheduler._rl_order).
import xgboost  # noqa: F401
import torch


def set_seed(seed, threads=2):
    """
    Seed every random source used in this project (Python, NumPy, PyTorch).

    Also fixes the PyTorch CPU thread count: with a different number of threads
    the float sums run in a different order, and over a training run the tiny
    differences grow, so the same seed gives different results. 2 threads lets
    4 runs share an 8-core machine and still match a run started on its own.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


# ── B9: honest splits ─────────────────────────────────────────────────────────
# The data covers Jan–Jun 2023. A time-based split: never train or tune on June.
TRAIN_END = "2023-05-01"   # train: Jan–Apr
VAL_END   = "2023-06-01"   # val:   May      test: June


def time_split(df):
    """Return (train, val, test) by the 'Start' timestamp: Jan–Apr / May / June."""
    train = df[df["Start"] < TRAIN_END]
    val   = df[(df["Start"] >= TRAIN_END) & (df["Start"] < VAL_END)]
    test  = df[df["Start"] >= VAL_END]
    return train, val, test


def held_out_persons(df, n=50, seed=7):
    """50 persons kept out of training entirely, for the cold-start test (Habit H1)."""
    persons = np.sort(df["Person ID"].unique())
    return set(np.random.RandomState(seed).choice(persons, n, replace=False).tolist())


def load_json(p):
    with open(p) as f:
        return json.load(f)


def save_json(o, p):
    with open(p, "w") as f:
        json.dump(o, f, indent=1)
