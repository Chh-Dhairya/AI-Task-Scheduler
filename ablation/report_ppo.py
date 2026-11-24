"""
Ablation 3 tables + learning-curve plot, from the runs of run_ppo_all.py.

    python -m ablation.report_ppo

Reads  ablation/results/ppo.csv, ppo_baselines.csv, ppo_curves.csv
Writes ablation/results/ppo_tables.md
       ablation/figures/ppo_learning_curves.png   (3b reward ablation, mean of 3 seeds)
"""
import os

import numpy as np
import pandas as pd

RESULTS = os.path.join("ablation", "results")
FIGURES = os.path.join("ablation", "figures")
NAMES = {"R0": "Full model (R0 = S0 = T0)", "R1": "− habit bonus", "R2": "− buffer bonus",
         "R3": "− priority bonus", "S1": "− mode flag", "S2": "− preferred hour",
         "T1": "Old buggy GAE (B1)", "T2": "No curriculum", "T3": "No reward normalisation",
         "random": "Random order", "habit_sort": "Habit-sort (preferred hour)", "edf": "EDF (earliest deadline)"}
METRICS = [("test_deadline_acc", "Deadline acc.", 100, "%"), ("test_habit_dev", "habit_dev", 1, " min"),
           ("test_habit_acc", "habit_acc (old)", 100, "%"), ("test_avg_reward", "Avg. episode reward", 1, "")]


def stats(df, v, m):
    x = df[(df.variant == v) & (df.metric == m)]["value"].astype(float)
    return x.mean(), x.std(ddof=0), len(x)


def table(df, variants, ref="R0", title=""):
    lines = [f"### {title}", "", "| Variant | " + " | ".join(n for _, n, _, _ in METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    for v in variants:
        cells = []
        for m, _, k, u in METRICS:
            mu, sd, n = stats(df, v, m)
            if np.isnan(mu):
                cells.append("–"); continue
            c = f"{mu * k:.2f}{u}" + (f" ± {sd * k:.2f}" if n > 1 else "")
            if v != ref and ref is not None:
                r_mu, r_sd, _ = stats(df, ref, m)
                d = (mu - r_mu) * k
                big = abs(d) > 2 * max(sd, r_sd) * k          # difference > 2 × seed std
                c += f" ({d:+.2f}{'*' if big else ''})"
            cells.append(c)
        lines.append(f"| {NAMES.get(v, v)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ppo = pd.read_csv(os.path.join(RESULTS, "ppo.csv"))
    base = pd.read_csv(os.path.join(RESULTS, "ppo_baselines.csv"))
    both = pd.concat([ppo, base])
    md = ["# Ablation 3: PPO scheduler (test bank rl_test.json, 2000 scenarios)", "",
          "Mean ± std over seeds 0, 1, 2 (80k episodes each). In brackets: change vs. the full model; "
          "`*` = larger than 2 × the seed std (so not just seed noise). habit_dev: lower is better. "
          "Avg. episode reward is always measured with the FULL reward function.", ""]
    md.append(table(both, ["random", "habit_sort", "edf", "R0"], ref=None, title="3a. Baselines vs. PPO"))
    md.append(table(ppo, ["R0", "R1", "R2", "R3"], title="3b. Reward ablation"))
    md.append(table(ppo, ["R0", "S1", "S2"], title="3c. State ablation"))
    md.append(table(ppo, ["R0", "T1", "T2", "T3"], title="3d. Training-trick ablation"))
    path = os.path.join(RESULTS, "ppo_tables.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    print("\n".join(md)); print(f"→ {path}")
    plot()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pd.read_csv(os.path.join(RESULTS, "ppo_curves.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (metric, label, k) in zip(axes, [("deadline_acc", "Deadline accuracy on rl_val (%)", 100),
                                             ("habit_dev", "habit_dev on rl_val (min)", 1)]):
        for v in ["R0", "R1", "R2", "R3"]:
            d = c[(c.variant == v) & (c.metric == metric)].groupby("episode")["value"]
            mu, sd = d.mean() * k, d.std(ddof=0) * k
            ax.plot(mu.index / 1000, mu, label=NAMES[v], lw=1.6)
            ax.fill_between(mu.index / 1000, mu - sd, mu + sd, alpha=0.15)
        for x in (5, 10):
            ax.axvline(x, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("Training episodes (thousands)"); ax.set_ylabel(label); ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Ablation 3b: PPO learning curves, reward ablation (mean ± std of 3 seeds; dotted = curriculum phase changes)", fontsize=10)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, "ppo_learning_curves.png"); fig.savefig(path, dpi=150)
    print(f"Figure → {path}")


if __name__ == "__main__":
    main()
