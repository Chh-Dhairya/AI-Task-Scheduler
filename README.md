# AI-Driven Personalized Life Planner

A hybrid AI task scheduler that builds a personalised daily schedule. It learns **when** a person usually does each activity, predicts **how long** each one takes, uses a **PPO reinforcement-learning agent** to decide the order, and corrects unhealthy timings with an **LLM (Gemini) plus a rule engine**. A fusion engine merges the ML and LLM schedules, and the result is shown in a Streamlit dashboard or a command-line interface.

## Pipeline

```text
User tasks (activity, priority, deadline)
    │
    ├─► Habit model ─────────► preferred time + confidence      (habit_model.py)
    ├─► Duration model ──────► minutes per task (XGBoost)        (duration_model.py)
    │
    ▼
PPO scheduler ───────────────► task order + placement 07:00–22:00 (scheduler.py, rl_agent.py)
    │   = ML schedule                                            (ml_pipeline.py)
    ▼
Gemini / rule fallback ──────► health-corrected times + score    (api_scheduler.py, config.py)
    │
    ▼
Fusion engine ───────────────► final = w_ML·ML + w_API·API,       (fusion_engine.py)
    │                            overlaps resolved by score
    ▼
Streamlit dashboard (app.py) / CLI (main.py)
```

## Components

| Component | File(s) | Method |
| --- | --- | --- |
| Data loading and features | `data_loader.py`, `preprocess.py` | Timestamps, 5 time buckets (shared `BUCKET_TO_HOUR`), label encoding, day of week |
| Habit model | `habit_model.py`, `train_habit.py` | 3-level frequency lookup: person × activity → activity → global, with confidence |
| Duration model | `duration_model.py`, `train_duration.py` | XGBoost regressor on activity, hour and weekday; p99 cap; early stopping on a validation month |
| RL environment | `rl_environment.py` | 62-dim state, masked discrete actions, habit/priority-mode shaped reward, priority 1–4 |
| PPO agent | `rl_agent.py` | Actor-critic, clipped objective, GAE (reset at episode ends), linear entropy decay, cosine LR, reward scaling |
| PPO training | `train_rl_scheduler.py`, `grid_search.py` | Curriculum (≤4 → ≤7 → ≤10 tasks), seeded runs, fixed validation/test scenario banks |
| Scheduler | `scheduler.py`, `ml_pipeline.py` | PPO ordering and habit-first placement; the same code is used by the app and the CLI |
| LLM correction | `api_scheduler.py`, `config.py` | Gemini (temperature 0, JSON mode) receives the ML start times; a rule engine with midnight-wrapping windows is the fallback |
| Fusion | `fusion_engine.py` | Weighted score, adaptive weights from habit confidence, weighted decision rule, score-aware overlap resolution |
| Interfaces | `app.py`, `main.py` | Streamlit dashboard (timeline, analytics, CSV export) and an interactive CLI |
| Ablation study | `ablation/` | Experiments for every component (see below) |

## Repository layout

```text
.
├── app.py                     # Streamlit dashboard
├── main.py                    # CLI
├── ml_pipeline.py             # shared ML-side pipeline (PPO schedule → API input)
├── config.py                  # weights, health rules, Gemini settings (.env)
├── data_loader.py, preprocess.py
├── habit_model.py, duration_model.py
├── rl_environment.py, rl_agent.py, scheduler.py
├── api_scheduler.py, fusion_engine.py
├── train_habit.py, train_duration.py, train_rl_scheduler.py
├── grid_search.py, grid_search_duration.py   # hyperparameter search
├── fix_dataset.py             # one-off dataset generator (already run; refuses to re-run)
├── habit_model.pkl, duration_model.pkl, rl_scheduler.pth   # trained models
├── activities_of_daily_living_fixed.csv                  # dataset
├── ablation/                  # ablation study: scripts, scenario banks, cache, figures
├── requirements.txt, requirements.lock
└── README.md
```

## Setup

```bash
git clone https://github.com/Chh-Dhairya/AI-Task-Scheduler.git
cd AI-Task-Scheduler
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # or requirements.lock for the exact tested versions
```

### Gemini API key (optional)

Create a file named `.env` in the project folder. It is gitignored, so never commit it.

```text
GEMINI_API_KEY="your-key"
# GEMINI_MODEL="gemini-3.6-flash"   # optional override
```

Without a key the app still runs: the rule engine in `api_scheduler.py` takes over the LLM's role.

## Run

```bash
streamlit run app.py
```

```bash
python main.py
```

The trained models are included, so both work right after installation.

## Train the models

```bash
python train_habit.py
```

```bash
python train_duration.py
```

```bash
python train_rl_scheduler.py --seed 0 --episodes 80000
```

The habit and duration models use a time-based split: train on January–April, validate on May, and test on June. The habit model also reports a cold-start test on 50 held-out people. The PPO agent picks its checkpoint on a fixed validation bank (`ablation/data/rl_val.json`) and reports its score once on a separate test bank (`ablation/data/rl_test.json`).

## Ablation study

Each script in `ablation/` removes or swaps one part of the system and measures the effect. Run them from the project root:

| Ablation | Command |
| --- | --- |
| 1. Duration model | `python -m ablation.run_duration` |
| 2. Habit model (personalisation, fallback levels, time resolution) | `python -m ablation.run_habit` |
| 3. PPO scheduler (variants × seeds, rule baselines) | `python -m ablation.run_ppo_all`, then `python -m ablation.report_ppo` |
| 4. LLM correction | `python -m ablation.run_llm --no-llm` (offline) |
| 5. Fusion engine | `python -m ablation.run_fusion` |
| Full system, one part removed at a time | `python -m ablation.run_full` |

Figures are in `ablation/figures/`. Cached Gemini responses in `ablation/cache/` make the LLM runs reproducible without a key.

## Dataset

`activities_of_daily_living_fixed.csv` has 50,000 activity logs from 500 people across 10 activities, January–June 2023. The start times were regenerated by `fix_dataset.py` so that each person has a consistent preferred time per activity. This makes it a semi-synthetic benchmark for checking the pipeline against known habits.

## Future work

- Google Calendar and Outlook integration
- Real-time rescheduling and multi-day planning
- Learning fusion weights from user feedback
- Validation on real (non-synthetic) activity logs

## Authors

- Dhairya Chhabra
- Karan Dhingra
- Nalin Chandola

**Mentor:** Dr. Malika Acharya
Department of Electronics and Communication Engineering
The LNM Institute of Information Technology (LNMIIT), Jaipur

## License

MIT, see [LICENSE](LICENSE).
