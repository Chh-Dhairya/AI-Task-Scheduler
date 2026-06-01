# AI-Driven Personalized Life Planner

An intelligent task scheduling system that combines **Machine Learning (XGBoost)**, **Reinforcement Learning (PPO)**, and **Large Language Models (Gemini)** to generate personalized, adaptive, and realistic daily schedules.

## Overview

Traditional productivity tools rely on manual planning and static scheduling. This project introduces a hybrid AI framework that predicts task durations, learns user habits, optimizes task sequencing, and refines schedules using semantic reasoning.

The system aims to maximize productivity while respecting deadlines, user preferences, and real-world behavioral constraints.

## Key Features

* **Task Duration Prediction** using XGBoost regression
* **Habit Learning Module** for personalized scheduling
* **PPO-based Reinforcement Learning Scheduler**
* **LLM-based Behavioral Correction** using Gemini API
* **Adaptive Fusion Engine** for combining ML, RL, and LLM outputs
* **Streamlit Dashboard** for user interaction

## System Pipeline

```text
User Tasks
    ↓
Input Processing
    ↓
Habit Learning
    ↓
Duration Prediction (XGBoost)
    ↓
RL Scheduler (PPO)
    ↓
LLM Schedule Refinement
    ↓
Fusion Engine
    ↓
Final Optimized Schedule
```

## Tech Stack

| Category        | Technologies                    |
| --------------- | ------------------------------- |
| Language        | Python                          |
| ML              | Scikit-Learn, XGBoost           |
| RL              | Stable-Baselines3, PPO, PyTorch |
| Data Processing | Pandas, NumPy                   |
| LLM             | Gemini API                      |
| UI              | Streamlit                       |
| Development     | VS Code                         |

## Reinforcement Learning Formulation

### State Space

* Current time
* Task features
* Completion status
* User habit information

### Actions

* Select next task to schedule

### Reward Function

* Deadline satisfaction
* Priority completion
* Habit alignment
* Early completion bonuses
* Penalties for invalid scheduling and missed deadlines

### Algorithm

* Proximal Policy Optimization (PPO)

## Evaluation Metrics

* Mean Absolute Error (MAE)
* Habit Accuracy
* Deadline Accuracy
* Scheduling Success Rate
* RL Reward
* Behavioral Compliance Score
* Composite Schedule Score

## Results

| Metric                  | Value        |
| ----------------------- | ------------ |
| Duration Prediction MAE | ~18.49 min   |
| Habit Accuracy          | ~81.72%      |
| Composite Score         | 0.949        |
| Scheduler               | PPO-based RL |

The hybrid architecture successfully balances task deadlines, user habits, and schedule feasibility while generating realistic and personalized schedules.

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── data/
├── models/
├── modules/
│   ├── preprocessing.py
│   ├── habit_learning.py
│   ├── duration_prediction.py
│   ├── rl_scheduler.py
│   ├── llm_correction.py
│   └── fusion_engine.py
├── screenshots/
├── Project_Report.pdf
└── README.md
```

## Installation

```bash
git clone <repository-url>
cd AI-Life-Planner

python -m venv venv
source venv/bin/activate   # Linux/macOS

# or

venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Future Work

* Google Calendar & Outlook integration
* Real-time schedule adaptation
* Multi-day planning
* Mobile application support
* Context-aware scheduling
* Multi-agent planning framework

## Authors

* Dhairya Chhabra
* Karan Dhingra
* Nalin Chandola

**Mentor:** Dr. Malika Acharya
Department of Electronics and Communication Engineering
The LNM Institute of Information Technology (LNMIIT), Jaipur

## Project Report

For detailed methodology, mathematical formulation, experiments, and results, refer to **Project_Report.pdf** included in this repository.
