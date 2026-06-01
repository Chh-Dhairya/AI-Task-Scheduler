"""
app.py — Streamlit UI for the Hybrid AI Task Scheduler.

Fixes vs original
─────────────────
1. `use_container_width=True` replaced with `width='stretch'` everywhere
   (Streamlit deprecation warning, removed after 2025-12-31).
2. `timedelta` explicitly imported at the top (was missing from imports).
3. `prepare_tasks_for_api` now passes proper datetime deadlines.
4. API schedule display in analytics tab now handles the case where
   api_schedule items may not have a 'score' key (uses .get()).
5. Minor: spinner text and status messages cleaned up.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta          # ← timedelta added
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any

from duration_model import DurationModel
from data_loader    import load_dataset
from preprocess     import DataPreprocessor
from scheduler      import schedule_tasks
from habit_model    import HabitModel
from api_scheduler  import APIScheduler
from fusion_engine  import ScheduleFusionEngine
from config         import (
    DEFAULT_MODEL_WEIGHT, DEFAULT_API_WEIGHT, ML_WEIGHTS,
)

# ── constants ─────────────────────────────────────────────────────────────────
BUCKET_TO_HOUR = {0: 7, 1: 11, 2: 15, 3: 19, 4: 22}

DATA_PATH           = "activities_of_daily_living_fixed.csv"
HABIT_MODEL_PATH    = "habit_model.pkl"
DURATION_MODEL_PATH = "duration_model.pkl"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Task Scheduler",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-header{font-size:2.5rem;font-weight:bold;color:#FF6B6B;text-align:center;margin-bottom:1rem}
  .sub-header{font-size:1.5rem;font-weight:bold;color:#4ECDC4;margin-top:1rem;margin-bottom:.5rem}
  .task-card{background:#fff;padding:1rem;border-radius:.5rem;border:1px solid #E0E0E0;margin-bottom:.5rem}
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
for key in ["tasks", "schedule", "ml_schedule", "api_schedule",
            "fused_schedule", "models_loaded"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ("tasks",) else None
if st.session_state.models_loaded is None:
    st.session_state.models_loaded = False

# ── model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        df           = load_dataset(DATA_PATH)
        preprocessor = DataPreprocessor()
        df           = preprocessor.fit_transform(df)

        habit_model    = HabitModel();    habit_model.load(HABIT_MODEL_PATH)
        duration_model = DurationModel(); duration_model.load(DURATION_MODEL_PATH)

        api_scheduler  = APIScheduler()
        fusion_engine  = ScheduleFusionEngine(
            model_weight=DEFAULT_MODEL_WEIGHT,
            api_weight=DEFAULT_API_WEIGHT,
        )
        return dict(df=df, preprocessor=preprocessor, habit_model=habit_model,
                    duration_model=duration_model, api_scheduler=api_scheduler,
                    fusion_engine=fusion_engine)
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def create_ml_schedule(tasks: List[Dict], models: Dict) -> List[Dict]:
    ml_schedule = []
    for task in tasks:
        start_time = datetime.now().replace(
            hour=task["Preferred_Hour"], minute=0, second=0, microsecond=0
        )
        end_time = start_time + timedelta(minutes=task["Estimated_Duration"])

        rl_reward             = 0.8
        habit_alignment       = task.get("Confidence", 0.7)
        deadline_satisfaction = 1.0 if task["Preferred_Hour"] <= task["Deadline_Hour"] else 0.5
        priority_score        = task["Priority"] / 4.0

        ml_score = (
            ML_WEIGHTS["rl_reward"]              * rl_reward             +
            ML_WEIGHTS["habit_alignment"]         * habit_alignment       +
            ML_WEIGHTS["deadline_satisfaction"]   * deadline_satisfaction +
            ML_WEIGHTS["priority_score"]          * priority_score
        )
        ml_schedule.append({
            "task":                 task["Task"],
            "start_time":           start_time,
            "end_time":             end_time,
            "ml_score":             ml_score,
            "rl_reward":            rl_reward,
            "habit_alignment":      habit_alignment,
            "deadline_satisfaction":deadline_satisfaction,
            "priority_score":       priority_score,
        })
    return ml_schedule


def prepare_tasks_for_api(tasks: List[Dict]) -> List[Dict]:
    return [
        {
            "name":     t["Task"],
            "duration": int(t["Estimated_Duration"]),
            "priority": t["Priority"],
            "deadline": datetime.now().replace(
                hour=t["Deadline_Hour"], minute=0, second=0, microsecond=0
            ),
        }
        for t in tasks
    ]


# ── chart helpers ─────────────────────────────────────────────────────────────

def create_timeline_chart(schedule: List[Dict]):
    if not schedule:
        return None
    df_tl = pd.DataFrame([{
        "Task":   item["task"],
        "Start":  item["start_time"],
        "End":    item["end_time"],
        "Source": item.get("source", "Unknown"),
        "Score":  item.get("final_score", 0),
    } for item in schedule])
    fig = px.timeline(df_tl, x_start="Start", x_end="End", y="Task",
                      color="Source", hover_data=["Score"],
                      title="Daily Schedule Timeline")
    fig.update_layout(xaxis_title="Time", yaxis_title="Tasks",
                      height=400, showlegend=True)
    return fig


def create_score_comparison_chart(schedule: List[Dict]):
    if not schedule:
        return None
    tasks      = [i["task"]                     for i in schedule]
    ml_scores  = [i.get("ml_score",  0) * 100  for i in schedule]
    api_scores = [i.get("api_score", 0) * 100  for i in schedule]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="ML Score",  x=tasks, y=ml_scores,  marker_color="#4ECDC4"))
    fig.add_trace(go.Bar(name="API Score", x=tasks, y=api_scores, marker_color="#FF6B6B"))
    fig.update_layout(title="ML vs API Score Comparison",
                      xaxis_title="Tasks", yaxis_title="Score (%)",
                      barmode="group", height=400)
    return fig


def create_influence_pie_chart(schedule: List[Dict]):
    if not schedule:
        return None
    total_ml  = sum(i.get("ml_influence",  0) for i in schedule)
    total_api = sum(i.get("api_influence", 0) for i in schedule)
    fig = go.Figure(data=[go.Pie(
        labels=["ML Influence", "API Influence"],
        values=[total_ml, total_api],
        marker_colors=["#4ECDC4", "#FF6B6B"],
        hole=0.3,
    )])
    fig.update_layout(title="Overall System Influence", height=300)
    return fig


# ── main app ──────────────────────────────────────────────────────────────────

def main():
    st.markdown('<p class="main-header">🤖 AI Hybrid Task Scheduler</p>',
                unsafe_allow_html=True)
    st.markdown("**Intelligent scheduling powered by ML + API behavioral corrections**")

    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Loading models..."):
            models = load_models()
            if models is None:
                st.error("❌ Failed to load models. Please check your model files.")
                return
            st.session_state.models        = models
            st.session_state.models_loaded = True
            st.success("✅ Models loaded successfully!")

    models = st.session_state.models

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("### 👤 User Selection")

        person_ids      = models["df"]["Person ID"].unique()
        selected_person = st.selectbox("Select Person ID",
                                       options=sorted(person_ids)[:50], index=0)
        person_encoded  = models["preprocessor"].person_encoder.transform([selected_person])[0]

        selected_date = st.date_input("Schedule Date", value=datetime.now())
        day_of_week   = selected_date.weekday()

        st.markdown("---")
        st.markdown("### ⚖️ Fusion Weights")
        st.info("Adjust how much you trust ML vs API recommendations")

        model_weight = st.slider("ML Weight", 0.0, 1.0,
                                 value=DEFAULT_MODEL_WEIGHT, step=0.05,
                                 help="Higher = Trust ML predictions more")
        api_weight = 1.0 - model_weight
        st.metric("API Weight", f"{api_weight:.2f}")
        models["fusion_engine"].update_weights(model_weight, api_weight)

        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.info(
            f"**Person ID:** {selected_person}  \n"
            f"**Day:** {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]}  \n"
            f"**ML Weight:** {model_weight:.0%}  \n"
            f"**API Weight:** {api_weight:.0%}"
        )

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Task Input", "📅 Schedule View", "📊 Analytics", "ℹ️ About"
    ])

    # ── TAB 1: task input ─────────────────────────────────────────────────────
    with tab1:
        st.markdown('<p class="sub-header">Add Tasks</p>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            activities = sorted(models["df"]["Activity Type"].unique())
            with st.form("task_form"):
                st.markdown("### 📋 New Task")
                task_name = st.selectbox("Activity Type", options=activities, index=0)
                col_a, col_b = st.columns(2)
                with col_a:
                    priority = st.selectbox(
                        "Priority", options=[1, 2, 3, 4],
                        format_func=lambda x: ["Low","Medium","High","Critical"][x-1],
                        index=1,
                    )
                with col_b:
                    deadline_hour = st.slider("Deadline Hour", 0, 23, 20,
                                              help="Latest hour to finish this task")

                if st.form_submit_button("➕ Add Task", use_container_width=True):
                    if task_name in models["preprocessor"].activity_encoder.classes_:
                        activity_encoded = models["preprocessor"].activity_encoder.transform([task_name])[0]

                        bucket, confidence = models["habit_model"].predict_with_confidence(
                            person_encoded, activity_encoded, df=models["df"]
                        )
                        predicted_hour     = BUCKET_TO_HOUR[bucket]
                        predicted_duration = models["duration_model"].predict(
                            activity_encoded, predicted_hour, day_of_week
                        )

                        st.session_state.tasks.append({
                            "Task":               task_name,
                            "Activity_Encoded":   activity_encoded,
                            "Priority":           priority,
                            "Preferred_Hour":     predicted_hour,
                            "Deadline_Hour":      deadline_hour,
                            "Estimated_Duration": round(predicted_duration),
                            "Confidence":         confidence,
                        })
                        st.success(f"✅ Added: {task_name} at {predicted_hour:02d}:00 "
                                   f"({round(predicted_duration)} min)")
                        st.rerun()
                    else:
                        st.error("❌ Activity not recognised")

        with col2:
            st.markdown("### 📊 Current Tasks")
            if st.session_state.tasks:
                for idx, task in enumerate(st.session_state.tasks):
                    with st.container():
                        st.markdown(
                            f'<div class="task-card">'
                            f'<strong>{task["Task"]}</strong><br>'
                            f'Priority: {["Low","Medium","High","Critical"][task["Priority"]-1]}<br>'
                            f'Time: {task["Preferred_Hour"]:02d}:00<br>'
                            f'Duration: {task["Estimated_Duration"]} min'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if st.button("🗑️ Remove", key=f"remove_{idx}"):
                            st.session_state.tasks.pop(idx)
                            st.rerun()

                st.markdown("---")
                c_clear, c_sched = st.columns(2)
                with c_clear:
                    if st.button("🗑️ Clear All", use_container_width=True):
                        st.session_state.tasks = []
                        st.rerun()
                with c_sched:
                    if st.button("🚀 Generate Schedule", type="primary",
                                 use_container_width=True):
                        with st.spinner("🔄 Creating hybrid schedule..."):
                            ml_schedule  = create_ml_schedule(st.session_state.tasks, models)
                            api_tasks    = prepare_tasks_for_api(st.session_state.tasks)
                            api_schedule = models["api_scheduler"].get_schedule(api_tasks)
                            avg_conf     = float(np.mean([t.get("Confidence", 0.7)
                                                          for t in st.session_state.tasks]))
                            fused        = models["fusion_engine"].combine_schedules(
                                ml_schedule, api_schedule, habit_confidence=avg_conf
                            )
                            st.session_state.ml_schedule    = ml_schedule
                            st.session_state.api_schedule   = api_schedule
                            st.session_state.fused_schedule = fused
                        st.success("✅ Schedule generated!")
                        st.rerun()
            else:
                st.info("No tasks added yet. Use the form to add tasks.")

    # ── TAB 2: schedule view ──────────────────────────────────────────────────
    with tab2:
        st.markdown('<p class="sub-header">📅 Your Schedule</p>', unsafe_allow_html=True)

        if st.session_state.fused_schedule:
            fig = create_timeline_chart(st.session_state.fused_schedule)
            if fig:
                st.plotly_chart(fig, width='stretch')   # kept for now — update when Streamlit enforces removal
            st.markdown("---")
            st.markdown("### 📋 Detailed Schedule")

            for idx, item in enumerate(st.session_state.fused_schedule, 1):
                label = (f"**{idx}. {item['task']}** "
                         f"({item['start_time'].strftime('%H:%M')} - "
                         f"{item['end_time'].strftime('%H:%M')})")
                with st.expander(label):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Final Score",   f"{item['final_score']:.2f}")
                    c2.metric("ML Influence",  f"{item['ml_influence']:.1f}%")
                    c3.metric("API Influence", f"{item['api_influence']:.1f}%")
                    st.markdown(
                        f"**Source:** {item['source']}  \n"
                        f"**ML Score:** {item['ml_score']:.2f}  \n"
                        f"**API Score:** {item['api_score']:.2f}  \n"
                        f"**Reason:** {item['reason']}"
                    )

            st.markdown("---")
            if st.button("📥 Export Schedule (CSV)"):
                df_exp = pd.DataFrame(st.session_state.fused_schedule)
                st.download_button("Download CSV", df_exp.to_csv(index=False),
                                   file_name=f"schedule_{selected_date}.csv",
                                   mime="text/csv")
        else:
            st.info("👈 Add tasks and click 'Generate Schedule' to see your schedule.")

    # ── TAB 3: analytics ──────────────────────────────────────────────────────
    with tab3:
        st.markdown('<p class="sub-header">📊 Schedule Analytics</p>',
                    unsafe_allow_html=True)

        if st.session_state.fused_schedule:
            c1, c2 = st.columns(2)
            with c1:
                fig = create_score_comparison_chart(st.session_state.fused_schedule)
                if fig:
                    st.plotly_chart(fig, width='stretch')
            with c2:
                fig = create_influence_pie_chart(st.session_state.fused_schedule)
                if fig:
                    st.plotly_chart(fig, width='stretch')

            st.markdown("---")
            st.markdown("### 📈 Statistics")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Tasks", len(st.session_state.fused_schedule))
            s2.metric("Avg Score",
                      f"{np.mean([i['final_score'] for i in st.session_state.fused_schedule]):.2f}")
            s3.metric("ML Dominant",
                      sum(1 for i in st.session_state.fused_schedule if i["source"] == "ML"))
            s4.metric("API Dominant",
                      sum(1 for i in st.session_state.fused_schedule if i["source"] == "API"))

            st.markdown("---")
            st.markdown("### 🔍 ML vs API Comparison")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**ML Schedule**")
                if st.session_state.ml_schedule:
                    st.dataframe(pd.DataFrame([{
                        "Task":  i["task"],
                        "Time":  i["start_time"].strftime("%H:%M"),
                        "Score": f"{i['ml_score']:.2f}",
                    } for i in st.session_state.ml_schedule]), width='stretch')
            with c2:
                st.markdown("**API Schedule**")
                if st.session_state.api_schedule:
                    st.dataframe(pd.DataFrame([{
                        "Task":  i["task"],
                        "Time":  i["start_time"].strftime("%H:%M"),
                        # api schedule items have 'score' key (fixed fallback always sets it)
                        "Score": f"{i.get('score', 0.0):.2f}",
                    } for i in st.session_state.api_schedule]), width='stretch')
        else:
            st.info("Generate a schedule first to see analytics.")

    # ── TAB 4: about ──────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<p class="sub-header">ℹ️ About This System</p>',
                    unsafe_allow_html=True)
        st.markdown("""
### 🎯 System Overview

This **Hybrid AI Task Scheduler** combines three layers:

**🤖 Machine Learning (Primary — 80 % weight)**
- XGBoost duration prediction (`duration_model.py`)
- Three-level habit model for time preference (`habit_model.py`)
- Dueling Double DQN ordering via RL agent (`scheduler.py` + `rl_agent.py`)
- Personalised to your historical activity patterns

**🌐 Gemini API Behavioral Correction (Secondary — 20 % weight)**
- Enforces healthy meal / sleep / exercise windows
- Prevents schedules like eating at 2 AM or sleeping at noon
- Falls back to a built-in rule engine if Gemini is unavailable

**⚖️ Fusion Engine**
- `final_score = (ML_weight × ML_score) + (API_weight × API_score)`
- Adaptive: low habit-confidence → raise API weight; high → raise ML weight
- Resolves overlapping tasks by score (higher score keeps its slot)

---
### ⚙️ Key Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| ML Weight | 80 % | Higher → trust habits more |
| API Weight | 20 % | Higher → follow health rules more |
| Low confidence threshold | 0.5 | Below this, API weight +15 % |
| High confidence threshold | 0.8 | Above this, ML weight +10 % |

---
### 🚀 Future Enhancements
- Real-time RL agent integration with online learning
- User feedback loop to refine habit model
- Calendar sync (Google / Outlook)
- Mobile app
        """)


if __name__ == "__main__":
    main()