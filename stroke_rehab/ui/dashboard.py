"""
ui/dashboard.py

NeuroRehab — Streamlit dashboard
Displays:
  - Live EMG signal chart
  - Camera feed with tracking overlay
  - Patient performance metrics
  - Recovery progress over time
  - Doctor report viewer
  - Session controls
"""

import sys
import os
import time
import json
import random
import math
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import (
    SESSION_DIR, REPORT_DIR, MODEL_DIR, DEFAULT_PATIENT_ID, EMG_SOURCE
)
from data_acquisition.emg_reader import create_emg_reader, SimulatedEMGReader
from data_acquisition.preprocessor import EMGPreprocessor, EMGFeatures
from ml.trainer import PerformancePredictor, ProgressTracker, ALL_FEATURES
from ml.doctor_report import DoctorReportParser, create_sample_report

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroRehab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #e0e8ff;
}

.stApp {
    background: #080d1a;
}

.block-container {
    padding-top: 1.5rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0f1729 0%, #141e38 100%);
    border: 1px solid #1e2d5a;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 6px 0;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #00c8aa;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #00c8aa;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.78rem;
    color: #6070a0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Status badges */
.badge-good    { background: #0a2a1a; color: #40dd80; border: 1px solid #40dd80; 
                 padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-comp    { background: #2a1e0a; color: #ffa030; border: 1px solid #ffa030;
                 padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-poor    { background: #2a0a0a; color: #ff4060; border: 1px solid #ff4060;
                 padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }

/* Section headers */
.section-header {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #3a4a7a;
    margin-bottom: 8px;
    margin-top: 16px;
    border-bottom: 1px solid #1a2040;
    padding-bottom: 6px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #060b18;
    border-right: 1px solid #141e38;
}

/* Plotly chart background */
.js-plotly-plot .plotly {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────
def init_state():
    defaults = {
        "emg_buffer": [],
        "emg_reader": None,
        "preprocessor": None,
        "predictor": None,
        "progress_tracker": None,
        "session_active": False,
        "session_start": None,
        "patient_id": DEFAULT_PATIENT_ID,
        "current_label": "compensating",
        "current_score": 50,
        "current_rom": 0.0,
        "current_stability": 1.0,
        "current_features": None,
        "therapy_plan": None,
        "session_log": [],
        "emg_pattern": "moderate",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Helper: load/create systems ─────────────────────────────────────────────
@st.cache_resource
def get_predictor():
    return PerformancePredictor()

@st.cache_resource
def get_preprocessor():
    return EMGPreprocessor()


def get_emg_samples(n: int = 256) -> np.ndarray:
    """Get fresh EMG samples, creating reader if needed."""
    if st.session_state.emg_reader is None:
        st.session_state.emg_reader = create_emg_reader(
            "simulated", pattern=st.session_state.emg_pattern
        )
        time.sleep(0.3)

    samples = st.session_state.emg_reader.read_available()
    if len(samples) < n:
        # Pad with previous tail or noise
        pad = [abs(np.random.randn() * 0.02) for _ in range(n - len(samples))]
        samples = pad + samples
    return np.array(samples[-n:])


def simulate_motion():
    """Simulate motion metrics for dashboard."""
    t = time.time()
    return {
        "arm_angle": 90 + 30 * math.sin(t * 0.4),
        "rom": 45 + 10 * math.sin(t * 0.2),
        "stability": 0.85 + 0.1 * math.sin(t * 2),
        "velocity": abs(5 * math.cos(t * 0.4)),
    }


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 NeuroRehab")
    st.markdown('<div class="section-header">Patient</div>', unsafe_allow_html=True)
    patient_id = st.text_input("Patient ID", value=st.session_state.patient_id)
    st.session_state.patient_id = patient_id

    st.markdown('<div class="section-header">EMG Source</div>', unsafe_allow_html=True)
    emg_src = st.radio("Source", ["Simulated", "Serial (ESP32)", "Bluetooth"],
                       horizontal=False)
    if emg_src == "Simulated":
        pattern = st.select_slider("Signal Pattern",
                                   ["poor", "moderate", "good"],
                                   value=st.session_state.emg_pattern)
        if pattern != st.session_state.emg_pattern:
            st.session_state.emg_reader = None  # Reset reader
            st.session_state.emg_pattern = pattern
    elif emg_src == "Serial (ESP32)":
        serial_port = st.text_input("Port", "/dev/ttyUSB0")
        st.caption("Set EMG_SOURCE='serial' in config.py")

    st.markdown('<div class="section-header">Doctor Report</div>', unsafe_allow_html=True)
    report_files = [f for f in os.listdir(REPORT_DIR) if f.endswith(".json")] if os.path.exists(REPORT_DIR) else []
    if not report_files:
        if st.button("📄 Create Sample Report"):
            create_sample_report()
            st.rerun()
    else:
        selected_report = st.selectbox("Load Report", report_files)
        if st.button("Load"):
            parser = DoctorReportParser()
            plan = parser.parse_file(os.path.join(REPORT_DIR, selected_report))
            st.session_state.therapy_plan = plan
            st.success(f"Loaded plan for {plan.patient_id}")

    st.markdown('<div class="section-header">Session</div>', unsafe_allow_html=True)
    if not st.session_state.session_active:
        if st.button("▶ Start Session", use_container_width=True, type="primary"):
            st.session_state.session_active = True
            st.session_state.session_start = time.time()
            st.session_state.session_log = []
    else:
        elapsed = time.time() - st.session_state.session_start
        st.metric("Session Time", f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
        if st.button("⏹ End Session", use_container_width=True):
            st.session_state.session_active = False

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (live)", value=True)
    refresh_rate = st.slider("Refresh rate (s)", 0.5, 3.0, 1.0, 0.5)


# ─── Main layout ──────────────────────────────────────────────────────────────
st.markdown("# 🧠 NeuroRehab — Rehabilitation Dashboard")

# Top metric row
col1, col2, col3, col4, col5 = st.columns(5)

# Get live data
raw_emg = get_emg_samples(256)
preprocessor = get_preprocessor()
filtered_emg, features = preprocessor.process_window(raw_emg)
predictor = get_predictor()
motion = simulate_motion()
prediction = predictor.predict(
    features,
    arm_angle=motion["arm_angle"],
    rom=motion["rom"],
    stability=motion["stability"],
    velocity=motion["velocity"],
)
st.session_state.current_label = prediction["label"]
st.session_state.current_score = prediction["score"]
st.session_state.current_rom = motion["rom"]
st.session_state.current_stability = motion["stability"]
st.session_state.current_features = features

# Log to session
if st.session_state.session_active:
    st.session_state.session_log.append({
        "time": time.time() - st.session_state.session_start,
        "rms": float(features.rms),
        "score": prediction["score"],
        "label": prediction["label"],
        "rom": motion["rom"],
        "stability": motion["stability"],
    })

label = prediction["label"]
badge_class = {"good": "badge-good", "compensating": "badge-comp", "poor": "badge-poor"}.get(label, "badge-comp")
label_emoji = {"good": "🟢", "compensating": "🟡", "poor": "🔴"}.get(label, "🟡")

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{prediction["score"]}</div>
        <div class="metric-label">Performance Score</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{features.rms:.3f}</div>
        <div class="metric-label">EMG RMS</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{motion['rom']:.1f}°</div>
        <div class="metric-label">Range of Motion</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{motion['stability']:.2f}</div>
        <div class="metric-label">Stability Index</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div style="margin-top:8px"><span class="{badge_class}">{label_emoji} {label.upper()}</span></div>
        <div class="metric-label" style="margin-top:10px">Performance Label</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Monitor", "📈 Progress", "🎮 Games", "📋 Doctor Report"
])


# ─── TAB 1: Live Monitor ──────────────────────────────────────────────────────
with tab1:
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown('<div class="section-header">Live EMG Signal</div>', unsafe_allow_html=True)

        # Build rolling EMG buffer
        st.session_state.emg_buffer.extend(filtered_emg.tolist())
        if len(st.session_state.emg_buffer) > 1000:
            st.session_state.emg_buffer = st.session_state.emg_buffer[-1000:]

        buffer = np.array(st.session_state.emg_buffer)
        t_axis = np.linspace(-len(buffer) / 1000, 0, len(buffer))

        fig_emg = go.Figure()
        fig_emg.add_trace(go.Scatter(
            x=t_axis, y=buffer,
            mode="lines",
            line=dict(color="#00c8aa", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(0,200,170,0.08)",
            name="EMG",
        ))
        # Activation threshold line
        fig_emg.add_hline(y=0.05, line_dash="dot",
                          line_color="rgba(255,160,0,0.5)",
                          annotation_text="threshold")
        fig_emg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,13,26,0.8)",
            height=220,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, color="#3a4a7a", title="Time (s)"),
            yaxis=dict(showgrid=True, gridcolor="#141e38", color="#3a4a7a",
                       range=[0, 1], title="Amplitude"),
            showlegend=False,
        )
        st.plotly_chart(fig_emg, use_container_width=True)

        st.markdown('<div class="section-header">EMG Feature Breakdown</div>', unsafe_allow_html=True)
        feat_names = EMGFeatures.feature_names()
        feat_vals = features.to_array()
        norm_vals = np.clip(feat_vals / (np.max(feat_vals) + 1e-6), 0, 1)

        fig_feat = go.Figure(go.Bar(
            x=feat_names,
            y=feat_vals,
            marker=dict(
                color=norm_vals,
                colorscale=[[0, "#0a1520"], [0.5, "#004488"], [1.0, "#00c8aa"]],
                line=dict(color="#00c8aa", width=0.5),
            ),
        ))
        fig_feat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,13,26,0.8)",
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, color="#3a4a7a", tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="#141e38", color="#3a4a7a"),
            showlegend=False,
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    with right_col:
        st.markdown('<div class="section-header">Performance Probabilities</div>', unsafe_allow_html=True)
        probs = prediction.get("probabilities", {})
        labels_ord = ["poor", "compensating", "good"]
        prob_vals = [probs.get(l, 0) for l in labels_ord]
        colors_p = ["#ff4060", "#ffa030", "#40dd80"]

        fig_probs = go.Figure(go.Bar(
            x=prob_vals, y=labels_ord,
            orientation="h",
            marker_color=colors_p,
            text=[f"{v*100:.0f}%" for v in prob_vals],
            textposition="inside",
        ))
        fig_probs.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,13,26,0.8)",
            height=180,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, color="#3a4a7a", range=[0, 1]),
            yaxis=dict(color="#6070a0"),
            showlegend=False,
        )
        st.plotly_chart(fig_probs, use_container_width=True)

        st.markdown('<div class="section-header">Motion Metrics</div>', unsafe_allow_html=True)
        # Gauge chart for stability
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=motion["stability"] * 100,
            title={"text": "Stability", "font": {"color": "#6070a0", "size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#3a4a7a"},
                "bar": {"color": "#00c8aa"},
                "bgcolor": "#0f1729",
                "bordercolor": "#1e2d5a",
                "steps": [
                    {"range": [0, 40], "color": "#2a0a0a"},
                    {"range": [40, 70], "color": "#2a1e0a"},
                    {"range": [70, 100], "color": "#0a2a1a"},
                ],
                "threshold": {
                    "line": {"color": "#00c8aa", "width": 2},
                    "thickness": 0.75,
                    "value": motion["stability"] * 100,
                },
            },
            number={"suffix": "%", "font": {"color": "#00c8aa", "size": 22}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=20, r=20, t=20, b=0),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ROM bar
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="metric-value">{motion['rom']:.1f}°</div>
                    <div class="metric-label">Range of Motion</div>
                </div>
                <div style="font-size:2rem">💪</div>
            </div>
            <div style="margin-top:10px; background:#0a0f1e; border-radius:4px; height:6px;">
                <div style="width:{min(100, motion['rom']/90*100):.0f}%; height:6px;
                            background:#00c8aa; border-radius:4px;"></div>
            </div>
            <div class="metric-label" style="margin-top:4px">Target: 90°</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Session Activity</div>', unsafe_allow_html=True)
        if st.session_state.session_active:
            elapsed = time.time() - st.session_state.session_start
            log = st.session_state.session_log
            n_windows = len(log)
            label_counts = {}
            for entry in log:
                l = entry.get("label", "")
                label_counts[l] = label_counts.get(l, 0) + 1
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Session Running</div>
                <div style="margin-top:6px; font-size:1.1rem; color:#00c8aa;">
                    ⏱ {int(elapsed//60):02d}:{int(elapsed%60):02d} &nbsp;|&nbsp;
                    📊 {n_windows} windows
                </div>
                <div style="margin-top:8px">
                    {''.join([f'<span class="badge-good">{label_counts.get("good",0)} good</span>&nbsp;'
                              f'<span class="badge-comp">{label_counts.get("compensating",0)} comp</span>&nbsp;'
                              f'<span class="badge-poor">{label_counts.get("poor",0)} poor</span>'])}
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Start a session to record data.")


# ─── TAB 2: Progress ──────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Recovery Progress Over Time</div>', unsafe_allow_html=True)

    # Generate simulated progress data for demo
    @st.cache_data
    def get_demo_progress():
        np.random.seed(7)
        n = 20
        dates = [datetime.now() - timedelta(days=n - i) for i in range(n)]
        base_score = 35
        scores = np.clip(
            [base_score + i * 2.5 + np.random.randn() * 8 for i in range(n)], 10, 100
        )
        roms = np.clip(
            [15 + i * 1.8 + np.random.randn() * 5 for i in range(n)], 5, 80
        )
        stabs = np.clip(
            [0.5 + i * 0.02 + np.random.randn() * 0.05 for i in range(n)], 0, 1
        )
        return pd.DataFrame({
            "date": dates, "score": scores, "rom": roms, "stability": stabs
        })

    df_prog = get_demo_progress()
    # Add live session if active
    if st.session_state.session_active and st.session_state.session_log:
        log_df = pd.DataFrame(st.session_state.session_log)
        if len(log_df) > 0:
            st.markdown("**Live session rolling average:**")
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(
                x=log_df["time"], y=log_df["score"].rolling(5, min_periods=1).mean(),
                mode="lines", line=dict(color="#00c8aa", width=2),
                fill="tozeroy", fillcolor="rgba(0,200,170,0.1)",
                name="Score (rolling avg)",
            ))
            fig_live.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,13,26,0.8)",
                height=180, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, color="#3a4a7a", title="Session time (s)"),
                yaxis=dict(showgrid=True, gridcolor="#141e38", color="#3a4a7a",
                           range=[0, 105]),
            )
            st.plotly_chart(fig_live, use_container_width=True)

    pc1, pc2 = st.columns(2)
    with pc1:
        fig_score = go.Figure()
        fig_score.add_trace(go.Scatter(
            x=df_prog["date"], y=df_prog["score"],
            mode="lines+markers",
            line=dict(color="#00c8aa", width=2),
            marker=dict(size=6, color="#00c8aa"),
            fill="tozeroy", fillcolor="rgba(0,200,170,0.06)",
            name="Score",
        ))
        # Trend line
        x_num = np.arange(len(df_prog))
        z = np.polyfit(x_num, df_prog["score"], 1)
        trend = np.polyval(z, x_num)
        fig_score.add_trace(go.Scatter(
            x=df_prog["date"], y=trend,
            mode="lines", line=dict(color="#ffa030", width=1, dash="dot"),
            name="Trend",
        ))
        fig_score.update_layout(
            title=dict(text="Performance Score", font=dict(color="#6070a0", size=13)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,13,26,0.8)",
            height=260, margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False, color="#3a4a7a"),
            yaxis=dict(showgrid=True, gridcolor="#141e38", color="#3a4a7a", range=[0, 105]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6070a0")),
        )
        st.plotly_chart(fig_score, use_container_width=True)

    with pc2:
        fig_rom = go.Figure()
        fig_rom.add_trace(go.Scatter(
            x=df_prog["date"], y=df_prog["rom"],
            mode="lines+markers",
            line=dict(color="#a060ff", width=2),
            marker=dict(size=6, color="#a060ff"),
            fill="tozeroy", fillcolor="rgba(160,96,255,0.06)",
            name="ROM",
        ))
        fig_rom.add_hline(y=45, line_dash="dot", line_color="rgba(255,160,0,0.5)",
                          annotation_text="target 45°")
        fig_rom.update_layout(
            title=dict(text="Range of Motion (°)", font=dict(color="#6070a0", size=13)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,13,26,0.8)",
            height=260, margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False, color="#3a4a7a"),
            yaxis=dict(showgrid=True, gridcolor="#141e38", color="#3a4a7a"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6070a0")),
        )
        st.plotly_chart(fig_rom, use_container_width=True)

    st.markdown('<div class="section-header">Session History</div>', unsafe_allow_html=True)
    display_df = df_prog.copy()
    display_df["date"] = display_df["date"].dt.strftime("%b %d")
    display_df["score"] = display_df["score"].round(1)
    display_df["rom"] = display_df["rom"].round(1)
    display_df["stability"] = display_df["stability"].round(3)
    st.dataframe(display_df, use_container_width=True, height=250)


# ─── TAB 3: Games ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Gamified Therapy — Mini Games</div>', unsafe_allow_html=True)
    st.info("🎮 Games run as standalone Pygame windows. Use the launcher below or run directly from terminal.")

    games = [
        {"name": "Bubble Pop", "icon": "🫧", "file": "bubble_pop.py",
         "exercise": "Pinch coordination",
         "desc": "Pinch floating bubbles before they escape. Easy to understand for any age."},
        {"name": "Flower Bloom", "icon": "🌸", "file": "flower_bloom.py",
         "exercise": "Hand open/close",
         "desc": "Garden blooms when you open your hand. Calm and encouraging for gentle sessions."},
        {"name": "Pump the Pump", "icon": "💦", "file": "pump_maze.py:PumpThePumpGame",
         "exercise": "Grip strength",
         "desc": "Squeeze to charge power meter, release to fire water cannon at targets."},
        {"name": "Maze Steering", "icon": "🌀", "file": "pump_maze.py:MazeSteeringGame",
         "exercise": "Wrist rotation",
         "desc": "Rotate your wrist to guide a marble through increasingly complex paths."},
    ]

    plan = st.session_state.therapy_plan
    if plan:
        st.markdown(f"**Prescribed game sequence for {plan.patient_id}:** "
                    + " → ".join([f"`{g}`" for g in plan.get_game_sequence()]))

    gcols = st.columns(4)
    for i, game in enumerate(games):
        with gcols[i]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center; min-height:200px;">
                <div style="font-size:2.5rem">{game['icon']}</div>
                <div style="font-size:1.1rem; font-weight:600; color:#e0e8ff; margin:8px 0 4px;">
                    {game['name']}
                </div>
                <div class="metric-label" style="color:#00c8aa;">{game['exercise']}</div>
                <div style="font-size:0.82rem; color:#5060a0; margin-top:8px; line-height:1.4;">
                    {game['desc']}
                </div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"▶ Launch", key=f"game_{i}", use_container_width=True):
                st.code(f"cd stroke_rehab\npython -m game.{game['file'].split('.')[0]}", language="bash")

    st.markdown("---")
    st.markdown('<div class="section-header">Run from Terminal</div>', unsafe_allow_html=True)
    st.code("""
# Run full demo pipeline (all modules integrated)
python scripts/run_demo.py

# Run a specific game
python -m game.bubble_pop
python -m game.flower_bloom
python -m game.pump_maze  # includes both Pump and Maze

# Full integration demo
python scripts/run_demo.py --game bubble_pop --pattern good --duration 60
""", language="bash")


# ─── TAB 4: Doctor Report ─────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Doctor Report & Therapy Plan</div>', unsafe_allow_html=True)

    plan = st.session_state.therapy_plan
    if plan is None:
        st.warning("No therapy plan loaded. Load a doctor report from the sidebar.")
        st.markdown("**Sample report format:**")
        from ml.doctor_report import SAMPLE_REPORT
        st.json(SAMPLE_REPORT)
    else:
        rc1, rc2 = st.columns([2, 3])
        with rc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Patient ID</div>
                <div style="font-size:1.3rem; color:#e0e8ff;">{plan.patient_id}</div>
                <div class="metric-label" style="margin-top:12px;">Condition</div>
                <div style="color:#e0e8ff;">{plan.condition}</div>
                <div class="metric-label" style="margin-top:12px;">Severity</div>
                <div><span class="{'badge-poor' if plan.severity=='severe' else 'badge-comp' if plan.severity=='moderate' else 'badge-good'}">
                    {plan.severity.upper()}
                </span></div>
                <div class="metric-label" style="margin-top:12px;">Affected Side</div>
                <div style="color:#e0e8ff;">{plan.affected_side.title()}</div>
                <div class="metric-label" style="margin-top:12px;">Target ROM</div>
                <div style="color:#00c8aa; font-family:'JetBrains Mono'; font-size:1.2rem;">{plan.target_rom_degrees}°</div>
                <div class="metric-label" style="margin-top:12px;">Session Duration</div>
                <div style="color:#e0e8ff;">{plan.session_duration_minutes} minutes</div>
            </div>""", unsafe_allow_html=True)

            if plan.contraindications:
                st.warning("⚠️ **Contraindications:** " + ", ".join(plan.contraindications))
            if plan.doctor_notes:
                st.info(f"📝 **Doctor notes:** {plan.doctor_notes}")

        with rc2:
            st.markdown('<div class="section-header">Prescribed Exercises</div>', unsafe_allow_html=True)
            for j, ex in enumerate(plan.exercises):
                icon = {"pump_the_pump": "💦", "flower_bloom": "🌸",
                        "bubble_pop": "🫧", "maze_steering": "🌀"}.get(ex.game, "🎮")
                diff_cls = {"easy": "badge-good", "medium": "badge-comp", "hard": "badge-poor"}.get(ex.difficulty, "badge-comp")
                st.markdown(f"""
                <div class="metric-card" style="margin:6px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <span style="font-size:1.3rem">{icon}</span>
                            <span style="font-size:1rem; font-weight:600; color:#e0e8ff; margin-left:8px;">
                                {ex.game.replace('_', ' ').title()}
                            </span>
                        </div>
                        <span class="{diff_cls}">{ex.difficulty.upper()}</span>
                    </div>
                    <div style="margin-top:8px; font-size:0.85rem; color:#6070a0;">
                        {ex.sets} sets × {ex.reps} reps &nbsp;|&nbsp; {ex.exercise_type.replace('_',' ')}
                    </div>
                    {f'<div style="font-size:0.8rem; color:#5060a0; margin-top:6px;">📌 {ex.notes}</div>' if ex.notes else ''}
                </div>""", unsafe_allow_html=True)


# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
