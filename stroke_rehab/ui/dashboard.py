"""
ui/dashboard.py

Stroke Rehab Monitor — Streamlit dashboard for remote patient monitoring.
Receives live MQTT data from patient-side rehabilitation system.

MQTT Topics (published by patient side):
  rehab/{PATIENT_ID}/emg           → EMG signal metrics
  rehab/{PATIENT_ID}/handpose      → OpenCV hand tracking data
  rehab/{PATIENT_ID}/fusion        → Fusion engine state classification
  rehab/{PATIENT_ID}/game          → Game progress (score, reps, level)
  rehab/{PATIENT_ID}/alerts        → Doctor alerts (fatigue, motor block)
  rehab/{PATIENT_ID}/session_summary → Session end summary
"""

import sys
import os
import time
import json
import threading
import queue
from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# MQTT client
try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
    # Check for paho-mqtt v2 API
    try:
        from paho.mqtt.enums import CallbackAPIVersion
        PAHO_V2 = True
    except ImportError:
        PAHO_V2 = False
except ImportError:
    PAHO_AVAILABLE = False
    PAHO_V2 = False

# Config and core imports
from config import (
    MQTT_BROKER, MQTT_PORT, PATIENT_ID,
    DASHBOARD_REFRESH_MS, EMG_HISTORY_LENGTH, ALERT_MAX_COUNT, REP_LOG_MAX_COUNT,
    FUSION_COLORS, EMG_THRESHOLD_PCT
)
from core.recovery_tracker import load_session_history, init_db

# Module-level queue for MQTT thread → main thread communication
# Use singleton pattern to persist across Streamlit reruns (module reloads)
import sys
if '_mqtt_queue_singleton' not in sys.modules:
    sys.modules['_mqtt_queue_singleton'] = queue.Queue()
_mqtt_queue = sys.modules['_mqtt_queue_singleton']


# ─── MQTT Callbacks (run in background thread) ──────────────────────────────────

def _on_connect(client, userdata, flags, rc, properties=None):
    """Callback when MQTT client connects to broker."""
    if rc == 0:
        _mqtt_queue.put({"type": "connection", "status": True})
        # Subscribe to all patient topics
        topics = [
            f"rehab/{PATIENT_ID}/emg",
            f"rehab/{PATIENT_ID}/handpose",
            f"rehab/{PATIENT_ID}/fusion",
            f"rehab/{PATIENT_ID}/game",
            f"rehab/{PATIENT_ID}/alerts",
            f"rehab/{PATIENT_ID}/session_summary",
        ]
        for topic in topics:
            client.subscribe(topic, qos=0)
            print(f"[DASHBOARD MQTT] Subscribed to {topic}")
    else:
        _mqtt_queue.put({"type": "connection", "status": False, "rc": rc})


def _on_disconnect(client, userdata, rc, properties=None):
    """Callback when MQTT client disconnects from broker."""
    _mqtt_queue.put({"type": "connection", "status": False, "rc": rc})


def _on_message(client, userdata, msg, properties=None):
    """Callback when MQTT message received. ONLY puts to queue — no st.* calls."""
    try:
        payload = json.loads(msg.payload.decode())
        print(f"[DASHBOARD MQTT] Received: {msg.topic}")
        _mqtt_queue.put({
            "type": "message",
            "topic": msg.topic,
            "payload": payload,
            "timestamp": time.time(),
        })
        print(f"[DASHBOARD MQTT] Put to queue, size now: {_mqtt_queue.qsize()}")
    except json.JSONDecodeError as e:
        print(f"[DASHBOARD MQTT] JSON error: {e}")
    except Exception as e:
        print(f"[DASHBOARD MQTT] Error: {e}")


def _start_mqtt_client():
    """Start MQTT client in a daemon thread. Called once per Streamlit session."""
    if not PAHO_AVAILABLE:
        st.error("paho-mqtt not installed. Run: pip install paho-mqtt")
        return None

    client_id = f"doctor_dashboard_{PATIENT_ID}_{int(time.time())}"
    st.session_state["mqtt_client_id"] = client_id
    
    # Use paho-mqtt v2 API if available
    if PAHO_V2:
        from paho.mqtt.enums import CallbackAPIVersion
        client = mqtt.Client(CallbackAPIVersion.VERSION1, client_id=client_id)
    else:
        client = mqtt.Client(client_id=client_id)
    
    client.on_connect = _on_connect
    client.on_disconnect = _on_disconnect
    client.on_message = _on_message

    try:
        st.info(f"Connecting to MQTT broker {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        # Start network loop in daemon thread
        mqtt_thread = threading.Thread(target=client.loop_forever, daemon=True)
        mqtt_thread.start()
        st.success(f"MQTT client started (ID: {client_id}, API={'v2' if PAHO_V2 else 'v1'})")
        return client
    except Exception as e:
        st.error(f"MQTT connection failed: {e}")
        _mqtt_queue.put({"type": "connection", "status": False, "error": str(e)})
        return None


# ─── Session State Initialization ───────────────────────────────────────────────

def _init_session_state():
    """Initialize all session state keys for dashboard."""
    defaults = {
        # MQTT connection
        "mqtt_connected": False,
        "mqtt_client_started": False,
        "mqtt_client": None,
        # EMG data
        "emg_history": deque(maxlen=EMG_HISTORY_LENGTH),
        "emg_latest": None,
        "emg_session_start": None,
        # Hand pose data
        "handpose_latest": None,
        # Fusion data
        "fusion_latest": None,
        "rep_log": [],
        # Game data
        "game_latest": None,
        # Alerts
        "alerts": [],
        # Session tracking
        "session_active": False,
        "session_start_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Queue Draining (main thread only) ────────────────────────────────────────

def _drain_mqtt_queue():
    """Drain MQTT queue into session_state. Called every rerun in main thread."""
    # Get reference to the singleton queue
    mqtt_queue = sys.modules.get('_mqtt_queue_singleton')
    if mqtt_queue is None:
        print(f"[DASHBOARD] Queue not initialized!")
        return
    
    queue_size = mqtt_queue.qsize()
    if queue_size > 0:
        print(f"[DASHBOARD] Queue has {queue_size} messages")
    
    # Process up to 100 messages per frame to avoid blocking rendering
    processed = 0
    for _ in range(100):
        try:
            item = mqtt_queue.get_nowait()
        except queue.Empty:
            break
        except Exception as e:
            print(f"[DASHBOARD] Queue error: {e}")
            break

        processed += 1
        if item["type"] == "connection":
            st.session_state["mqtt_connected"] = item.get("status", False)
            print(f"[DASHBOARD] Connection status: {item.get('status')}")

        elif item["type"] == "message":
            topic = item["topic"]
            payload = item["payload"]
            ts = item.get("timestamp", time.time())
            print(f"[DASHBOARD] Processing: {topic}")

            # Route by topic suffix
            if topic.endswith("/emg"):
                st.session_state["emg_latest"] = payload
                st.session_state["msg_count_emg"] = st.session_state.get("msg_count_emg", 0) + 1
                st.session_state["last_topic"] = topic
                print(f"[DASHBOARD] EMG #{st.session_state['msg_count_emg']}, effort={payload.get('effort_pct', 0):.1f}%")
                # Add to history for chart
                effort = payload.get("effort_pct", 0)
                st.session_state["emg_history"].append({
                    "time": ts,
                    "effort_pct": effort,
                    "contracting": payload.get("contracting", False),
                    "rms": payload.get("rms", 0),
                    "median_freq": payload.get("median_freq", 0),
                    "fatigue_alert": payload.get("fatigue_alert", False),
                })
                # Track session start from first EMG message
                if st.session_state["session_start_time"] is None:
                    st.session_state["session_start_time"] = ts
                    st.session_state["session_active"] = True

            elif topic.endswith("/handpose"):
                st.session_state["handpose_latest"] = payload
                print(f"[DASHBOARD] Handpose updated")

            elif topic.endswith("/fusion"):
                st.session_state["fusion_latest"] = payload
                st.session_state["msg_count_fusion"] = st.session_state.get("msg_count_fusion", 0) + 1
                st.session_state["last_topic"] = topic
                print(f"[DASHBOARD] Fusion #{st.session_state['msg_count_fusion']}, state={payload.get('state', 'N/A')}")
                # Track rep completion
                if payload.get("rep_completed"):
                    rep_entry = {
                        "rep": payload.get("rep_number", len(st.session_state["rep_log"]) + 1),
                        "state": payload.get("state", "REST"),
                        "effort_pct": payload.get("effort_pct", 0),
                        "score_delta": payload.get("score_delta", 0),
                        "timestamp": ts,
                    }
                    st.session_state["rep_log"].append(rep_entry)
                    # Trim to max
                    if len(st.session_state["rep_log"]) > REP_LOG_MAX_COUNT:
                        st.session_state["rep_log"] = st.session_state["rep_log"][-REP_LOG_MAX_COUNT:]

            elif topic.endswith("/game"):
                st.session_state["game_latest"] = payload
                st.session_state["msg_count_game"] = st.session_state.get("msg_count_game", 0) + 1
                st.session_state["last_topic"] = topic
                print(f"[DASHBOARD] Game #{st.session_state['msg_count_game']}")

            elif topic.endswith("/alerts"):
                alert_entry = {
                    "type": payload.get("type", "INFO"),
                    "message": payload.get("message", ""),
                    "severity": payload.get("severity", "info"),
                    "timestamp": ts,
                }
                st.session_state["alerts"].insert(0, alert_entry)  # Newest first
                # Trim to max
                if len(st.session_state["alerts"]) > ALERT_MAX_COUNT:
                    st.session_state["alerts"] = st.session_state["alerts"][:ALERT_MAX_COUNT]

            elif topic.endswith("/session_summary"):
                # Session ended — reset timer but keep history
                st.session_state["session_start_time"] = None
                st.session_state["session_active"] = False
                st.session_state["rep_log"] = []
                # Clear non-fatigue alerts
                st.session_state["alerts"] = [
                    a for a in st.session_state["alerts"]
                    if a.get("type") != "fatigue"
                ]
    if processed > 0:
        print(f"[DASHBOARD] Processed {processed} messages this frame")


# ─── UI Components ─────────────────────────────────────────────────────────────

def _render_top_bar():
    """Render top bar with title, status, and metrics."""
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.title("Stroke Rehab Monitor")
        # Patient ID badge + connection status
        connected = st.session_state.get("mqtt_connected", False)
        status_dot = "🟢" if connected else "🔴"
        status_text = "Connected" if connected else "Offline"

        elapsed_str = "--:--"
        if st.session_state.get("session_start_time"):
            elapsed = time.time() - st.session_state["session_start_time"]
            elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

        st.markdown(f"""
        <div style="display:flex; gap:20px; align-items:center; margin-bottom:10px;">
            <span style="background:#1e2d5a; padding:4px 12px; border-radius:12px; font-size:0.9rem;">
                Patient: <b>{PATIENT_ID}</b>
            </span>
            <span style="font-size:0.9rem;">{status_dot} {status_text}</span>
            <span style="font-size:0.9rem; color:#00c8aa;">⏱ {elapsed_str}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # 4 metric cards in 2x2 grid
        game = st.session_state.get("game_latest") or {}
        emg = st.session_state.get("emg_latest") or {}

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Score", game.get("score", 0))
        with m2:
            st.metric("Reps", game.get("reps", 0))

        m3, m4 = st.columns(2)
        with m3:
            st.metric("Level", game.get("level", 1))
        with m4:
            effort = emg.get("effort_pct", 0)
            st.metric("Effort", f"{effort:.0f}%")

    return connected


def _render_emg_chart():
    """Render live EMG signal chart with history."""
    st.markdown("### Live EMG Signal")

    emg_history = st.session_state.get("emg_history", deque())
    if not emg_history:
        st.info("Waiting for EMG data...")
        return

    # Convert deque to list for plotting
    history_list = list(emg_history)
    times = [(h["time"] - history_list[0]["time"]) - (history_list[-1]["time"] - history_list[0]["time"])
             for h in history_list]
    efforts = [h["effort_pct"] for h in history_list]
    contracting = [h["contracting"] for h in history_list]

    # Build color array based on contracting state
    colors = [FUSION_COLORS["GOOD"] if c else FUSION_COLORS["REST"] for c in contracting]

    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=times, y=efforts,
        mode="lines",
        line=dict(color="#00c8aa", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,200,170,0.1)",
        name="Effort %",
    ))

    # Threshold line
    threshold = EMG_THRESHOLD_PCT * 100
    fig.add_hline(y=threshold, line_dash="dash", line_color="rgba(255,160,0,0.7)",
                  annotation_text=f"Threshold ({threshold:.0f}%)")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,26,0.8)",
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, color="#3a4a7a", title="Time (s)",
                   range=[times[0] if times else -30, 0]),
        yaxis=dict(showgrid=True, gridcolor="#141e38", color="#3a4a7a",
                   range=[0, 105], title="Effort %"),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Inline metrics
    if history_list:
        latest = history_list[-1]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Peak", f"{latest.get('effort_pct', 0):.0f}%")
        with c2:
            st.metric("RMS", f"{latest.get('rms', 0)*1000:.1f} mV")
        with c3:
            st.metric("Freq", f"{latest.get('median_freq', 0):.0f} Hz")

        # Fatigue alert
        if latest.get("fatigue_alert"):
            st.warning("⚠️ Fatigue Alert — Consider rest period")


def _render_hand_pose():
    """Render hand pose finger extension bars."""
    st.markdown("### Hand Pose (OpenCV)")

    handpose = st.session_state.get("handpose_latest")
    if handpose is None:
        st.info("Waiting for hand detection...")
        return

    extensions = handpose.get("extensions", [0, 0, 0, 0, 0])
    moving = handpose.get("moving", [False, False, False, False, False])
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    for name, ext, mov in zip(finger_names, extensions, moving):
        indicator = "●" if mov else "○"
        label = f"{name}: {ext:.0%} {indicator}"
        st.progress(ext, text=label)

    # Confidence and moving count
    conf = handpose.get("confidence", 0)
    moving_count = sum(moving)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Confidence", f"{conf:.0%}")
    with c2:
        st.metric("Moving", f"{moving_count} fingers")


def _render_fusion_state():
    """Render fusion state badge and rep history."""
    st.markdown("### Fusion State")

    fusion = st.session_state.get("fusion_latest")
    if fusion is None:
        st.info("Waiting for fusion data...")
        return

    state = fusion.get("state", "REST")
    color = FUSION_COLORS.get(state, FUSION_COLORS["REST"])

    # Large colored badge
    st.markdown(f"""
    <div style="background:{color}; padding:15px; border-radius:12px; text-align:center;
                color:white; font-size:1.4rem; font-weight:bold; margin-bottom:15px;">
        {state.replace('_', ' ')}
    </div>
    """, unsafe_allow_html=True)

    # State explanation
    explanations = {
        "GOOD": "Intent + movement detected — reward rep",
        "INTENT_BLOCKED": "Trying but blocked — encourage patient",
        "PASSIVE_MOVE": "Movement without intent — passive assist",
        "REST": "Patient resting",
    }
    st.caption(explanations.get(state, ""))

    # Last 5 reps table
    rep_log = st.session_state.get("rep_log", [])
    if rep_log:
        st.markdown("**Recent Reps:**")
        recent_reps = rep_log[-5:][::-1]  # Newest first
        for r in recent_reps:
            rep_num = r.get("rep", 0)
            rep_state = r.get("state", "REST")
            effort = r.get("effort_pct", 0)
            st.markdown(f"• Rep {rep_num}: {rep_state} ({effort:.0f}% effort)")


def _render_game_progress():
    """Render game progress section."""
    st.markdown("### Session Progress")

    game = st.session_state.get("game_latest") or {}
    score = game.get("score", 0)
    reps = game.get("reps", 0)
    level = game.get("level", 1)

    # Progress bar (target 30 reps)
    target_reps = 30
    progress = min(1.0, reps / target_reps)
    st.progress(progress, text=f"Reps: {reps} / {target_reps}")

    # Rep quality chart (last 20 reps)
    rep_log = st.session_state.get("rep_log", [])
    if rep_log:
        recent = rep_log[-20:]
        rep_nums = [r["rep"] for r in recent]
        efforts = [r["effort_pct"] for r in recent]
        colors = [FUSION_COLORS.get(r["state"], FUSION_COLORS["REST"]) for r in recent]

        fig = go.Figure(go.Bar(
            x=rep_nums, y=efforts,
            marker_color=colors,
            text=[f"{e:.0f}%" for e in efforts],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,13,26,0.8)",
            height=200,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False, color="#3a4a7a", title="Rep #"),
            yaxis=dict(showgrid=True, gridcolor="#141e38", color="#3a4a7a",
                       range=[0, 105], title="Effort %"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Performance label from fusion state
    fusion = st.session_state.get("fusion_latest")
    if fusion:
        from core.fusion_engine import FusionEngine
        perf_label = FusionEngine.FUSION_TO_PERFORMANCE.get(fusion.get("state"), None)
        if perf_label:
            perf_color = {"good": "#1D9E75", "compensating": "#EF9F27", "poor": "#378ADD"}.get(perf_label, "#888780")
            st.markdown(f"""
            <div style="display:inline-block; background:{perf_color}; padding:6px 14px;
                        border-radius:20px; color:white; font-weight:600;">
                Performance: {perf_label.upper()}
            </div>
            """, unsafe_allow_html=True)


def _render_recovery_trend():
    """Render recovery trend from SQLite database."""
    st.markdown("### Recovery Trend")

    try:
        con = init_db()
        history = load_session_history(con, PATIENT_ID, last_n=15)
        con.close()
    except Exception:
        history = []

    if len(history) < 2:
        st.info("Collect 2+ sessions to see trend")
        return

    # Extract data
    sessions = list(range(1, len(history) + 1))
    mvc_norms = [h.get("mvc_normalised", 100) for h in history]
    avg_efforts = [h.get("avg_effort_pct", 0) for h in history]
    good_ratios = [h.get("good_ratio_pct", 0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sessions, y=mvc_norms,
        mode="lines+markers",
        name="MVC% (Strength)",
        line=dict(color="#1D9E75", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=sessions, y=avg_efforts,
        mode="lines+markers",
        name="Avg Effort%",
        line=dict(color="#378ADD", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=sessions, y=good_ratios,
        mode="lines+markers",
        name="Good Rep Ratio%",
        line=dict(color="#a060ff", width=2),
        marker=dict(size=6),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,26,0.8)",
        height=280,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False, color="#3a4a7a", title="Session #"),
        yaxis=dict(showgrid=True, gridcolor="#141e38", color="#3a4a7a", title="%"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6070a0")),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_alerts():
    """Render alerts panel."""
    st.markdown("### Doctor Alerts")

    alerts = st.session_state.get("alerts", [])

    if st.button("Clear Alerts"):
        st.session_state["alerts"] = []
        st.rerun()

    if not alerts:
        st.success("No alerts this session")
        return

    for alert in alerts[:ALERT_MAX_COUNT]:
        severity = alert.get("severity", "info")
        msg = alert.get("message", "")
        alert_type = alert.get("type", "INFO")
        ts = alert.get("timestamp", time.time())
        time_str = datetime.fromtimestamp(ts).strftime("%H:%M")

        display = f"[{time_str}] {alert_type} — {msg}"

        if severity == "danger":
            st.error(display)
        elif severity == "warning":
            st.warning(display)
        else:
            st.info(display)


# ─── Main Dashboard Layout ─────────────────────────────────────────────────────

def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="Stroke Rehab Monitor",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Initialize session state
    _init_session_state()

    # Start MQTT client exactly once (guard against Streamlit reruns)
    if not st.session_state.get("mqtt_client_started"):
        st.session_state["mqtt_client_started"] = True
        if PAHO_AVAILABLE:
            client = _start_mqtt_client()
            st.session_state["mqtt_client"] = client
        else:
            st.error("paho-mqtt not installed. Run: pip install paho-mqtt")

    # Auto-refresh every 500ms (non-blocking)
    st_autorefresh(interval=DASHBOARD_REFRESH_MS, limit=None, key="dashboard_refresh")

    # Drain MQTT queue (main thread only)
    _drain_mqtt_queue()

    # ─── TOP BAR ──────────────────────────────────────────────
    connected = _render_top_bar()

    # Always show debug info for troubleshooting
    with st.expander("Debug Info", expanded=not connected):
        st.write(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
        st.write(f"Patient ID: {PATIENT_ID}")
        st.write(f"Client ID: {st.session_state.get('mqtt_client_id', 'N/A')}")
        st.write(f"PAHO Available: {PAHO_AVAILABLE}")
        st.write(f"PAHO V2 API: {PAHO_V2}")
        st.write(f"Client Started: {st.session_state.get('mqtt_client_started', False)}")
        st.write(f"MQTT Connected: {st.session_state.get('mqtt_connected', False)}")
        st.write(f"Queue size: {_mqtt_queue.qsize()}")
        st.write(f"EMG msgs: {st.session_state.get('msg_count_emg', 0)}")
        st.write(f"Fusion msgs: {st.session_state.get('msg_count_fusion', 0)}")
        st.write(f"Game msgs: {st.session_state.get('msg_count_game', 0)}")
        st.write(f"Last topic: {st.session_state.get('last_topic', 'None')}")
        st.write(f"Session active: {st.session_state.get('session_active', False)}")
        st.write(f"EMG history len: {len(st.session_state.get('emg_history', []))}")

    if not connected:
        st.warning("Patient device offline — waiting for connection...")

    st.markdown("---")

    # ─── ROW 1: EMG + HAND POSE ───────────────────────────────
    col_emg, col_hand = st.columns([3, 2])

    with col_emg:
        _render_emg_chart()

    with col_hand:
        _render_hand_pose()

    st.markdown("---")

    # ─── ROW 2: FUSION STATE + GAME PROGRESS ──────────────────
    col_fusion, col_game = st.columns([2, 3])

    with col_fusion:
        _render_fusion_state()

    with col_game:
        _render_game_progress()

    st.markdown("---")

    # ─── ROW 3: RECOVERY TREND + ALERTS ───────────────────────
    col_trend, col_alerts = st.columns([3, 2])

    with col_trend:
        _render_recovery_trend()

    with col_alerts:
        _render_alerts()


if __name__ == "__main__":
    main()

