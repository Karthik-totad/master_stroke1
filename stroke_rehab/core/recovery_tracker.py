"""
core/recovery_tracker.py

Recovery tracking with SQLite storage and MQTT publishing.
Tracks long-term muscle recovery across sessions using EMG metrics.
"""

import argparse
import sqlite3
import json
import time
import os
import sys
from datetime import datetime
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EMG_SAMPLE_RATE, CONTRACTION_MIN_MS, PATIENT_ID

# Database path
DB_PATH = "rehab_recovery.db"


def init_db(path=DB_PATH):
    """Initialize SQLite database with patients, sessions, and rep_log tables."""
    con = sqlite3.connect(path)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id          TEXT PRIMARY KEY,
            name        TEXT,
            stroke_date TEXT,
            created_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      TEXT,
            session_date    TEXT,
            session_number  INTEGER,
            duration_secs   REAL,

            mvc_raw         REAL,
            mvc_normalised  REAL,
            baseline_mvc    REAL,

            activation_latency_ms   REAL,
            sustained_time_ms       REAL,
            avg_effort_pct          REAL,
            peak_effort_pct         REAL,

            median_freq_start_hz    REAL,
            median_freq_end_hz      REAL,
            freq_drop_pct           REAL,

            total_reps          INTEGER,
            good_reps           INTEGER,
            intent_blocked_reps INTEGER,
            passive_reps        INTEGER,

            rep_effort_json     TEXT,
            notes               TEXT,
            created_at          TEXT
        );

        CREATE TABLE IF NOT EXISTS rep_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      INTEGER,
            rep_number      INTEGER,
            state           TEXT,
            effort_pct      REAL,
            duration_ms     REAL,
            median_freq_hz  REAL,
            timestamp       REAL
        );
    """)
    con.commit()
    return con


def ensure_patient(con, patient_id, name="Patient"):
    """Create patient record if it doesn't exist."""
    cur = con.cursor()
    cur.execute("SELECT id FROM patients WHERE id=?", (patient_id,))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO patients VALUES (?,?,?,?)",
            (patient_id, name, None, datetime.now().isoformat())
        )
        con.commit()
        print(f"[DB] New patient record created: {patient_id}")


def next_session_number(con, patient_id):
    """Get next session number for patient."""
    cur = con.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM sessions WHERE patient_id=?", (patient_id,)
    )
    return (cur.fetchone()[0] or 0) + 1


def save_session(con, patient_id, metrics, rep_log):
    """Save session metrics and rep log to database."""
    cur = con.cursor()
    now = datetime.now().isoformat()
    session_num = next_session_number(con, patient_id)

    # Compute baseline MVC (first session) for normalisation
    cur.execute(
        "SELECT mvc_raw FROM sessions WHERE patient_id=? ORDER BY session_number LIMIT 1",
        (patient_id,)
    )
    row = cur.fetchone()
    baseline_mvc = row[0] if row else metrics.get("mvc_raw", 1.0)

    mvc_normalised = round(metrics.get("mvc_raw", 1.0) / max(baseline_mvc, 1e-6) * 100, 1)

    cur.execute("""
        INSERT INTO sessions (
            patient_id, session_date, session_number, duration_secs,
            mvc_raw, mvc_normalised, baseline_mvc,
            activation_latency_ms, sustained_time_ms, avg_effort_pct, peak_effort_pct,
            median_freq_start_hz, median_freq_end_hz, freq_drop_pct,
            total_reps, good_reps, intent_blocked_reps, passive_reps,
            rep_effort_json, notes, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        patient_id, now, session_num, metrics.get("duration_secs", 0),
        metrics.get("mvc_raw", 1.0), mvc_normalised, baseline_mvc,
        metrics.get("activation_latency_ms"),
        metrics.get("sustained_time_ms"),
        metrics.get("avg_effort_pct"),
        metrics.get("peak_effort_pct"),
        metrics.get("median_freq_start_hz"),
        metrics.get("median_freq_end_hz"),
        metrics.get("freq_drop_pct"),
        metrics.get("total_reps", 0),
        metrics.get("good_reps", 0),
        metrics.get("intent_blocked_reps", 0),
        metrics.get("passive_reps", 0),
        json.dumps([r["effort_pct"] for r in rep_log]),
        metrics.get("notes", ""),
        now,
    ))
    session_id = cur.lastrowid

    for rep in rep_log:
        cur.execute("""
            INSERT INTO rep_log
            (session_id, rep_number, state, effort_pct, duration_ms, median_freq_hz, timestamp)
            VALUES (?,?,?,?,?,?,?)
        """, (
            session_id, rep["rep_number"], rep["state"],
            rep["effort_pct"], rep["duration_ms"],
            rep.get("median_freq_hz", 0), rep["timestamp"],
        ))

    con.commit()
    print(f"[DB] Session #{session_num} saved (id={session_id})")
    return session_id, session_num, mvc_normalised, baseline_mvc


def load_session_history(con, patient_id, last_n=20):
    """Load last N sessions for patient."""
    cur = con.cursor()
    cur.execute("""
        SELECT session_number, session_date,
               mvc_normalised, activation_latency_ms, sustained_time_ms,
               avg_effort_pct, median_freq_start_hz, freq_drop_pct,
               total_reps, good_reps, intent_blocked_reps
        FROM sessions
        WHERE patient_id=?
        ORDER BY session_number DESC
        LIMIT ?
    """, (patient_id, last_n))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return list(reversed(rows))  # chronological order


class RecoveryTracker:
    """
    Tracks recovery metrics across a session.
    Consumes preprocessed EMG features and fusion results.
    """

    def __init__(self, patient_id, con, mqtt_pub):
        self.pid = patient_id
        self.con = con
        self.mqtt = mqtt_pub

        self.rep_log = []
        self.rep_count = 0
        self.session_start = time.time()

        # Metrics tracking
        self.all_efforts = []
        self.freq_at_start = None
        self.freq_at_end = None
        self._mvc = 1.0

    def log_rep(self, fusion_result):
        """Log a completed rep from fusion result."""
        if not fusion_result or not fusion_result.get("rep_completed"):
            return None

        self.rep_count += 1
        rep_effort = fusion_result.get("rep_effort", 0.0)
        self.all_efforts.append(rep_effort)

        rep = {
            "rep_number": self.rep_count,
            "state": fusion_result.get("state", "REST"),
            "effort_pct": rep_effort,
            "duration_ms": fusion_result.get("rep_duration_ms", 0),
            "median_freq_hz": fusion_result.get("median_freq", 0),
            "timestamp": time.time(),
        }
        self.rep_log.append(rep)

        # Update frequency tracking
        if fusion_result.get("median_freq"):
            if self.freq_at_start is None:
                self.freq_at_start = fusion_result["median_freq"]
            self.freq_at_end = fusion_result["median_freq"]

        quality = "GOOD" if rep_effort >= 60 else "OK" if rep_effort >= 35 else "WEAK"
        print(f"  Rep #{self.rep_count:3d}  effort={rep_effort:5.1f}%  "
              f"dur={rep['duration_ms']:5.0f}ms  freq={rep['median_freq_hz']:.0f}Hz  [{quality}]")

        return rep

    def build_metrics(self):
        """Build session summary metrics."""
        efforts = [r["effort_pct"] for r in self.rep_log if r["effort_pct"] > 0]
        states = [r["state"] for r in self.rep_log]

        freq_drop = None
        if self.freq_at_start and self.freq_at_end and self.freq_at_start > 0:
            freq_drop = round(
                (self.freq_at_start - self.freq_at_end) / self.freq_at_start * 100, 1
            )

        return {
            "mvc_raw": round(self._mvc, 5),
            "activation_latency_ms": None,  # Not measured in basic fusion mode
            "sustained_time_ms": None,      # Not measured in basic fusion mode
            "avg_effort_pct": round(float(np.mean(efforts)), 1) if efforts else 0,
            "peak_effort_pct": round(float(np.max(efforts)), 1) if efforts else 0,
            "median_freq_start_hz": round(self.freq_at_start or 0, 1),
            "median_freq_end_hz": round(self.freq_at_end or 0, 1),
            "freq_drop_pct": freq_drop,
            "total_reps": self.rep_count,
            "good_reps": states.count("GOOD"),
            "intent_blocked_reps": states.count("INTENT_BLOCKED"),
            "passive_reps": states.count("PASSIVE_MOVE"),
            "duration_secs": round(time.time() - self.session_start, 1),
            "notes": "",
        }


def print_recovery_report(history, patient_id):
    """Print recovery report to terminal."""
    if not history:
        print("[Report] No session history found.")
        return

    print("\n" + "=" * 62)
    print(f"  RECOVERY REPORT — {patient_id}")
    print("=" * 62)

    latest = history[-1]
    first = history[0]

    print(f"  Sessions completed : {len(history)}")
    print(f"  First session      : {first['session_date'][:10]}")
    print(f"  Latest session     : {latest['session_date'][:10]}")

    # Strength recovery
    mvc_trend = [s["mvc_normalised"] for s in history if s["mvc_normalised"]]
    if len(mvc_trend) >= 2:
        delta = mvc_trend[-1] - mvc_trend[0]
        print(f"\n  Strength (MVC normalised)")
        print(f"    Session 1  : 100%")
        print(f"    Latest     : {mvc_trend[-1]:.1f}%")
        arrow = "↑" if delta >= 0 else "↓"
        print(f"    Change     : {arrow} {abs(delta):.1f}%")

    # Latency recovery
    lat = [s["activation_latency_ms"] for s in history if s["activation_latency_ms"]]
    if len(lat) >= 2:
        print(f"\n  Activation latency (lower = better)")
        print(f"    Session 1  : {lat[0]:.0f} ms")
        print(f"    Latest     : {lat[-1]:.0f} ms")
        improvement = lat[0] - lat[-1]
        arrow = "↓" if improvement >= 0 else "↑"
        print(f"    Change     : {arrow} {abs(improvement):.0f} ms")

    # Effort trend
    efforts = [s["avg_effort_pct"] for s in history if s["avg_effort_pct"]]
    if efforts:
        print(f"\n  Average effort per session")
        for s in history[-5:]:
            bar = int((s["avg_effort_pct"] or 0) / 5)
            print(f"    S{s['session_number']:2d}  [{'#'*bar:.<20}] {s['avg_effort_pct']:.1f}%")

    # Rep quality
    print(f"\n  Latest session rep quality")
    total = latest["total_reps"] or 1
    good = latest["good_reps"] or 0
    blocked = latest["intent_blocked_reps"] or 0
    passive = latest.get("passive_reps") or 0
    print(f"    GOOD            : {good:3d} ({good/total*100:.0f}%)")
    print(f"    INTENT_BLOCKED  : {blocked:3d} ({blocked/total*100:.0f}%)")
    print(f"    PASSIVE_MOVE    : {passive:3d} ({passive/total*100:.0f}%)")

    print("=" * 62)


# Matplotlib is optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


def plot_recovery_charts(history, patient_id):
    """Plot recovery charts using matplotlib."""
    if not PLOT_AVAILABLE or not history:
        return

    sessions = [s["session_number"] for s in history]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Recovery Progress — {patient_id}", fontsize=14, fontweight="medium")
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.4)

    colors = {"bar": "#1D9E75", "line": "#378ADD", "warn": "#E24B4A", "neutral": "#888780"}

    # 1. MVC Strength
    ax1 = fig.add_subplot(gs[0, 0])
    mvc = [s["mvc_normalised"] for s in history]
    ax1.plot(sessions, mvc, color=colors["line"], marker="o", markersize=5, linewidth=2)
    ax1.axhline(100, color=colors["neutral"], linewidth=0.5, linestyle="--")
    ax1.fill_between(sessions, mvc, alpha=0.1, color=colors["line"])
    ax1.set_title("Strength — MVC %", fontsize=11)
    ax1.set_xlabel("Session", fontsize=9)
    ax1.set_ylabel("% of baseline", fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(labelsize=8)

    # 2. Activation Latency
    ax2 = fig.add_subplot(gs[0, 1])
    lat = [s["activation_latency_ms"] for s in history]
    valid = [(s, l) for s, l in zip(sessions, lat) if l]
    if valid:
        sx, ly = zip(*valid)
        ax2.plot(sx, ly, color=colors["warn"], marker="o", markersize=5, linewidth=2)
        ax2.fill_between(sx, ly, alpha=0.1, color=colors["warn"])
    ax2.set_title("Activation latency (lower = better)", fontsize=11)
    ax2.set_xlabel("Session", fontsize=9)
    ax2.set_ylabel("Milliseconds", fontsize=9)
    ax2.tick_params(labelsize=8)

    # 3. Average effort
    ax3 = fig.add_subplot(gs[0, 2])
    efforts = [s["avg_effort_pct"] or 0 for s in history]
    bars = ax3.bar(sessions, efforts, color=colors["bar"], width=0.6)
    for b, v in zip(bars, efforts):
        if v > 0:
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                     f"{v:.0f}", ha="center", fontsize=7, color="#333")
    ax3.set_title("Avg effort per session", fontsize=11)
    ax3.set_xlabel("Session", fontsize=9)
    ax3.set_ylabel("% MVC", fontsize=9)
    ax3.set_ylim(0, 105)
    ax3.tick_params(labelsize=8)

    # 4. Rep quality stacked bar
    ax4 = fig.add_subplot(gs[1, 0])
    good = [s["good_reps"] or 0 for s in history]
    blocked = [s["intent_blocked_reps"] or 0 for s in history]
    passive = [s.get("passive_reps") or 0 for s in history]
    ax4.bar(sessions, good, label="GOOD", color=colors["bar"], width=0.6)
    ax4.bar(sessions, blocked, bottom=good, label="INTENT_BLOCKED",
            color="#EF9F27", width=0.6)
    ax4.bar(sessions, passive,
            bottom=[g + b for g, b in zip(good, blocked)],
            label="PASSIVE", color=colors["neutral"], width=0.6)
    ax4.set_title("Rep quality over sessions", fontsize=11)
    ax4.set_xlabel("Session", fontsize=9)
    ax4.set_ylabel("Rep count", fontsize=9)
    ax4.legend(fontsize=7)
    ax4.tick_params(labelsize=8)

    # 5. Spectral frequency (fatigue marker)
    ax5 = fig.add_subplot(gs[1, 1])
    freq_s = [s["median_freq_start_hz"] for s in history if s["median_freq_start_hz"]]
    freq_e = [s["median_freq_end_hz"] for s in history if s["median_freq_end_hz"]]
    fs_sess = [s["session_number"] for s in history if s["median_freq_start_hz"]]
    if freq_s:
        ax5.plot(fs_sess, freq_s, color=colors["line"], label="Start", marker="o",
                 markersize=4, linewidth=2)
        ax5.plot(fs_sess, freq_e, color=colors["warn"], label="End", marker="s",
                 markersize=4, linewidth=2, linestyle="--")
        ax5.fill_between(fs_sess, freq_s, freq_e, alpha=0.1, color="#888")
    ax5.set_title("EMG median frequency (fatigue)", fontsize=11)
    ax5.set_xlabel("Session", fontsize=9)
    ax5.set_ylabel("Hz", fontsize=9)
    ax5.legend(fontsize=7)
    ax5.tick_params(labelsize=8)

    # 6. Recovery score composite
    ax6 = fig.add_subplot(gs[1, 2])
    scores = []
    for s in history:
        sc = 0
        wt = 0
        if s["mvc_normalised"]:
            sc += min(s["mvc_normalised"], 150) * 0.40
            wt += 0.40
        if s["avg_effort_pct"]:
            sc += s["avg_effort_pct"] * 0.30
            wt += 0.30
        total = s["total_reps"] or 1
        good_ratio = (s["good_reps"] or 0) / total * 100
        sc += good_ratio * 0.30
        wt += 0.30
        scores.append(round(sc / max(wt, 1e-6), 1))

    sc_colors = [colors["bar"] if s >= 70 else
                 "#EF9F27" if s >= 40 else colors["warn"] for s in scores]
    ax6.bar(sessions, scores, color=sc_colors, width=0.6)
    ax6.axhline(70, color=colors["bar"], linewidth=0.7, linestyle="--", label="Target 70")
    ax6.set_title("Composite recovery score", fontsize=11)
    ax6.set_xlabel("Session", fontsize=9)
    ax6.set_ylabel("Score (0–100)", fontsize=9)
    ax6.set_ylim(0, 105)
    ax6.legend(fontsize=7)
    ax6.tick_params(labelsize=8)

    chart_path = f"recovery_report_{patient_id}.png"
    plt.savefig(chart_path, dpi=120, bbox_inches="tight")
    print(f"[Chart] Saved to {chart_path}")
    plt.show()
