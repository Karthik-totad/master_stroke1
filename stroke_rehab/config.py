"""
config.py — Global configuration for NeuroRehab system
"""

import os

# ─── EMG Source ───────────────────────────────────────────────────────────────
# Options: "simulated" | "serial" | "bluetooth"
EMG_SOURCE = "serial"

EMG_PORT = "COM13"       # Windows serial port (was /dev/ttyUSB0 for Linux)
EMG_BAUD = 115200
EMG_BT_ADDRESS = ""             # BLE MAC address if using Bluetooth

EMG_SAMPLE_RATE = 1000          # Hz
EMG_WINDOW_SIZE = 256           # Samples per feature window
EMG_STEP_SIZE = 64              # Overlap step

# ─── Signal Processing ────────────────────────────────────────────────────────
EMG_LOWPASS_HZ = 450
EMG_HIGHPASS_HZ = 20
EMG_NOTCH_HZ = 50               # Power line noise (60 for USA)

# ─── Vision ───────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# ─── ML Model ─────────────────────────────────────────────────────────────────
MODEL_PATH = "data/models/performance_classifier.pkl"
SCALER_PATH = "data/models/scaler.pkl"
LABEL_ENCODER_PATH = "data/models/label_encoder.pkl"
PERFORMANCE_LABELS = ["poor", "compensating", "good"]

# ─── Game ─────────────────────────────────────────────────────────────────────
GAME_FPS = 60
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
DEFAULT_GAME = "bubble_pop"     # bubble_pop | maze_steering | pump_the_pump | flower_bloom

# Adaptive difficulty thresholds
DIFFICULTY_INCREASE_SCORE = 80  # % score to go up
DIFFICULTY_DECREASE_SCORE = 40  # % score to go down

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_DIR = os.path.join(BASE_DIR, "data", "sessions")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
REPORT_DIR = os.path.join(BASE_DIR, "data", "doctor_reports")

os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ─── Patient ──────────────────────────────────────────────────────────────────
PATIENT_ID = "PT001"

# Backward compatibility alias
DEFAULT_PATIENT_ID = PATIENT_ID

# ─── MQTT ─────────────────────────────────────────────────────────────────────────
MQTT_BROKER = "localhost"
MQTT_PORT   = 1883

# ─── Fusion + Recovery ──────────────────────────────────────────────────────────
EMG_THRESHOLD_PCT   = 0.35     # 35% of MVC = contraction detected
CONTRACTION_MIN_MS  = 150      # ms, will be converted to samples using EMG_SAMPLE_RATE
FATIGUE_FREQ_DROP   = 0.30     # 30% median frequency drop = fatigue alert
FINGER_MOVE_THRESH  = 0.04     # normalised landmark delta for movement detection

# Derived window sizes — always computed, never hardcoded
CONTRACTION_MIN_SAMPLES = int(CONTRACTION_MIN_MS / 1000 * EMG_SAMPLE_RATE)
FFT_WINDOW_SAMPLES      = int(0.256 * EMG_SAMPLE_RATE)
RMS_WINDOW_SAMPLES      = int(0.020 * EMG_SAMPLE_RATE)

# ─── Dashboard Settings ─────────────────────────────────────────────────────────
DASHBOARD_REFRESH_MS   = 500    # MQTT poll interval in milliseconds
EMG_HISTORY_LENGTH     = 300    # samples to show in live EMG chart (~30s at 10Hz)
ALERT_MAX_COUNT        = 10     # max alerts to keep in dashboard
REP_LOG_MAX_COUNT      = 50     # max reps to keep in rep quality chart

# Fusion state colors for dashboard
FUSION_COLORS = {
    "GOOD":           "#1D9E75",
    "INTENT_BLOCKED": "#EF9F27",
    "PASSIVE_MOVE":   "#378ADD",
    "REST":           "#888780",
}

# Import FUSION_TO_PERFORMANCE from core.fusion_engine to avoid duplication
# (defined there as class attribute: FusionEngine.FUSION_TO_PERFORMANCE)
