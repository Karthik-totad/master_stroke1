"""
config.py — Global configuration for NeuroRehab system
"""

import os

# ─── EMG Source ───────────────────────────────────────────────────────────────
# Options: "simulated" | "serial" | "bluetooth"
EMG_SOURCE = "simulated"

EMG_PORT = "/dev/ttyUSB0"       # Serial port (Linux) or "COM3" (Windows)
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
DEFAULT_PATIENT_ID = "PT001"
