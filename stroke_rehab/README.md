# 🧠 NeuroRehab — Smart Stroke Rehabilitation System

A complete end-to-end stroke rehabilitation platform combining **real EMG biofeedback**, **computer vision hand tracking**, **machine learning**, and **gamified therapy**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)

---

## 🎮 Four Mini-Games

| Game | Control | Exercise |
|------|---------|----------|
| 🫧 Bubble Pop | Pinch thumb + index | Pinch coordination |
| 🌸 Flower Bloom | Open / close hand | Finger extension |
| 💦 Pump the Pump | Squeeze EMG sensor | Grip strength |
| 🌀 Maze Steering | Tilt wrist | Wrist rotation |

---

## 🏗️ Project Structure

```
neurorehab/
├── data_acquisition/
│   ├── emg_reader.py        # ESP32 serial / BLE / simulated EMG
│   ├── preprocessor.py      # Bandpass filter, RMS, MAV, ZC features
│   └── session_logger.py    # CSV session logging
├── vision/
│   └── tracker.py           # MediaPipe hand + pose tracking
├── ml/
│   ├── trainer.py           # ML classifier (poor/compensating/good)
│   └── doctor_report.py     # JSON therapy plan parser
├── game/
│   ├── game_engine.py       # Adaptive difficulty + scoring
│   ├── bubble_pop.py        # Bubble Pop (Pygame)
│   ├── flower_bloom.py      # Flower Bloom (Pygame)
│   └── pump_maze.py         # Pump + Maze (Pygame)
├── ui/
│   ├── dashboard.py         # Streamlit live dashboard
│   └── neurorehab_games.html # Browser game (ESP32 + MediaPipe)
├── scripts/
│   ├── run_demo.py          # Full pipeline demo
│   ├── train_model.py       # Train ML model
│   └── run_dashboard.sh     # Launch Streamlit
├── data/
│   └── doctor_reports/
│       └── PT001_report.json
├── config.py
└── requirements.txt
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/neurorehab.git
cd neurorehab
pip install -r requirements.txt
python scripts/train_model.py
```

### Run browser game (no hardware needed)
```bash
cd ui
python3 -m http.server 8080
# Open Chrome → http://localhost:8080/neurorehab_games.html
```

### Run full pipeline demo
```bash
python scripts/run_demo.py --pipeline-only --duration 30
```

### Launch Streamlit dashboard
```bash
streamlit run ui/dashboard.py
```

---

## 🔌 ESP32 EMG Setup

Flash this to your ESP32:
```cpp
void setup() { Serial.begin(115200); }
void loop() {
  Serial.println("EMG:" + String(analogRead(A0)));
  delay(1);
}
```

Then click **"🔌 Connect ESP32 EMG"** in the browser game.

> Requires **Chrome or Edge** (Web Serial API). Serve via `python3 -m http.server`, not `file://`.

---

## 🤖 ML Model

Classifies as **poor / compensating / good** from EMG + motion features.

```bash
python scripts/train_model.py              # train on synthetic data
python scripts/train_model.py --n-synthetic 5000  # more data
```

---

## 📋 Doctor Report Format

```json
{
  "patient_id": "PT001",
  "severity": "moderate",
  "prescribed_exercises": [
    { "type": "grip_strength", "game": "pump_the_pump", "difficulty": "easy" }
  ],
  "target_rom_degrees": 45
}
```

---

## 🛠️ Tech Stack

Python · OpenCV · MediaPipe · scikit-learn · Streamlit · Pygame · Web Serial API

---

## 📄 License

MIT License
