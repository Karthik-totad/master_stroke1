"""
data_acquisition/session_logger.py

Logs EMG + motion features to CSV/JSON for later analysis and model retraining.
"""

import os
import csv
import json
import time
from datetime import datetime
from typing import Optional
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SESSION_DIR
from data_acquisition.preprocessor import EMGFeatures


class SessionLogger:
    """
    Logs one therapy session to disk.
    Creates:
      - {session_id}_emg_features.csv   — per-window EMG features
      - {session_id}_summary.json       — session summary stats
    """

    EMG_FEATURE_COLS = EMGFeatures.feature_names()
    MOTION_COLS = ["arm_angle", "rom", "stability", "velocity"]
    GAME_COLS = ["game_name", "score", "difficulty", "performance_label"]

    def __init__(self, patient_id: str, session_id: Optional[str] = None):
        self.patient_id = patient_id
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or f"{patient_id}_{ts}"
        self.start_time = time.time()

        self._csv_path = os.path.join(SESSION_DIR, f"{self.session_id}_features.csv")
        self._summary_path = os.path.join(SESSION_DIR, f"{self.session_id}_summary.json")

        self._rows: list[dict] = []
        self._csv_file = open(self._csv_path, "w", newline="")
        all_cols = (
            ["timestamp", "patient_id", "session_id"]
            + self.EMG_FEATURE_COLS
            + self.MOTION_COLS
            + self.GAME_COLS
        )
        self._writer = csv.DictWriter(self._csv_file, fieldnames=all_cols)
        self._writer.writeheader()

        print(f"[SessionLogger] Session started: {self.session_id}")

    def log(
        self,
        emg_features: EMGFeatures,
        arm_angle: float = 0.0,
        rom: float = 0.0,
        stability: float = 1.0,
        velocity: float = 0.0,
        game_name: str = "",
        score: float = 0.0,
        difficulty: str = "easy",
        performance_label: str = "",
    ):
        """Log one feature window."""
        row = {
            "timestamp": round(time.time() - self.start_time, 3),
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            # EMG features
            **{k: round(v, 6) for k, v in zip(
                self.EMG_FEATURE_COLS, emg_features.to_array()
            )},
            # Motion
            "arm_angle": round(arm_angle, 2),
            "rom": round(rom, 2),
            "stability": round(stability, 4),
            "velocity": round(velocity, 4),
            # Game
            "game_name": game_name,
            "score": round(score, 2),
            "difficulty": difficulty,
            "performance_label": performance_label,
        }
        self._writer.writerow(row)
        self._rows.append(row)

    def close(self) -> dict:
        """Flush and write session summary. Returns summary dict."""
        self._csv_file.flush()
        self._csv_file.close()

        summary = self._compute_summary()
        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[SessionLogger] Session saved: {self.session_id}")
        print(f"  Rows: {len(self._rows)}  |  Duration: {summary['duration_seconds']:.1f}s")
        return summary

    def _compute_summary(self) -> dict:
        if not self._rows:
            return {"session_id": self.session_id, "rows": 0}

        rms_vals = [r["rms"] for r in self._rows]
        rom_vals = [r["rom"] for r in self._rows]
        scores = [r["score"] for r in self._rows if r["score"] > 0]
        labels = [r["performance_label"] for r in self._rows if r["performance_label"]]

        label_counts = {}
        for l in labels:
            label_counts[l] = label_counts.get(l, 0) + 1
        dominant_label = max(label_counts, key=label_counts.get) if label_counts else "unknown"

        return {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration_seconds": round(time.time() - self.start_time, 1),
            "total_windows": len(self._rows),
            "emg": {
                "mean_rms": round(float(np.mean(rms_vals)), 4),
                "max_rms": round(float(np.max(rms_vals)), 4),
                "mean_contraction_ratio": round(
                    float(np.mean([r["contraction_ratio"] for r in self._rows])), 4
                ),
            },
            "motion": {
                "mean_rom": round(float(np.mean(rom_vals)), 2),
                "max_rom": round(float(np.max(rom_vals)), 2),
                "mean_stability": round(
                    float(np.mean([r["stability"] for r in self._rows])), 4
                ),
            },
            "game": {
                "mean_score": round(float(np.mean(scores)), 2) if scores else 0.0,
                "dominant_performance": dominant_label,
                "label_distribution": label_counts,
            },
            "csv_path": self._csv_path,
        }


if __name__ == "__main__":
    from data_acquisition.preprocessor import EMGFeatures
    logger = SessionLogger("PT_TEST")
    for i in range(5):
        feat = EMGFeatures(
            rms=0.3+i*0.05, mav=0.25, zc=12, ssc=8, wl=1.2,
            var=0.01, mean_freq=145.0, median_freq=120.0,
            peak_amp=0.8, contraction_ratio=0.4,
        )
        logger.log(feat, arm_angle=30+i, rom=45, stability=0.9, score=75.0,
                   game_name="bubble_pop", performance_label="good")
    summary = logger.close()
    print(json.dumps(summary, indent=2))
