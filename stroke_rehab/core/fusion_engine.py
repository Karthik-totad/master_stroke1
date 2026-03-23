"""
core/fusion_engine.py

FusionEngine for post-stroke rehabilitation.
Fuses EMG contraction state with OpenCV finger movement to produce
4 clinical states per rep.
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EMG_THRESHOLD_PCT, FATIGUE_FREQ_DROP, CONTRACTION_MIN_MS


class FusionEngine:
    """
    Fuses EMG contraction state with finger movement to produce
    4 clinical states per rep.

    State table:
    ┌──────────────────┬────────────┬───────────────────────────────────────┐
    │ State            │ EMG | CV   │ Clinical meaning                      │
    ├──────────────────┼────────────┼───────────────────────────────────────┤
    │ GOOD             │  ✓  |  ✓  │ Intent + movement — healthy rep       │
    │ INTENT_BLOCKED   │  ✓  |  ✗  │ Trying but can't move — motor damage  │
    │ PASSIVE_MOVE     │  ✗  |  ✓  │ Finger moved without muscle intent    │
    │ REST             │  ✗  |  ✗  │ Resting / not attempting              │
    └──────────────────┴────────────┴───────────────────────────────────────┘
    """

    GOOD = "GOOD"
    INTENT_BLOCKED = "INTENT_BLOCKED"
    PASSIVE_MOVE = "PASSIVE_MOVE"
    REST = "REST"

    # Mapping from fusion states to ML performance labels
    FUSION_TO_PERFORMANCE = {
        "GOOD":           "good",
        "INTENT_BLOCKED": "compensating",
        "PASSIVE_MOVE":   "poor",
        "REST":           None,            # REST reps are not labelled
    }

    def __init__(self):
        self.rep_states = []
        self.score = 0
        self.reps = 0
        self.level = 1
        self.consecutive_blocked = 0

        # Contraction tracking
        self.is_contracting = False
        self.contraction_start = None
        self._rep_rms_buf = []
        self._freq_history = []

    def classify(self, emg_features, finger_extensions, finger_moving):
        """
        Classify current state from EMG features and finger data.

        Args:
            emg_features: dict from EMGPreprocessor containing:
                - rms, mav, zero_crossings, slope_sign_changes, waveform_length
                - variance, mean_freq, median_freq, peak_amplitude, contraction_ratio
            finger_extensions: list[float] of 5 values [0.0-1.0] for thumb, index, middle, ring, pinky
            finger_moving: list[bool] of 5 values indicating movement

        Returns:
            dict with fusion state and metrics
        """
        if emg_features is None:
            return None

        # Map from existing preprocessor output
        contraction_ratio = emg_features.get("contraction_ratio", 0.0)
        emg_active = contraction_ratio >= EMG_THRESHOLD_PCT
        effort_pct = min(100.0, round(contraction_ratio * 100, 1))
        median_freq = emg_features.get("median_freq", 0.0)
        rms = emg_features.get("rms", 0.0)

        # Any finger moving
        finger_moving_any = any(finger_moving) if finger_moving else False

        # Determine state
        if emg_active and finger_moving_any:
            state = self.GOOD
            self.consecutive_blocked = 0
        elif emg_active and not finger_moving_any:
            state = self.INTENT_BLOCKED
            self.consecutive_blocked += 1
        elif not emg_active and finger_moving_any:
            state = self.PASSIVE_MOVE
            self.consecutive_blocked = 0
        else:
            state = self.REST
            self.consecutive_blocked = 0

        # Track contraction for rep detection
        rep_completed = False
        rep_effort = 0.0
        rep_duration = 0

        if emg_active and not self.is_contracting:
            # Contraction started
            self.is_contracting = True
            self.contraction_start = time.time()
            self._rep_rms_buf = []

        if self.is_contracting and emg_active:
            self._rep_rms_buf.append(rms)

        if not emg_active and self.is_contracting:
            # Contraction ended
            self.is_contracting = False
            duration_ms = int((time.time() - self.contraction_start) * 1000)

            if duration_ms >= CONTRACTION_MIN_MS:
                rep_effort = min(100.0, round(
                    np.mean(self._rep_rms_buf) / max(rms, 1e-6) * 100 if self._rep_rms_buf else effort_pct, 1
                ))
                rep_completed = True
                rep_duration = duration_ms

        # Fatigue detection
        fatigue_alert = False
        if median_freq > 0:
            self._freq_history.append(median_freq)
            if len(self._freq_history) >= 10:
                baseline = np.mean(list(self._freq_history)[:5])
                current = np.mean(list(self._freq_history)[-5:])
                if baseline > 0 and (baseline - current) / baseline >= FATIGUE_FREQ_DROP:
                    fatigue_alert = True

        # Score rep on completion
        score_delta = 0
        if rep_completed:
            self.reps += 1
            if state == self.GOOD:
                score_delta = 10
            elif state == self.INTENT_BLOCKED:
                score_delta = 7  # reward the attempt
            elif state == self.PASSIVE_MOVE:
                score_delta = 3
            self.score += score_delta
            self.rep_states.append({
                "rep": self.reps,
                "state": state,
                "effort_pct": rep_effort,
                "duration_ms": rep_duration,
                "score_delta": score_delta,
                "timestamp": time.time(),
            })
            if self.reps % 10 == 0:
                self.level += 1

        # Alert for consecutive blocked attempts
        alert = None
        if self.consecutive_blocked >= 3:
            alert = {
                "type": "MOTOR_BLOCK",
                "message": f"Patient has attempted {self.consecutive_blocked} reps "
                           f"with EMG activity but no finger movement. "
                           f"Consider adjusting exercise difficulty.",
                "severity": "warning",
            }

        return {
            "state": state,
            "emg_active": emg_active,
            "finger_moving": finger_moving_any,
            "effort_pct": effort_pct,
            "extensions": finger_extensions if finger_extensions else [0.0] * 5,
            "rep_completed": rep_completed,
            "rep_effort": rep_effort,
            "rep_duration_ms": rep_duration,
            "score": self.score,
            "score_delta": score_delta,
            "reps": self.reps,
            "level": self.level,
            "alert": alert,
            "fatigue_alert": fatigue_alert,
            "median_freq": median_freq,
            "timestamp": time.time(),
        }

    def get_game_state(self):
        """Return current game state for MQTT publishing."""
        return {
            "score": self.score,
            "reps": self.reps,
            "level": self.level,
            "ts": time.time(),
        }

    def get_performance_label(self) -> str | None:
        """
        Returns the ML performance label for the most recent completed rep,
        or None if the last state was REST.
        Maps fusion states to trainer.py performance labels.
        """
        if not self.rep_states:
            return None
        last_state = self.rep_states[-1]["state"]
        return self.FUSION_TO_PERFORMANCE.get(last_state)
