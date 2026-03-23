"""
ml/trainer.py + ml/predictor.py — combined ML module

Trains a Random Forest classifier on EMG + motion features
to classify patient performance as: poor / compensating / good

Also includes ProgressTracker for session-over-session metrics.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_DIR, SESSION_DIR, PERFORMANCE_LABELS

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[ml] scikit-learn not installed")

from data_acquisition.preprocessor import EMGFeatures


# ─── Feature vector ───────────────────────────────────────────────────────────

ALL_FEATURES = (
    EMGFeatures.feature_names()
    + ["arm_angle", "rom", "stability", "velocity"]
)


def make_feature_vector(emg: EMGFeatures, arm_angle=0., rom=0.,
                         stability=1., velocity=0.) -> np.ndarray:
    return np.concatenate([
        emg.to_array(),
        [arm_angle, rom, stability, velocity]
    ])


# ─── Synthetic training data ──────────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """
    Generate labelled synthetic training data based on domain knowledge.
    Used when no real session data is available.

    # NOTE: These distributions are calibrated to EMGPreprocessor normalised output.
    # Replace with real labelled session data as soon as 20+ sessions are available.
    # Real data always outperforms synthetic — retrain when data is available.
    """
    np.random.seed(42)
    rows = []

    for label, props in [
        ("poor", dict(
            rms=(0.03, 0.01),      mav=(0.025, 0.01),
            zc=(3, 1),             ssc=(2, 1),
            wl=(0.08, 0.03),       var=(0.0003, 0.0001),
            mean_freq=(55, 15),    median_freq=(45, 12),
            peak=(0.08, 0.03),     cr=(0.05, 0.02),
            angle=(40, 12),        rom=(8, 4),
            stab=(0.35, 0.12),     vel=(0.5, 0.3)
        )),
        ("compensating", dict(
            rms=(0.10, 0.03),      mav=(0.09, 0.025),
            zc=(10, 4),            ssc=(8, 3),
            wl=(0.45, 0.12),       var=(0.0015, 0.0006),
            mean_freq=(95, 20),    median_freq=(80, 15),
            peak=(0.25, 0.07),     cr=(0.22, 0.07),
            angle=(75, 18),        rom=(22, 8),
            stab=(0.60, 0.12),     vel=(2.0, 0.8)
        )),
        ("good", dict(
            rms=(0.22, 0.05),      mav=(0.19, 0.04),
            zc=(22, 6),            ssc=(18, 5),
            wl=(1.2, 0.25),        var=(0.007, 0.002),
            mean_freq=(135, 18),   median_freq=(115, 14),
            peak=(0.48, 0.08),     cr=(0.52, 0.09),
            angle=(105, 14),       rom=(42, 9),
            stab=(0.82, 0.07),     vel=(4.0, 1.2)
        )),
    ]:
        n = n_samples // 3
        p = props
        for _ in range(n):
            rows.append({
                "rms": max(0, np.random.normal(p["rms"][0], p["rms"][1])),
                "mav": max(0, np.random.normal(p["mav"][0], p["mav"][1])),
                "zero_crossings": max(0, np.random.normal(p["zc"][0], p["zc"][1])),
                "slope_sign_changes": max(0, np.random.normal(p["ssc"][0], p["ssc"][1])),
                "waveform_length": max(0, np.random.normal(p["wl"][0], p["wl"][1])),
                "variance": max(0, np.random.normal(p["var"][0], p["var"][1])),
                "mean_freq": max(0, np.random.normal(p["mean_freq"][0], p["mean_freq"][1])),
                "median_freq": max(0, np.random.normal(p["median_freq"][0], p["median_freq"][1])),
                "peak_amplitude": np.clip(np.random.normal(p["peak"][0], p["peak"][1]), 0, 1),
                "contraction_ratio": np.clip(np.random.normal(p["cr"][0], p["cr"][1]), 0, 1),
                "arm_angle": np.clip(np.random.normal(p["angle"][0], p["angle"][1]), 0, 180),
                "rom": max(0, np.random.normal(p["rom"][0], p["rom"][1])),
                "stability": np.clip(np.random.normal(p["stab"][0], p["stab"][1]), 0, 1),
                "velocity": max(0, np.random.normal(p["vel"][0], p["vel"][1])),
                "label": label,
            })

    return pd.DataFrame(rows)


# ─── Trainer ──────────────────────────────────────────────────────────────────

class PerformanceTrainer:
    """Train and evaluate EMG + motion performance classifier."""

    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model_path = os.path.join(MODEL_DIR, "performance_classifier.pkl")
        self.scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        self.le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        self.meta_path = os.path.join(MODEL_DIR, "model_meta.json")

    def load_session_data(self) -> Optional[pd.DataFrame]:
        """Load all logged session CSVs."""
        dfs = []
        for f in os.listdir(SESSION_DIR):
            if f.endswith("_features.csv"):
                try:
                    df = pd.read_csv(os.path.join(SESSION_DIR, f))
                    if "performance_label" in df.columns:
                        df = df[df["performance_label"].isin(PERFORMANCE_LABELS)]
                        dfs.append(df)
                except Exception:
                    pass
        return pd.concat(dfs, ignore_index=True) if dfs else None

    def train(self, use_real_data: bool = True, n_synthetic: int = 3000) -> dict:
        """
        Train the classifier.
        Merges real session data (if any) with synthetic samples.
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not installed"}

        # Load data
        df = None
        if use_real_data:
            df = self.load_session_data()

        synth = generate_synthetic_dataset(n_synthetic)

        if df is not None and len(df) > 50:
            print(f"[trainer] Real session data: {len(df)} rows")
            # Rename columns for consistency
            if "performance_label" in df.columns:
                df = df.rename(columns={"performance_label": "label"})
            combined = pd.concat([
                df[ALL_FEATURES + ["label"]],
                synth[ALL_FEATURES + ["label"]],
            ], ignore_index=True)
        else:
            print(f"[trainer] Using synthetic dataset only ({n_synthetic} samples)")
            combined = synth

        combined = combined.dropna()
        X = combined[ALL_FEATURES].values
        y = combined["label"].values

        # Encode labels
        le = LabelEncoder()
        le.fit(PERFORMANCE_LABELS)
        y_enc = le.transform(y)

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        cv_scores = cross_val_score(model, X_scaled, y_enc, cv=5, scoring="accuracy")

        print(f"[trainer] Accuracy: {report['accuracy']:.3f}")
        print(f"[trainer] CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Save
        joblib.dump(model, self.model_path)
        joblib.dump(scaler, self.scaler_path)
        joblib.dump(le, self.le_path)

        meta = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(combined),
            "n_real_samples": len(df) if df is not None else 0,
            "accuracy": float(report["accuracy"]),
            "cv_accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "feature_names": ALL_FEATURES,
            "classes": list(le.classes_),
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[trainer] Model saved to {self.model_path}")
        return meta

    def log_labelled_sample(self, emg_features: dict, motion: dict,
                            performance_label: str, session_id: str):
        """
        Append one labelled sample to the patient's session CSV.
        Called during live sessions when a rep completes.
        performance_label must be one of: poor / compensating / good
        This accumulates real training data automatically.
        """
        if performance_label not in PERFORMANCE_LABELS:
            return

        row = {**emg_features, **motion, "performance_label": performance_label,
               "session_id": session_id, "timestamp": time.time()}

        path = os.path.join(SESSION_DIR, f"{session_id}_features.csv")
        df   = pd.DataFrame([row])
        header = not os.path.exists(path)
        df.to_csv(path, mode="a", header=header, index=False)

    def should_retrain(self) -> bool:
        """
        Returns True if enough new real data has accumulated to justify retraining.
        Threshold: 200+ new labelled samples since last training.
        """
        if not os.path.exists(self.meta_path):
            return True
        with open(self.meta_path) as f:
            meta = json.load(f)
        last_trained = meta.get("n_real_samples", 0)
        current      = sum(
            len(pd.read_csv(os.path.join(SESSION_DIR, f)))
            for f in os.listdir(SESSION_DIR)
            if f.endswith("_features.csv")
        ) if os.path.exists(SESSION_DIR) else 0
        return (current - last_trained) >= 200


# ─── Predictor ────────────────────────────────────────────────────────────────

class PerformancePredictor:
    """Real-time inference — classifies one feature window."""

    def __init__(self):
        self._model = None
        self._scaler = None
        self._le = None
        self._loaded = False
        self._load()

    def _load(self):
        mp = os.path.join(MODEL_DIR, "performance_classifier.pkl")
        sp = os.path.join(MODEL_DIR, "scaler.pkl")
        lp = os.path.join(MODEL_DIR, "label_encoder.pkl")

        if os.path.exists(mp) and SKLEARN_AVAILABLE:
            try:
                self._model = joblib.load(mp)
                self._scaler = joblib.load(sp)
                self._le = joblib.load(lp)
                self._loaded = True
            except Exception as e:
                print(f"[predictor] Failed to load model: {e}")

    def predict(
        self,
        emg: EMGFeatures,
        arm_angle: float = 0.,
        rom: float = 0.,
        stability: float = 1.,
        velocity: float = 0.,
    ) -> dict:
        """
        Returns:
            {
                "label": "good" | "compensating" | "poor",
                "confidence": 0.0–1.0,
                "probabilities": {"poor": 0.x, "compensating": 0.x, "good": 0.x},
                "score": 0–100
            }
        """
        if not self._loaded:
            return self._heuristic_predict(emg, stability, rom)

        X = make_feature_vector(emg, arm_angle, rom, stability, velocity).reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        idx = self._model.predict(X_scaled)[0]
        probs = self._model.predict_proba(X_scaled)[0]
        label = self._le.inverse_transform([idx])[0]
        prob_dict = {c: float(p) for c, p in zip(self._le.classes_, probs)}

        score_map = {"poor": 20, "compensating": 55, "good": 85}
        confidence_bonus = int(float(np.max(probs)) * 15)
        score = int(np.clip(score_map[label] + confidence_bonus, 0, 100))

        return {
            "label": label,
            "confidence": float(np.max(probs)),
            "probabilities": prob_dict,
            "score": int(np.clip(score, 0, 100)),
        }

    def _heuristic_predict(self, emg: EMGFeatures, stability: float, rom: float) -> dict:
        """Fallback rule-based predictor when model isn't loaded."""
        score = (
            emg.rms * 30
            + emg.contraction_ratio * 30
            + stability * 20
            + min(rom / 60, 1.0) * 20
        ) * 100

        if score >= 65:
            label = "good"
        elif score >= 35:
            label = "compensating"
        else:
            label = "poor"

        return {
            "label": label,
            "confidence": 0.75,
            "probabilities": {"poor": 0.1, "compensating": 0.2, "good": 0.7},
            "score": int(np.clip(score, 0, 100)),
        }


# ─── Progress Tracker ─────────────────────────────────────────────────────────

class ProgressTracker:
    """Tracks recovery metrics across sessions."""

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self._path = os.path.join(SESSION_DIR, f"{patient_id}_progress.json")
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            with open(self._path) as f:
                return json.load(f)
        return {"patient_id": self.patient_id, "sessions": []}

    REQUIRED_KEYS = {
        "session_id": str,
        "game": dict,
        "emg": dict,
        "motion": dict,
    }

    def record_session(self, summary: dict):
        """Record a completed session summary."""
        missing = [k for k in self.REQUIRED_KEYS if k not in summary]
        if missing:
            print(f"[ProgressTracker] WARNING: missing keys {missing} — using defaults")

        game   = summary.get("game",   {})
        emg    = summary.get("emg",    {})
        motion = summary.get("motion", {})

        self._data["sessions"].append({
            "date":                   datetime.now().isoformat(),
            "session_id":             summary.get("session_id", "unknown"),
            "mean_score":             float(game.get("mean_score",             0)),
            "dominant_performance":   str(game.get("dominant_performance",    "unknown")),
            "mean_rms":               float(emg.get("mean_rms",               0)),
            "mean_rom":               float(motion.get("mean_rom",            0)),
            "mean_stability":         float(motion.get("mean_stability",      0)),
        })
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get_trend(self) -> dict:
        """Compute recovery trend metrics."""
        sessions = self._data["sessions"]
        if not sessions:
            return {}
        scores = [s["mean_score"] for s in sessions]
        roms = [s["mean_rom"] for s in sessions]
        return {
            "total_sessions": len(sessions),
            "score_trend": float(np.polyfit(range(len(scores)), scores, 1)[0]) if len(scores) > 1 else 0,
            "rom_trend": float(np.polyfit(range(len(roms)), roms, 1)[0]) if len(roms) > 1 else 0,
            "latest_score": scores[-1] if scores else 0,
            "best_score": max(scores) if scores else 0,
            "sessions": sessions[-10:],  # Last 10
        }


if __name__ == "__main__":
    print("Training performance classifier...")
    trainer = PerformanceTrainer()
    meta = trainer.train(use_real_data=False)
    print(json.dumps(meta, indent=2))

    print("\nTesting predictor...")
    pred = PerformancePredictor()
    feat = EMGFeatures(rms=0.45, mav=0.4, zc=30, ssc=25, wl=2.5,
                       var=0.02, mean_freq=170, median_freq=150,
                       peak_amp=0.85, contraction_ratio=0.65)
    result = pred.predict(feat, arm_angle=120, rom=55, stability=0.9, velocity=5)
    print(f"Prediction: {result}")
