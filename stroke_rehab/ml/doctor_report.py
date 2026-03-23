"""
ml/doctor_report.py

Parses doctor reports (JSON) and converts them into structured therapy plans.
Adjusts game difficulty and exercise sequence based on patient condition.
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import REPORT_DIR


@dataclass
class Exercise:
    game: str               # bubble_pop | maze_steering | pump_the_pump | flower_bloom
    exercise_type: str      # grip_strength | finger_extension | wrist_rotation | pinch
    sets: int = 3
    reps: int = 10
    difficulty: str = "easy"   # easy | medium | hard
    notes: str = ""
    duration_seconds: Optional[int] = None


@dataclass
class TherapyPlan:
    patient_id: str
    condition: str
    affected_side: str
    severity: str               # mild | moderate | severe
    exercises: list[Exercise] = field(default_factory=list)
    session_duration_minutes: int = 20
    target_rom_degrees: float = 45.0
    contraindications: list[str] = field(default_factory=list)
    doctor_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "condition": self.condition,
            "affected_side": self.affected_side,
            "severity": self.severity,
            "session_duration_minutes": self.session_duration_minutes,
            "target_rom_degrees": self.target_rom_degrees,
            "contraindications": self.contraindications,
            "doctor_notes": self.doctor_notes,
            "exercises": [
                {
                    "game": e.game,
                    "type": e.exercise_type,
                    "sets": e.sets,
                    "reps": e.reps,
                    "difficulty": e.difficulty,
                    "notes": e.notes,
                }
                for e in self.exercises
            ],
        }

    def get_game_sequence(self) -> list[str]:
        """Ordered list of games to play this session."""
        return [e.game for e in self.exercises]

    def adjust_difficulty(self, performance_label: str) -> None:
        """
        Dynamically adjust exercise difficulty based on ML performance label.
        Mutates in-place; returns None.
        """
        difficulty_map = {
            "poor":         "easy",
            "compensating": "medium",
            "good":         "hard",
        }
        target = difficulty_map.get(performance_label)
        if target is None:
            print(f"[TherapyPlan] Unknown performance label '{performance_label}' — no change")
            return
        changed = 0
        for ex in self.exercises:
            if ex.difficulty != target:
                ex.difficulty = target
                changed += 1
        if changed:
            print(f"[TherapyPlan] Adjusted {changed} exercise(s) to difficulty='{target}' "
                  f"based on performance='{performance_label}'")


# ─── Default Game Mapping ─────────────────────────────────────────────────────

EXERCISE_TO_GAME = {
    "grip_strength": "pump_the_pump",
    "finger_extension": "flower_bloom",
    "wrist_rotation": "maze_steering",
    "pinch": "bubble_pop",
    "pinch_coordination": "bubble_pop",
    "general": "bubble_pop",
}

SEVERITY_DEFAULTS = {
    "mild": dict(sets=3, reps=12, difficulty="medium"),
    "moderate": dict(sets=3, reps=8, difficulty="easy"),
    "severe": dict(sets=2, reps=5, difficulty="easy"),
}


# ─── Parser ───────────────────────────────────────────────────────────────────

class DoctorReportParser:
    """Parses a doctor JSON report into a TherapyPlan."""

    def parse_file(self, path: str) -> TherapyPlan:
        with open(path) as f:
            data = json.load(f)
        return self.parse_dict(data)

    def parse_dict(self, data: dict) -> TherapyPlan:
        patient_id = data.get("patient_id", "UNKNOWN")
        condition = data.get("condition", "stroke")
        affected_side = data.get("affected_side", "right")
        severity = data.get("severity", "moderate")
        target_rom = data.get("target_rom_degrees", 45.0)
        session_dur = data.get("session_duration_minutes", 20)
        contraindications = data.get("contraindications", [])
        doctor_notes = data.get("notes", "")

        defaults = SEVERITY_DEFAULTS.get(severity, SEVERITY_DEFAULTS["moderate"])

        exercises = []
        for ex in data.get("prescribed_exercises", []):
            ex_type = ex.get("type", "general")
            game = ex.get("game") or EXERCISE_TO_GAME.get(ex_type, "bubble_pop")
            exercises.append(Exercise(
                game=game,
                exercise_type=ex_type,
                sets=ex.get("sets", defaults["sets"]),
                reps=ex.get("reps", defaults["reps"]),
                difficulty=ex.get("difficulty", defaults["difficulty"]),
                notes=ex.get("notes", ""),
                duration_seconds=ex.get("duration_seconds"),
            ))

        # If no exercises prescribed, create defaults based on severity
        if not exercises:
            exercises = self._default_exercises(severity)

        return TherapyPlan(
            patient_id=patient_id,
            condition=condition,
            affected_side=affected_side,
            severity=severity,
            exercises=exercises,
            session_duration_minutes=session_dur,
            target_rom_degrees=target_rom,
            contraindications=contraindications,
            doctor_notes=doctor_notes,
        )

    def _default_exercises(self, severity: str) -> list[Exercise]:
        defaults = SEVERITY_DEFAULTS.get(severity, SEVERITY_DEFAULTS["moderate"])
        return [
            Exercise(
                game="flower_bloom",
                exercise_type="finger_extension",
                difficulty="easy",
                **{k: v for k, v in defaults.items() if k != "difficulty"},
            ),
            Exercise(
                game="bubble_pop",
                exercise_type="pinch",
                difficulty=defaults["difficulty"],
                **{k: v for k, v in defaults.items() if k != "difficulty"},
            ),
        ]

    def load_from_directory(self, directory: str = REPORT_DIR) -> dict[str, TherapyPlan]:
        """Load all report files from a directory."""
        plans = {}
        if not os.path.exists(directory):
            return plans
        for fname in os.listdir(directory):
            if fname.endswith(".json"):
                try:
                    plan = self.parse_file(os.path.join(directory, fname))
                    plans[plan.patient_id] = plan
                except Exception as e:
                    print(f"[DoctorReport] Failed to parse {fname}: {e}")
        return plans


# ─── Sample report ────────────────────────────────────────────────────────────

SAMPLE_REPORT = {
    "patient_id": "PT001",
    "date": "2026-03-21",
    "condition": "right_hemiplegia",
    "affected_side": "right",
    "severity": "moderate",
    "prescribed_exercises": [
        {
            "type": "grip_strength",
            "game": "pump_the_pump",
            "sets": 3,
            "reps": 10,
            "difficulty": "easy",
            "notes": "Avoid overextension"
        },
        {
            "type": "finger_extension",
            "game": "flower_bloom",
            "sets": 2,
            "reps": 15,
            "difficulty": "easy",
            "notes": "Focus on full extension"
        },
        {
            "type": "pinch_coordination",
            "game": "bubble_pop",
            "sets": 3,
            "reps": 20,
            "difficulty": "easy",
            "notes": "Pinch coordination"
        },
        {
            "type": "wrist_rotation",
            "game": "maze_steering",
            "sets": 2,
            "reps": 10,
            "difficulty": "easy",
            "notes": "Gentle wrist rotation only"
        }
    ],
    "contraindications": ["avoid_high_resistance"],
    "target_rom_degrees": 45,
    "session_duration_minutes": 20,
    "notes": "Patient has moderate spasticity. Start all exercises at easy level."
}


def create_sample_report():
    """Write sample report to disk."""
    path = os.path.join(REPORT_DIR, "PT001_report.json")
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(SAMPLE_REPORT, f, indent=2)
    print(f"[DoctorReport] Sample report written to {path}")
    return path


if __name__ == "__main__":
    create_sample_report()
    parser = DoctorReportParser()
    plan = parser.parse_dict(SAMPLE_REPORT)
    print(json.dumps(plan.to_dict(), indent=2))
    print(f"\nGame sequence: {plan.get_game_sequence()}")
