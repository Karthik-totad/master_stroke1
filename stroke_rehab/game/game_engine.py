"""
game/game_engine.py

Shared game engine infrastructure:
  - Adaptive difficulty system
  - Scoring engine
  - EMG + motion signal routing to game controls
  - Session feedback overlay
"""

import time
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from data_acquisition.preprocessor import EMGFeatures
from vision.tracker import MotionFrame


# ─── Colours ──────────────────────────────────────────────────────────────────
C_BG        = (12, 15, 26)
C_PANEL     = (20, 25, 45)
C_ACCENT    = (0, 200, 170)
C_WARN      = (255, 160, 0)
C_DANGER    = (220, 60, 80)
C_GOOD      = (80, 220, 120)
C_TEXT      = (220, 230, 255)
C_TEXT_DIM  = (100, 110, 140)
C_WHITE     = (255, 255, 255)


@dataclass
class GameInputs:
    """
    Normalised game control inputs derived from EMG + motion.
    All values in 0.0–1.0 unless noted.
    """
    # From EMG
    grip_strength: float = 0.0        # How hard they're squeezing
    contraction: float = 0.0          # Binary: are muscles activated?
    pinch_strength: float = 0.0       # Pinch amplitude

    # From vision
    pinch_closed: bool = False         # Threshold pinch detection
    hand_open: bool = True
    wrist_angle_norm: float = 0.5     # 0=full left, 0.5=neutral, 1=full right
    arm_elevation: float = 0.5        # 0=down, 1=raised
    stability: float = 1.0
    rom: float = 0.0                  # Degrees

    # Combined
    performance_label: str = "compensating"
    performance_score: int = 50


@dataclass
class DifficultyConfig:
    """Parameters for each difficulty level."""
    level: str = "easy"
    target_speed: float = 1.0         # Relative speed multiplier
    target_size: float = 1.0          # Relative target size
    time_limit: float = 1.0           # Relative time pressure
    n_targets: int = 3

    @classmethod
    def from_label(cls, level: str) -> "DifficultyConfig":
        configs = {
            "easy":   cls(level="easy",   target_speed=0.5, target_size=1.4, time_limit=1.5, n_targets=3),
            "medium": cls(level="medium", target_speed=1.0, target_size=1.0, time_limit=1.0, n_targets=5),
            "hard":   cls(level="hard",   target_speed=1.8, target_size=0.7, time_limit=0.7, n_targets=7),
        }
        return configs.get(level, configs["easy"])


class ScoreEngine:
    """Tracks score, streak, and difficulty adaptation."""

    INCREASE_THRESHOLD = 80   # Score % → harder
    DECREASE_THRESHOLD = 40   # Score % → easier

    def __init__(self, initial_difficulty: str = "easy"):
        self.difficulty = initial_difficulty
        self.config = DifficultyConfig.from_label(initial_difficulty)
        self.score = 0
        self.max_score = 100
        self.streak = 0
        self.hits = 0
        self.misses = 0
        self._history: list[float] = []

    def hit(self, bonus: float = 1.0):
        self.hits += 1
        self.streak += 1
        points = int(10 * bonus * (1 + self.streak * 0.1))
        self.score = min(self.score + points, self.max_score * 5)
        self._history.append(1.0)
        self._adapt()

    def miss(self):
        self.misses += 1
        self.streak = 0
        self._history.append(0.0)
        self._adapt()

    def accuracy(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.5

    def _adapt(self):
        if len(self._history) < 10:
            return
        recent = sum(self._history[-10:]) / 10 * 100
        if recent >= self.INCREASE_THRESHOLD and self.difficulty != "hard":
            self._set_difficulty("hard" if self.difficulty == "medium" else "medium")
        elif recent <= self.DECREASE_THRESHOLD and self.difficulty != "easy":
            self._set_difficulty("easy" if self.difficulty == "medium" else "medium")

    def _set_difficulty(self, level: str):
        if level != self.difficulty:
            self.difficulty = level
            self.config = DifficultyConfig.from_label(level)
            print(f"[GameEngine] Difficulty → {level}")


# ─── HUD Renderer ─────────────────────────────────────────────────────────────

class HUDRenderer:
    """Renders a consistent HUD overlay across all games."""

    def __init__(self, screen_w: int, screen_h: int):
        self.w = screen_w
        self.h = screen_h
        self._font_lg = None
        self._font_md = None
        self._font_sm = None

    def _init_fonts(self):
        if self._font_lg is None and PYGAME_AVAILABLE:
            pygame.font.init()
            self._font_lg = pygame.font.SysFont("Arial", 28, bold=True)
            self._font_md = pygame.font.SysFont("Arial", 20)
            self._font_sm = pygame.font.SysFont("Arial", 15)

    def draw(self, surface, score_engine: ScoreEngine, inputs: GameInputs,
             game_name: str, time_remaining: Optional[float] = None):
        self._init_fonts()

        # Top bar
        pygame.draw.rect(surface, C_PANEL, (0, 0, self.w, 55))
        pygame.draw.line(surface, C_ACCENT, (0, 55), (self.w, 55), 1)

        # Game title
        title = self._font_lg.render(f"◈ {game_name.upper().replace('_', ' ')}", True, C_ACCENT)
        surface.blit(title, (20, 14))

        # Score
        score_surf = self._font_lg.render(f"Score: {score_engine.score}", True, C_WHITE)
        surface.blit(score_surf, (self.w // 2 - 60, 14))

        # Difficulty badge
        diff_colors = {"easy": C_GOOD, "medium": C_WARN, "hard": C_DANGER}
        dc = diff_colors.get(score_engine.difficulty, C_ACCENT)
        diff_surf = self._font_md.render(score_engine.difficulty.upper(), True, dc)
        pygame.draw.rect(surface, dc, (self.w - 140, 12, 110, 30), border_radius=6)
        pygame.draw.rect(surface, C_PANEL, (self.w - 138, 14, 106, 26), border_radius=5)
        surface.blit(diff_surf, (self.w - 128, 18))

        # Streak
        if score_engine.streak > 2:
            streak_surf = self._font_sm.render(f"🔥 x{score_engine.streak}", True, C_WARN)
            surface.blit(streak_surf, (self.w // 2 + 70, 20))

        # Timer
        if time_remaining is not None:
            frac = max(0, time_remaining / 60)
            bar_w = int(200 * frac)
            pygame.draw.rect(surface, C_PANEL, (self.w - 240, 62, 200, 8))
            color = C_GOOD if frac > 0.5 else (C_WARN if frac > 0.2 else C_DANGER)
            pygame.draw.rect(surface, color, (self.w - 240, 62, bar_w, 8))

        # EMG bar (bottom left)
        self._draw_emg_bar(surface, inputs)

        # Performance label
        label_colors = {"good": C_GOOD, "compensating": C_WARN, "poor": C_DANGER}
        lc = label_colors.get(inputs.performance_label, C_TEXT)
        perf_surf = self._font_sm.render(
            f"● {inputs.performance_label.upper()}", True, lc
        )
        surface.blit(perf_surf, (20, self.h - 30))

    def _draw_emg_bar(self, surface, inputs: GameInputs):
        # Vertical EMG bar
        bx, by, bw, bh = 20, self.h - 120, 18, 80
        pygame.draw.rect(surface, C_PANEL, (bx, by, bw, bh), border_radius=4)
        fill = int(bh * inputs.grip_strength)
        if fill > 0:
            color = C_GOOD if inputs.grip_strength > 0.4 else C_WARN
            pygame.draw.rect(surface, color, (bx, by + bh - fill, bw, fill),
                             border_radius=4)
        pygame.draw.rect(surface, C_ACCENT, (bx, by, bw, bh), 1, border_radius=4)
        emg_lbl = self._font_sm.render("EMG", True, C_TEXT_DIM)
        surface.blit(emg_lbl, (bx - 2, by - 16))

    def draw_feedback(self, surface, message: str, color=C_GOOD, duration: float = 1.0):
        """Animated feedback pop (call each frame, fades out)."""
        # Simple implementation — caller manages fade timer
        self._init_fonts()
        surf = self._font_lg.render(message, True, color)
        surface.blit(surf, (self.w // 2 - surf.get_width() // 2, self.h // 2 - 40))


# ─── Input Adapter ────────────────────────────────────────────────────────────

def adapt_inputs(
    emg: Optional[EMGFeatures],
    motion: Optional[MotionFrame],
    prediction: Optional[dict],
) -> GameInputs:
    """Convert raw EMG + motion data into normalised game inputs."""
    inp = GameInputs()

    if emg:
        inp.grip_strength = float(min(emg.rms * 2.5, 1.0))
        inp.contraction = float(emg.contraction_ratio)
        inp.pinch_strength = float(min(emg.mav * 3.0, 1.0))

    if motion:
        hand = motion.hand
        pose = motion.pose
        inp.pinch_closed = hand.is_pinching
        inp.hand_open = hand.is_open
        inp.stability = motion.stability
        inp.rom = motion.rom
        # Normalise wrist angle -90..+90 → 0..1
        inp.wrist_angle_norm = float((hand.wrist_angle + 90) / 180)
        inp.arm_elevation = float(min(pose.elevation / 90, 1.0)) if pose.detected else 0.5

    if prediction:
        inp.performance_label = prediction.get("label", "compensating")
        inp.performance_score = prediction.get("score", 50)

    return inp
