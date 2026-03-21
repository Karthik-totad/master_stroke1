"""
game/bubble_pop.py

Mini-game: Bubble Pop
Exercise: Pinch coordination
Mechanic: Pinch floating bubbles before they escape off-screen.
          Bubble size/speed adapts to difficulty.
          Controlled by: EMG pinch strength + vision pinch detection.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from game.game_engine import (
    GameInputs, ScoreEngine, HUDRenderer, DifficultyConfig,
    C_BG, C_ACCENT, C_GOOD, C_WARN, C_DANGER, C_TEXT, C_WHITE, C_PANEL
)


@dataclass
class Bubble:
    x: float
    y: float
    radius: float
    color: tuple
    vx: float
    vy: float
    wobble_phase: float = 0.0
    alive: bool = True
    age: float = 0.0
    max_age: float = 5.0
    popped: bool = False
    pop_anim: float = 0.0     # 0 = not popping, 1 = fully popped

    COLORS = [
        (80, 200, 255), (100, 255, 180), (255, 160, 100),
        (200, 100, 255), (255, 220, 80), (100, 220, 255),
    ]

    @classmethod
    def spawn(cls, screen_w: int, screen_h: int, config: DifficultyConfig) -> "Bubble":
        base_r = int(50 * config.target_size)
        r = random.randint(int(base_r * 0.7), int(base_r * 1.3))
        return cls(
            x=float(random.randint(r, screen_w - r)),
            y=float(screen_h + r),
            radius=float(r),
            color=random.choice(cls.COLORS),
            vx=random.uniform(-1.5, 1.5) * config.target_speed,
            vy=-random.uniform(1.5, 3.5) * config.target_speed,
            wobble_phase=random.uniform(0, math.tau),
            max_age=6.0 / config.target_speed,
        )


class BubblePopGame:
    """
    Bubble Pop — pinch coordination game.
    """

    def __init__(self, screen_w: int = 1280, screen_h: int = 720,
                 initial_difficulty: str = "easy"):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.score_engine = ScoreEngine(initial_difficulty)
        self.hud = HUDRenderer(screen_w, screen_h)
        self.bubbles: list[Bubble] = []
        self._spawn_timer = 0.0
        self._feedback: list[dict] = []
        self._cursor_x = screen_w // 2
        self._cursor_y = screen_h // 2
        self._was_pinching = False
        self._particles: list[dict] = []
        self.running = False
        self.start_time = 0.0
        self.session_duration = 60.0  # seconds per session
        self._screen = None
        self._clock = None

    def start_pygame(self):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame not installed. Run: pip install pygame")
        pygame.init()
        self._screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("NeuroRehab — Bubble Pop")
        self._clock = pygame.time.Clock()
        self.start_time = time.time()
        self.running = True

    def update(self, inputs: GameInputs, dt: float):
        """Update game state given current inputs and delta time."""
        config = self.score_engine.config

        # Spawn bubbles
        spawn_rate = 1.5 / config.n_targets
        self._spawn_timer += dt
        if self._spawn_timer >= spawn_rate and len(self.bubbles) < config.n_targets:
            self.bubbles.append(Bubble.spawn(self.screen_w, self.screen_h, config))
            self._spawn_timer = 0.0

        # Update cursor from inputs
        # Wrist angle steers X; arm elevation steers Y (normalized 0–1)
        self._cursor_x = int(inputs.wrist_angle_norm * self.screen_w)
        self._cursor_y = int((1.0 - inputs.arm_elevation) * (self.screen_h - 80) + 80)

        # Pinch detection
        is_pinching = inputs.pinch_closed or inputs.pinch_strength > 0.55
        just_pinched = is_pinching and not self._was_pinching
        self._was_pinching = is_pinching

        # Update bubbles
        for bub in self.bubbles:
            if bub.popped:
                bub.pop_anim += dt * 4
                if bub.pop_anim >= 1.0:
                    bub.alive = False
                continue

            bub.age += dt
            bub.x += bub.vx + math.sin(bub.wobble_phase + bub.age * 2) * 0.5
            bub.y += bub.vy
            bub.wobble_phase += dt * 0.5

            # Bounce off walls
            if bub.x - bub.radius < 0:
                bub.x = bub.radius; bub.vx = abs(bub.vx)
            if bub.x + bub.radius > self.screen_w:
                bub.x = self.screen_w - bub.radius; bub.vx = -abs(bub.vx)

            # Escape off top
            if bub.y + bub.radius < 0 or bub.age > bub.max_age:
                bub.alive = False
                self.score_engine.miss()
                self._add_feedback("MISSED!", C_DANGER, bub.x, bub.y + 100)
                continue

            # Hit detection
            if just_pinched:
                dist = math.hypot(self._cursor_x - bub.x, self._cursor_y - bub.y)
                if dist < bub.radius + 25:
                    bub.popped = True
                    quality = max(0.1, 1.0 - dist / (bub.radius + 25))
                    self.score_engine.hit(bonus=quality)
                    self._add_feedback(
                        ["POP! 💥", "NICE! ✨", "GREAT! ⚡"][self.score_engine.streak % 3],
                        C_GOOD, bub.x, bub.y
                    )
                    self._spawn_particles(bub.x, bub.y, bub.color)

        # Clean dead bubbles
        self.bubbles = [b for b in self.bubbles if b.alive]

        # Update feedback timers
        self._feedback = [f for f in self._feedback if f["ttl"] > 0]
        for f in self._feedback:
            f["ttl"] -= dt
            f["y"] -= 40 * dt

        # Update particles
        for p in self._particles:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["vy"] += 200 * dt  # gravity
            p["ttl"] -= dt
        self._particles = [p for p in self._particles if p["ttl"] > 0]

    def render(self, surface, inputs: GameInputs):
        """Render one frame."""
        self._draw_background(surface)
        self._draw_bubbles(surface)
        self._draw_particles(surface)
        self._draw_cursor(surface, inputs)
        self._draw_feedback_texts(surface)
        time_remaining = self.session_duration - (time.time() - self.start_time)
        self.hud.draw(surface, self.score_engine, inputs, "Bubble Pop", time_remaining)

    def _draw_background(self, surface):
        surface.fill(C_BG)
        # Subtle grid
        for x in range(0, self.screen_w, 80):
            pygame.draw.line(surface, (20, 25, 45), (x, 55), (x, self.screen_h), 1)
        for y in range(55, self.screen_h, 80):
            pygame.draw.line(surface, (20, 25, 45), (0, y), (self.screen_w, y), 1)

    def _draw_bubbles(self, surface):
        for bub in self.bubbles:
            if bub.popped:
                # Expand + fade
                r = int(bub.radius * (1 + bub.pop_anim * 1.5))
                alpha = int(255 * (1 - bub.pop_anim))
                s = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(s, (*bub.color, alpha), (r + 2, r + 2), r, 4)
                surface.blit(s, (int(bub.x) - r - 2, int(bub.y) - r - 2))
            else:
                # Wobble radius
                wr = int(bub.radius + 3 * math.sin(bub.wobble_phase * 3))
                # Outer glow
                glow = pygame.Surface((wr * 2 + 20, wr * 2 + 20), pygame.SRCALPHA)
                pygame.draw.circle(glow, (*bub.color, 30), (wr + 10, wr + 10), wr + 10)
                surface.blit(glow, (int(bub.x) - wr - 10, int(bub.y) - wr - 10))
                # Main bubble
                pygame.draw.circle(surface, bub.color, (int(bub.x), int(bub.y)), wr, 3)
                # Highlight
                hx = int(bub.x - wr * 0.3)
                hy = int(bub.y - wr * 0.3)
                pygame.draw.circle(surface, (255, 255, 255), (hx, hy), max(2, wr // 5))
                # Age timer ring
                age_frac = 1 - (bub.age / bub.max_age)
                color = C_GOOD if age_frac > 0.5 else (C_WARN if age_frac > 0.25 else C_DANGER)
                arc_rect = pygame.Rect(int(bub.x) - wr, int(bub.y) - wr, wr * 2, wr * 2)
                if age_frac > 0:
                    pygame.draw.arc(surface, color, arc_rect,
                                    0, math.tau * age_frac, 3)

    def _draw_cursor(self, surface, inputs: GameInputs):
        x, y = self._cursor_x, self._cursor_y
        pinching = inputs.pinch_closed or inputs.pinch_strength > 0.55
        color = C_GOOD if pinching else C_ACCENT
        r = 20 if pinching else 28
        pygame.draw.circle(surface, color, (x, y), r, 3)
        # Inner dot
        pygame.draw.circle(surface, color, (x, y), 5 if pinching else 3)
        # Crosshairs
        pygame.draw.line(surface, color, (x - 40, y), (x - r - 5, y), 1)
        pygame.draw.line(surface, color, (x + r + 5, y), (x + 40, y), 1)
        pygame.draw.line(surface, color, (x, y - 40), (x, y - r - 5), 1)
        pygame.draw.line(surface, color, (x, y + r + 5), (x, y + 40), 1)

    def _draw_particles(self, surface):
        for p in self._particles:
            alpha = int(255 * p["ttl"] / 0.6)
            s = pygame.Surface((8, 8), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p["color"], alpha), (4, 4), 4)
            surface.blit(s, (int(p["x"]) - 4, int(p["y"]) - 4))

    def _draw_feedback_texts(self, surface):
        font = pygame.font.SysFont("Arial", 24, bold=True)
        for f in self._feedback:
            alpha = int(255 * min(f["ttl"], 1.0))
            surf = font.render(f["text"], True, f["color"])
            surf.set_alpha(alpha)
            surface.blit(surf, (int(f["x"]) - surf.get_width() // 2, int(f["y"])))

    def _add_feedback(self, text: str, color: tuple, x: float, y: float):
        self._feedback.append({"text": text, "color": color, "x": x, "y": y, "ttl": 1.2})

    def _spawn_particles(self, x: float, y: float, color: tuple):
        import random
        for _ in range(12):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(80, 250)
            self._particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed - 100,
                "color": color,
                "ttl": random.uniform(0.3, 0.7),
            })

    def run_loop(self, input_callback: Optional[callable] = None):
        """
        Main game loop. input_callback() → GameInputs (called each frame).
        If None, uses demo/keyboard inputs.
        """
        self.start_pygame()
        prev_time = time.time()

        while self.running:
            now = time.time()
            dt = min(now - prev_time, 0.05)
            prev_time = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False

            # Get inputs
            if input_callback:
                inputs = input_callback()
            else:
                inputs = self._keyboard_demo_inputs()

            self.update(inputs, dt)
            self.render(self._screen, inputs)
            pygame.display.flip()
            self._clock.tick(60)

            if time.time() - self.start_time > self.session_duration:
                self.running = False

        pygame.quit()
        return self.score_engine.score

    def _keyboard_demo_inputs(self) -> GameInputs:
        """Keyboard fallback for testing without EMG/camera."""
        keys = pygame.key.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_btn = pygame.mouse.get_pressed()[0]
        from game.game_engine import GameInputs
        return GameInputs(
            grip_strength=0.7 if mouse_btn else 0.1,
            pinch_strength=0.8 if mouse_btn else 0.1,
            pinch_closed=mouse_btn,
            hand_open=not mouse_btn,
            wrist_angle_norm=mouse_x / self.screen_w,
            arm_elevation=1.0 - mouse_y / self.screen_h,
            stability=0.9,
            rom=45.0,
            performance_label="good",
        )


if __name__ == "__main__":
    game = BubblePopGame()
    score = game.run_loop()
    print(f"Final score: {score}")
