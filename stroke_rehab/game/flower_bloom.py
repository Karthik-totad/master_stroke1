"""
game/flower_bloom.py

Mini-game: Flower Bloom
Exercise: Hand open/close (finger extension)
Mechanic: A garden of flowers that open when the patient opens their hand,
          fold when they close it. Calmer, better suited for patients
          who need gentler encouragement.
Controlled by: EMG contraction ratio + vision hand open/close
"""

import math
import random
import time
from dataclasses import dataclass, field

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from game.game_engine import (
    GameInputs, ScoreEngine, HUDRenderer,
    C_BG, C_ACCENT, C_GOOD, C_WARN, C_DANGER, C_TEXT, C_WHITE, C_PANEL
)


@dataclass
class Flower:
    x: float
    y: float
    stem_h: float
    petal_count: int
    petal_color: tuple
    center_color: tuple
    bloom: float = 0.0        # 0 = closed, 1 = fully open
    sway: float = 0.0         # Gentle sway animation
    sway_speed: float = 0.8
    sway_phase: float = 0.0
    scale: float = 1.0
    scored: bool = False      # Did we award points this bloom cycle?

    PETAL_PALETTES = [
        ((255, 160, 200), (255, 220, 50)),
        ((180, 120, 255), (255, 200, 80)),
        ((100, 200, 255), (255, 255, 150)),
        ((255, 180, 100), (200, 255, 100)),
        ((200, 255, 180), (255, 200, 200)),
    ]

    @classmethod
    def create(cls, x: float, ground_y: float) -> "Flower":
        palette = random.choice(cls.PETAL_PALETTES)
        return cls(
            x=x, y=ground_y,
            stem_h=random.uniform(80, 160),
            petal_count=random.choice([5, 6, 7, 8]),
            petal_color=palette[0],
            center_color=palette[1],
            sway_speed=random.uniform(0.5, 1.2),
            sway_phase=random.uniform(0, math.tau),
            scale=random.uniform(0.8, 1.3),
        )

    def update(self, target_bloom: float, dt: float):
        # Smooth bloom animation
        speed = 2.5 if target_bloom > self.bloom else 1.5
        self.bloom += (target_bloom - self.bloom) * speed * dt
        self.bloom = max(0.0, min(1.0, self.bloom))
        # Sway
        self.sway_phase += self.sway_speed * dt
        self.sway = math.sin(self.sway_phase) * 6

    def draw(self, surface):
        if not PYGAME_AVAILABLE:
            return

        cx = int(self.x + self.sway)
        head_y = int(self.y - self.stem_h)

        # Stem
        stem_color = (60, 160, 60)
        stem_sway_x = int(self.x + self.sway * 0.4)
        pygame.draw.line(surface, stem_color,
                         (int(self.x), int(self.y)),
                         (cx, head_y + 5), 4)

        # Leaves
        for side in [-1, 1]:
            lx = int(self.x + side * 25 + self.sway * 0.3)
            ly = int(self.y - self.stem_h * 0.45)
            leaf_pts = [
                (int(self.x + self.sway * 0.2), ly),
                (lx, ly - 15),
                (lx + side * 10, ly + 5),
            ]
            pygame.draw.polygon(surface, stem_color, leaf_pts)

        # Petals (bloom animation)
        if self.bloom > 0.01:
            petal_r = int(18 * self.scale * self.bloom)
            spread = int(22 * self.scale * self.bloom)
            for i in range(self.petal_count):
                angle = (i / self.petal_count) * math.tau
                px = cx + int(math.cos(angle) * spread)
                py = head_y + int(math.sin(angle) * spread)
                if petal_r > 2:
                    pygame.draw.ellipse(surface, self.petal_color,
                                        (px - petal_r, py - petal_r,
                                         petal_r * 2, petal_r * 2))

        # Center
        center_r = max(3, int(10 * self.scale * (0.3 + self.bloom * 0.7)))
        pygame.draw.circle(surface, self.center_color, (cx, head_y), center_r)

        # Sparkle when fully bloomed
        if self.bloom > 0.95:
            t = time.time()
            for i in range(4):
                sa = i * math.pi / 2 + t * 2
                sx = cx + int(math.cos(sa) * (center_r + 8))
                sy = head_y + int(math.sin(sa) * (center_r + 8))
                pygame.draw.circle(surface, (255, 255, 200), (sx, sy), 2)


class FlowerBloomGame:
    """
    Flower Bloom — gentle hand open/close game.
    Opens flowers when hand opens, closes when hand closes.
    Calm, encouraging, suitable for severe/moderate patients.
    """

    N_FLOWERS = 7

    def __init__(self, screen_w: int = 1280, screen_h: int = 720,
                 initial_difficulty: str = "easy"):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.score_engine = ScoreEngine(initial_difficulty)
        self.hud = HUDRenderer(screen_w, screen_h)
        self.flowers: list[Flower] = []
        self._feedback: list[dict] = []
        self._bg_elements: list[dict] = []
        self.running = False
        self.start_time = 0.0
        self.session_duration = 60.0
        self._screen = None
        self._clock = None
        self._font = None
        self._target_bloom = 0.0
        self._prev_hand_open = False
        self._bloom_hold_timer = 0.0
        self._butterflies: list[dict] = []
        self._total_blooms = 0
        self._ambient_t = 0.0

    def _init(self):
        ground_y = self.screen_h - 80
        spacing = self.screen_w // (self.N_FLOWERS + 1)
        self.flowers = [
            Flower.create(spacing * (i + 1), ground_y)
            for i in range(self.N_FLOWERS)
        ]
        # Clouds
        self._bg_elements = [
            {"x": random.uniform(0, self.screen_w),
             "y": random.uniform(80, 220),
             "speed": random.uniform(15, 40),
             "r": random.randint(30, 70)}
            for _ in range(6)
        ]
        pygame.font.init()
        self._font = pygame.font.SysFont("Georgia", 22, bold=False)

    def start_pygame(self):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame not installed")
        pygame.init()
        self._screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("NeuroRehab — Flower Bloom")
        self._clock = pygame.time.Clock()
        self.start_time = time.time()
        self.running = True
        self._init()

    def update(self, inputs: GameInputs, dt: float):
        self._ambient_t += dt

        # Compute target bloom from EMG + vision
        # Hand open → bloom opens; hand close → bloom closes
        emg_open = inputs.contraction < 0.3   # Low contraction = relaxed/open
        vision_open = inputs.hand_open
        target = 1.0 if (emg_open or vision_open) else 0.0

        # Smooth blend (EMG weighted more)
        self._target_bloom += (target - self._target_bloom) * 3.0 * dt

        # Update each flower
        for f in self.flowers:
            f.update(self._target_bloom, dt)

            # Score when fully bloomed
            fully_bloomed = f.bloom > 0.92
            if fully_bloomed and not f.scored:
                f.scored = True
                self._total_blooms += 1
                self.score_engine.hit(bonus=inputs.stability)
                self._add_feedback(
                    random.choice(["Beautiful! 🌸", "Lovely! 🌺", "Wonderful! 🌼"]),
                    (255, 180, 220), f.x, f.y - f.stem_h - 40
                )
                self._spawn_butterfly(f.x, f.y - f.stem_h)
            elif not fully_bloomed and f.scored:
                f.scored = False  # Reset for next bloom cycle

        # Update butterflies
        for b in self._butterflies:
            b["x"] += math.cos(b["phase"]) * 60 * dt
            b["y"] += math.sin(b["phase"] * 1.3) * 30 * dt - 10 * dt
            b["phase"] += dt * 2
            b["ttl"] -= dt
            b["wing_phase"] += dt * 8
        self._butterflies = [b for b in self._butterflies if b["ttl"] > 0 and b["y"] > 60]

        # Clouds drift
        for c in self._bg_elements:
            c["x"] = (c["x"] + c["speed"] * dt) % (self.screen_w + 200)

        # Feedback timers
        for f in self._feedback:
            f["ttl"] -= dt
            f["y"] -= 25 * dt
        self._feedback = [f for f in self._feedback if f["ttl"] > 0]

    def render(self, surface, inputs: GameInputs):
        self._draw_sky(surface)
        self._draw_clouds(surface)
        self._draw_ground(surface)
        for flower in self.flowers:
            flower.draw(surface)
        self._draw_butterflies(surface)
        self._draw_hand_guide(surface, inputs)
        self._draw_feedback_texts(surface)
        self._draw_bloom_counter(surface)
        time_remaining = self.session_duration - (time.time() - self.start_time)
        self.hud.draw(surface, self.score_engine, inputs, "Flower Bloom", time_remaining)

    def _draw_sky(self, surface):
        # Gradient sky
        sky_top = (20, 40, 80)
        sky_bot = (120, 190, 240)
        h = self.screen_h
        for y in range(h):
            frac = y / h
            r = int(sky_top[0] + (sky_bot[0] - sky_top[0]) * frac)
            g = int(sky_top[1] + (sky_bot[1] - sky_top[1]) * frac)
            b = int(sky_top[2] + (sky_bot[2] - sky_top[2]) * frac)
            pygame.draw.line(surface, (r, g, b), (0, y), (self.screen_w, y))

    def _draw_clouds(self, surface):
        for c in self._bg_elements:
            for dr, dg, db, dx, dy in [
                (240, 248, 255, 0, 0),
                (220, 235, 245, -20, 10),
                (230, 242, 250, 20, 5),
            ]:
                pygame.draw.circle(surface, (dr, dg, db),
                                   (int(c["x"] + dx), int(c["y"] + dy)), c["r"])

    def _draw_ground(self, surface):
        gy = self.screen_h - 80
        pygame.draw.rect(surface, (50, 120, 50), (0, gy, self.screen_w, 80))
        # Grass blades
        for x in range(0, self.screen_w, 12):
            h = random.randint(5, 18)
            sway = int(3 * math.sin(self._ambient_t + x * 0.1))
            pygame.draw.line(surface, (70, 150, 60),
                             (x, gy), (x + sway, gy - h), 2)

    def _draw_butterflies(self, surface):
        for b in self._butterflies:
            wing_open = abs(math.sin(b["wing_phase"]))
            w = int(15 * wing_open)
            h_b = 10
            x, y = int(b["x"]), int(b["y"])
            color = b["color"]
            if w > 1:
                pygame.draw.ellipse(surface, color, (x - w, y - h_b // 2, w, h_b))
                pygame.draw.ellipse(surface, color, (x, y - h_b // 2, w, h_b))
            pygame.draw.circle(surface, (50, 30, 10), (x, y), 3)

    def _draw_hand_guide(self, surface, inputs: GameInputs):
        """Show a visual prompt for hand state."""
        gx, gy = self.screen_w - 180, self.screen_h - 160
        open_pct = self._target_bloom

        # Guide circle
        pygame.draw.circle(surface, (30, 40, 60), (gx, gy), 55)
        pygame.draw.circle(surface, C_ACCENT, (gx, gy), 55, 2)

        # Finger arcs
        for i in range(5):
            angle = math.pi * 1.1 + (i / 4) * math.pi * 0.8
            length = 30 + 15 * open_pct
            ex = gx + int(math.cos(angle) * length)
            ey = gy + int(math.sin(angle) * length)
            color = C_GOOD if open_pct > 0.6 else C_WARN
            pygame.draw.line(surface, color, (gx, gy), (ex, ey), 4)

        label = self._font.render("OPEN ✋" if inputs.hand_open else "CLOSE ✊",
                                  True, C_GOOD if inputs.hand_open else C_ACCENT)
        surface.blit(label, (gx - label.get_width() // 2, gy + 65))

    def _draw_bloom_counter(self, surface):
        text = self._font.render(f"🌸 Blooms: {self._total_blooms}", True, (255, 200, 220))
        surface.blit(text, (self.screen_w // 2 - text.get_width() // 2, 65))

    def _draw_feedback_texts(self, surface):
        font = pygame.font.SysFont("Georgia", 24, italic=True)
        for f in self._feedback:
            alpha = int(255 * min(f["ttl"], 1.0))
            surf = font.render(f["text"], True, f["color"])
            surf.set_alpha(alpha)
            surface.blit(surf, (int(f["x"]) - surf.get_width() // 2, int(f["y"])))

    def _add_feedback(self, text: str, color: tuple, x: float, y: float):
        self._feedback.append({"text": text, "color": color, "x": x, "y": y, "ttl": 2.0})

    def _spawn_butterfly(self, x: float, y: float):
        colors = [(255, 150, 200), (200, 150, 255), (255, 220, 100), (150, 220, 255)]
        self._butterflies.append({
            "x": x, "y": y,
            "phase": random.uniform(0, math.tau),
            "wing_phase": 0.0,
            "color": random.choice(colors),
            "ttl": 5.0,
        })

    def run_loop(self, input_callback=None):
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
        keys = pygame.key.get_pressed()
        space = keys[pygame.K_SPACE]
        return GameInputs(
            grip_strength=0.1 if space else 0.6,
            contraction=0.1 if space else 0.6,
            hand_open=space,
            pinch_closed=False,
            stability=0.9,
            rom=35.0,
            performance_label="good",
        )


if __name__ == "__main__":
    game = FlowerBloomGame()
    score = game.run_loop()
    print(f"Final score: {score}")
