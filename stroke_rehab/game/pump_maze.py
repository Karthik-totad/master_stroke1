"""
game/pump_the_pump.py

Mini-game: Pump the Pump
Exercise: Grip squeeze / release (grip strength)
Mechanic: Squeeze hand to charge a power meter, release to fire water cannon at targets.
Controlled by: EMG RMS + vision grip aperture
"""

import math, random, time
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
class Target:
    x: float
    y: float
    radius: float
    color: tuple
    hit: bool = False
    hit_anim: float = 0.0
    vx: float = 0.0
    points: int = 10
    COLORS = [(255, 80, 60), (255, 160, 40), (220, 60, 200), (80, 200, 255)]

    @classmethod
    def spawn(cls, w: int, h: int, speed: float) -> "Target":
        side = random.choice([-1, 1])
        return cls(
            x=float(random.randint(100, w - 100)),
            y=float(random.randint(100, h // 2)),
            radius=float(random.randint(25, 50)),
            color=random.choice(cls.COLORS),
            vx=side * speed * random.uniform(30, 80),
            points=max(5, int(800 / (random.randint(25, 50) + 1))),
        )


@dataclass
class WaterStream:
    x: float
    y: float
    power: float
    active: bool = True
    particles: list = field(default_factory=list)

    def spawn_particles(self):
        for _ in range(int(self.power * 8)):
            angle = -math.pi / 2 + random.uniform(-0.3, 0.3)
            speed = self.power * random.uniform(300, 600)
            self.particles.append({
                "x": self.x, "y": self.y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "ttl": random.uniform(0.3, 0.8),
                "r": random.randint(3, 8),
            })


class PumpThePumpGame:
    def __init__(self, screen_w=1280, screen_h=720, initial_difficulty="easy"):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.score_engine = ScoreEngine(initial_difficulty)
        self.hud = HUDRenderer(screen_w, screen_h)
        self.targets: list[Target] = []
        self._streams: list[WaterStream] = []
        self._charge = 0.0       # 0–1, builds while squeezing
        self._was_contracted = False
        self._feedback: list[dict] = []
        self.running = False
        self.start_time = 0.0
        self.session_duration = 60.0
        self._screen = None
        self._clock = None
        self._cannon_y = screen_h - 60
        self._spawn_timer = 0.0

    def start_pygame(self):
        pygame.init()
        self._screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("NeuroRehab — Pump the Pump")
        self._clock = pygame.time.Clock()
        self.start_time = time.time()
        self.running = True

    def update(self, inputs: GameInputs, dt: float):
        config = self.score_engine.config
        # Spawn targets
        self._spawn_timer += dt
        if self._spawn_timer > 1.5 / config.n_targets and len(self.targets) < config.n_targets:
            self.targets.append(Target.spawn(self.screen_w, self.screen_h, config.target_speed))
            self._spawn_timer = 0.0

        # Update targets
        for t in self.targets:
            if not t.hit:
                t.x += t.vx * dt
                if t.x < t.radius or t.x > self.screen_w - t.radius:
                    t.vx *= -1
            else:
                t.hit_anim += dt * 3
                if t.hit_anim >= 1.0:
                    t.hit = True  # keep for removal

        # Charge / fire logic
        contracted = inputs.grip_strength > 0.35 or inputs.contraction > 0.4
        if contracted:
            self._charge = min(1.0, self._charge + dt * 1.2)
        elif self._was_contracted and self._charge > 0.1:
            # Fire!
            stream = WaterStream(x=self.screen_w // 2, y=self._cannon_y, power=self._charge)
            stream.spawn_particles()
            self._check_hits(stream)
            self._streams.append(stream)
            self._charge = 0.0
        elif not contracted:
            self._charge = max(0.0, self._charge - dt * 0.5)
        self._was_contracted = contracted

        # Update stream particles
        for s in self._streams:
            for p in s.particles:
                p["x"] += p["vx"] * dt
                p["y"] += p["vy"] * dt
                p["vy"] += 400 * dt  # gravity
                p["ttl"] -= dt
            s.particles = [p for p in s.particles if p["ttl"] > 0]
        self._streams = [s for s in self._streams if s.particles]

        # Remove dead targets
        self.targets = [t for t in self.targets if not (t.hit and t.hit_anim >= 1.0)]

        # Feedback
        for f in self._feedback:
            f["ttl"] -= dt
            f["y"] -= 30 * dt
        self._feedback = [f for f in self._feedback if f["ttl"] > 0]

    def _check_hits(self, stream: WaterStream):
        cx = self.screen_w // 2
        for t in self.targets:
            if t.hit:
                continue
            # Check if any particle hits target
            for _ in range(20):
                angle = -math.pi / 2 + random.uniform(-0.3 * stream.power, 0.3 * stream.power)
                speed = stream.power * 500
                px = cx + math.cos(angle) * speed * 0.5
                py = self._cannon_y + math.sin(angle) * speed * 0.5
                if math.hypot(px - t.x, py - t.y) < t.radius + 20:
                    t.hit = True
                    self.score_engine.hit(bonus=stream.power)
                    self._feedback.append({
                        "text": f"💦 SPLASH! +{t.points}",
                        "color": (100, 180, 255), "x": t.x, "y": t.y, "ttl": 1.2
                    })
                    break

    def render(self, surface, inputs: GameInputs):
        surface.fill((15, 20, 40))
        # Background water shimmer
        t = time.time()
        for i in range(0, self.screen_w, 40):
            y = self.screen_h - 50 + int(3 * math.sin(t * 2 + i * 0.1))
            pygame.draw.line(surface, (20, 50, 100), (i, y), (i + 35, y), 2)

        # Targets
        for tgt in self.targets:
            if tgt.hit:
                r = int(tgt.radius * (1 + tgt.hit_anim))
                alpha = int(255 * (1 - tgt.hit_anim))
                s = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(s, (*tgt.color, alpha), (r + 2, r + 2), r)
                surface.blit(s, (int(tgt.x) - r - 2, int(tgt.y) - r - 2))
            else:
                pygame.draw.circle(surface, tgt.color, (int(tgt.x), int(tgt.y)), int(tgt.radius))
                pygame.draw.circle(surface, C_WHITE, (int(tgt.x), int(tgt.y)), int(tgt.radius), 2)
                # Rings
                for ring in [0.6, 0.35]:
                    pygame.draw.circle(surface, C_WHITE,
                                       (int(tgt.x), int(tgt.y)), int(tgt.radius * ring), 1)

        # Water particles
        for s in self._streams:
            for p in s.particles:
                alpha = int(200 * p["ttl"] / 0.8)
                col = (100, 180, 255)
                circ_s = pygame.Surface((p["r"] * 2, p["r"] * 2), pygame.SRCALPHA)
                pygame.draw.circle(circ_s, (*col, min(255, alpha)), (p["r"], p["r"]), p["r"])
                surface.blit(circ_s, (int(p["x"]) - p["r"], int(p["y"]) - p["r"]))

        # Cannon
        cx = self.screen_w // 2
        pygame.draw.rect(surface, (80, 90, 110), (cx - 20, self._cannon_y - 20, 40, 40),
                         border_radius=6)
        pygame.draw.rect(surface, (120, 140, 160), (cx - 8, self._cannon_y - 50, 16, 35),
                         border_radius=4)

        # Charge meter
        meter_x, meter_y = cx - 80, self._cannon_y - 80
        pygame.draw.rect(surface, C_PANEL, (meter_x, meter_y, 160, 18), border_radius=6)
        fill_w = int(160 * self._charge)
        color = C_GOOD if self._charge < 0.6 else (C_WARN if self._charge < 0.85 else C_DANGER)
        if fill_w > 0:
            pygame.draw.rect(surface, color, (meter_x, meter_y, fill_w, 18), border_radius=6)
        pygame.draw.rect(surface, C_ACCENT, (meter_x, meter_y, 160, 18), 1, border_radius=6)
        font = pygame.font.SysFont("Arial", 14)
        lbl = font.render("CHARGE", True, C_TEXT)
        surface.blit(lbl, (meter_x + 55, meter_y - 18))

        # Feedback
        font2 = pygame.font.SysFont("Arial", 22, bold=True)
        for f in self._feedback:
            surf = font2.render(f["text"], True, f["color"])
            surf.set_alpha(int(255 * min(f["ttl"], 1.0)))
            surface.blit(surf, (int(f["x"]) - surf.get_width() // 2, int(f["y"])))

        time_remaining = self.session_duration - (time.time() - self.start_time)
        self.hud.draw(surface, self.score_engine, inputs, "Pump the Pump", time_remaining)

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
                keys = pygame.key.get_pressed()
                space = keys[pygame.K_SPACE]
                inputs = GameInputs(grip_strength=0.8 if space else 0.05,
                                    contraction=0.8 if space else 0.05,
                                    stability=0.9, rom=40.0, performance_label="good")
            self.update(inputs, dt)
            self.render(self._screen, inputs)
            pygame.display.flip()
            self._clock.tick(60)
            if time.time() - self.start_time > self.session_duration:
                self.running = False
        pygame.quit()
        return self.score_engine.score


# ─── Maze Steering ────────────────────────────────────────────────────────────

class MazeSteeringGame:
    """
    Maze Steering — wrist rotation game.
    Rotate wrist like joystick to guide marble through maze.
    Controlled by: EMG wrist signal + vision wrist angle.
    """

    MAZES = [
        # Each maze: list of wall rects (x, y, w, h) relative to screen center
        [(-200, -150, 400, 20), (-200, 150, 400, 20),
         (-200, -150, 20, 300), (180, -80, 20, 160)],
        [(-200, -150, 400, 20), (-200, 150, 400, 20),
         (-200, -150, 20, 300), (180, -150, 20, 220),
         (-80, -30, 160, 20), (-80, 60, 120, 20)],
    ]

    def __init__(self, screen_w=1280, screen_h=720, initial_difficulty="easy"):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.score_engine = ScoreEngine(initial_difficulty)
        self.hud = HUDRenderer(screen_w, screen_h)
        self.running = False
        self.start_time = 0.0
        self.session_duration = 60.0
        self._screen = None
        self._clock = None
        self._marble_x = float(screen_w // 2 - 160)
        self._marble_y = float(screen_h // 2)
        self._marble_vx = 0.0
        self._marble_vy = 0.0
        self._goal_x = float(screen_w // 2 + 150)
        self._goal_y = float(screen_h // 2)
        self._maze_idx = 0
        self._trail: list[tuple] = []
        self._feedback: list[dict] = []

    def start_pygame(self):
        pygame.init()
        self._screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("NeuroRehab — Maze Steering")
        self._clock = pygame.time.Clock()
        self.start_time = time.time()
        self.running = True

    def update(self, inputs: GameInputs, dt: float):
        speed = 150 * self.score_engine.config.target_speed
        # Wrist angle → X direction, arm elevation → Y direction
        ax = (inputs.wrist_angle_norm - 0.5) * 2  # -1 to +1
        ay = (inputs.arm_elevation - 0.5) * -2

        self._marble_vx += ax * speed * dt * 3
        self._marble_vy += ay * speed * dt * 3
        drag = 0.85
        self._marble_vx *= drag
        self._marble_vy *= drag

        new_x = self._marble_x + self._marble_vx * dt
        new_y = self._marble_y + self._marble_vy * dt

        # Wall collisions
        cx, cy = self.screen_w // 2, self.screen_h // 2
        for (wx, wy, ww, wh) in self.MAZES[self._maze_idx % len(self.MAZES)]:
            rx = cx + wx; ry = cy + wy
            mr = 12
            if rx < new_x + mr < rx + ww and ry < new_y + mr < ry + wh:
                self._marble_vx *= -0.6
                self._marble_vy *= -0.6
                new_x = self._marble_x
                new_y = self._marble_y
                self.score_engine.miss()

        # Boundary
        new_x = max(20, min(self.screen_w - 20, new_x))
        new_y = max(75, min(self.screen_h - 20, new_y))

        self._marble_x, self._marble_y = new_x, new_y
        self._trail.append((int(new_x), int(new_y)))
        if len(self._trail) > 60:
            self._trail.pop(0)

        # Goal
        if math.hypot(new_x - self._goal_x, new_y - self._goal_y) < 25:
            self.score_engine.hit(bonus=1.0)
            self._maze_idx += 1
            self._marble_x = float(self.screen_w // 2 - 160)
            self._marble_y = float(self.screen_h // 2)
            self._marble_vx = self._marble_vy = 0.0
            self._trail.clear()
            self._feedback.append({"text": "GOAL! 🎯", "color": C_GOOD,
                                    "x": self._goal_x, "y": self._goal_y, "ttl": 1.5})

        for f in self._feedback:
            f["ttl"] -= dt
            f["y"] -= 30 * dt
        self._feedback = [f for f in self._feedback if f["ttl"] > 0]

    def render(self, surface, inputs: GameInputs):
        surface.fill((10, 12, 25))
        # Grid
        for x in range(0, self.screen_w, 60):
            pygame.draw.line(surface, (18, 22, 45), (x, 55), (x, self.screen_h), 1)
        for y in range(55, self.screen_h, 60):
            pygame.draw.line(surface, (18, 22, 45), (0, y), (self.screen_w, y), 1)

        cx, cy = self.screen_w // 2, self.screen_h // 2
        # Walls
        for (wx, wy, ww, wh) in self.MAZES[self._maze_idx % len(self.MAZES)]:
            pygame.draw.rect(surface, (60, 80, 140),
                             (cx + wx, cy + wy, ww, wh), border_radius=4)
            pygame.draw.rect(surface, C_ACCENT,
                             (cx + wx, cy + wy, ww, wh), 1, border_radius=4)

        # Trail
        for i, (tx, ty) in enumerate(self._trail):
            alpha = int(180 * i / len(self._trail))
            s = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(s, (0, 200, 170, alpha), (3, 3), 3)
            surface.blit(s, (tx - 3, ty - 3))

        # Goal
        t = time.time()
        gr = int(20 + 5 * math.sin(t * 3))
        pygame.draw.circle(surface, (50, 220, 100),
                           (int(self._goal_x), int(self._goal_y)), gr)
        pygame.draw.circle(surface, C_WHITE,
                           (int(self._goal_x), int(self._goal_y)), gr, 2)
        font = pygame.font.SysFont("Arial", 14)
        lbl = font.render("GOAL", True, C_WHITE)
        surface.blit(lbl, (int(self._goal_x) - lbl.get_width() // 2,
                           int(self._goal_y) - 10))

        # Marble
        mx, my = int(self._marble_x), int(self._marble_y)
        pygame.draw.circle(surface, (200, 220, 255), (mx, my), 14)
        pygame.draw.circle(surface, C_WHITE, (mx - 4, my - 4), 4)
        pygame.draw.circle(surface, (100, 120, 200), (mx, my), 14, 2)

        # Wrist angle indicator
        ax = (inputs.wrist_angle_norm - 0.5) * 2
        ay = (inputs.arm_elevation - 0.5) * -2
        ix = int(self.screen_w // 2 + ax * 60)
        iy = int(self.screen_h - 40 + ay * 30)
        pygame.draw.circle(surface, C_PANEL, (self.screen_w // 2, self.screen_h - 40), 30)
        pygame.draw.circle(surface, C_ACCENT, (self.screen_w // 2, self.screen_h - 40), 30, 1)
        pygame.draw.circle(surface, C_ACCENT, (ix, iy), 8)

        for f in self._feedback:
            font2 = pygame.font.SysFont("Arial", 26, bold=True)
            surf = font2.render(f["text"], True, f["color"])
            surf.set_alpha(int(255 * min(f["ttl"], 1.0)))
            surface.blit(surf, (int(f["x"]) - surf.get_width() // 2, int(f["y"])))

        time_remaining = self.session_duration - (time.time() - self.start_time)
        self.hud.draw(surface, self.score_engine, inputs, "Maze Steering", time_remaining)

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
                keys = pygame.key.get_pressed()
                ax = (1 if keys[pygame.K_RIGHT] else 0) - (1 if keys[pygame.K_LEFT] else 0)
                ay = (1 if keys[pygame.K_DOWN] else 0) - (1 if keys[pygame.K_UP] else 0)
                inputs = GameInputs(
                    wrist_angle_norm=0.5 + ax * 0.4,
                    arm_elevation=0.5 - ay * 0.4,
                    stability=0.9, rom=40.0, performance_label="good"
                )
            self.update(inputs, dt)
            self.render(self._screen, inputs)
            pygame.display.flip()
            self._clock.tick(60)
            if time.time() - self.start_time > self.session_duration:
                self.running = False
        pygame.quit()
        return self.score_engine.score
