"""
vision/tracker.py

Webcam-based movement tracker using MediaPipe Hands + Pose.
Detects:
  - Hand landmarks (for grip/pinch gestures)
  - Arm pose (for ROM, angle)
  - Outputs structured motion data per frame
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
import math

try:
    import mediapipe as mp
    # Check that solutions API is available (older-style API)
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        MEDIAPIPE_AVAILABLE = True
    else:
        MEDIAPIPE_AVAILABLE = False
        print("[tracker] MediaPipe solutions API not available — using simulated motion.")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[tracker] MediaPipe not installed — using simulated motion.")


@dataclass
class HandState:
    """Per-frame hand tracking state."""
    detected: bool = False
    landmarks: list = field(default_factory=list)  # 21 (x,y,z) landmarks
    pinch_distance: float = 1.0        # 0=closed, 1=fully open
    grip_aperture: float = 1.0         # Average finger spread
    is_open: bool = True
    is_pinching: bool = False
    wrist_angle: float = 0.0           # Degrees from neutral
    palm_center: tuple = (0.0, 0.0)


@dataclass
class PoseState:
    """Per-frame arm pose state."""
    detected: bool = False
    shoulder: Optional[tuple] = None
    elbow: Optional[tuple] = None
    wrist: Optional[tuple] = None
    arm_angle: float = 0.0             # Elbow angle (degrees)
    elevation: float = 0.0            # Arm elevation from horizontal
    velocity: float = 0.0             # Wrist velocity (px/frame)


@dataclass
class MotionFrame:
    """Combined motion data for one frame."""
    timestamp: float = 0.0
    hand: HandState = field(default_factory=HandState)
    pose: PoseState = field(default_factory=PoseState)
    stability: float = 1.0            # 1=stable, 0=tremor
    rom: float = 0.0                  # Range of motion (degrees)


class MovementTracker:
    """
    Real-time movement tracker using webcam + MediaPipe.
    Outputs MotionFrame per camera frame.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._prev_wrist = None
        self._wrist_history = []
        self._stability_window = []

        if MEDIAPIPE_AVAILABLE:
            self._mp_hands = mp.solutions.hands
            self._mp_pose = mp.solutions.pose
            self._mp_draw = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            self._pose = self._mp_pose.Pose(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )

    def start(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            print("[tracker] WARNING: Could not open camera — using simulated mode")
            self._cap = None

    def stop(self):
        if self._cap:
            self._cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def read_frame(self) -> tuple[Optional[np.ndarray], MotionFrame]:
        """
        Read one frame and return (annotated_frame, MotionFrame).
        Returns (None, simulated_frame) if camera unavailable.
        """
        if self._cap is None or not MEDIAPIPE_AVAILABLE:
            return None, self._simulate_motion()

        ret, frame = self._cap.read()
        if not ret:
            return None, self._simulate_motion()

        frame = cv2.flip(frame, 1)  # Mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        hand_results = self._hands.process(rgb)
        pose_results = self._pose.process(rgb)

        rgb.flags.writeable = True
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        hand_state = self._process_hands(hand_results, annotated)
        pose_state = self._process_pose(pose_results, annotated)
        stability = self._compute_stability(pose_state.wrist)
        rom = self._compute_rom(pose_state)

        self._draw_overlay(annotated, hand_state, pose_state, stability, rom)

        return annotated, MotionFrame(
            timestamp=time.time(),
            hand=hand_state,
            pose=pose_state,
            stability=stability,
            rom=rom,
        )

    def _process_hands(self, results, frame: np.ndarray) -> HandState:
        state = HandState()
        if not results or not results.multi_hand_landmarks:
            return state

        h, w = frame.shape[:2]
        lm = results.multi_hand_landmarks[0]

        self._mp_draw.draw_landmarks(
            frame, lm, self._mp_hands.HAND_CONNECTIONS,
            self._mp_draw.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=4),
            self._mp_draw.DrawingSpec(color=(0, 180, 80), thickness=2),
        )

        def lmxy(idx):
            return (lm.landmark[idx].x * w, lm.landmark[idx].y * h)

        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        thumb_tip = lmxy(4)
        index_tip = lmxy(8)
        middle_tip = lmxy(12)
        ring_tip = lmxy(16)
        pinky_tip = lmxy(20)
        wrist = lmxy(0)
        middle_mcp = lmxy(9)

        # Normalize by hand size
        hand_size = dist(wrist, middle_mcp) + 1e-6

        pinch_dist = dist(thumb_tip, index_tip) / hand_size
        grip = np.mean([
            dist(thumb_tip, middle_tip),
            dist(thumb_tip, ring_tip),
            dist(thumb_tip, pinky_tip),
        ]) / hand_size

        # Wrist angle (relative to vertical)
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        wrist_angle = math.degrees(math.atan2(dx, -dy))

        state.detected = True
        state.landmarks = [(lm.landmark[i].x, lm.landmark[i].y, lm.landmark[i].z)
                           for i in range(21)]
        state.pinch_distance = float(np.clip(pinch_dist, 0, 1))
        state.grip_aperture = float(np.clip(grip / 2, 0, 1))
        state.is_open = grip > 0.6
        state.is_pinching = pinch_dist < 0.3
        state.wrist_angle = wrist_angle
        state.palm_center = (
            float(np.mean([lmxy(i)[0] for i in range(21)])),
            float(np.mean([lmxy(i)[1] for i in range(21)])),
        )

        return state

    def _process_pose(self, results, frame: np.ndarray) -> PoseState:
        state = PoseState()
        if not results or not results.pose_landmarks:
            return state

        h, w = frame.shape[:2]
        lm = results.pose_landmarks.landmark

        def lmxy(idx):
            return (lm[idx].x * w, lm[idx].y * h)

        # Use right arm (indices 12, 14, 16) — adjust for left if needed
        try:
            shoulder = lmxy(12)
            elbow = lmxy(14)
            wrist = lmxy(16)
        except Exception:
            return state

        # Draw arm
        for a, b in [(shoulder, elbow), (elbow, wrist)]:
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     (255, 100, 0), 3)
        for pt in [shoulder, elbow, wrist]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, (0, 120, 255), -1)

        # Elbow angle
        v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
        v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
        cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (
            (math.hypot(*v1) + 1e-6) * (math.hypot(*v2) + 1e-6)
        )
        arm_angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))

        # Elevation (how far shoulder is above wrist)
        elevation = max(0.0, (shoulder[1] - wrist[1]) / h * 90)

        # Velocity
        velocity = 0.0
        if self._prev_wrist:
            velocity = math.hypot(
                wrist[0] - self._prev_wrist[0],
                wrist[1] - self._prev_wrist[1],
            )
        self._prev_wrist = wrist

        state.detected = True
        state.shoulder = shoulder
        state.elbow = elbow
        state.wrist = wrist
        state.arm_angle = float(arm_angle)
        state.elevation = float(elevation)
        state.velocity = float(velocity)

        # Annotate angle
        cv2.putText(frame, f"{arm_angle:.0f}°",
                    (int(elbow[0]) + 10, int(elbow[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return state

    def _compute_stability(self, wrist_pos) -> float:
        """Stability score based on wrist jitter (1=stable, 0=tremor)."""
        if wrist_pos is None:
            return 1.0
        self._stability_window.append(wrist_pos)
        if len(self._stability_window) > 30:
            self._stability_window.pop(0)
        if len(self._stability_window) < 5:
            return 1.0
        xs = [p[0] for p in self._stability_window]
        ys = [p[1] for p in self._stability_window]
        jitter = math.hypot(np.std(xs), np.std(ys))
        return float(np.clip(1.0 - jitter / 30.0, 0.0, 1.0))

    def _compute_rom(self, pose: PoseState) -> float:
        """Range of motion in degrees."""
        if not self._wrist_history and pose.wrist:
            self._wrist_history.append(pose.elevation)
        elif pose.wrist:
            self._wrist_history.append(pose.elevation)
            self._wrist_history = self._wrist_history[-60:]
        if len(self._wrist_history) < 2:
            return pose.arm_angle
        return float(max(self._wrist_history) - min(self._wrist_history))

    def _draw_overlay(self, frame, hand: HandState, pose: PoseState,
                      stability: float, rom: float):
        """Draw HUD overlay on frame."""
        h, w = frame.shape[:2]

        # Semi-transparent sidebar
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 220, 0), (w, 160), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        def txt(text, y, color=(200, 220, 255)):
            cv2.putText(frame, text, (w - 210, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        txt("NeuroRehab Track", 25, (100, 200, 255))
        txt(f"ROM: {rom:.1f} deg", 50)
        txt(f"Stability: {stability:.2f}", 75,
            (0, 255, 0) if stability > 0.7 else (0, 165, 255))
        if hand.detected:
            txt(f"Pinch: {hand.pinch_distance:.2f}", 100)
            txt(f"Grip: {'OPEN' if hand.is_open else 'CLOSED'}", 125,
                (0, 255, 100) if hand.is_open else (100, 150, 255))
        if pose.detected:
            txt(f"Elbow: {pose.arm_angle:.0f} deg", 150)

    def _simulate_motion(self) -> MotionFrame:
        """Generate synthetic motion data when camera is unavailable."""
        t = time.time()
        hand = HandState(
            detected=True,
            pinch_distance=float(0.5 + 0.4 * math.sin(t * 0.8)),
            grip_aperture=float(0.5 + 0.3 * math.cos(t * 0.5)),
            is_open=(math.sin(t * 0.5) > 0),
            is_pinching=(math.sin(t * 0.8) < -0.5),
            wrist_angle=float(20 * math.sin(t * 0.3)),
        )
        pose = PoseState(
            detected=True,
            arm_angle=float(90 + 30 * math.sin(t * 0.4)),
            elevation=float(30 + 15 * math.sin(t * 0.4)),
            velocity=float(abs(5 * math.cos(t * 0.4))),
        )
        return MotionFrame(
            timestamp=t,
            hand=hand,
            pose=pose,
            stability=float(0.85 + 0.1 * math.sin(t * 2)),
            rom=float(45 + 10 * math.sin(t * 0.2)),
        )
