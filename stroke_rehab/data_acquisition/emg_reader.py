"""
data_acquisition/emg_reader.py

Unified EMG data reader supporting:
  - Simulated (demo/testing)
  - Serial (ESP32 via USB)
  - Bluetooth LE
"""

import time
import math
import random
import threading
import queue
from abc import ABC, abstractmethod
import numpy as np

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import asyncio
    import bleak
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False


class BaseEMGReader(ABC):
    """Abstract base for all EMG reader backends."""

    def __init__(self, sample_rate: int = 1000):
        self.sample_rate = sample_rate
        self._running = False
        self._queue = queue.Queue(maxsize=10000)
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def read(self, n_samples: int = 1) -> list[float]:
        """Blocking read of n samples."""
        samples = []
        while len(samples) < n_samples:
            try:
                samples.append(self._queue.get(timeout=1.0))
            except queue.Empty:
                break
        return samples

    def read_available(self) -> list[float]:
        """Non-blocking drain of all available samples."""
        samples = []
        while not self._queue.empty():
            try:
                samples.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return samples

    @abstractmethod
    def _run(self):
        """Background acquisition loop — implemented per backend."""
        ...


# ─── Simulated EMG ────────────────────────────────────────────────────────────

class SimulatedEMGReader(BaseEMGReader):
    """
    Generates realistic synthetic EMG signals for demo/testing.
    Simulates:
      - Baseline noise
      - Voluntary muscle contractions (bursts)
      - Fatigue effects over time
      - Stroke-pattern artifacts (intermittent weak activations)
    """

    def __init__(self, sample_rate: int = 1000, pattern: str = "moderate"):
        super().__init__(sample_rate)
        self.pattern = pattern   # "poor" | "moderate" | "good"
        self._elapsed = 0.0

    def _run(self):
        interval = 1.0 / self.sample_rate
        t = 0.0
        contraction_phase = 0.0
        contraction_active = False
        contraction_timer = 0.0
        next_contraction = random.uniform(0.5, 1.5)
        fatigue = 0.0

        while self._running:
            # Base noise
            noise = np.random.randn() * 0.02

            # Contraction pattern
            if not contraction_active:
                contraction_timer += interval
                if contraction_timer >= next_contraction:
                    contraction_active = True
                    contraction_phase = 0.0
                    contraction_timer = 0.0
                    next_contraction = random.uniform(0.3, 1.0)
            else:
                contraction_phase += interval
                duration = self._contraction_duration()
                if contraction_phase >= duration:
                    contraction_active = False
                    fatigue = min(fatigue + 0.05, 1.0)  # accumulate fatigue

            # EMG signal
            if contraction_active:
                amp = self._amplitude(fatigue)
                # Simulate motor unit firing
                signal = amp * abs(
                    np.random.randn() * 0.5
                    + math.sin(2 * math.pi * 120 * t) * 0.3
                    + math.sin(2 * math.pi * 240 * t) * 0.15
                )
            else:
                signal = abs(noise) * 0.1

            # Add power line artifact
            signal += math.sin(2 * math.pi * 50 * t) * 0.005

            # Normalize to 0–1 (simulating ADC)
            signal = float(np.clip(signal + abs(noise), 0.0, 1.0))

            if not self._queue.full():
                self._queue.put(signal)

            t += interval
            self._elapsed += interval
            time.sleep(interval * 0.95)

    def _amplitude(self, fatigue: float) -> float:
        base = {"poor": 0.2, "moderate": 0.55, "good": 0.85}.get(self.pattern, 0.55)
        return base * (1.0 - fatigue * 0.4)

    def _contraction_duration(self) -> float:
        return {"poor": 0.15, "moderate": 0.35, "good": 0.5}.get(self.pattern, 0.35)


# ─── Serial (ESP32) EMG ───────────────────────────────────────────────────────

class SerialEMGReader(BaseEMGReader):
    """
    Reads EMG from ESP32 over USB serial.
    Expected format per line: "EMG:<int_value>\\n"
    where value is 0–4095 (12-bit ADC).
    """

    def __init__(self, port: str, baud: int = 115200, sample_rate: int = 1000):
        super().__init__(sample_rate)
        self.port = port
        self.baud = baud
        self._ser = None

    def start(self):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial not installed. Run: pip install pyserial")
        self._ser = serial.Serial(self.port, self.baud, timeout=1.0)
        super().start()

    def stop(self):
        super().stop()
        if self._ser and self._ser.is_open:
            self._ser.close()

    def _run(self):
        while self._running:
            try:
                line = self._ser.readline().decode("utf-8").strip()
                if line.startswith("EMG:"):
                    raw = int(line.split(":")[1])
                    normalized = raw / 4095.0
                    if not self._queue.full():
                        self._queue.put(normalized)
            except Exception:
                time.sleep(0.001)


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_emg_reader(source: str = "simulated", **kwargs) -> BaseEMGReader:
    """
    Factory function — creates the appropriate reader.

    Args:
        source: "simulated" | "serial" | "bluetooth"
        **kwargs: passed to reader constructor

    Returns:
        Started EMG reader instance
    """
    if source == "simulated":
        pattern = kwargs.get("pattern", "moderate")
        reader = SimulatedEMGReader(pattern=pattern)
    elif source == "serial":
        port = kwargs.get("port", "/dev/ttyUSB0")
        baud = kwargs.get("baud", 115200)
        reader = SerialEMGReader(port=port, baud=baud)
    else:
        raise ValueError(f"Unknown EMG source: '{source}'. Use 'simulated', 'serial', or 'bluetooth'.")

    reader.start()
    return reader


if __name__ == "__main__":
    print("Testing SimulatedEMGReader for 3 seconds...")
    reader = create_emg_reader("simulated", pattern="moderate")
    time.sleep(3)
    samples = reader.read_available()
    print(f"Collected {len(samples)} samples")
    print(f"  Mean: {np.mean(samples):.4f}")
    print(f"  Max:  {np.max(samples):.4f}")
    print(f"  Min:  {np.min(samples):.4f}")
    reader.stop()
