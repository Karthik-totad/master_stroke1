"""
data_acquisition/preprocessor.py

EMG signal preprocessing pipeline:
  - Bandpass filtering (20–450 Hz)
  - Notch filtering (50/60 Hz power line)
  - Rectification
  - Normalization
  - Feature extraction: RMS, MAV, ZC, SSC, WL, MNF, MDF
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Optional


@dataclass
class EMGFeatures:
    """Extracted features from one EMG window."""
    rms: float           # Root Mean Square — overall activation level
    mav: float           # Mean Absolute Value
    zc: int              # Zero Crossings — frequency indicator
    ssc: int             # Slope Sign Changes — frequency indicator
    wl: float            # Waveform Length — complexity
    var: float           # Variance
    mean_freq: float     # Mean Frequency (spectral)
    median_freq: float   # Median Frequency (spectral)
    peak_amp: float      # Peak amplitude
    contraction_ratio: float  # Fraction of time above activation threshold

    def to_array(self) -> np.ndarray:
        return np.array([
            self.rms, self.mav, self.zc, self.ssc, self.wl,
            self.var, self.mean_freq, self.median_freq,
            self.peak_amp, self.contraction_ratio
        ])

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "rms", "mav", "zero_crossings", "slope_sign_changes",
            "waveform_length", "variance", "mean_freq", "median_freq",
            "peak_amplitude", "contraction_ratio"
        ]


class EMGPreprocessor:
    """
    Real-time EMG preprocessing pipeline.
    Designed for 1000 Hz sampling rate.
    """

    ACTIVATION_THRESHOLD = 0.05  # Normalized units

    def __init__(
        self,
        sample_rate: int = 1000,
        lowpass_hz: float = 450.0,
        highpass_hz: float = 20.0,
        notch_hz: float = 50.0,
        window_size: int = 256,
    ):
        self.fs = sample_rate
        self.window_size = window_size
        self._buffer = np.zeros(window_size * 4)  # Circular buffer
        self._buf_idx = 0
        self._buf_count = 0

        # Design filters once
        self._bp_sos = self._bandpass_filter(highpass_hz, lowpass_hz)
        self._notch_b, self._notch_a = self._notch_filter(notch_hz)

        # MVC (Maximum Voluntary Contraction) for normalization — updated dynamically
        self._mvc = 1.0

    # ─── Filter Design ────────────────────────────────────────────────────────

    def _bandpass_filter(self, low: float, high: float):
        nyq = self.fs / 2.0
        return signal.butter(
            4, [low / nyq, high / nyq], btype="bandpass", output="sos"
        )

    def _notch_filter(self, freq: float, Q: float = 30.0):
        return signal.iirnotch(freq, Q, self.fs)

    # ─── Preprocessing ────────────────────────────────────────────────────────

    def process_window(self, raw_window: np.ndarray) -> tuple[np.ndarray, EMGFeatures]:
        """
        Process a raw EMG window.

        Args:
            raw_window: 1D array of raw ADC values (0–1 normalized)

        Returns:
            (filtered_signal, EMGFeatures)
        """
        raw = np.asarray(raw_window, dtype=np.float64)

        # 1. Bandpass filter
        filtered = signal.sosfilt(self._bp_sos, raw)

        # 2. Notch filter
        filtered = signal.lfilter(self._notch_b, self._notch_a, filtered)

        # 3. Full-wave rectification
        rectified = np.abs(filtered)

        # 4. MVC normalization (update running max)
        peak = float(np.max(rectified))
        if peak > self._mvc:
            self._mvc = peak * 0.9 + self._mvc * 0.1  # Smooth update
        normalized = np.clip(rectified / max(self._mvc, 1e-6), 0.0, 1.0)

        # 5. Extract features
        features = self._extract_features(filtered, normalized)

        return normalized, features

    def _extract_features(self, filtered: np.ndarray, normalized: np.ndarray) -> EMGFeatures:
        """Extract time-domain and frequency-domain EMG features."""
        n = len(filtered)

        # Time domain
        rms = float(np.sqrt(np.mean(filtered ** 2)))
        mav = float(np.mean(np.abs(filtered)))
        var = float(np.var(filtered))
        peak_amp = float(np.max(np.abs(filtered)))
        wl = float(np.sum(np.abs(np.diff(filtered))))

        # Zero crossings (with deadband)
        threshold = 0.01
        sign = np.sign(filtered)
        zc = int(np.sum(
            (np.diff(sign) != 0) & (np.abs(np.diff(filtered)) > threshold)
        ))

        # Slope sign changes
        diff = np.diff(filtered)
        ssc = int(np.sum(
            (np.diff(np.sign(diff)) != 0) & (np.abs(diff[:-1]) > threshold)
        ))

        # Frequency domain
        freqs, psd = signal.welch(filtered, self.fs, nperseg=min(n, 128))
        valid = (freqs >= 20) & (freqs <= 450)
        freqs_v, psd_v = freqs[valid], psd[valid]

        if len(psd_v) > 0 and np.sum(psd_v) > 0:
            mean_freq = float(np.sum(freqs_v * psd_v) / np.sum(psd_v))
            cumulative = np.cumsum(psd_v)
            median_freq = float(freqs_v[np.searchsorted(cumulative, cumulative[-1] / 2)])
        else:
            mean_freq = median_freq = 150.0

        # Contraction ratio
        contraction_ratio = float(np.mean(normalized > self.ACTIVATION_THRESHOLD))

        return EMGFeatures(
            rms=rms, mav=mav, zc=zc, ssc=ssc, wl=wl,
            var=var, mean_freq=mean_freq, median_freq=median_freq,
            peak_amp=peak_amp, contraction_ratio=contraction_ratio,
        )

    def set_mvc(self, mvc_value: float):
        """Manually set MVC for calibration."""
        self._mvc = max(mvc_value, 1e-6)

    def smooth_envelope(self, signal_in: np.ndarray, window_ms: int = 50) -> np.ndarray:
        """Apply moving average envelope smoothing."""
        k = max(1, int(window_ms * self.fs / 1000))
        kernel = np.ones(k) / k
        return np.convolve(signal_in, kernel, mode="same")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from data_acquisition.emg_reader import create_emg_reader
    import time

    print("Testing EMG preprocessor...")
    reader = create_emg_reader("simulated", pattern="moderate")
    preprocessor = EMGPreprocessor()

    time.sleep(1.0)  # Collect samples

    raw = np.array(reader.read_available())
    if len(raw) >= 256:
        window = raw[:256]
        filtered, features = preprocessor.process_window(window)
        print(f"Window: {len(window)} samples")
        print(f"Features:")
        for name, val in zip(EMGFeatures.feature_names(), features.to_array()):
            print(f"  {name:25s}: {val:.4f}")
    else:
        print(f"Only got {len(raw)} samples — wait longer")

    reader.stop()
