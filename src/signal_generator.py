"""
Signal generation module for ultrasonic A-scan signals.

Generates synthetic ultrasonic signals in two classes:
- defect: Signal with an anomalous echo/spike at a random position
- no_defect: Clean signal with Gaussian noise only
"""

import numpy as np
from typing import Tuple, List


class UltrasonicSignalGenerator:
    """Generate synthetic ultrasonic A-scan signals."""
    
    def __init__(
        self,
        signal_length: int = 256,
        sampling_rate: float = 50e6,
        noise_std: float = 0.05,
        defect_amplitude: float = 0.5,
        seed: int = None
    ):
        """
        Initialize the signal generator.
        
        Args:
            signal_length: Length of each signal sample
            sampling_rate: Sampling rate in Hz
            noise_std: Standard deviation of Gaussian noise
            defect_amplitude: Amplitude of the defect spike
            seed: Random seed for reproducibility
        """
        self.signal_length = signal_length
        self.sampling_rate = sampling_rate
        self.noise_std = noise_std
        self.defect_amplitude = defect_amplitude
        
        if seed is not None:
            np.random.seed(seed)
    
    def _generate_base_signal(self) -> np.ndarray:
        """
        Generate a base ultrasonic signal (clean).
        
        Simulates a simple ultrasonic pulse with reflections.
        
        Returns:
            Base signal of shape (signal_length,)
        """
        signal = np.zeros(self.signal_length)
        
        # Initial pulse
        pulse_width = 10
        pulse_center = 30
        signal[pulse_center:pulse_center + pulse_width] = np.hanning(pulse_width)
        
        # Natural reflections
        reflection_positions = [80, 150, 220]
        for pos in reflection_positions:
            if pos < self.signal_length:
                amplitude = 0.3 / (1 + (pos - 30) / 100)  # Decay with distance
                signal[pos:pos + pulse_width] += amplitude * np.hanning(pulse_width)
        
        return signal
    
    def generate_no_defect(self) -> np.ndarray:
        """
        Generate a signal without defect (clean signal + noise).
        
        Returns:
            Signal array of shape (signal_length,)
        """
        signal = self._generate_base_signal()
        noise = np.random.normal(0, self.noise_std, self.signal_length)
        return signal + noise
    
    def generate_defect(self) -> np.ndarray:
        """
        Generate a signal with defect (spike at random position + noise).
        
        Returns:
            Signal array of shape (signal_length,)
        """
        signal = self._generate_base_signal()
        
        # Add defect spike at random position
        defect_position = np.random.randint(50, self.signal_length - 50)
        spike_width = 8
        defect_spike = self.defect_amplitude * np.hanning(spike_width)
        
        start_idx = max(0, defect_position - spike_width // 2)
        end_idx = min(self.signal_length, start_idx + spike_width)
        signal[start_idx:end_idx] += defect_spike[:end_idx - start_idx]
        
        noise = np.random.normal(0, self.noise_std, self.signal_length)
        return signal + noise
    
    def generate_dataset(
        self,
        n_defect: int = 500,
        n_no_defect: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a dataset of signals.
        
        Args:
            n_defect: Number of defect signals
            n_no_defect: Number of no-defect signals
        
        Returns:
            signals: Array of shape (n_defect + n_no_defect, signal_length)
            labels: Array of shape (n_defect + n_no_defect,) with 0=no_defect, 1=defect
        """
        signals = []
        labels = []
        
        # Generate no-defect signals
        for _ in range(n_no_defect):
            signals.append(self.generate_no_defect())
            labels.append(0)
        
        # Generate defect signals
        for _ in range(n_defect):
            signals.append(self.generate_defect())
            labels.append(1)
        
        signals = np.array(signals, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Shuffle
        indices = np.random.permutation(len(signals))
        signals = signals[indices]
        labels = labels[indices]
        
        return signals, labels
