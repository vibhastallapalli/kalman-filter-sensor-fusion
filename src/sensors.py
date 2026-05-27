"""
Sensor models for GPS (low-frequency, high-noise) and accelerometer (high-frequency, low-noise).
"""

import numpy as np


class GPSSensor:
    """
    Simulates a GPS receiver.

    Characteristics:
      - Measures position only.
      - High measurement noise (std dev ~5 m).
      - Low update rate (1 Hz by default → one reading per 100 simulation steps at 100 Hz).
    """

    def __init__(self, noise_std: float = 5.0, update_rate_hz: float = 1.0,
                 sim_rate_hz: float = 100.0, rng: np.random.Generator = None):
        self.noise_std    = noise_std
        self.update_rate  = update_rate_hz
        self.sim_rate     = sim_rate_hz
        # How many simulation ticks between GPS updates
        self.update_every = max(1, int(round(sim_rate_hz / update_rate_hz)))
        self._rng         = rng if rng is not None else np.random.default_rng()

    def measure(self, true_position: float) -> float:
        """Return a noisy GPS position reading."""
        return true_position + self._rng.normal(0.0, self.noise_std)

    def is_update_step(self, step_index: int) -> bool:
        """Return True if a new GPS fix is available at this simulation step."""
        return (step_index % self.update_every) == 0


class Accelerometer:
    """
    Simulates a MEMS accelerometer.

    Characteristics:
      - Measures acceleration directly (plus bias drift modelled as fixed offset).
      - Low measurement noise (std dev ~0.3 m/s²).
      - Updates every simulation tick (100 Hz).
    """

    def __init__(self, noise_std: float = 0.3, bias: float = 0.05,
                 rng: np.random.Generator = None):
        self.noise_std = noise_std
        self.bias      = bias            # small constant bias, typical of MEMS devices
        self._rng      = rng if rng is not None else np.random.default_rng()

    def measure(self, true_acceleration: float) -> float:
        """Return a noisy accelerometer reading."""
        return true_acceleration + self.bias + self._rng.normal(0.0, self.noise_std)
