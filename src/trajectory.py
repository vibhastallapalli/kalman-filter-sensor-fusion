"""
True trajectory generator for a vehicle that accelerates, cruises, then decelerates.
"""

import numpy as np


class TrueTrajectory:
    """
    Generates the ground-truth position, velocity, and acceleration of a vehicle.

    Phase 1 (0–3s):   Accelerate at +3 m/s²  (0  → 9 m/s)
    Phase 2 (3–7s):   Cruise at 9 m/s        (constant velocity)
    Phase 3 (7–10s):  Decelerate at –3 m/s²  (9  → 0 m/s)
    """

    ACCEL_PHASE_END   = 3.0   # seconds
    CRUISE_PHASE_END  = 7.0   # seconds
    TOTAL_DURATION    = 10.0  # seconds

    ACCEL_VALUE       = 3.0   # m/s²
    CRUISE_SPEED      = 9.0   # m/s

    def __init__(self):
        self.initial_position = 0.0   # metres
        self.initial_velocity = 0.0   # m/s

    def get_acceleration(self, t: float) -> float:
        """Return true acceleration at time t."""
        if t < self.ACCEL_PHASE_END:
            return self.ACCEL_VALUE
        elif t < self.CRUISE_PHASE_END:
            return 0.0
        else:
            return -self.ACCEL_VALUE

    def get_state(self, t: float) -> tuple[float, float, float]:
        """
        Return (position, velocity, acceleration) at time t using kinematics.
        Computed analytically from phase boundaries so it is exact.
        """
        a = self.get_acceleration(t)

        if t <= self.ACCEL_PHASE_END:
            v = self.initial_velocity + self.ACCEL_VALUE * t
            p = self.initial_position + self.initial_velocity * t + 0.5 * self.ACCEL_VALUE * t**2

        elif t <= self.CRUISE_PHASE_END:
            # State at end of acceleration phase
            v_end_accel = self.ACCEL_VALUE * self.ACCEL_PHASE_END
            p_end_accel = 0.5 * self.ACCEL_VALUE * self.ACCEL_PHASE_END**2

            dt = t - self.ACCEL_PHASE_END
            v = v_end_accel
            p = p_end_accel + v_end_accel * dt

        else:
            # State at end of cruise phase
            v_end_accel = self.ACCEL_VALUE * self.ACCEL_PHASE_END
            p_end_accel = 0.5 * self.ACCEL_VALUE * self.ACCEL_PHASE_END**2

            dt_cruise = self.CRUISE_PHASE_END - self.ACCEL_PHASE_END
            p_end_cruise = p_end_accel + v_end_accel * dt_cruise
            v_end_cruise = v_end_accel

            dt = t - self.CRUISE_PHASE_END
            v = v_end_cruise - self.ACCEL_VALUE * dt
            p = p_end_cruise + v_end_cruise * dt - 0.5 * self.ACCEL_VALUE * dt**2

        return p, v, a

    def generate(self, timesteps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate arrays of (positions, velocities, accelerations) for all timesteps."""
        positions     = np.zeros(len(timesteps))
        velocities    = np.zeros(len(timesteps))
        accelerations = np.zeros(len(timesteps))

        for i, t in enumerate(timesteps):
            positions[i], velocities[i], accelerations[i] = self.get_state(t)

        return positions, velocities, accelerations
