"""
Simulation engine: ties trajectory, sensors, and Kalman filter together.
Logs every timestep and computes summary statistics.

RMS comparison methodology
---------------------------
GPS-only baseline: at every timestep the "GPS estimate" is the last
  received GPS fix, held constant until the next fix arrives.  This
  captures the full cost of using a 1 Hz GPS -- large errors between
  updates as the vehicle moves, plus noise at each fix.

Kalman estimate: produced at every timestep (100 Hz) using both the
  accelerometer (predict) and occasional GPS (update).

Both are evaluated against the true position at every timestep, making
the comparison apples-to-apples and demonstrating the real benefit of
sensor fusion.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .trajectory    import TrueTrajectory
from .sensors       import GPSSensor, Accelerometer
from .kalman_filter import KalmanFilter


@dataclass
class StepLog:
    """All data recorded at a single simulation timestep."""
    time:              float
    true_position:     float
    true_velocity:     float
    true_acceleration: float
    accel_reading:     float
    gps_fix:           Optional[float]  # None when GPS did not update this step
    gps_held:          float            # last GPS fix held constant (GPS-only baseline)
    kf_position:       float
    kf_velocity:       float
    kf_pos_std:        float
    kf_error:          float            # |kf_position - true_position|
    gps_error:         float            # |gps_held   - true_position|


@dataclass
class SimulationResults:
    """Aggregated results after the simulation completes."""
    logs:              List[StepLog]
    timesteps:         np.ndarray

    # Full-resolution arrays (all timesteps)
    true_positions:    np.ndarray = field(default_factory=lambda: np.array([]))
    gps_held:          np.ndarray = field(default_factory=lambda: np.array([]))
    kf_positions:      np.ndarray = field(default_factory=lambda: np.array([]))
    kf_stds:           np.ndarray = field(default_factory=lambda: np.array([]))
    accel_readings:    np.ndarray = field(default_factory=lambda: np.array([]))
    kf_errors:         np.ndarray = field(default_factory=lambda: np.array([]))
    gps_errors:        np.ndarray = field(default_factory=lambda: np.array([]))

    # Sparse GPS fix arrays (only at fix timesteps)
    gps_fix_times:     np.ndarray = field(default_factory=lambda: np.array([]))
    gps_fix_values:    np.ndarray = field(default_factory=lambda: np.array([]))

    rms_gps:           float = 0.0   # GPS-only baseline RMS at all timesteps
    rms_kalman:        float = 0.0   # Kalman RMS at all timesteps
    improvement_pct:   float = 0.0

    def compute_statistics(self) -> None:
        """Populate convenience arrays and compute RMS metrics."""
        self.true_positions = np.array([s.true_position  for s in self.logs])
        self.gps_held       = np.array([s.gps_held       for s in self.logs])
        self.kf_positions   = np.array([s.kf_position    for s in self.logs])
        self.kf_stds        = np.array([s.kf_pos_std     for s in self.logs])
        self.accel_readings = np.array([s.accel_reading  for s in self.logs])
        self.kf_errors      = np.array([s.kf_error       for s in self.logs])
        self.gps_errors     = np.array([s.gps_error      for s in self.logs])

        fix_logs = [(s.time, s.gps_fix) for s in self.logs if s.gps_fix is not None]
        if fix_logs:
            self.gps_fix_times  = np.array([f[0] for f in fix_logs])
            self.gps_fix_values = np.array([f[1] for f in fix_logs])

        self.rms_gps    = float(np.sqrt(np.mean(self.gps_errors**2)))
        self.rms_kalman = float(np.sqrt(np.mean(self.kf_errors**2)))

        if self.rms_gps > 0:
            self.improvement_pct = (1.0 - self.rms_kalman / self.rms_gps) * 100.0


class Simulation:
    """
    Runs the sensor fusion simulation.

    Parameters
    ----------
    duration_s      : total simulation time in seconds
    sim_rate_hz     : simulation (and accelerometer) frequency in Hz
    gps_rate_hz     : GPS update frequency in Hz
    gps_noise_std   : GPS position measurement noise std dev (metres)
    accel_noise_std : accelerometer noise std dev (m/s^2)
    process_std     : Kalman filter process noise std dev (m/s^2)
    seed            : random seed for reproducibility
    """

    def __init__(self,
                 duration_s:      float = 10.0,
                 sim_rate_hz:     float = 100.0,
                 gps_rate_hz:     float = 0.5,
                 gps_noise_std:   float = 8.0,
                 accel_noise_std: float = 0.8,
                 process_std:     float = 1.0,
                 seed:            int   = 42):

        self.duration_s = duration_s
        self.dt         = 1.0 / sim_rate_hz
        self.rng        = np.random.default_rng(seed)

        self.trajectory = TrueTrajectory()
        self.gps        = GPSSensor(noise_std=gps_noise_std,
                                    update_rate_hz=gps_rate_hz,
                                    sim_rate_hz=sim_rate_hz,
                                    rng=self.rng)
        self.accel      = Accelerometer(noise_std=accel_noise_std,
                                        bias=0.0,       # zero bias: noise only
                                        rng=self.rng)
        self.kf         = KalmanFilter(dt=self.dt,
                                       process_accel_std=process_std,
                                       gps_noise_std=gps_noise_std)

        self.timesteps  = np.arange(0.0, duration_s + self.dt / 2, self.dt)

    def run(self) -> SimulationResults:
        """Execute the simulation and return a fully populated SimulationResults."""
        true_positions, true_velocities, true_accels = \
            self.trajectory.generate(self.timesteps)

        logs: List[StepLog] = []
        last_gps_held: float = 0.0   # GPS-only baseline: last fix held constant

        for step, t in enumerate(self.timesteps):
            true_p = true_positions[step]
            true_v = true_velocities[step]
            true_a = true_accels[step]

            # -- Accelerometer reading (every step) ---------------------------
            accel_meas = self.accel.measure(true_a)

            # -- Kalman predict using accelerometer ---------------------------
            self.kf.predict(accel_meas)

            # -- GPS fix and Kalman update (sparse) ---------------------------
            gps_fix = None
            if self.gps.is_update_step(step):
                gps_fix      = self.gps.measure(true_p)
                last_gps_held = gps_fix
                self.kf.update(gps_fix)

            kf_err  = abs(self.kf.position - true_p)
            gps_err = abs(last_gps_held    - true_p)

            logs.append(StepLog(
                time              = t,
                true_position     = true_p,
                true_velocity     = true_v,
                true_acceleration = true_a,
                accel_reading     = accel_meas,
                gps_fix           = gps_fix,
                gps_held          = last_gps_held,
                kf_position       = self.kf.position,
                kf_velocity       = self.kf.velocity,
                kf_pos_std        = self.kf.position_std,
                kf_error          = kf_err,
                gps_error         = gps_err,
            ))

        results = SimulationResults(logs=logs, timesteps=self.timesteps)
        results.compute_statistics()
        return results
