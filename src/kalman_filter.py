"""
Discrete-time Kalman filter for position/velocity estimation.

State vector:  x = [position, velocity]^T
Control input: u = [acceleration]          (from accelerometer)
Measurement:   z = [position]              (from GPS)

State transition:
    x[k+1] = F·x[k] + B·u[k] + process_noise

    F = [[1, dt],     B = [[0.5·dt²],
         [0,  1]]          [dt     ]]

Measurement model:
    z[k] = H·x[k] + measurement_noise
    H = [[1, 0]]
"""

import numpy as np


class KalmanFilter:
    """
    Full Kalman filter with explicit predict/update steps and covariance propagation.
    """

    def __init__(self, dt: float, process_accel_std: float, gps_noise_std: float,
                 init_position: float = 0.0, init_velocity: float = 0.0):
        """
        Parameters
        ----------
        dt                : simulation time step (seconds)
        process_accel_std : std dev of unmodelled acceleration (process noise)
        gps_noise_std     : std dev of GPS position measurement noise
        init_position     : initial position estimate
        init_velocity     : initial velocity estimate
        """
        self.dt = dt

        # ── State vector ───────────────────────────────────────────────────────
        self.x = np.array([[init_position],
                           [init_velocity]], dtype=float)

        # ── State covariance ──────────────────────────────────────────────────
        # Vehicle starts at known rest position -- begin with tight uncertainty.
        # This lets the filter rely primarily on accelerometer dead-reckoning
        # and accept GPS as a periodic sanity check rather than the primary signal.
        self.P = np.diag([1.0**2, 0.5**2])    # pos ±1 m, vel ±0.5 m/s

        # ── State transition matrix ────────────────────────────────────────────
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])

        # ── Control-input matrix ───────────────────────────────────────────────
        self.B = np.array([[0.5 * dt**2],
                           [dt]])

        # ── Measurement matrix (GPS observes position only) ────────────────────
        self.H = np.array([[1.0, 0.0]])

        # ── Process noise covariance ───────────────────────────────────────────
        # Derived from continuous white-noise acceleration model:
        #   Q = sigma_a² · [[dt⁴/4, dt³/2],
        #                    [dt³/2, dt²  ]]
        sa2 = process_accel_std**2
        self.Q = sa2 * np.array([[0.25 * dt**4, 0.5 * dt**3],
                                  [0.5  * dt**3, dt**2      ]])

        # ── Measurement noise covariance ───────────────────────────────────────
        self.R = np.array([[gps_noise_std**2]])

        # Diagnostic stores for the last update step
        self.last_innovation      = None
        self.last_kalman_gain     = None
        self.last_innovation_cov  = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, acceleration: float) -> None:
        """
        Propagate state and covariance forward one time step using
        the accelerometer reading as the control input.
        """
        u = np.array([[acceleration]])

        # State prediction
        self.x = self.F @ self.x + self.B @ u

        # Covariance prediction (adds process noise)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, gps_position: float) -> None:
        """
        Correct the predicted state with a GPS position measurement.
        Uses the standard Kalman update equations.
        """
        z = np.array([[gps_position]])

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(2) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Store diagnostics
        self.last_innovation     = float(y[0, 0])
        self.last_kalman_gain    = K.copy()
        self.last_innovation_cov = float(S[0, 0])

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def position(self) -> float:
        return float(self.x[0, 0])

    @property
    def velocity(self) -> float:
        return float(self.x[1, 0])

    @property
    def position_std(self) -> float:
        """One-sigma position uncertainty from diagonal of P."""
        return float(np.sqrt(self.P[0, 0]))

    @property
    def velocity_std(self) -> float:
        """One-sigma velocity uncertainty from diagonal of P."""
        return float(np.sqrt(self.P[1, 1]))
