"""
Kalman Filter Sensor Fusion Simulation
Entry point -- run this file to execute the full simulation.

GPS-only baseline: last GPS fix held constant between updates (1 Hz).
Kalman estimate:   position fused from GPS + accelerometer at 100 Hz.
Both evaluated vs true position at every timestep (apples-to-apples).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.simulation import Simulation
from src.plotting   import plot_results


# -- Simulation parameters ---------------------------------------------------
SIM_PARAMS = dict(
    duration_s      = 10.0,   # seconds
    sim_rate_hz     = 100.0,  # Hz (accelerometer rate)
    gps_rate_hz     = 0.5,    # Hz (GPS fix rate -- one fix every 2 seconds)
    gps_noise_std   = 8.0,    # metres (realistic automotive-grade GPS uncertainty)
    accel_noise_std = 0.8,    # m/s^2
    process_std     = 1.0,    # m/s^2 (Kalman process noise)
    seed            = 42,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOT_PATH   = os.path.join(RESULTS_DIR, "simulation.png")
LOG_PATH    = os.path.join(RESULTS_DIR, "simulation_log.csv")


def print_summary(results) -> None:
    """Print a formatted summary table to stdout."""
    sep = "-" * 62
    dt  = float(results.timesteps[1] - results.timesteps[0])

    print(f"\n{sep}")
    print("  KALMAN FILTER SENSOR FUSION  -  SIMULATION SUMMARY")
    print(sep)
    print(f"  Duration          : {results.timesteps[-1]:.1f} s")
    print(f"  Simulation rate   : {1.0/dt:.0f} Hz")
    print(f"  Total steps       : {len(results.logs)}")
    print(f"  GPS fixes         : {sum(1 for s in results.logs if s.gps_fix is not None)}")
    print(sep)
    print("  RMS error evaluated at ALL 1001 timesteps (100 Hz)")
    print(f"  GPS-only baseline : {results.rms_gps:.3f} m  (last fix held constant)")
    print(f"  Kalman estimate   : {results.rms_kalman:.3f} m")
    print(f"  Improvement       : {results.improvement_pct:.1f} %")
    print(sep)

    # Phase breakdown
    phases = [
        ("Acceleration  (0-3 s)",  0.0,  3.0),
        ("Cruise        (3-7 s)",  3.0,  7.0),
        ("Deceleration  (7-10 s)", 7.0, 10.0),
    ]
    print("  Kalman RMS per driving phase:")
    for name, t0, t1 in phases:
        subset = [s for s in results.logs if t0 <= s.time < t1]
        if subset:
            rms = np.sqrt(np.mean([s.kf_error**2 for s in subset]))
            print(f"    {name} : {rms:.3f} m")
    print(sep + "\n")


def save_csv(results) -> None:
    """Write per-step log to a CSV file."""
    import csv
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s", "true_position_m", "true_velocity_ms",
            "true_accel_ms2", "accel_reading_ms2",
            "gps_fix_m", "gps_held_m",
            "kf_position_m", "kf_velocity_ms", "kf_pos_std_m",
            "kf_error_m", "gps_error_m",
        ])
        for s in results.logs:
            writer.writerow([
                f"{s.time:.4f}",
                f"{s.true_position:.4f}",
                f"{s.true_velocity:.4f}",
                f"{s.true_acceleration:.4f}",
                f"{s.accel_reading:.4f}",
                f"{s.gps_fix:.4f}" if s.gps_fix is not None else "",
                f"{s.gps_held:.4f}",
                f"{s.kf_position:.4f}",
                f"{s.kf_velocity:.4f}",
                f"{s.kf_pos_std:.4f}",
                f"{s.kf_error:.4f}",
                f"{s.gps_error:.4f}",
            ])
    print(f"  Log  saved -> {LOG_PATH}")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nInitialising simulation ...")
    sim = Simulation(**SIM_PARAMS)

    n_steps = len(sim.timesteps)
    print(f"Running {n_steps} steps at {1.0/sim.dt:.0f} Hz ...")
    results = sim.run()

    print("Simulation complete. Generating outputs ...")
    print_summary(results)
    save_csv(results)
    plot_results(results, save_path=PLOT_PATH)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
