"""
Visualization module: three-panel diagnostic figure saved to disk.

Panel 1 - True trajectory vs GPS-held-constant baseline vs Kalman estimate
Panel 2 - Estimation error over time (GPS baseline vs Kalman)
Panel 3 - Accelerometer noise profile vs true acceleration
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .simulation import SimulationResults


def plot_results(results: SimulationResults, save_path: str = "results/simulation.png") -> None:
    logs = results.logs
    t    = results.timesteps

    true_pos   = results.true_positions
    gps_held   = results.gps_held
    kf_pos     = results.kf_positions
    kf_std     = results.kf_stds
    true_accel = np.array([s.true_acceleration for s in logs])
    accel_meas = results.accel_readings
    kf_err     = results.kf_errors
    gps_err    = results.gps_errors

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle("Kalman Filter Sensor Fusion - GPS + Accelerometer",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    # ---- Panel 1: Trajectory comparison -------------------------------------
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(t, true_pos, color="black", lw=2.5, label="True position", zorder=5)
    ax1.plot(t, gps_held, color="tomato", lw=1.2, alpha=0.75,
             label="GPS-only baseline (1 Hz, held constant)")
    ax1.scatter(results.gps_fix_times, results.gps_fix_values,
                c="tomato", s=80, zorder=6, label="GPS fixes (sigma=8 m, 0.5 Hz)")
    ax1.plot(t, kf_pos, color="royalblue", lw=1.8,
             label="Kalman estimate (100 Hz)", zorder=4)
    ax1.fill_between(t, kf_pos - kf_std, kf_pos + kf_std,
                     alpha=0.20, color="royalblue", label="+/-1 sigma uncertainty")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")
    ax1.set_title("Position: True vs GPS-Only Baseline vs Kalman Estimate")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ymin, ymax = ax1.get_ylim()
    for x0, x1, col, label in [(0, 3, "green", "Accel"),
                                (3, 7, "gold",  "Cruise"),
                                (7, 10, "salmon", "Decel")]:
        ax1.axvspan(x0, x1, alpha=0.06, color=col)
        ax1.text((x0+x1)/2, ymin + 0.05*(ymax-ymin), label,
                 fontsize=8, ha="center",
                 color={"Accel": "darkgreen", "Cruise": "goldenrod",
                        "Decel": "crimson"}[label])

    # ---- Panel 2: Error comparison ------------------------------------------
    ax2 = fig.add_subplot(gs[1])

    ax2.plot(t, gps_err, color="tomato",     lw=1.2, alpha=0.8,
             label="GPS-only error")
    ax2.plot(t, kf_err,  color="royalblue",  lw=1.5,
             label="Kalman error")
    ax2.axhline(results.rms_gps,    color="tomato",    lw=1.5, ls="--",
                label=f"GPS RMS = {results.rms_gps:.2f} m")
    ax2.axhline(results.rms_kalman, color="royalblue", lw=1.5, ls="--",
                label=f"Kalman RMS = {results.rms_kalman:.2f} m")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("|Error| (m)")
    ax2.set_title(f"Position Error  (improvement: {results.improvement_pct:.1f}%)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # ---- Panel 3: Accelerometer noise ---------------------------------------
    ax3 = fig.add_subplot(gs[2])

    ax3.plot(t, true_accel, color="black",     lw=2.0, label="True acceleration")
    ax3.plot(t, accel_meas, color="steelblue", lw=0.7, alpha=0.65,
             label="Accelerometer reading (noise std = 0.8 m/s^2, 100 Hz)")

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Acceleration (m/s^2)")
    ax3.set_title("Sensor Noise Comparison - Accelerometer")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {save_path}")
