# Kalman Filter Sensor Fusion

A Python simulation that fuses a noisy GPS position sensor and a high-rate accelerometer using a discrete-time Kalman filter to accurately estimate the position and velocity of a moving vehicle.

## Overview

The simulation models a vehicle that:
1. **Accelerates** from 0 to 9 m/s over 3 seconds (+3 m/s^2)
2. **Cruises** at 9 m/s for 4 seconds
3. **Decelerates** from 9 to 0 m/s over 3 seconds (-3 m/s^2)

Two sensors observe the vehicle:

| Sensor | Frequency | Noise (sigma) | Measures |
|---|---|---|---|
| GPS | 0.5 Hz | ±8 m | Position |
| Accelerometer | 100 Hz | ±0.8 m/s² | Acceleration |

The GPS-only baseline holds the last received fix constant between updates, which is how a position-only GPS would behave at 0.5 Hz. The Kalman filter predicts state with the accelerometer at every 100 Hz tick and corrects with GPS when a fix arrives, achieving **>90% RMS error reduction** compared to the GPS-only baseline.

## Project Structure

```
kalman-filter-sensor-fusion/
├── src/
│   ├── __init__.py
│   ├── trajectory.py       # True kinematic trajectory generator
│   ├── sensors.py          # GPS and accelerometer sensor models
│   ├── kalman_filter.py    # Kalman filter (predict + update + covariance)
│   ├── simulation.py       # Simulation engine + step logging
│   └── plotting.py         # Three-panel diagnostic figure
├── results/                # Generated plots and CSV logs (git-ignored)
├── main.py                 # Entry point
├── requirements.txt
└── README.md
```

## Kalman Filter Design

**State vector**: `x = [position, velocity]^T`

**State transition** (driven by accelerometer as control input):
```
x[k+1] = F·x[k] + B·u[k]

F = [[1, dt],    B = [[0.5·dt²],
     [0,  1]]         [dt     ]]
```

**Measurement model** (GPS observes position only):
```
z[k] = H·x[k] + v[k]
H = [[1, 0]]
```

**Process noise** derived from white-noise acceleration model:
```
Q = sigma_a² · [[dt⁴/4, dt³/2],
                [dt³/2, dt²  ]]
```

**Covariance update** uses the Joseph form for numerical stability:
```
P[k|k] = (I - K·H)·P[k|k-1]·(I - K·H)^T + K·R·K^T
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

This will:
- Run the full 10-second, 100 Hz simulation (1001 timesteps)
- Print a summary table with RMS errors and improvement %
- Save `results/simulation.png` (three-panel diagnostic plot)
- Save `results/simulation_log.csv` (per-step log of all signals)

## Example Output

```
--------------------------------------------------------------
  KALMAN FILTER SENSOR FUSION  -  SIMULATION SUMMARY
--------------------------------------------------------------
  Duration          : 10.0 s
  Simulation rate   : 100 Hz
  Total steps       : 1001
  GPS fixes         : 6
--------------------------------------------------------------
  RMS error evaluated at ALL 1001 timesteps (100 Hz)
  GPS-only baseline : 10.915 m  (last fix held constant)
  Kalman estimate   : 0.682 m
  Improvement       : 93.7 %
--------------------------------------------------------------
  Kalman RMS per driving phase:
    Acceleration  (0-3 s)  : 0.099 m
    Cruise        (3-7 s)  : 0.708 m
    Deceleration  (7-10 s) : 0.933 m
--------------------------------------------------------------
```

## Output Plot

The saved figure contains three panels:

1. **Position comparison** — True trajectory, GPS-held-constant baseline (sparse fixes + frozen between updates), and the smooth Kalman estimate with ±1σ uncertainty band.
2. **Estimation error** — Absolute position error for both GPS baseline and Kalman over the full 10 seconds, with RMS reference lines.
3. **Sensor noise** — Raw accelerometer readings vs true acceleration, illustrating the 100 Hz noise profile.

## Algorithm Details

The filter runs a **predict → (conditional) update** loop at every simulation tick:

```
for each step k at 100 Hz:
    accel_measurement  <- accelerometer.measure(true_accel[k])
    kf.predict(accel_measurement)        # propagate state + covariance

    if gps.is_update_step(k):            # every 200th step (0.5 Hz)
        gps_measurement <- gps.measure(true_position[k])
        kf.update(gps_measurement)       # correct with GPS fix
```

**Why fusion wins**: Between GPS updates the vehicle travels up to 18 m (9 m/s for 2 s). A GPS-only system is completely blind to this motion and accumulates large position errors. The Kalman filter integrates the high-frequency accelerometer to track this displacement, then uses the rare GPS fix to prevent long-term drift. The two sensors are *complementary*: GPS prevents bias accumulation; the accelerometer provides dense, low-noise dead-reckoning between fixes.

## Files Generated

| File | Description |
|---|---|
| `results/simulation.png` | Three-panel diagnostic plot |
| `results/simulation_log.csv` | Per-step log: true state, sensor readings, Kalman estimate, errors |
