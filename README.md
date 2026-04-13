# polypath-motion

> A calculus-based trajectory planner for autonomous vehicles, built as an academic project exploring how polynomial mathematics drives real-world motion planning.

---

## Overview

This project implements a **quintic (5th-degree) polynomial trajectory planner** that generates smooth, dynamically feasible paths for an autonomous vehicle navigating a procedurally generated road with randomized obstacles.

The core idea: by fitting 5th-degree polynomials to motion states (position, velocity, acceleration), the planner produces paths with **continuous derivatives up to jerk** — a key requirement for passenger comfort and vehicle stability.

---

## How It Works

### 1. Quintic Polynomial Engine
Each trajectory segment is described by two independent 1D quintic polynomials — one for `x`, one for `y`. Given boundary conditions (start/end position, velocity, and acceleration), the 6 coefficients are solved via a linear system:

$$s(t) = a_0 + a_1t + a_2t^2 + a_3t^3 + a_4t^4 + a_5t^5$$

Derivatives are computed analytically to yield velocity $v(t)$, acceleration $a(t)$, and jerk $j(t)$.

### 2. Candidate Sampling
The planner samples a grid of:
- **Lateral offsets** — 15 positions across the lane width
- **Maneuver durations** — T ∈ [4s, 10s]

For each combination, a quintic trajectory pair is generated and evaluated.

### 3. Cost Function
Safe (collision-free) trajectories are ranked by:

$$J = T + w_j \cdot \sum j^2 + w_d \cdot |d_{lat}|$$

| Term | Weight | Penalizes |
|---|---|---|
| Duration $T$ | 1.0 | Slow maneuvers |
| Jerk $\sum j^2$ | 0.01 | Discomfort |
| Lateral deviation $\|d_{lat}\|$ | 2.0 | Lane wandering |

### 4. Collision & Boundary Checking
Each candidate path is checked against:
- Rectangular obstacle bounding boxes (with 1.3m safety margin)
- Road boundary limits (sine-wave road, 14m wide)

Only fully safe paths are considered for selection.

---

## Visualizations

**Plot 1 — Kinematic Profile**
Position, velocity, and acceleration time-series for a reference "stop" maneuver — demonstrating smooth boundary satisfaction.

**Plot 2 — Trajectory Comparison**
Side-by-side view of the straight-line unoptimized path vs. the cost-optimized trajectory, with road boundaries and obstacles shown.

**Plot 3 — Animated Simulation**
A real-time animation of the vehicle following the optimized path, with a camera that tracks the vehicle.

---

## Tech Stack

- **Python 3**
- `numpy` — linear algebra, polynomial evaluation
- `matplotlib` — static plots and animation

---

## Running It

```bash
pip install numpy matplotlib
python mathp.py
```

Each run randomizes the road curvature and obstacle placement to test planner robustness.

---

**Course project:** Calculus-Based Trajectory Planning For Autonomous Vehicles  
Demonstrates applied use of derivatives, integrals, and polynomial modeling in the context of autonomous vehicle motion planning.
