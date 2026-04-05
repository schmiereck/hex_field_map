# Results: Proper Time Investigation — 1+1D Equilateral Triangular Lattice

**Source file:** [`quantum_proper_time.py`](quantum_proper_time.py)

**Parameters:** ε = 0.1,  T = 20,  σ = 8,  c = √3 ≈ 1.7321

---

## Lattice Structure

The **1+1D equilateral triangular lattice** has three move directions, all with edge length 1:

| Direction | Δx | Δt | Type | Proper time Δτ |
|-----------|-----|-----|------|----------------|
| d=0  left-diagonal  | −√3/2 | 0.5 | lightlike | 0 |
| d=1  straight-up    | 0     | 1.0 | timelike  | 1 |
| d=2  right-diagonal | +√3/2 | 0.5 | lightlike | 0 |

The speed of light follows geometrically: `c = (√3/2) / 0.5 = √3`.

Time evolution uses a **second-order recurrence** in half-steps:
- Diagonal moves (d=0, d=2) use `amp_prev` (1 half-step ago)
- Straight move (d=1) uses `amp_pprev` (2 half-steps ago)

Encoded in a **6×6 transfer matrix** `TM_full = TM_half²`:

```
TM_half = [[A, B],
           [I₃, 0]]

A[0,:] = exp(+ip) · C[0,:]     (left-diagonal,  p = k · √3/2)
A[2,:] = exp(−ip) · C[2,:]     (right-diagonal)
B[1,:] = C[1,:]                 (straight, from prev)
C[d,d'] = iε (d≠d'),  1 (d=d') (amplitude rule)
```

---

## Key Finding: Physical Eigenvector is Purely Lightlike

The most important result of this investigation:

> **The physical propagating eigenmode (E ≈ 2ε, k = 0) has zero straight component.**

At k = 0 and ε = 0.1, the eigenvector of TM_full at the propagating eigenvalue E = 0.1993 is:

```
vc_phys = [−0.707,  0.000,  +0.707]   (left, straight, right)
```

The straight (timelike) component is exactly **zero**. This mode is a pure left/right diagonal standing wave. A particle in this mode traverses only lightlike paths — every constituent path accumulates **zero proper time** in the path-integral sense.

The mass m ≈ 2ε arises entirely from the interference between left-diagonal and right-diagonal paths — a discrete lattice analogue of **Zitterbewegung**.

In contrast, the timelike mode (E ≈ ε, k = 0) has `|vc_straight| ≈ 0.75`, but this mode's energy decreases with k (unphysical dispersion) and is not the propagating band.

---

## Physical Mass: m ≈ 2ε

The physical mass is the eigenvalue of TM_full at k = 0 closest to 2ε:

```
m_phys = 0.19934   (ε = 0.1,  2ε = 0.200)
```

This is **not** m ≈ ε as listed in CLAUDE.md for 1+1D models — the correct scaling is m ≈ 2ε, identical to the 2+1D hexagonal model. The CLAUDE.md entry for the 1+1D triangular lattice should be updated accordingly.

---

## Proper Time Measurements

Four quantities were computed for each velocity:

| Quantity | Definition | Notes |
|----------|------------|-------|
| τ_classical | T · √(1 − v²/c²) | Special relativity prediction |
| τ_quantum | T · m · ⟨1/E(k)⟩_G | Gaussian-weighted QM prediction |
| τ_phase | T · \|d(phase)/dt\| / m | Phase slope at CoM; overestimates for broad σ_k |
| τ_dist | ⟨n_straight⟩ × 0.5 | Mean proper time from timelike mode (uniform init) |

### Velocity Sweep Results (ε = 0.1, T = 20, σ = 8)

| v/c | τ_classical | τ_quantum | τ_phase | τ_dist |
|-----|-------------|-----------|---------|--------|
| 0.0 | 20.000 | 15.414 | 24.359 | 1.964 |
| 0.1 | 19.900 | 15.394 | 22.970 | 1.966 |
| 0.3 | 19.079 | 15.221 | 20.202 | 1.979 |
| 0.5 | 17.321 | 14.777 | 18.933 | 2.013 |
| 0.7 | 14.283 | 13.695 | 18.085 | 2.103 |
| 0.9 |  8.718 |  9.808 | 15.524 | 2.496 |

**σ_k = 1/σ = 0.125,  σ_k/m = 0.627** — the packet is broad relative to the mass.

### τ_quantum: Correct QM Time Dilation ✓

The quantum proper time `τ_quantum = T · m · ⟨1/E(k)⟩_G` correctly decreases with velocity, confirming **relativistic time dilation** at the quantum level. For a narrow packet (σ → ∞), τ_quantum → τ_classical.

The deviation between τ_quantum and τ_classical is a **quantum correction** due to the k-spread of the Gaussian: high-k components (faster, higher energy) contribute shorter proper times, pulling ⟨1/E⟩ below 1/E(k₀).

### τ_phase: Overestimates Due to Broad Packet

The phase at the centre-of-mass evolves as:

```
d(phase)/dt|_{x=x_com} = k₀ · v_g_eff − ⟨E(k)⟩
```

For a narrow packet at momentum k₀: this equals −m/γ (the correct relativistic rest-frame rate). For a **broad packet** (σ_k ≳ m), the ⟨E(k)⟩ average exceeds E(k₀), causing τ_phase to overestimate τ. Despite the wrong absolute scale, τ_phase **decreases monotonically** with velocity, reproducing the qualitative time-dilation signature.

### τ_dist: Physical Eigenvector has Zero Proper Time

Since the physical propagating mode has `|vc_straight| = 0`, path-integral trajectories in that mode never trigger the straight step and accumulate τ_acc = 0. The small non-zero τ_dist ≈ 2 is contributed by the **timelike mode** (E ≈ ε), which is activated by the uniform initialisation used for this measurement. The timelike mode does show an increasing trend with v (slower particles stay more timelike), but the signal is mixed with the dominant lightlike propagation.

---

## Figures

### Figure 1 — Spacetime Spread and Worldlines

![worldlines_proper_time.png](worldlines_proper_time.png)

**Left:** v = 0 (at rest). The wave packet stays centred; the white dashed lines mark the light cone r = √3·t. The classical worldline (x = 0) is coloured by accumulated proper time τ_acc = t (maximum, since γ = 1).

**Right:** v = 0.5c. The packet drifts to the right. The worldline is coloured by τ_acc = t · √(1 − v²/c²) = 0.866·t, showing time dilation relative to coordinate time.

Both panels confirm that the probability density is strictly causal — no weight outside the light cone.

---

### Figure 2 — Phase Oscillation at Centre-of-Mass

![phase_vs_time.png](phase_vs_time.png)

Phase `arg(ψ_center(t))` (unwrapped, magenta) and its linear fit (dashed) for three velocities.

- **v = 0 (top):** Oscillation frequency ≈ 0.243 rad/step. For a pure eigenstate this would equal m = 0.1993; the excess is due to k-spread (⟨E⟩ > m).
- **v = 0.5c (middle):** Frequency decreases to ≈ 0.189 rad/step — time dilation is visible.
- **v = 0.9c (bottom):** Frequency further decreases to ≈ 0.155 rad/step.

The **monotone decrease of oscillation frequency with velocity** is the lattice signature of relativistic time dilation, even if the absolute scale is shifted by the k-spread correction.

---

### Figure 3 — Time-Dilation Curve τ/T vs v/c

![dilation_curve.png](dilation_curve.png)

**Main panel:** All four proper-time measures normalised to T = 20, plotted against v/c.

- Black solid line: SR classical prediction τ_cl / T = √(1 − v²/c²)
- Blue dashed with circles: τ_quantum / T — Gaussian k-average ⟨m/E⟩ (primary QM result)
- Red squares: τ_phase / T — phase-slope measurement (overestimates, but correct trend)
- Green triangles: τ_dist / T — timelike-mode simulation (small, ≈ 0.1 T)

**Residual panel:** (τ_phase − τ_quantum) / τ_quantum in percent. The ~32–58% overestimate of τ_phase is consistent with the measured ratio σ_k/m ≈ 0.63.

**Key result:** τ_quantum decreases correctly from 15.41 (v=0) to 9.81 (v=0.9c), reproducing the hyperbolic shape of the SR curve with a quantum offset set by the packet width.

---

### Figure 4 — Proper-Time Distributions

![proper_time_distribution.png](proper_time_distribution.png)

Histogram of P(τ_acc) at t = T from the straight-step accumulator (timelike mode, uniform initialisation) for v = 0, 0.3, 0.7, 0.9c.

All distributions are peaked near τ_acc ≈ 0–3, far from τ_classical (red line) and τ_quantum (orange dashed). This is the expected consequence of the physical eigenvector being purely lightlike: paths in the dominant propagating mode accumulate zero proper time. The small τ_dist ≈ 2 that appears comes from residual population in the timelike mode (E ≈ ε).

As velocity increases, the distribution shifts slightly upward (τ_dist: 1.96 → 2.50), reflecting that faster-moving packets mix more of the timelike component via the Gaussian k-spread.

---

## Summary

| Property | Value | Notes |
|----------|-------|-------|
| c | √3 = 1.7321 | Geometrically exact |
| m_phys (ε=0.1) | 0.1993 ≈ 2ε | NOT ε — same as 2+1D hexagonal |
| vc_straight (physical mode) | **0** | Purely lightlike eigenvector |
| τ_quantum (v=0) | 15.41 | T·m·⟨1/E⟩_G with σ=8 |
| τ_quantum (v=0.9c) | 9.81 | 36% less than v=0 → time dilation ✓ |
| τ_phase trend | decreasing ✓ | Correct direction, wrong scale (k-spread) |
| τ_dist | ≈ 2 (small) | Physical mode is lightlike — no τ_acc |

### Physical Interpretation

The 1+1D equilateral triangular lattice realizes a **purely lightlike massive particle**: the physical propagating mode (E ≈ 2ε) propagates entirely via left and right diagonal (lightlike) steps. The rest mass m ≈ 2ε and the relativistic dispersion E² = 3k² + m² emerge from quantum interference between these two diagonal directions — a discrete Zitterbewegung mechanism. No classical "rest frame" proper time accumulates along the constituent paths; the particle is, at the path-integral level, moving at the speed of light in alternating directions, just as in Feynman's original checkerboard model.

This is consistent with the general principle: **mass without timelike paths** — a quantum lattice particle can be massive without any single path being timelike.
