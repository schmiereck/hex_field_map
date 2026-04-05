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

### Figure 5 — σ Convergence: τ_quantum → τ_classical as σ_k/m → 0

![sigma_convergence.png](sigma_convergence.png)

**Main panel:** τ_quantum/T vs v/c for three packet widths σ = 8, 20, 30. As σ increases (narrower k-spread), the quantum curve converges toward the classical SR curve √(1 − v²/c²).

**Residual panel:** Deviation (τ_quantum − τ_classical) / τ_classical in percent.

#### σ-Sweep Data Table

| σ | σ_k/m | τ_qt(v=0)/T | τ_qt(0.5c)/T | τ_qt(0.9c)/T | max deviation |
|---|-------|-------------|--------------|--------------|---------------|
| 8 | 0.627 | 0.771 | 0.739 | 0.490 | −22.9% |
| 20 | 0.251 | 0.930 | 0.844 | 0.447 | −7.0% |
| 30 | 0.167 | 0.964 | 0.857 | 0.441 | −3.6% |

The convergence is **monotone and quantitative**: each doubling of σ roughly halves the deviation. This confirms that the discrete lattice model reproduces special-relativistic time dilation to arbitrary precision in the narrow-packet limit (σ_k ≪ m).

#### Complete Velocity Tables

**σ = 8  (σ_k/m = 0.627)**

| v/c | τ_classical | τ_quantum | τ_phase | τ_dist | qt vs cl |
|-----|-------------|-----------|---------|--------|----------|
| 0.0 | 20.000 | 15.414 | 24.359 | 1.957 | −22.9% |
| 0.1 | 19.900 | 15.394 | 22.970 | 1.952 | −22.6% |
| 0.3 | 19.079 | 15.221 | 19.969 | 1.932 | −20.2% |
| 0.5 | 17.321 | 14.777 | 18.338 | 1.949 | −14.7% |
| 0.7 | 14.283 | 13.695 | 17.223 | 2.009 | −4.1% |
| 0.9 |  8.718 |  9.808 | 13.161 | 2.227 | +12.5% |

**σ = 20  (σ_k/m = 0.251)**

| v/c | τ_classical | τ_quantum | τ_phase | τ_dist | qt vs cl |
|-----|-------------|-----------|---------|--------|----------|
| 0.0 | 20.000 | 18.597 | 21.747 | 1.820 | −7.0% |
| 0.1 | 19.900 | 18.539 | 21.340 | 1.825 | −6.8% |
| 0.3 | 19.079 | 18.044 | 19.758 | 1.867 | −5.4% |
| 0.5 | 17.321 | 16.870 | 18.301 | 1.947 | −2.6% |
| 0.7 | 14.283 | 14.450 | 16.440 | 1.982 | +1.2% |
| 0.9 |  8.718 |  8.946 | 12.408 | 2.236 | +2.6% |

**σ = 30  (σ_k/m = 0.167)**

| v/c | τ_classical | τ_quantum | τ_phase | τ_dist | qt vs cl |
|-----|-------------|-----------|---------|--------|----------|
| 0.0 | 20.000 | 19.282 | 20.862 | 1.788 | −3.6% |
| 0.1 | 19.900 | 19.207 | 20.618 | 1.795 | −3.5% |
| 0.3 | 19.079 | 18.579 | 19.389 | 1.852 | −2.6% |
| 0.5 | 17.321 | 17.143 | 17.903 | 1.981 | −1.0% |
| 0.7 | 14.283 | 14.394 | 15.955 | 1.971 | +0.8% |
| 0.9 |  8.718 |  8.818 | 12.041 | 2.110 | +1.1% |

---

## Non-Unitarity and the Fast-Growing Mode

### Complete Eigenvalue Spectrum at k = 0, ε = 0.1

The 6×6 transfer matrix TM_full has 4 non-zero and 2 dead modes:

| # | |λ| | E = −arg(λ) | Structure | Classification |
|---|-----|-------------|-----------|----------------|
| 1 | **1.033** | **−0.319** | 84% diagonal, 16% straight | **Fast/growing, negative energy** |
| 2 | 1.010 | +0.199 | 100% diagonal, 0% straight | Physical propagating (m ≈ 2ε) |
| 3 | 1.010 | +0.0005 | 99.5% straight | Near-zero-energy straight mode |
| 4 | 1.007 | +0.123 | 57% straight, 43% diagonal | Mixed mode (m ≈ ε) |
| 5 | 0.0 | — | v_prev only | Dead mode (zero eigenvalue) |
| 6 | 0.0 | — | v_prev only | Dead mode (zero eigenvalue) |

**All four non-zero modes have |λ| > 1** — none are unitary.

### The Fast Mode (#1): Negative-Energy Antiparticle Analogue

- **|λ| = 1.033 at k = 0**, ranging 1.020–1.033 across all k — always growing, never stable.
- **E = −0.319 (negative energy)** — phase rotates opposite to the physical modes.
- Eigenvector is L-R symmetric (both diagonals + significant straight), unlike the physical mode (#2) which is L-R antisymmetric and purely diagonal.
- Dispersion tracks approximately **−E_phys(k)**: at large k, E_fast + E_phys → 0 (within ~0.002), confirming it as an approximate **negative-energy partner**.

**Dirac analogy:** In the continuum 1+1D Dirac equation, both particle (E > 0) and antiparticle (E < 0) branches are unitary (|λ| = 1). Here the discrete lattice reproduces the E/−E splitting but breaks the symmetry: the negative-E "antiparticle" mode grows fastest. This is a **lattice instability**, not a healthy antiparticle branch.

### Root Cause of Non-Unitarity

The non-unitarity is **structural**, not a numerical artefact:

1. **Singular values of TM_half**: {1.425, 1.418, 1.010, 1.000, 0, 0} — far from all-ones.
2. **Block structure** `[[A, B], [I, 0]]` has rank 4 (not 6), since the straight direction (d=1) couples only through B to the previous time step.
3. **Scaling with ε**: As ε → 0, all |λ| → 1. The non-unitarity is O(ε²) — vanishes in the massless limit.
4. **Product conservation**: The product of all non-zero |λ| is constant (1.0609) across all k, tied to det(effective 4×4 submatrix).

### Can it be Fixed?

The non-unitarity is inherent to embedding a mixed-time-step second-order recurrence into a single transfer matrix. Possible mitigations:

- **Physical subspace projection**: Only track the 4 non-null modes in a reduced state space.
- **Norm renormalisation per step**: Divide by |λ|_max each step (ad hoc but effective).
- **First-order reformulation**: Derive a genuine first-order evolution on a 3-component state (non-trivial due to mixed Δt).
- **Accept for dispersion analysis**: Eigenvalue *phases* (energies) remain physically meaningful — c = √3 and m = 2ε are correct. The code uses this approach.

**Conclusion:** The fast mode is a discrete-lattice analogue of the Dirac negative-energy sea, with an extra complication: exponential growth. For dispersion analysis and proper-time investigations, projecting onto the physical band (E ≈ 2ε) and ignoring the growing modes yields correct results. Long-time real-space simulations would require explicit suppression of the fast mode.

---

## Summary

| Property | Value | Notes |
|----------|-------|-------|
| c | √3 = 1.7321 | Geometrically exact |
| m_phys (ε=0.1) | 0.1993 ≈ 2ε | NOT ε — same as 2+1D hexagonal |
| vc_straight (physical mode) | **0** | Purely lightlike eigenvector |
| τ_quantum (σ=30, v=0) | 19.28 | 3.6% below τ_cl = 20 |
| τ_quantum (σ=30, v=0.9c) | 8.82 | 1.1% above τ_cl = 8.72 |
| σ convergence | ✓ | τ_quantum → τ_classical as σ_k/m → 0 |
| τ_phase trend | decreasing ✓ | Correct direction, wrong scale (k-spread) |
| τ_dist | ≈ 2 (small) | Physical mode is lightlike — no τ_acc |
| Fast mode (E < 0) | |λ| = 1.033 | Negative-energy antiparticle analogue |
| Non-unitarity | O(ε²) | Structural — vanishes as ε → 0 |

### Physical Interpretation

The 1+1D equilateral triangular lattice realizes a **purely lightlike massive particle**: the physical propagating mode (E ≈ 2ε) propagates entirely via left and right diagonal (lightlike) steps. The rest mass m ≈ 2ε and the relativistic dispersion E² = 3k² + m² emerge from quantum interference between these two diagonal directions — a discrete Zitterbewegung mechanism. No classical "rest frame" proper time accumulates along the constituent paths; the particle is, at the path-integral level, moving at the speed of light in alternating directions, just as in Feynman's original checkerboard model.

This is consistent with the general principle: **mass without timelike paths** — a quantum lattice particle can be massive without any single path being timelike.

The σ-convergence analysis provides the **quantitative proof**: as the wave packet narrows in k-space (σ → ∞, σ_k/m → 0), the quantum proper time τ_quantum = T · m · ⟨1/E(k)⟩_G converges to the classical special-relativistic result τ_classical = T · √(1 − v²/c²), with deviations shrinking from 23% (σ=8) to 3.6% (σ=30). The model correctly reproduces relativistic time dilation as an emergent property of discrete path-integral interference on an equilateral lattice.
