# CLAUDE.md — Project Overview: hex_field_map

## Project Goal

Implementation and analysis of **discrete quantum path-integral models** on various lattices.
Core question: Which lattice geometry reproduces relativistic quantum mechanics
(Dirac equation) with correct speed of light c, physical mass m ∝ ε, and causality?

### Amplitude Rule (all models)
- Same direction as previous step → factor **1**
- Direction change → factor **iε** (ε is the mass parameter)
- Total amplitude = product of all step factors
- Probability = |sum over all paths|²

---

## Models (developed chronologically)

### 1+1D Models (`quantum_path_integral.py`, `quantum_dispersion_phys.py`)

| Model | File | Moves | Edge length | c | m |
|-------|------|-------|-------------|---|---|
| Feynman Checkerboard | `quantum_path_integral.py` | (±1, +1) | √2 | 1 | ≈ ε |
| Square + Rest | `quantum_path_integral.py` | (−1,+1),(0,+1),(+1,+1) | √2,1,√2 | ≈1 | ≈ ε |
| Equilateral Triangle | `quantum_dispersion_phys.py` | (±√3/2,+½),(0,+1) | **1** (all equal) | **√3** | **≈ 2ε** |

The equilateral triangular lattice is the most physically natural: all edges equal length,
c = √3 geometrically exact, no scaling artefacts.

**Note on the triangular lattice mass:** The physical propagating eigenmode (E ≈ 2ε at k=0)
has **zero straight (timelike) component** — it is a purely lightlike standing wave.
Mass arises from interference between left and right diagonal paths (discrete Zitterbewegung).
The timelike mode sits at E ≈ ε but has unphysical (decreasing) dispersion.

### 2+1D Model (`quantum_hex_2d.py`) ← **current main model**

Extension to 2 spatial dimensions with hexagonal lattice:
- **7 directions**: 6 diagonal (0°, 60°, 120°, 180°, 240°, 300°) + 1 straight
- All edge lengths = 1
- Δt = 0.5 for diagonal moves, Δt = 1.0 for straight move
- **14×14 transfer matrix** (7 amplitudes each for current and previous time)

---

## Files

### Python Scripts

| File | Description |
|------|-------------|
| `quantum_path_integral.py` | 1+1D simulation (3 models), produces comparison plots |
| `quantum_dispersion.py` | Dispersion analysis (older version) |
| `quantum_dispersion_phys.py` | Physically correct dispersion analysis (1+1D) |
| `quantum_phase_patterns.py` | Phase pattern analysis |
| `quantum_lattice_viz.py` | Lattice visualisation |
| `quantum_hex_2d.py` | **2+1D hexagonal model** (main file) |
| `quantum_proper_time.py` | **Proper time investigation** (1+1D equilateral triangular) |

### Result Files

| File | Description |
|------|-------------|
| `RESULTS.md` | Results of the 1+1D models (original) |
| `RESULTS_1D_en.md` | English version — 1+1D results with corrected m ≈ 2ε |
| `RESULTS_2D_de.md` | German version — 2+1D hexagonal model |
| `RESULTS_2D_en.md` | English version — 2+1D results including wave packet simulation |
| `RESULT_Proper_Time_1D_en.md` | Proper time investigation — 1+1D equilateral triangular |

### Generated Figures (2+1D)

| File | Content |
|------|---------|
| `lattice_geometry_2d.png` | Lattice geometry, edge lengths |
| `spacetime_spread_2d.png` | \|ψ(x,y,t)\|² with light cone r=√3·t |
| `dispersion_relation_2d.png` | E(k) + 2D heatmap |
| `group_velocity_2d.png` | Group velocity |
| `epsilon_sweep_2d.png` | m(ε) dependence |

### Generated Figures (1+1D proper time)

| File | Content |
|------|---------|
| `worldlines_proper_time.png` | Spacetime heatmap with worldlines coloured by τ |
| `phase_vs_time.png` | Phase oscillation at CoM for v=0, 0.5c, 0.9c |
| `dilation_curve.png` | τ/T vs v/c comparing all methods |
| `proper_time_distribution.png` | P(τ_acc) histograms for selected velocities |

---

## Core Physics: 2+1D Hexagonal Model

### Transfer Matrix (`TM14_half`)
```
M_half = [[A, B],   (14×14)
          [I7, 0]]

A[d,d'] = exp(i·kx·Δx[d] + i·ky·Δy[d]) · C[d,d']   (diagonal moves)
B[d,6]  = exp(i·kx·Δx[6] + ...) · C[d,6]              (straight move)
C[d,d'] = iε if d≠d', else 1                           (amplitude rule)

M_full = M_half @ M_half   (one full time step)
```

### Key Functions
- `simulate_hex_2d(T, eps)` — time evolution via recurrence on (Nx,Ny,7) array
- `TM14_half(kx, ky, eps)` — single k-point
- `TM14_full_batch(kx_arr, ky_arr, eps)` — batched (n_k,n_k,14,14)
- `fit_rel_2d_direct(eps)` — measures c=√3 (geometrically) and m=arctan(2ε/(1−ε²))
- `physical_band_2d(...)` — selects physical band via E_ref selector

### Physical Mass
The propagating band starts at k=0 at the **5-fold degenerate eigenvalue**:
```
m_phys = arctan(2ε / (1−ε²))  ≈  2ε   (for small ε)
```
Not the single eigenvalue at ε (that is a non-propagating mode).

---

## Confirmed Results

### 2+1D Hexagonal

| Property | Value | Note |
|----------|-------|------|
| c | √3 = 1.7321 | Geometrically exact, confirmed by simulation |
| m(ε=0.1) | 0.1993 ≈ 2ε | 5-fold degenerate k=0 eigenvalue |
| Isotropy error | 0.0000 | At \|k\| ≤ 0.4, 6-fold symmetry |
| max\|v_g\| | 1.88 ≈ c | Minor lattice artefact at zone boundary |
| Causality | strict | Light cone r=√3·t respected |

### 1+1D Equilateral Triangular (proper time investigation)

| Property | Value | Note |
|----------|-------|------|
| c | √3 = 1.7321 | Geometrically exact |
| m(ε=0.1) | 0.1993 ≈ **2ε** | Physical eigenvector: purely lightlike (vc_straight = 0) |
| τ_quantum (v=0) | 15.41 | T·m·⟨1/E(k)⟩_G with σ=8, T=20 |
| τ_quantum (v=0.9c) | 9.81 | 36% less → time dilation confirmed |
| Causality | strict | Light cone r=√3·t respected |

---

## Development Branch

Current work runs on: `claude/quantum-path-integral-simulator-GZNoz`

---

## Notes for Future Development

- **Band tracking**: The 5-fold degenerate band at m=2ε splits into sub-bands for k>0.
  The isotropic sub-band is the physically relevant one. Best tracking:
  `E_guide = 0.5*(E_ref(m=2ε) + E_prev)` with filter `|vg| < 1.5c`.

- **Isotropy**: Exact only at |k| ≤ 0.4; lattice corrections for larger k are normal.

- **Proper time**: The physical propagating mode of the EQ triangular lattice is purely
  lightlike (vc_straight = 0). Mass arises from diagonal interference. Measured proper time
  per path is zero; the quantum average τ_quantum = T·m·⟨1/E(k)⟩_G gives the correct
  relativistic time dilation.

- **Non-unitarity**: All non-zero eigenmodes of TM_full have |λ| > 1. The fastest-growing
  mode (|λ|=1.033, E=−0.319 at k=0) is a negative-energy antiparticle analogue.
  Non-unitarity is O(ε²) and structural (rank-deficient block matrix). Eigenvalue
  *phases* (energies) remain correct; only amplitudes grow. Project onto physical
  subspace or renormalise per step for long-time simulations.

- **3+1D extension**: The next logical step would be a 3+1D face-centred cubic (FCC)
  lattice with an analogous amplitude rule.
