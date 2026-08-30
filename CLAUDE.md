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

### 2+1D Model (`quantum_hex_2d.py`)

Extension to 2 spatial dimensions with hexagonal lattice:
- **7 directions**: 6 diagonal (0°, 60°, 120°, 180°, 240°, 300°) + 1 straight
- All edge lengths = 1
- Δt = 0.5 for diagonal moves, Δt = 1.0 for straight move
- **14×14 transfer matrix** (7 amplitudes each for current and previous time)

---

### 2+1D Turning-Phase Model (`quantum_hex_turning.py`) ← **current main model**

Path weight = **sum of turning angles**, scaled by a free parameter α:

```
w_path = prod_steps  a(|n|) · exp(i·α·60°·n)      n = signed 60°-units turned
```

State lives on **directed edges** (node + heading) — the same `amp[x,y,d]` as above.

- For every closed walk returning to the same directed edge, Σθ = 360°·w exactly
  (discrete Whitney turning-number theorem, verified for all walks up to L=9).
  Hence the raw angle sum gives every loop phase 1 — **α is what makes it physical**.
- Loop phase = exp(i·2πα·w): α is an **Aharonov–Bohm flux in heading space**,
  physical only mod 1. α = 1/2 → (−1)^w = spinor double cover.
- Coin operator `C = expm(i·ε·G_α)` with `G_α = e^{iπα/3}R + h.c.`, R = +60° rotation.
  **Exactly unitary** (2e-16) — fixes the |λ|>1 problem of the model above.

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
| `quantum_hex_turning.py` | **2+1D turning-phase model** (model + transfer matrix + simulation) |
| `quantum_hex_turning_figs.py` | Figures and numerical report for the turning-phase model |
| `quantum_hex_magnetic.py` | **Magnetic field** (Peierls phase) on the turning-phase model |
| `quantum_hex_magnetic_figs.py` | Figures and numerical report for the magnetic model |
| `quantum_fcc_holonomy.py` | **3D FCC holonomy** — does the turning-number theorem survive? |
| `quantum_hex_su3.py` | **SU(3) colour** — non-abelian links, Wilson loops |
| `quantum_qcd_3d_report.py` | Report + overview figure for the 3D-spin and QCD questions |
| `quantum_hex_su3_mc.py` | **SU(3) Wilson-action Monte Carlo** (dynamical gauge field) |
| `quantum_hex_su3_mc_figs.py` | Report + figure for the Monte Carlo |
| `quantum_fcc_3d.py` | **3+1D FCC walker with an SU(2) coin** (spin 0 vs spin 1/2) |
| `quantum_fcc_3d_figs.py` | Report + figure for the 3+1D walker |
| `quantum_two_particle.py` | **Two-particle Hilbert space** — exclusion, statistics, collisions |
| `quantum_two_particle_figs.py` | Report + figure for the two-particle work |

### Result Files

| File | Description |
|------|-------------|
| `RESULTS.md` | Results of the 1+1D models (original) |
| `RESULTS_1D_en.md` | English version — 1+1D results with corrected m ≈ 2ε |
| `RESULTS_2D_de.md` | German version — 2+1D hexagonal model |
| `RESULTS_2D_en.md` | English version — 2+1D results including wave packet simulation |
| `RESULT_Proper_Time_1D_en.md` | Proper time investigation — 1+1D equilateral triangular |
| `RESULTS_Turning_2D_de.md` | German — 2+1D turning-phase model (angle sum as path integral) |
| `RESULTS_Magnetic_2D_de.md` | German — magnetic field, chirality families, Onsager quantisation |
| `ROADMAP_QCD_3D_de.md` | German — why spin needs 3D, and what QCD needs (with measurements) |
| `RESULTS_SU3_MC_de.md` | German — SU(3) Wilson-action Monte Carlo: area law vs perimeter law |
| `RESULTS_FCC_3D_de.md` | German — 3+1D FCC walker: spin becomes rigid, Kramers automatic |
| `RESULTS_TwoParticle_de.md` | German — two-particle space: Pauli hole, bound states, colour singlet |

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

### Generated Figures (2+1D turning phase)

| File | Content |
|------|---------|
| `turning_geometry.png` | Headings, turn table n, coin operator arg C at α=0 and 1/2 |
| `turning_loops.png` | Winding numbers of all closed walks up to L=9, structure factor A(α) |
| `turning_spectrum.png` | E_m(α) numeric vs analytic, Kramers doubling, mass knob |
| `turning_dispersion.png` | 6 bands at α=0/0.25/0.5, cone zoom, isotropy, v_g(k) |
| `turning_motion.png` | CoM tracks in 11 directions, direction error, null test for δ |
| `turning_two_packets.png` | Crossing packets (fringes) vs head-on packets (none) |

---

## Confirmed Results — 2+1D Turning Phase

| Property | Value | Note |
|----------|-------|------|
| Whitney theorem | 0 violations | all closed walks up to L=9 |
| Unitarity | 2.2e-16 | `mode="unitary"`; norm 1.00000000 over 120 steps |
| Rest spectrum | E_m = −4ε·cos(π(α−m)/3) | numeric vs analytic: 2.4e-15 |
| Kramers doubling | 3 doublets at α=1/2 | vs 4 distinct levels at α=0, 6 generically |
| Mass knob | massive ∀α **except** α=1/2 | at α=1/2 the top pair opens a massless cone |
| Cone slope | **√3/2 = c/2 = 0.86602** | independent of ε (0.05, 0.1, 0.2) |
| Motion, direction error | ≤ 0.070° | 11 angles, \|k\|=0.8, 120 steps |
| Isotropy of \|v_g\| | 3.5 % at \|k\|=0.3, 10.1 % at \|k\|=0.8 | 6-fold lattice ripple |
| Constant δ per step | E → E − δ/Δt, trajectory identical | cannot move anything |
| Head-on spinor overlap | 3.7e-16 (exactly 0) | counter-propagating packets do not interfere |

### Generated Figures (2+1D magnetic)

| File | Content |
|------|---------|
| `magnetic_geometry.png` | Peierls construction, flux per plaquette, gauge invariance, loop families vs B |
| `magnetic_orbits.png` | Cyclotron orbits, mirror orbits at ±B, R vs k/B, chirality order parameter L_z |
| `magnetic_landau.png` | Time-domain spectrum, level spacings, Onsager line with γ = 1/2 vs 0 |
| `magnetic_standing_waves.png` | Standing ring waves, ⟨r⟩ vs n, coherent sum running the orbit |

---

## Confirmed Results — Magnetic Field

Peierls phase exp(i·(B/2)(x·dy − y·dx)); the midpoint rule is exact, and the sum
around any closed walk is B·(signed enclosed area) — the same shoelace quantity
that classifies the loop families.

| Property | Value | Note |
|----------|-------|------|
| Flux per plaquette | exact to 1e-9 | triangle 3√3/16, hexagon 6×, reversed −1× |
| Unitarity | 1.000000000000 | diagonal phase × shift × unitary coin |
| Gauge invariance of ρ | 1.7e-15 | symmetric vs Landau gauge |
| Cyclotron radius | R = k/B, ratio 0.94 | 6 % deficit from wave-packet spreading |
| Chirality separation | L_z flips sign exactly with B | −2.36 at B=+0.03, +2.36 at B=−0.03 |
| **Bare excited edge** | L_z = −1.7e-14 | populates all 6 bands at exactly 1/6 → exact cancellation |
| Onsager quantisation | A_k/(2πB) = n + γ, slope 0.998 | measured via time-domain spectroscopy |
| **Berry offset γ** | **0.518 at α=0, 0.020 at α=1/2** | γ=1/2 → 0 is the Dirac Berry phase π = the same (−1)^w spinor |
| Ring states | stationary to 0.2–1.0 % / 40 steps | ⟨r⟩ ∝ √(n+1/2) |
| Superposition of rings | closed orbit, returns at t=67.5 | direction emerges from Fourier superposition |

**Magnetic breakdown:** at ε=0.1 the effective mass is tiny (m\*=0.087) so
ħω_c exceeds the band gap and the packet splits.  All magnetic runs use ε=0.5,
where ω_c/gap = 0.05.

### Winding-number families (closed walks, no 180° reversal)

L=3,4,5 have **only** w=±1 (pure left/right triangles).  w=0 ("figure eights")
first appears at L=6 (24 of 74 walks).  w=±3 first appears at L=9.

---

## Development Branch

Current work runs on: `claude/hexagonal-lattice-wave-model-kjg71o`

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

- **3+1D extension / spin rigidity** (measured, see `ROADMAP_QCD_3D_de.md`): the 2D
  turning-number theorem does NOT survive in 3D.  On the FCC lattice the holonomy of a
  closed walk is a rotation about the initial heading (axis exact to 8e-16) but its angle
  is not quantised — 23 distinct values of φ/2π up to L=6, with cos(φ/2) ∈ {1/3, 1/√3,
  √(2/3), 2√2/3}, and holonomies about different axes do not commute (0.333).
  Since π₁(S¹)=ℤ but π₁(S²)=0, the continuous α of the 2D model has no 3D counterpart:
  SU(2) is simple, so only its centre ℤ₂ survives → **α ∈ {0, 1/2}, boson or fermion**.
  Spin is quantised in 3D precisely because SO(3) is doubly, not infinitely, connected.
  The α=1/2 point of the 2D model is therefore the shadow of the only nontrivial 3D option.
  **Built** in `quantum_fcc_3d.py` (see `RESULTS_FCC_3D_de.md`): the walker carries a
  spinor and turning rotates it by the SU(2) transport, so α is gone and only spin 0 or
  1/2 remain.  Coin C = expm(i·ε·G), G = Σ_{d'~d} |d'⟩⟨d| ⊗ Q(d→d') over the 60° edges
  of the cuboctahedron — exactly unitary (4e-16), and the 180° reversal never occurs.
  Kramers doubling is automatic and holds at **every** k (splitting ~1e-15), only for
  spin 1/2.  The belt trick is measured directly: the shortest closed heading loop
  (L=6) returns the spinor with −1, and +1 first appears at L=12.  m = 8ε (boson),
  4√3·ε (fermion), exactly linear.  Causality structural: max |v_g| = 1.7223 < c = √3.
  Limitation found: no particle–antiparticle symmetry, because the cuboctahedron
  contains triangles and is therefore not bipartite (the 2D 6-cycle was).

- **QCD** (measured, see `ROADMAP_QCD_3D_de.md`): the magnetic module is already a U(1)
  lattice gauge theory.  `quantum_hex_su3.py` promotes the link phase to SU(3):
  tr W is gauge invariant to 9e-16; U(1) fluxes add exactly (0.0e+00) while SU(3) needs a
  transporter to a common base point (9.7e-17 with, 0.065 without) and its plaquettes do
  not commute (0.82).  A **static** random background gives a perimeter law
  ⟨W⟩ = c^perimeter (matches to 3 digits) — i.e. **no confinement**.  An area law needs the
  links correlated by the Wilson action (Monte Carlo) — **done**, see
  `RESULTS_SU3_MC_de.md`: the triangular lattice is perfectly 3-colourable (every
  plaquette holds exactly one link of each direction), so the Metropolis sweep is fully
  vectorised; the plaquette matches the exact single-plaquette integral at all 7
  couplings within 0.98σ; and loops of equal area but different perimeter — (2,2) and
  (4,1) — give 0.044716 vs 0.044699 under the Wilson action (identical, area law) but
  0.35263 vs 0.27240 in a static ensemble at the same plaquette (perimeter law).
  In 2D this area law is exact and kinematic, so it validates the machinery rather than
  discovering confinement; the same measurement first says something nontrivial in 3+1D.
  Real QCD additionally needs 3+1D (asymptotic freedom), unquenching, and — the hardest
  barrier — real-time evolution has a sign problem.

- **Two-particle space** (built, see `RESULTS_TwoParticle_de.md`): for a non-interacting
  pair everything factorises into two evolved orbitals, so the pair correlation is exact
  and cheap.  Measured in the 3+1D FCC model: Ψ ≡ 0 for two fermions in the same orbital
  (1.9e-9); coincidence ratio exactly 0 (fermion) and 2 (boson) at full internal overlap;
  a Pauli hole along x₁=x₂ against a bosonic ridge.  Caveat: head-on packets have nearly
  orthogonal internal states (0.009 on FCC, 3.7e-16 in 2D) and show NO exchange effect.
  With a genuine contact interaction (full two-particle wavefunction, 1+1D): 1D contact
  scattering is integrable — only a phase shift, max|ΔP| = 2.4e-5 — and a contact acting
  on the full coincidence (x,d) leaves fermions **exactly** unchanged (6e-17) at any
  strength.  Bound states verified by exact diagonalisation in the relative coordinate
  (reduction checked against the full evolution to 5.2e-15): a site contact binds both
  species, a full-coincidence contact binds only bosons.  Gauge averaging keeps only the
  colour singlet (norm 1.000000) and annihilates the octet.

- **Turning phase — open points**: (a) exclusion/collisions need either a two-particle
  Hilbert space on pairs of directed edges (exact, expensive) or a nonlinear
  self-interaction (cheap, ad hoc); the linear model gives interference only.
  (b) Chirality families are combinatorially established but need a position-dependent
  Peierls factor A = (B/2)(−y, x) to separate them dynamically (cyclotron orbits).
  (c) No closed derivation yet for why the cone slope is exactly c/2.

- **Magnetic — open points**: (a) the 6 % deficit in R = k/B is only argued as a
  spreading effect, not derived.  (b) The Landau degeneracy per level was not
  measured; a magnetic unit cell at rational flux would give the full Hofstadter
  spectrum on this lattice.  (c) Chirality separation requires selecting one band —
  a bare excited edge populates all six equally and its circulation cancels exactly.
