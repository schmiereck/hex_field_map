# CLAUDE.md — Project Overview: hex_field_map

## Project Goal

Implementation and analysis of **discrete quantum path-integral models** on various lattices.
Core question: Which lattice geometry reproduces relativistic quantum mechanics
(Dirac equation) with correct speed of light c, physical mass m ∝ ε, and causality?

### Papers

| Paper | Files | Status |
|-------|-------|--------|
| 1 — Relativistic QM from Lattice Geometry | `paper.tex/pdf`, `paper_de.tex/pdf`, `README1_lattice_geometry.md`, `PAPER_SUMMARY.md` | published (tag `v1.0.0`, DOI 10.5281/zenodo.22178585; concept DOI …584; see `CITATION.cff`) |
| 2 — Internal Structure of the Lattice Path Integral | `paper2_internal_structure.tex/pdf`, `README2_internal_structure.md` | published (tag `v2.0.0`, DOI 10.5281/zenodo.23888331; Zenodo metadata in `.zenodo.json` — update title/version/description before each new release) |
| 3 — Discrete states / gravity as computation time | `ROADMAP_Paper3_de.md`, `SOURCE_Paper3_Gemini_Thread_de.md`, `RESULTS_Gravity_Lapse_de.md`, `RESULTS_PhaseBit_S1_de.md`, `RESULTS_PhaseBit_S2_de.md`, `RESULTS_PhaseBit_S3_de.md`, `RESULTS_PhaseBit_S4_de.md`, `RESULTS_PhaseBit_S4_Open_de.md`, `RESULTS_PhaseBit_S4_Open2_de.md`, `RESULTS_PhaseBit_S4_Open3_de.md`, `RESULTS_PhaseBit_S4_Open4_de.md`, `RESULTS_PhaseBit_S4_Open5_de.md`, `RESULTS_PhaseBit_S4_Open6_de.md`, `RESULTS_H1_Antimatter_ProperTime_de.md`, `RESULTS_H2_BudgetRule_de.md`, `RESULTS_H2b_Compensation_Beta_de.md`, `RESULTS_G0i_Gravitomagnetism_de.md`, `RESULTS_G0i_Causal_de.md`, `RESULTS_Exp3_Isotropy_de.md`, `RESULTS_Gravity3D_de.md`, `RESULTS_Observer_Perihelion_LT_de.md`, `RESULTS_GravAtom_de.md`, `RESULTS_GravSpinOrbit_de.md` | in preparation (main focus: spin vortices from phase bits; S1–S4 + six rounds of S4 open points done, gravity 1a–1d done) |

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

Layout: `src/` (all `.py`), `figures/` (`.png/.gif/.mp4`), `data/` (`.npz`),
`docs/` (`RESULTS_*.md`, notes, paper READMEs, HTML), `papers/` (`.tex/.pdf`).
Scripts write output via `src/paths.py` (`fig_path`, `data_path`, `docs_path`) —
use these for every new output file. Papers use `\graphicspath{{../figures/}}`.
The tables below list bare file names; find them in the directory above.

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
| `quantum_scatter_2d.py` | **2+1D deflection scattering** (relative-coordinate reduction) |
| `quantum_scatter_2d_figs.py` | Report + figure for the scattering work |
| `quantum_helicity_test.py` | **Does spin lift the exchange ceiling?** (overlap diagnostics) |
| `quantum_3d.py` | **3+1D tetrahedral-octahedral (cuboctahedron) lattice** (Paper 2) |
| `quantum_spin_structure.py` | Internal / spin structure of the physical band (Paper 2) |
| `quantum_spin_cluster.py` | Spin-cluster analysis of the degenerate k=0 band |
| `quantum_mzero_check.py` | Chirality and m=0 exclusion check |
| `experiment_double_slit.py` | Double slit on the hex lattice |
| `experiment_double_slit_square.py` | Double slit, square-lattice comparison |
| `experiment_farfield_slit.py` | Double slit, far-field (Fraunhofer) |
| `experiment_box_3d.py` | Box states in 3D (symmetry breaking by confinement) |
| `quantum_gravity_lapse.py` | **Paper 3, Exp. 1a** — lapse on the internal clock eps(r)=eps0·N(r), free fall, EP ratio R, coupling kappa(k) |
| `quantum_gravity_lapse_figs.py` | Figure for Exp. 1a (`gravity_lapse.png`) |
| `phase_bit_automaton.py` | **Paper 3, Exp. S1** — integer phase-bit automaton N[x,y,d,p,T]; Z12 double cover, spinor sign, vortex modes |
| `phase_bit_automaton_figs.py` | Figure for Exp. S1 (`phase_bit_automaton.png`) |
| `phase_bit_vortex_dynamics.py` | **Paper 3, Exp. S2** — unitary Z12 phase-bit walker, superselection, mass by chiral reversal, Zitterbewegung |
| `phase_bit_vortex_dynamics_figs.py` | Figure for Exp. S2 (`phase_bit_vortex_dynamics.png`) |
| `phase_bit_fcc_2O.py` | **Paper 3, Exp. S3** — FCC walker with a finite 2O (binary octahedral) spin register |
| `phase_bit_fcc_2O_figs.py` | Figure for Exp. S3 (`phase_bit_fcc_2O.png`) |
| `phase_bit_mass_s4.py` | **Paper 3, Exp. S4** — mass-term scan at the 3D Dirac point, mu·Pi mass, 3+1D dynamics |
| `phase_bit_mass_s4_figs.py` | Figure for Exp. S4 (`phase_bit_mass_s4.png`) |
| `phase_bit_s4_open.py` | **Paper 3, S4 open points** — T vs PT, exact lattice chirality H, 2D Kramers protection, background compensation, exact k-patch dynamics (`Patch` class) |
| `phase_bit_s4_open_figs.py` | Figure for the S4 open points (`phase_bit_s4_open.png`) |
| `phase_bit_s4_open2.py` | **Paper 3, S4 open points round 2** — 2O + lambda* W90 (no background, E<->-E), theta with H-breaking mass, mass domain walls in a z-slab (`slab_T`), ZB amplitude via band projectors |
| `phase_bit_s4_open2_figs.py` | Figure for round 2 (`phase_bit_s4_open2.png`) |
| `phase_bit_s4_open3.py` | **Paper 3, round 3 / Exp. 1b** — gravity on 2O* (vacuum counterterm, clock vs full lapse on an x-chain), Z2 wall test (parts `B2`), theta in a B field (`C2`), Landau levels; run parts via `python phase_bit_s4_open3.py [main] [B2] [C2]` |
| `phase_bit_s4_open3_figs.py` | Figure for round 3 (`phase_bit_s4_open3.png`) |
| `phase_bit_s4_open4.py` | **Paper 3, round 4 / Exp. 1c** — local full lapse by a wait buffer (`f_dwell`, `dwell_k`), spatial metric, counterterm offset scan, E∥B pumping in time; run parts via `python phase_bit_s4_open4.py [A] [B] [C]` |
| `phase_bit_s4_open4_figs.py` | Figure for round 4 (`phase_bit_s4_open4.png`) |
| `phase_bit_s4_open5.py` | **Paper 3, round 5 / Exp. 1d** — strong/time-dependent buffer lapse, horizon approach, lapse sourced by the local load (two-packet Hartree, `chain_T`, `Buffered`), anisotropic metric via heading-dependent coin couplings, GW interferometer, periodic magnetic chain (det winding); run parts via `python phase_bit_s4_open5.py [A] [B] [C] [D]` |
| `phase_bit_s4_open5_figs.py` | Figure for round 5 (`phase_bit_s4_open5.png`) |
| `phase_bit_s4_open6.py` | **Paper 3, round 6** — GR source 2e − e_M (`make_runner`: two-packet Hartree, Green or local wave equation for Phi), causal gravity, lattice mirror for massive particles, band structure of the helicity sector; run parts via `python phase_bit_s4_open6.py [A] [B] [C] [D]` |
| `phase_bit_s4_open6_figs.py` | Figure for round 6 (`phase_bit_s4_open6.png`) |
| `phase_bit_s4_open7.py` | **Paper 3, hypothesis H1 + proper time** — loads of particle/antiparticle, pair runaway (single-particle) vs hole picture, mass load = m dtau/dt (`Walker` class); run parts via `python phase_bit_s4_open7.py [A] [B]` |
| `phase_bit_s4_open7_figs.py` | Figure for H1 / proper time (`phase_bit_s4_open7.png`) |
| `phase_bit_s4_open8.py` | **Paper 3, hypothesis H2** — one budget rule per node: wait gate vs stay gate (`Scheme` class, Bloch and chain), no-go for wait buffers, gamma_PPN from the band, stay-gate background, x-chain deflection vs 1b; run parts via `python phase_bit_s4_open8.py [A] [B]` |
| `phase_bit_s4_open8_figs.py` | Figure for H2 (`phase_bit_s4_open8.png`) |
| `phase_bit_s4_open9.py` | **Paper 3, H2 continued** — lambda(N) compensation of the stay-gate background (`Xl(lam)`, `band_alpha`, sparse `shift_chain` with per-layer coins), x-chain deflection, beta_PPN from the composition of two loads (`U_compose`: linear / parallel / nested), source redshift; run parts via `python phase_bit_s4_open9.py [A] [B] [C]` |
| `phase_bit_s4_open9_figs.py` | Figure for H2b (`phase_bit_s4_open9.png`) |
| `phase_bit_s4_open10.py` | **Paper 3, g0i** — gravitomagnetism: heading-dependent stay gate (rejected) vs heading-independent drag hop b sin(k_y l)cos(k_z l)/l, two moving sheets on the x-chain (`Chain`, `make_packet`, force from d<k_x>/dt, ideal-GR packet semiclassics); run parts via `python phase_bit_s4_open10.py [A] [B]` |
| `phase_bit_s4_open10_figs.py` | Figure for g0i (`phase_bit_s4_open10.png`) |
| `phase_bit_s4_open11.py` | **Paper 3, causal g0i** — Phi and b as leapfrog wave equations (c_g = c' N0^2), `DynChain` (coin lookup table in N, fields updated every tick), variants instant / inst_nob / causal / mixed / nob, light-cone prediction |
| `phase_bit_s4_open11_figs.py` | Figure for causal g0i (`phase_bit_s4_open11.png`) |
| `phase_bit_exp3_isotropy.py` | **Paper 3, Exp. 3** — anisotropy power C_l (l = 4, 6, 8) of the cuboctahedral shells, of the walker's long-time direction distribution (`bz_velocities`, Hellmann-Feynman), smooth sources (Fibonacci x Gauss quadrature), 3D point source (`step3`, symmetrised over 24 rotations), front speed; run parts via `python phase_bit_exp3_isotropy.py [A] [B] [C] [D]` |
| `phase_bit_exp3_isotropy_figs.py` | Figure for Exp. 3 (`phase_bit_exp3_isotropy.png`) |
| `phase_bit_s4_open12.py` | **Paper 3, gravity in 3D** — 3D Poisson / causal wave equation for Phi (1/r, no secular growth), full 3D walker `Walker3D` (W or H2 gates, lambda(N) coins in N bins, periodic box, complex64), `packet3d` (band-projected massless packet), light deflection by a point mass; run parts via `python phase_bit_s4_open12.py [A] [B]` (B: ~45 min) |
| `phase_bit_s4_open12_figs.py` | Figure for 3D gravity (`phase_bit_s4_open12.png`) |
| `phase_bit_s4_open13.py` | **Paper 3, observer / perihelion / Lense-Thirring** — Michelson-Morley in the lattice (group velocities over 2000 directions), 3D walker force law (`packet3d_m`), classical orbits of the lattice metric (perihelion, Lense-Thirring nodes), gravitomagnetic dipole, local 3D drag hop (`drag3d`, brick wall of bond rotations), internal rotation generator `J_int` = (2i/pi) log R90; run parts via `python phase_bit_s4_open13.py [A] [B1] [B2] [C1] [C2]` |
| `phase_bit_s4_open13_figs.py` | Figure (`phase_bit_s4_open13.png`) |
| `phase_bit_s4_open14.py` | **Paper 3, gravitational atom** — 1D lattice atom on the x-chain (levels from the time signal, `spectrum_peaks` with Blackman-Harris) vs the effective Hamiltonian (`eff_levels_1d`), 3D radial effective atom (`eff_levels_radial`, Newtonian reference) for H2 / W / linear lapse; run parts via `python phase_bit_s4_open14.py [A] [B]` |
| `phase_bit_s4_open14_figs.py` | Figure (`phase_bit_s4_open14.png`) |
| `phase_bit_s4_open15.py` | **Paper 3, gravitational spin-orbit coupling** — 2D radial Dirac reference in the metric (`dirac_radial`, staggered grid, kappa = +-j, optional lattice Wilson term r_b), 2D gravitational atom in a slab of `Walker3D` (C4-projected 2p start with J_z = +-1/2, signal checkpoint every 1000 ticks), filter diagonalisation from the autocorrelation (`fdm`); run parts via `python phase_bit_s4_open15.py [A] [B variant spin] [C]` (B: 1.2-2.6 h per run) |
| `phase_bit_s4_open15_figs.py` | Figure (`phase_bit_s4_open15.png`) |
| `create_animation.py` | Hex lattice wave animation (`animation.gif/mp4`) |

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
| `RESULTS_Scatter_2D_de.md` | German — 2+1D deflection scattering, Mott interference, helicity ceiling |
| `RESULTS_Helicity_de.md` | German — spin does not lift the ceiling; the non-relativistic limit does |
| `RESULTS_3D_en.md` | English — 3+1D tetrahedral-octahedral lattice |
| `RESULTS_Spin_Structure.md` | Spin / internal structure of the physical band |
| `RESULTS_Experiments_en.md` | English — double slit and box states |
| `RESULTS_Farfield_en.md` | English — far-field double slit |
| `NOTES_Caching_de.md` | German — memoised walk enumeration, lattice caching assessment |
| `ROADMAP_Paper3_de.md` | German — Paper 3 roadmap: critical review of the Gemini thread, experiments 1–4 |
| `SOURCE_Paper3_Gemini_Thread_de.md` | German — unverified source idea (Gemini export), kept verbatim |
| `RESULTS_Gravity_Lapse_de.md` | German — Exp. 1a: internal-clock lapse = Nordström-type gravity, R = 2 |
| `RESULTS_PhaseBit_S1_de.md` | German — Exp. S1: spin vortices from phase bits (integer automaton) |
| `RESULTS_PhaseBit_S2_de.md` | German — Exp. S2: dynamics of spin vortices (massless/massive, Zitterbewegung) |
| `RESULTS_PhaseBit_S3_de.md` | German — Exp. S3: spin 1/2 in 3D from a finite 2O register; sign-convention issue in quantum_fcc_3d |
| `RESULTS_PhaseBit_S4_de.md` | German — Exp. S4: T-preserving mass for the 3D spin vortex (mu·Pi = lattice beta) |
| `RESULTS_PhaseBit_S4_Open_de.md` | German — S4 open points: Theta is PT, chirality H, 2D no-go, background compensation, Dirac-regime dynamics |
| `RESULTS_PhaseBit_S4_Open2_de.md` | German — round 2: 2O without background, theta observable, domain-wall Dirac cones, ZB amplitude |
| `RESULTS_PhaseBit_S4_Open3_de.md` | German — round 3: gravity (Exp. 1b) on the clean particle, Z2 wall test, gauge field and anomaly |
| `RESULTS_PhaseBit_S4_Open4_de.md` | German — round 4: local full lapse (wait buffer), spatial metric (Exp. 1c), origin of the counterterm, interior-node chiral branch |
| `RESULTS_PhaseBit_S4_Open5_de.md` | German — round 5: one buffer for all N, horizon = mirror, lapse from the local load (Exp. 1d), anisotropic metric / GW, Landau det winding |
| `RESULTS_H1_Antimatter_ProperTime_de.md` | German — H1: antimatter gravitates normally only in the hole picture; proper time = mass load / m |
| `RESULTS_H2_BudgetRule_de.md` | German — H2: wait buffers are always a pure lapse; wait + stay gate (clock 1 node, hop 2) gives A = 1/N, gamma_PPN = 1 |
| `RESULTS_H2b_Compensation_Beta_de.md` | German — H2b: lambda(N) removes the stay-gate background; beta_PPN = 1 from nested gates (N = N1 N2) |
| `RESULTS_G0i_Gravitomagnetism_de.md` | German — g0i: drag hop from the momentum current; parallel light beams do not attract, co-moving sheets obey 1/gamma^3 |
| `RESULTS_G0i_Causal_de.md` | German — causal g0i: Phi and b as wave equations; parallel light beams feel nothing at any time only if both travel together |
| `RESULTS_Exp3_Isotropy_de.md` | German — Exp. 3: shells never isotropic (C_4 = 3.2e-3), particle isotropic with C_4 ~ sigma^-4, front at 0.5 c inside the shell |
| `RESULTS_Gravity3D_de.md` | German — 3D: 1/r field (causal, no growth), light deflection H2/W = 2.07 in the full 3D walker; observer inside the lattice sees spheres via his own signals |
| `RESULTS_Observer_Perihelion_LT_de.md` | German — Michelson-Morley in the lattice ((ka)^2), Newton in the 3D walker, perihelion (2+2g-b)/3, Lense-Thirring and the gyroscope rule |
| `RESULTS_GravAtom_de.md` | German — gravitational atom: l-splitting = quantum perihelion (ratios -> 1/3, 7/6), 1D lattice atom validates the effective Hamiltonian |
| `RESULTS_GravSpinOrbit_de.md` | German — gravitational spin-orbit coupling: fine structure ~ 2 gamma + 1 (de Sitter), 2D atom in the 3D walker matches Dirac in the metric to 0.8-1.5 % |
| `RESULTS_PhaseBit_S4_Open6_de.md` | German — round 6: GR source T00 + T_ii from the load, causal gravity (local wave equation), mirror for massive particles, no global gap in the helicity sector |

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

All work is merged into `main` (default branch). Create feature branches from `main`.

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
  (L=6) returns the spinor with −1, and +1 first appears at L=12.  m = 8ε (boson).
  **Spin 1/2 is massless:** the lowest level −4√3·ε is 4-fold and splits linearly
  into two Kramers doublets ±(c/3)|k|, isotropic and ε-independent — a massless 3D
  Dirac cone (2D analogue: Kramers cone c/2 at α=1/2).  (Corrected Nov 2026: the
  earlier "m = 4√3ε, relativistic to 0.8–4.7 %" came from a sign error in
  su2_transport, now exp(−iθ n·σ/2).)  Causality structural: max |v_g| = 1.7224 < c.
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
  orthogonal internal states (exactly 0 on FCC spin 1/2 after the transport fix, 3.7e-16 in 2D) and show NO exchange effect.
  With a genuine contact interaction (full two-particle wavefunction, 1+1D): 1D contact
  scattering is integrable — only a phase shift, max|ΔP| = 2.4e-5 — and a contact acting
  on the full coincidence (x,d) leaves fermions **exactly** unchanged (6e-17) at any
  strength.  Bound states verified by exact diagonalisation in the relative coordinate
  (reduction checked against the full evolution to 5.2e-15): a site contact binds both
  species, a full-coincidence contact binds only bosons.  Gauge averaging keeps only the
  colour singlet (norm 1.000000) and annihilates the octet.

- **Deflection scattering in 2+1D** (built, see `RESULTS_Scatter_2D_de.md`): the
  relative-coordinate reduction at fixed total momentum is exact (0.000e+00 against the
  full two-particle evolution, with and without contact; [step, X] = 1e-17), which makes
  2D two-body scattering affordable at (N_sites x 36).  Deflection works — 6.2 % of the
  state is scattered with a forward/backward ratio of 8.0 and a six-fold lattice ray
  pattern — which 1+1D forbids.  Mott interference is present (fermion minimum at 90°,
  ~10x below the boson) but **capped by |⟨u(k)|u(−k)⟩|**: best contrast 0.752 against a
  ceiling of 0.794 at k=0.3, 0.401 against 0.516 at k=0.6.  The cause is structural —
  the band eigenvector is locked to k (helicity-like), so counter-propagating particles
  are partly distinguishable; at **α=1/2 the overlap is exactly 0 and no exchange
  interference exists at all**.  This same fact has now appeared four times (3.7e-16 in
  the 2D two-packet test, exactly 0 on FCC (was 0.009 before the transport fix), in step 3, and here) and is a central property of
  this model class.  **Followed up** in `quantum_helicity_test.py` (see
  `RESULTS_Helicity_de.md`): a momentum-independent spin does NOT lift the ceiling — in
  FCC both Kramers singular values are EXACTLY 0 (massless Dirac cone, opposite
  helicity; corrected Nov 2026, previously 0.416 at k=0.2) while the heading fidelity is
  finite, and spin 0 overlaps better (0.692 at k=0.2), so the spinor makes it worse.  The bottleneck is that the internal index IS the direction of motion.  What
  does recover the full Mott zero is the non-relativistic limit: at overlap 0.965 the
  contrast is 0.9937 and the fermionic cross section at 90° is 0.37 % of the bosonic one.
  The Dirac guess |⟨u(k)|u(−k)⟩| = m/E = 1/γ was tested and FAILS (ratio falls from 0.98
  to 0.18) — the lattice overlap decays faster.  Correction to the wording of step 4: the
  overlap is a scaling guide, not a strict cap (measured contrast/overlap 0.78 … 1.12).  Two analysis traps recorded: the contrast dephases with radius (a
  broad annulus gave +0.28 instead of +0.75), and the mirror partner of angular bin i is
  71−i, not 72−i (which faked a 29 % asymmetry where there is 8.4 %).

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

- **Gravity as computation time** (Paper 3, see `ROADMAP_Paper3_de.md`,
  `RESULTS_Gravity_Lapse_de.md`): Exp. 1a slows only the internal clock,
  eps(r) = eps0·N(r), with the shift untouched (exactly unitary).  Free fall is
  universal as eps→0 (EP violation O(eps²): R = 1.998 at 0.1, 1.917 at 0.7) but
  R = (m0/m*)/c² → **2** (α=0), 2.548 (α=1/4) instead of the GR value 1, because the
  turning band has m* = m0/(2c²).  Lattice vs semiclassics with the exact band agree
  to 1e-3.  Moving particles couple with kappa = eps·dE/deps/E ≈ 1.5–1.7/γ² →
  Nordström-type scalar gravity, no light bending.  Next: 1b (full lapse incl.
  propagation, expect Einstein-1911 half deflection), 1c (spatial metric), and find a
  band with R = 1.

- **Spin vortices from phase bits** (Paper 3 main focus, `RESULTS_PhaseBit_S1_de.md`):
  walker label (d in Z6, phase bit p in Z4, turn count T); a turn by n does
  d+=n, p+=n, T+=|n|; the state is a table of non-negative integer path counts,
  numbers enter only at readout A = sum N·eps^T·i^(a p + b T).  Integer vs float
  3e-11; (a,b)=(1,1) = graded alpha=3/2 ≡ alpha=1/2 (spectra 5e-15).  Exactly 12 of
  24 (d,p) labels reachable (d ≡ p mod 2) = Z12 double cover.  Closed walks: bit −1
  exactly for odd winding (integer match with enumeration up to L=9).  k=0 modes on
  Z12 are vortices e^{2 pi i j N/12}; odd j = half-quantum vortices (spin 1/2,
  Kramers doublets), selected by the spin readout; even j by the boson readout.
  Counts grow 1.79 bits/step → no finite-state exact automaton.  Next: S2 (moving
  vortex packet), S3 (3D with finite binary-octahedral spin register).
  **S2** (`RESULTS_PhaseBit_S2_de.md`): unitary walker directly on Z12 labels,
  C = expm(i(eps·G0 + mu·G3)), G0 = S+S⁻¹, G3 = i(S³−S⁻³) (180° reversal always counted
  as a left turn).  R6: N→N+6 commutes with everything → spin/boson superselection,
  drift 8e-16 even with random eps(r).  mu=0 equals alpha=1/2 (7e-15); the vortex
  doublet j=5/7 is a massless Kramers cone (slope c/2): a k=0 vortex becomes a ring at
  c/2.  mu>0 splits it by exactly 8·mu (j=7 up) → massive, rests, v < c; costs Kramers
  (T broken; 2+1D parity anomaly).  Zitterbewegung: pure j=7 with k0=0.3 oscillates
  j=7↔5 and in transverse CoM at 0.598 vs band splitting 0.595; damping = band
  separation (sigma/dv).
  **S3** (`RESULTS_PhaseBit_S3_de.md`): FCC transport replaced by 2O: for each 60°
  heading pair the unique closest rotation in O is a 90° quarter turn about a cube
  axis, Q = (1 + i n·σ)/√2.  Only 48 of 576 (d, g) labels reachable (Z4 bit per
  heading); integer table read out with 2O irreps = float walk (2e-15).  Holonomy
  quantised: heading triangle π, square 2π (minimal SU(2): 0.551, 1.360 rad); closed
  walks up to L=8 all −1.  Kramers at every k (6e-15), causal, rest levels ±2√2ε,
  ±4√2ε, and **exact E ↔ −E symmetry** (3.6e-15; minimal SU(2): 0.217) — the
  non-bipartite limitation was the continuous curvature, not the geometry.  Price:
  more anisotropy (17 % vs 8 % at k=0.3).
  **S4** (`RESULTS_PhaseBit_S4_de.md`): at the 4-fold Dirac point
  H_eff = E0 + Σ k_i Γ_i, {Γ_i,Γ_j} = 2(c/3)²δ (2e-15).  A term V opens a gap iff the
  traceless part of P†VP anticommutes with all Γ_i; Kramers-safe iff [V,Θ]=0
  (Θ = (1⊗iσ_y)K keeps k, so it is PT; true T = Π·Θ — see open points below).  Scan: all
  couplings with the spin transported only shift (∝1); with the spin NOT rotated
  (identity, i n·σ) → Π-even masses; i u_d·σ (K+K†) → Π-odd masses; on-site u·σ and
  i(K−K†)·u_d·σ split the cone and break T.  The 180° reversal without spin rotation
  IS the inversion Π (= lattice β): [Π,G]=0, gap exactly 4μ, Kramers at every k; with
  2O also E ↔ −E exact (4e-15).  The massive quartet = E_c(k) ± Δ(k), Δ Dirac with
  c/3 at small k, plus a common background −αk², αε ≈ 0.125 → clean Dirac regime
  μ ≪ ε.  3+1D (81³, ε=0.3, μ=0.15): rest packet stays (drift 1e-8), boosted |v| ≪ c
  (upper doublet runs against k at μ/ε = 0.5), Zitterbewegung in transverse CoM
  0.645 vs band splitting 0.653.
  **S4 open points** (`RESULTS_PhaseBit_S4_Open_de.md`): true time reversal T = Π·Θ
  (symmetric frame W = C^½ ph C^½: T W(k)* T⁻¹ = W(−k)⁻¹ to 8e-16).  μΠ keeps P, T, PT;
  the parity-odd mass = Π·(iH) breaks P and T (keeps PT) like ψ̄iγ5ψ; on-site helicity
  H = Σ|d⟩⟨d|⊗u_d·σ keeps T, breaks PT.  H commutes with the massless step (exact lattice
  chirality, spin transport keeps helicity), {Π,H}=0, exp(iθH/2) rotates μΠ into
  μ(cosθ Π + sinθ Π iH): spectrum θ-independent to 5e-15.  Weyl nodes per helicity
  sector: partner of opposite chirality at the same k=0 but another quasi-energy.
  2D: T = S³K, T² = R6 = −1; Dirac point = one Kramers pair at k=0 → no T-even mass
  (500 random T-even terms: splitting ≤ 3e-15; T-odd ≥ 6e-3).  Background: fixed sign
  at the extremal (lowest) Dirac level (αε 0.13 min SU(2), 0.196 2O); at the interior
  4-fold Dirac level (E/ε=+2.309, min SU(2)) exactly cancelled by λ*·ε·W90 (90° pairs,
  spin transport), λ* = −0.40827, ε-independent, all symmetries kept, gap 4μ.
  Exact k-patch method (`Patch`: step ψ̂ ← ph(k)·Cψ̂ on the packet's k support,
  envelope by IFFT) reproduces the 81³ runs (0.0508/−0.0875/0.6450) with < 1 GB.
  Dirac regime ε=0.3, μ=0.03, σ=40: compensated v = 0.124/0.231/0.369/0.425 vs Dirac
  0.135/0.250/0.400/0.512 at k = 0.025…0.2 (plain 0.105…0.308); upper doublet now moves
  along +k; ZB 0.1275 vs band 0.1248.
  **Round 2** (`RESULTS_PhaseBit_S4_Open2_de.md`): only one 2O-valued coupling keeps
  P, T, PT, H and E<->-E: W90 (90° pairs, quarter turn about the cube axis); it commutes
  with G_2O and just shifts its levels.  The Dirac-quartet background is second-order
  repulsion from the two 8-fold levels (velocity weights 2 and 6, zero to the opposite
  quartet) → cancels at lambda = −5 (second order), −4.979 exact for eps→0; residual
  O(eps²) (αε 2.3e-3 at eps=0.1), retuned lambda*(eps) = −4.556 (0.1), −3.204 (0.3).
  No doubling (quartets at G = ±4.21).  At eps=0.3 the massive band is Dirac to
  −0.3 %/−1.0 % (μ/ε=0.1) and −0.6 %/−2.0 % (μ/ε=0.5) at k=0.2/0.4, isotropic,
  max|v_g| 0.80–0.87.  Trap: at lambda=−2 the two quartets meet at E=0; only the
  flavour average is background-free (per flavour centre +0.021 at k=0.4).
  theta: with an H-breaking spin-identity mass (projects to Π) the gap is
  4μ|1+e^{iθ}| to 1e-14.  Domain walls (z-slab, `slab_T`, 120 layers, compensated
  SU(2), eps 0.3, mu 0.05): sign flip μΠ→−μΠ gives a massless 2D Dirac cone per wall
  (zero mode −6e-4, slope ≈ c/3), chiral rotation 0→π gives gapped wall states
  (≈0.07 vs bulk 0.2) → 2D spin vortex = boundary state of the 3D one (TI / axion θ=π).
  Floquet bands at |k|>1 overlap the gap energy in ~17 % of the BZ (no global gap).
  ZB amplitude with a well-defined helicity +1 spinor: sim vs band-projector theory
  rms 0.003 of 0.556, vs Dirac 4x4 0.010.
  **Round 3** (`RESULTS_PhaseBit_S4_Open3_de.md`, also Exp. 1b): a lapse scales the
  quasi-energy, so the Floquet offset of the Dirac quartet gravitates (2O*: E_c=+1.02 vs
  m=0.06 → 18x too strong) → vacuum counterterm −eps·lv·1 puts the Dirac point at E=0.
  Then R' = E0/(m* c'^2) (c'=c/3) = 0.997/0.997 (2O*, μ/ε=0.1), 0.979 (SU(2) comp),
  0.73/1.20 particle/antiparticle without compensation (background breaks the EP).
  x-chain (160 layers, Bloch y,z, N=1+gx, g=2e-3, μ=0.15, σ=8), a_x/(−g c'^2):
  clock lapse 0.89/0.65/0.32 (rest, k=0.3, 0.6) and ≈0 for massless beams (Nordström,
  κ ≈ (m/E)^2 now without the 1.5–1.7 prefactor of 1a); full lapse ½{N,H} (Floquet H
  smoothly projected to |E|<1.2) 0.86/0.85/0.79 and 0.73/0.88 for massless beams
  (Einstein 1911); semiclassics with the exact band agrees to 2–3 % (15 % for the
  narrow massless beam).  Z2 test (z-slab, random T-even term at one wall): single
  quartet → central branches touch at k=0 (6e-15), cone survives; two quartets
  (λ=−2) → avoided crossing, min spacing 4.3e-3/4.9e-3 (1.2μ/2.4μ).  Gauge field:
  [H, W_B] = 3.5e-16 for any Peierls field → theta unobservable even with B
  (3.4e-14; 0.26 with an H-breaking mass).  Landau levels (B=0.05): lowest node has
  one chiral branch per helicity, slopes −0.549/+0.590 (≈∓c/3); interior node branch
  not separable.  H is not the anomalous axial charge; node chirality is.
  **Round 4** (`RESULTS_PhaseBit_S4_Open4_de.md`, also Exp. 1c): strictly local full
  lapse = wait buffer per directed-edge state, tick U = R(a)·[S C ⊕ (−1)],
  R(a) = cos a + i sin a σ_x (active↔buffer, same site).  Exact: eigenphase
  f_a(θ) = θ/2 + arcsin(cos a sin(θ/2)) (vs 48x48: 4e-15), slope N = (1+cos a)/2 at 0,
  buffer weight 1−N, Kramers and E<->-E kept; only buffer phase π keeps E = 0 fixed
  (otherwise the vacuum energy depends on N and gravitates); range N ≥ 1/2 per buffer.
  x-chain (N0 = 0.75, g = 2e-3, 120 ticks): a/(−g N0² c'²) = 0.86/0.85/0.79/0.72/0.88 =
  the global full lapse of round 3 (Einstein 1911), semiclassics f_{a(x)}(θ(k)) 1–4 %.
  Exp. 1c: buffer lapse N/A plus mass μ·A (A = 1 − gx) -> H = (N/A)K + N M; ratio 1c/1b
  = 1.999/1.989 for light, 0.97 at rest, 1.24/1.58 moving vs GR 1 + (pc'/E)² = 1.25/1.57
  (worse with the lattice group velocity: 1.20/1.43).  Counterterm forced: residual
  offset δ gives R' = 1 ± δ/m for particle/antiparticle (lattice to 3 digits), and the
  vacuum must be the fixed point of the buffer rule; no lattice symmetry pins a single
  quartet (E<->-E swaps ±E_c; only the doubled λ = −2 point sits at 0).  E∥B in time
  (k_z ramp 2e-4/tick, helicity +1): both nodes isotropic Weyl with v = c'; LLL pumped
  dE/dk_z = −0.988/−0.992 c' (lower) vs +0.698/+0.851 c' (interior) at B = 0.05/0.035;
  interior < c' because its Weyl range is small (4-fold levels 0.57 away).  Long sweeps
  delocalise (open orbits), so the global connection of the branches is not traced.
  **Round 5** (`RESULTS_PhaseBit_S4_Open5_de.md`, also Exp. 1d): one buffer covers
  0 < N <= 1 (a > pi/2 allowed; round-4 claim N >= 1/2 was wrong); for ANY local wait
  scheme the lapse = active weight of the vacuum fixed vector (Hellmann-Feynman, random
  two-buffer schemes to 1e-3).  Horizon approach (N 0.9 -> 0.04): coordinate speed c'N,
  k = E'/(N c') blueshifts until the band maximum at k = pi/a, then the photon turns back
  at f_N(E_max) = E' (N = 0.089) -> mirror, not horizon; exact-band ray matches.  Uniform
  N(t) ramp: leakage ~T^-3, phase = sum N E dt.  Exp. 1d: load e(x) = Re psi^+ i(U psi - psi)/dt
  (local, vacuum 0) sources Phi'' = kappa e (1D Green fn, Hartree without self-term):
  sim vs packet semiclassics 0.4 %, third law to 1.9 % (= per-species EP factors 0.965/0.978),
  moving source pulls with E (3.02/5.69 vs E ratio 3.04/5.74), not GR's E(1+v^2).  Trap:
  N clipped at 1 in packet tails faked 7 %.  Heading-dependent waits only give the mean
  lapse 1/<1/N_d> (velocity numerator unchanged); heading-dependent coin weights
  s_d s_d' (s_d = 1 + u_d^T S u_d) give an exact ellipsoidal cone, response E_g -1.81,
  T_2g -1.11, any traceless g^-1 to first order, vacuum shift O(h^2).  GW interferometer:
  x-y phase = GR for k -> 0 (0.9 % at k = 0.03, error linear in k).  Periodic chain with one
  flux quantum (B = 2 pi/(Nx a^2)): det U(k_z) winding 0 at B = 0.1 and 0.05 -> no Floquet
  flow across the zone edge; tracing the single branch failed (dense LL spectrum).
  **Round 6** (`RESULTS_PhaseBit_S4_Open6_de.md`): source s = 2e - e_M with the mass load
  e_M = mu Re[i (U psi)^+ (dU/dmu psi)]/dt (local; = mu dE/dmu) gives T00 + T_ii: ratio to the
  load source 1.887/1.964 vs 1 + v^2 = 1.892/1.970, light 1.991/1.990.  Local wave equation
  for Phi (leapfrog, c_g = c', Courant 0.47): test particle starts falling only after D/c'
  (sharp-onset fit 245 vs 276; prediction a*P(|dx| < c't) from the initial densities rms 0.12);
  1D retarded field grows ~c't, so Phi is referenced at each particle.  Lattice mirror for
  mass: mu = 0.15 falls from rest, turns at -30.2 (pred. -30.5), bounces back to +18.8 of +20;
  mu = 0.03 has f_Nmin(E_max) < E' -> no turning, runs into the low-N region.  Helicity
  sector +1 (no field): 12 bands cover the whole quasi-energy circle, neighbouring bands
  touch (min gap <= 6e-3, also across the zone edge) -> only the net flow (0) is topological;
  a Weyl census with Berry-flux chirality failed (nodal lines, multi-fold points).
  **H1 + proper time** (`RESULTS_H1_Antimatter_ProperTime_de.md`): loads e, e_M of the
  lower band are negative (-|E|); passive fall normal (R' 0.997).  Pair particle +
  antiparticle, single-particle sources: both accelerate the same way (CoM -1.85e-4,
  momentum not conserved) -> excluded; hole picture (source = -load of the missing state)
  attracts exactly like two particles (+-1.858e-4).  Price: the sea load must be normal
  ordered (lattice form of the cosmological-constant problem).  Mass load Q_M/m = dtau/dt =
  N A m/sqrt(k^2 c'^2 + A^2 m^2) = N sqrt(1 - v_phys^2): 40 cases (N 0.4-1, A 0.9-1.1, k_y to
  0.6) match the lattice band to 0.9 % (moving), Dirac to 2 % at k <= 0.3; E = m dtau/dt + p.v,
  clock share 1/gamma^2; space-time integral of e_M = m tau = -S; equals Paper 1's
  tau_quantum = T m <1/E>.
  **H2** (`RESULTS_H2_BudgetRule_de.md`): any combination of wait buffers (also separate
  ones for shift and coin) is a pure lapse: both see the active weight of the joint fixed
  vector, 1/(1/N1 + 1/N2 - 1), so gamma = 0.  Stay gate R(b)[S C (+) -C] (hop refused, coin
  still applied, phase pi keeps E = 0) slows only the hop.  Rule "every process needs each
  node it involves to be free" (clock 1 node, hop 2): U = W(a)[H(b)(SC (+) -C) (+) -1] with
  N_w = N_h = N gives gap = N m, v = N^2 c' -> H = N^2 K + N M = Exp. 1c with A = 1/N,
  gamma = 1.0000 (band, k -> 0 extrapolated) for N = 0.3 ... 0.95.  Price: extra background
  -alpha k^2 with alpha = 0.245 N_h (1 - N_h) (mass-independent, O(dt); relative to the
  kinetic term ~ m dt or k dt).  x-chain (N0 0.75, g 2e-3, sigma 16): (a_H2/N0^2)/a_1b =
  1.986/1.922/1.860 for light k = 0.1/0.2/0.4 (gamma 0.99-1.03 after the background);
  'GR gamma = 1 plus measured alpha' matches the lattice semiclassics to <= 0.011 in all 8
  cases (incl. mu = 0.15, where alpha dominates: 0.717 at rest).  Traps: broad packets need
  the GR prediction with the same k distribution (rest is not 1 since A0 = 1/N0); FD
  curvature at small mu has a quartic error -> compare with exact Dirac E(k).
  **H2b** (`RESULTS_H2b_Compensation_Beta_de.md`): lambda (W90) changes only the background
  (Dirac point, gap, velocity fixed), so a node retunes lambda(N_h) (root of alpha(mu=0) = vacuum;
  d lambda ~ 3.3 (1 - N), quartic fit; no root below N_h ~ 0.4, lambda would cross -2).  Chain:
  light (a_H2c/N0^2)/a_1b = 2.005/1.998/2.007 (gamma 1.005/0.998/1.007), mu = 0.03 band gamma
  0.99/1.00; mu = 0.15 keeps the walker's own mass-dependent alpha (0.048) unscaled by N^2
  (0.940/1.077 explained exactly).  beta: ln N12 - ln N1 - ln N2 = 2(beta-1) ln N1 ln N2; one gate
  with linear Phi -> 1/2, parallel wait registers (waits add) -> 3/2, nested gates (N1 N2) -> 1.0000
  up to N12 = 0.49; nested stay+wait keeps A = 1/(N1 N2).  Rest load in H2 = N m (2e-5) -> with
  source E(1+v^2) and the product rule g00 = -1 + 2U* - 2U*^2 + 3 Phi1* - 2 Phi2* = GR 1PN in the
  conserved density rho*.  Coupling must be N = N0 exp(Phi), not N0 (1 + Phi).  g0i (gravitomagnetism)
  is missing: only scalars N, A.
  **g0i** (`RESULTS_G0i_Gravitomagnetism_de.md`): heading-dependent stay gate N_h(1 + u_d.w) gives
  a linear term but splits Kramers pairs (1.4e-4 at w = k = 0.02, k perp w) and shifts the vacuum
  (-0.74 w^2) -> rejected (the internal index IS the direction of motion).  Drag hop: every state,
  any heading, handed on along y: phase exp(-i dt b sin(k_y l)cos(k_z l)/l) per layer, l = 0.612;
  exact scalar (5e-15).  b = -4 c'^2 N0^4 (kappa/2)|x| * (k_s rho_s) (4 = GR / relativity principle,
  N0^4 = lattice vs physical coordinates around N0, A0 = 1/N0).  Two sheets, N0 0.6, sigma 32, 640
  layers, force F = d<k_x>/dt (ZB at 2E fitted), mean over both Kramers partners: lapse force vs
  ideal-GR packet semiclassics 0.996-1.002; with/without g0i: light parallel 0.019/0.053 (packet GR
  0.022/0.055, beams 0), antiparallel 1.99/1.96; massive 0.378/0.088 par., 1.626/1.921 antipar.;
  co-moving a(k,k)/a(0,0) = 0.669/0.319 vs packet GR 0.675/0.339 (1/gamma^3 0.655/0.292; without
  g0i 1.77/3.63).  Traps: single Kramers partner has a spin-dependent force (+-8 %); CoM fit biased
  by ZB at 2E; notch filter (U - e^{iE dt}) reweights the packet; packets < 4 sigma from the open
  chain end give up to 20 % too much force.
  **Causal g0i** (`RESULTS_G0i_Causal_de.md`): Phi_tt = c_g^2 (Phi_xx - kappa s/dx), same for b with
  kappa_b p/dx (the /dx makes the static limit equal the Green-function SUM; round 6 had it as
  lam2*(lap - kappa*LAT_A*q)), c_g = c' N0^2.  Two sheets at +-46, N0 0.8, sigma 16, gradient 3e-4,
  640 layers, 800 ticks: force = time-resolved static force x light-cone fraction P (rms 0.5-2.5 %);
  before arrival causal +0.01 vs mixed (b instantaneous) +1030 (1e-3 kappa) -> instantaneous shift
  beside retarded lapse = superluminal repulsion; parallel light at t = 400: -57 vs -1612 (Phi only).
  Traps: missing /dx (fields x0.612), too strong gradient (<k_x> changes 0.4 > k_y), spreading packet
  reaches the open end at 400 layers, constant instead of time-resolved static force in the prediction.
  **Exp. 3** (`RESULTS_Exp3_Isotropy_de.md`): C_l = 4pi/(2l+1) sum_m |a_lm|^2 (0 isotropic, 1 point);
  O_h sets are 3-designs, so l = 4, 6, 8 test.  Shell n nodes: C_4 = 3.220e-3, C_6 = 1.006e-3,
  C_8 = 1.19e-4 from n ~ 8 = cuboctahedron surface limit (face densities 1/sqrt3 {111}, 1/2 {100})
  -> never isotropic.  2O* walker: point source long-time limit (all 24 modes) C_4 ~ 9e-4; smooth
  source of width sigma: Dirac quartet C_4 = 1.06e-4/7.5e-6/4.7e-7/3.0e-8/2.1e-9 at sigma 2..32
  (~ sigma^-4), other 20 modes saturate at 2.3e-3 (warping of degenerate multiplets).  Front max v.n
  0.94/0.87/0.94 ([100]/[110]/[111]); 3D point source to n = 64 (131^3, complex64, 108 s): C_4 -> 8.9e-4,
  99 % front at r/t 0.89/0.86/0.92, weight on the geometric shell (graph dist >= 0.8 n) < 1e-3.
  Traps: offset k grid with a Gaussian weight picks the 8 [111] points near 0 (fake C_l ~ 0.3) -> isotropic
  quadrature; random sampling has a 1/N floor.
  **Gravity 3D** (`RESULTS_Gravity3D_de.md`): Poisson on 96^3 gives -kappa/(4 pi r) to 0-2 % (r 6-12,
  periodic images beyond); causal 3D wave equation: pulse at the front (switched-on source), then the
  static value, centre Phi constant (-0.0319 at t = 22 and 43) -> no 1D-type growth.  Full 3D walker
  112x88x80 (788k sites, 4 registers complex64, coins in 48 N bins), U = GM/sqrt(r^2+rc^2), GM 0.25,
  rc 2, N = exp(-U); massless packet k0 0.6, sigma 5, both Kramers partners identical (no spin force):
  delta k_y W/H2 = -0.0217/-0.0450 (b 7.35), -0.0191/-0.0393 (b 12.25): H2/W = 2.071/2.058 at equal
  time (straight ray per path 1.99/1.98); local force n_s N^n_s c' <|k|> dU/dy from the density matches
  to <= 2.7 % (with <k_x> instead of <|k|>: 5.5 % low); thin straight ray fails at b ~ sigma (0.64).
  Observer question: spheres are defined by the observer's own signals; testable is only agreement
  between probes, (a/lambda)^2 for the quartet.
  **Observer / perihelion / LT** (`RESULTS_Observer_Perihelion_LT_de.md`): light speed over 2000 directions:
  direction spread 1.9e-5/1.1e-4/3.6e-4 at k 0.02/0.05/0.1 (~k^1.83, k^3 Y4 term; extremes swap near
  k 0.3), energy dependence 1 - 0.42 (ka)^2 (aniso +-0.048).  3D walker (64^3, mu 0.15, sigma 5, mass
  may lie outside the box since U is analytic): sim / local metric force 1.034 ... 0.989 (W), 1.010 ... 0.998
  (H2); above Newton by 8 %/28 % because the packet's k spread (v^2 ~ 0.18) gravitates; H2/W 1.18.
  Wave-packet orbit impossible (L = m r v/c'^2 ~ 1 hbar, spreading << period) -> classical orbits of
  H = N sqrt(m^2 + c'^2 p^2/A^2): advance/PPN 1.0106/1.0026 (H2, a 400/1600), 1.0042/1.0011 (W = 1/3 GR),
  1.016/1.004 (linear lapse = 7/6 GR).  Rotating ball: b = 2c'^2 (kappa/4pi) J x r/r^3 (0.991 at r 8);
  classical node precession / LT = 1.001 (use the measured a, e: 1.04 with the Newtonian start values).
  drag3d: bond (x, x+s) rotation angle dt (b.s)/(8 l^2), brick wall over 6 directions x 2 parities
  (m = (x.s)//2 mod 2, box multiple of 4).  Rigid b = Omega z x r (Omega 0.004): orbit 0.00406, spin
  0.00026 without rule, 0.00399 with exp(-i dt (1/2) curl b . J).  J_a = (2i/pi) logm(R90_a), R90 =
  heading permutation (right-handed +90 deg, Rodrigues) x exp(-i pi/4 sigma_a), commutes with C; g = 1 on
  both rest doublets.  Bare spin index P(1 x sigma/2)P: g = -1/3 (particle), +1 (antiparticle).
  **Gravitational atom** (`RESULTS_GravAtom_de.md`): direct 3D walker atom infeasible (2p needs ~200^3 x 96,
  ~1e4 ticks).  Effective H = N^1/2 sqrt(m^2 + c'^2 Pi^2) N^1/2, Pi^2 = N^(n/2)(-div N^n grad)N^(n/2).  1D lattice
  atom (512 layers, mu 0.08, GM 0.6, rc = a0/2, 60000 ticks, even/odd starts): levels vs effective (free
  lattice band) 0.2 % (ground) ... 3 %, H2 - W shift to ~3 %.  Traps: Hann side lobe (-31 dB) faked a level
  below the potential minimum; 20000 ticks merged neighbouring lines (8-13 % off).  3D radial (3500 pts to
  70 a0, rc = 0.1 a0, Newtonian softening splitting subtracted): relativistic l-splitting positive, W/H2
  0.22/0.28/0.31 (l 0-1/1-2/2-3) -> 1/3, linear/H2 1.16-1.18 -> 7/6, ~ GM^4, vs omega_p = 3GM^4/(L^2 n^3)
  factor 1.1-1.25 at n <= 4.  No spin (fine structure open).
  **Gravitational spin-orbit** (`RESULTS_GravSpinOrbit_de.md`): Dirac in the metric H = beta m N + (c'/2){N/A, alpha.p}, FW term
  (2 gamma + 1) c'^2/(4m)(1/r)U' sigma.L -> fine structure E(2p3/2) - E(2p1/2) A=N : W : H2 : linear = -1 : 1 : 3 : 3 (weak field,
  radial 2D Dirac -0.978 : 1 : 2.997 : 3.011), 1 : 3.64 : 4.50 at alpha_f = 0.3.  Lattice: 2D atom in a 340x340x2 slab of Walker3D
  (mu 0.1, alpha_f 0.3, a0 9.62, rc a0/2, 8000 ticks), start = reference 2p x e^{i phi} x rest spinor J_z = +-1/2, projected on the
  exact C4 sector; levels by filter diagonalisation of <psi0|psi(t)> (FFT resolution 4-15x too coarse).  Fine structure 1.0715/3.9928/
  4.9590e-4 vs Dirac + lattice Wilson term (r_b = -0.037, beta r_b c'^2/2m N^1/2 Pi^2 N^1/2) 1.0881/3.9621/4.9009e-4: ratio 0.985/1.008/
  1.012 (pure Dirac 1.058/1.082/1.089); 2p3/2 above 2p1/2; absolute levels 0.3-1.1 % deeper (spin-independent).  Traps: kappa < 0
  needs the r=0-regular component on the half grid and P = 0 at the wall; three parallel runs hit the memory limit (checkpointing).
- **Fixed (Nov 2026, found in S3):** `quantum_fcc_3d.su2_transport` returned
  cos(θ/2) + i sin(θ/2) n·σ, which rotates by −θ (anti-homomorphism).  Now
  exp(−iθ n·σ/2).  Recomputed: RESULTS_FCC_3D (spin 1/2 = massless Dirac cone c/3,
  not massive), RESULTS_Helicity (FCC doublet overlaps exactly 0), RESULTS_TwoParticle
  (late-time ratios 0.73/1.27 instead of 0.65/1.35).  Belt trick and spin 0 unchanged.
