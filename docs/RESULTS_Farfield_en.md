# Far-Field (Fraunhofer) Double Slit on the Hex Lattice — Findings

This document reports a **negative result with explanation**: under the
constraints of the 2+1D hexagonal path-integral model used in this
project, the textbook Fraunhofer regime
`Δy = λ L / d` cannot be reached on grids of practical size.

## What was attempted

`experiment_farfield_slit.py` — same hex lattice and physical-band
projection as `experiment_double_slit.py`, but with parameters chosen
to satisfy

1. **Fraunhofer condition** `F = L / (d² / λ) ≥ 3`
2. **Off-axis maxima exist** `d > λ` (so `sin θ_n = n λ / d` has solutions for `n ≥ 1`)
3. **Several fringes on screen** `N_y · Δy_lat ≥ 4 Δy`
4. **Grid ≤ 220 × 300** (for tractable runtime)
5. **Time-integrated screen intensity** `I(y) = Σ_t |ψ_phys(x_scr, y, t)|²`,
   driven by a long pulse (σ_x = 50, σ_y = 40 in lattice indices), so the
   coherence time exceeds the path-difference delay across the screen.

Best parameter set found: `NX=220, NY=280, k₀=0.4, d_idx=22 (d=16.5 phys),
L_idx=113 (L=48.9 phys), λ=15.7 phys`, giving `F=2.82` and a *predicted*
`Δy=46.6 phys ≈ 62 idx`.

## What was observed

- The time-integrated screen pattern is a **single smooth Gaussian-like
  envelope** centred between the two slits.
- Single-slit runs (block A, block B) give two displaced Gaussian
  envelopes whose sum reproduces the double-slit profile to within ~1 %.
- No interference modulation at the predicted fringe scale `Δy ≈ 62 idx`
  is detectable.
- `find_peaks` flags small (~1 %) ripples at Δy ≈ 10 idx, but their
  m-mode content (m=±1 vector) is identical at "peaks" and "minima" —
  these are lattice noise, not interference fringes.

## Why the regime is unreachable on this lattice

The **k=0 physical band projector** used in `experiment_double_slit.py`
projects the 14-component state onto the 5-fold degenerate eigenspace of
M_full at `k=0` and `E = 2ε ≈ 0.199`. This projector is valid only for
wave content with `|k| ≪ 0.4` — it removes everything outside that
small-momentum window.

But the band dispersion is approximately

> `E²(k) ≈ c²|k|² + m²`,  with `c=√3`, `m=2ε`

so at the band bottom (`E = m`) the propagating wave has `k → 0`,
i.e. **infinite physical wavelength**. A finite-k packet that we try to
inject in the y-direction is suppressed by the projector before it can
form a y-periodic interference pattern.

In short:

| Want | Constraint | Conflict |
|---|---|---|
| Multiple Fraunhofer maxima | `d > λ` ⇒ `λ < d` | possible |
| Fraunhofer regime | `L > 3 d² / λ` | grid grows as `d²` |
| Use k=0 projector | `k₀ ≲ 0.4` ⇒ `λ ≳ 16 phys` | OK |
| Wave actually carries that k | dispersion at `k₀≈0.4` is `E ≈ 0.7` ≫ band bottom `0.2` | **fails** — wave is outside the projector subspace |

The k=0 physical band is the *band-bottom* — the only momentum at which
the projector is exactly correct. The band-bottom mode has k=0 and
therefore no spatial periodicity to diffract into Fraunhofer fringes.
The original Fresnel-regime double-slit (`experiment_double_slit.py`)
worked because **all** of its physics is at the band bottom and the
geometric/Fresnel near-field pattern does not require finite-k coherent
oscillations.

## What this implies

A proper Fraunhofer experiment on this lattice would require either:

1. **A k-dependent physical-band projector** — built at every (kx, ky)
   along the band, using the band's actual dispersion. This is a
   non-trivial extension of `physical_band_2d`.
2. **Many ε of separation between band bottom and band continuum**
   so that `k₀` of order `1/d` still lies in the band well below the
   continuum. With `ε=0.1` the band is too narrow.
3. **A different lattice / different transfer-matrix construction** in
   which the propagating-mode dispersion has a non-trivial momentum
   range that survives the projector.

None of these is a small change. The Fresnel near-field result of
`experiment_double_slit.py` (one central peak + slit-A + slit-B peaks)
is the only diffraction signal extractable from the present model.

## Square-lattice comparison

The same script runs the 2+1D square (Feynman-checkerboard) variant
with `T_sq = T·√3` to match physical time. The square screen pattern
shows the same qualitative behaviour — a smooth envelope, no Fraunhofer
fringes — for the same reason: the rank-3 physical band built at k=0
does not extend to the k₀ needed for d > λ at any reasonable grid size.

## Files

- `experiment_farfield_slit.py` — script (final parameter set above)
- `farfield_results.npz` — numerical arrays
- `fig_farfield_heatmap.png`, `fig_farfield_screen.png`,
  `fig_farfield_modes.png` — diagnostic plots showing the smooth envelope
  and the absence of fringe-scale modulation.

## Bottom-line answer

> **Q: Does the hexagonal lattice reproduce the Fraunhofer fringe
> spacing `Δy = λ L / d` to within < 5 %?**
>
> **A: Not with the k=0 physical-band projector.** No fringes at the
> predicted scale form on grids of practical size. The reason is
> structural: the band-bottom propagating mode has k → 0 and therefore
> no y-periodic interference can develop in the projected subspace.
> The Fresnel-regime three-peak pattern reported in
> `RESULTS_Experiments_en.md` is the strongest interference signal
> achievable in this model without rebuilding the band projector
> as a function of (kx, ky).
