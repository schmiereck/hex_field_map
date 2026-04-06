# Paper Summary

**Title:** Relativistic Quantum Mechanics from Lattice Geometry:
Emergent Speed of Light, Mass, and Time Dilation on an Equilateral Triangular Lattice

**Author:** Thomas Schmiereck
**File:** `paper.tex` (arXiv category: quant-ph)

---

## Plain-Language Summary

Imagine a triangular grid where time flows upward and a particle can take
three kinds of steps at each moment: lean left, lean right, or go straight up.
All three step lengths are identical (edge length = 1).
The only rule: if the particle changes direction, multiply its quantum
amplitude by a small imaginary number $i\varepsilon$; if it keeps going the
same way, multiply by 1.
That's it — no equations of motion, no postulated mass, no built-in speed
of light.

From just this geometric rule, the simulation produces: a speed of light
$c = \sqrt{3}$ (because the diagonal step covers $\sqrt{3}/2$ space in
$1/2$ time), a particle mass $m \approx 2\varepsilon$, relativistic energy
$E^2 = c^2 p^2 + m^2$, and — in the 2D extension using a hexagonal lattice
— perfect rotational symmetry with no spurious extra particles (no
"fermion doubling").
The particle's internal clock (proper time) slows down exactly as Einstein's
special relativity predicts, and the quantum trembling motion known as
Zitterbewegung appears at exactly the right frequency.
An antiparticle mode even shows up automatically as a negative-energy
eigenvalue of the evolution matrix.

Everything relativistic emerges from the shape of the lattice alone.

---

## Five Most Important Numerical Results

- **Speed of light:** $c = \sqrt{3} = 1.7321$ — exact from geometry,
  confirmed by strict light-cone causality in simulation
  ($r = \sqrt{3}\,t$ never violated).

- **Physical mass:** $m_{\rm phys}(\varepsilon{=}0.1) = 0.1993 \approx 2\varepsilon$,
  with $m = \arctan(2\varepsilon/(1{-}\varepsilon^2))$ reproducing the
  measured eigenvalue across $\varepsilon \in \{0.01, 0.1, 0.5, 1.0\}$.

- **Relativistic dispersion:** $E^2 = 3k^2 + m^2$ with
  RMSE $= 0.0066$ at $k \le 0.05$, and 6-fold isotropy error $= 0.0000$
  for $|k| \le 0.4$ in the 2+1D hexagonal model.

- **Zitterbewegung:** Measured period $= 15.76$,
  theoretical $2\pi/(2m) = 15.77$ — agreement to $0.06\%$.

- **Time dilation convergence:** At packet width $\sigma = 30$
  ($\sigma_k/m = 0.167$), the quantum proper time $\tau_{\rm quantum}$
  lies within $3.6\%$ of the SR prediction $T\sqrt{1{-}v^2/c^2}$ at
  $v = 0$ and within $1.1\%$ at $v = 0.9c$, with monotone convergence
  as $\sigma \to \infty$.

---

## Suggested arXiv Keywords

`discrete quantum mechanics` · `path integral` · `lattice field theory` ·
`Feynman checkerboard` · `Dirac equation` · `relativistic dispersion` ·
`Zitterbewegung` · `fermion doubling` · `emergent Lorentz symmetry` ·
`quantum proper time` · `hexagonal lattice` · `equilateral triangular lattice`
