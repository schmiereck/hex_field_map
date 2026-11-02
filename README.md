# Triangular Lattice Geometry

[![DOI](https://zenodo.org/badge/1201004719.svg)](https://doi.org/10.5281/zenodo.22178584)

*Thomas Schmiereck — thomas@schmiereck.de*

---

## Paper 1: Relativistic Quantum Mechanics from Lattice Geometry

**Emergent Speed of Light, Mass, and Time Dilation on an Equilateral Triangular Lattice**

### Abstract

A discrete path-integral model on a **1+1D equilateral triangular lattice** and its **2+1D hexagonal extension** in which all relativistic structure emerges purely from lattice geometry.

**README**: [`README1_lattice_geometry.md`](docs/README1_lattice_geometry.md)  
LaTeX version: [`paper.tex`](papers/paper.tex)  
PDF version: [`paper.pdf`](papers/paper.pdf)  
German version: [`paper_de.tex`](papers/paper_de.tex) / [`paper_de.pdf`](papers/paper_de.pdf)  
DOI: [10.5281/zenodo.22178585](https://doi.org/10.5281/zenodo.22178585) (release v1.0.0, see [Citation](#citation))

---

## Paper 2: Internal Structure of the Equilateral Lattice Path Integral

**Crystal Angular Momentum, Emergent Vector Ground State, and Symmetry Breaking by Confinement**

### Abstract

In a previous paper we showed that an equilateral triangular (1+1D) and
hexagonal (2+1D) path-integral lattice with the single amplitude rule
"direction change = iε" reproduces relativistic dispersion, exact isotropy
and the speed of light c = √3 from geometry alone. Here we analyse the
**internal structure** of the physical propagating band of the same model,
in 2+1D and in a natural 3+1D tetrahedral-octahedral (cuboctahedron)
extension.

follow-up to [`paper.tex`](papers/paper.tex) / [`paper.pdf`](papers/paper.pdf) / [`README1_lattice_geometry.md`](docs/README1_lattice_geometry.md)

**README**: [`README2_internal_structure.md`](docs/README2_internal_structure.md)  
LaTeX version: [`paper2_internal_structure.tex`](papers/paper2_internal_structure.tex)  
PDF version: [`paper2_internal_structure.pdf`](papers/paper2_internal_structure.pdf)  
DOI: [10.5281/zenodo.23888331](https://doi.org/10.5281/zenodo.23888331) (release v2.0.0, see [Citation](#citation))

---

## Further Investigations (work in progress, notes in German)

Building on the 2+1D hexagonal model, a **turning-phase model** replaces the
amplitude rule by the *sum of turning angles*, scaled by a parameter α:
`w_path = ∏ a(|n|) · exp(i·α·60°·n)`. For every closed walk the angle sum is
exactly 360°·w (discrete Whitney theorem), so α acts as an Aharonov–Bohm flux
in heading space; α = ½ gives the spinor double cover. The coin operator is
exactly unitary.

| Topic | Result | Notes |
|-------|--------|-------|
| Turning-phase model (2+1D) | Exact unitarity, Kramers doubling at α = ½, massless cone with slope c/2 | [`RESULTS_Turning_2D_de.md`](docs/RESULTS_Turning_2D_de.md) |
| Magnetic field (Peierls phase) | Cyclotron orbits, chirality families, Onsager quantisation | [`RESULTS_Magnetic_2D_de.md`](docs/RESULTS_Magnetic_2D_de.md) |
| 3D spin and QCD roadmap | Why spin needs 3D (no winding number on S²), what QCD needs | [`ROADMAP_QCD_3D_de.md`](docs/ROADMAP_QCD_3D_de.md) |
| SU(3) Wilson-action Monte Carlo | Area law vs perimeter law on the triangular lattice | [`RESULTS_SU3_MC_de.md`](docs/RESULTS_SU3_MC_de.md) |
| 3+1D FCC walker with SU(2) coin | α disappears: spin 0 or spin ½ only, Kramers automatic | [`RESULTS_FCC_3D_de.md`](docs/RESULTS_FCC_3D_de.md) |
| Two-particle Hilbert space | Pauli hole, bound states, colour singlet | [`RESULTS_TwoParticle_de.md`](docs/RESULTS_TwoParticle_de.md) |
| 2+1D deflection scattering | Mott interference, capped by the helicity-like band eigenvector | [`RESULTS_Scatter_2D_de.md`](docs/RESULTS_Scatter_2D_de.md) |
| Helicity test | Internal spin does not lift the cap; the non-relativistic limit does | [`RESULTS_Helicity_de.md`](docs/RESULTS_Helicity_de.md) |
| Caching on the regular lattice | Memoised walk enumeration | [`NOTES_Caching_de.md`](docs/NOTES_Caching_de.md) |

---

## Repository Layout

| Directory | Content |
|-----------|---------|
| [`papers/`](papers) | LaTeX sources and PDFs of the papers, `PAPER_SUMMARY.md` |
| [`docs/`](docs) | Paper READMEs, result write-ups (`RESULTS_*.md`), notes, interactive HTML |
| [`src/`](src) | Python simulations and figure scripts |
| [`figures/`](figures) | Generated figures and animations |
| [`data/`](data) | Numerical results (`.npz`) |

Scripts can be run from any directory, e.g. `python src/quantum_hex_2d.py`;
output paths are resolved by [`src/paths.py`](src/paths.py).
The papers are built from inside `papers/` (`pdflatex paper.tex`) and take
their figures from `../figures/`.
Dependencies: [`requirements.txt`](requirements.txt).

---

## Citation

Both papers are archived on Zenodo as releases of this repository.

**Paper 1** (release v1.0.0):

> Schmiereck, T. (2026). *Relativistic Quantum Mechanics from Lattice Geometry:
> Emergent Speed of Light, Mass, and Time Dilation on an Equilateral Triangular
> Lattice* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.22178585

```bibtex
@software{schmiereck_2026_paper1,
  author    = {Schmiereck, Thomas},
  title     = {Relativistic Quantum Mechanics from Lattice Geometry:
               Emergent Speed of Light, Mass, and Time Dilation
               on an Equilateral Triangular Lattice},
  year      = {2026},
  month     = aug,
  version   = {v1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22178585},
  url       = {https://doi.org/10.5281/zenodo.22178585}
}
```

**Paper 2** (release v2.0.0):

> Schmiereck, T. (2026). *Internal Structure of the Equilateral Lattice Path
> Integral: Crystal Angular Momentum, Emergent Vector Ground State, and Symmetry
> Breaking by Confinement* (v2.0.0). Zenodo. https://doi.org/10.5281/zenodo.23888331

```bibtex
@software{schmiereck_2026_paper2,
  author    = {Schmiereck, Thomas},
  title     = {Internal Structure of the Equilateral Lattice Path Integral:
               Crystal Angular Momentum, Emergent Vector Ground State,
               and Symmetry Breaking by Confinement},
  year      = {2026},
  month     = nov,
  version   = {v2.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23888331},
  url       = {https://doi.org/10.5281/zenodo.23888331}
}
```

To cite the repository as a whole (always resolving to the latest release), use
the concept DOI [10.5281/zenodo.22178584](https://doi.org/10.5281/zenodo.22178584).
Citation metadata is also provided in [`CITATION.cff`](CITATION.cff)
(GitHub: *Cite this repository*).
