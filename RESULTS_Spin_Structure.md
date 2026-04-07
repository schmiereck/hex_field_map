# Spin Structure of the 5-fold Band (2+1D Hexagonal)

ε = 0.1,  M_full = M_half²,  k = 0

rank(M_full) = 8 / 14

## Task 1 — C_6v invariance & irrep decomposition

M_full is invariant under C_6v: **True**

Multiplicities of C_6v irreps in the 14-dim representation:

| Irrep | dim | multiplicity |
|---|---|---|
| A1 | 1 | +4.000 |
| A2 | 1 | +0.000 |
| B1 | 1 | +2.000 |
| B2 | 1 | +0.000 |
| E1 | 2 | +2.000 |
| E2 | 2 | +2.000 |

## Task 2+3 — Angular-momentum content of the 5-fold band

Angular momenta m (R_60 eigenvalues = exp(i m π/3)):

`m = ['+1.00', '+2.00', '-0.44', '-1.00', '-2.00', '-3.00']`

Rounded: **[-3, -2, -1, 0, 1, 2]**

See `spin_eigenvectors_k0.png` for polar plots of the
five angular-momentum eigenstates.

## Task 4 — Splitting for k>0

See `spin_band_splitting.png`.

## Conclusion

The 5-fold subspace carries angular momenta [-3, -2, -1, 0, 1, 2] (mod 6).

**Verdict:** (C/D) Lattice-specific reducible representation — see m-list above.
