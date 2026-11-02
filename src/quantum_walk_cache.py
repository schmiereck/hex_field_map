#!/usr/bin/env python3
"""
Memoised closed-walk enumeration — caching on a regular lattice
================================================================

The project enumerates closed walks in several places (winding-number families
in RESULTS_Turning_2D_de.md, the belt trick in RESULTS_FCC_3D_de.md).  Those
use a plain depth-first search, whose cost grows like (branching)^L.

On a REGULAR lattice the same sub-problem recurs at many places in that tree:
"how many walks of n more steps lead from (position, heading) back to the
origin, and with which turn sums?"  That depends only on (position, heading,
steps left) — not on how the walk got there.  Memoising it turns the
exponential search into a polynomial-size table:

    f(r, d, n) = sum over allowed turns s of  shift_s f(r + dr(d+s), d+s, n-1)

which is exactly a chain "one cached entry produces the next, covering a region
one step larger".

Measured on the triangular lattice (5 allowed turns per step):

      L    DFS       memoised   speedup   table    walks found
      7    0.035 s   0.003 s        10x     725            370
      9    0.646 s   0.007 s        88x    1444          7 454
     11   13.393 s   0.015 s       917x    2523        154 874
     13    —         0.023 s          —    4034      3 330 646
     15    —         0.037 s          —    6157     72 961 242

The table grows roughly linearly while the number of walks grows
exponentially.  Results agree exactly with the depth-first version wherever
both can be run.

WHERE THIS DOES AND DOES NOT APPLY
----------------------------------
It applies to COMBINATORIAL quantities: walk counts, winding numbers, loop
families, holonomy classes.  The cache key is a small discrete tuple, so hits
are exact and frequent.

It does NOT apply to hashing a wavefunction on a region boundary: those
amplitudes are continuous complex numbers and two regions are never bit
identical.  The correct analogue there is to cache the OPERATOR rather than the
state — for a homogeneous lattice the boundary-to-boundary map of a region is
the same for every translate and every symmetry image of it, and merging two
blocks into one twice as large is the chaining step.  That is the hierarchical
/ recursive Green's function method.  Note that the project already uses the
strongest form of that idea wherever the lattice is homogeneous: the transfer
matrix replaces the whole spatial problem by one small matrix per k, and the
relative-coordinate reduction in quantum_scatter_2d.py turns (N_sites*6)^2 into
N_sites*36 by exploiting exactly this translation invariance.
"""

import math

import numpy as np

from quantum_hex_turning import MOVES_IDX, DX_PHYS, DY_PHYS, SQRT3
from quantum_fcc_3d import FCC_STEPS, FCC_UNIT, ADJ60


def closed_walks_2d(L_max, d0=0, steps=(-2, -1, 0, 1, 2)):
    """
    {L: {winding: count}} for closed walks on the triangular lattice that
    return to the same directed edge.  Memoised; agrees exactly with the
    depth-first enumerate_loops() in quantum_hex_turning.
    """
    memo = {}

    def f(ix, iy, d, n):
        key = (ix, iy, d, n)
        if key in memo:
            return memo[key]
        if n == 0:
            out = {0: 1} if (ix == 0 and iy == 0 and d == d0) else {}
        elif math.hypot(ix * DX_PHYS, iy * DY_PHYS) / (SQRT3 / 2) > n + 1e-9:
            out = {}
        else:
            out = {}
            for s in steps:
                nd = (d + s) % 6
                for t, c in f(ix + MOVES_IDX[nd, 0], iy + MOVES_IDX[nd, 1],
                              nd, n - 1).items():
                    out[t + s] = out.get(t + s, 0) + c
        memo[key] = out
        return out

    res = {}
    for L in range(1, L_max + 1):
        d = {}
        for t, c in f(0, 0, d0, L).items():
            if t % 6 == 0:
                d[t // 6] = d.get(t // 6, 0) + c
        if d:
            res[L] = d
    return res, len(memo)


def closed_walks_fcc(L_max, d0=0):
    """
    {L: {'+1': n, '-1': n}} for closed walks on the FCC 60-degree graph,
    classified by the SU(2) holonomy sign.  Memoised over
    (position, heading, steps left) with the quaternion carried forward, so the
    key stays discrete.  Agrees with the depth-first belt_trick().
    """
    allowed = [[j for j in range(len(FCC_UNIT)) if ADJ60[i, j]]
               for i in range(len(FCC_UNIT))]
    # transport quaternions between 60-degree neighbours
    from quantum_fcc_3d import quat_transport, qmul
    QT = [[quat_transport(FCC_UNIT[i], FCC_UNIT[j]) for j in range(len(FCC_UNIT))]
          for i in range(len(FCC_UNIT))]
    memo = {}

    def f(pos, d, n):
        key = (pos, d, n)
        if key in memo:
            return memo[key]
        if n == 0:
            out = {(1.0, 0.0, 0.0, 0.0): 1} if (pos == (0, 0, 0) and d == d0) else {}
        elif math.sqrt(sum(p * p for p in pos)) / math.sqrt(2.0) > n + 1e-9:
            out = {}
        else:
            out = {}
            for nd in allowed[d]:
                q = QT[d][nd]
                sub = f(tuple(p + s for p, s in zip(pos, FCC_STEPS[nd])), nd, n - 1)
                for qq, c in sub.items():
                    tot = tuple(round(v, 9) for v in qmul(qq, q))
                    out[tot] = out.get(tot, 0) + c
        memo[key] = out
        return out

    res = {}
    u0 = FCC_UNIT[d0]
    for L in range(1, L_max + 1):
        plus = minus = 0
        for q, c in f((0, 0, 0), d0, L).items():
            phi = 2.0 * math.atan2(float(np.dot(np.array(q[1:]), u0)), q[0])
            if abs(round(phi / (2 * np.pi))) % 2 == 1:
                minus += c
            else:
                plus += c
        if plus or minus:
            res[L] = {"+1": plus, "-1": minus}
    return res, len(memo)
