#!/usr/bin/env python3
"""
Does the turning-number theorem survive in 3D?  —  FCC holonomy test
=====================================================================

In 2D the heading of a walker lives on a CIRCLE.  For a closed walk that
returns to the same directed edge the accumulated turning is exactly
360 deg * w with an integer winding number w (see RESULTS_Turning_2D_de.md).
That integer is what allows a *continuous* phase exp(i 2 pi alpha w): alpha is
a flux threaded through the circle of headings, and pi_1(S^1) = Z admits any
flux.  This is the anyon situation — spin is a free real parameter.

In 3D the heading lives on a SPHERE.  This module tests, on the 12-neighbour
FCC lattice, what happens to the theorem.  Headings are parallel-transported
with the minimal rotation from one heading to the next, accumulated as a
quaternion (i.e. in SU(2), the double cover of SO(3)).

Result (see ROADMAP_QCD_3D_de.md):
  * the holonomy of a closed walk is a rotation about the initial heading
    (verified to 8e-16), but its ANGLE is not quantised — 23 distinct values
    already at length 6, e.g. phi/2pi = 0.19591... with cos(phi/2) = sqrt(2/3),
    an irrational multiple of 2 pi.  There is no winding number.
  * holonomies based at different headings do NOT commute.
  * pi_1(S^2) = 0: there is no loop in the heading space to thread a flux
    through, so the continuous alpha of the 2D model has no 3D counterpart.
    The only freedom left is the Z_2 centre of SU(2): +1 or -1, i.e. boson or
    fermion.  Spin becomes rigid exactly because the 3D rotation group is
    doubly — not infinitely — connected.
"""

import itertools
import math
import numpy as np

# ─── FCC geometry ────────────────────────────────────────────────────────────

FCC_STEPS = np.array([v for v in itertools.product([-1, 0, 1], repeat=3)
                      if sum(abs(x) for x in v) == 2], dtype=int)   # 12 vectors
FCC_UNIT = FCC_STEPS / np.sqrt(2.0)
N_FCC = len(FCC_STEPS)


# ─── quaternions (SU(2)) ─────────────────────────────────────────────────────

def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)


def transport(u, v):
    """Minimal rotation taking heading u to heading v, as a unit quaternion.

    None for the 180 deg reversal, whose rotation axis is undefined — the same
    ambiguity that the 2D model resolves by forbidding reversals.
    """
    c = float(np.dot(u, v))
    if c > 1 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    n = np.cross(u, v)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return None
    n = n / nn
    th = math.acos(max(-1.0, min(1.0, c)))
    s = math.sin(th / 2.0)
    return (math.cos(th / 2.0), s * n[0], s * n[1], s * n[2])


def transport_table(unit):
    n = len(unit)
    return [[transport(unit[i], unit[j]) for j in range(n)] for i in range(n)]


# ─── closed-walk holonomies ──────────────────────────────────────────────────

def closed_holonomies(L_max=6, d0=0, unit=None, steps=None, scale=None):
    """
    All closed walks that return to the same directed edge, with their SU(2)
    holonomy.  Returns {L: [(quaternion, phi, axis_residual), ...]}.

    phi is the rotation angle about the initial heading; axis_residual is the
    component of the rotation axis perpendicular to that heading and must
    vanish (it does, to ~1e-15).
    """
    if unit is None:
        unit, steps, scale = FCC_UNIT, FCC_STEPS, math.sqrt(2.0)
    QT = transport_table(unit)
    n_d = len(unit)
    dim = len(steps[0])
    u0 = unit[d0]
    out = {L: [] for L in range(1, L_max + 1)}

    def rec(pos, d, q, depth):
        if depth > 0 and all(p == 0 for p in pos) and d == d0:
            vec = np.array(q[1:])
            phi = 2.0 * math.atan2(float(np.dot(vec, u0)), q[0])
            res = float(np.linalg.norm(vec - np.dot(vec, u0) * u0))
            out[depth].append((q, phi, res))
        if depth == L_max:
            return
        if math.sqrt(sum(p * p for p in pos)) / scale > (L_max - depth) + 1e-9:
            return
        for nd in range(n_d):
            if QT[d][nd] is None:
                continue
            rec(tuple(p + s for p, s in zip(pos, steps[nd])),
                nd, qmul(QT[d][nd], q), depth + 1)

    rec(tuple([0] * dim), d0, (1.0, 0.0, 0.0, 0.0), 0)
    return out


def hex_2d_control(L_max=6):
    """The same computation for the 2D triangular lattice, as a control."""
    ang = np.arange(6) * np.pi / 3
    unit = np.stack([np.cos(ang), np.sin(ang), np.zeros(6)], axis=1)
    steps = np.array([[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1]])
    return closed_holonomies(L_max, 0, unit, steps, 1.0)


def commutator_across_headings(L_max=5, d_a=0, d_b=1):
    """
    Holonomies based at one heading are all rotations about that heading, so
    they commute trivially.  The non-abelian structure appears between
    different base headings.  Returns the largest |Qa Qb - Qb Qa|.
    """
    A = [q for L in closed_holonomies(L_max, d_a).values() for q, _, _ in L]
    B = [q for L in closed_holonomies(L_max, d_b).values() for q, _, _ in L]
    mx = 0.0
    for a in A:
        for b in B:
            ab, ba = qmul(a, b), qmul(b, a)
            mx = max(mx, max(abs(x - y) for x, y in zip(ab, ba)))
    return mx, len(A), len(B)
