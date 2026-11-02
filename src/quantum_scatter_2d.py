#!/usr/bin/env python3
"""
Two-body scattering in 2+1D: deflection collisions
===================================================

Step 4 of ROADMAP_QCD_3D_de.md.

RESULTS_TwoParticle_de.md showed that in 1+1D a contact interaction between
identical particles is integrable: it produces only a phase shift, never a
redistribution of momentum.  A genuine deflection needs at least two spatial
dimensions.

The full two-particle amplitude in 2D would be (N_sites * 6)^2, which is far
too large.  But the contact interaction is translation invariant, so the total
momentum Q is conserved and the problem reduces to the RELATIVE coordinate:

    Psi(r1,d1,r2,d2) = e^{-i Q.r2} phi(r1 - r2, d1, d2)

One step sends  r -> r + dr(d1) - dr(d2)  and multiplies by e^{i Q.dr(d2)},
so the reduced state is (N_sites x 36) — a few megabytes.

At Q = 0 (centre-of-mass frame) the exchange operator is simply
    (X phi)(r, d1, d2) = phi(-r, d2, d1)
with no phase, so bosons and fermions are the symmetric and antisymmetric
sectors of the same operator.

Expected (and this is a TEST, not a discovery): the scattered wave sits on a
ring of fixed |k_rel|, hexagonally warped by the lattice, and for identical
particles the amplitude is f(theta) -+ f(theta+pi).  For a central interaction
f(pi/2) = f(3pi/2), so the FERMIONIC cross section must vanish at 90 degrees in
the centre-of-mass frame while the bosonic one is doubled there.
"""

import numpy as np

from quantum_hex_turning import (MOVES_IDX, MOVES_PHYS, DX_PHYS, DY_PHYS,
                                 DT_HALF, N_D, coin, bands, pick_band)


# ─── reduced (relative-coordinate) evolution ─────────────────────────────────

def _shift2(a, sx, sy, periodic=False):
    if periodic:
        return np.roll(a, (sx, sy), axis=(0, 1))
    out = np.zeros_like(a)
    nx, ny = a.shape[0], a.shape[1]
    xs = slice(max(0, -sx), nx - max(0, sx))
    xd = slice(max(0, sx), nx - max(0, -sx))
    ys = slice(max(0, -sy), ny - max(0, sy))
    yd = slice(max(0, sy), ny - max(0, -sy))
    out[xd, yd] = a[xs, ys]
    return out


def step_rel(phi, C, Qphase, U=0.0, cx=None, cy=None, mode="site",
             periodic=False):
    """
    One step of the reduced two-particle evolution.

    phi has shape (Nx, Ny, 6, 6) indexed by (r, d1, d2).
    Qphase[d2] = exp(i Q . dr(d2)).
    The contact acts at r = 0, i.e. at grid index (cx, cy).
    """
    w = np.einsum('ae,xyeb->xyab', C, phi)
    w = np.einsum('bf,xyaf->xyab', C, w)
    w = w * Qphase[None, None, None, :]
    out = np.zeros_like(w)
    for d1 in range(N_D):
        for d2 in range(N_D):
            sx = MOVES_IDX[d1, 0] - MOVES_IDX[d2, 0]
            sy = MOVES_IDX[d1, 1] - MOVES_IDX[d2, 1]
            out[:, :, d1, d2] = _shift2(w[:, :, d1, d2], sx, sy, periodic)
    if U:
        if mode == "site":
            out[cx, cy, :, :] *= np.exp(1j * U)
        else:                                   # full coincidence (r=0, d1=d2)
            for d in range(N_D):
                out[cx, cy, d, d] *= np.exp(1j * U)
    return out


def q_phase(Q):
    return np.exp(1j * (MOVES_PHYS @ np.asarray(Q, dtype=float)))


def exchange(phi, cx=0, cy=0):
    """
    (X phi)(r,d1,d2) = phi(-r,d2,d1)  — valid at Q = 0.

    r = 0 sits at grid index (cx, cy), so negating r means reflecting the index
    about that point: i -> (2*cx - i) mod N.  Note that phi[::-1] is NOT this
    reflection unless the grid is odd and centred (2*cx = N-1); getting that
    wrong makes X fail to commute with the step.
    """
    nx, ny = phi.shape[0], phi.shape[1]
    ix = (2 * cx - np.arange(nx)) % nx
    iy = (2 * cy - np.arange(ny)) % ny
    return np.transpose(phi[np.ix_(ix, iy)], (0, 1, 3, 2))


def symmetrise(phi, theta, cx=0, cy=0):
    """theta = 0 boson (X = +1), theta = pi fermion (X = -1)."""
    return 0.5 * (phi + np.exp(1j * theta) * exchange(phi, cx, cy))


# ─── setting up and running a collision ──────────────────────────────────────

def grid_phys(Nx, Ny, cx, cy):
    IX, IY = np.meshgrid(np.arange(Nx) - cx, np.arange(Ny) - cy, indexing='ij')
    return IX * DX_PHYS, IY * DY_PHYS, ((IX + IY) % 2 == 0)


def initial_relative(Nx, Ny, cx, cy, r0, k, sigma, eps, alpha, band=-1):
    """
    Relative-coordinate state for two band eigenstates with momenta +k and -k
    (total momentum zero):  phi(r) = env(r-r0) e^{i k.r} u(+k)_d1 u(-k)_d2.
    """
    X, Y, sub = grid_phys(Nx, Ny, cx, cy)
    E1, V1, _ = bands(k[0], k[1], eps, alpha)
    E2, V2, _ = bands(-k[0], -k[1], eps, alpha)
    u1 = V1[:, band] / np.linalg.norm(V1[:, band])
    u2 = V2[:, band] / np.linalg.norm(V2[:, band])
    env = (np.exp(-((X - r0[0]) ** 2 + (Y - r0[1]) ** 2) / (2 * sigma ** 2))
           * np.exp(1j * (k[0] * X + k[1] * Y)) * sub)
    phi = env[:, :, None, None] * np.einsum('a,b->ab', u1, u2)[None, None, :, :]
    return phi / np.sqrt((np.abs(phi) ** 2).sum())


def run_scatter(phi0, eps, alpha, U, n_steps, cx, cy, mode="site"):
    C = coin(eps, alpha, "unitary")
    Qp = q_phase(np.zeros(2))
    phi = phi0.copy()
    for _ in range(n_steps):
        phi = step_rel(phi, C, Qp, U, cx, cy, mode)
    return phi


def angular_profile(phi, cx, cy, r_min, r_max, n_bin=72):
    """Bin the density over an annulus by the polar angle of r."""
    Nx, Ny = phi.shape[0], phi.shape[1]
    X, Y, _ = grid_phys(Nx, Ny, cx, cy)
    R = np.hypot(X, Y)
    TH = np.degrees(np.arctan2(Y, X)) % 360.0
    dens = (np.abs(phi) ** 2).sum(axis=(2, 3))
    m = (R >= r_min) & (R <= r_max)
    edges = np.linspace(0, 360, n_bin + 1)
    idx = np.clip(np.digitize(TH[m], edges) - 1, 0, n_bin - 1)
    out = np.zeros(n_bin)
    np.add.at(out, idx, dens[m])
    return 0.5 * (edges[:-1] + edges[1:]), out
