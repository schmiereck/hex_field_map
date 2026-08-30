#!/usr/bin/env python3
"""
Non-abelian (SU(3)) colour on the triangular lattice
=====================================================

The magnetic model replaces the link phase by exp(i*A.dl) in U(1).  QCD needs
the same construction with the phase promoted to a matrix:

    U_link  in  SU(3),      W_C = tr( ordered product of U along C )

Three things change, and all three are checked here:

1.  The state carries a colour index:  amp[x, y, d, c],  c = 1,2,3.
2.  Only the TRACE of a closed loop is gauge invariant, not the loop
    "flux" — under U -> g(x) U g(y)^dagger, tr(W_C) is unchanged.
3.  Fluxes no longer ADD.  For U(1) we verified exactly that the flux of a
    rhombus is the sum of the fluxes of its two triangles.  For SU(3) the
    matrices do not commute, the product is path ordered, and no such
    additivity exists.  This non-commutativity is the whole of QCD's
    difference from electromagnetism.

Scope / honesty
---------------
The gauge field here is a STATIC, quenched background: links are drawn from a
fixed distribution, they have no action and no dynamics.  That is enough to
demonstrate the machinery and to measure a Wilson-loop area law, but it is NOT
lattice QCD: real QCD needs the links to be sampled from the Wilson action
(Monte Carlo), and asymptotic freedom needs 3+1 dimensions.  See
ROADMAP_QCD_3D_de.md.

Lattice conventions
-------------------
Abstract triangular lattice with integer coordinates s = (i, j) and three
positive link directions

    mu = 0: s -> s + (1, 0)
    mu = 1: s -> s + (0, 1)
    mu = 2: s -> s + (-1, 1)

The elementary (up) triangle is the path [+0, +2, -1]:
    (1,0) + (-1,1) - (0,1) = (0,0).
"""

import numpy as np
from scipy.linalg import expm

N_C = 3
DIRS = np.array([[1, 0], [0, 1], [-1, 1]])
PLAQ_UP = [(+1, 0), (+1, 2), (-1, 1)]          # signed directions of the path


# ─── SU(3) elements ──────────────────────────────────────────────────────────

def haar_su3(rng, n=N_C):
    """Haar-random SU(n) via QR of a complex Gaussian."""
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q * (np.diag(r) / np.abs(np.diag(r)))[None, :]
    return q / np.linalg.det(q) ** (1.0 / n)


def su3_near_identity(g, rng, n=N_C):
    """U = exp(i*g*H) with H a normalised traceless Hermitian matrix."""
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (a + a.conj().T) / 2.0
    H -= np.trace(H) / n * np.eye(n)
    H /= np.sqrt((np.abs(H) ** 2).sum()) + 1e-30
    return expm(1j * g * H)


def make_links(L, g, rng, group="su3"):
    """links[i, j, mu] = SU(3) matrix on the link s -> s + DIRS[mu]."""
    n = 1 if group == "u1" else N_C
    U = np.empty((L, L, 3, n, n), dtype=complex)
    for i in range(L):
        for j in range(L):
            for mu in range(3):
                if group == "haar":
                    U[i, j, mu] = haar_su3(rng, n)
                elif group == "u1":
                    U[i, j, mu] = np.exp(1j * g * rng.normal())
                else:
                    U[i, j, mu] = su3_near_identity(g, rng, n)
    return U


# ─── Wilson loops ────────────────────────────────────────────────────────────

def walk_loop(U, start, path):
    """
    Ordered product of link matrices along `path`, a list of (sign, mu).
    Returns (matrix, end_site).  The path must close for a Wilson loop.
    """
    L = U.shape[0]
    n = U.shape[-1]
    M = np.eye(n, dtype=complex)
    i, j = start
    for sgn, mu in path:
        if sgn > 0:
            M = U[i % L, j % L, mu] @ M
            i += DIRS[mu, 0]; j += DIRS[mu, 1]
        else:
            i -= DIRS[mu, 0]; j -= DIRS[mu, 1]
            M = U[i % L, j % L, mu].conj().T @ M
    return M, (i, j)


def wilson(U, start, path):
    """(1/N) tr of the ordered product; the loop must close."""
    M, end = walk_loop(U, start, path)
    assert (end[0] - start[0]) % U.shape[0] == 0 and \
           (end[1] - start[1]) % U.shape[0] == 0, "path does not close"
    return np.trace(M) / U.shape[-1]


def rhombus_path(a, b):
    """Closed loop enclosing a*b rhombi = 2*a*b elementary triangles."""
    return ([(+1, 0)] * a + [(+1, 1)] * b + [(-1, 0)] * a + [(-1, 1)] * b)


PLAQ_DOWN_PARTNER = [(+1, 1), (-1, 0), (-1, 2)]   # completes the 1x1 rhombus


def gauge_transform(U, rng):
    """
    U[s,mu] transports colour FROM s TO s+mu, so a gauge rotation acts as
        U[s,mu] -> G(s+mu) U[s,mu] G(s)^dagger .
    """
    L, n = U.shape[0], U.shape[-1]
    if n == 1:
        G = np.exp(1j * rng.normal(size=(L, L)))
        V = U.copy()
        for mu in range(3):
            sh = np.roll(np.roll(G, -DIRS[mu, 0], axis=0), -DIRS[mu, 1], axis=1)
            V[:, :, mu, 0, 0] = sh * U[:, :, mu, 0, 0] * np.conj(G)
        return V
    G = np.empty((L, L, n, n), dtype=complex)
    for i in range(L):
        for j in range(L):
            G[i, j] = haar_su3(rng, n)
    V = np.empty_like(U)
    for i in range(L):
        for j in range(L):
            for mu in range(3):
                i2 = (i + DIRS[mu, 0]) % L
                j2 = (j + DIRS[mu, 1]) % L
                V[i, j, mu] = G[i2, j2] @ U[i, j, mu] @ G[i, j].conj().T
    return V


def loop_average(U, path, n_sites=None, rng=None):
    """<(1/N) Re tr W> over all starting sites (translation average)."""
    L = U.shape[0]
    vals = []
    if n_sites is None:
        pts = [(i, j) for i in range(L) for j in range(L)]
    else:
        pts = [(int(rng.integers(L)), int(rng.integers(L))) for _ in range(n_sites)]
    for s in pts:
        vals.append(wilson(U, s, path).real)
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))
