#!/usr/bin/env python3
"""
3+1D walker on the FCC lattice with an SU(2) coin
==================================================

Step 2 of ROADMAP_QCD_3D_de.md.

In the 2D model the heading lives on a circle, so a loop phase exp(i*2*pi*alpha*w)
is available for ANY real alpha: spin is a free knob that has to be set by hand
(alpha = 1/2 was chosen to get Kramers doubling and the Dirac Berry phase).

In 3D the heading lives on a sphere.  quantum_fcc_holonomy.py showed there is
no winding number and the holonomies do not commute, so no continuous alpha
exists.  What replaces it is the parallel transport itself: the walker carries
a spinor, and turning the heading rotates that spinor by the SU(2) element of
the minimal rotation.  **alpha is gone; what is left is the choice of
representation, spin 0 or spin 1/2 — boson or fermion.**

Geometry
--------
12 FCC headings (the cuboctahedron).  Turn angles are 60, 90, 120, 180 deg;
each heading has 4 neighbours at 60 deg.  The coin connects only those, so the
180 deg reversal — whose rotation axis is undefined and which forced an
ad-hoc convention in 2D — never enters.

Scaling follows the 2D convention (spacetime edge length 1):
    |step| = sqrt(3)/2,  dt = 1/2   =>   c = sqrt(3)
so integer FCC coordinates carry a lattice constant a = sqrt(6)/4.

Coin
----
    G = sum_{d' ~ d}  |d'><d| (x) Q(d -> d')          (Hermitian by construction)
    C = expm(i * eps * G)                              (exactly unitary)

with Q the SU(2) transport for spin 1/2, and Q = 1 for spin 0.  To first order
in eps this is "straight = 1, 60 deg turn = i*eps*(spinor rotation)" — the same
amplitude rule as before, with the turning PHASE replaced by the turning
ROTATION.
"""

import itertools
import math
import numpy as np
from scipy.linalg import expm

SQRT3 = np.sqrt(3.0)
DT_HALF = 0.5
STEP_LEN = SQRT3 / 2.0                 # physical length of one hop
C_LIGHT = STEP_LEN / DT_HALF           # = sqrt(3)
LAT_A = STEP_LEN / np.sqrt(2.0)        # physical length per integer unit

FCC_STEPS = np.array([v for v in itertools.product([-1, 0, 1], repeat=3)
                      if sum(abs(x) for x in v) == 2], dtype=int)
N_D = len(FCC_STEPS)                   # 12
FCC_UNIT = FCC_STEPS / np.sqrt(2.0)
MOVES_PHYS = FCC_UNIT * STEP_LEN       # physical displacement per hop

DOT = FCC_UNIT @ FCC_UNIT.T
ADJ60 = np.isclose(DOT, 0.5)           # cuboctahedron edges


# ─── SU(2) transport ─────────────────────────────────────────────────────────

def su2_transport(u, v):
    """
    2x2 SU(2) matrix of the minimal rotation taking heading u to heading v.
    Only called for 60 deg pairs, so the axis is always well defined.
    """
    c = float(np.dot(u, v))
    if c > 1 - 1e-12:
        return np.eye(2, dtype=complex)
    n = np.cross(u, v)
    n = n / np.linalg.norm(n)
    th = math.acos(max(-1.0, min(1.0, c)))
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    ns = n[0] * sx + n[1] * sy + n[2] * sz
    return math.cos(th / 2) * np.eye(2) + 1j * math.sin(th / 2) * ns


def generator(spin=0.5):
    """
    G = sum_{d'~d} |d'><d| (x) Q(d->d').  Hermitian: the reverse pair carries
    Q(d'->d) = Q(d->d')^dagger.
    """
    ns = 2 if spin == 0.5 else 1
    G = np.zeros((N_D * ns, N_D * ns), dtype=complex)
    for d in range(N_D):
        for dp in range(N_D):
            if not ADJ60[d, dp]:
                continue
            Q = su2_transport(FCC_UNIT[d], FCC_UNIT[dp]) if ns == 2 \
                else np.eye(1, dtype=complex)
            G[dp * ns:(dp + 1) * ns, d * ns:(d + 1) * ns] = Q
    return G


def coin(eps, spin=0.5):
    """C = expm(i*eps*G): exactly unitary."""
    return expm(1j * eps * generator(spin))


# ─── transfer matrix ─────────────────────────────────────────────────────────

def TM(k, eps, spin=0.5):
    """
    One-step transfer matrix at physical momentum k = (kx, ky, kz).

    Real-space rule: new[r + dr_d', d'] = sum_d C[d',d] psi[r, d]
      => T(k)[d',d] = exp(-i k.dr_d') C[d',d]
    """
    ns = 2 if spin == 0.5 else 1
    C = coin(eps, spin)
    ph = np.exp(-1j * (MOVES_PHYS @ np.asarray(k, dtype=float)))
    return np.repeat(ph, ns)[:, None] * C


def bands(k, eps, spin=0.5):
    lam, vec = np.linalg.eig(TM(k, eps, spin))
    E = -np.angle(lam) / DT_HALF
    o = np.argsort(E)
    return E[o], vec[:, o], np.abs(lam)[o]


def group_velocity(u, spin=0.5):
    """
    v_g = <dr>/dt, exact for a unitary coin (Hellmann-Feynman), so
    |v_g| <= |step|/dt = c is structural.
    """
    ns = 2 if spin == 0.5 else 1
    w = (np.abs(u) ** 2).reshape(N_D, ns).sum(axis=1)
    w = w / w.sum()
    return (w[:, None] * MOVES_PHYS).sum(axis=0) / DT_HALF


def rest_spectrum(eps, spin=0.5):
    """E at k=0: eigenvalues of G give E = -eps*lambda/dt."""
    lam = np.linalg.eigvalsh(generator(spin))
    return np.sort(-eps * lam / DT_HALF), np.sort(lam)


# ─── the belt trick on the lattice ───────────────────────────────────────────

def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)


def quat_transport(u, v):
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


def belt_trick(L_max=6, d0=0, only_60=True):
    """
    Closed walks that return to the same directed edge, classified by the SU(2)
    holonomy.  A walk whose heading path is a full 2*pi rotation returns the
    SO(3) identity but the SU(2) element -1: the spinor comes back with a minus
    sign.  That is the belt trick, measured on the lattice.

    Returns counts of (rounded scalar part of the holonomy quaternion).
    """
    QT = [[quat_transport(FCC_UNIT[i], FCC_UNIT[j]) for j in range(N_D)]
          for i in range(N_D)]
    allowed = [[j for j in range(N_D)
                if (ADJ60[i, j] if only_60 else QT[i][j] is not None)]
               for i in range(N_D)]
    hits = {}
    u0 = FCC_UNIT[d0]

    def rec(pos, d, q, depth):
        if depth > 0 and pos == (0, 0, 0) and d == d0:
            phi = 2.0 * math.atan2(float(np.dot(np.array(q[1:]), u0)), q[0])
            key = round(phi / (2 * np.pi), 6)
            hits[key] = hits.get(key, 0) + 1
        if depth == L_max:
            return
        if math.sqrt(sum(p * p for p in pos)) / math.sqrt(2.0) > (L_max - depth) + 1e-9:
            return
        for nd in allowed[d]:
            rec(tuple(p + s for p, s in zip(pos, FCC_STEPS[nd])),
                nd, qmul(QT[d][nd], q), depth + 1)

    rec((0, 0, 0), d0, (1.0, 0.0, 0.0, 0.0), 0)
    return hits


# ─── real-space simulation in 3+1D ───────────────────────────────────────────

def _shift3(a, s):
    """Shift a (N,N,N,...) array by integer FCC step s, zero fill."""
    out = np.zeros_like(a)
    sl_src, sl_dst = [], []
    for ax in range(3):
        d = int(s[ax])
        n = a.shape[ax]
        sl_src.append(slice(max(0, -d), n - max(0, d)))
        sl_dst.append(slice(max(0, d), n - max(0, -d)))
    out[tuple(sl_dst)] = a[tuple(sl_src)]
    return out


def step3(psi, C, ns):
    """One lattice step: turn (coin), then move each heading."""
    sh = psi.shape[:3]
    w = (psi.reshape(-1, N_D * ns) @ C.T).reshape(*sh, N_D, ns)
    new = np.zeros_like(psi)
    for d in range(N_D):
        new[..., d, :] = _shift3(w[..., d, :], FCC_STEPS[d])
    return new


def packet3(N, sigma, k_vec, eps, spin=0.5, band=0, dtype=np.complex64):
    """
    Gaussian packet on the FCC sublattice with the internal state set to the
    band eigenvector at k_vec.  Returns (psi, centre index, v_g predicted).
    """
    ns = 2 if spin == 0.5 else 1
    c = N // 2
    ix = np.arange(N) - c
    IX, IY, IZ = np.meshgrid(ix, ix, ix, indexing='ij')
    sub = ((IX + IY + IZ) % 2 == 0)
    X, Y, Z = IX * LAT_A, IY * LAT_A, IZ * LAT_A
    E, V, _ = bands(k_vec, eps, spin)
    u = V[:, band] / np.linalg.norm(V[:, band])
    vg = group_velocity(u, spin)
    env = (np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
           * np.exp(1j * (k_vec[0] * X + k_vec[1] * Y + k_vec[2] * Z)) * sub)
    psi = (env[..., None] * u[None, None, None, :]).astype(dtype)
    psi = psi / np.sqrt((np.abs(psi)**2).sum())
    return psi.reshape(N, N, N, N_D, ns), c, vg, E[band]


def com3(psi, c):
    p = (np.abs(psi)**2).sum(axis=(-2, -1))
    tot = p.sum()
    N = psi.shape[0]
    ix = (np.arange(N) - c) * LAT_A
    return np.array([(p.sum(axis=(1, 2)) * ix).sum() / tot,
                     (p.sum(axis=(0, 2)) * ix).sum() / tot,
                     (p.sum(axis=(0, 1)) * ix).sum() / tot]), float(tot)


def run_packet3(N, n_steps, k_vec, eps=0.1, spin=0.5, sigma=4.0, band=0,
                store_every=2):
    ns = 2 if spin == 0.5 else 1
    psi, c, vg, E = packet3(N, sigma, np.asarray(k_vec, float), eps, spin, band)
    C = coin(eps, spin).astype(psi.dtype)
    ts, tr, nm = [0.0], [com3(psi, c)[0]], [1.0]
    for t in range(1, n_steps + 1):
        psi = step3(psi, C, ns)
        if t % store_every == 0 or t == n_steps:
            r, tot = com3(psi, c)
            ts.append(t * DT_HALF); tr.append(r); nm.append(tot)
    return dict(t=np.array(ts), r=np.array(tr), norm=np.array(nm),
                vg=vg, E=E, psi=psi, c=c)
