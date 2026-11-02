#!/usr/bin/env python3
"""
Turning-Phase Model on the 2D Triangular (Hex) Lattice
=======================================================

Idea
----
A directed edge (node + heading d) is excited.  Each step the walk turns by
a multiple of 60 degrees.  The *sum of turning angles* along a path is used
as the path-integral phase:

    w_path = prod_steps  a(|n|) * exp(i * alpha * 60deg * n)

where n = signed number of 60-degree units turned in that step.

Key facts (see RESULTS_Turning_2D_de.md):

1.  For any closed walk that returns to the SAME directed edge the total
    turning is exactly 360deg * (integer winding number w).  This is the
    discrete Whitney turning-number theorem and it is automatic, not a
    coincidence.  Hence exp(i*Sum(theta)) == 1 for every loop: the raw
    angle sum carries NO interference.  The scale factor alpha is what
    makes the construction physical.

2.  The loop phase is exp(i * 2*pi * alpha * w).  alpha is therefore an
    Aharonov-Bohm flux in *heading space*, physical only mod 1:
    alpha -> alpha + 1 is a gauge transformation psi_d -> e^{i*60deg*d} psi_d.

3.  alpha = 1/2  =>  loop phase (-1)^w : the spinor double cover (spin 1/2).
    At k = 0 the whole rest-energy spectrum becomes exactly 2-fold
    degenerate (Kramers).

4.  Motion requires a phase that depends on the STEP DISPLACEMENT
    (Peierls phase exp(i k.dr)).  A constant extra angle per step -- or
    every n-th step -- is a pure global phase and can never move anything.

Geometry (identical to quantum_hex_2d.py)
-----------------------------------------
6 headings at 0,60,...,300 deg, spatial step sqrt(3)/2, half time step 0.5,
so the spacetime edge length is 1 and c = (sqrt(3)/2)/0.5 = sqrt(3).

Index space:  dx_idx unit = sqrt(3)/4,  dy_idx unit = 3/4
  d=0: (+2, 0)   d=1: (+1,+1)   d=2: (-1,+1)
  d=3: (-2, 0)   d=4: (-1,-1)   d=5: (+1,-1)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

# ─── geometry ────────────────────────────────────────────────────────────────

SQRT3    = np.sqrt(3)
DX_PHYS  = SQRT3 / 4          # physical x per x-index
DY_PHYS  = 0.75               # physical y per y-index
DT_HALF  = 0.5                # physical time per lattice step
C_LIGHT  = SQRT3              # (sqrt(3)/2) / 0.5
N_D      = 6                  # headings

MOVES_PHYS = np.array([
    [ SQRT3/2,  0.00],   # d=0    0 deg
    [ SQRT3/4,  0.75],   # d=1   60 deg
    [-SQRT3/4,  0.75],   # d=2  120 deg
    [-SQRT3/2,  0.00],   # d=3  180 deg
    [-SQRT3/4, -0.75],   # d=4  240 deg
    [ SQRT3/4, -0.75],   # d=5  300 deg
])

MOVES_IDX = np.array([[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1]])

DIR_COLORS = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']


def turn_steps():
    """NSTEP[d_new, d_old] = signed number of 60-deg units turned, in -2..+3.

    +3 is the 180-degree reversal, whose turning sense is genuinely ambiguous
    (+3 and -3 describe the same move).  It is excluded by default.
    """
    d_new = np.arange(N_D)[:, None]
    d_old = np.arange(N_D)[None, :]
    return ((d_new - d_old + 2) % 6) - 2


NSTEP = turn_steps()


# ─── the coin (turning) operator ─────────────────────────────────────────────

def rotation_op():
    """R with (R v)_d = v_{d-1}: rotate the heading by +60 deg."""
    R = np.zeros((N_D, N_D), dtype=complex)
    for d in range(N_D):
        R[(d + 1) % N_D, d] = 1.0
    return R


def coin(eps, alpha, mode="unitary", allow_reversal=False):
    """
    6x6 turning operator C[d_new, d_old].

    mode = "unitary"  (recommended)
        C = expm(i * eps * G_alpha),  G_alpha = e^{i pi alpha/3} R + h.c.
        Exactly unitary; to first order in eps it reproduces
        "straight -> 1, +-60deg turn -> i*eps".  Multi-step turns and the
        180deg move are generated consistently, so no sign ambiguity arises.

    mode = "graded"
        w(n) = (i*eps)^{|n|} * exp(i*alpha*60deg*n)   -- every turn is a
        product of elementary 60deg turns.

    mode = "flat"
        w(0) = 1, w(n != 0) = i*eps * exp(i*alpha*60deg*n)
        With alpha = 0 and allow_reversal = True this is exactly the
        amplitude rule of quantum_hex_2d.py (restricted to the 6 diagonal
        moves, i.e. without the "straight/rest" direction).

    Parameters
    ----------
    eps   : mass parameter (turn amplitude)
    alpha : turning phase per 360 deg of heading rotation.  Physical mod 1.
            alpha = 0.5 gives the spinor (-1)^winding.
    """
    if mode == "unitary":
        R  = rotation_op()
        ph = np.exp(1j * np.pi * alpha / 3.0)
        G  = ph * R + np.conj(ph) * R.conj().T
        return expm(1j * eps * G)

    W = np.zeros((N_D, N_D), dtype=complex)
    for dn in range(N_D):
        for do in range(N_D):
            n = NSTEP[dn, do]
            if n == 0:
                W[dn, do] = 1.0
            elif abs(n) <= 2:
                amp = (1j * eps) if mode == "flat" else (1j * eps) ** abs(n)
                W[dn, do] = amp * np.exp(1j * np.pi * alpha * n / 3.0)
            else:                                  # n == 3: 180deg reversal
                if allow_reversal:
                    amp = (1j * eps) if mode == "flat" else (1j * eps) ** 3
                    # symmetric over both turning senses (+3 and -3)
                    W[dn, do] = amp * np.cos(np.pi * alpha)
    return W


# ─── transfer matrix / bands ─────────────────────────────────────────────────

def TM(kx, ky, eps, alpha, mode="unitary", extra_phase=0.0):
    """
    6x6 one-step transfer matrix at physical momentum (kx, ky).

    Real-space rule:  new[r + dr_{d'}, d'] = sum_d C[d',d] * psi[r, d]
    => T(k)[d',d] = exp(-i k.dr_{d'}) * C[d',d]

    extra_phase is a *direction-independent* phase per step -- the naive
    "add a constant amount every step" idea.  It multiplies every eigenvalue
    by the same factor and therefore only shifts the energy; it cannot
    produce motion.
    """
    C  = coin(eps, alpha, mode)
    ph = np.exp(-1j * (kx * MOVES_PHYS[:, 0] + ky * MOVES_PHYS[:, 1]))
    return np.exp(1j * extra_phase) * ph[:, None] * C


def bands(kx, ky, eps, alpha, mode="unitary", extra_phase=0.0):
    """Sorted energies E_j(k) and eigenvectors.  E = -arg(lambda)/dt."""
    lam, vec = np.linalg.eig(TM(kx, ky, eps, alpha, mode, extra_phase))
    E = -np.angle(lam) / DT_HALF
    o = np.argsort(E)
    return E[o], vec[:, o], np.abs(lam)[o]


def group_velocity(vec_col):
    """
    v_g = <dr>_u / dt, exact for a unitary coin.

    Derivation: T(k) = diag(e^{-i k.dr_d}) C, so dT/dk = diag(-i dr_d) T and
    Hellmann-Feynman gives dE/dk = <dr>/dt with <dr> = sum_d |u_d|^2 dr_d.
    Because |<dr>| <= sqrt(3)/2, causality |v| <= c = sqrt(3) is automatic.
    """
    w = np.abs(vec_col) ** 2
    w = w / w.sum()
    return (w[:, None] * MOVES_PHYS).sum(axis=0) / DT_HALF


def rest_spectrum(eps, alpha, mode="unitary"):
    """Rest energies at k = 0 (analytic for mode='unitary')."""
    E, _, _ = bands(0.0, 0.0, eps, alpha, mode)
    return E


def rest_spectrum_analytic(eps, alpha):
    """E_m = -2*eps*cos(pi*(alpha-m)/3)/dt for the unitary coin, m = 0..5."""
    m = np.arange(N_D)
    return np.sort(-2.0 * eps * np.cos(np.pi * (alpha - m) / 3.0) / DT_HALF)


# ─── closed loops and winding numbers ────────────────────────────────────────

def enumerate_loops(L_max=9, allow_reversal=False):
    """
    All closed walks of length <= L_max that start and end on the same
    directed edge (origin, heading 0).

    Returns {L: {winding: count}}.  Verifies that the total turning of every
    such walk is a multiple of 360 deg.
    """
    steps  = [-2, -1, 0, 1, 2] + ([3] if allow_reversal else [])
    counts = {L: {} for L in range(1, L_max + 1)}
    bad    = [0]

    def rec(ix, iy, d, tsum, depth):
        if depth > 0 and ix == 0 and iy == 0 and d == 0:
            if tsum % 6 != 0:
                bad[0] += 1
            counts[depth][tsum // 6] = counts[depth].get(tsum // 6, 0) + 1
        if depth == L_max:
            return
        # prune: cannot get home in the remaining steps
        dist = np.hypot(ix * DX_PHYS, iy * DY_PHYS) / (SQRT3 / 2)
        if dist > (L_max - depth) + 1e-9:
            return
        for s in steps:
            nd = (d + s) % 6
            rec(ix + MOVES_IDX[nd, 0], iy + MOVES_IDX[nd, 1], nd,
                tsum + s, depth + 1)

    rec(0, 0, 0, 0, 0)
    return counts, bad[0]


def loop_structure_factor(counts, alpha_arr):
    """A(alpha) = sum_loops exp(i 2 pi alpha w), summed over all lengths."""
    tot = {}
    for c in counts.values():
        for w, n in c.items():
            tot[w] = tot.get(w, 0) + n
    A = np.zeros_like(alpha_arr, dtype=complex)
    for w, n in tot.items():
        A += n * np.exp(2j * np.pi * alpha_arr * w)
    return A, tot


# ─── real-space simulation ───────────────────────────────────────────────────

def _shift(a, sx, sy):
    """Shift a 2D-plus-channel array by (sx, sy) index units, zero fill."""
    out = np.zeros_like(a)
    nx, ny = a.shape[0], a.shape[1]
    xs = slice(max(0, -sx), nx - max(0, sx))
    xd = slice(max(0, sx),  nx - max(0, -sx))
    ys = slice(max(0, -sy), ny - max(0, sy))
    yd = slice(max(0, sy),  ny - max(0, -sy))
    out[xd, yd] = a[xs, ys]
    return out


def step(psi, C, extra_phase=0.0):
    """One lattice step: turn (coin), then move."""
    w = psi @ C.T                       # w[...,d'] = sum_d C[d',d] psi[...,d]
    if extra_phase:
        w = w * np.exp(1j * extra_phase)
    new = np.zeros_like(psi)
    for d in range(N_D):
        new[:, :, d] = _shift(w[:, :, d], MOVES_IDX[d, 0], MOVES_IDX[d, 1])
    return new


def make_grid(n_steps, margin_idx=30):
    nx = 2 * (2 * n_steps + margin_idx) + 1
    ny = 2 * (1 * n_steps + margin_idx) + 1
    return nx, ny, nx // 2, ny // 2


def pick_band(kx, ky, eps, alpha, mode="unitary", prefer=None, extra_phase=0.0):
    """
    Pick the band whose group velocity points along `prefer` (default: k).
    Returns (E, u, v_g).
    """
    E, V, _ = bands(kx, ky, eps, alpha, mode, extra_phase)
    if prefer is None:
        prefer = np.array([kx, ky])
    nrm = np.linalg.norm(prefer)
    ref = prefer / nrm if nrm > 1e-12 else np.array([1.0, 0.0])
    best, best_score = 0, -np.inf
    for j in range(N_D):
        v = group_velocity(V[:, j])
        s = float(v @ ref)
        if s > best_score:
            best, best_score = j, s
    return E[best], V[:, best], group_velocity(V[:, best])


def gaussian_packet(nx, ny, cx, cy, sigma_phys, kx, ky, u):
    """Gaussian envelope * plane wave * internal spinor u, on the sublattice."""
    ix = (np.arange(nx) - cx)
    iy = (np.arange(ny) - cy)
    IX, IY = np.meshgrid(ix, iy, indexing='ij')
    X, Y = IX * DX_PHYS, IY * DY_PHYS
    sub = ((IX + IY) % 2 == 0)          # only this sublattice is reachable
    env = np.exp(-(X**2 + Y**2) / (2 * sigma_phys**2)) * np.exp(1j*(kx*X + ky*Y))
    env = env * sub
    psi = env[:, :, None] * (u / np.linalg.norm(u))[None, None, :]
    return psi / np.sqrt((np.abs(psi)**2).sum())


def com_track(psi_hist, cx, cy):
    """Centre of mass in physical units for each stored frame."""
    out = []
    nx, ny = psi_hist[0].shape[0], psi_hist[0].shape[1]
    IX, IY = np.meshgrid(np.arange(nx) - cx, np.arange(ny) - cy, indexing='ij')
    X, Y = IX * DX_PHYS, IY * DY_PHYS
    for psi in psi_hist:
        p = (np.abs(psi)**2).sum(axis=-1)
        s = p.sum()
        out.append([(X * p).sum() / s, (Y * p).sum() / s, s])
    return np.array(out)


def run_packet(n_steps, eps=0.5, alpha=0.5, mode="unitary", sigma_phys=6.0,
               k_mag=0.6, angle_deg=0.0, extra_phase=0.0, store_every=5,
               margin_idx=30):
    """Propagate one boosted Gaussian packet; returns history and diagnostics."""
    ang = np.radians(angle_deg)
    kx, ky = k_mag * np.cos(ang), k_mag * np.sin(ang)
    E, u, vg = pick_band(kx, ky, eps, alpha, mode, extra_phase=extra_phase)

    nx, ny, cx, cy = make_grid(n_steps, margin_idx)
    psi = gaussian_packet(nx, ny, cx, cy, sigma_phys, kx, ky, u)

    C = coin(eps, alpha, mode)
    hist, times = [psi.copy()], [0.0]
    for t in range(1, n_steps + 1):
        psi = step(psi, C, extra_phase)
        if t % store_every == 0 or t == n_steps:
            hist.append(psi.copy())
            times.append(t * DT_HALF)
    return dict(hist=hist, times=np.array(times), cx=cx, cy=cy,
                kx=kx, ky=ky, E=E, u=u, vg=vg, nx=nx, ny=ny)


# ─── mass / cone diagnostics ─────────────────────────────────────────────────

def top_band(k, eps, alpha, mode="unitary"):
    """Highest-energy band at k = (k, 0): the 'particle' band."""
    E, V, _ = bands(k, 0.0, eps, alpha, mode)
    return E[-1], V[:, -1]


def rest_energy_top(eps, alpha):
    """
    Analytic top rest level:  m(alpha) = 4*eps*cos(pi*delta/3)
    with delta = distance from alpha to the nearest integer (in [0, 1/2]).
    """
    delta = abs(alpha - np.round(alpha))
    return 4.0 * eps * np.cos(np.pi * delta / 3.0)


def cone_slope(eps, alpha, k=1e-3):
    """
    Splitting rate of the top *pair* of bands, (E5 - E4) / (2k).

    Away from alpha = 1/2 the top band is non-degenerate and this is a plain
    band gap -> the value diverges as k -> 0 is *not* the case; instead the
    pair splitting stays finite and the top band is quadratic (massive).
    At alpha = 1/2 the top level is 2-fold degenerate and splits linearly:
    the slope converges to exactly sqrt(3)/2 = c/2, independent of eps.
    """
    E, _, _ = bands(k, 0.0, eps, alpha, "unitary")
    return (E[-1] - E[-2]) / (2.0 * k)


def band_slope_top(eps, alpha, k=1e-3):
    """dE/dk of the top band at k -> 0.  ~0 = massive, finite = massless."""
    E0 = rest_energy_top(eps, alpha)
    E, _ = top_band(k, eps, alpha)
    return (E - E0) / k
