#!/usr/bin/env python3
"""
Magnetic field on top of the turning-phase model
=================================================

Adds a position-dependent Peierls phase to `quantum_hex_turning`:

    A(r) = (B/2) * (-y, x)          (symmetric gauge)

A step that starts at r and moves by dr picks up

    exp( i * integral A.dl )  =  exp( i * (B/2) * (x*dy - y*dx) )

(the midpoint rule is exact here: the dx*dy/2 cross terms cancel).  The sum
of this phase around any closed walk is exactly B * (signed enclosed area),
because (1/2)*sum(x*dy - y*dx) is the shoelace formula.

Consequences
------------
* The evolution stays exactly unitary (diagonal phase x shift x unitary coin).
* Translation invariance is broken, so there is no k-space transfer matrix;
  everything here is real space plus time-domain spectroscopy.
* Left- and right-circulating orbits are no longer degenerate: this is the
  dynamical separation of the chirality families that the winding-number
  combinatorics only established statically.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from quantum_hex_turning import (
    SQRT3, DX_PHYS, DY_PHYS, DT_HALF, C_LIGHT, N_D,
    MOVES_PHYS, MOVES_IDX, DIR_COLORS, NSTEP,
    coin, bands, group_velocity, pick_band, rest_energy_top,
    _shift, make_grid, com_track,
)

AREA_TRI = (SQRT3 / 4.0) * (SQRT3 / 2.0) ** 2     # 3*sqrt(3)/16 = 0.32476


# ─── geometry helpers ────────────────────────────────────────────────────────

def phys_grid(nx, ny, cx, cy):
    IX, IY = np.meshgrid(np.arange(nx) - cx, np.arange(ny) - cy, indexing='ij')
    return IX * DX_PHYS, IY * DY_PHYS


def peierls_field(B, nx, ny, cx, cy):
    """ph[x, y, d] = exp(i*(B/2)*(x*dy_d - y*dx_d)), evaluated at the SOURCE."""
    X, Y = phys_grid(nx, ny, cx, cy)
    ph = np.empty((nx, ny, N_D), dtype=complex)
    for d in range(N_D):
        dx, dy = MOVES_PHYS[d]
        ph[:, :, d] = np.exp(1j * (B / 2.0) * (X * dy - Y * dx))
    return ph


def step_B(psi, C, ph):
    """One lattice step with a magnetic field: turn, gauge phase, move."""
    w = (psi @ C.T) * ph
    new = np.zeros_like(psi)
    for d in range(N_D):
        new[:, :, d] = _shift(w[:, :, d], MOVES_IDX[d, 0], MOVES_IDX[d, 1])
    return new


def angular_momentum(psi, C, ph, nx, ny, cx, cy):
    """
    L_z = sum_{r,d} |w_d(r)|^2 * (x*dy_d - y*dx_d) / dt

    i.e. twice the rate at which the probability sweeps area — the natural
    chirality order parameter.  Exactly 0 for B = 0 by mirror symmetry.
    """
    X, Y = phys_grid(nx, ny, cx, cy)
    w = (psi @ C.T) * ph
    tot = 0.0
    for d in range(N_D):
        dx, dy = MOVES_PHYS[d]
        tot += (np.abs(w[:, :, d])**2 * (X * dy - Y * dx)).sum()
    return tot / DT_HALF


# ─── verification: flux per plaquette ────────────────────────────────────────

def loop_flux(B, dirs):
    """
    Accumulated Peierls phase along a closed sequence of headings, starting
    at the origin.  Returns (phase, signed area).  Verifies phase = B * area.
    """
    x = y = 0.0
    tot = 0.0
    for d in dirs:
        dx, dy = MOVES_PHYS[d]
        tot += (B / 2.0) * (x * dy - y * dx)
        x += dx; y += dy
    assert abs(x) < 1e-12 and abs(y) < 1e-12, "walk is not closed"
    return tot, tot / B if B else 0.0


# ─── closed loops with signed enclosed area ──────────────────────────────────

def enumerate_loops_area(L_max=8, allow_reversal=False):
    """
    Closed walks returning to the same directed edge, classified by
    (winding number w, signed enclosed area S).

    Returns {L: {(w, S_rounded): count}}.
    """
    steps = [-2, -1, 0, 1, 2] + ([3] if allow_reversal else [])
    out = {L: {} for L in range(1, L_max + 1)}

    def rec(ix, iy, d, tsum, area2, depth):
        if depth > 0 and ix == 0 and iy == 0 and d == 0:
            key = (tsum // 6, round(area2 / 2.0, 6))
            out[depth][key] = out[depth].get(key, 0) + 1
        if depth == L_max:
            return
        px, py = ix * DX_PHYS, iy * DY_PHYS
        if np.hypot(px, py) / (SQRT3 / 2) > (L_max - depth) + 1e-9:
            return
        for s in steps:
            nd = (d + s) % 6
            dx, dy = MOVES_PHYS[nd]
            rec(ix + MOVES_IDX[nd, 0], iy + MOVES_IDX[nd, 1], nd,
                tsum + s, area2 + (px * dy - py * dx), depth + 1)

    rec(0, 0, 0, 0, 0.0, 0)
    return out


def return_amplitude(loops, alpha, B, eps):
    """
    Sum over closed walks of  eps^(#turns) * exp(i*2*pi*alpha*w) * exp(i*B*S),
    split into the left (w>0), right (w<0) and figure-eight (w=0) families.
    """
    fam = {"left": 0j, "right": 0j, "eight": 0j}
    for L, d in loops.items():
        for (w, S), n in d.items():
            a = n * (eps ** L) * np.exp(2j * np.pi * alpha * w) * np.exp(1j * B * S)
            fam["left" if w > 0 else ("right" if w < 0 else "eight")] += a
    return fam


# ─── wave packets in a field ─────────────────────────────────────────────────

def gaussian_packet_B(nx, ny, cx, cy, x0, y0, sigma, k_mag, ang_deg, eps, alpha):
    a = np.radians(ang_deg)
    kx, ky = k_mag * np.cos(a), k_mag * np.sin(a)
    _, u, vg = pick_band(kx, ky, eps, alpha)
    u = u / np.linalg.norm(u)
    IX, IY = np.meshgrid(np.arange(nx) - cx, np.arange(ny) - cy, indexing='ij')
    X, Y = IX * DX_PHYS, IY * DY_PHYS
    sub = ((IX + IY) % 2 == 0)
    env = (np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
           * np.exp(1j * (kx * X + ky * Y)) * sub)
    psi = env[:, :, None] * u[None, None, :]
    return psi / np.sqrt((np.abs(psi)**2).sum()), vg


# ─── band-projected wave packet (k-space construction) ───────────────────────

def band_projected_packet(nx, ny, cx, cy, x0, y0, sigma, k_mag, ang_deg,
                          eps, alpha, band=-1, thresh=1e-7):
    """
    Build a packet that lives in ONE band only.

    A packet whose internal spinor is fixed to u(k0) is *not* a pure band
    state away from k0: in the flanks of the packet k points in other
    directions, where the correct band eigenvector differs.  The admixture of
    the other five bands travels at different velocities and shows up as six
    ballistic jets along the lattice directions.

    Here the envelope is Fourier transformed, every k is dressed with its own
    band eigenvector u(k) (in a smooth gauge fixed by u(k0)), and the result
    is transformed back.  Because T(k + G) = T(k) exactly for the folding
    vector G = (pi/dx, pi/dy), the sublattice structure is preserved.
    """
    IX, IY = np.meshgrid(np.arange(nx) - cx, np.arange(ny) - cy, indexing='ij')
    X, Y = IX * DX_PHYS, IY * DY_PHYS
    sub = ((IX + IY) % 2 == 0)
    a = np.radians(ang_deg)
    kx0, ky0 = k_mag * np.cos(a), k_mag * np.sin(a)

    g = (np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
         * np.exp(1j * (kx0 * X + ky0 * Y)) * sub)
    gh = np.fft.fft2(g)

    KX = (2 * np.pi * np.fft.fftfreq(nx) / DX_PHYS)[:, None] * np.ones((1, ny))
    KY = (2 * np.pi * np.fft.fftfreq(ny) / DY_PHYS)[None, :] * np.ones((nx, 1))

    mask = np.abs(gh) > thresh * np.abs(gh).max()
    kx, ky = KX[mask], KY[mask]

    C = coin(eps, alpha, "unitary")
    ph = np.exp(-1j * (kx[:, None] * MOVES_PHYS[None, :, 0]
                       + ky[:, None] * MOVES_PHYS[None, :, 1]))     # (n_k, 6)
    T = ph[:, :, None] * C[None, :, :]                              # (n_k, 6, 6)
    lam, vec = np.linalg.eig(T)
    E = -np.angle(lam) / DT_HALF
    order = np.argsort(E, axis=1)
    sel = order[:, band]
    u = vec[np.arange(len(kx)), :, sel]                             # (n_k, 6)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)

    _, uref, _ = pick_band(kx0, ky0, eps, alpha)
    uref = uref / np.linalg.norm(uref)
    ov = u @ uref.conj()
    u = u * (np.conj(ov) / (np.abs(ov) + 1e-30))[:, None]           # smooth gauge

    psih = np.zeros((nx, ny, N_D), dtype=complex)
    psih[mask] = gh[mask][:, None] * u
    psi = np.fft.ifft2(psih, axes=(0, 1))
    return psi / np.sqrt((np.abs(psi)**2).sum())


def run_cyclotron(B, eps=0.1, alpha=0.0, k_mag=0.8, ang_deg=0.0, sigma=6.0,
                  n_steps=400, store_every=4, n_grid=None):
    """Propagate a boosted packet in a uniform field; return the CoM track."""
    if n_grid is None:
        # the packet starts ON the orbit, whose centre is offset by R, so it
        # reaches 2R from the origin; plus room for wave-packet spreading
        R = k_mag / max(abs(B), 1e-9)
        span = 2.4 * R + 8 * sigma
        nx = 2 * int(span / DX_PHYS) + 1
        ny = 2 * int(span / DY_PHYS) + 1
    else:
        nx, ny = n_grid
    cx, cy = nx // 2, ny // 2
    psi = band_projected_packet(nx, ny, cx, cy, 0.0, 0.0, sigma,
                                k_mag, ang_deg, eps, alpha)
    _, _, vg = pick_band(k_mag * np.cos(np.radians(ang_deg)),
                         k_mag * np.sin(np.radians(ang_deg)), eps, alpha)
    C = coin(eps, alpha, "unitary")
    ph = peierls_field(B, nx, ny, cx, cy)

    hist, times, lz = [psi.copy()], [0.0], [angular_momentum(psi, C, ph, nx, ny, cx, cy)]
    for t in range(1, n_steps + 1):
        psi = step_B(psi, C, ph)
        if t % store_every == 0 or t == n_steps:
            hist.append(psi.copy()); times.append(t * DT_HALF)
            lz.append(angular_momentum(psi, C, ph, nx, ny, cx, cy))
    return dict(hist=hist, times=np.array(times), lz=np.array(lz),
                nx=nx, ny=ny, cx=cx, cy=cy, vg=vg, B=B)


def excited_edge(nx, ny, cx, cy, d=0):
    """The original setup: a single excited directed edge at the origin."""
    psi = np.zeros((nx, ny, N_D), dtype=complex)
    psi[cx, cy, d] = 1.0
    return psi




# ─── time-domain spectroscopy: which orbits resonate ─────────────────────────

def autocorrelation(psi0, C, ph, n_steps, filters=None):
    """
    c(t) = <psi0 | psi(t)>, and optionally the spectral projections

        phi_n = sum_t w(t) * exp(i*E_n*t*dt) * psi(t)

    which pick out the eigenstate at E_n (a standing wave on the orbit).
    `filters` is a list of energies E_n.  w(t) is a Hann window.
    """
    psi = psi0.copy()
    c = np.zeros(n_steps + 1, dtype=complex)
    c[0] = 1.0
    phi = None
    if filters is not None:
        phi = [np.zeros_like(psi0) for _ in filters]
    win = lambda t: 0.5 * (1 - np.cos(2 * np.pi * t / n_steps))
    if phi is not None:
        for j, E in enumerate(filters):
            phi[j] += win(0) * psi
    for t in range(1, n_steps + 1):
        psi = step_B(psi, C, ph)
        c[t] = np.vdot(psi0, psi)
        if phi is not None:
            w = win(t)
            for j, E in enumerate(filters):
                phi[j] += w * np.exp(1j * E * t * DT_HALF) * psi
    return c, phi


def spectrum_from_autocorr(c, E_grid):
    """S(E) = |sum_t w(t) c(t) exp(i E t dt)|, Hann-windowed."""
    n = len(c) - 1
    t = np.arange(n + 1)
    w = 0.5 * (1 - np.cos(2 * np.pi * t / n))
    ph = np.exp(1j * E_grid[:, None] * t[None, :] * DT_HALF)
    return np.abs((ph * (w * c)[None, :]).sum(axis=1))


def find_peaks(E_grid, S, rel=0.15):
    """Local maxima of S above rel*max, with parabolic sub-grid refinement."""
    out = []
    thr = rel * S.max()
    for i in range(1, len(S) - 1):
        if S[i] > thr and S[i] >= S[i - 1] and S[i] > S[i + 1]:
            y0, y1, y2 = S[i - 1], S[i], S[i + 1]
            d = y0 - 2 * y1 + y2
            shift = 0.5 * (y0 - y2) / d if abs(d) > 1e-30 else 0.0
            out.append(E_grid[i] + shift * (E_grid[1] - E_grid[0]))
    return np.array(out)


# ─── Onsager quantisation: which orbit sizes reinforce ───────────────────────

def contour_area_k(E_target, eps, alpha, n_theta=48, k_max=3.0):
    """
    Area enclosed by the constant-energy contour of the top band in k space.
    Handles hexagonal warping by shooting a ray in each direction.
    """
    ks = np.linspace(1e-4, k_max, 600)
    A = 0.0
    for th in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
        cx_, sy_ = np.cos(th), np.sin(th)
        E = np.array([bands(k * cx_, k * sy_, eps, alpha)[0][-1] for k in ks])
        i = np.where(E >= E_target)[0]
        if len(i) == 0:
            return np.nan
        i = i[0]
        if i == 0:
            kr = ks[0]
        else:
            kr = ks[i - 1] + (E_target - E[i - 1]) * (ks[i] - ks[i - 1]) / (E[i] - E[i - 1])
        A += 0.5 * kr**2 * (2 * np.pi / n_theta)
    return A


def onsager_index(E_levels, B, eps, alpha):
    """
    A_k(E_n) / (2*pi*B).  Semiclassically this equals n + gamma, with
    gamma = 1/2 for an ordinary band and gamma = 0 for a Dirac point
    (Berry phase pi) — the same spinor factor as alpha = 1/2.
    """
    return np.array([contour_area_k(E, eps, alpha) / (2 * np.pi * B)
                     for E in E_levels])
