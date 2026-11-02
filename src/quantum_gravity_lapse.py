#!/usr/bin/env python3
"""
Gravity as Computation Time — Experiment 1a: the internal-clock lapse
=====================================================================

Idea (Paper 3, see docs/ROADMAP_Paper3_de.md)
---------------------------------------------
Near a large mass the lattice nodes "compute more slowly".  The cheapest,
strictly local and exactly unitary way to slow a node down in the
turning-phase model is to slow its *internal clock*: the turning rate.
Every node gets a lapse N(r) and turns with

    eps(r) = eps0 * N(r),        C(r) = expm(i * eps(r) * G_alpha)

The shift (the light-cone hop) is untouched.  Mass in this model IS the
turning rate (discrete Zitterbewegung), so this is "fewer turns per tick
near the mass".

What is measured
----------------
1.  Equivalence principle.  Semiclassically dk/dt = -dE/dx, and at k = 0
    the rest energy is exactly linear in eps (E_top = 4*eps at alpha = 0),
    so the force is F = -m0 * grad N.  The inertia is the band curvature
    1/m*.  Free fall is universal iff m0/m* does not depend on eps0:
        a = -(m0/m*) * grad N  =:  -R * c^2 * grad N
    GR (Newtonian limit, N = sqrt(g00)) gives R = 1.
2.  Coupling of moving particles.  The clock lapse couples to
    kappa(k) = eps * dE/deps / E   (Dirac: m dE/dm / E = m^2/E^2 = 1/gamma^2).
    A full lapse (all of the computation slowed) couples to E itself,
    kappa = 1.  kappa -> 0 for fast particles means: no light bending —
    the behaviour of Nordström's scalar gravity (1913).
3.  Real-space check.  Packets in a linear lapse N = 1 + g*x, propagated
    with the position-dependent coin, compared with a semiclassical
    ensemble that uses the exact lattice band (no parabolic approximation).

dE/deps is exact by Hellmann–Feynman: T = P(k) C, dC/deps = i C G, so
d(lambda)/deps = i*lambda*<u|G|u> and dE/deps = -<u|G|u>/dt.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_hex_turning import (C_LIGHT, DT_HALF, DX_PHYS, DY_PHYS, MOVES_IDX,
                                 MOVES_PHYS, N_D, _shift, rotation_op)
from paths import data_path

C2 = C_LIGHT ** 2


# ─── generator and band quantities ───────────────────────────────────────────

def generator(alpha):
    """Hermitian turning generator G_alpha (C = expm(i eps G))."""
    R = rotation_op()
    ph = np.exp(1j * np.pi * alpha / 3.0)
    return ph * R + np.conj(ph) * R.conj().T


def top_band_batch(kx, ky, eps, alpha):
    """
    Top band at arrays of (kx, ky, eps): returns E, v_g (n,2), dE/deps.
    eps may be a scalar or an array of the same length as kx.
    """
    kx = np.atleast_1d(kx).astype(float)
    ky = np.atleast_1d(ky).astype(float)
    eps = np.broadcast_to(np.asarray(eps, float), kx.shape)
    G = generator(alpha)
    g, V = np.linalg.eigh(G)
    C = np.einsum('ij,nj,kj->nik', V, np.exp(1j * eps[:, None] * g[None, :]),
                  V.conj())                                     # (n,6,6)
    ph = np.exp(-1j * (kx[:, None] * MOVES_PHYS[None, :, 0]
                       + ky[:, None] * MOVES_PHYS[None, :, 1]))
    T = ph[:, :, None] * C
    lam, U = np.linalg.eig(T)
    E = -np.angle(lam) / DT_HALF
    j = np.argmax(E, axis=1)
    n = np.arange(len(kx))
    u = U[n, :, j]
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    w = np.abs(u) ** 2
    vg = (w @ MOVES_PHYS) / DT_HALF
    dEde = -np.real(np.einsum('ni,ij,nj->n', u.conj(), G, u)) / DT_HALF
    return E[n, j], vg, dEde


def equivalence_ratio(eps0, alpha, dk=None):
    """R = (m0/m*)/c^2 with m0 = eps*dE/deps and 1/m* = d2E/dk2 at k = 0."""
    if dk is None:
        dk = 0.02 * eps0
    E, _, d = top_band_batch(np.array([-dk, 0.0, dk]), np.zeros(3), eps0, alpha)
    curv = (E[0] - 2 * E[1] + E[2]) / dk ** 2
    m0 = eps0 * d[1]
    return m0 * curv / C2, m0, 1.0 / curv


def kappa_curve(k, eps0, alpha):
    """kappa(k) = eps dE/deps / E  and the Dirac comparison (m0/E)^2."""
    E, vg, d = top_band_batch(k, np.zeros_like(k), eps0, alpha)
    m0 = top_band_batch(np.array([0.0]), np.array([0.0]), eps0, alpha)[0][0]
    return eps0 * d / E, (m0 / E) ** 2, E, np.linalg.norm(vg, axis=1)


# ─── real-space evolution with a position-dependent coin ─────────────────────

class LapseWalker:
    """Turning-phase walker with eps(r) = eps0 * N(r) on a fixed grid."""

    def __init__(self, nx, ny, eps0, alpha, lapse):
        self.nx, self.ny = nx, ny
        self.cx, self.cy = nx // 2, ny // 2
        IX, IY = np.meshgrid(np.arange(nx) - self.cx, np.arange(ny) - self.cy,
                             indexing='ij')
        self.X, self.Y = IX * DX_PHYS, IY * DY_PHYS
        self.sub = ((IX + IY) % 2 == 0)
        self.N = lapse(self.X, self.Y)
        if np.any(self.N <= 0):
            raise ValueError("lapse must stay positive on the grid")
        self.eps0, self.alpha = eps0, alpha
        g, V = np.linalg.eigh(generator(alpha))
        self.V, self.Vh = V, V.conj().T
        # phases of the local coin in the G eigenbasis, per site
        self.phase = np.exp(1j * eps0 * self.N[:, :, None] * g[None, None, :])

    def step(self, psi):
        w = ((psi @ self.Vh.T) * self.phase) @ self.V.T      # C(r) psi(r)
        new = np.zeros_like(psi)
        for d in range(N_D):
            new[:, :, d] = _shift(w[:, :, d], MOVES_IDX[d, 0], MOVES_IDX[d, 1])
        return new

    def packet(self, x0, y0, sigma, kx0, ky0, thresh=1e-7):
        """
        Band-projected top-band packet: every Fourier component carries its
        own top-band eigenvector of the UNIFORM walk with eps0 (smooth gauge).
        """
        nx, ny = self.nx, self.ny
        gfun = (np.exp(-((self.X - x0) ** 2 + (self.Y - y0) ** 2) / (2 * sigma ** 2))
                * np.exp(1j * (kx0 * self.X + ky0 * self.Y)) * self.sub)
        gh = np.fft.fft2(gfun)
        KX = (2 * np.pi * np.fft.fftfreq(nx) / DX_PHYS)[:, None] * np.ones((1, ny))
        KY = (2 * np.pi * np.fft.fftfreq(ny) / DY_PHYS)[None, :] * np.ones((nx, 1))
        mask = np.abs(gh) > thresh * np.abs(gh).max()
        kx, ky = KX[mask], KY[mask]

        G = generator(self.alpha)
        g, V = np.linalg.eigh(G)
        C = V @ np.diag(np.exp(1j * self.eps0 * g)) @ V.conj().T

        def top_vecs(kxa, kya):
            ph = np.exp(-1j * (kxa[:, None] * MOVES_PHYS[None, :, 0]
                               + kya[:, None] * MOVES_PHYS[None, :, 1]))
            lam, U = np.linalg.eig(ph[:, :, None] * C[None])
            j = np.argmax(-np.angle(lam), axis=1)
            u = U[np.arange(len(kxa)), :, j]
            return u / np.linalg.norm(u, axis=1, keepdims=True)

        u = top_vecs(kx, ky)
        uref = top_vecs(np.array([kx0]), np.array([ky0]))[0]
        ov = u @ uref.conj()
        u = u * (np.conj(ov) / (np.abs(ov) + 1e-30))[:, None]

        psih = np.zeros((nx, ny, N_D), dtype=complex)
        psih[mask] = gh[mask][:, None] * u
        psi = np.fft.ifft2(psih, axes=(0, 1))
        return psi / np.sqrt((np.abs(psi) ** 2).sum())

    def com(self, psi):
        p = (np.abs(psi) ** 2).sum(axis=-1)
        s = p.sum()
        return (self.X * p).sum() / s, (self.Y * p).sum() / s, s


def run_fall(eps0, alpha=0.0, g=5e-4, sigma=12.0, k0=(0.0, 0.0), n_steps=300,
             half_x=200.0, half_y=150.0, every=5):
    """Packet in the linear lapse N = 1 + g*x.  Returns times and CoM track."""
    nx = 2 * int(half_x / DX_PHYS) + 1
    ny = 2 * int(half_y / DY_PHYS) + 1
    walker = LapseWalker(nx, ny, eps0, alpha, lambda X, Y: 1.0 + g * X)
    psi = walker.packet(0.0, 0.0, sigma, *k0)
    ts, track = [0.0], [walker.com(psi)]
    for t in range(1, n_steps + 1):
        psi = walker.step(psi)
        if t % every == 0:
            ts.append(t * DT_HALF)
            track.append(walker.com(psi))
    return np.array(ts), np.array(track)


# ─── semiclassical ensemble with the exact lattice band ──────────────────────

def semiclassical(eps0, alpha=0.0, g=5e-4, sigma=12.0, k0=(0.0, 0.0),
                  t_max=150.0, dt=0.5, n_k=21):
    """
    Ensemble of (x, k) trajectories with initial |psi(k)|^2 ∝ exp(-|k-k0|^2 sigma^2):
        dx/dt = v_g(k; eps(x)),   dk/dt = -g * eps0 * dE/deps(k; eps(x)) * x_hat
    Returns times and the weighted mean position.
    """
    s = 3.0 / sigma
    q = np.linspace(-s, s, n_k)
    QX, QY = np.meshgrid(q, q, indexing='ij')
    kx = (QX + k0[0]).ravel()
    ky = (QY + k0[1]).ravel()
    wts = np.exp(-(QX ** 2 + QY ** 2).ravel() * sigma ** 2)
    wts /= wts.sum()
    x = np.zeros_like(kx)
    y = np.zeros_like(kx)

    def rhs(x, kx, ky):
        e = eps0 * (1.0 + g * x)
        _, vg, d = top_band_batch(kx, ky, e, alpha)
        return vg[:, 0], vg[:, 1], -g * eps0 * d

    ts, mx, my = [0.0], [0.0], [0.0]
    n = int(round(t_max / dt))
    for i in range(n):                                   # RK2 (midpoint)
        vx, vy, fx = rhs(x, kx, ky)
        vx2, vy2, fx2 = rhs(x + 0.5 * dt * vx, kx + 0.5 * dt * fx, ky)
        x = x + dt * vx2
        y = y + dt * vy2
        kx = kx + dt * fx2
        ts.append((i + 1) * dt)
        mx.append((wts * x).sum())
        my.append((wts * y).sum())
    return np.array(ts), np.array(mx), np.array(my)


def fit_accel(t, x, t_min=0.0):
    """Fit x = x0 + v0 t + a t^2 / 2 and return a."""
    m = t >= t_min
    A = np.vstack([np.ones(m.sum()), t[m], 0.5 * t[m] ** 2]).T
    return np.linalg.lstsq(A, x[m], rcond=None)[0][2]


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    out = {}
    t0 = time.time()

    # 1. equivalence ratio R(eps0) for several alpha
    eps_list = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    for alpha in (0.0, 0.25):
        R = np.array([equivalence_ratio(e, alpha)[0] for e in eps_list])
        out[f'R_alpha{alpha}'] = R
        print(f"alpha={alpha}: R(eps0) =", np.round(R, 5))
    out['eps_list'] = eps_list

    # 2. coupling of moving particles
    k = np.linspace(0.0, 3.0, 61)
    for e in (0.1, 0.3):
        kap, dirac, E, v = kappa_curve(k, e, 0.0)
        out[f'kappa_eps{e}'], out[f'kdirac_eps{e}'] = kap, dirac
        out[f'E_eps{e}'], out[f'v_eps{e}'] = E, v
    out['k_kappa'] = k

    # 3. real-space free fall vs semiclassics
    g = 5e-4
    fall_eps = [0.2, 0.3, 0.5, 0.7]
    a_sim, a_sc, a_par = [], [], []
    for e in fall_eps:
        ts, tr = run_fall(e, g=g)
        tsc, mx, _ = semiclassical(e, g=g)
        out[f'fall_t_eps{e}'], out[f'fall_x_eps{e}'] = ts, tr[:, 0]
        out[f'fall_norm_eps{e}'] = tr[:, 2]
        out[f'sc_t_eps{e}'], out[f'sc_x_eps{e}'] = tsc, mx
        R0 = equivalence_ratio(e, 0.0)[0]
        a_sim.append(fit_accel(ts, tr[:, 0]))
        a_sc.append(fit_accel(tsc, mx))
        a_par.append(-R0 * C2 * g)
        print(f"fall eps0={e}: a_sim={a_sim[-1]:.4e}  a_semicl={a_sc[-1]:.4e}  "
              f"a_parab={a_par[-1]:.4e}  a_GR={-C2*g:.4e}  norm={tr[-1,2]:.6f}  "
              f"[{time.time()-t0:.0f}s]")
    out['fall_eps'] = np.array(fall_eps)
    out['a_sim'], out['a_sc'], out['a_par'] = map(np.array, (a_sim, a_sc, a_par))
    out['g'] = g

    # 4. transverse deflection of moving packets (field along x, motion along y)
    e = 0.3
    defl_k = [0.0, 0.4, 0.8, 1.2]
    ax_sim, ax_sc, ax_full = [], [], []
    for kk in defl_k:
        ts, tr = run_fall(e, g=g, k0=(0.0, kk), n_steps=240, half_x=150.0,
                          half_y=260.0)
        tsc, mx, my = semiclassical(e, g=g, k0=(0.0, kk), t_max=120.0)
        out[f'defl_t_k{kk}'], out[f'defl_x_k{kk}'] = ts, tr[:, 0]
        out[f'defl_y_k{kk}'] = tr[:, 1]
        out[f'dsc_t_k{kk}'], out[f'dsc_x_k{kk}'] = tsc, mx
        ax_sim.append(fit_accel(ts, tr[:, 0]))
        ax_sc.append(fit_accel(tsc, mx))
        # full lapse (couples to E): a_x = -g * E * v/k  (isotropic estimate)
        E, vg, d = top_band_batch(np.array([0.0]), np.array([max(kk, 1e-4)]), e, 0.0)
        if kk > 0:
            ax_full.append(-g * E[0] * vg[0, 1] / kk)
        else:                          # at rest both lapses coincide
            ax_full.append(-g * equivalence_ratio(e, 0.0)[0] * C2)
        print(f"deflection k={kk}: a_x sim={ax_sim[-1]:.4e}  semicl={ax_sc[-1]:.4e}  "
              f"full-lapse={ax_full[-1]:.4e}  [{time.time()-t0:.0f}s]")
    out['defl_k'] = np.array(defl_k)
    out['ax_sim'], out['ax_sc'], out['ax_full'] = map(np.array, (ax_sim, ax_sc, ax_full))

    np.savez(data_path('gravity_lapse.npz'), **out)
    print(f"saved data/gravity_lapse.npz  [{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()
