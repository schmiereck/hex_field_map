#!/usr/bin/env python3
"""
Paper 3, Experiment S2 — Dynamics of spin vortices from phase bits
==================================================================

The walker of S1 lives on the double cover: label N in Z12 (net turning
mod 12), heading d = N mod 6, phase bit p = N mod 4.  Here the dynamics is
made exactly unitary:

    C = expm(i * (eps * G0 + mu * G3))
    G0 = S + S^-1                 +-60 deg turns           (integer generator)
    G3 = i * (S^3 - S^-3)          180 deg reversal that always counts as a
                                   LEFT turn (N -> N+3, phase bit +3)
    S : N -> N + 1

then each label hops along its heading d = N mod 6.

Facts tested
------------
1.  Superselection.  The 360-deg rotation R6 : N -> N+6 commutes with every
    step (hop depends on N mod 6 only; G0, G3 are functions of S).  The spin
    sector (R6 = -1, half-quantum vortices) and the boson sector (R6 = +1)
    never mix — even with a random position-dependent eps(r).
2.  mu = 0: the spin sector is the turning model at alpha = 1/2 (densities
    identical).  The vortex doublet at k = 0 splits LINEARLY (cone slope c/2):
    a spin vortex cannot be at rest; a k = 0 vortex packet becomes a ring
    expanding at c/2.
3.  mu > 0: the chiral reversal splits the Kramers pair j <-> 12-j by
    +-4 mu (j = 7 up, j = 5 down) and opens a gap: a massive spin vortex.
    A band-projected packet stays put, a boosted one moves at v_g < c/2 < c.
4.  Zitterbewegung.  A pure j = 7 vortex with momentum k0 overlaps both bands
    of the split doublet.  Its centre trembles and its vortex content
    oscillates between j = 7 and j = 5 at the band splitting
    omega = E_up(k0) - E_down(k0).
"""

import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_hex_turning import (MOVES_IDX, MOVES_PHYS, DX_PHYS, DY_PHYS,
                                 DT_HALF, C_LIGHT, _shift, coin, step)
from paths import data_path

NL = 12
LAB = np.arange(NL)
HEAD = LAB % 6
S = np.roll(np.eye(NL), 1, axis=0)                     # (S v)_N = v_{N-1}
G0 = S + S.T
G3 = 1j * (np.linalg.matrix_power(S, 3) - np.linalg.matrix_power(S.T, 3))
R6 = np.linalg.matrix_power(S, 6)
P_SPIN = (np.eye(NL) - R6) / 2
MODES = np.exp(2j * np.pi * np.outer(LAB, LAB) / NL) / np.sqrt(NL)  # column j


def z12_coin(eps, mu):
    return expm(1j * (eps * G0 + mu * G3))


# ─── real-space evolution ────────────────────────────────────────────────────

class Grid:
    def __init__(self, half_x, half_y):
        self.nx = 2 * int(half_x / DX_PHYS) + 1
        self.ny = 2 * int(half_y / DY_PHYS) + 1
        self.cx, self.cy = self.nx // 2, self.ny // 2
        IX, IY = np.meshgrid(np.arange(self.nx) - self.cx,
                             np.arange(self.ny) - self.cy, indexing='ij')
        self.X, self.Y = IX * DX_PHYS, IY * DY_PHYS
        self.sub = ((IX + IY) % 2 == 0)


def z12_step(psi, C):
    """psi[x, y, N]: coin on the label, then hop along heading N mod 6."""
    w = psi @ C.T
    new = np.zeros_like(psi)
    for N in range(NL):
        d = N % 6
        new[:, :, N] = _shift(w[:, :, N], MOVES_IDX[d, 0], MOVES_IDX[d, 1])
    return new


def z12_step_local(psi, V, gvals, eps_map, mu):
    """Coin with position-dependent eps (mu fixed): C(r) = expm(i(eps(r) G0 + mu G3)).
    G0 and G3 commute (both functions of S) and share the eigenbasis MODES."""
    ph = np.exp(1j * (eps_map[:, :, None] * gvals[0][None, None, :]
                      + mu * gvals[1][None, None, :]))
    w = ((psi @ V.conj()) * ph) @ V.T
    new = np.empty_like(psi)
    for N in range(NL):                          # periodic: no boundary loss
        d = N % 6
        new[:, :, N] = np.roll(w[:, :, N], (MOVES_IDX[d, 0], MOVES_IDX[d, 1]),
                               axis=(0, 1))
    return new


def stats(grid, psi):
    p = (np.abs(psi) ** 2).sum(axis=-1)
    s = p.sum()
    mx, my = (grid.X * p).sum() / s, (grid.Y * p).sum() / s
    r2 = (((grid.X - mx) ** 2 + (grid.Y - my) ** 2) * p).sum() / s
    jpop = (np.abs(psi @ MODES.conj()) ** 2).sum(axis=(0, 1)) / s
    spin = (np.abs(psi @ P_SPIN.T) ** 2).sum() / s
    return mx, my, np.sqrt(r2), jpop, spin, s


# ─── k-space ─────────────────────────────────────────────────────────────────

def z12_T(kx, ky, C):
    ph = np.exp(-1j * (kx * MOVES_PHYS[HEAD, 0] + ky * MOVES_PHYS[HEAD, 1]))
    return ph[:, None] * C


def spin_bands(kx, ky, C):
    """Spin-sector bands: sorted energies, eigenvectors, group velocities."""
    lam, V = np.linalg.eig(z12_T(kx, ky, C))
    keep = np.linalg.norm(P_SPIN @ V, axis=0) > 0.5
    E = -np.angle(lam[keep]) / DT_HALF
    V = V[:, keep]
    o = np.argsort(E)
    E, V = E[o], V[:, o] / np.linalg.norm(V[:, o], axis=0)
    w = np.abs(V) ** 2
    vg = (w.T @ MOVES_PHYS[HEAD]) / DT_HALF
    return E, V, vg


def band_packet(grid, sigma, kx0, ky0, C, band, x0=0.0, y0=0.0, thresh=1e-7):
    """Packet in one spin-sector band (index into the sorted spin bands)."""
    g = (np.exp(-((grid.X - x0) ** 2 + (grid.Y - y0) ** 2) / (2 * sigma ** 2))
         * np.exp(1j * (kx0 * grid.X + ky0 * grid.Y)) * grid.sub)
    gh = np.fft.fft2(g)
    KX = (2 * np.pi * np.fft.fftfreq(grid.nx) / DX_PHYS)[:, None] * np.ones((1, grid.ny))
    KY = (2 * np.pi * np.fft.fftfreq(grid.ny) / DY_PHYS)[None, :] * np.ones((grid.nx, 1))
    mask = np.abs(gh) > thresh * np.abs(gh).max()
    kx, ky = KX[mask], KY[mask]
    ph = np.exp(-1j * (kx[:, None] * MOVES_PHYS[HEAD][None, :, 0]
                       + ky[:, None] * MOVES_PHYS[HEAD][None, :, 1]))
    lam, U = np.linalg.eig(ph[:, :, None] * C[None])
    spin_w = np.linalg.norm(np.einsum('ij,njk->nik', P_SPIN, U), axis=1)
    E = np.where(spin_w > 0.5, -np.angle(lam) / DT_HALF, np.inf)
    order = np.argsort(E, axis=1)                         # spin bands first
    sel = order[:, band]
    u = U[np.arange(len(kx)), :, sel]
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    _, Vref, _ = spin_bands(kx0, ky0, C)
    uref = Vref[:, band]
    ov = u @ uref.conj()
    u = u * (np.conj(ov) / (np.abs(ov) + 1e-30))[:, None]
    psih = np.zeros((grid.nx, grid.ny, NL), dtype=complex)
    psih[mask] = gh[mask][:, None] * u
    psi = np.fft.ifft2(psih, axes=(0, 1))
    return psi / np.sqrt((np.abs(psi) ** 2).sum())


def vortex_packet(grid, sigma, kx0, ky0, j, x0=0.0, y0=0.0):
    """Gaussian envelope times the pure vortex mode j on every site."""
    g = (np.exp(-((grid.X - x0) ** 2 + (grid.Y - y0) ** 2) / (2 * sigma ** 2))
         * np.exp(1j * (kx0 * grid.X + ky0 * grid.Y)) * grid.sub)
    psi = g[:, :, None] * MODES[:, j][None, None, :]
    return psi / np.sqrt((np.abs(psi) ** 2).sum())


def run(grid, psi, C, n_steps, every=2):
    rec = [stats(grid, psi)]
    ts = [0.0]
    for t in range(1, n_steps + 1):
        psi = z12_step(psi, C)
        if t % every == 0:
            rec.append(stats(grid, psi))
            ts.append(t * DT_HALF)
    ts = np.array(ts)
    return dict(t=ts, x=np.array([r[0] for r in rec]), y=np.array([r[1] for r in rec]),
                rms=np.array([r[2] for r in rec]), jpop=np.array([r[3] for r in rec]),
                spin=np.array([r[4] for r in rec]), norm=np.array([r[5] for r in rec]),
                psi=psi)


def fit_damped(t, y, omega0):
    """Fit y = a + b t + c t^2 + A exp(-g t) cos(w t + phi); returns (w, g, A)."""
    from scipy.optimize import curve_fit
    f = lambda t, a, b, c, A, g, w, ph: a + b*t + c*t*t + A*np.exp(-g*t)*np.cos(w*t + ph)
    p0 = [y.mean(), 0, 0, 0.5 * (y.max() - y.min()), 0.02, omega0, 0.0]
    best = None
    for ph0 in np.linspace(0, 2*np.pi, 8, endpoint=False):
        p0[-1] = ph0
        try:
            p, _ = curve_fit(f, t, y, p0=p0, maxfev=20000)
            res = np.sum((f(t, *p) - y) ** 2)
            if best is None or res < best[0]:
                best = (res, p)
        except RuntimeError:
            pass
    p = best[1]
    return abs(p[5]), p[4], abs(p[3])


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    out = {}
    EPS = 0.3

    # 1. superselection with a random position-dependent eps
    g = Grid(40, 30)
    rng = np.random.default_rng(7)
    psi = (rng.normal(size=(g.nx, g.ny, NL)) + 1j * rng.normal(size=(g.nx, g.ny, NL)))
    psi *= np.exp(-(g.X ** 2 + g.Y ** 2) / 200.0)[:, :, None]
    psi /= np.sqrt((np.abs(psi) ** 2).sum())
    eps_map = EPS * (1 + 0.5 * rng.uniform(-1, 1, size=(g.nx, g.ny)))
    gvals = (np.real(np.diag(MODES.conj().T @ G0 @ MODES)),
             np.real(np.diag(MODES.conj().T @ G3 @ MODES)))
    w0 = (np.abs(psi @ P_SPIN.T) ** 2).sum()
    drift = 0.0
    for t in range(60):
        psi = z12_step_local(psi, MODES, gvals, eps_map, 0.05)
        w = (np.abs(psi @ P_SPIN.T) ** 2).sum() / (np.abs(psi) ** 2).sum()
        drift = max(drift, abs(w - w0))
    print(f"1. spin-sector weight {w0:.6f}, max drift over 60 steps with random eps(r) "
          f"and mu=0.05: {drift:.1e}  [{time.time()-t0:.0f}s]")
    out['sector_drift'] = drift

    # 2a. mu = 0 equals the turning model at alpha = 1/2 (densities)
    g = Grid(40, 30)
    psi12 = vortex_packet(g, 5.0, 0.4, 0.2, 5)
    a6 = np.zeros((g.nx, g.ny, 6), dtype=complex)
    for N in range(NL):                       # spin readout  A_d = sum i^p psi_N
        a6[:, :, N % 6] += 1j ** (N % 4) * psi12[:, :, N]
    C12 = z12_coin(EPS, 0.0)
    C6 = coin(EPS, 1.5, 'unitary')            # alpha = 3/2 ≡ 1/2 (gauge)
    for t in range(60):
        psi12 = z12_step(psi12, C12)
        a6 = step(a6, C6)
    b6 = np.zeros_like(a6)
    for N in range(NL):
        b6[:, :, N % 6] += 1j ** (N % 4) * psi12[:, :, N]
    dev = np.abs(b6 - a6).max() / np.abs(a6).max()
    print(f"2a. Z12 walk (mu=0) read out with i^p vs 6-heading turning model alpha=3/2: "
          f"max rel. deviation {dev:.1e}")
    out['readout_dev'] = dev

    # band structure: splitting at k = 0 and cone / gap
    k = np.linspace(0, 1.2, 61)
    for mu in (0.0, 0.05):
        C = z12_coin(EPS, mu)
        Eb = np.array([spin_bands(kk, 0.0, C)[0] for kk in k])
        out[f'bands_mu{mu}'] = Eb
    out['k_bands'] = k
    mus = np.array([0.0, 0.01, 0.02, 0.05, 0.1])
    gaps = []
    for mu in mus:
        E0 = spin_bands(0.0, 0.0, z12_coin(EPS, mu))[0]
        gaps.append(E0[-1] - E0[-2])
    out['mus'], out['gaps'] = mus, np.array(gaps)
    print("   top-doublet splitting at k=0 vs mu:",
          ", ".join(f"mu={m}: {gp:.5f} (8mu={8*m:.5f})" for m, gp in zip(mus, gaps)))
    E0, V0, _ = spin_bands(0.0, 0.0, z12_coin(EPS, 0.05))
    jtop = np.argmax(np.abs(MODES.conj().T @ V0[:, -1]) ** 2)
    jsec = np.argmax(np.abs(MODES.conj().T @ V0[:, -2]) ** 2)
    print(f"   at mu=0.05 the upper level is the pure vortex j={jtop}, the lower j={jsec}")
    out['j_up'], out['j_down'] = jtop, jsec

    # 2b. massless: k = 0 vortex packet becomes a ring expanding at c/2
    g = Grid(110, 110)
    C = z12_coin(EPS, 0.0)
    r = run(g, vortex_packet(g, 6.0, 0.0, 0.0, 7), C, 200)
    slope = np.polyfit(r['t'][r['t'] > 40], r['rms'][r['t'] > 40], 1)[0]
    print(f"2b. mu=0, j=7 vortex at k=0: rms radius grows at {slope:.4f} "
          f"(c/2 = {C_LIGHT/2:.4f}); CoM stays at ({r['x'][-1]:.2e}, {r['y'][-1]:.2e}); "
          f"spin weight {r['spin'][-1]:.12f}  [{time.time()-t0:.0f}s]")
    out['ring_t'], out['ring_rms'], out['ring_slope'] = r['t'], r['rms'], slope
    out['ring_rho'] = (np.abs(r['psi']) ** 2).sum(axis=-1)

    # 3. massive vortex (mu > 0): rest packet and boosted packets
    MU = 0.05
    C = z12_coin(EPS, MU)
    r = run(g, band_packet(g, 6.0, 0.0, 0.0, C, band=5), C, 200)
    slope_m = np.polyfit(r['t'][r['t'] > 40], r['rms'][r['t'] > 40], 1)[0]
    print(f"3a. mu={MU}, upper band at k=0: rms radius grows at {slope_m:.4f} "
          f"(massless: {slope:.4f}); vortex content j={jtop}: {r['jpop'][-1, jtop]:.4f}")
    out['rest_t'], out['rest_rms'], out['rest_slope'] = r['t'], r['rms'], slope_m
    out['rest_rho'] = (np.abs(r['psi']) ** 2).sum(axis=-1)
    out['rest_jpop'] = r['jpop']

    ks = [0.1, 0.2, 0.4, 0.6]
    v_sim, v_band = [], []
    gb = Grid(150, 60)
    for kk in ks:
        r = run(gb, band_packet(gb, 8.0, kk, 0.0, C, band=5, x0=-60.0), C, 160)
        v = np.polyfit(r['t'], r['x'], 1)[0]
        vb = spin_bands(kk, 0.0, C)[2][-1, 0]
        v_sim.append(v); v_band.append(vb)
        print(f"3b. k={kk}: v_CoM = {v:.4f}, v_g(band) = {vb:.4f}, c/2 = {C_LIGHT/2:.4f}, "
              f"c = {C_LIGHT:.4f}  [{time.time()-t0:.0f}s]")
    out['boost_k'], out['v_sim'], out['v_band'] = np.array(ks), np.array(v_sim), np.array(v_band)

    # 4. Zitterbewegung of a pure j=7 vortex with momentum
    k0, SIG = 0.3, 16.0
    gz = Grid(170, 90)
    r = run(gz, vortex_packet(gz, SIG, k0, 0.0, jtop, x0=-60.0), C, 200, every=1)
    Eb, Vb, vgb = spin_bands(k0, 0.0, C)
    omega_band = Eb[-1] - Eb[-2]
    win = r['t'] <= 60.0
    om_j, gam_j, A_j = fit_damped(r['t'][win], r['jpop'][win, jtop], omega_band)
    om_y, gam_y, A_y = fit_damped(r['t'][win], r['y'][win], omega_band)
    print(f"4. ZB at k0={k0}, sigma={SIG}: band splitting omega = {omega_band:.4f}; "
          f"fit j={jtop} population: omega = {om_j:.4f}, amplitude {A_j:.3f}, "
          f"decay {gam_j:.4f}; transverse CoM y: omega = {om_y:.4f}, amplitude {A_y:.3f}, "
          f"decay {gam_y:.4f}")
    print(f"   overlap of the j={jtop} vortex with the two bands: "
          f"{np.abs(Vb[:, -1].conj() @ MODES[:, jtop])**2:.3f} / "
          f"{np.abs(Vb[:, -2].conj() @ MODES[:, jtop])**2:.3f}; "
          f"v_g of the two bands {vgb[-1,0]:.3f} / {vgb[-2,0]:.3f} -> separation time "
          f"sigma/dv = {SIG/abs(vgb[-1,0]-vgb[-2,0]):.1f}; spin weight {r['spin'][-1]:.12f}"
          f"  [{time.time()-t0:.0f}s]")
    out.update(zb_t=r['t'], zb_x=r['x'], zb_y=r['y'], zb_jpop=r['jpop'],
               zb_omega_band=omega_band, zb_omega_j=om_j, zb_omega_y=om_y,
               zb_gamma_j=gam_j, zb_spin=r['spin'])
    out['EPS'], out['MU'] = EPS, MU

    np.savez(data_path('phase_bit_vortex_dynamics.npz'), **out)
    print(f"saved data/phase_bit_vortex_dynamics.npz  [{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()
