#!/usr/bin/env python3
"""
Paper 3: the gravitational atom
==============================================================================

Quantum version of the perihelion advance: in a pure 1/r potential the levels
depend on n only (accidental degeneracy, closed Kepler orbits).  The metric's
relativistic terms split the l-degeneracy; semiclassically the splitting
E(n, l+1) - E(n, l) is the precession frequency of the orbit.

Effective Hamiltonian of the walker in a static field (from H2 band level:
gap N m, light speed c' N^(1+n)):
    H = N^(1/2) sqrt(m^2 + c'^2 Pi^2) N^(1/2),   Pi^2 = N^(n/2) (-div N^n grad) N^(n/2)
n = 1 (H2: A = 1/N) or 0 (wait only: A = 1); N = e^-U (product rule) or 1 - U (linear lapse).

A.  1D lattice atom (x-chain, Bloch k_y = k_z = 0), H2c and W schemes: levels from the time signal
    <psi0|psi(t)> of even/odd start packets, against the 1D effective Hamiltonian.
B.  3D atom from the effective Hamiltonian (radial equation per l, dense diagonalisation):
    l-splitting at fixed n for the three metric variants vs the perihelion factors
    1 : 1/3 : 7/6, and vs the classical precession frequency for larger n.

Run parts: python phase_bit_s4_open14.py [A] [B]
"""

import functools
import os
import sys
import time

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import data_path

print = functools.partial(print, flush=True)

VARIANTS = {'H2': (lambda U: np.exp(-U), 1, (1, 1)),          # N, n (A = N^-n), (gamma, beta)
            'W': (lambda U: np.exp(-U), 0, (0, 1)),
            'linear': (lambda U: 1 - U, 1, (1, 0.5))}


def eff_levels_radial(Nr, n_exp, l, r, m, c, nlev):
    """levels of H = N^1/2 sqrt(m^2 + c^2 Pi^2) N^1/2 for angular momentum l on the radial grid r (u = r R)."""
    dr = r[1] - r[0]
    M = len(r)
    w = Nr ** n_exp                                            # N^n at the nodes
    wh = (np.r_[w[1:], w[-1]] + w) / 2                         # at half points i + 1/2
    whm = np.r_[w[0], wh[:-1]]
    # -d/dr (w d/dr) + w l(l+1)/r^2 acting on u (Dirichlet at r = 0 and r_max)
    D = (np.diag(wh + whm) - np.diag(wh[:-1], 1) - np.diag(wh[:-1], -1)) / dr ** 2 + np.diag(w * l * (l + 1) / r ** 2)
    s = Nr ** (n_exp / 2)
    P2 = s[:, None] * D * s[None, :]
    lam, V = eigh(P2)
    E = (V * np.sqrt(m ** 2 + c ** 2 * np.clip(lam, 0, None))) @ V.T
    h = np.sqrt(Nr)
    H = h[:, None] * E * h[None, :]
    ev = eigh(H, eigvals_only=True, subset_by_index=[0, nlev - 1])
    return ev


def newton_levels_radial(Ur, l, r, m, c, nlev):
    """nonrelativistic reference: H = m - m U + c^2 p^2/(2m) (same softened U)."""
    dr = r[1] - r[0]
    D = (np.diag(np.full(len(r), 2.0)) - np.diag(np.ones(len(r) - 1), 1) - np.diag(np.ones(len(r) - 1), -1)) / dr ** 2
    H = c ** 2 / (2 * m) * (D + np.diag(l * (l + 1) / r ** 2)) + np.diag(m - m * Ur)
    return eigh(H, eigvals_only=True, subset_by_index=[0, nlev - 1])


def eff_levels_1d(Nx_, nexp, x, m, c, Efun=None, nlev=12):
    """1D: H = N^1/2 E(Pi^2) N^1/2, Pi^2 = N^(n/2) (-d N^n d) N^(n/2) on the grid x (Dirichlet)."""
    dx = x[1] - x[0]
    w = Nx_ ** nexp
    wh = (np.r_[w[1:], w[-1]] + w) / 2
    whm = np.r_[w[0], wh[:-1]]
    D = (np.diag(wh + whm) - np.diag(wh[:-1], 1) - np.diag(wh[:-1], -1)) / dx ** 2
    s = Nx_ ** (nexp / 2)
    lam, V = eigh(s[:, None] * D * s[None, :])
    lam = np.clip(lam, 0, None)
    f = np.sqrt(m ** 2 + c ** 2 * lam) if Efun is None else Efun(lam)
    E = (V * f) @ V.T
    h = np.sqrt(Nx_)
    ev, vec = eigh(h[:, None] * E * h[None, :], subset_by_index=[0, nlev - 1])
    par = np.sign(np.sum(vec * vec[::-1], axis=0))                 # +1 even, -1 odd (grid symmetric)
    return ev, par


def spectrum_peaks(sig, dt, emin, emax, pad=16):
    """levels E from the time signal sum_j c_j exp(-i E_j t): Blackman-Harris window (side lobes -92 dB),
    zero padding, parabolic refinement of the log magnitude."""
    from scipy.signal.windows import blackmanharris
    n = len(sig)
    w = blackmanharris(n)
    F = np.fft.fft(sig * w, n * pad)
    om = -2 * np.pi * np.fft.fftfreq(n * pad, d=dt)              # E such that exp(-i E t)
    A = np.abs(F)
    sel = (om > emin) & (om < emax)
    idx = np.where(sel)[0]
    peaks = []
    for i in idx:
        if 0 < i < len(A) - 1 and A[i] > A[i - 1] and A[i] >= A[i + 1] and A[i] > 1e-3 * A[idx].max():
            y0, y1, y2 = np.log(A[i - 1]), np.log(A[i]), np.log(A[i + 1])
            d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            peaks.append((om[i] + d * (om[i + 1] - om[i]), A[i] / A[idx].max()))
    return sorted(peaks)


def part_A(out):
    print("A. 1D lattice atom (x-chain, Bloch k_y = k_z = 0): levels from the time signal vs the effective Hamiltonian")
    import scipy.sparse as sp
    from scipy.linalg import expm
    from quantum_fcc_3d import DT_HALF, LAT_A
    from phase_bit_s4_open import CP
    from phase_bit_mass_s4 import bands_C
    from phase_bit_s4_open8 import Scheme
    from phase_bit_s4_open9 import Xl, LAM0, shift_chain, lam_of
    from phase_bit_s4_open10 import A_FIT
    t0 = time.time()
    mu, GM, Nx, nst = 0.08, 0.6, 512, 60000
    m = 2 * mu
    a0 = CP ** 2 / (m ** 2 * GM)
    rc = 0.5 * a0
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    U = GM / np.sqrt(xs ** 2 + rc ** 2)
    print(f"   m = {m}, GM = {GM}: a0 = {a0:.1f}, rc = a0/2, U(0) = {U.max():.4f}, chain {Nx} layers ({Nx*LAT_A:.0f}), {nst} ticks")
    # free-lattice band along x (upper doublet) as a function of k^2, for the effective theory with the lattice dispersion
    C0 = expm(1j * Xl(LAM0)(mu))
    kt = np.linspace(0, 1.2, 241)
    Eb = np.array([np.sort(bands_C(np.array([k, 0, 0]), C0)[0][np.argsort(np.abs(bands_C(np.array([k, 0, 0]), C0)[0]
                                                                               - np.hypot(m, CP * k)))[:2]]).mean() for k in kt])
    xg = np.linspace(-150, 150, 1500)
    Ug = GM / np.sqrt(xg ** 2 + rc ** 2)
    res = {}
    for kind, nexp in (('W', 0), ('H2', 1)):
        Nf = np.exp(-U)
        lams = np.full(Nx, LAM0) if kind == 'W' else lam_of(np.clip(Nf, 0.4, 0.999), A_FIT)
        Cs = [expm(1j * Xl(l)(mu)) for l in lams]
        Cb = sp.block_diag([sp.csr_matrix(C_) for C_ in Cs], format='csr')
        sc = Scheme('W' if kind == 'W' else 'WH', shift_chain((0.0, 0.0), Nx) @ Cb, Cb)
        sc.set_N(Nf, Nf if kind == 'H2' else np.full(Nx, 1 - 1e-13))
        nr = sc.nreg()
        # start: displaced Gaussian on the rest doublet (excites even and odd levels)
        E0, V0 = bands_C(np.zeros(3), C0)
        u = V0[:, np.argsort(np.abs(E0 - m))[0]]
        lev_par = {}
        for parity, env in (('even', np.exp(-xs ** 2 / (2 * (0.6 * rc) ** 2))),
                            ('odd', xs / rc * np.exp(-xs ** 2 / (2 * (0.8 * rc) ** 2)))):
            psi = (env[:, None] * u[None, :]).ravel().astype(complex)
            st = [psi / np.linalg.norm(psi)] + [np.zeros_like(psi) for _ in range(nr - 1)]
            ph = np.exp(1j * m * DT_HALF)
            for _ in range(4):
                st2 = sc.tick(st)
                st = [(a + ph * b) / 2 for a, b in zip(st, st2)]
            nrm = np.sqrt(sum(np.vdot(a, a).real for a in st))
            st = [a / nrm for a in st]
            s0 = [a.copy() for a in st]
            sig = np.empty(nst, complex)
            for it in range(nst):
                sig[it] = sum(np.vdot(a0_, a) for a0_, a in zip(s0, st))
                st = sc.tick(st)
            pk = spectrum_peaks(sig, DT_HALF, m * np.exp(-U.max()), m)
            lev_par[parity] = [p[0] - m for p in pk if p[1] > 0.01]
        lev_l = np.array(sorted(lev_par['even'] + lev_par['odd']))
        print(f"   {kind}: lattice bound levels, even: " + " ".join(f"{e:+.5e}" for e in lev_par['even'])
              + " | odd: " + " ".join(f"{e:+.5e}" for e in lev_par['odd']))
        # effective theory on a fine grid: Dirac and lattice dispersion
        Ngr = np.exp(-Ug)
        ev_d, par_d = eff_levels_1d(Ngr, nexp, xg, m, CP)
        ev_l, par_l = eff_levels_1d(Ngr, nexp, xg, m, CP, Efun=lambda lam: np.interp(np.sqrt(lam), kt, Eb))
        ev_d, ev_l = ev_d - m, ev_l - m
        bd = ev_l < 0
        print(f"   {kind}: effective (lattice band), even: " + " ".join(f"{e:+.5e}" for e, q in zip(ev_l, par_l) if q > 0 and e < 0)
              + " | odd: " + " ".join(f"{e:+.5e}" for e, q in zip(ev_l, par_l) if q < 0 and e < 0))
        res[kind] = (lev_l, ev_d, ev_l)
        out[f'A_{kind}'] = np.array([np.r_[lev_l, [np.nan] * (12 - len(lev_l))][:12], ev_d[:12], ev_l[:12]])
        print(f"   {kind}: effective (Dirac) bound: " + " ".join(f"{e:+.5e}" for e in ev_d[ev_d < 0]) + f"  [{time.time()-t0:.0f}s]")
    out['A_par'] = np.array([mu, GM, rc, Nx, nst])


def part_B(out):
    print("B. 3D gravitational atom from the effective Hamiltonian (scalar; m = c = 1, U = GM/sqrt(r^2 + rc^2), rc = 0.1 a0)")
    t0 = time.time()
    m, c = 1.0, 1.0
    rows = []
    for GM in (0.1, 0.05):
        a0 = c ** 2 / (m ** 2 * GM)
        rc = 0.1 * a0
        rmax, M = 70 * a0, 3500
        r = np.linspace(rmax / M, rmax, M)
        Ur = GM / np.sqrt(r ** 2 + rc ** 2)
        res = {}
        for name, (Nf, nexp, (gam, bet)) in VARIANTS.items():
            Nr = Nf(Ur)
            res[name] = {l: eff_levels_radial(Nr, nexp, l, r, m, c, 5 - l) - m for l in range(4)}
        res['Newton'] = {l: newton_levels_radial(Ur, l, r, m, c, 5 - l) - m for l in range(4)}
        for name in ('H2', 'W', 'linear', 'Newton'):
            lev = res[name]
            print(f"   GM = {GM}, {name:6s}: " + " | ".join(
                f"n={n}: " + " ".join(f"{lev[l][n-l-1]:+.7e}" for l in range(min(n, 4))) for n in (2, 3, 4))
                + f"   (Bohr -GM^2/2n^2: {-GM**2/8:.4e}, {-GM**2/18:.4e}, {-GM**2/32:.4e})  [{time.time()-t0:.0f}s]")
        for n in (2, 3, 4):
            for l in range(0, n - 1):
                if l + 1 > 3:
                    continue
                d = {k: res[k][l + 1][n - l - 2] - res[k][l][n - l - 1] for k in res}
                rel = {k: d[k] - d['Newton'] for k in ('H2', 'W', 'linear')}
                rows.append((GM, n, l, d['Newton'], rel['H2'], rel['W'], rel['linear']))
                print(f"      GM = {GM}, n = {n}, E(l={l+1}) - E(l={l}): Newton (softening) {d['Newton']:+.3e}; relativistic part "
                      f"H2 {rel['H2']:+.4e}, W {rel['W']:+.4e} (W/H2 {rel['W']/rel['H2']:.3f}; classical 1/3), "
                      f"linear {rel['linear']:+.4e} (lin/H2 {rel['linear']/rel['H2']:.3f}; classical 7/6 = 1.167)")
        out[f'B_levels_{GM}'] = np.array([[res[k][l][i] if i < len(res[k][l]) else np.nan for i in range(5)]
                                          for k in ('H2', 'W', 'linear', 'Newton') for l in range(4)])
    out['B_rows'] = np.array(rows)
    print("   correspondence with the classical precession frequency omega_p = 3 GM^4/(L^2 n^3) (H2 = GR; L = mean l + 1/2):")
    for row in rows:
        GM, n, l, dN, h2 = row[:5]
        L = l + 1.0
        wp = 3 * GM ** 4 / (L ** 2 * n ** 3)
        print(f"      GM = {GM}, n = {int(n)}, l {int(l)} -> {int(l)+1}: H2 splitting {h2:.4e}, omega_p {wp:.4e}, ratio {h2/wp:.3f}")


def main():
    parts = [p.upper() for p in sys.argv[1:]] or ['A', 'B']
    path = data_path('phase_bit_s4_open14.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in parts:
        {'A': part_A, 'B': part_B}[p](out)
        np.savez_compressed(path, **out)
    print("saved data/phase_bit_s4_open14.npz")


if __name__ == '__main__':
    main()
