#!/usr/bin/env python3
"""
Paper 3 follow-up, gravitational waves 2: the detector
==============================================================================

Does a lattice photon respond to all ten metric components the way a massless particle
in GR does?  Compensated H2 walker (2O*, eps 0.3, mu 0, wait gate N_w, stay gate per
heading N_h,d, lambda(N0)), background N_w = N_h = N0 = 0.75 (c_eff = N0^2 c'), plus the
uniform drag hop exp(-i dt b.p_lat), p_lat(k) = sum_s s sin(k.s) / (4|s|^2) (Bloch symbol
of drag3d).

GR: g^{mu nu} k_mu k_nu = 0 gives  d ln w = -1/2 h_{mu nu} n^mu n^nu,  n^mu = (1, n).
Lattice map, calibrated at k -> 0:
    lapse     N_w = N0 (1 + phi)                         h_00 = -2 phi
    drag      b = -c_eff h_0i                            (exact scalar phase)
    bond      N_h,d = N0 (1 + delta_axis(d))             h_ij from the cone tensor (open17)

A.  static uniform responses at finite k, all directions: ratio lattice / GR for each
    component (lapse, drag, bond trace, bond E_g, bond T_2g) vs |k|.
B.  combinations a gravitational wave along n presents to a photon:
      TT (h_+ on the two transverse axes): photon along a transverse axis sees -h_+/2,
          photon along n sees nothing;
      gauge xi_0(t - n.x) (h_00 = 2G', h_0n = -G') and xi_n (h_0n = F', h_nn = -2F'):
          a co-moving photon must see exactly nothing;
    residual of the gauge combinations relative to the TT signal vs |k|.
    (For a compact pulse any linear response to a gauge wave is a total derivative along
    a non-co-moving path and integrates to zero; only the co-moving photon tests the
    relative normalisation of lapse, drag and bond gate.)
C.  real time: uniform h_+(t) = h0 sin(W t) for half a period, photons along x, y, z;
    phase relative to free evolution vs GR  dphi = (E/2) int h_nn dt.
D.  origin of the O(k) term of the bond response: lambda is fixed at lambda(N0), so a
    bond strain changes the lattice background -alpha k^2.  Bond-local retuning: W90
    coupling of the heading pair (d, d') at lambda(sqrt(N_d N_d')), counterterm per
    heading at lambda(N_d); bond map recalibrated; A and B repeated.

Run parts: python phase_bit_s4_open18.py [A] [B] [C] [D]
"""

import functools
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import MOVES_PHYS, DT_HALF
from phase_bit_s4_open import CP
from phase_bit_s4_open4 import a_of
from phase_bit_s4_open5 import DIRS
from phase_bit_s4_open8 import Sk, Scheme
from phase_bit_s4_open9 import Xl, lam_of, W90, G2, lvf
from phase_bit_s4_open4 import EPS
from phase_bit_s4_open17 import AX, AXU, SYM, UD
from paths import data_path

print = functools.partial(print, flush=True)
N0 = 0.75
CEFF = N0 ** 2 * CP
A_FIT = np.load(data_path('phase_bit_s4_open9.npz'))['A_fit']
LAM = lam_of(N0, A_FIT)
MODE = {'lam': 'fixed'}
HD = np.repeat(np.arange(12), 2)
C0 = expm(1j * Xl(LAM)(0.0))
S2 = np.sum(MOVES_PHYS[0] ** 2)
AX_OF = np.zeros(12, int)
for _a, (_d, _r) in enumerate(AX):
    AX_OF[_d] = AX_OF[_r] = _a


def p_lat(k):
    return (MOVES_PHYS * np.sin(MOVES_PHYS @ k)[:, None]).sum(0) / (4 * S2)


def U_full(k, phi=0.0, dax=np.zeros(6), b=np.zeros(3)):
    k = np.asarray(k, float)
    S = Sk(k)
    C = C0
    if MODE['lam'] == 'pair' and np.any(dax):
        Nd = N0 * (1 + np.asarray(dax)[AX_OF])
        Lp = lam_of(np.sqrt(np.outer(Nd, Nd)), A_FIT)[HD][:, HD]
        C = expm(1j * (EPS * (G2 + Lp * W90) - EPS * np.diag(lvf(lam_of(Nd, A_FIT)[HD]))))
    sc = Scheme('WH', S @ C, C, S)
    aw = a_of(N0 * (1 + phi))
    ah = np.repeat(a_of(N0 * (1 + dax[AX_OF])), 2)
    sc.cw, sc.sw = np.full(24, np.cos(aw)), np.full(24, np.sin(aw))
    sc.ch, sc.sh = np.cos(ah), np.sin(ah)
    cols = []
    for j in range(96):
        e = np.zeros(96, complex)
        e[j] = 1
        cols.append(np.concatenate(sc.tick(list(e.reshape(4, 24)))))
    return np.exp(-1j * DT_HALF * (b @ p_lat(k))) * np.array(cols).T


def photon(U, k):
    """upper massless doublet: mean of the two levels nearest +c_eff |k|, and one eigenvector."""
    w, V = np.linalg.eig(U)
    E = -np.angle(w) / DT_HALF
    idx = np.argsort(np.abs(E - CEFF * np.linalg.norm(k)))[:2]
    return E[idx].mean(), V[:, idx[0]], abs(E[idx[0]] - E[idx[1]])


def dlnE(k, eps=1e-5, **pert):
    """symmetric finite difference of ln E for a perturbation scaled by eps."""
    sc = lambda s: {kk: s * np.asarray(v) for kk, v in pert.items()}
    return (np.log(photon(U_full(k, **sc(eps)), k)[0]) - np.log(photon(U_full(k, **sc(-eps)), k)[0])) / (2 * eps)


def bond_map():
    """6x6 map delta_axis -> h_ij at k -> 0: d ln E(n) = -1/2 n.h.n fitted over DIRS."""
    kk = 0.004
    A = np.array([[n[i] * n[j] * (1 if i == j else 2) for i, j in SYM] for n in DIRS])
    cols = []
    for a in range(6):
        d = np.zeros(6)
        d[a] = 1.0
        y = np.array([-2 * dlnE(kk * n, dax=d) for n in DIRS])
        cols.append(np.linalg.lstsq(A, y, rcond=None)[0])
    return np.array(cols).T                    # h_sym = Mb @ delta


def hsym(h):
    return np.array([h[i, j] for i, j in SYM])


def lattice_dlnE(k, h00=0.0, h0=np.zeros(3), h=np.zeros((3, 3)), Minv=None):
    """lattice d ln E for a metric perturbation (linear superposition of the calibrated maps)."""
    r = 0.0
    if h00:
        r += dlnE(k, phi=-h00 / 2)
    if np.any(h0):
        r += dlnE(k, b=-CEFF * np.asarray(h0))
    if np.any(h):
        r += dlnE(k, dax=Minv @ hsym(h))
    return r


def gr_dlnE(n, h00=0.0, h0=np.zeros(3), h=np.zeros((3, 3))):
    return -0.5 * (h00 + 2 * np.asarray(h0) @ n + n @ h @ n)


KS = (0.01, 0.03, 0.1, 0.2, 0.3)


def part_A(out):
    print(f"A. static uniform responses of the photon (H2c, N0 = {N0}, lambda = {LAM:.4f}): lattice / GR")
    t0 = time.time()
    Mb = bond_map()
    Minv = np.linalg.inv(Mb)
    out['Mb'] = Mb
    sv = np.linalg.svd(Mb, compute_uv=False)
    print(f"   bond map delta -> h (k -> 0): singular values " + ", ".join(f"{v:.3f}" for v in sv))
    comps = {
        'lapse h00': dict(h00=1.0),
        'drag h0x': dict(h0=np.array([1.0, 0, 0])),
        'drag h0(111)': dict(h0=np.ones(3) / np.sqrt(3)),
        'bond trace': dict(h=np.eye(3)),
        'bond E_g': dict(h=np.diag([1.0, -1, 0])),
        'bond T_2g': dict(h=np.array([[0, 1.0, 0], [1, 0, 0], [0, 0, 0]])),
    }
    rows = []
    for name, c in comps.items():
        line = []
        for km in KS:
            rat = []
            for n in DIRS:
                g = gr_dlnE(n, **c)
                if abs(g) < 0.05:
                    continue
                rat.append(lattice_dlnE(km * n, Minv=Minv, **c) / g)
            rat = np.array(rat)
            line.append((rat.mean(), rat.min(), rat.max()))
            rows.append((list(comps).index(name), km, rat.mean(), rat.min(), rat.max()))
        print(f"   {name:13s} " + "  ".join(f"k={km}: {m:.4f} [{lo:.4f},{hi:.4f}]" for km, (m, lo, hi) in zip(KS, line)))
    out['A'] = np.array(rows)
    out['Minv'] = Minv
    print(f"   [{time.time()-t0:.0f}s]")
    return Minv


def part_B(out, Minv):
    print("B. gravitational wave along n: TT signal vs gauge residual for the co-moving photon")
    t0 = time.time()
    rows = []
    for wn, tr in ((np.array([1.0, 0, 0]), (np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))),
                   (np.array([1.0, 1, 0]) / np.sqrt(2), (np.array([1.0, -1, 0]) / np.sqrt(2), np.array([0, 0, 1.0]))),
                   (np.ones(3) / np.sqrt(3), (np.array([1.0, -1, 0]) / np.sqrt(2), np.array([1.0, 1, -2]) / np.sqrt(6)))):
        e1, e2 = tr
        hp = np.outer(e1, e1) - np.outer(e2, e2)
        for km in KS:
            tt_trans = lattice_dlnE(km * e1, h=hp, Minv=Minv)            # GR -1/2
            tt_long = lattice_dlnE(km * wn, h=hp, Minv=Minv)             # GR 0
            g0 = lattice_dlnE(km * wn, h00=2.0, h0=-wn, Minv=Minv)        # xi_0, GR 0
            gn = lattice_dlnE(km * wn, h0=wn, h=-2 * np.outer(wn, wn), Minv=Minv)   # xi_n, GR 0
            ref = abs(tt_trans)
            rows.append((*wn, km, tt_trans, tt_long, g0, gn))
            print(f"   n = {np.round(wn,3)}, |k| = {km}: TT transverse {tt_trans:+.5f} (GR -0.5), along n "
                  f"{tt_long/ref:+.1e}; gauge xi_0 {g0/ref:+.1e}, xi_n {gn/ref:+.1e} (relative to TT)")
    out['B'] = np.array(rows)
    print(f"   [{time.time()-t0:.0f}s]")


def part_C(out, Minv):
    print("C. real time: uniform h_+(t) = h0 sin(W t) (wave along x: h_yy = -h_zz), half a period")
    t0 = time.time()
    h0, W = 1e-3, 0.02
    nt = int(round(np.pi / W / DT_HALF))
    Tend = nt * DT_HALF
    hp = np.diag([0.0, 1.0, -1.0])
    dax1 = Minv @ hsym(hp)
    rows = []
    for km in (0.03, 0.1, 0.3):
        res = []
        for n in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
            k = km * n
            U0 = U_full(k)
            E, v, _ = photon(U0, k)
            psi, ref = v.copy(), v.copy()
            integ = 0.0
            for it in range(nt):
                t = (it + 0.5) * DT_HALF
                ht = h0 * np.sin(W * t)
                psi = U_full(k, dax=ht * dax1) @ psi
                ref = U0 @ ref
                integ += ht * DT_HALF
            dphi = np.angle(np.vdot(ref, psi))
            gr = E * 0.5 * (n @ hp @ n) * integ            # phase = -int dE dt, dE = -E/2 h_nn
            leak = 1 - abs(np.vdot(ref, psi))
            res.append((dphi, gr, leak))
        (px, gx, lx), (py, gy, ly), (pz, gz, lz) = res
        rows.append((km, px, py, pz, gy, gz, lx, ly, lz))
        print(f"   |k| = {km}: phase x {px:+.3e}, y {py:+.3e} (GR {gy:+.3e}, ratio {py/gy:.4f}), "
              f"z {pz:+.3e} (GR {gz:+.3e}, ratio {pz/gz:.4f}); (y - z)/GR {(py-pz)/(gy-gz):.4f}; "
              f"x / y {px/py:+.1e}; leakage {max(lx,ly,lz):.1e}  [{time.time()-t0:.0f}s]")
    out['C'] = np.array(rows)


def part_D(out):
    print("D. bond-local lambda retuning (W90 pair (d, d') at lambda(sqrt(N_d N_d'))): A and B repeated")
    MODE['lam'] = 'pair'
    sub = {}
    Minv = part_A(sub)
    part_B(sub, Minv)
    for k, v in sub.items():
        out['D_' + k] = v
    MODE['lam'] = 'fixed'


def main():
    parts = sys.argv[1:] or ['A', 'B', 'C', 'D']
    out = {}
    Minv = part_A(out)
    if 'B' in parts:
        part_B(out, Minv)
    if 'C' in parts:
        part_C(out, Minv)
    if 'D' in parts:
        part_D(out)
    np.savez(data_path('phase_bit_s4_open18.npz'), **out)


if __name__ == '__main__':
    main()
