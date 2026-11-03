#!/usr/bin/env python3
"""
Paper 3, hypothesis H1 (active gravitational mass of antimatter) and the proper
time as the mass part of the computational load
==============================================================================

A.  H1.  Loads of upper (particle) and lower (antiparticle) band states:
    e = Re psi^+ i (U psi - psi)/dt, e_M = mu dE/dmu (local), s = 2e - e_M.
    A1  band level: signs and sizes for both bands.
    A2  particle + antiparticle on the x-chain (1D Green function, Hartree):
        (i) single-particle picture: the antiparticle sources with its own load
            (negative) -> both run in the same direction (runaway, momentum not
            conserved);
        (ii) hole picture (filled Dirac sea, load relative to the vacuum): the
             antiparticle is a missing negative-energy state, its source is
             minus the load of that state -> mutual attraction.
B.  Proper time.  The mass load is m dtau/dt:
    B1  free packets moving transversely (k_y), uniform lapse N:
        Q_M / (N0 m) vs N sqrt(1 - v^2) (Dirac) and the lattice band.
    B2  with the spatial metric of Exp. 1c (buffer lapse N/A, mass mu A):
        Q_M = m N sqrt(1 - v_phys^2), v_phys from the physical momentum k/A.
    The space-time integral of the mass load is m tau = -(action).

Run parts: python phase_bit_s4_open7.py [A] [B]   (default: both)
"""

import functools
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import DT_HALF, LAT_A
from phase_bit_mass_s4 import bands_C
from phase_bit_s4_open import CP
from phase_bit_s4_open4 import f_dwell, a_of, model_2Ostar, N0
from phase_bit_s4_open5 import chain_T, Buffered, packet, on_branch
from paths import data_path

print = functools.partial(print, flush=True)
UPPER = lambda E: np.sort(np.argsort(np.abs(E))[:4])[2:]
LOWER = lambda E: np.sort(np.argsort(np.abs(E))[:4])[:2]
DMU = 1e-4


class Walker:
    """One packet on the buffered x-chain with local loads e(x), e_M(x)."""

    def __init__(self, X, mu, kyz, sel, Nx, xs, x0, sig, Nbase=N0, A=1.0):
        self.mu_eff = mu * A
        C = expm(1j * X(self.mu_eff))
        self.Nx = Nx
        psi, self.Eb = packet(C, kyz, sel, Nx, xs, x0, sig)
        self.dT = chain_T((expm(1j * X(self.mu_eff + DMU)) - expm(1j * X(self.mu_eff - DMU))) / (2 * DMU), kyz, Nx)
        self.bt = Buffered(chain_T(C, kyz, Nx), Nx)
        self.Nbase = Nbase / A                      # buffer lapse N/A (Exp. 1c)
        aN = a_of(self.Nbase)
        self.bt.set_a(np.full(Nx, aN))
        self.m, self.b = on_branch(self.bt, psi, self.Eb * DT_HALF, np.full(Nx, aN), aN)
        self.xs = xs
        self.e, self.eM = self.loads()[2:]

    def loads(self):
        bt = self.bt
        m2, b2 = bt.tick(self.m, self.b)
        ld = (np.conj(self.m) * 1j * (m2 - self.m) + np.conj(self.b) * 1j * (b2 - self.b)).real / DT_HALF
        dm = self.dT @ self.m
        lm = (1j * (np.conj(m2) * bt.ca * dm + np.conj(b2) * 1j * bt.sa * dm)).real * self.mu_eff / DT_HALF
        return m2, b2, ld.reshape(self.Nx, 24).sum(1), lm.reshape(self.Nx, 24).sum(1)

    def step(self, Phi=None):
        if Phi is not None:
            self.bt.set_a(a_of(np.clip(self.Nbase * (1 + Phi), 1e-3, 1 - 1e-9)))
        self.m, self.b, self.e, self.eM = self.loads()

    def com(self):
        w = (np.abs(self.m) ** 2 + np.abs(self.b) ** 2).reshape(self.Nx, 24).sum(1)
        return (w @ self.xs) / w.sum()


# ═════════════════════════════════════════════════════════════════════════════
def part_A(out):
    print("A. H1: active gravitational mass of the antiparticle (2O*, eps = 0.3, buffer lapse N0 = 0.75)")
    t0 = time.time()
    X, lam = model_2Ostar()
    Nx, SIG, nst, D = 1000, 60.0, 200, 400
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    ts = np.arange(nst + 1) * DT_HALF

    # A1: loads of single packets
    rows = []
    for name, sel in (('particle (upper band)', UPPER), ('antiparticle (lower band)', LOWER)):
        for ky in (0.0, 0.3):
            w = Walker(X, 0.03, (ky, 0.0), sel, Nx, xs, 0.0, SIG)
            Q, QM = w.e.sum(), w.eM.sum()
            rows.append((sel is LOWER, ky, w.Eb, Q, QM, 2 * Q - QM))
            print(f"   A1  {name:26s} k_y = {ky}: E = {w.Eb:+.4f}; Q = {Q:+.5f} (N0 E = {N0*w.Eb:+.5f}), "
                  f"Q_M = {QM:+.5f}, source 2Q - Q_M = {2*Q-QM:+.5f}")
    out['A1'] = np.array(rows)

    # A2: particle at -D/2, antiparticle at +D/2
    kap = 2e-3 / (N0 * 0.06)
    absdx = np.abs(xs[:, None] - xs[None, :])
    res = {}
    for picture, sgn in (('single-particle', +1.0), ('hole (Dirac sea)', -1.0)):
        ws = [Walker(X, 0.03, (0.0, 0.0), UPPER, Nx, xs, -D / 2 * LAT_A, SIG),
              Walker(X, 0.03, (0.0, 0.0), LOWER, Nx, xs, +D / 2 * LAT_A, SIG)]
        signs = [1.0, sgn]                           # sign of the source of each packet
        xc = [[], []]
        for it in range(nst + 1):
            srcs = [sg * (2 * w.e - w.eM) for sg, w in zip(signs, ws)]
            phis = [0.5 * kap * absdx @ q for q in srcs]
            for i, w in enumerate(ws):
                xc[i].append(w.com())
                if it == nst:
                    continue
                Phi = phis[1 - i]
                w.step(Phi - np.interp(xc[i][-1], xs, Phi))
        xc = np.array(xc)
        acc = [2 * np.polyfit(ts, xc[i], 2)[0] for i in range(2)]
        res[picture] = (acc, xc)
        com_acc = 0.5 * (acc[0] + acc[1])
        print(f"   A2  {picture:17s}: a_particle = {acc[0]:+.3e}, a_antiparticle = {acc[1]:+.3e}; "
              f"centre of mass {com_acc:+.3e}, relative (towards each other > 0) {acc[0]-acc[1]:+.3e}  "
              f"[{time.time()-t0:.0f}s]")
        out[f'A2_{picture[:4]}'] = xc
        out[f'A2_{picture[:4]}_acc'] = np.array(acc)
    # reference: particle-particle pair with the same masses
    ws = [Walker(X, 0.03, (0.0, 0.0), UPPER, Nx, xs, -D / 2 * LAT_A, SIG),
          Walker(X, 0.03, (0.0, 0.0), UPPER, Nx, xs, +D / 2 * LAT_A, SIG)]
    xc = [[], []]
    for it in range(nst + 1):
        phis = [0.5 * kap * absdx @ (2 * w.e - w.eM) for w in ws]
        for i, w in enumerate(ws):
            xc[i].append(w.com())
            if it == nst:
                continue
            Phi = phis[1 - i]
            w.step(Phi - np.interp(xc[i][-1], xs, Phi))
    xc = np.array(xc)
    acc = [2 * np.polyfit(ts, xc[i], 2)[0] for i in range(2)]
    print(f"   A2  particle-particle  : a_1 = {acc[0]:+.3e}, a_2 = {acc[1]:+.3e} (reference)")
    out['A2_pp'] = xc
    out['A2_pp_acc'] = np.array(acc)
    out['A2_ts'] = ts


# ═════════════════════════════════════════════════════════════════════════════
def part_B(out):
    print("B. proper time = mass load / m: Q_M = m dtau/dt (packets moving transversely, uniform N, A)")
    t0 = time.time()
    X, lam = model_2Ostar()
    Nx, SIG, nst = 400, 30.0, 60
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    rows = []
    for mu in (0.03, 0.15):
        m = 2 * mu
        for Nb, A in ((0.75, 1.0), (0.4, 1.0), (0.999, 1.0), (0.75, 0.9), (0.75, 1.1)):
            for ky in (0.0, 0.15, 0.3, 0.6):
                w = Walker(X, mu, (ky, 0.0), UPPER, Nx, xs, 0.0, SIG, Nbase=Nb, A=A)
                QM, QQ = [], []
                for _ in range(nst):
                    QM.append(w.eM.sum())
                    QQ.append(w.e.sum())
                    w.step()
                QM, QQ = np.mean(QM), np.mean(QQ)
                # Dirac with physical momentum k/A: dtau/dt = N sqrt(1 - v_phys^2) = N A m / sqrt(k^2 c'^2 + A^2 m^2)
                dirac = Nb * A * m / np.hypot(CP * ky, A * m)
                # lattice band: mu_eff dE/dmu_eff of the walker band, times the lapse N/A
                C = lambda mm: expm(1j * X(mm))
                Ep = bands_C(np.array([0, ky, 0]), C(mu * A + DMU))[0]
                Em = bands_C(np.array([0, ky, 0]), C(mu * A - DMU))[0]
                band = (Nb / A) * mu * A * (Ep[UPPER(Ep)].mean() - Em[UPPER(Em)].mean()) / (2 * DMU)
                rows.append((mu, Nb, A, ky, QM / m, dirac, band / m, QM / QQ))
        print(f"   mu = {mu} done  [{time.time()-t0:.0f}s]")
    r = np.array(rows)
    out['B_rows'] = r
    print("   dtau/dt = Q_M / m   (simulation | lattice band | Dirac N sqrt(1 - v_phys^2)):")
    for row in r:
        mu, Nb, A, ky = row[:4]
        print(f"      mu = {mu:.2f}, N = {Nb:.3f}, A = {A:.1f}, k_y = {ky:.2f}: {row[4]:.4f} | {row[6]:.4f} | {row[5]:.4f}")
    dev_band = np.abs(r[:, 4] / r[:, 6] - 1).max()
    small = r[:, 0] == 0.03
    dev_d = np.abs(r[small, 4] / r[small, 5] - 1).max()
    print(f"   max |sim/band - 1| = {dev_band:.1e}; max |sim/Dirac - 1| = {dev_d:.3f} (mu = 0.03), "
          f"{np.abs(r[~small, 4] / r[~small, 5] - 1).max():.3f} (mu = 0.15)")


def main():
    parts = [p.upper() for p in sys.argv[1:]] or ['A', 'B']
    path = data_path('phase_bit_s4_open7.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in parts:
        {'A': part_A, 'B': part_B}[p](out)
        np.savez(path, **out)
    print("saved data/phase_bit_s4_open7.npz")


if __name__ == '__main__':
    main()
