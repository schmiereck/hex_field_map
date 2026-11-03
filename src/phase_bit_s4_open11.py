#!/usr/bin/env python3
"""
Paper 3, causal g0i: lapse Phi and shift b as local wave equations
==============================================================================

Both fields live on the nodes and obey a local wave equation (leapfrog) with the
local light speed in lattice coordinates, c_g = c' N0^2 (H2: N/A = N^2):
    Phi_tt = c_g^2 (Phi_xx - kappa s),        s = 2e - e_M      (round 6)
    b_tt   = c_g^2 (b_xx   - kappa_b p),      p = k_s rho_s,  kappa_b = -4 c'^2 N0^4 kappa
Static limit = the Green functions of open10.  Retarded 1D solution: the gradient
at the test is the static one restricted to the source part inside the light
cone -> F(t) = F_static * P(|dx| < c_g t).

Two moving sheets (source at -D/2, test at +D/2, Bloch k_y), fields start at 0.
Variants: 'instant' (both Green functions), 'causal' (both waves), 'mixed'
(Phi causal, b instantaneous), 'nob' (Phi causal, no b).
For parallel light beams the static forces cancel; only if Phi and b travel
together does the test feel nothing at ALL times.

Run: python phase_bit_s4_open11.py
"""

import functools
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import DT_HALF, LAT_A
from phase_bit_s4_open import CP
from phase_bit_s4_open8 import Scheme
from phase_bit_s4_open9 import Xl, shift_chain, lam_of
from phase_bit_s4_open10 import A_FIT, kdrag, Chain, make_packet, filt
from paths import data_path

print = functools.partial(print, flush=True)


class _Op:
    def __init__(self, f):
        self.f = f

    def __matmul__(self, v):
        return self.f(v)


class DynChain:
    """H2 compensated chain whose lapse N(x) and drag b(x) can change every tick.
    Coins from a lookup table in N (lambda(N) compensation)."""

    def __init__(self, mu, ky, Nx):
        self.Nx, self.ky = Nx, ky
        self.Ng = np.linspace(0.30, 0.999, 1400)
        self.Ctab = np.array([expm(1j * Xl(lam_of(max(N, 0.4), A_FIT))(mu)) for N in self.Ng])
        self.P = shift_chain((ky, 0.0), Nx)
        coin = lambda v: np.einsum('xij,xj->xi', self.C, v.reshape(Nx, 24)).ravel()
        self.sc = Scheme('WH', _Op(lambda v: self.P @ coin(v)), _Op(coin))
        self.ph = None

    def update(self, Nf, bf=None):
        idx = np.clip(np.searchsorted(self.Ng, Nf), 0, len(self.Ng) - 1)
        self.C = self.Ctab[idx]
        self.sc.set_N(Nf, Nf)
        self.ph = None if bf is None else np.repeat(np.exp(-1j * DT_HALF * bf * kdrag(self.ky)), 24)

    def tick(self, st):
        st = self.sc.tick(st)
        return st if self.ph is None else [self.ph * a for a in st]


def main():
    t0 = time.time()
    N0, Nx, SIG, D, gphi, nst = 0.8, 640, 16.0, 150, 3e-4, 800      # weak: <k_x> changes << k_y
    cg = CP * N0 ** 2
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    dx = LAT_A
    xS, xT = -D / 2 * LAT_A, D / 2 * LAT_A
    ts = np.arange(nst + 1) * DT_HALF
    KX = 2 * np.pi * np.fft.fftfreq(Nx) / LAT_A
    absdx = np.abs(xs[:, None] - xs[None, :])
    Nuni = np.full(Nx, N0)
    print(f"causal g0i: N0 = {N0}, c_g = c' N0^2 = {cg:.4f}, D = {D * LAT_A:.1f} -> D/c_g = {D * LAT_A / cg:.0f} "
          f"(= {D * LAT_A / cg / DT_HALF:.0f} ticks), {nst} ticks; Courant {cg * DT_HALF / dx:.2f}")
    out = {'ts': ts, 'par': np.array([N0, Nx, SIG, D, gphi, nst, cg])}
    cases = [('light, parallel', 0.0, 0.2, 0.2), ('light, antiparallel', 0.0, 0.2, -0.2),
             ('mu=0.03, parallel', 0.03, 0.2, 0.2), ('mu=0.03, antiparallel', 0.03, 0.2, -0.2)]

    def kmean(st):
        w = sum((np.abs(np.fft.fft(a.reshape(Nx, 24), axis=0)) ** 2).sum(1) for a in st)
        return (w @ KX) / w.sum()

    def com(st):
        w = sum(np.abs(a) ** 2 for a in st).reshape(Nx, 24).sum(1)
        return (w @ xs) / w.sum()

    def lap(f):
        L = np.zeros_like(f)
        L[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dx ** 2
        return L

    for cname, mu, ks, kt in cases:
        # source loads at uniform N0 (static sheet)
        st, _ = make_packet(mu, ks, N0, Nx, xs, xS, SIG)
        ch = Chain(mu, ks, Nx, Nuni)
        st2 = ch.tick(st)
        e = sum((np.conj(a) * 1j * (b2 - a)).real for a, b2 in zip(st, st2)).reshape(Nx, 24).sum(1) / DT_HALF
        if mu > 0:
            dm = 1e-5
            sp_, sm_ = Chain(mu, ks, Nx, Nuni, dmu=dm).tick(st), Chain(mu, ks, Nx, Nuni, dmu=-dm).tick(st)
            eM = mu * sum((1j * np.conj(u2) * (p - m) / (2 * dm)).real for u2, p, m in zip(st2, sp_, sm_)
                          ).reshape(Nx, 24).sum(1) / DT_HALF
        else:
            eM = np.zeros(Nx)
        rho = sum(np.abs(a) ** 2 for a in st).reshape(Nx, 24).sum(1)
        s = 2 * e - eM
        kap = 2 * gphi / s.sum()
        kapb = -4 * CP ** 2 * N0 ** 4 * kap
        p = ks * rho
        Phi_st = 0.5 * kap * absdx @ s
        b_st = 0.5 * kapb * absdx @ p

        # light-cone prediction: F(t) = F_static * P(|x_T - x_S| < c_g t), test- and source-weighted
        stT, ET = make_packet(mu, kt, N0, Nx, xs, xT, SIG)
        wT = sum(np.abs(a) ** 2 for a in stT).reshape(Nx, 24).sum(1)
        wT /= wT.sum()
        dist = np.abs(xs[:, None] - xs[None, :])
        sw = s / s.sum()
        P = np.array([wT @ ((dist < cg * t) @ sw) for t in ts])
        out[f'{cname}|P'] = P

        dyn = DynChain(mu, kt, Nx)
        res = {}
        for var in ('instant', 'inst_nob', 'causal', 'mixed', 'nob'):
            kcs, xcs = [], []
            for member in (0, 1):
                stt, Et = make_packet(mu, kt, N0, Nx, xs, xT, SIG, member)
                dyn.update(Nuni)
                stt = filt(dyn, stt, Et)
                Phi, Phi_old = np.zeros(Nx), np.zeros(Nx)
                b, b_old = np.zeros(Nx), np.zeros(Nx)
                kc, xc = [kmean(stt)], [com(stt)]
                for it in range(nst):
                    t = it * DT_HALF
                    # fields at this tick
                    if var == 'instant':
                        Pf, bfld = Phi_st, b_st
                    elif var == 'inst_nob':
                        Pf, bfld = Phi_st, None
                    else:
                        Pf = Phi
                        bfld = b_st if var == 'mixed' else (b if var == 'causal' else None)
                    xr = xc[-1]
                    Nf = np.clip(N0 * np.exp(Pf - np.interp(xr, xs, Pf)), 1e-3, 0.999)
                    bf = None if bfld is None else bfld - np.interp(xr, xs, bfld)
                    dyn.update(Nf, bf)
                    stt = dyn.tick(stt)
                    kc.append(kmean(stt))
                    xc.append(com(stt))
                    # advance the wave equations (leapfrog, fixed ends)
                    c2 = (cg * DT_HALF) ** 2
                    # source per node / dx: the static limit is then exactly the Green-function sum
                    Phi_new = 2 * Phi - Phi_old + c2 * (lap(Phi) - kap * s / dx)
                    b_new = 2 * b - b_old + c2 * (lap(b) - kapb * p / dx)
                    Phi_new[[0, -1]] = 0.0
                    b_new[[0, -1]] = 0.0
                    Phi_old, Phi, b_old, b = Phi, Phi_new, b, b_new
                kcs.append(kc)
                xcs.append(xc)
            kc = np.mean(kcs, 0)
            res[var] = kc
            out[f'{cname}|{var}|k'] = kc
            out[f'{cname}|{var}|x'] = np.mean(xcs, 0)
            print(f"   {cname:22s} {var:8s}: <k_x>(t) - k0 at t = 100/200/300/400: "
                  + " ".join(f"{(kc[int(tt / DT_HALF)] - kc[0]) / kap * 1e3:+.4f}" for tt in (100, 200, 300, 400))
                  + f"  (1e-3 kappa)  [{time.time()-t0:.0f}s]")
        # static forces from the instantaneous runs; light-cone predictions for the retarded runs
        F_all = np.polyfit(ts, res['instant'], 1)[0]
        F_lap = np.polyfit(ts, res['inst_nob'], 1)[0]
        F_drag = F_all - F_lap
        # time-resolved static forces (sliding linear fit, +-20 time units) folded with the light cone
        w = 40
        Fw = lambda k: np.array([np.polyfit(ts[max(0, i - w):i + w + 1], k[max(0, i - w):i + w + 1], 1)[0]
                                 for i in range(len(ts))])
        Fa, Fl = Fw(res['instant']), Fw(res['inst_nob'])
        cum = lambda f: np.concatenate([[0], np.cumsum(f[:-1]) * DT_HALF])
        preds = {'causal': cum(Fa * P), 'nob': cum(Fl * P), 'mixed': cum(Fa - Fl) + cum(Fl * P)}
        out[f'{cname}|F_inst'] = np.array([Fa, Fl])
        line = f"      static: total {F_all/kap*1e3:+.4f}, lapse {F_lap/kap*1e3:+.4f}, drag {F_drag/kap*1e3:+.4f};"
        for var, pr in preds.items():
            out[f'{cname}|pred_{var}'] = pr
            d_ = res[var] - res[var][0]
            scale = np.abs(pr).max()
            rel = np.sqrt(np.mean((d_ - pr) ** 2)) / scale
            line += f" {var}: end {d_[-1]/kap*1e3:+.4f} (pred {pr[-1]/kap*1e3:+.4f}, rms/max {rel:.3f});"
        early = ts < 0.8 * (D * LAT_A - 3 * SIG) / cg
        dc = res['causal'] - res['causal'][0]
        dm_ = res['mixed'] - res['mixed'][0]
        line += f" before arrival (t < {ts[early][-1]:.0f}): causal {dc[early][-1]/kap*1e3:+.5f}, mixed {dm_[early][-1]/kap*1e3:+.4f}"
        print(line + "  (1e-3 kappa)")
        out[f'{cname}|summary'] = np.array([F_all, F_lap, F_drag, kap, dc[early][-1], dm_[early][-1]])
    np.savez(data_path('phase_bit_s4_open11.npz'), **out)
    print("saved data/phase_bit_s4_open11.npz")


if __name__ == '__main__':
    main()
