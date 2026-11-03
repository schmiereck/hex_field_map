#!/usr/bin/env python3
"""
Paper 3, g0i: gravitomagnetism from the momentum current of the load
==============================================================================

Linearised GR: a moving source adds a shift (g0i) to the metric; for a test
particle H = N E(p/A) + b.p.  For two bodies moving along y the transverse
attraction carries the factor (1 + v_s^2)(1 + v_t^2) - C v_s v_t with C = 4:
parallel light beams do not attract, antiparallel ones twice as strongly as
without g0i, and co-moving bodies obey the relativity principle
(lab acceleration = rest acceleration / gamma^3 for sheets) only for C = 4.

A.  Band level.
    A1  first attempt: heading-dependent stay gate N_h(d) = N_h (1 + u_d.w)
        -> linear term, but Kramers splitting and a vacuum shift O(w^2).
    A2  drag hop: every state, independent of its heading, is handed on along
        y with the generator b sin(k_y l)/l (l = 0.612, the y step of the FCC
        neighbours (0, l, +-l)); exact scalar: E -> E + b sin(k_y l)/l for all
        bands, Dirac point and Kramers untouched.
B.  x-chain, two sheets (Bloch k_y): source at -D/2, test at +D/2, H2
    compensated scheme (wait + stay gates, lambda(N)), N = N0 exp(Phi),
    Phi = (kappa/2)|x| * s with s = 2e - e_M of the source (measured), drag
        b(x) = -4 c'^2 N0^4 (kappa/2) |x| * (k_s rho_s)     (momentum density)
    (N0^4: lattice vs physical coordinates around the background N0, A0 = 1/N0).
    Test acceleration with drag off / on for parallel, antiparallel, co-moving
    and light pairs.

Run parts: python phase_bit_s4_open10.py [A] [B]   (default: both)
"""

import functools
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import MOVES_PHYS, DT_HALF, LAT_A
from phase_bit_s4_open import CP
from phase_bit_s4_open4 import a_of
from phase_bit_s4_open8 import Scheme, energies, near, ONE
from phase_bit_s4_open9 import Xl, LAM0, Sk, shift_chain, lam_of
from paths import data_path

print = functools.partial(print, flush=True)
UD = MOVES_PHYS / np.linalg.norm(MOVES_PHYS, axis=1)[:, None]
ELL = LAT_A                                    # y step of the neighbours (0, l, +-l)
kdrag = lambda ky, kz=0.0: np.sin(ky * ELL) * np.cos(kz * ELL) / ELL   # pair (0, l, +l), (0, l, -l)
A_FIT = np.load(data_path('phase_bit_s4_open9.npz'))['A_fit']


def bloch(k, mu, Nw, Nh, w=None, b=0.0, lam=LAM0):
    """96x96 H2 tick at Bloch k; optional heading-dependent stay gate (w) and drag b along y."""
    C = expm(1j * Xl(lam)(mu))
    sc = Scheme('WH', Sk(k) @ C, C)
    sc.set_N(Nw, Nh)
    if w is not None:
        Nd = np.repeat(Nh * (1 + UD @ np.asarray(w, float)), 2)
        sc.ch, sc.sh = np.cos(a_of(Nd)), np.sin(a_of(Nd))
    cols = []
    for j in range(96):
        e = np.zeros(96, complex)
        e[j] = 1
        cols.append(np.concatenate(sc.tick(list(e.reshape(4, 24)))))
    U = np.array(cols).T
    return np.exp(-1j * DT_HALF * b * kdrag(k[1], k[2])) * U


def low4(U):
    E = energies(U)
    return np.sort(E[np.argsort(np.abs(E))[:4]])


# ═════════════════════════════════════════════════════════════════════════════
def part_A(out):
    print("A. band level (2O*, N_w = N_h = 0.75)")
    N = 0.75
    rows1 = []
    print("   A1  heading-dependent stay gate N_h(d) = N_h (1 + u_d.w), w = 0.02 along y")
    for mu in (0.0, 0.03):
        for kv in ((0, 0, 0), (0, 0.02, 0), (0, -0.02, 0), (0.02, 0, 0)):
            E0 = low4(bloch(kv, mu, N, N))
            E1 = low4(bloch(kv, mu, N, N, w=(0, 0.02, 0)))
            dE = E1 - E0
            split = max(dE[1] - dE[0], dE[3] - dE[2])
            rows1.append((mu, *kv, *dE, split))
            print(f"      mu = {mu}, k = {kv}: dE = {np.round(dE, 6)}, doublet splitting {split:.1e}")
    out['A1'] = np.array(rows1)
    rows2 = []
    print("   A2  drag hop b sin(k_y l)/l, b = 0.02")
    for mu in (0.0, 0.03):
        for kv in ((0, 0, 0), (0, 0.05, 0), (0, -0.05, 0), (0.05, 0, 0), (0, 0.2, 0.1), (0.1, 0.1, 0.1)):
            E0 = low4(bloch(kv, mu, N, N))
            E1 = low4(bloch(kv, mu, N, N, b=0.02))
            pred = 0.02 * kdrag(kv[1], kv[2])
            dev = np.abs(E1 - E0 - pred).max()
            rows2.append((mu, *kv, pred, dev))
            print(f"      mu = {mu}, k = {kv}: dE = {np.round(E1 - E0, 7)} vs b sin(k_y l)cos(k_z l)/l = {pred:+.7f}; max dev {dev:.1e}")
    out['A2'] = np.array(rows2)


# ═════════════════════════════════════════════════════════════════════════════
class Chain:
    """H2 compensated scheme on the x-chain with lapse field N(x) and drag b(x)."""

    def __init__(self, mu, ky, Nx, Nf, bf=None, dmu=0.0):
        lam = lam_of(np.clip(Nf, 0.4, 0.999), A_FIT)
        Cs = [expm(1j * Xl(l)(mu + dmu)) for l in lam]
        Cb = sp.block_diag([sp.csr_matrix(C_) for C_ in Cs], format='csr')
        self.sc = Scheme('WH', shift_chain((ky, 0.0), Nx) @ Cb, Cb)
        self.sc.set_N(Nf, Nf)
        self.ph = None if bf is None else np.repeat(np.exp(-1j * DT_HALF * bf * kdrag(ky)), 24)

    def tick(self, st):
        st = self.sc.tick(st)
        return st if self.ph is None else [self.ph * a for a in st]


def make_packet(mu, ky, N0, Nx, xs, x0, SIG, member=0):
    lam = lam_of(N0, A_FIT)
    kk = abs(ky)
    guess = lambda q: N0 * np.hypot(N0 * CP * q, 2 * mu) if mu > 0 else N0 * N0 * CP * q
    E0, V0 = np.linalg.eig(bloch((0, ky, 0), mu, N0, N0, lam=lam))
    E0 = -np.angle(E0) / DT_HALF
    Eb = near(E0, guess(kk), 2).mean()
    Q0, _ = np.linalg.qr(V0[:, np.argsort(np.abs(E0 - Eb))[:2]])
    uref = Q0[:, member]                      # one member of the Kramers doublet
    psi = np.zeros((Nx, 96), complex)
    for kx in 2 * np.pi * np.fft.fftfreq(Nx) / LAT_A:
        gk = np.exp(-SIG ** 2 * kx ** 2 / 2)
        if gk < 1e-8:
            continue
        Ek, Vk = np.linalg.eig(bloch((kx, ky, 0), mu, N0, N0, lam=lam))
        Ek = -np.angle(Ek) / DT_HALF
        Q, _ = np.linalg.qr(Vk[:, np.argsort(np.abs(Ek - guess(np.hypot(kx, kk))))[:2]])
        u = Q @ (Q.conj().T @ uref)
        psi += gk * np.exp(1j * kx * (xs - x0))[:, None] * (u / np.linalg.norm(u))[None, :]
    psi /= np.linalg.norm(psi)
    st = [psi[:, 24 * q:24 * (q + 1)].ravel() for q in range(4)]
    return st, Eb


def band_v(mu, ky, N0):
    lam = lam_of(N0, A_FIT)
    guess = lambda q: N0 * np.hypot(N0 * CP * q, 2 * mu) if mu > 0 else N0 * N0 * CP * abs(q)
    E = lambda q: near(energies(bloch((0, q, 0), mu, N0, N0, lam=lam)), guess(q), 2).mean()
    return (E(ky + 1e-4) - E(ky - 1e-4)) / 2e-4, E(ky)


def filt(ch, st, Eb):
    """in the actual field chain: (1 + ph U)/2 against the pi branches of the gates."""
    ph = np.exp(1j * Eb * DT_HALF)
    for _ in range(3):
        st2 = ch.tick(st)
        st = [(a + ph * b) / 2 for a, b in zip(st, st2)]
    nrm = np.sqrt(sum(np.vdot(a, a).real for a in st))
    return [a / nrm for a in st]


def part_B(out):
    N0, Nx, SIG, nst, D, gphi = 0.6, 640, 32.0, 320, 220, 3e-3     # both packets >= 4 sigma from the open ends
    print(f"B. x-chain: source and test sheets (Bloch k_y), H2 compensated, N0 = {N0}, D = {D} layers")
    t0 = time.time()
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    xS, xT = -D / 2 * LAT_A, D / 2 * LAT_A
    ts = np.arange(nst + 1) * DT_HALF
    absdx = np.abs(xs[:, None] - xs[None, :])
    KX = 2 * np.pi * np.fft.fftfreq(Nx) / LAT_A
    Nuni = np.full(Nx, N0)
    rows = []
    cases = [(0.03, 0.0, 0.03, 0.0), (0.03, 0.1, 0.03, 0.1), (0.03, 0.2, 0.03, 0.2),
             (0.03, 0.1, 0.03, -0.1), (0.03, 0.2, 0.03, -0.2), (0.03, 0.2, 0.03, 0.0), (0.03, 0.0, 0.03, 0.2),
             (0.0, 0.2, 0.0, 0.2), (0.0, 0.2, 0.0, -0.2), (0.0, 0.1, 0.0, 0.1), (0.0, 0.1, 0.0, -0.1)]
    for mus, ks, mut, kt in cases:
        # source loads (uniform N0): e, e_M (dU/dmu by central difference), momentum density k_s rho
        st, Es = make_packet(mus, ks, N0, Nx, xs, xS, SIG)
        ch = Chain(mus, ks, Nx, Nuni)
        st2 = ch.tick(st)
        e = sum((np.conj(a) * 1j * (b2 - a)).real for a, b2 in zip(st, st2)).reshape(Nx, 24).sum(1) / DT_HALF
        if mus > 0:
            dm = 1e-5
            sp_, sm_ = Chain(mus, ks, Nx, Nuni, dmu=dm).tick(st), Chain(mus, ks, Nx, Nuni, dmu=-dm).tick(st)
            eM = mus * sum((1j * np.conj(u2) * (p - m) / (2 * dm)).real for u2, p, m in zip(st2, sp_, sm_)
                           ).reshape(Nx, 24).sum(1) / DT_HALF
        else:
            eM = np.zeros(Nx)
        rho = sum(np.abs(a) ** 2 for a in st).reshape(Nx, 24).sum(1)
        s = 2 * e - eM
        kap = 2 * gphi / s.sum()                      # fixed field gradient; results quoted per unit kappa
        Phi = 0.5 * kap * absdx @ s
        bq = -4 * CP ** 2 * N0 ** 4 * 0.5 * kap * absdx @ (ks * rho)
        iT = np.argmin(np.abs(xs - xT))
        Nf = np.clip(N0 * np.exp(Phi - Phi[iT]), 1e-3, 0.999)
        bf = bq - bq[iT]
        vs_, _ = band_v(mus, ks, N0)
        vt_, _ = band_v(mut, kt, N0)
        vs, vt = vs_ / (CP * N0 ** 2), vt_ / (CP * N0 ** 2)
        res = [mus, ks, mut, kt, s.sum(), e.sum(), eM.sum(), vs, vt, kap]
        # ideal GR with the same fields: H = N E(k / A) + b sin(k_y l)/l, A = 1/N (lattice coordinates),
        # averaged over the test packet's x and k_x distributions
        st0, _ = make_packet(mut, kt, N0, Nx, xs, xT, SIG)
        wx = sum(np.abs(a) ** 2 for a in st0).reshape(Nx, 24).sum(1)
        wk = sum((np.abs(np.fft.fft(a.reshape(Nx, 24), axis=0)) ** 2).sum(1) for a in st0)
        wx, wk = wx / wx.sum(), wk / wk.sum()
        dN, db = np.gradient(Nf, xs), np.gradient(bf, xs)
        kk = np.hypot(KX, kt)
        Eloc = np.hypot(2 * mut, CP * Nf[:, None] * kk[None, :])
        dH = dN[:, None] * Eloc + Nf[:, None] ** 2 * (CP * kk[None, :]) ** 2 * dN[:, None] / Eloc
        F_gr_off = -np.sum(wx[:, None] * wk[None, :] * dH)
        F_gr_on = F_gr_off - np.sum(wx * db) * kdrag(kt)

        def kmean(st):
            w = sum((np.abs(np.fft.fft(a.reshape(Nx, 24), axis=0)) ** 2).sum(1) for a in st)
            return (w @ KX) / w.sum()
        per = []
        for drag in (False, True):
            chT = Chain(mut, kt, Nx, Nf, bf if drag else None)
            am = []
            for member in (0, 1):                 # both Kramers partners: removes the spin-dependent part
                stt, Et = make_packet(mut, kt, N0, Nx, xs, xT, SIG, member)
                stt = filt(chT, stt, Et)
                kc = [kmean(stt)]
                for _ in range(nst):
                    stt = chT.tick(stt)
                    kc.append(kmean(stt))
                # force F = d<k_x>/dt; the lapse gradient mixes in the opposite band -> fit the
                # Zitterbewegung at omega = 2 E together with the linear rise
                w_ = 2 * abs(Et)
                Mf = np.stack([np.ones_like(ts), ts, np.cos(w_ * ts), np.sin(w_ * ts)], 1)
                am.append(np.linalg.lstsq(Mf, np.array(kc), rcond=None)[0][1])
            per.append(am)
            res.append(np.mean(am))
        res += [per[0][0], per[0][1], per[1][0], per[1][1], F_gr_off, F_gr_on]
        a_off, a_on = res[10], res[11]
        pred = 1 - 4 * vs * vt / ((1 + vs ** 2) * (1 + vt ** 2))
        res += [a_on / a_off, pred]
        rows.append(res)
        print(f"   source mu {mus}, k {ks:+.1f} (v {vs:+.3f}) | test mu {mut}, k {kt:+.1f} (v {vt:+.3f}): "
              f"Q_s = {s.sum():.4f}; F/kappa: off {a_off/kap:+.4e}, on {a_on/kap:+.4e}; on/off {a_on/a_off:+.3f} "
              f"(GR 1 - 4 v_s v_t/((1+v_s^2)(1+v_t^2)) = {pred:+.3f})  [{time.time()-t0:.0f}s]")
        print(f"      ideal GR for these packets: F/kappa off {F_gr_off/kap:+.4e}, on {F_gr_on/kap:+.4e}, "
              f"on/off {F_gr_on/F_gr_off:+.3f};  sim/GR: off {a_off/F_gr_off:.3f}, on {a_on/F_gr_on:.3f}")
        print(f"      Kramers members F/kappa: off {per[0][0]/kap:+.4e} / {per[0][1]/kap:+.4e}, on {per[1][0]/kap:+.4e} / {per[1][1]/kap:+.4e}")
    r = np.array(rows)
    out['B_rows'] = r
    out['B_par'] = np.array([N0, Nx, SIG, nst, D, gphi])
    # relativity principle: co-moving identical sheets vs both at rest (per unit kappa, per particle)
    lam = lam_of(N0, A_FIT)
    guess = lambda q, ky: N0 * np.hypot(N0 * CP * np.hypot(q, ky), 0.06)
    Eb = lambda q, ky: near(energies(bloch((q, ky, 0), 0.03, N0, N0, lam=lam)), guess(q, ky), 2).mean()
    d2 = lambda ky: (Eb(1e-3, ky) - 2 * Eb(0.0, ky) + Eb(-1e-3, ky)) / 1e-6
    ref = r[0, 11] / r[0, 9] * d2(0.0)
    print("   co-moving identical sheets, a(k,k)/a(0,0) = F d2E/dk_x^2 per unit kappa (same particle number per cell):")
    cm = []
    for i in (1, 2):
        v = r[i, 7]
        g = 1 / np.sqrt(1 - v ** 2)
        on, off = r[i, 11] / r[i, 9] * d2(r[i, 1]) / ref, r[i, 10] / r[i, 9] * d2(r[i, 1]) / ref
        grp = (r[i, 17] / r[i, 9] * d2(r[i, 1])) / (r[0, 17] / r[0, 9] * d2(0.0))
        cm.append((r[i, 1], v, on, off, g ** -3, g * (1 + v ** 2) ** 2, grp))
        print(f"      k = {r[i,1]}: v = {v:.3f}, drag on {on:.3f}, drag off {off:.3f}; GR for these packets {grp:.3f}, "
              f"ideal sheets 1/gamma^3 = {g**-3:.3f}, lapse only gamma(1+v^2)^2 = {g*(1+v**2)**2:.3f}")
    out['B_comove'] = np.array(cm)


def main():
    parts = [p.upper() for p in sys.argv[1:]] or ['A', 'B']
    path = data_path('phase_bit_s4_open10.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in parts:
        {'A': part_A, 'B': part_B}[p](out)
        np.savez(path, **out)
    print("saved data/phase_bit_s4_open10.npz")


if __name__ == '__main__':
    main()
