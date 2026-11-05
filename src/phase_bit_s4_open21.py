#!/usr/bin/env python3
"""
Bit test 2: free fall with a deterministic busy schedule that varies in space
==============================================================================

Bit test 1 (open20): with a uniform schedule, a node that parks its content while busy
(toggle: NOT of the active/waiting flag at the start and end of a busy interval) gives an
exact lapse N = 1 - p.  Here the schedule depends on the layer x of the x-chain
(Bloch in y, z), so free neighbours send amplitude into busy nodes and vice versa.

Schedules (busy fraction p(x) = 1 - N(x), isolated busy ticks for p < 1/2):
    common phase  sigma-delta b = floor((t+1) p + phi) - floor(t p + phi), phi = 0 everywhere
    random phase  same with a random phi per layer
    nested        events only on odd ticks, event k busy where u_k < 2 p(x) with ONE common
                  sequence u_k = frac(k alpha + u0): neighbours differ only for u_k between their
                  thresholds (mismatch ~ grad p, constant in time)
A busy tick at x toggles at t and t + 1: content arriving at x in tick t skips tick t + 1.

Gates (2O*, eps 0.3, lambda*, as open8/open9):
    wait toggle  W: clock and hop of the content are skipped (bit form of 1b)
    stay toggle  H: the hop is refused, the coin runs (bit form of the stay gate)

A.  Bloch, periodic patterns: gap = (1 - p_w) 2 mu, v = (1 - p_w - p_h) c' exactly, as long as
    the W and H events of a node are >= 2 ticks apart; colliding toggles mix the sectors.
B.  x-chain, wait toggle, N(x) = N0 (1 + g x), N0 = 0.75, g = 2e-3, sigma 16, 160 ticks:
    acceleration / semiclassics N(x) E_free(k) for the three schedules, continuous 1b for
    comparison; leakage = weight outside the active register on layers that are not parked in
    this tick.  Control: free walker with the ideal skip phase exp(+i E dt) on the busy layers
    of the nested schedule (no buffer).
    B2  nested schedule: other sequences, packet widths, gradient.

    python phase_bit_s4_open21.py [A] [B]   (default: both; ~2 min)
"""

import functools
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import DT_HALF, LAT_A
from phase_bit_s4_open import CP
from phase_bit_s4_open8 import Sk, Scheme, bloch_U, energies, near, ONE
from phase_bit_s4_open9 import Xl, LAM0, shift_chain
from paths import data_path

print = functools.partial(print, flush=True)
P2 = np.pi / 2
X0 = Xl(LAM0)


# ═════════════════════════════════════════════════════════════════════════════
def tick_bloch(k, mu, aw, ah):
    C = expm(1j * X0(mu))
    S = Sk(k)
    sc = Scheme('WH', S @ C, C, S)
    sc.cw, sc.sw = np.full(24, np.cos(aw)), np.full(24, np.sin(aw))
    sc.ch, sc.sh = np.full(24, np.cos(ah)), np.full(24, np.sin(ah))
    cols = []
    for j in range(96):
        e = np.zeros(96, complex)
        e[j] = 1
        cols.append(np.concatenate(sc.tick(list(e.reshape(4, 24)))))
    return np.array(cols).T


def floquet(bw, bh, k, mu):
    U = np.eye(96, dtype=complex)
    pw, ph = bw[-1], bh[-1]
    for w, h in zip(bw, bh):
        U = tick_bloch(k, mu, P2 * (w != pw), P2 * (h != ph)) @ U
        pw, ph = w, h
    return U


def branch(U, steps, k, mu):
    w0, V0 = np.linalg.eig(tick_bloch(k, mu, 0.0, 0.0))
    E0 = -np.angle(w0) / DT_HALF
    Q, _ = np.linalg.qr(V0[:, np.argsort(np.abs(E0 - np.hypot(CP * np.linalg.norm(k), 2 * mu)))[:2]])
    w, V = np.linalg.eig(U)
    E = -np.angle(w) / (steps * DT_HALF)
    ov = np.linalg.norm(Q.conj().T @ V, axis=0) / np.linalg.norm(V, axis=0)
    ov[(np.abs(E) > 0.5) | (E < 0)] = -1
    j = int(np.argmax(ov))
    return E[j], ov[j]


def part_A(out):
    print("A. Bloch: wait (W) and stay (H) toggles, periodic patterns starting free (1 = event)")
    KK, MU = 0.01, 0.03
    pats = [([1, 0, 0, 0], [0, 0, 0, 0]), ([0, 0, 0, 0], [1, 0, 0, 0]), ([1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0, 0], [0, 0, 1, 0, 0]), ([1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]),
            ([1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, 0]), ([1, 0, 0, 0], [1, 0, 0, 0]), ([1, 0, 0, 0], [0, 0, 0, 1])]
    rows = []
    for bw, bh in pats:
        m = len(bw)
        Eg, og = branch(floquet(bw, bh, np.zeros(3), MU), m, np.zeros(3), MU)
        Ek, ok = branch(floquet(bw, bh, np.array([0, KK, 0]), 0.0), m, np.array([0, KK, 0]), 0.0)
        pg, pv = 1 - sum(bw) / m, 1 - (sum(bw) + sum(bh)) / m
        rows.append((m, sum(bw), sum(bh), Eg / (2 * MU), Ek / (CP * KK), og, ok, pg, pv))
        print(f"   W {str(bw):26s} H {str(bh):26s}: gap/(2 mu) = {Eg/(2*MU):.4f} (pred {pg:.4f}), "
              f"v/c' = {Ek/(CP*KK):.4f} (pred {pv:.4f}), overlap {og:.2f}/{ok:.2f}")
    out['A'] = np.array(rows)


# ═════════════════════════════════════════════════════════════════════════════
def sigma_delta(p, phi, nt):
    """busy bits b[t, x] for t = 0 .. nt - 1 (busy fraction p per layer, phase phi)."""
    t = np.arange(nt + 1)[:, None]
    f = np.floor(t * p[None, :] + phi[None, :])
    return (f[1:] - f[:-1]).astype(int)


def toggles(b):
    """gate angle pi/2 where the bit changes (prev = 0 before t = 0)."""
    prev = np.vstack([np.zeros((1, b.shape[1]), int), b[:-1]])
    return (b != prev)


def free_packet(mu, ky, Nx, SIG, xs):
    """band-projected packet of the free walker (24 components per layer, active register)."""
    C = expm(1j * X0(mu))
    guess = lambda kx: np.hypot(CP * np.hypot(kx, ky), 2 * mu)
    E0, V0 = np.linalg.eig(Sk((0, ky, 0)) @ C)
    E0 = -np.angle(E0) / DT_HALF
    uref = V0[:, np.argsort(np.abs(E0 - guess(0.0)))[0]]
    psi = np.zeros((Nx, 24), complex)
    for kx in 2 * np.pi * np.fft.fftfreq(Nx) / LAT_A:
        gk = np.exp(-SIG ** 2 * kx ** 2 / 2)
        if gk < 1e-8:
            continue
        Ek, Vk = np.linalg.eig(Sk((kx, ky, 0)) @ C)
        Ek = -np.angle(Ek) / DT_HALF
        Q, _ = np.linalg.qr(Vk[:, np.argsort(np.abs(Ek - guess(kx)))[:2]])
        u = Q @ (Q.conj().T @ uref)
        psi += gk * np.exp(1j * kx * xs)[:, None] * (u / np.linalg.norm(u))[None, :]
    return psi / np.linalg.norm(psi)


def scheme_packet(kind, mu, ky, Nw0, Nh0, Nx, SIG, xs):
    """packet from Bloch eigenvectors of the continuous scheme at the base lapse (as open8 B)."""
    nr = Scheme(kind, None, None).nreg()
    bd = lambda kx: Nw0 * np.hypot(Nh0 * CP * np.hypot(kx, ky), 2 * mu)
    E0, V0 = np.linalg.eig(bloch_U(X0, kind, (0, ky, 0), mu, Nw0, Nh0))
    E0 = -np.angle(E0) / DT_HALF
    Eb = near(E0, bd(0.0), 2).mean()
    uref = V0[:, np.argsort(np.abs(E0 - Eb))[0]]
    psi = np.zeros((Nx, 24 * nr), complex)
    for kx in 2 * np.pi * np.fft.fftfreq(Nx) / LAT_A:
        gk = np.exp(-SIG ** 2 * kx ** 2 / 2)
        if gk < 1e-8:
            continue
        Ek, Vk = np.linalg.eig(bloch_U(X0, kind, (kx, ky, 0), mu, Nw0, Nh0))
        Ek = -np.angle(Ek) / DT_HALF
        Q, _ = np.linalg.qr(Vk[:, np.argsort(np.abs(Ek - near(Ek, bd(kx), 2).mean()))[:2]])
        u = Q @ (Q.conj().T @ uref)
        psi += gk * np.exp(1j * kx * xs)[:, None] * (u / np.linalg.norm(u))[None, :]
    return psi / np.linalg.norm(psi), Eb


class Chain:
    def __init__(self, mu, ky, Nx):
        C = expm(1j * X0(mu))
        self.Cb = sp.kron(sp.eye(Nx), sp.csr_matrix(C), format='csr')
        self.T = shift_chain((ky, 0.0), Nx) @ self.Cb
        self.Nx = Nx

    def com(self, st, xs):
        w = sum((np.abs(a) ** 2).reshape(self.Nx, 24).sum(1) for a in st)
        return (w @ xs) / w.sum()

    def run_cont(self, kind, Nw, Nh, st, nst, xs, Eb):
        sc = Scheme(kind, self.T, self.Cb)
        sc.set_N(Nw, Nh)
        ph = np.exp(1j * Eb * DT_HALF)
        for _ in range(3):                                   # spectral filter (as open8 B)
            st2 = sc.tick(st)
            st = [(a + ph * b) / 2 for a, b in zip(st, st2)]
        nrm = np.sqrt(sum(np.vdot(a, a).real for a in st))
        st = [a / nrm for a in st]
        xc = [self.com(st, xs)]
        for _ in range(nst):
            st = sc.tick(st)
            xc.append(self.com(st, xs))
        return np.array(xc)

    def run_bit(self, bw, bh, psi, nst, xs):
        """bit schedule: bw, bh [t, x]; returns CoM and leakage (weight outside register 0 on
        layers whose bits are 0 in this tick, i.e. not parked)."""
        kind = 'WH' if bh is not None else 'W'
        sc = Scheme(kind, self.T, self.Cb)
        nr = sc.nreg()
        st = [psi.ravel()] + [np.zeros(24 * self.Nx, complex) for _ in range(nr - 1)]
        tw = toggles(bw)
        th = toggles(bh) if bh is not None else None
        xc, leak = [self.com(st, xs)], [0.0]
        for t in range(nst):
            aw = np.where(tw[t], P2, 0.0)
            sc.cw, sc.sw = np.repeat(np.cos(aw), 24), np.repeat(np.sin(aw), 24)
            busy = bw[t].astype(bool)
            if th is not None:
                ah = np.where(th[t], P2, 0.0)
                sc.ch, sc.sh = np.repeat(np.cos(ah), 24), np.repeat(np.sin(ah), 24)
                busy = busy | bh[t].astype(bool)
            st = sc.tick(st)
            xc.append(self.com(st, xs))
            off = sum((np.abs(a) ** 2).reshape(self.Nx, 24).sum(1) for a in st[1:])
            leak.append(off[~busy].sum())
        return np.array(xc), np.array(leak)


def accel(ts, xc):
    return 2 * np.polyfit(ts, xc, 2)[0]


def semicl(H, SIG):
    kq = np.linspace(-3 / SIG, 3 / SIG, 25)
    wq = np.exp(-SIG ** 2 * kq ** 2)
    Fx = np.array([(H(k, 1.0) - H(k, -1.0)) / 2 for k in kq])
    d2 = np.array([(H(k + 1e-3, 0) - 2 * H(k, 0) + H(k - 1e-3, 0)) / 1e-6 for k in kq])
    return -np.sum(wq * Fx * d2) / wq.sum()


def free_band(mu, kx, ky):
    E = energies(Sk((kx, ky, 0)) @ expm(1j * X0(mu)))
    return near(E, np.hypot(CP * np.hypot(kx, ky), 2 * mu), 2).mean()


CASES = [('mu=0.03, rest', 0.03, 0.0), ('mu=0.03, k_y=0.2', 0.03, 0.2), ('mu=0.15, rest', 0.15, 0.0),
         ('massless, k_y=0.2', 0.0, 0.2), ('massless, k_y=0.4', 0.0, 0.4)]
GOLD = (np.sqrt(5) - 1) / 2


def threshold(p, nt, alpha=GOLD, u0=0.0):
    """nested schedule: events only on odd ticks, event k busy where u_k < 2 p(x), one common
    low-discrepancy sequence u_k = frac(k alpha + u0) for all nodes -> the busy set of a layer with
    smaller p is a subset; neighbours differ only for u_k between their thresholds."""
    t = np.arange(nt)
    u = np.mod(((t - 1) // 2) * alpha + u0, 1.0)
    return ((t % 2 == 1)[:, None] & (u[:, None] < 2 * p[None, :])).astype(int)


def run_phase(ch, psi, bw, E, nst, xs):
    """control: free walker, busy layers get the phase exp(+i E dt) of a skipped tick (ideal skip)."""
    st, xc = psi.ravel().copy(), []
    xc.append(ch.com([st], xs))
    for t in range(nst):
        st = ch.T @ st
        st = np.where(np.repeat(bw[t].astype(bool), 24), st * np.exp(1j * E * DT_HALF), st)
        xc.append(ch.com([st], xs))
    return np.array(xc)


def part_B(out):
    N0, g, Nx, SIG, nst = 0.75, 2e-3, 300, 16.0, 160
    print(f"B. x-chain, wait toggle: bit schedules vs continuous 1b (N0 = {N0}, g = {g}, sigma {SIG}, {nst} ticks)")
    t0 = time.time()
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    ts = np.arange(nst + 1) * DT_HALF
    N = N0 * (1 + g * xs)
    p = 1 - N
    rng = np.random.default_rng(1)
    scheds = {'common phase': sigma_delta(p, np.zeros(Nx), nst), 'random phase': sigma_delta(p, rng.random(Nx), nst),
              'nested': threshold(p, nst)}
    c = slice(Nx // 2 - 50, Nx // 2 + 50)
    for sn, b in scheds.items():
        mm = (b[:, 1:] != b[:, :-1])[:, c]
        print(f"   schedule {sn:12s}: busy fraction at x = 0 {b[:, Nx//2].mean():.3f} (p {p[Nx//2]:.3f}); neighbour "
              f"mismatch per tick: t < 40 {mm[:40].mean():.4f}, t >= 120 {mm[120:].mean():.4f}")
    rows, traj = [], {}
    for cname, mu, ky in CASES:
        ch = Chain(mu, ky, Nx)
        psi_c, Eb = scheme_packet('W', mu, ky, N0, ONE, Nx, SIG, xs)
        xc = ch.run_cont('W', N, ONE * np.ones(Nx), [psi_c[:, :24].ravel(), psi_c[:, 24:].ravel()], nst, xs, Eb)
        a_c = accel(ts, xc)
        s_b = semicl(lambda kx, x: N0 * (1 + g * x) * free_band(mu, kx, ky), SIG)
        psi = free_packet(mu, ky, Nx, SIG, xs)
        E = free_band(mu, 0.0, ky)
        a_ph = accel(ts, run_phase(ch, psi, scheds['nested'], E, nst, xs))
        s_ph = semicl(lambda kx, x: free_band(mu, kx, ky) + E * (N0 * (1 + g * x) - 1), SIG)
        traj[f'{cname}|cont'] = xc
        res = [cname, a_c / s_b, a_ph / s_ph]
        line = f"   {cname:18s} continuous 1b / semicl {a_c/s_b:.3f};  ideal skip phase (nested) {a_ph/s_ph:.3f}"
        for sn, bw in scheds.items():
            xb, lk = ch.run_bit(bw, None, psi, nst, xs)
            traj[f'{cname}|{sn}'] = xb
            traj[f'{cname}|{sn}|leak'] = lk
            res += [accel(ts, xb) / s_b, lk[40], lk.max()]
            line += f";  {sn}: {accel(ts, xb)/s_b:+.3f} (leak t=40 {lk[40]:.3f}, max {lk.max():.3f})"
        print(line + f"  [{time.time()-t0:.0f}s]")
        rows.append(res)
    out['B_rows'] = np.array(rows, dtype=object)
    out['B_ts'] = ts
    for k, v in traj.items():
        out['B_traj_' + k] = v

    print("   B2  nested schedule: other sequences u_k, packet width, gradient (mu = 0.03 rest / massless k_y = 0.2)")
    rows = []
    var = [('golden, u0 0', GOLD, 0.0, SIG, g, Nx), ('golden, u0 0.5', GOLD, 0.5, SIG, g, Nx),
           ('sqrt2, u0 0', np.sqrt(2) - 1, 0.0, SIG, g, Nx), ('pi, u0 0.5', np.pi - 3, 0.5, SIG, g, Nx),
           ('sigma 8', GOLD, 0.0, 8.0, g, Nx), ('sigma 32', GOLD, 0.0, 32.0, g, 520), ('g 1e-3', GOLD, 0.0, SIG, 1e-3, Nx)]
    for vn, al, u0, sg, gg, nx in var:
        xs_ = (np.arange(nx) - nx // 2) * LAT_A
        p_ = 1 - N0 * (1 + gg * xs_)
        bw = threshold(p_, nst, al, u0)
        r = [vn]
        for mu, ky in ((0.03, 0.0), (0.0, 0.2)):
            ch = Chain(mu, ky, nx)
            psi = free_packet(mu, ky, nx, sg, xs_)
            E = free_band(mu, 0.0, ky)
            xb, lk = ch.run_bit(bw, None, psi, nst, xs_)
            s_b = semicl(lambda kx, x: N0 * (1 + gg * x) * free_band(mu, kx, ky), sg)
            s_ph = semicl(lambda kx, x: free_band(mu, kx, ky) + E * (N0 * (1 + gg * x) - 1), sg)
            a_ph = accel(ts, run_phase(ch, psi, bw, E, nst, xs_))
            r += [accel(ts, xb) / s_b, a_ph / s_ph, lk.max()]
        rows.append(r)
        print(f"      {vn:15s}: bit {r[1]:.3f} / {r[4]:.3f}, ideal skip phase {r[2]:.3f} / {r[5]:.3f}, "
              f"leakage max {r[3]:.3f} / {r[6]:.3f}  [{time.time()-t0:.0f}s]")
    out['B2_rows'] = np.array(rows, dtype=object)


def main():
    parts = [q.upper() for q in sys.argv[1:]] or ['A', 'B']
    path = data_path('phase_bit_s4_open21.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for q in parts:
        {'A': part_A, 'B': part_B}[q](out)
        np.savez(path, **out)
    print("saved data/phase_bit_s4_open21.npz")


if __name__ == '__main__':
    main()
