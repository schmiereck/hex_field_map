#!/usr/bin/env python3
"""
Paper 3, Experiment S1 — Spin vortices from phase bits (integer automaton)
==========================================================================

A walker on the 2D triangular lattice carries
    d in Z6   heading (0, 60, ..., 300 deg)
    p in Z4   phase bit, +1, +i, -1, -i
    T in N    number of elementary 60-deg turns taken so far
and a step turns by n in {-2,...,+2} units of 60 deg (no reversal), then hops:
    d -> d + n (mod 6),   p -> p + n (mod 4),   T -> T + |n|.
The state is a table of NON-NEGATIVE INTEGER path counts N[x, y, d, p, T].
No phase, no epsilon, no floating point enters the dynamics.

Only at readout are numbers attached:
    A_ab(x, y, d; eps) = sum_{p,T} N[x,y,d,p,T] * eps^T * i^(a*p + b*T)
      a = 1 : the phase bit is read as a phase      (spin readout)
      a = 0 : the phase bit is ignored               (boson readout)
      b = 1 : every elementary turn carries i        (Paper-1 rule "turn = i eps")

Facts tested here
-----------------
1.  Exactness: the integer table reproduces the floating-point walk with coin
    W[d',d] = eps^|n| * i^(a n + b |n|) to machine precision.
2.  Double cover: only pairs with d = p (mod 2) are reachable -> 12 of 24
    labels = Z12 = net turning mod 12, the double cover of the 6 headings.
3.  Spinor sign: a closed walk (back to the same directed edge) has net turning
    6w, so p = 6w mod 4 in {0, 2}: the phase bit is -1 exactly for odd winding.
    Checked against an independent enumeration of all closed walks.
4.  (a=1, b=1) is the graded turning model at alpha = 3/2, gauge-equivalent to
    alpha = 1/2: identical spectrum at every k.
5.  Vortex modes: at k = 0 the integer transfer is circulant on Z12, so its
    eigenmodes are e^{2 pi i j N / 12}.  A full 360-deg rotation (N -> N+6)
    multiplies them by (-1)^j.  Odd j = half-quantum vortices = spin 1/2,
    and they form Kramers doublets (j <-> 12 - j); even j = integer vortices.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_hex_turning import (MOVES_IDX, MOVES_PHYS, N_D, DT_HALF, _shift,
                                 coin, enumerate_loops)
from paths import data_path

TURNS = (-2, -1, 0, 1, 2)


# ─── integer automaton ───────────────────────────────────────────────────────

def make_state(t_max):
    """Integer table N[ix, iy, d, p, T] with a single walker at the centre."""
    nx, ny = 4 * t_max + 1, 2 * t_max + 1
    N = np.zeros((nx, ny, N_D, 4, 2 * t_max + 1), dtype=np.int64)
    N[nx // 2, ny // 2, 0, 0, 0] = 1
    return N


def int_step(N):
    """One step of the phase-bit automaton.  Pure integer additions."""
    new = np.zeros_like(N)
    for d in range(N_D):
        src = N[:, :, d]                                  # (nx, ny, 4, T)
        for n in TURNS:
            dn = (d + n) % N_D
            moved = np.roll(src, n, axis=2)               # p -> p + n
            if abs(n):
                moved = np.concatenate([np.zeros_like(moved[..., :abs(n)]),
                                        moved[..., :-abs(n)]], axis=-1)
            new[:, :, dn] += _shift(moved, MOVES_IDX[dn, 0], MOVES_IDX[dn, 1])
    return new


def readout(N, eps, a, b):
    """A[x, y, d] = sum_{p,T} N * eps^T * i^(a p + b T)."""
    nT = N.shape[-1]
    wp = 1j ** (a * np.arange(4))
    wT = (eps ** np.arange(nT)) * 1j ** (b * np.arange(nT))
    return np.einsum('xydpt,p,t->xyd', N.astype(float), wp, wT)


# ─── floating-point reference ────────────────────────────────────────────────

def phase_bit_coin(eps, a, b):
    """W[d', d] = eps^|n| * i^(a n + b |n|), n = signed turn from d to d'."""
    W = np.zeros((N_D, N_D), dtype=complex)
    for d in range(N_D):
        for n in TURNS:
            W[(d + n) % N_D, d] += eps ** abs(n) * 1j ** (a * n + b * abs(n))
    return W


def float_step(psi, W):
    w = psi @ W.T
    new = np.zeros_like(psi)
    for d in range(N_D):
        new[:, :, d] = _shift(w[:, :, d], MOVES_IDX[d, 0], MOVES_IDX[d, 1])
    return new


# ─── double-cover (Z12) spectrum at k = 0 ────────────────────────────────────

def z12_transfer(eps, b):
    """Integer-label transfer on Z12 at k = 0 (turn weights eps^|n| i^(b|n|))."""
    M = np.zeros((12, 12), dtype=complex)
    for N in range(12):
        for n in TURNS:
            M[(N + n) % 12, N] += eps ** abs(n) * 1j ** (b * abs(n))
    return M


def z12_eigen_analytic(eps, b):
    """lambda_j = sum_n eps^|n| i^(b|n|) e^{-2 pi i j n / 12}, j = 0..11."""
    j = np.arange(12)
    lam = np.zeros(12, dtype=complex)
    for n in TURNS:
        lam += eps ** abs(n) * 1j ** (b * abs(n)) * np.exp(-2j * np.pi * j * n / 12)
    return lam


def sector_of_mode(v):
    """Eigenvalue of the 360-deg rotation N -> N+6 on a Z12 vector: +1 or -1."""
    rot = np.roll(v, 6)
    return np.real(np.vdot(v, rot) / np.vdot(v, v))


def readout_projection(a):
    """6x12 map from Z12 labels to headings: A_d = sum_{N = d mod 6} i^(a N)."""
    P = np.zeros((N_D, 12), dtype=complex)
    for N in range(12):
        P[N % 6, N] = 1j ** (a * (N % 4))
    return P


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    out = {}
    T_MAX = 24
    N = make_state(T_MAX)
    nx, ny = N.shape[:2]
    cx, cy = nx // 2, ny // 2

    configs = [(1, 0), (0, 0), (1, 1), (0, 1)]
    eps_test = (0.1, 0.3, 0.7)
    psis = {(a, b, e): None for a, b in configs for e in eps_test}
    for key in psis:
        psi = np.zeros((nx, ny, N_D), dtype=complex)
        psi[cx, cy, 0] = 1.0
        psis[key] = psi
    coins = {(a, b, e): phase_bit_coin(e, a, b) for a, b, e in psis}

    max_rel = 0.0
    loops_p2, loops_p0, bits = [], [], []
    for t in range(1, T_MAX + 1):
        N = int_step(N)
        for key in psis:
            psis[key] = float_step(psis[key], coins[key])
            if t % 4 == 0 or t == T_MAX:
                A = readout(N, key[2], key[0], key[1])
                ref = psis[key]
                rel = np.abs(A - ref).max() / np.abs(ref).max()
                max_rel = max(max_rel, rel)
        # closed walks: back at the origin with heading 0
        loops_p0.append(int(N[cx, cy, 0, 0].sum()))
        loops_p2.append(int(N[cx, cy, 0, 2].sum()))
        bits.append(float(np.log2(max(N.max(), 1))))
        if t == T_MAX:
            occ = (N.sum(axis=(0, 1, 4)) > 0)
    print(f"1. integer vs float: max relative deviation {max_rel:.2e} "
          f"over t<={T_MAX}, eps={eps_test}, (a,b)={configs}  [{time.time()-t0:.0f}s]")
    out['max_rel'] = max_rel

    # 2. reachable (d, p) labels
    pairs = [(d, p) for d in range(6) for p in range(4) if occ[d, p]]
    ok_parity = all((d - p) % 2 == 0 for d, p in pairs)
    print(f"2. reachable (d,p) labels: {len(pairs)} of 24, all with d = p mod 2: {ok_parity}")
    out['occ'] = occ.astype(int)

    # 3. closed walks vs independent enumeration
    L = 9
    counts, bad = enumerate_loops(L_max=L)
    odd = [sum(c for w, c in counts[l].items() if w % 2) for l in range(1, L + 1)]
    even = [sum(c for w, c in counts[l].items() if w % 2 == 0) for l in range(1, L + 1)]
    match = (odd == loops_p2[:L]) and (even == loops_p0[:L])
    print(f"3. closed walks up to L={L}: p=2 (sign -1) counts {loops_p2[:L]}")
    print(f"   odd-winding enumeration      {odd}")
    print(f"   p=0 counts {loops_p0[:L]} vs even winding {even} -> exact match: {match}, "
          f"Whitney violations {bad}")
    out['loops_p0'], out['loops_p2'] = np.array(loops_p0), np.array(loops_p2)
    out['enum_odd'], out['enum_even'] = np.array(odd), np.array(even)
    out['bits'] = np.array(bits)
    print(f"   integer size: max count needs {bits[-1]:.1f} bits at t={T_MAX} "
          f"({bits[-1]/T_MAX:.3f} bits/step; log2 5 = {np.log2(5):.3f})")

    # 4. gauge equivalence with the graded turning model at alpha = 1/2
    dev = 0.0
    rng = np.random.default_rng(1)
    for _ in range(20):
        kx, ky = rng.uniform(-3, 3, 2)
        e = rng.uniform(0.05, 0.8)
        ph = np.exp(-1j * (kx * MOVES_PHYS[:, 0] + ky * MOVES_PHYS[:, 1]))
        ev1 = np.linalg.eigvals(ph[:, None] * phase_bit_coin(e, 1, 1))
        ev2 = np.linalg.eigvals(ph[:, None] * coin(e, 0.5, 'graded'))
        d12 = np.abs(ev1[:, None] - ev2[None, :])          # match as sets
        dev = max(dev, d12.min(axis=1).max(), d12.min(axis=0).max())
    print(f"4. spectrum (a=1,b=1) vs graded turning model alpha=1/2: max diff {dev:.1e}")
    out['gauge_dev'] = dev

    # 5. vortex modes on Z12 at k = 0
    for b in (0, 1):
        e = 0.3
        M = z12_transfer(e, b)
        lam_num = np.linalg.eigvals(M)
        lam_ana = z12_eigen_analytic(e, b)
        err = max(np.abs(lam_num[:, None] - lam_ana[None, :]).min(axis=1))
        j = np.arange(12)
        modes = np.exp(2j * np.pi * np.outer(np.arange(12), j) / 12)    # columns
        sectors = np.array([sector_of_mode(modes[:, k]) for k in j])
        # which modes survive which readout
        surv = {a: np.linalg.norm(readout_projection(a) @ modes, axis=0) for a in (0, 1)}
        # 6x6 readout spectra
        ev_spin = np.linalg.eigvals(phase_bit_coin(e, 1, b))
        ev_bos = np.linalg.eigvals(phase_bit_coin(e, 0, b))
        in_odd = max(np.abs(ev_spin[:, None] - lam_ana[j % 2 == 1][None]).min(axis=1))
        in_even = max(np.abs(ev_bos[:, None] - lam_ana[j % 2 == 0][None]).min(axis=1))
        print(f"5. b={b}, eps={e}: Z12 eigenvalues analytic vs numeric {err:.1e}; "
              f"360-deg sign of mode j: {np.round(sectors).astype(int).tolist()}")
        print(f"   spin readout (a=1) keeps j = {j[surv[1] > 1e-9].tolist()}, "
              f"boson readout (a=0) keeps j = {j[surv[0] > 1e-9].tolist()}")
        print(f"   6x6 spin spectrum = odd-j levels: {in_odd:.1e};  "
              f"boson spectrum = even-j levels: {in_even:.1e}")
        for name, sel in (("odd j, spin ", j % 2 == 1), ("even j, boson", j % 2 == 0)):
            print(f"   lambda ({name}): " + ", ".join(
                f"j={jj}:{np.round(lam_ana[jj], 6)}" for jj in j[sel]))
        out[f'lam_b{b}'] = lam_ana
    out['eps_vortex'] = 0.3

    # readout amplitudes at t = T_MAX for the figure
    for a in (0, 1):
        A = readout(N, 0.3, a, 1)
        out[f'rho_a{a}'] = (np.abs(A) ** 2).sum(axis=-1)
    out['T_MAX'] = T_MAX

    np.savez(data_path('phase_bit_automaton.npz'), **out)
    print(f"saved data/phase_bit_automaton.npz  [{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()
