#!/usr/bin/env python3
"""
Bit test 1: does the lapse survive if a node is really busy or free in each tick?
==============================================================================

The wait gate R(a) = cos a + i sin a sigma_x acts on a flag qubit "active / waiting" of
every directed-edge state; R(pi/2) = i sigma_x is a NOT.  The continuous angle
a = arccos(2N - 1) is a coherent "partial NOT".  Here the gate angle is binary per tick,
a in {0, pi/2}, driven by a busy bit b(t) (2O*, eps 0.3, lambda*, wait gate only, Bloch).

A1  'every': NOT in every busy tick.
A2  'toggle': NOT when the bit changes (start and end of a busy interval): the content of
    the node is parked while busy.
A3  random independent bits with busy probability p: the coherent amplitude evolves with
    the averaged one-tick map (1 - p) U(0) + p U(pi/2) (exact for independent ticks and
    nodes); its eigenvalue gives the coherent lapse and the decay per tick; coherence
    between two nodes with independent bits decays by (1 - p)^2 + p^2 per tick.

Physical branch: eigenvector with the largest overlap with the free upper Dirac doublet
(embedded in the active register) among quasi-energies |E| < 0.5 (the buffer sector, phase
(-1)^m per period, folds onto E ~ 0 for some patterns).

    python phase_bit_s4_open20.py      (< 1 min)
"""

import functools
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import DT_HALF
from phase_bit_s4_open import CP
from phase_bit_s4_open8 import Sk, Scheme
from phase_bit_s4_open9 import Xl, LAM0
from paths import data_path

print = functools.partial(print, flush=True)
P2 = np.pi / 2
KK, MU = 0.01, 0.03


def tick(k, mu, aw):
    C = expm(1j * Xl(LAM0)(mu))
    S = Sk(k)
    sc = Scheme('W', S @ C, C, S)
    sc.cw, sc.sw = np.full(24, np.cos(aw)), np.full(24, np.sin(aw))
    cols = []
    for j in range(48):
        e = np.zeros(48, complex)
        e[j] = 1
        cols.append(np.concatenate(sc.tick(list(e.reshape(2, 24)))))
    return np.array(cols).T


def branch(U, steps, k, mu):
    w0, V0 = np.linalg.eig(tick(k, mu, 0.0))
    E0 = -np.angle(w0) / DT_HALF
    guess = np.hypot(CP * np.linalg.norm(k), 2 * mu)
    Q, _ = np.linalg.qr(V0[:, np.argsort(np.abs(E0 - guess))[:2]])
    w, V = np.linalg.eig(U)
    E = -np.angle(w) / (steps * DT_HALF)
    ov = np.linalg.norm(Q.conj().T @ V, axis=0) / np.linalg.norm(V, axis=0)
    ov[np.abs(E) > 0.5] = -1
    ov[E < 0] = -1                        # upper doublet
    j = int(np.argmax(ov))
    return E[j], ov[j], abs(w[j])


def floquet(pat, k, mu, sem):
    U = np.eye(48, dtype=complex)
    prev = pat[-1] if sem == 'toggle' else 0
    for b in pat:
        a = P2 * b if sem == 'every' else P2 * (b != prev)
        prev = b
        U = tick(k, mu, a) @ U
    return U


def main():
    out = {}
    rows = []
    print("A1/A2. deterministic busy patterns (Floquet over one period)")
    pats = ([0, 0, 1], [0, 0, 0, 1], [1], [0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1])
    for sem in ('every', 'toggle'):
        for pat in pats:
            if sem == 'toggle' and sum(pat) == len(pat):
                continue
            m, p = len(pat), sum(pat) / len(pat)
            Eg, og, _ = branch(floquet(pat, np.zeros(3), MU, sem), m, np.zeros(3), MU)
            Ek, ok, _ = branch(floquet(pat, np.array([0, KK, 0]), 0.0, sem), m, np.array([0, KK, 0]), 0.0)
            rows.append((0 if sem == 'every' else 1, m, p, Eg / (2 * MU), Ek / (CP * KK), og, ok))
            print(f"   {sem:6s} {str(pat):26s} p = {p:.3f}: gap/(2 mu) = {Eg/(2*MU):.4f}, v/c' = {Ek/(CP*KK):.4f} "
                  f"(overlap {og:.2f}/{ok:.2f});  1 - p = {1-p:.4f}")
    out['A12'] = np.array(rows)
    print("A3. random independent bits: averaged one-tick map (coherent part)")
    rows = []
    for p in (0.02, 0.05, 0.1, 0.25, 0.5):
        Ub = lambda k, mu: (1 - p) * tick(k, mu, 0.0) + p * tick(k, mu, P2)
        Eg, _, lg = branch(Ub(np.zeros(3), MU), 1, np.zeros(3), MU)
        Ek, _, lk = branch(Ub(np.array([0, KK, 0]), 0.0), 1, np.array([0, KK, 0]), 0.0)
        a = np.arctan2(p, 1 - p)
        rows.append((p, Eg / (2 * MU), Ek / (CP * KK), lk, (1 + np.cos(a)) / 2, (1 - p) ** 2 + p ** 2))
        print(f"   p = {p:4}: coherent gap/(2 mu) = {Eg/(2*MU):.4f}, v/c' = {Ek/(CP*KK):.4f} (continuous gate with "
              f"tan a = p/(1-p): N = {(1+np.cos(a))/2:.4f}; 1 - p = {1-p:.2f}); |lambda| per tick {lk:.4f}; "
              f"node-node coherence per tick {(1-p)**2+p**2:.4f}")
    out['A3'] = np.array(rows)
    np.savez(data_path('phase_bit_s4_open20.npz'), **out)


if __name__ == '__main__':
    main()
