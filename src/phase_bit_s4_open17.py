#!/usr/bin/env python3
"""
Paper 3 follow-up, gravitational waves 1: can the budget rule carry a tensor?
==============================================================================

The published model has two scalars (lapse N, spatial metric A = 1/N) and a vector
(drag hop b).  A gravitational wave needs a traceless spatial tensor h_ij.

Idea: the budget rule counts the nodes a hop involves.  A hop along the FCC bond d
involves the bond (x, x + d); if the load of a node is resolved by bond direction,
the stay gate becomes N_h,d.  The FCC lattice has exactly 6 unoriented bond axes,
and the 6 dyads u_d u_d^T are a basis of the symmetric 3x3 matrices, so a scalar per
bond axis IS a symmetric tensor (trace = isotropic A, 5 traceless components).

A0  algebra: rank and conditioning of the bond basis, dual basis.
A1  band level: stay gate N_h,d = N0 (1 + u_d^T S u_d), even under d -> -d (unlike the
    rejected heading-dependent gate of g0i, which was odd).  Fit the cone tensor M of
    the Dirac quartet (E^2 = c'^2 k^T M k): Kramers, Dirac point, ellipsoid residual,
    linear response to the trace, E_g and T_2g parts of S.
A2  GR target M = N0^2 g^-1, g = 1 + 2h (h traceless): S from the measured response,
    random strains, error vs |h|.
A3  bond-local law: is M = sum_d w(N_d) u_d u_d^T with one function w (the hop rate of
    bond d)?  Compare with w = N_d^2 (both nodes free) and w = N_d.
B   source: bond load L_a = dE/d ln N_a (Hellmann-Feynman, like the mass load e_M = mu dE/dmu
    of round 6) for massless and massive band states, compared with the stress tensor in the
    bond basis, (N^2/2) u_a^T T u_a with T^ij = c'^2 p^i p^j / E (lambda(N) compensated).

Run parts: python phase_bit_s4_open17.py [A] [B]
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
from phase_bit_s4_open8 import Sk
from phase_bit_s4_open9 import Xl, LAM0, lam_of
from paths import data_path

print = functools.partial(print, flush=True)
UD = MOVES_PHYS / np.linalg.norm(MOVES_PHYS, axis=1)[:, None]
N0 = 0.75
# 6 unoriented axes: heading d and its reverse
AX = []
for d in range(12):
    r = int(np.argmin(np.linalg.norm(UD + UD[d], axis=1)))
    if r > d:
        AX.append((d, r))
AXU = np.array([UD[d] for d, _ in AX])
SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def dyad_matrix():
    """rows: bond axes, columns: symmetric components (xx, yy, zz, xy, xz, yz) of u u^T."""
    return np.array([[u[i] * u[j] for i, j in SYM] for u in AXU])


def U_stay(k, Nh_d, mu=0.0, lam=LAM0):
    """stay-gate tick R(b_d)[S C (+) -C] at Bloch vector k with one N_h per heading."""
    C = expm(1j * Xl(lam)(mu))
    S = Sk(k)
    T = S @ C
    b = np.repeat(a_of(np.asarray(Nh_d)), 2)
    c, s = np.diag(np.cos(b)), np.diag(np.sin(b))
    Z = np.zeros((24, 24))
    D = np.block([[T, Z], [Z, -C]])
    R = np.block([[c, 1j * s], [1j * s, c]])
    return R @ D


def levels(U):
    return -np.angle(np.linalg.eigvals(U)) / DT_HALF


def cone(Nh_d, kk=0.01, mu=0.0, lam=LAM0):
    E0 = levels(U_stay(np.zeros(3), Nh_d, mu, lam))
    q0 = np.sort(E0[np.argsort(np.abs(E0))[:4]])
    Ec = q0.mean()
    A, b, spl = [], [], np.abs(q0 - Ec).max()
    for n in DIRS:
        E = levels(U_stay(kk * n, Nh_d, mu, lam))
        e = np.sort(E[np.argsort(np.abs(E - Ec))[:4]]) - Ec
        spl = max(spl, abs(e[3] - e[2]), abs(e[1] - e[0]))
        A.append([n[0] ** 2, n[1] ** 2, n[2] ** 2, 2 * n[0] * n[1], 2 * n[0] * n[2], 2 * n[1] * n[2]])
        b.append(((e[3] - e[0]) / 2 / kk) ** 2)
    s, res, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    M = np.array([[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]]) / CP ** 2
    resid = np.sqrt(res[0] / len(b)) / CP ** 2 if len(res) else 0.0
    return M, Ec, spl, resid


def Nh_of(S):
    return N0 * (1 + np.einsum('di,ij,dj->d', UD, S, UD))


def part_A(out):
    print("A. bond-resolved stay gate: does the budget rule carry a tensor?")
    t0 = time.time()
    B = dyad_matrix()
    sv = np.linalg.svd(B, compute_uv=False)
    print(f"   A0  bond axes: {len(AX)}; dyad matrix rank {np.linalg.matrix_rank(B)}, singular values "
          + ", ".join(f"{v:.4f}" for v in sv) + f" (condition {sv[0]/sv[-1]:.3f})")
    out['A0_sv'] = sv

    M0, Ec0, spl0, r0 = cone(np.full(12, N0))
    print(f"   A1  isotropic N_h = {N0}: M = {np.diag(M0).round(6)} (N_h^2 = {N0**2:.6f}), off-diagonal "
          f"{np.abs(M0 - np.diag(np.diag(M0))).max():.1e}, Dirac point {Ec0:+.1e}, Kramers/quartet {spl0:.1e}, "
          f"fit residual {r0:.1e}")
    e = 0.01
    rows = {}
    for name, S in (('trace', np.eye(3) * e), ('E_g', np.diag([e, -e, 0.0])),
                    ('T_2g', np.array([[0, e, 0], [e, 0, 0], [0, 0, 0.0]]))):
        M, Ec, spl, res = cone(Nh_of(S))
        dM = (M - M0) / M0[0, 0]
        rows[name] = (M, Ec, spl, res)
        print(f"       S = {name} ({e}): dM/M0 = {np.array2string(dM.round(5), separator=',').replace(chr(10), '')}, "
              f"Dirac point {Ec:+.1e}, Kramers {spl:.1e}, ellipsoid residual {res:.1e}")
    rT = (rows['trace'][0][0, 0] - M0[0, 0]) / M0[0, 0] / e
    rE = (rows['E_g'][0][0, 0] - rows['E_g'][0][1, 1]) / 2 / M0[0, 0] / e
    rO = rows['T_2g'][0][0, 1] / M0[0, 0] / e
    print(f"       response per unit S: trace {rT:+.4f} (2 = M ~ N_h^2), E_g {rE:+.4f}, T_2g {rO:+.4f}")
    out['A1_resp'] = np.array([rT, rE, rO])

    # A2: GR target M = N0^2 (1 + 2h)^-1 for traceless h, S = -2 h_E/rE - 2 h_T/rO
    rng = np.random.default_rng(17)
    res2 = []
    for amp in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02):
        for trial in range(3):
            hm = rng.normal(size=(3, 3))
            hm = (hm + hm.T) / 2
            hm -= np.trace(hm) / 3 * np.eye(3)
            hm *= amp / np.abs(hm).max()
            hE, hO = np.diag(np.diag(hm)), hm - np.diag(np.diag(hm))
            S = -2 * hE / rE - 2 * hO / rO
            M, Ec, spl, res = cone(Nh_of(S))
            tgt = M0[0, 0] * np.linalg.inv(np.eye(3) + 2 * hm)
            dev = np.abs(M - tgt).max() / M0[0, 0]
            res2.append((amp, dev, Ec, spl, res))
        r = np.array([x for x in res2 if x[0] == amp])
        print(f"   A2  |h| = {amp}: |M - N0^2 g^-1|/N0^2 = {r[:,1].mean():.1e} (relative to 2|h|: "
              f"{r[:,1].mean()/(2*amp):.3f}), Dirac point {np.abs(r[:,2]).max():.1e}, Kramers {r[:,3].max():.1e}")
    out['A2'] = np.array(res2)

    # A3: bond-local law M = sum_d w_d u_d u_d^T (normalised to M0)
    #     for 12 headings sum_d u u^T = 4 * 1, so isotropic w gives M = 4 w; w = N^2 / 4 matches M0 = N^2
    print("   A3  bond-local law: M vs sum over headings w(N_d) u_d u_d^T / 4")
    for name in ('trace', 'E_g', 'T_2g'):
        S = {'trace': np.eye(3) * e, 'E_g': np.diag([e, -e, 0.0]),
             'T_2g': np.array([[0, e, 0], [e, 0, 0], [0, 0, 0.0]])}[name]
        Nd = Nh_of(S)
        M = rows[name][0]
        for lab, w in (('N^2', Nd ** 2), ('N', Nd * N0)):
            P = np.einsum('d,di,dj->ij', w, UD, UD) / 4
            print(f"       {name:5s} w = {lab:3s}: |dM_lattice - dM_law| / |dM_lattice| = "
                  f"{np.abs((M - M0) - (P - N0**2*np.eye(3))).max() / np.abs(M - M0).max():.3f}")
    print(f"   [{time.time()-t0:.0f}s]")


def part_B(out):
    print("B. source: bond load dE/d ln N_a vs the stress tensor (N^2/2) u_a^T T u_a, T^ij = c'^2 p^i p^j / E")
    t0 = time.time()
    lam = lam_of(N0, np.load(data_path('phase_bit_s4_open9.npz'))['A_fit'])
    ax_of = np.zeros(12, int)
    for a, (d, r) in enumerate(AX):
        ax_of[d] = ax_of[r] = a
    rows = []
    for mu in (0.0, 0.03):
        for kmag in (0.02, 0.05, 0.1, 0.2):
            for n in DIRS[[0, 3, 7, 8]]:
                k = kmag * n

                def Eup(Na):
                    E = levels(U_stay(k, Na[ax_of], mu, lam))
                    Eg = np.hypot(N0 * CP * kmag, 2 * mu)
                    return E[np.argsort(np.abs(E - Eg))[:2]].mean()
                E = Eup(np.full(6, N0))
                L = []
                for a in range(6):
                    dN = np.zeros(6)
                    dN[a] = 1e-5 * N0
                    L.append((Eup(np.full(6, N0) + dN) - Eup(np.full(6, N0) - dN)) / 2e-5)
                L = np.array(L)
                T = CP ** 2 * N0 ** 2 * np.outer(k, k) / E       # p = k, velocity factor N0^2 c'^2 in E^2
                Lt = np.array([u @ T @ u for u in AXU]) / 2 * N0 ** 2 / N0 ** 2
                rows.append((mu, kmag, *n, E, np.abs(L - Lt).max() / np.abs(Lt).max(), L.sum() / Lt.sum()))
        r = np.array([x for x in rows if x[0] == mu])
        for kmag in (0.02, 0.05, 0.1, 0.2):
            q = r[r[:, 1] == kmag]
            print(f"   mu = {mu:4}: |k| = {kmag:4}: max |L - u T u| / max|u T u| = {q[:,6].max():.1e}, "
                  f"sum L / sum u T u = {q[:,7].min():.4f} ... {q[:,7].max():.4f}")
    out['B'] = np.array(rows)
    print(f"   [{time.time()-t0:.0f}s]")


def main():
    parts = sys.argv[1:] or ['A', 'B']
    out = {}
    if 'A' in parts:
        part_A(out)
    if 'B' in parts:
        part_B(out)
    np.savez(data_path('phase_bit_s4_open17.npz'), **out)


if __name__ == '__main__':
    main()
