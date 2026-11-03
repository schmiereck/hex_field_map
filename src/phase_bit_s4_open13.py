#!/usr/bin/env python3
"""
Paper 3: what an observer inside the lattice can measure, Newton and perihelion, Lense-Thirring
==============================================================================

A.  "Michelson-Morley in the lattice" (2O*, eps = 0.3, mu = 0, no field).
    The observer defines lengths with long-wave light (radar).  Measurable is only
    whether two probes agree when the apparatus is turned:
        Q(n) = v(k_hi, n) / v(k_lo, n)       (radar times of two light energies)
    over 2000 directions n (Fibonacci sphere), group velocity of the upper Weyl
    doublet by Hellmann-Feynman (exact), plus the walk-off angle between the ray
    (group velocity) and the wave vector.

B.  Newton and perihelion.
    B1  full 3D walker (W and H2 gates, lambda(N) coins): force on a massive packet at rest (mu = 0.15,
        m = 0.3) at distance r = 6 ... 26 from a point mass U = GM/sqrt(r^2 + rc^2), F = d<k_x>/dt, against
        the density- and k-averaged local force of H = N sqrt(m^2 + (N^n c' k)^2) (n = 1 H2, 0 W).
    B2  classical orbits of H = N(r) sqrt(m^2 + c'^2 p^2/A(r)^2) (static isotropic metric): perihelion advance
        vs the PPN value (2 + 2 gamma - beta)/3 * 6 pi GM/p for H2 (gamma = beta = 1), W (gamma = 0) and a
        linear lapse (beta = 1/2).  A wave-packet orbit is out of reach (L = m r v << hbar, printed).

C.  Lense-Thirring.
    C1  gravitomagnetic field of a rotating ball from the 3D field equation (dipole 2 c'^2 (kappa/4pi) J x r/r^3,
        i.e. the GR coefficient from the factor 4) and the nodal precession of a classical orbit.
    C2  frame dragging in the 3D walker: rigid drag field b = Omega z x r realised by a LOCAL drag hop
        (brick wall of real bond rotations along the 6 FCC directions, the same for all internal states);
        a packet at rest is carried around with Omega.  Gyroscope: the scalar drag couples to L only, the spin
        stays fixed; adding the internal rotation exp(-i dt (1/2) curl b . J) with J_a = (2i/pi) log R90_a
        (the lattice's own 90-degree rotation, g = 1 on every spin-1/2 doublet) lets the spin turn with Omega.

Run parts: python phase_bit_s4_open13.py [A] [B1] [B2] [C1] [C2]
"""

import functools
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import MOVES_PHYS, DT_HALF, LAT_A
from phase_bit_s4_open import CP
from phase_bit_s4_open9 import Xl, LAM0
from phase_bit_s4_open12 import Walker3D
from phase_bit_mass_s4 import bands_C
from paths import data_path

print = functools.partial(print, flush=True)
W24 = np.repeat(MOVES_PHYS, 2, axis=0) / DT_HALF          # velocity of each internal state


def fib_dirs(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    a = np.pi * (1 + np.sqrt(5)) * i
    return np.stack([np.sqrt(1 - z ** 2) * np.cos(a), np.sqrt(1 - z ** 2) * np.sin(a), z], 1)


def cone_velocity(C, K):
    """group velocity (3-vector) of the upper Weyl doublet (2 states nearest +c'|k|) at the k points K."""
    ph = np.exp(-1j * K @ MOVES_PHYS.T)
    U = np.repeat(ph, 2, axis=1)[:, :, None] * C[None]
    lam, V = np.linalg.eig(U)
    E = -np.angle(lam) / DT_HALF
    kk = np.linalg.norm(K, axis=1)
    idx = np.argsort(np.abs(E - CP * kk[:, None]), axis=1)[:, :2]
    out = np.zeros((len(K), 3))
    for j in range(2):
        v = np.take_along_axis(V, idx[:, None, j:j + 1].repeat(24, 1), axis=2)[:, :, 0]
        p = np.abs(v) ** 2 / (np.abs(v) ** 2).sum(1, keepdims=True)
        out += p @ W24 / 2
    return out


# ═════════════════════════════════════════════════════════════════════════════
def part_A(out):
    print("A. Michelson-Morley in the lattice: radar with two light energies, apparatus turned over 2000 directions")
    t0 = time.time()
    C = expm(1j * Xl(LAM0)(0.0))
    n = fib_dirs(2000)
    ks = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
    V = {}
    for k in ks:
        v = cone_velocity(C, k * n)
        V[k] = v
        vpar = np.sum(v * n, 1)
        walk = np.degrees(np.arccos(np.clip(vpar / np.linalg.norm(v, axis=1), -1, 1)))
        out[f'A_v_{k}'] = v
        print(f"   k = {k:4.2f}: |v|/c' along k: mean {vpar.mean()/CP:.6f}, spread (max-min)/mean {np.ptp(vpar)/vpar.mean():.2e}; "
              f"walk-off ray vs wave vector: max {walk.max():.2e} deg  [{time.time()-t0:.0f}s]")
    rows = []
    klo = 0.02
    vlo = np.sum(V[klo] * n, 1)
    for k in ks[1:]:
        Q = np.sum(V[k] * n, 1) / vlo
        dQ = np.ptp(Q) / Q.mean()
        # directions of extremes
        imax, imin = np.argmax(Q), np.argmin(Q)
        rows.append((k, dQ, Q.mean()))
        print(f"   Q = v(k = {k})/v(k = {klo}): mean {Q.mean():.6f}, anisotropy (max-min)/mean = {dQ:.2e}  "
              f"(fastest along {np.round(np.abs(n[imax]), 2)}, slowest along {np.round(np.abs(n[imin]), 2)})")
    r = np.array(rows)
    out['A_rows'] = r
    spread = np.array([np.ptp(np.sum(V[k] * n, 1)) / np.mean(np.sum(V[k] * n, 1)) for k in ks])
    p = np.polyfit(np.log(np.array(ks[:3])), np.log(spread[:3]), 1)
    print(f"   direction spread of the light speed ~ k^{p[0]:.2f} for k = 0.02 ... 0.1 (cubic k^3 Y_4 term in E(k)); "
          f"extremes swap between k = 0.2 and 0.4 (higher orders)")
    out['A_spread'] = np.array([ks, spread])
    mean_lo = np.array([np.mean(np.sum(V[k] * n, 1)) / CP for k in ks])
    print(f"   energy dependence (isotropic, time of flight): 1 - <v>/c' = " + ", ".join(f"{1-m:.2e}" for m in mean_lo)
          + f" at k = {ks}")
    # express as an anisotropic dispersion v = c'(1 - xi(n) (k a)^2): xi range
    k = 0.1
    xi = (1 - np.sum(V[k] * n, 1) / CP) / (k * LAT_A) ** 2
    print(f"   v = c'(1 - xi(n) (k a)^2): xi between {xi.min():.4f} and {xi.max():.4f} (a = {LAT_A:.3f}); "
          f"isotropic part {xi.mean():.4f}, anisotropic part +-{np.ptp(xi)/2:.4f}")
    out['A_xi'] = np.array([xi.min(), xi.max(), xi.mean()])


# ═════════════════════════════════════════════════════════════════════════════
def packet3d_m(shape, k0, sig, mu, member):
    """upper doublet (E ~ +sqrt(m^2 + c'^2 k^2), m = 2 mu) packet, Gaussian envelope sig, centred at x = 0."""
    C = expm(1j * Xl(LAM0)(mu))
    m = 2 * mu
    ks = [2 * np.pi * np.fft.fftfreq(n, d=LAT_A) for n in shape]
    KX, KY, KZ = np.meshgrid(*ks, indexing='ij')
    g = np.exp(-sig ** 2 * ((KX - k0[0]) ** 2 + (KY - k0[1]) ** 2 + (KZ - k0[2]) ** 2) / 2)
    target = lambda kk: np.hypot(m, CP * kk)
    E0, V0 = bands_C(np.asarray(k0, float), C)
    Q0, _ = np.linalg.qr(V0[:, np.argsort(np.abs(E0 - target(np.linalg.norm(k0))))[:2]])
    uref = Q0[:, member]
    psik = np.zeros((*shape, 24), np.complex64)
    for i in zip(*np.where(g > 1e-6)):
        k = np.array([KX[i], KY[i], KZ[i]])
        E, V = bands_C(k, C)
        Q, _ = np.linalg.qr(V[:, np.argsort(np.abs(E - target(np.linalg.norm(k))))[:2]])
        u = Q @ (Q.conj().T @ uref)
        psik[i] = g[i] * u / max(np.linalg.norm(u), 1e-12)
    psi = np.fft.ifftn(psik, axes=(0, 1, 2))
    psi = np.roll(psi, tuple(n // 2 for n in shape), axis=(0, 1, 2))
    I = np.indices(shape).sum(0) - sum(n // 2 for n in shape)
    psi *= (I % 2 == 0)[..., None]
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
    return psi.reshape(-1, 24).astype(np.complex64)


def part_B1(out):
    print("B1. Newton in the 3D walker: force on a massive packet at rest at distance r from a point mass")
    t0 = time.time()
    shape = (64, 64, 64)
    mu, sig, GM, rc, nst = 0.15, 5.0, 0.25, 2.0, 64
    m = 2 * mu
    xs = [(np.arange(n) - n // 2) * LAT_A for n in shape]
    X, Y, Z = np.meshgrid(*xs, indexing='ij')
    ks = [2 * np.pi * np.fft.fftfreq(n, d=LAT_A) for n in shape]
    KX, KY, KZ = np.meshgrid(*ks, indexing='ij')
    near = KX ** 2 + KY ** 2 + KZ ** 2 < 2.0 ** 2
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    S = int(np.prod(shape))
    print(f"   box {shape}, {S*24*8/1e9:.2f} GB per register; packet at the centre, mass at (-r, 0, 0) (may lie outside the box)")
    rows = []
    for r in (6.0, 9.0, 13.0, 18.0, 26.0):
        R = np.sqrt((X + r) ** 2 + Y ** 2 + Z ** 2)
        U = GM / np.sqrt(R ** 2 + rc ** 2)
        Nf = np.exp(-U)
        dNdx = Nf * GM * (X + r) / (R ** 2 + rc ** 2) ** 1.5      # dN/dx = -N dU/dx
        res = [r]
        for kind, nexp in (('W', 0), ('H2', 1)):
            wk = Walker3D(shape, 'W' if kind == 'W' else 'WH', Nf, mu=mu)
            nreg = 2 if kind == 'W' else 4
            Fs, Ps = [], []
            for member in (0, 1):
                psi = packet3d_m(shape, (0.0, 0.0, 0.0), sig, mu, member)
                st = [psi] + [np.zeros_like(psi) for _ in range(nreg - 1)]
                ph = np.exp(1j * m * Nf[shape[0] // 2, shape[1] // 2, shape[2] // 2] * DT_HALF)
                for _ in range(3):
                    st2 = wk.tick(st)
                    st = [(a + ph * c) / 2 for a, c in zip(st, st2)]
                nrm = np.sqrt(sum(np.vdot(a, a).real for a in st))
                st = [a / nrm for a in st]
                kx, ts, pf = [], [], []
                for it in range(nst + 1):
                    if it % 2 == 0:
                        spec = sum((np.abs(np.fft.fftn(a.reshape(*shape, 24), axes=(0, 1, 2))) ** 2).sum(-1) for a in st) * near
                        rho = sum((np.abs(a) ** 2).sum(1) for a in st).reshape(shape)
                        kx.append((spec * KX).sum() / spec.sum())
                        ts.append(it * DT_HALF)
                        # local force -<dH/dx> of H = N sqrt(m^2 + (N^n c' k)^2), k average with the mean N
                        k2m = (spec * K2).sum() / spec.sum()
                        Nc = (rho * Nf).sum() / rho.sum()
                        Eloc = np.sqrt(m ** 2 + (Nc ** nexp * CP) ** 2 * k2m)
                        fac = Eloc + nexp * Nc ** (2 * nexp) * CP ** 2 * k2m / Eloc
                        pf.append(-(rho * dNdx).sum() / rho.sum() * fac)
                    if it < nst:
                        st = wk.tick(st)
                ts = np.array(ts)
                w_ = 2 * m * Nc
                M = np.stack([np.ones_like(ts), ts, np.cos(w_ * ts), np.sin(w_ * ts)], 1)
                Fs.append(np.linalg.lstsq(M, np.array(kx), rcond=None)[0][1])
                Ps.append(np.mean(pf))
            F, P = np.mean(Fs), np.mean(Ps)
            res += [F, P]
            print(f"   r = {r:5.1f}, {kind:2s}: F = {F:+.4e}, local force {P:+.4e} (ratio {F/P:.3f}); "
                  f"Newton -m GM r/(r^2+rc^2)^1.5 = {-m*GM*r/(r*r+rc*rc)**1.5:+.4e}  [{time.time()-t0:.0f}s]")
        rows.append(res)
    r = np.array(rows)
    out['B1_rows'] = r
    soft = np.polyfit(np.log(r[2:, 0]), np.log(r[2:, 0] / (r[2:, 0] ** 2 + 4) ** 1.5), 1)[0]
    for j, kind in ((1, 'W'), (3, 'H2')):
        sl = np.polyfit(np.log(r[2:, 0]), np.log(np.abs(r[2:, j])), 1)[0]
        print(f"   {kind}: slope d ln|F| / d ln r for r >= 13: {sl:.3f} (point mass -2, softened {soft:.3f})")
    print("   H2 / W force: " + ", ".join(f"{a/b:.3f}" for a, b in zip(r[:, 3], r[:, 1]))
          + "  (Newton limit: 1; kinetic weighting of the packet's k spread differs slightly)")


def part_B2(out):
    print("B2. classical orbits of H = N sqrt(m^2 + c'^2 p^2/A^2): perihelion advance")
    from scipy.integrate import solve_ivp
    m, c, GM = 1.0, CP, 1.0
    rows = []
    variants = {'H2 (N = e^-U, A = 1/N)': (lambda U: np.exp(-U), lambda U: np.exp(U), 1, 1),
                'W (N = e^-U, A = 1)': (lambda U: np.exp(-U), lambda U: 1.0 + 0 * U, 0, 1),
                'linear lapse (N = 1 - U, A = 1/N)': (lambda U: 1 - U, lambda U: 1 / (1 - U), 1, 0.5)}
    for name, (Nfun, Afun, gam, bet) in variants.items():
        for a_semi in (400.0, 1600.0):
            e = 0.3
            rp = a_semi * (1 - e)
            v_n = c * np.sqrt(GM / a_semi * (1 + e) / (1 - e))
            p = m * v_n / c ** 2
            y0 = [rp, 0.0, 0.0, 0.0, p, 0.0]

            def rhs(t, y):
                x, pv = y[:3], y[3:]
                r = np.linalg.norm(x)
                U = GM / r
                p2 = pv @ pv
                N, A = Nfun(U), Afun(U)
                E = np.sqrt(m ** 2 + c ** 2 * p2 / A ** 2)
                dx = N * c ** 2 * pv / (A ** 2 * E)
                H = lambda UU: Nfun(UU) * np.sqrt(m ** 2 + c ** 2 * p2 / Afun(UU) ** 2)
                h = 1e-6 * U
                dHdU = (H(U + h) - H(U - h)) / (2 * h)
                return np.concatenate([dx, dHdU * GM * x / r ** 3])
            T = 2 * np.pi * np.sqrt(a_semi ** 3 / GM) / c
            sol = solve_ivp(rhs, (0, 6.3 * T), y0, method='DOP853', rtol=1e-12, atol=1e-12 * a_semi,
                            dense_output=True, max_step=T / 300)
            tt = np.linspace(0, 6.3 * T, 600000)
            Y = sol.sol(tt)
            rr = np.hypot(Y[0], Y[1])
            ph = np.unwrap(np.arctan2(Y[1], Y[0]))
            ip = [i for i in range(1, len(rr) - 1) if rr[i] < rr[i - 1] and rr[i] <= rr[i + 1]]
            angs = []
            for i in ip:
                yy = rr[i - 1:i + 2]
                d = (yy[0] - yy[2]) / (2 * (yy[0] - 2 * yy[1] + yy[2]))
                angs.append(ph[i] + d * (ph[i + 1] - ph[i - 1]) / 2)
            adv = np.diff(angs) - 2 * np.pi
            rmin, rmax = rr.min(), rr.max()
            psemi = 2 * rmin * rmax / (rmin + rmax)
            gr = 6 * np.pi * GM / psemi
            ppn = (2 + 2 * gam - bet) / 3 * gr
            rows.append((a_semi, psemi, adv.mean(), ppn, gr))
            print(f"   {name:34s} a = {a_semi:6.0f}: advance per orbit {adv.mean():.6e} rad ({len(adv)} orbits, spread {np.ptp(adv):.1e}); "
                  f"PPN (2+2g-b)/3 x 6 pi GM/p = {ppn:.6e} (ratio {adv.mean()/ppn:.4f})")
    out['B2_rows'] = np.array(rows)
    for r_, GMl in ((10.0, 0.25), (30.0, 0.5)):
        v = CP * np.sqrt(GMl / r_)
        print(f"   wave-packet orbit, m = 0.3, r = {r_}, GM = {GMl}: v = {v:.3f}, L = m r v / c'^2 = {0.3*r_*v/CP**2:.2f} hbar "
              f"(classical needs L >> 1); spreading time m sigma^2/c'^2 for sigma = r/3: {0.3*(r_/3)**2/CP**2:.0f} vs period "
              f"{2*np.pi*r_/v:.0f}")


# ═════════════════════════════════════════════════════════════════════════════
SIG3 = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1.0 + 0j, -1.0])]


def R90_int(axis):
    """internal action of the right-handed +90 deg rotation about a cube axis: heading permutation x spin rotation."""
    from quantum_fcc_3d import FCC_STEPS
    e = np.zeros(3); e[axis] = 1
    K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
    M = np.rint(np.eye(3) + K + K @ K).astype(int)          # Rodrigues, angle 90 deg
    P = np.zeros((12, 12))
    for d in range(12):
        t = M @ FCC_STEPS[d]
        P[[k for k in range(12) if (FCC_STEPS[k] == t).all()][0], d] = 1
    return np.kron(P, expm(-1j * np.pi / 4 * SIG3[axis]))


def J_int():
    """internal rotation generators J_a = (2i/pi) log R90_a (24 x 24, Hermitian)."""
    from scipy.linalg import logm
    J = []
    for a in range(3):
        j = 1j * 2 / np.pi * logm(R90_int(a))
        J.append((j + j.conj().T) / 2)
    return J


def drag3d(psi, theta_fn):
    """local drag hop: brick wall of bond rotations along the 6 FCC directions (both parities).
    psi: (Nx, Ny, Nz, 24); theta_fn(s) -> bond angle array for bonds (x, x + s) (indexed at x)."""
    from quantum_fcc_3d import FCC_STEPS
    sh = psi.shape[:3]
    I = np.indices(sh)
    dirs = [s for s in FCC_STEPS if tuple(s) > tuple(-s)]          # one of each +-s pair (6)
    for s in dirs:
        th = theta_fn(s)
        m = (I[0] * s[0] + I[1] * s[1] + I[2] * s[2]) // 2
        for par in (0, 1):
            M = ((m % 2) == par)[..., None]
            c, sn = np.cos(th)[..., None] * M, np.sin(th)[..., None] * M
            B = np.roll(psi, tuple(-s), axis=(0, 1, 2))            # psi(x + s) at x
            newA = np.where(M, c * psi - sn * B, psi)
            newB_at_x = c * B + sn * psi                              # new value for site x + s, stored at x
            moved = np.roll(np.where(M, newB_at_x, 0), tuple(s), axis=(0, 1, 2))
            Mb = np.roll(M, tuple(s), axis=(0, 1, 2))
            psi = np.where(Mb, moved, newA)
    return psi


def part_C1(out):
    print("C1. gravitomagnetic field of a rotating ball (3D field equation) and Lense-Thirring nodal precession")
    n, h = 96, LAT_A
    x = (np.arange(n) - n // 2) * h
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Rb, Om = 5.0, 0.02
    rho = (R < Rb).astype(float)
    rho /= rho.sum() * h ** 3
    px, py = -Om * Y * rho, Om * X * rho                     # momentum density of rigid rotation about z (unit mass)
    Jz = np.sum(X * py - Y * px) * h ** 3
    k = 2 * np.pi * np.fft.fftfreq(n, d=h)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    lapk = -(4 / h ** 2) * (np.sin(KX * h / 2) ** 2 + np.sin(KY * h / 2) ** 2 + np.sin(KZ * h / 2) ** 2)
    lapk[0, 0, 0] = 1
    kap = 1.0
    kapb = -4 * CP ** 2 * kap
    solve = lambda f: np.real(np.fft.ifftn(np.where(lapk == 1, 0, kapb * np.fft.fftn(f) / lapk)))
    bx, by = solve(px), solve(py)
    # far field of a rotating body: b = 2 c'^2 (kappa/4pi) (J x r)/r^3   (from the factor 4)
    rows = []
    for r_ in (8.0, 10.0, 12.0, 15.0):
        i = n // 2 + int(round(r_ / h))
        bpred = 2 * CP ** 2 * kap / (4 * np.pi) * Jz * x[i] / x[i] ** 3     # (J z x r)_y at (r, 0, 0) = J r
        rows.append((x[i], by[i, n // 2, n // 2], bpred))
        print(f"   b_y at (r, 0, 0), r = {x[i]:.2f}: {by[i, n//2, n//2]:+.4e} vs 2 c'^2 (kappa/4pi) J / r^2 = {bpred:+.4e} "
              f"(ratio {by[i, n//2, n//2]/bpred:.3f})")
    out['C1_field'] = np.array(rows)
    # classical orbit with H = N sqrt(m^2 + c'^2 p^2/A^2) + b.p, b = B (J_hat x x)/r^3: node of an inclined orbit
    from scipy.integrate import solve_ivp
    m, c, GM, B = 1.0, CP, 1.0, 2.0 * 3.0                     # B = 2 G J / c'^0 in these units (strong to see it)
    a_s, e, inc = 300.0, 0.2, np.radians(60)
    rp = a_s * (1 - e)
    v = c * np.sqrt(GM / a_s * (1 + e) / (1 - e))
    x0 = np.array([rp, 0, 0])
    v0 = v * np.array([0, np.cos(inc), np.sin(inc)])
    y0 = np.concatenate([x0, m * v0 / c ** 2])

    def rhs(t, y):
        xx, pv = y[:3], y[3:]
        r = np.linalg.norm(xx)
        U = GM / r
        N, A = np.exp(-U), np.exp(U)
        E = np.sqrt(m ** 2 + c ** 2 * (pv @ pv) / A ** 2)
        bvec = B * np.cross([0, 0, 1.0], xx) / r ** 3
        dx = N * c ** 2 * pv / (A ** 2 * E) + bvec
        H = lambda UU: np.exp(-UU) * np.sqrt(m ** 2 + c ** 2 * (pv @ pv) * np.exp(-2 * UU))
        hU = 1e-6 * U
        F = (H(U + hU) - H(U - hU)) / (2 * hU) * GM * xx / r ** 3
        # - grad (b . p) = - B grad[(z_hat x x) . p / r^3]
        L = np.cross(xx, pv)
        g = B * (np.cross(pv, [0, 0, 1.0]) / r ** 3 - 3 * L[2] * xx / r ** 5)
        return np.concatenate([dx, F - g])
    T = 2 * np.pi * np.sqrt(a_s ** 3 / GM) / c
    sol = solve_ivp(rhs, (0, 8 * T), y0, method='DOP853', rtol=1e-11, atol=1e-9, dense_output=True, max_step=T / 300)
    tt = np.linspace(0, 8 * T, 160000)
    Y = sol.sol(tt)
    Lv = np.cross(Y[:3].T, Y[3:].T)
    node = np.unwrap(np.arctan2(Lv[:, 0], -Lv[:, 1]))          # longitude of the ascending node
    rr = np.linalg.norm(Y[:3], axis=0)
    ip = [i for i in range(1, len(rr) - 1) if rr[i] < rr[i - 1] and rr[i] <= rr[i + 1]]
    rate = np.polyfit(tt[ip], node[ip], 1)[0]                  # sampled once per orbit (at perihelion)
    rmin, rmax = rr.min(), rr.max()
    a_m, e_m = (rmin + rmax) / 2, (rmax - rmin) / (rmax + rmin)   # measured orbital elements
    pred = B / (a_m ** 3 * (1 - e_m ** 2) ** 1.5)
    print(f"   classical orbit (a = {a_s}, e = {e}, i = 60 deg) with b = B (J x x)/r^3: node precession {rate:.4e} per time, "
          f"Lense-Thirring B/(a^3 (1-e^2)^1.5) = {pred:.4e} (ratio {rate/pred:.3f})")
    out['C1_orbit'] = np.array([rate, pred])


def part_C2(out):
    print("C2/C3. frame dragging in the 3D walker: rigid drag field b = Omega z x r, local drag hop; gyroscope")
    from quantum_fcc_3d import step3
    t0 = time.time()
    shape = (64, 64, 64)
    mu, sig, nst = 0.15, 4.0, 60          # short: the spreading packet must not reach the open box edge
    Om = 0.004
    C = expm(1j * Xl(LAM0)(mu)).astype(np.complex64)
    J = J_int()
    E0, V0 = bands_C(np.zeros(3), expm(1j * Xl(LAM0)(mu)))
    Q, _ = np.linalg.qr(V0[:, np.argsort(np.abs(E0 - 2 * mu))[:2]])
    Jd = [Q.conj().T @ j @ Q for j in J]
    w, v = np.linalg.eigh(Jd[0])
    uref = Q @ v[:, np.argmax(w)]                              # spin along +x (J_x = +1/2)
    print(f"   rest doublet: J_x, J_y, J_z restricted have eigenvalues +-1/2: {np.round(np.linalg.eigvalsh(Jd[2]), 4)}; "
          f"<J_x> of the start state {np.real(uref.conj() @ J[0] @ uref):+.4f}")
    xs = [(np.arange(n) - n // 2) * LAT_A for n in shape]
    X, Y, Z = np.meshgrid(*xs, indexing='ij')
    R0 = 8.0
    res = {}
    for label, spin_rule in (('drag hop only', False), ('drag hop + internal rotation (1/2) curl b . J', True)):
        # packet: Gaussian at (R0, 0, 0) with the fixed internal state uref (rest doublet, sigma large)
        env = np.exp(-((X - R0) ** 2 + Y ** 2 + Z ** 2) / (4 * sig ** 2))
        I = np.indices(shape).sum(0) - sum(n // 2 for n in shape)
        env = env * (I % 2 == 0)
        psi = (env[..., None] * uref[None, None, None, :]).astype(np.complex64)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        dt = DT_HALF

        def theta(s):
            sp = s * LAT_A
            mid = [X + sp[0] / 2, Y + sp[1] / 2]
            bx, by = -Om * mid[1], Om * mid[0]
            return (dt * (bx * sp[0] + by * sp[1]) / (8 * LAT_A ** 2)).astype(np.float32)
        th_cache = {}
        Urot = expm(-1j * dt * Om * J[2]).astype(np.complex64)   # (1/2) curl b = Omega z
        ang, spin, ts = [], [], []
        for it in range(nst + 1):
            if it % 4 == 0:
                rho = (np.abs(psi) ** 2).sum(-1)
                cx, cy = (rho * X).sum() / rho.sum(), (rho * Y).sum() / rho.sum()
                ang.append(np.arctan2(cy, cx))
                flat = psi.reshape(-1, 24)
                jx = np.real(np.sum(np.conj(flat) * (flat @ J[0].T)))
                jy = np.real(np.sum(np.conj(flat) * (flat @ J[1].T)))
                spin.append(np.arctan2(jy, jx))
                ts.append(it * dt)
            if it == nst:
                break
            psi = step3(psi.reshape(*shape, 12, 2), C, 2).reshape(*shape, 24)
            psi = drag3d(psi, lambda s: th_cache.setdefault(tuple(s), theta(s)))
            if spin_rule:
                psi = (psi.reshape(-1, 24) @ Urot.T).reshape(*shape, 24)
        ts, ang, spin = np.array(ts), np.unwrap(ang), np.unwrap(spin)
        ra, rs = np.polyfit(ts, ang, 1)[0], np.polyfit(ts, spin, 1)[0]
        res[label] = (ra, rs)
        out[f'C2_{int(spin_rule)}'] = np.array([ts, ang, spin])
        print(f"   {label:46s}: orbit angle rate {ra:+.5f} (Omega = {Om}), spin angle rate {rs:+.5f}; "
              f"norm {np.sum(np.abs(psi)**2):.6f}  [{time.time()-t0:.0f}s]")
    out['C2_rates'] = np.array([[*res['drag hop only'], *res['drag hop + internal rotation (1/2) curl b . J']], [Om, Om, Om, Om]])


def main():
    parts = [p.upper() for p in sys.argv[1:]] or ['A']
    path = data_path('phase_bit_s4_open13.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in parts:
        {'A': part_A, 'B1': part_B1, 'B2': part_B2, 'C1': part_C1, 'C2': part_C2}[p](out)
        np.savez_compressed(path, **out)
    print("saved data/phase_bit_s4_open13.npz")


if __name__ == '__main__':
    main()
