#!/usr/bin/env python3
"""
Paper 3, fifth follow-up to Experiment S4 — strong and time-dependent lapse,
lapse from the local load (Exp. 1d), anisotropic metric, global Landau branches
==============================================================================

A.  Strong fields with the wait buffer of round 4:
    A1  one buffer already covers 0 < N <= 1 (a in (pi/2, pi) gives N < 1/2);
        general rule for any local wait scheme: slope N = active weight of the
        vacuum fixed vector (random 3-level schemes).
    A2  approach to a horizon: massless packet running into N -> N_min on an
        x-chain; coordinate speed N c', blueshift of the lattice momentum k = E/(N c')
        until the lattice cutoff.
    A3  time-dependent uniform lapse N(t): phase = integral of N E dt, leakage
        out of the physical branch vs ramp time.
B.  Exp. 1d: lapse sourced by the local load (energy density per node).
    Two packets on an x-chain (separate Bloch momenta), each moves in the lapse
    N0 (1 + Phi) sourced by the other one, Phi'' = kappa e(x) (1D Green function),
    e(x) = Re psi^+(x) i (U psi - psi)(x) / dt  (local).  Third law, active mass
    of a moving source.
C.  Anisotropic metric (variable edge lengths):
    C1  heading-dependent waits only give the average lapse (proof + numbers);
    C2  heading-dependent coin couplings give an ellipsoidal cone; two response
        constants (E_g, T_2g of the cubic group) calibrated, then any traceless
        strain h reproduces g^-1 = (1 + 2h)^-1;
    C3  gravitational-wave interferometer: h_plus(t) uniform, photons along x and y.
D.  Global connection of the chiral Landau branches: periodic x-chain with one
    flux quantum (B = 2 pi / (Nx a^2)), helicity sector +1, spectrum vs k_z over
    the whole zone and the winding of det U(k_z).

Run parts: python phase_bit_s4_open5.py [A] [B] [C] [D]   (default: all)
"""

import functools
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import MOVES_PHYS, DT_HALF, FCC_STEPS, LAT_A
from phase_bit_mass_s4 import model_generators, candidates, bands_C
from phase_bit_s4_open import CP
from phase_bit_s4_open4 import f_dwell, a_of, dwell_k, model_2Ostar, quartet, EPS, N0
from paths import data_path

print = functools.partial(print, flush=True)
UD = MOVES_PHYS / np.linalg.norm(MOVES_PHYS, axis=1)[:, None]


def chain_T(C, kyz, Nx):
    """Sparse x-chain step (open ends), uniform coin C, Bloch (ky, kz)."""
    T = None
    for dl in (-1, 0, 1):
        M = np.zeros((24, 24), complex)
        for d in np.where(FCC_STEPS[:, 0] == dl)[0]:
            ph = np.exp(-1j * (kyz[0] * MOVES_PHYS[d, 1] + kyz[1] * MOVES_PHYS[d, 2]))
            M[2 * d:2 * d + 2] += ph * C[2 * d:2 * d + 2]
        S = sp.eye(Nx, k=-dl, format='csr')          # layer l -> l + dl
        blk = sp.kron(S, sp.csr_matrix(M), format='csr')
        T = blk if T is None else T + blk
    return T


class Buffered:
    """Buffered tick on the x-chain: m (active), b (buffer), angle a(x) per layer."""

    def __init__(self, T, Nx):
        self.T, self.Nx = T, Nx

    def set_a(self, a):
        self.ca, self.sa = np.repeat(np.cos(a), 24), np.repeat(np.sin(a), 24)

    def tick(self, m, b):
        m2, b2 = self.T @ m, -b
        return self.ca * m2 + 1j * self.sa * b2, 1j * self.sa * m2 + self.ca * b2


def packet(C, kyz, sel, Nx, xs, x0, sig, kx0=0.0):
    """Gaussian packet on the band(s) sel(E) at (kx0, ky, kz), centred at x0."""
    E, U = bands_C(np.array([kx0, *kyz]), C)
    uref = U[:, sel(E)[0]]
    uref /= np.linalg.norm(uref)
    psi = np.zeros((Nx, 24), complex)
    for kx in 2 * np.pi * np.fft.fftfreq(Nx) / LAT_A:
        gk = np.exp(-sig ** 2 * (kx - kx0) ** 2 / 2)
        if gk < 1e-8:
            continue
        Ek, Uk = bands_C(np.array([kx, *kyz]), C)
        Q, _ = np.linalg.qr(Uk[:, sel(Ek)])
        u = Q @ (Q.conj().T @ uref)
        psi += gk * np.exp(1j * kx * (xs - x0))[:, None] * (u / np.linalg.norm(u))[None, :]
    return (psi / np.linalg.norm(psi)).ravel(), E[sel(E)].mean()


def on_branch(bt, psi, th_c, a, aref, nfilt=3):
    """Buffer amplitude from the local 2x2 eigenvector + spectral filter against the pi branch."""
    lam_c = np.exp(-1j * f_dwell(th_c, a))
    r = (np.cos(a) * np.exp(-1j * th_c) - lam_c) / (1j * np.sin(a))
    m, b = psi.copy(), np.repeat(r, 24) * psi
    ph = np.exp(1j * f_dwell(th_c, aref))
    for _ in range(nfilt):
        m2, b2 = bt.tick(m, b)
        m, b = (m + ph * m2) / 2, (b + ph * b2) / 2
    nrm = np.sqrt(np.vdot(m, m).real + np.vdot(b, b).real)
    return m / nrm, b / nrm


# ═════════════════════════════════════════════════════════════════════════════
def part_A(out):
    print("A. strong and time-dependent lapse with the wait buffer (2O*, eps = 0.3)")
    X, lam = model_2Ostar()
    C = expm(1j * X(0.03))
    C0 = expm(1j * X(0.0))

    # A1: one buffer below N = 1/2
    rng = np.random.default_rng(1)
    err = 0.0
    for _ in range(10):
        k = rng.uniform(-1.5, 1.5, 3)
        th0 = -np.angle(np.linalg.eigvals(dwell_k(k, C, 0.0)[:24, :24]))
        for N in (0.4, 0.2, 0.05):
            a = a_of(N)
            l = np.linalg.eigvals(dwell_k(k, C, a))
            err = max(err, max(np.min(np.abs(l - np.exp(-1j * f_dwell(t, a)))) for t in th0))
    q0 = quartet(C)
    rows = []
    for N in (0.4, 0.2, 0.1, 0.05, 0.02):
        a = a_of(N)
        E_, V_ = np.linalg.eig(dwell_k(np.zeros(3), C, a))
        Ed = -np.angle(E_) / DT_HALF
        idx = np.argsort(np.abs(Ed))[:4]
        wb = ((np.abs(V_[24:, idx]) ** 2).sum(0) / (np.abs(V_[:, idx]) ** 2).sum(0)).mean()
        rows.append((N, np.sort(Ed[idx])[3] / q0[3], wb, 2 * (np.pi - a) / DT_HALF))
    print(f"   A1  formula f_a vs exact 48x48 for N = 0.4, 0.2, 0.05: max error {err:.1e}")
    for N, r, wb, gap in rows:
        print(f"       N = {N:5.2f}: quartet ratio {r:.5f}, buffer weight {wb:.4f} (1-N = {1-N:.2f}), "
              f"branch gap at the zone edge 2(pi-a)/dt = {gap:.3f}")
    out['A1_rows'] = np.array(rows)
    out['A1_err'] = err

    # general rule: random local wait schemes with two buffers (phases -1, +-1)
    rows = []
    for trial in range(6):
        nb = 2
        v = rng.normal(size=nb + 1) + 1j * rng.normal(size=nb + 1)
        v /= np.linalg.norm(v)
        D0 = np.diag([1.0, -1.0, (-1.0) ** trial])
        u = D0 @ v
        W, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
        w = W @ u
        # unitary K with K w = v: Householder-type with phase
        ph = np.vdot(w, v) / abs(np.vdot(w, v))
        x = ph * w - v
        K = (np.eye(3) - 2 * np.outer(x, x.conj()) / np.vdot(x, x)) * ph
        M = K @ W                                   # M D0 v = v
        assert np.allclose(M @ D0 @ v, v)
        # replace the active 1 by the walker: U = (M (x) 1)(SC (+) phases)
        Mbig = np.kron(M, np.eye(24))

        def Ufull(k):
            u_ = np.repeat(np.exp(-1j * (MOVES_PHYS @ k)), 2)[:, None] * C0
            Dg = sp.block_diag([u_, D0[1, 1] * np.eye(24), D0[2, 2] * np.eye(24)]).toarray()
            return Mbig @ Dg
        kk = 0.01
        E = -np.angle(np.linalg.eigvals(Ufull(np.array([0, 0, kk])))) / DT_HALF
        e = np.sort(E[np.argsort(np.abs(E))[:4]])
        rows.append((abs(v[0]) ** 2, e[3] / (CP * kk)))
    out['A1_general'] = np.array(rows)
    print("   A1  two-buffer schemes, slope/c' vs active weight |v_0|^2 of the fixed vector: " +
          ", ".join(f"{s:.4f}/{w:.4f}" for w, s in rows))

    # A2: horizon approach (massless packet along -x into N -> N_min)
    print("   A2  massless packet running into a strong lapse gradient (x-chain, buffer lapse)")
    t0 = time.time()
    Nx = 360
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    Nmax, Nmin, x1, x2 = 0.9, 0.04, -40.0, 40.0   # N(x): Nmin for x < x1, ramp, Nmax for x > x2
    Nprof = lambda x: np.clip(Nmin + (Nmax - Nmin) * (x - x1) / (x2 - x1), Nmin, Nmax)
    a = a_of(Nprof(xs))
    sel = lambda E: np.sort(np.argsort(np.abs(E - CP * 0.3))[:2])
    T = chain_T(C0, (0.0, 0.0), Nx)
    bt = Buffered(T, Nx)
    bt.set_a(a)
    psi, Eb = packet(C0, (0.0, 0.0), sel, Nx, xs, 60.0, 12.0, kx0=-0.3)
    m, b = on_branch(bt, psi, Eb * DT_HALF, a, a_of(Nmax))
    kgrid = 2 * np.pi * np.fft.fftfreq(Nx) / LAT_A
    rec = []
    nst = 2400
    for it in range(nst + 1):
        if it % 10 == 0:
            w = (np.abs(m) ** 2 + np.abs(b) ** 2).reshape(Nx, 24).sum(1)
            mk = np.abs(np.fft.fft(m.reshape(Nx, 24), axis=0)) ** 2
            pk = mk.sum(1) / mk.sum()
            kc = np.angle(np.sum(pk * np.exp(1j * kgrid * LAT_A))) / LAT_A   # circular mean of k
            kc = kc - 2 * np.pi / LAT_A if kc > 0.5 else kc
            z = np.vdot(np.concatenate([m, b]), np.concatenate(bt.tick(m, b)))
            refl = w[xs > 60].sum() / w.sum() if it > 300 else 0.0
            rec.append((it * DT_HALF, (w @ xs) / w.sum(), kc, -np.angle(z) / DT_HALF, refl,
                        np.sqrt(((w @ xs ** 2) / w.sum()) - ((w @ xs) / w.sum()) ** 2)))
        m, b = bt.tick(m, b)
    rec = np.array(rec)
    out['A2_rec'] = rec
    out['A2_prof'] = np.array([Nmax, Nmin, x1, x2])
    # rays: (i) linear cone dx/dt = -c' N(x); (ii) exact band H(x, k) = f_{a(x)}(theta_ph(k)) / dt
    kb = np.linspace(-0.05, -2 * np.pi / LAT_A + 0.3, 3000)
    Eb_ = [CP * 0.05]
    for q in kb[1:]:
        E_ = bands_C(np.array([q, 0, 0]), C0)[0]
        Eb_.append(E_[np.argmin(np.abs(E_ - Eb_[-1]))])
    Eb_ = np.array(Eb_)
    Eph = lambda q: np.interp(q, kb[::-1], Eb_[::-1])
    Hs = lambda x, q: f_dwell(Eph(q) * DT_HALF, a_of(Nprof(x))) / DT_HALF
    E0c = Hs(60.0, -0.3)

    def rhs(y):
        x, q = y
        dq, dx = 1e-4, 1e-3
        return np.array([(Hs(x, q + dq) - Hs(x, q - dq)) / (2 * dq), -(Hs(x + dx, q) - Hs(x - dx, q)) / (2 * dx)])
    yr, tr, xl = np.array([60.0, -0.3]), [0.0], [60.0]
    traj = [yr.copy()]
    dt = 0.05
    while tr[-1] < rec[-1, 0]:
        k1 = rhs(yr); k2 = rhs(yr + dt / 2 * k1); k3 = rhs(yr + dt / 2 * k2); k4 = rhs(yr + dt * k3)
        yr = yr + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(yr.copy())
        xl.append(xl[-1] - Nprof(xl[-1]) * CP * dt)
        tr.append(tr[-1] + dt)
    traj = np.array(traj)
    out['A2_ray'] = np.column_stack([tr, xl, traj])
    out['A2_band'] = np.column_stack([kb, Eb_])
    imax = np.argmax(Eb_)
    print(f"       photon band along -x: E rises to {Eb_[imax]:.3f} at k = {kb[imax]:.2f} (linear c'k up to ~1); "
          f"coordinate energy E' = {E0c:.4f}, sim. drift {np.ptp(rec[:,3]):.1e}")
    for t_ in (50, 150, 300, 600, 900, 1200):
        j = np.argmin(np.abs(rec[:, 0] - t_))
        jr = np.argmin(np.abs(np.array(tr) - t_))
        print(f"       t = {t_:4d}: <x> = {rec[j,1]:+7.2f} (exact-band ray {traj[jr,0]:+7.2f}, linear ray {xl[jr]:+7.2f}), "
              f"N = {Nprof(rec[j,1]):.3f}, k = {rec[j,2]:+.3f} (ray {traj[jr,1]:+.3f}), width {rec[j,5]:.1f}")
    from scipy.optimize import brentq
    Nturn = brentq(lambda N: f_dwell(Eb_[imax] * DT_HALF, a_of(N)) / DT_HALF - E0c, 1e-3, 0.999)
    xturn = x1 + (Nturn - Nmin) / (Nmax - Nmin) * (x2 - x1)
    jmin = np.argmin(rec[:, 1])
    print(f"       turning point (band maximum reached, f_N(E_max) = E'): N = {Nturn:.4f}, x = {xturn:+.2f}; "
          f"simulated deepest <x> = {rec[jmin,1]:+.2f} at t = {rec[jmin,0]:.0f}")
    out['A2_turn'] = np.array([Nturn, xturn, rec[jmin, 1], rec[jmin, 0], E0c, Eb_[imax]])
    print(f"       returned to x > 60 at the end: {rec[-1,4]:.3f}   [{time.time()-t0:.0f}s]")

    # A3: uniform time-dependent lapse
    print("   A3  uniform ramp N: 0.9 -> 0.3 (k = (0.2, 0, 0), massive mu = 0.03): phase and leakage")
    k = np.array([0.2, 0.0, 0.0])
    E, V = bands_C(k, C)
    j = np.argsort(np.abs(E))[:4]
    j = np.sort(j)[3]
    th = E[j] * DT_HALF
    rows = []
    for Tr in (4, 10, 30, 100, 300, 1000):
        Nt = lambda t: 0.9 - 0.6 * np.clip(t / Tr, 0, 1) if Tr > 0 else 0.3
        a = a_of(0.9)
        lam_c = np.exp(-1j * f_dwell(th, a))
        r = (np.cos(a) * np.exp(-1j * th) - lam_c) / (1j * np.sin(a))
        psi = np.concatenate([V[:, j], r * V[:, j]])
        psi /= np.linalg.norm(psi)
        phase = 0.0
        nt = Tr + 200
        for it in range(nt):
            Uk = dwell_k(k, C, a_of(Nt(it + 0.5)))
            psi = Uk @ psi
            phase += f_dwell(th, a_of(Nt(it + 0.5)))
        a1 = a_of(0.3)
        lam1 = np.exp(-1j * f_dwell(th, a1))
        r1 = (np.cos(a1) * np.exp(-1j * th) - lam1) / (1j * np.sin(a1))
        target = np.concatenate([V[:, j], r1 * V[:, j]])
        target /= np.linalg.norm(target)
        # weight on the physical branch of the final tick (projector onto the 24 states near 0)
        Ef, Vf = np.linalg.eig(dwell_k(k, C, a1))
        phys = np.abs(np.angle(Ef)) < np.pi / 2
        Q, _ = np.linalg.qr(Vf[:, phys])
        wphys = np.linalg.norm(Q.conj().T @ psi) ** 2
        ov = np.vdot(target, psi)
        dphi = np.angle(ov * np.exp(1j * phase))
        rows.append((Tr, 1 - wphys, 1 - abs(ov) ** 2, dphi))
        print(f"       ramp {Tr:5d} ticks: leaked to pi branch {1-wphys:.2e}, left the band state {1-abs(ov)**2:.2e}, "
              f"phase - sum f(N(t)) = {dphi:+.2e}")
    out['A3_rows'] = np.array(rows)


# ═════════════════════════════════════════════════════════════════════════════
def part_B(out):
    print("B. Exp. 1d: lapse sourced by the local load e(x) (two packets, 2O*, buffer lapse)")
    t0 = time.time()
    X, lam = model_2Ostar()
    Nx, SIG, nst, D = 1000, 60.0, 200, 400
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    ts = np.arange(nst + 1) * DT_HALF
    upper = lambda E: np.sort(np.argsort(np.abs(E))[:4])[2:]
    kap = 2e-3 / (N0 * 0.06)                # slope 1e-3 from a rest source with E = 0.06 (N < 1 over the packets)

    # R' per species on the band level (as in round 4)
    def Rprime(mu):
        Cm = expm(1j * X(mu))
        h, dN = 1e-3, 1e-4
        Eb = lambda kx, N: f_dwell(quartet(Cm, (kx, 0, 0)) * DT_HALF, a_of(N)) / DT_HALF
        e0 = Eb(0, N0)
        dEdN = (Eb(0, N0 + dN) - Eb(0, N0 - dN)) / (2 * dN)
        curv = 2 * (Eb(h, N0) - e0) / h ** 2
        return (dEdN * curv / (N0 * CP ** 2))[3]

    def a_unit(mu, kyz=(0.0, 0.0)):
        """Semiclassical acceleration per unit lapse gradient for the packet's k spread (as in round 4)."""
        Cm = expm(1j * X(mu))
        Hsc = lambda kx, x: (lambda E_: f_dwell(E_[upper(E_)].mean() * DT_HALF, a_of(N0 * (1 + 1e-3 * x))) / DT_HALF)(
            bands_C(np.array([kx, *kyz]), Cm)[0])
        kq = np.linspace(-3 / (SIG * LAT_A), 3 / (SIG * LAT_A), 41)
        wq = np.exp(-(SIG * LAT_A) ** 2 * kq ** 2)
        Fx = np.array([(Hsc(k, 1.0) - Hsc(k, -1.0)) / 2 for k in kq])
        d2 = np.array([(Hsc(k + 1e-3, 0) - 2 * Hsc(k, 0) + Hsc(k - 1e-3, 0)) / 1e-6 for k in kq])
        return -np.sum(wq * Fx * d2) / wq.sum() / 1e-3

    def run(specs, kaps=None, self_grav=False):
        """specs: list of (mu, (ky, kz), x0); kaps: coupling of the field sourced by each packet."""
        kaps = kaps or [kap] * len(specs)
        st = []
        for mu, kyz, x0 in specs:
            C = expm(1j * X(mu))
            psi, Eb = packet(C, kyz, upper, Nx, xs, x0, SIG)
            bt = Buffered(chain_T(C, kyz, Nx), Nx)
            bt.set_a(np.full(Nx, a_of(N0)))
            m, b = on_branch(bt, psi, Eb * DT_HALF, np.full(Nx, a_of(N0)), a_of(N0))
            e = N0 * Eb * (np.abs(m) ** 2 + np.abs(b) ** 2).reshape(Nx, 24).sum(1)
            st.append(dict(bt=bt, m=m, b=b, e=e, Eb=Eb, xc=[], Q=[]))
        absdx = np.abs(xs[:, None] - xs[None, :])
        for it in range(nst + 1):
            phis = [0.5 * kj * absdx @ s['e'] for kj, s in zip(kaps, st)]
            for i, s in enumerate(st):
                Phi = sum(p for j, p in enumerate(phis) if j != i or self_grav)
                w = (np.abs(s['m']) ** 2 + np.abs(s['b']) ** 2).reshape(Nx, 24).sum(1)
                s['xc'].append((w @ xs) / w.sum())
                Phi = Phi - np.interp(s['xc'][-1], xs, Phi)       # reference at the particle itself
                s['Q'].append(s['e'].sum())
                if it == nst:
                    continue
                s['bt'].set_a(a_of(np.clip(N0 * (1 + Phi), 1e-3, 1 - 1e-9)))   # far tails only
                m2, b2 = s['bt'].tick(s['m'], s['b'])
                # local load: Re psi^+ i (U psi - psi) / dt, summed per layer
                ld = (np.conj(s['m']) * 1j * (m2 - s['m']) + np.conj(s['b']) * 1j * (b2 - s['b'])).real / DT_HALF
                s['e'] = ld.reshape(Nx, 24).sum(1)
                s['m'], s['b'] = m2, b2
        acc = [2 * np.polyfit(ts, np.array(s['xc']), 2)[0] for s in st]
        return acc, [np.array(s['Q']) for s in st], [s['Eb'] for s in st], [np.array(s['xc']) for s in st]

    pref = N0 ** 2 * CP ** 2 * kap / 2
    R = {mu: Rprime(mu) for mu in (0.03, 0.06)}
    AU = {mu: a_unit(mu) for mu in (0.03, 0.06)}
    print(f"   kappa = {kap:.4f}; R' (band level, buffer lapse N0 = {N0}): mu = 0.03: {R[0.03]:.4f}, mu = 0.06: {R[0.06]:.4f}; "
          f"with the packet's k spread (sigma = {SIG:.0f} layers): {AU[0.03]/(-N0**2*CP**2):.4f} / {AU[0.06]/(-N0**2*CP**2):.4f}")

    # (1) two different masses at rest
    acc, Q, Eb, xc = run([(0.03, (0.0, 0.0), -D / 2 * LAT_A), (0.06, (0.0, 0.0), +D / 2 * LAT_A)])
    print(f"   (1) mu = 0.03 at x = {-D/2*LAT_A:.0f}, mu = 0.06 at x = {D/2*LAT_A:+.0f} (m = 0.06 / 0.12):")
    print(f"       load integral Q = sum e: {Q[0][0]:.5f} -> {Q[0][-1]:.5f} and {Q[1][0]:.5f} -> {Q[1][-1]:.5f} "
          f"(N0 E = {N0*Eb[0]:.5f} / {N0*Eb[1]:.5f})")
    p1 = R[0.03] * pref * Q[1].mean()
    p2 = -R[0.06] * pref * Q[0].mean()
    print(f"       a_1 = {acc[0]:+.3e} (pred. R'_1 N0^2 c'^2 kappa Q_2/2 = {p1:+.3e}), "
          f"a_2 = {acc[1]:+.3e} (pred. {p2:+.3e})")
    s1, s2 = -AU[0.03] * kap / 2 * Q[1].mean(), AU[0.06] * kap / 2 * Q[0].mean()
    print(f"       semiclassics with the packets' k spread: a_1 = {s1:+.3e}, a_2 = {s2:+.3e}")
    third = (Eb[0] * acc[0] + Eb[1] * acc[1]) / (Eb[0] * abs(acc[0]))
    print(f"       third law: (E_1 a_1 + E_2 a_2)/(E_1 |a_1|) = {third:+.4f}  "
          f"(expected from the species' fall factors: {(Eb[0]*s1+Eb[1]*s2)/(Eb[0]*abs(s1)):+.4f}; band-level R' only: {R[0.03]/R[0.06]-1:+.4f})")
    out['B_two'] = np.array([acc[0], acc[1], p1, p2, Eb[0], Eb[1], third, R[0.03], R[0.06], Q[0].mean(), Q[1].mean(), s1, s2])
    out['B_xc1'], out['B_xc2'] = xc[0], xc[1]

    # (2) active mass of a moving source: test particle at rest, source at rest vs moving (k_y)
    res = []
    for ky in (0.0, 0.3, 0.6):
        Es = bands_C(np.array([0.0, ky, 0.0]), expm(1j * X(0.03)))[0]
        ks = kap * 0.06 / Es[upper(Es)].mean()           # keep the field slope small
        acc, Q, Eb, xc = run([(0.03, (0.0, 0.0), -D / 2 * LAT_A), (0.03, (ky, 0.0), +D / 2 * LAT_A)], [kap, ks])
        res.append((ky, acc[0] * kap / ks, Q[1].mean(), Eb[1]))
        print(f"   (2) source mu = 0.03 with k_y = {ky}: E_source = {Eb[1]:.4f}, Q = {Q[1].mean():.5f}; "
              f"a_test (rescaled to kappa) = {acc[0]*kap/ks:+.3e}  [{time.time()-t0:.0f}s]")
    r = np.array(res)
    print("       a_test / a_test(rest source) = " + ", ".join(f"{x:.3f}" for x in r[:, 1] / r[0, 1]) +
          "   vs E/E(rest) = " + ", ".join(f"{x:.3f}" for x in r[:, 3] / r[0, 3]))
    out['B_moving'] = r


# ═════════════════════════════════════════════════════════════════════════════
DIRS = np.array([v / np.linalg.norm(v) for v in map(np.array, [
    (1., 0, 0), (0, 1., 0), (0, 0, 1.), (1., 1, 0), (1., -1, 0), (1., 0, 1), (0, 1., 1), (1., 1, 1),
    (1., -1, 1), (1., 0, -1), (0, 1., -1), (1., 1, -1), (-1., 1, 1)])])


def cone_tensor(C, kk=0.01):
    """Fit E^2 = c'^2 k^T M k to the upper level of the Dirac quartet (centre removed)."""
    E0 = bands_C(np.zeros(3), C)[0]
    Ec = np.sort(E0[np.argsort(np.abs(E0))[:4]]).mean()
    A, b, spl = [], [], 0.0
    for n in DIRS:
        E = bands_C(kk * n, C)[0]
        e = np.sort(E[np.argsort(np.abs(E - Ec))[:4]]) - Ec
        spl = max(spl, abs(e[3] - e[2]), abs(e[1] - e[0]), abs(e[3] + e[0]))
        A.append([n[0] ** 2, n[1] ** 2, n[2] ** 2, 2 * n[0] * n[1], 2 * n[0] * n[2], 2 * n[1] * n[2]])
        b.append((e[3] / kk) ** 2)
    s, res, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    M = np.array([[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]]) / CP ** 2
    resid = np.sqrt(res[0] / len(b)) / CP ** 2 if len(res) else 0.0
    return M, Ec, spl, resid


def part_C(out):
    print("C. anisotropic metric: heading-dependent waits vs heading-dependent coin couplings")
    models, _ = model_generators()
    W90 = candidates()['90° transport, K+K†']
    X, lam = model_2Ostar()
    Gm = models['2O register'] + lam * W90
    lv = 2 * np.sqrt(2) + lam * np.sqrt(2)

    # C1: heading-dependent buffer (wait time proportional to the stretched edge length)
    def dwell_d(k, C, ad):
        u = np.repeat(np.exp(-1j * (MOVES_PHYS @ np.asarray(k, float))), 2)[:, None] * C
        I, Z = np.eye(24), np.zeros((24, 24))
        Dg = np.block([[u, Z], [Z, -I]])
        c, s = np.diag(np.repeat(np.cos(ad), 2)), np.diag(np.repeat(np.sin(ad), 2))
        return np.block([[c, 1j * s], [1j * s, c]]) @ Dg

    C0 = expm(1j * X(0.0))
    h = 0.05
    A = np.diag([1 + h, 1 - h, 1.0])
    L = np.linalg.norm(UD @ A.T, axis=1)
    ad = a_of(N0 / L)
    kk = 0.01
    sl = []
    for n in DIRS[:3]:
        E = -np.angle(np.linalg.eigvals(dwell_d(kk * n, C0, ad))) / DT_HALF
        sl.append(np.sort(E[np.argsort(np.abs(E))[:4]])[3] / (CP * kk))
    # first-order theory: velocity numerator unchanged, normalisation = mean over headings of 1/N_d
    pred = 1 / np.mean(1 / (N0 / L))
    print(f"   C1  waits N_d = N0/L_d for A = diag(1+h, 1-h, 1), h = {h}: slopes along x, y, z = "
          + ", ".join(f"{s:.5f}" for s in sl) + f" c'  (isotropic; 1/<1/N_d> = {pred:.5f}; "
          f"GR would need {N0/(1+h):.4f}, {N0/(1-h):.4f}, {N0:.4f})")
    out['C1'] = np.array([*sl, pred, N0 / (1 + h), N0 / (1 - h), N0])

    # C2: heading-dependent coin couplings s_d s_d' (s_d = 1 + u_d^T S u_d)
    def coinX(Sm):
        s = np.repeat(1 + np.einsum('di,ij,dj->d', UD, Sm, UD), 2)
        return EPS * (s[:, None] * Gm * s[None, :])

    def Mof(Sm):
        Xg = coinX(Sm) - EPS * lv * np.eye(24)
        M, Ec, spl, resid = cone_tensor(expm(1j * Xg))
        M2, Ec2, spl2, resid2 = cone_tensor(expm(1j * (Xg + Ec * DT_HALF * np.eye(24))))
        return M2, Ec, spl2, resid2

    eps_ = 0.01
    Mp, Ecp, *_ = Mof(np.diag([eps_, -eps_, 0.0]))
    Mc, Ecc, *_ = Mof(np.array([[0, eps_, 0], [eps_, 0, 0], [0, 0, 0.0]]))
    rE = (Mp[0, 0] - Mp[1, 1]) / 2 / eps_          # response per unit S (E_g)
    rT = Mc[0, 1] / eps_                           # response per unit S (T_2g)
    print(f"   C2  response of the cone tensor M to S: E_g {rE:+.4f}, T_2g {rT:+.4f} "
          f"(different -> two constants, cubic lattice)")
    # GR target for metric g = 1 + 2h (h traceless): M = g^-1 ~ 1 - 2h  ->  S = -2 h_E / rE - 2 h_T / rT
    rng = np.random.default_rng(5)
    rows = []
    for trial in range(5):
        hm = rng.normal(size=(3, 3)) * 0.01
        hm = (hm + hm.T) / 2
        hm -= np.trace(hm) / 3 * np.eye(3)
        hE, hT = np.diag(np.diag(hm)), hm - np.diag(np.diag(hm))
        Sm = -2 * hE / rE - 2 * hT / rT
        M, Ec, spl, resid = Mof(Sm)
        target = np.linalg.inv(np.eye(3) + 2 * hm)
        dev = np.abs(M - target).max() / np.abs(2 * hm).max()
        rows.append((np.abs(2 * hm).max(), np.abs(M - target).max(), dev, Ec, spl, resid))
        print(f"       random strain |2h| = {np.abs(2*hm).max():.4f}: |M - g^-1| = {np.abs(M-target).max():.1e} "
              f"(relative {dev:.3f}), vacuum shift before retuning {Ec:+.1e}, quartet splitting {spl:.1e}, "
              f"ellipsoid fit residual {resid:.1e}")
    # the plus example of C1 (h = 0.05) with coin couplings instead of waits
    hp = np.diag([h, -h, 0.0])
    Mpl, _, _, _ = Mof(-2 * hp / rE)
    gpl = np.sqrt(np.diag(np.linalg.inv(np.eye(3) + 2 * hp)))
    print(f"       plus h = {h}: speeds along x, y, z with coin couplings " +
          ", ".join(f"{v:.4f}" for v in np.sqrt(np.diag(Mpl))) + " c'  (GR, g = 1 + 2h: " +
          ", ".join(f"{v:.4f}" for v in gpl) + ")")
    out['C2_plus'] = np.vstack([np.sqrt(np.diag(Mpl)), gpl])
    out['C2_resp'] = np.array([rE, rT])
    out['C2_rows'] = np.array(rows)
    # vacuum shift scaling with the strain amplitude
    vs = []
    for amp in (0.005, 0.01, 0.02, 0.04):
        _, Ec, _, _ = Mof(np.diag([amp, -amp, 0.0]) * (-2 / rE))
        vs.append((amp, Ec))
    print("       vacuum shift vs h (plus): " + ", ".join(f"h = {a}: {e:+.2e}" for a, e in vs) + "  (second order)")
    out['C2_vac'] = np.array(vs)

    # C3: gravitational-wave interferometer (uniform h_plus(t), photons along x and y)
    print("   C3  gravitational wave h_plus(t) = h0 sin(W t): photons along x and y (exact tick, no buffer)")
    h0, W, nt = 0.01, 0.05, 800
    Sbase = np.diag([1.0, -1.0, 0.0]) * (-2 / rE)
    Cfree = expm(1j * (EPS * Gm - EPS * lv * np.eye(24)))
    phot = lambda E, kph: np.argmin(np.abs(E - CP * kph))
    for kph in (0.03, 0.1, 0.3):
        res = {}
        for name, n in (('x', np.array([1.0, 0, 0])), ('y', np.array([0, 1.0, 0]))):
            k = kph * n
            E, V = bands_C(k, Cfree)
            j = phot(E, kph)
            psi = V[:, j] / np.linalg.norm(V[:, j])
            ph_ad, rec = 0.0, []
            for it in range(nt):
                t = (it + 0.5) * DT_HALF
                Xg = coinX(Sbase * h0 * np.sin(W * t)) - EPS * lv * np.eye(24)
                Cg = expm(1j * Xg)
                Uk = np.repeat(np.exp(-1j * (MOVES_PHYS @ k)), 2)[:, None] * Cg
                Ek = bands_C(k, Cg)[0]
                ph_ad += Ek[phot(Ek, kph)] * DT_HALF
                psi = Uk @ psi
                if (it + 1) % 10 == 0:
                    rec.append(((it + 1) * DT_HALF, ph_ad, np.angle(np.vdot(V[:, j], psi) * np.exp(1j * ph_ad))))
            res[name] = (np.array(rec), abs(np.vdot(V[:, j], psi)))
        tx, ty = res['x'][0], res['y'][0]
        t = tx[:, 0]
        dphi_ad = tx[:, 1] - ty[:, 1]                  # adiabatic band phases (x minus y)
        # GR, first order (g = 1 + 2h): omega_x - omega_y = -2 h k v(k); with v = c' and with the lattice v_g(k)
        Eph = lambda q: (lambda E_: E_[phot(E_, q)])(bands_C(np.array([q, 0, 0]), Cfree)[0])
        vg = (Eph(kph + 1e-4) - Eph(kph - 1e-4)) / 2e-4
        gr = -2 * h0 * CP * kph * (1 - np.cos(W * t)) / W
        grl = gr * vg / CP
        dev_ex = np.abs(tx[:, 2] - ty[:, 2]).max()
        print(f"       k = {kph}: phase difference x - y max {np.abs(dphi_ad).max():.5f} rad; GR with c': "
              f"{np.abs(gr).max():.5f}, GR with lattice v_g = {vg/CP:.3f} c': {np.abs(grl).max():.5f}; "
              f"exact tick vs adiabatic band {dev_ex:.1e}; fidelity {res['x'][1]:.6f}/{res['y'][1]:.6f}")
        out[f'C3_k{kph}'] = np.column_stack([t, dphi_ad, gr, grl, tx[:, 2] - ty[:, 2]])


# ═════════════════════════════════════════════════════════════════════════════
def part_D(out):
    print("D. periodic x-chain with one flux quantum, helicity sector +1 (minimal SU(2), mu = 0)")
    t0 = time.time()
    models, _ = model_generators()
    H = candidates()['on-site u_d.s']
    C = expm(1j * EPS * models['minimal SU(2)'])
    wH, VH = np.linalg.eigh(H)
    P1 = VH[:, np.isclose(wH, 1)]
    CP1 = C @ P1
    Md = [P1[2 * d:2 * d + 2].conj().T @ CP1[2 * d:2 * d + 2] for d in range(12)]
    Mz = [P1[2 * d:2 * d + 2].conj().T @ P1[2 * d:2 * d + 2] * MOVES_PHYS[d, 2] for d in range(12)]
    Mz = sum(Mz)                                   # 12 x 12 sector velocity weight (Hellmann-Feynman)
    nodes = (-4 * np.sqrt(3) * EPS, 4 * EPS / np.sqrt(3))

    def sector(Nx, B, ky, kz):
        xs = np.arange(Nx) * LAT_A
        blocks = np.zeros((Nx, 3, 12, 12), complex)
        for d in range(12):
            kyl = ky + B * (xs + 0.5 * MOVES_PHYS[d, 0])
            ph = np.exp(-1j * (kyl * MOVES_PHYS[d, 1] + kz * MOVES_PHYS[d, 2]))
            blocks[:, FCC_STEPS[d, 0] + 1] += ph[:, None, None] * Md[d][None]
        T = np.zeros((12 * Nx, 12 * Nx), complex)
        for l in range(Nx):
            for o in range(3):
                l2 = (l + o - 1) % Nx
                T[12 * l2:12 * l2 + 12, 12 * l:12 * l + 12] += blocks[l, o]
        return T

    for Nx, nk, vec in ((168, 97, True), (336, 0, False)):
        B = 2 * np.pi / (Nx * LAT_A ** 2)
        T = sector(Nx, B, 0.0, 0.0)
        un = np.abs(T @ T.conj().T - np.eye(len(T))).max()
        del T
        kzs = np.linspace(-np.pi / LAT_A, np.pi / LAT_A, 241)
        ph = []
        for kz in kzs:
            s, _ = np.linalg.slogdet(sector(Nx, B, 0.0, kz))
            ph.append(np.angle(s))
        wind = (np.unwrap(ph)[-1] - np.unwrap(ph)[0]) / (2 * np.pi)
        print(f"   Nx = {Nx}, B = {B:.5f} (one flux quantum), sector dim {12*Nx}: unitarity {un:.1e}; "
              f"winding of det U(k_z) over the zone: {wind:+.3f}  [{time.time()-t0:.0f}s]")
        out[f'D_wind_{Nx}'] = wind
        if not vec:
            continue
        # ky independence
        E1 = np.sort(-np.angle(np.linalg.eigvals(sector(Nx, B, 0.0, 0.1))) / DT_HALF)
        E2 = np.sort(-np.angle(np.linalg.eigvals(sector(Nx, B, 0.37, 0.1))) / DT_HALF)
        print(f"       spectrum k_y = 0 vs 0.37: max difference {np.abs(E1-E2).max():.1e}")
        kzs = np.linspace(-np.pi / LAT_A, np.pi / LAT_A, nk)
        allE, allV = [], []
        Mzb = np.kron(np.eye(Nx), Mz)
        for i, kz in enumerate(kzs):
            lamv, U = np.linalg.eig(sector(Nx, B, 0.0, kz))
            U /= np.linalg.norm(U, axis=0)
            vz = np.einsum('ij,ij->j', U.conj(), Mzb @ U).real / DT_HALF
            allE.append(-np.angle(lamv) / DT_HALF)
            allV.append(vz)
            if i % 16 == 0:
                print(f"       k_z = {kz:+.3f} done  [{time.time()-t0:.0f}s]")
        out['D_kz'], out['D_E'], out['D_vz'] = kzs, np.array(allE), np.array(allV)
        out['D_B'] = B
        # chiral branches: states with |v_z| close to c' inside the LL gaps near each node
        for En in nodes:
            i0 = np.argmin(np.abs(kzs))
            E0, v0 = allE[i0], allV[i0]
            near = np.argsort(np.abs(E0 - En))[:6]
            print(f"       node {En:+.3f}: states at k_z = 0 nearest: " +
                  ", ".join(f"{E0[q]-En:+.3f} (v_z {v0[q]/CP:+.2f} c')" for q in near))


def main():
    parts = [p.upper() for p in sys.argv[1:]] or ['A', 'B', 'C', 'D']
    path = data_path('phase_bit_s4_open5.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in parts:
        {'A': part_A, 'B': part_B, 'C': part_C, 'D': part_D}[p](out)
        np.savez(path, **out)
    print("saved data/phase_bit_s4_open5.npz")


if __name__ == '__main__':
    main()
