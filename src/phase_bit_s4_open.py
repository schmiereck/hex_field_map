#!/usr/bin/env python3
"""
Paper 3, follow-up to Experiment S4 — the four open points
==========================================================

A.  Symmetries of the mass terms.  The antiunitary Theta = (1 (x) i sigma_y) K
    used in S4 maps k -> k, so it is PT, not T.  True time reversal is
    T = Pi Theta (heading reversed), checked in the symmetric frame
    W(k) = C^1/2 ph(k) C^1/2:  T W(k)* T^-1 = W(-k)^-1.  The helicity
    H = sum_d |d><d| (x) u_d.sigma commutes with the massless step (exact
    lattice chiral symmetry), the parity-odd mass is Pi.(iH), and
    exp(i theta H/2) rotates mu.Pi into mu(cos theta Pi + sin theta Pi iH):
    the theta angle is unobservable for a free particle.  Weyl nodes per
    helicity sector and their chirality (Nielsen-Ninomiya bookkeeping).
B.  2D counterpart: T = S^3 K on the Z12 walker, T^2 = R6 = -1 on the spin
    sector.  The 2D Dirac point is a single Kramers pair at k = 0; random
    T-even terms never split it, T-odd terms always do.
C.  Background -alpha k^2: at the extremal (lowest) Dirac level it has a fixed
    sign (second-order level repulsion).  At the interior 4-fold Dirac level
    of the minimal-SU(2) walker it can be tuned to zero by a spin-transporting
    90 deg coupling, structurally (alpha*eps independent of eps).
D.  Dynamics in the Dirac regime mu << eps with an exact k-patch method
    (lattice step done in k space on the support of the packet; validated
    against the 81^3 real-space runs of S4).
"""

import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import MOVES_PHYS, DT_HALF, C_LIGHT, N_D
from phase_bit_mass_s4 import (PI, THETA_U, DIRV, SX, SY, SZ, model_generators, candidates,
                               bands_C, group_velocity, fit_damped)
import phase_bit_vortex_dynamics as Z
from paths import data_path

CP = C_LIGHT / 3                       # Dirac slope c/3
T_TRUE = PI @ THETA_U                  # true time reversal T = Pi Theta (times K)


def sym_report(V, H):
    return dict(P=np.abs(PI @ V @ PI - V).max(),
                PT=np.abs(THETA_U @ V.conj() @ THETA_U.conj().T - V).max(),
                T=np.abs(T_TRUE @ V.conj() @ T_TRUE.conj().T - V).max(),
                H=np.abs(H @ V - V @ H).max())


def sym_frame_T(G, V, eps, mu, k):
    Ch = expm(0.5j * (eps * G + mu * V))
    ph = lambda k: np.repeat(np.exp(-1j * (MOVES_PHYS @ k)), 2)
    W = lambda k: Ch @ np.diag(ph(k)) @ Ch
    return np.abs(T_TRUE @ W(k).conj() @ T_TRUE.conj().T - np.linalg.inv(W(-k))).max()


def fourfold_levels(G):
    w = np.linalg.eigvalsh(G)
    vals, cnt = np.unique(np.round(w, 7), return_counts=True)
    return [v for v, n in zip(vals, cnt) if n == 4]


def level_idx(C, E0, n=4):
    E = bands_C(np.zeros(3), C)[0]
    return np.sort(np.argsort(np.abs(E - E0))[:n])


def alpha_eps(G, eps, lv, h=1e-3):
    """alpha*eps of the common background at the G-level lv (direction [100])."""
    C = expm(1j * eps * G)
    idx = level_idx(C, -eps * lv / DT_HALF)
    e0 = bands_C(np.zeros(3), C)[0][idx]
    e1 = bands_C(np.array([h, 0, 0]), C)[0][idx]
    return -np.mean(2 * (e1 - e0) / h ** 2) / 2 * eps


# ─── exact k-patch dynamics ──────────────────────────────────────────────────

class Patch:
    """Band-limited packet on the infinite FCC lattice, evolved exactly in k space.
    psi_hat(k) lives on a cubic patch k0 + q, |q_i| < nsig/sigma; one lattice
    step is psi_hat <- ph(k) * (C psi_hat).  The envelope in real space is the
    inverse FFT over the patch (box 2 pi/dk, resolution 2 pi/(patch width))."""

    def __init__(self, k0, sigma, M, nsig=5.0):
        self.k0, self.M = np.asarray(k0, float), M
        self.dk = 2 * nsig / sigma / M
        q = self.dk * np.fft.fftfreq(M) * M
        QX, QY, QZ = np.meshgrid(q, q, q, indexing='ij')
        self.q = np.stack([QX.ravel(), QY.ravel(), QZ.ravel()], 1)
        self.k = self.k0 + self.q
        self.g = np.exp(-sigma ** 2 * (self.q ** 2).sum(1) / 2)
        self.ph = np.repeat(np.exp(-1j * (self.k @ MOVES_PHYS.T)), 2, axis=1)
        self.L = 2 * np.pi / self.dk
        x = self.L * np.fft.fftfreq(M)
        X, Y, Z_ = np.meshgrid(x, x, x, indexing='ij')
        self.X = np.stack([X.ravel(), Y.ravel(), Z_.ravel()], 1)

    def band_state(self, C, band, chunk=20000):
        """Kramers doublet (band, band+1), smooth gauge by projecting u_ref(k0)."""
        uref = bands_C(self.k0, C)[1][:, band]
        uref = uref / np.linalg.norm(uref)
        out = np.empty((len(self.k), 2 * N_D), complex)
        for a in range(0, len(self.k), chunk):
            T = self.ph[a:a + chunk, :, None] * C[None]
            lam, U = np.linalg.eig(T)
            o = np.argsort(-np.angle(lam) / DT_HALF, axis=1)
            n = np.arange(len(T))
            A = np.stack([U[n, :, o[:, band + j]] for j in range(2)], axis=2)
            Qd, _ = np.linalg.qr(A)
            u = np.einsum('nij,nj->ni', Qd, np.einsum('nij,i->nj', Qd.conj(), uref))
            out[a:a + chunk] = u / np.linalg.norm(u, axis=1, keepdims=True)
        self.psi = self.g[:, None] * out
        self.psi /= np.linalg.norm(self.psi)

    def plain_state(self, u):
        self.psi = self.g[:, None] * (u / np.linalg.norm(u))[None]
        self.psi /= np.linalg.norm(self.psi)

    def step(self, C, n=1):
        CT = C.T
        for _ in range(n):
            self.psi = self.ph * (self.psi @ CT)

    def observe(self):
        M = self.M
        f = np.fft.ifftn(self.psi.reshape(M, M, M, -1), axes=(0, 1, 2)).reshape(-1, self.psi.shape[1])
        p = (np.abs(f) ** 2).sum(1)
        p /= p.sum()
        r = p @ self.X
        return r, np.sqrt((((self.X - r) ** 2).sum(1) * p).sum())


def run_patch(pt, C, n_steps, every):
    ts, rs, rm = [0.0], [], []
    r, s = pt.observe(); rs.append(r); rm.append(s)
    for t in range(every, n_steps + 1, every):
        pt.step(C, every)
        r, s = pt.observe(); rs.append(r); rm.append(s); ts.append(t * DT_HALF)
    return np.array(ts), np.array(rs), np.array(rm)


def dirac_rms(t, sigma, m, c=CP, n=200000, seed=0):
    """rms(t) of a Gaussian packet with v = c^2 k / sqrt(m^2 + c^2 k^2) (ballistic)."""
    k = np.random.default_rng(seed).normal(scale=1 / (np.sqrt(2) * sigma), size=(n, 3))
    v2 = np.mean(((c ** 2 * k) ** 2).sum(1) / (m ** 2 + c ** 2 * (k ** 2).sum(1)))
    return np.sqrt(3 * sigma ** 2 / 2 + v2 * t ** 2)   # |psi|^2 has width sigma/sqrt2


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    out = {}
    models, _ = model_generators()
    cands = candidates()
    H = cands['on-site u_d.s']
    VODD = cands['180° i u_d.s, K+K†']
    rng = np.random.default_rng(5)
    kr = rng.normal(size=(20, 3))
    kr *= (rng.uniform(0.05, 1.5, 20) / np.linalg.norm(kr, axis=1))[:, None]

    # ── A. symmetries ───────────────────────────────────────────────────────
    print("A. symmetries of the mass terms (P, PT = Theta of S4, T = Pi Theta, chirality H)")
    for name, V in (('mu Pi', PI), ('parity-odd 180° i u_d.s', VODD), ('on-site helicity', H)):
        s = sym_report(V, H)
        print(f"   {name:26s} P {s['P']:.0e}  PT {s['PT']:.0e}  T {s['T']:.0e}  [H,V] {s['H']:.0e}")
    for mname, G in models.items():
        tf = [sym_frame_T(G, V, 0.3, 0.1, kr[0]) for V in (0 * PI, PI, VODD)]
        ekk = max(np.abs(bands_C(k, expm(1j * (0.3 * G + 0.1 * VODD)))[0]
                         - bands_C(-k, expm(1j * (0.3 * G + 0.1 * VODD)))[0]).max() for k in kr)
        print(f"   {mname}: symmetric frame T W(k)* T^-1 = W(-k)^-1: none {tf[0]:.0e}, "
              f"Pi {tf[1]:.0e}, parity-odd {tf[2]:.2f};  [G,H] {np.abs(G @ H - H @ G).max():.0e}, "
              f"{{Pi,H}} {np.abs(PI @ H + H @ PI).max():.0e};  E(k)=E(-k) with parity-odd mass {ekk:.0e}")
    print(f"   parity-odd mass = Pi (iH): {np.abs(VODD - PI @ (1j * H)).max():.0e}")
    theta = np.linspace(0, np.pi, 7)
    th_dev = {}
    for mname, G in models.items():
        ref = None
        devs = []
        for th in theta:
            V = np.cos(th) * PI + np.sin(th) * VODD
            X = expm(0.5j * th * H)
            C = expm(1j * (0.3 * G + 0.1 * V))
            E = np.array([bands_C(k, C)[0] for k in kr])
            ref = E if ref is None else ref
            devs.append((np.abs(E - ref).max(), np.abs(X.conj().T @ PI @ X - V).max()))
        th_dev[mname] = np.array(devs)
        print(f"   {mname}: mu(cos th Pi + sin th Pi iH), th = 0..pi: max |E(th) - E(0)| over "
              f"20 k = {th_dev[mname][:, 0].max():.1e};  X^dag Pi X = V to {th_dev[mname][:, 1].max():.0e}")
    out['theta'], out['theta_dev_min'] = theta, th_dev['minimal SU(2)']
    weyl = []
    for mname, G in models.items():
        w, U = np.linalg.eigh(H)
        for s in (1, -1):
            B = U[:, np.isclose(w, s)]
            ev, EV = np.linalg.eigh(B.conj().T @ G @ B)
            for lv in np.unique(np.round(ev, 6)):
                Pn = EV[:, np.isclose(ev, lv, atol=1e-6)]
                if Pn.shape[1] != 2:
                    continue
                Pf = B @ Pn
                v = np.array([[np.real(np.trace(Pf.conj().T @ D @ Pf @ sg)) / 2 for sg in (SX, SY, SZ)]
                              for D in DIRV])
                weyl.append((mname, s, -lv / DT_HALF, np.sign(np.linalg.det(v)),
                             abs(np.linalg.det(v)) ** (1 / 3)))
    for r in weyl:
        print(f"   Weyl node: {r[0]:14s} helicity {r[1]:+d}  E/eps = {r[2]:+.3f}  chirality {r[3]:+.0f}  "
              f"|det v|^(1/3) = {r[4]:.4f}")
    print(f"   [{time.time()-t0:.0f}s]")

    # ── B. 2D ───────────────────────────────────────────────────────────────
    U3 = np.linalg.matrix_power(Z.S, 3)
    print("B. 2D Z12 walker: T = S^3 K")
    print(f"   T^2 = S^3 (S^3)* = R6 (-1 on spin sector): {np.abs(U3 @ U3.conj() - Z.R6).max():.0e};  "
          f"G0 T-even {np.abs(U3 @ Z.G0.conj() @ U3.T - Z.G0).max():.0e};  "
          f"G3 (chiral reversal, S2 mass) T-odd {np.abs(U3 @ Z.G3.conj() @ U3.T + Z.G3).max():.0e}")
    eps2 = 0.3
    Ch = expm(0.5j * eps2 * Z.G0)
    ph2 = lambda kx, ky: np.exp(-1j * (kx * Z.MOVES_PHYS[Z.HEAD, 0] + ky * Z.MOVES_PHYS[Z.HEAD, 1]))
    W2 = lambda kx, ky: Ch @ np.diag(ph2(kx, ky)) @ Ch
    print(f"   symmetric frame T W(k)* T^-1 = W(-k)^-1: "
          f"{np.abs(U3 @ W2(0.3, -0.2).conj() @ U3.T - np.linalg.inv(W2(-0.3, 0.2))).max():.0e}")
    E2, _, _ = Z.spin_bands(0, 0, expm(1j * eps2 * Z.G0))
    sl2 = (Z.spin_bands(1e-4, 0, expm(1j * eps2 * Z.G0))[0] - E2) / 1e-4
    print(f"   spin sector at k=0: E = {np.round(E2, 4).tolist()}, slopes {np.round(sl2, 4).tolist()} "
          f"(two Kramers pairs are cones with c/2 = {C_LIGHT/2:.4f})")
    rng2 = np.random.default_rng(7)
    res2 = {}
    for sign, name in ((1, 'T-even'), (-1, 'T-odd')):
        sp = []
        for _ in range(500):
            A = rng2.normal(size=(12, 12)) + 1j * rng2.normal(size=(12, 12))
            V = (A + A.conj().T) / 2
            V = (V + Z.R6 @ V @ Z.R6.T) / 2
            V = (V + sign * U3 @ V.conj() @ U3.T) / 2
            V /= np.abs(np.linalg.eigvalsh(V)).max()
            E, _, _ = Z.spin_bands(0, 0, expm(1j * (eps2 * Z.G0 + 0.05 * V)))
            sp.append(np.sort(E)[1::2] - np.sort(E)[0::2])
        sp = np.array(sp)
        res2[name] = sp
        print(f"   500 random {name} terms (strength 0.05, superselection kept): largest splitting of "
              f"the three k=0 pairs = {sp.max():.1e}, smallest = {sp.min(axis=1).min():.1e}")
    out['split2d_even'], out['split2d_odd'] = res2['T-even'].max(1), res2['T-odd'].min(1)
    print(f"   [{time.time()-t0:.0f}s]")

    # ── C. background ───────────────────────────────────────────────────────
    print("C. common background -alpha k^2")
    G = models['minimal SU(2)']
    G2O = models['2O register']
    W90, W120 = cands['90° transport, K+K†'], cands['120° transport, K+K†']
    for mname, Gm in models.items():
        lv = max(fourfold_levels(Gm))
        print(f"   {mname}, lowest Dirac level: alpha*eps = "
              + ", ".join(f"{alpha_eps(Gm, e, lv):.4f} (eps={e})" for e in (0.05, 0.1, 0.3)))
    lam_scan = np.linspace(-1.0, 0.35, 28)
    scan = {}
    for wname, W in (('90°', W90), ('120°', W120)):
        s = W.copy()
        print(f"   {wname} transport term: symmetries " + str({k: f'{v:.0e}' for k, v in sym_report(W, H).items()})
              + f", [G_min, W] {np.abs(G @ W - W @ G).max():.2f}, [G_2O, W] {np.abs(G2O @ W - W @ G2O).max():.2f}")
        rows = []
        for l in lam_scan:
            Gc = G + l * W
            rows.append([alpha_eps(Gc, 0.3, max(fourfold_levels(Gc))),
                         alpha_eps(Gc, 0.3, min(fourfold_levels(Gc))),
                         alpha_eps(Gc, 0.1, min(fourfold_levels(Gc)))])
        scan[wname] = np.array(rows)
    out['lam_scan'], out['scan90'], out['scan120'] = lam_scan, scan['90°'], scan['120°']
    for e in (0.05, 0.3):
        lv2 = max(fourfold_levels(G2O + 2.0 * W90))
        print(f"   2O + 2.0 eps W90 (commutes with G_2O), lowest level: alpha*eps = "
              f"{alpha_eps(G2O + 2.0 * W90, e, lv2):.4f} at eps = {e}")
    pick = lambda Gc: min(fourfold_levels(Gc))
    lam_star = brentq(lambda l: alpha_eps(G + l * W90, 0.1, pick(G + l * W90)), -0.6, -0.2, xtol=1e-12)
    GC = G + lam_star * W90
    LVC = pick(GC)
    print(f"   compensated interior Dirac point: lambda* = {lam_star:+.5f} (term lambda* eps W90); "
          f"level E/eps = {-LVC/DT_HALF:+.4f}; alpha*eps = "
          + ", ".join(f"{alpha_eps(GC, e, LVC):+.1e} (eps={e})" for e in (0.05, 0.1, 0.3)))
    w, V = np.linalg.eigh(GC)
    P = V[:, np.isclose(w, LVC, atol=1e-7)]
    Gam = [P.conj().T @ D @ P for D in DIRV]
    cl = np.sqrt(np.trace(Gam[0] @ Gam[0]).real / 4)
    acg = max(np.abs(Gam[i] @ Gam[j] + Gam[j] @ Gam[i] - 2 * (i == j) * cl ** 2 * np.eye(4)).max()
              for i in range(3) for j in range(3))
    Mp = P.conj().T @ PI @ P
    print(f"   Dirac slope {cl:.5f} (c/3 = {CP:.5f}), Clifford {acg:.0e}; P^dag Pi P = "
          f"{np.round(np.linalg.eigvalsh(Mp), 4).tolist()}, anticommutes {max(np.abs(Mp @ g + g @ Mp).max() for g in Gam):.0e}")
    comp_rows = []
    for gname, Gm, lvm in (('plain lowest', G, max(fourfold_levels(G))), ('compensated', GC, LVC)):
        for mu in (0.015, 0.03, 0.15):
            eps = 0.3
            C = expm(1j * (eps * Gm + mu * PI))
            idx = level_idx(C, -eps * lvm / DT_HALF)
            E0 = bands_C(np.zeros(3), C)[0][idx]
            m = 0.5 * (E0[2] - E0[1]); ec = 0.5 * (E0[2] + E0[1])
            h = 1e-3
            E1 = bands_C(np.array([h, 0, 0]), C)[0]
            E1 = np.sort(E1[np.argsort(np.abs(E1 - ec))[:4]])
            cu = 2 * (E1 - E0) / h ** 2
            kram = max(np.abs(np.diff(bands_C(k, C)[0])[0::2]).max() for k in kr)
            vmax = max(np.linalg.norm([group_velocity(u) for u in bands_C(k, C)[1].T], axis=1).max()
                       for k in kr)
            comp_rows.append((gname, mu, (E0[2] - E0[1]) / mu, cu[0], cu[3], CP ** 2 / m, kram, vmax))
            print(f"   {gname:12s} eps=0.3 mu={mu:<5}: gap/mu {(E0[2]-E0[1])/mu:.5f}; d2E/dk2 lower "
                  f"{cu[0]:+.3f} upper {cu[3]:+.3f} (Dirac ±{CP**2/m:.3f}); Kramers {kram:.0e}; "
                  f"max|v_g| {vmax:.3f}")
    out['comp_rows'] = np.array(comp_rows, dtype=object)
    # dispersion curves of the upper doublet for the figure
    kl = np.linspace(0, 0.6, 31)
    for gname, Gm, lvm in (('plain', G, max(fourfold_levels(G))), ('comp', GC, LVC)):
        C = expm(1j * (0.3 * Gm + 0.03 * PI))
        idx = level_idx(C, -0.3 * lvm / DT_HALF)
        E0 = bands_C(np.zeros(3), C)[0][idx]
        ec = 0.5 * (E0[2] + E0[1])
        curves = []
        for d in ([1, 0, 0], [1, 1, 1]):
            v = np.array(d, float) / np.linalg.norm(d)
            cc = []
            for k in kl:
                E = bands_C(k * v, C)[0]
                cc.append(np.sort(E[np.argsort(np.abs(E - ec))[:4]]) - ec)
            curves.append(cc)
        out[f'disp_{gname}'] = np.array(curves)
        out[f'disp_{gname}_m'] = 0.5 * (E0[2] - E0[1])
    out['disp_k'] = kl
    out['lam_star'] = lam_star
    print(f"   [{time.time()-t0:.0f}s]")

    # ── D. dynamics in the Dirac regime ─────────────────────────────────────
    print("D. exact k-patch dynamics")
    # validation against S4 real space (81^3): eps 0.3, mu 0.15, sigma 4
    Cm = expm(1j * (0.3 * G + 0.15 * PI))
    pt = Patch([0, 0, 0], 4.0, 40); pt.band_state(Cm, 2)
    t, r, rms = run_patch(pt, Cm, 50, 2)
    s_rest = np.polyfit(t[t > 12], rms[t > 12], 1)[0]
    pt = Patch([0.2, 0, 0], 4.0, 40); pt.band_state(Cm, 2)
    t, r, rms = run_patch(pt, Cm, 40, 2)
    v_b = np.polyfit(t, r[:, 0], 1)[0]
    print(f"   validation vs S4 81^3 real space: rest rms growth {s_rest:.4f} (S4: 0.051), "
          f"boost k=0.2 v {v_b:+.4f} (S4: -0.088)  [{time.time()-t0:.0f}s]")
    out['val'] = np.array([s_rest, v_b])

    EPS, MU, SIG, MP = 0.3, 0.03, 40.0, 40
    dyn = {}
    for gname, Gm, lvm in (('plain', G, max(fourfold_levels(G))), ('comp', GC, LVC)):
        C = expm(1j * (EPS * Gm + MU * PI))
        E0all = bands_C(np.zeros(3), C)[0]
        idx = level_idx(C, -EPS * lvm / DT_HALF)
        up = idx[2]                                       # upper massive doublet (bands up, up+1)
        m = 0.5 * (E0all[idx[2]] - E0all[idx[1]])
        print(f"   [{gname}] eps={EPS} mu={MU} (mu/eps={MU/EPS:.2f}) m={m:.4f}, sigma={SIG} "
              f"(m sigma/c' = {m*SIG/CP:.1f}), patch {MP}^3, box {2*np.pi/(2*5/SIG/MP):.0f}")
        # rest
        pt = Patch([0, 0, 0], SIG, MP); pt.band_state(C, up)
        t, r, rms = run_patch(pt, C, 800, 20)
        pred = dirac_rms(t, SIG, m)
        print(f"      rest: drift {np.abs(r[-1]).max():.1e}; rms {rms[0]:.2f} -> {rms[-1]:.2f} at t={t[-1]:.0f}; "
              f"Dirac {pred[0]:.2f} -> {pred[-1]:.2f}  [{time.time()-t0:.0f}s]")
        dyn[f'{gname}_rest'] = (t, rms, pred)
        # boosts
        kb = np.array([0.025, 0.05, 0.1, 0.2])
        vs, vg = [], []
        for kk in kb:
            k0 = np.array([kk, 0, 0])
            pt = Patch(k0, SIG, MP); pt.band_state(C, up)
            t, r, rms = run_patch(pt, C, 300, 20)
            vs.append(np.polyfit(t, r[:, 0], 1)[0])
            vg.append(group_velocity(bands_C(k0, C)[1][:, up])[0])
        vd = CP ** 2 * kb / np.hypot(m, CP * kb)
        print("      boosts k = " + ", ".join(f"{k}: v {a:+.4f} (band {b:+.4f}, Dirac {c:+.4f})"
                                             for k, a, b, c in zip(kb, vs, vg, vd))
              + f"  [{time.time()-t0:.0f}s]")
        dyn[f'{gname}_boost'] = (kb, np.array(vs), np.array(vg), vd)
        # Zitterbewegung
        k0 = np.array([0.03, 0, 0])
        Ek, Uk = bands_C(k0, C)
        Uk = Uk / np.linalg.norm(Uk, axis=0)
        u = (Uk[:, idx[1]] + Uk[:, idx[2]]) / np.sqrt(2)
        pt = Patch(k0, SIG, MP); pt.plain_state(u)
        t, r, rms = run_patch(pt, C, 600, 2)
        om_pred = Ek[idx[2]] - Ek[idx[1]]
        om, g_, A_ = fit_damped(t, r[:, 1], om_pred)
        print(f"      Zitterbewegung k0=0.03: omega fit {om:.4f}, band splitting {om_pred:.4f}, Dirac "
              f"2 sqrt(m^2+(c/3 k)^2) = {2*np.hypot(m, CP*0.03):.4f}; amplitude {A_:.2f} "
              f"(Dirac ~ c'/(2E) = {CP/(2*np.hypot(m, CP*0.03)):.2f})  [{time.time()-t0:.0f}s]")
        dyn[f'{gname}_zb'] = (t, r[:, 1], om, om_pred)
    for key, val in dyn.items():
        for j, a in enumerate(val):
            out[f'dyn_{key}_{j}'] = np.asarray(a)
    out['EPS'], out['MU'], out['SIG'] = EPS, MU, SIG

    np.savez(data_path('phase_bit_s4_open.npz'), **out)
    print(f"saved data/phase_bit_s4_open.npz  [{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()
