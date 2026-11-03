#!/usr/bin/env python3
"""
Paper 3, third follow-up to Experiment S4 — gravity on the clean particle,
Z2 of the domain wall, gauge field and anomaly
=========================================================================

A.  Gravity (Exp. 1b) on the clean 2O* particle (2O + lambda*(eps) eps W90).
    Vacuum counterterm: the Dirac point is put at quasi-energy 0 (otherwise the
    Floquet offset of the quartet gravitates).  Clock lapse C(r) = expm(i N(r) X)
    vs full lapse H -> (N H + H N)/2.
    - band level: R' = E0/(m* c'^2) (c' = c/3, the particle's own limiting
      speed) for particle and antiparticle, models with and without the
      background compensation; kappa(k) of the clock lapse vs (m/E)^2.
    - x-chain (FCC layers along x, Bloch in y,z), linear lapse N = 1 + g x:
      transverse acceleration of rest packets, massive and massless beams.
B.  Z2 in practice: spectral flow of the mass-sign domain wall in a z-slab,
    single quartet (2O*) vs two quartets (2O at lambda = -2), with and without
    a random T-even perturbation at one wall.
C.  Gauge field: Landau gauge A = (0, B x, 0) on the x-chain.  [H, W_B] = 0,
    and the lowest Landau level of each Weyl node per helicity sector.
"""

import functools
import os
import sys
import time

import numpy as np
from scipy.linalg import expm, schur, eigh
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import MOVES_PHYS, DT_HALF, FCC_STEPS, LAT_A
from phase_bit_mass_s4 import PI, model_generators, candidates, bands_C
from phase_bit_s4_open import CP, T_TRUE, fourfold_levels, alpha_eps
from phase_bit_s4_open2 import slab_T, LAM_STAR
from paths import data_path

print = functools.partial(print, flush=True)
EPS = 0.3


def x_chain(Cs, kyz, Nx, B=0.0, xs=None, periodic=True):
    """FCC layers along x; Bloch (ky, kz); optional Landau-gauge field A_y = B x."""
    D = 24 * Nx
    T = np.zeros((D, D), complex)
    for l in range(Nx):
        for d in range(12):
            l2 = l + FCC_STEPS[d, 0]
            if periodic:
                l2 %= Nx
            elif l2 < 0 or l2 >= Nx:
                continue
            ky = kyz[0] + (B * (xs[l] + 0.5 * MOVES_PHYS[d, 0]) if B else 0.0)
            ph = np.exp(-1j * (ky * MOVES_PHYS[d, 1] + kyz[1] * MOVES_PHYS[d, 2]))
            T[24 * l2 + 2 * d:24 * l2 + 2 * d + 2, 24 * l:24 * l + 24] += ph * Cs[l][2 * d:2 * d + 2, :]
    return T


def main():
    t0 = time.time()
    out = {}
    models, _ = model_generators()
    cands = candidates()
    H = cands['on-site u_d.s']
    W90 = cands['90° transport, K+K†']
    G, G2 = models['minimal SU(2)'], models['2O register']
    lvf = lambda l: 2 * np.sqrt(2) + l * np.sqrt(2)
    lam = brentq(lambda l: alpha_eps(G2 + l * W90, EPS, lvf(l)), -6, -2.2, xtol=1e-11)
    print(f"lambda*(eps={EPS}) = {lam:.4f}")

    def X_of(name, mu):
        if name == '2O*':
            Gm, lv = G2 + lam * W90, lvf(lam)
        elif name == '2O plain':
            Gm, lv = G2, max(fourfold_levels(G2))
        else:
            Gm = G + LAM_STAR * W90
            lv = min(fourfold_levels(Gm))
        return EPS * Gm + mu * PI - EPS * lv * np.eye(24)       # vacuum counterterm

    # ── A1. band level ─────────────────────────────────────────────────────
    print("A1. R' = E0/(m* c'^2) with the Dirac point at E = 0 (clock or full lapse: same at k = 0)")
    rowsR = []
    for name in ('2O plain', 'SU2 comp', '2O*'):
        for mu in (0.015, 0.03, 0.15):
            C = expm(1j * X_of(name, mu))
            E0 = bands_C(np.zeros(3), C)[0]
            q = np.sort(E0[np.argsort(np.abs(E0))[:4]])
            h = 1e-3
            E1 = bands_C(np.array([h, 0, 0]), C)[0]
            E1 = np.sort(E1[np.argsort(np.abs(E1))[:4]])
            cu = 2 * (E1 - q) / h ** 2
            Rp, Ra = q[3] * cu[3] / CP ** 2, q[0] * cu[0] / CP ** 2
            rowsR.append((name, mu, Rp, Ra))
            print(f"   {name:9s} mu/eps={mu/EPS:.2f}: R' particle {Rp:.4f}, antiparticle {Ra:.4f}")
    out['A_R'] = np.array(rowsR, dtype=object)
    # without counterterm (raw Floquet offset) for 2O*
    Xr = EPS * (G2 + lam * W90) + 0.03 * PI
    C = expm(1j * Xr)
    E0 = bands_C(np.zeros(3), C)[0]
    Ec = -EPS * lvf(lam) / DT_HALF
    q = np.sort(E0[np.argsort(np.abs(E0 - Ec))[:4]])
    print(f"   without counterterm: quartet at E = {Ec:+.3f}, upper doublet E0 = {q[3]:+.3f} vs mass {q[3]-Ec:.3f} "
          f"-> R' would be {q[3]/(q[3]-Ec):.1f} (the Floquet offset gravitates)")
    kap = []
    for mu in (0.03, 0.15):
        X = X_of('2O*', mu)

        def Eup(k, N):
            E = bands_C(np.array([k, 0, 0]), expm(1j * N * X))[0]
            return np.sort(E[np.argsort(np.abs(E))[:4]])[3]
        m = Eup(0, 1)
        for k in (0.1, 0.3, 0.6, 1.0):
            E = Eup(k, 1)
            kp = (Eup(k, 1 + 1e-4) - Eup(k, 1 - 1e-4)) / 2e-4 / E
            kap.append((mu, k, kp, (m / E) ** 2))
    print("   clock-lapse coupling kappa(k) vs (m/E)^2: " +
          ", ".join(f"mu={a}, k={b}: {c_:.3f}/{d_:.3f}" for a, b, c_, d_ in kap))
    out['A_kappa'] = np.array(kap)
    print(f"   [{time.time()-t0:.0f}s]")

    # ── A2. chain: clock vs full lapse ─────────────────────────────────────
    print("A2. x-chain with N = 1 + g x: transverse acceleration")
    Nx, g, SIG = 160, 2e-3, 8.0
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    Nl = 1 + g * xs

    def packet(X, kyz, sel):
        C = expm(1j * X)
        E, U = bands_C(np.array([0.0, *kyz]), C)
        uref = U[:, sel(E)[0]]
        uref = uref / np.linalg.norm(uref)
        psi = np.zeros((Nx, 24), complex)
        for kx in 2 * np.pi * np.fft.fftfreq(Nx) / LAT_A:
            gk = np.exp(-SIG ** 2 * kx ** 2 / 2)
            if gk < 1e-8:
                continue
            Ek, Uk = bands_C(np.array([kx, *kyz]), C)
            Q, _ = np.linalg.qr(Uk[:, sel(Ek)])
            u = Q @ (Q.conj().T @ uref)
            psi += gk * np.exp(1j * kx * xs)[:, None] * (u / np.linalg.norm(u))[None, :]
        return (psi / np.linalg.norm(psi)).ravel()

    com = lambda psi: ((np.abs(psi.reshape(Nx, 24)) ** 2).sum(1) @ xs) / (np.abs(psi) ** 2).sum()
    accel = lambda t, x: 2 * np.polyfit(t, x, 2)[0]
    cases = [('massive, rest', 0.15, (0.0, 0.0)), ('massive, k_y=0.3', 0.15, (0.3, 0.0)),
             ('massive, k_y=0.6', 0.15, (0.6, 0.0)), ('massless, k_y=0.2', 0.0, (0.2, 0.0)),
             ('massless, k_y=0.4', 0.0, (0.4, 0.0))]
    rowsL = []
    for cname, mu, kyz in cases:
        X = X_of('2O*', mu)
        if mu > 0:
            sel = lambda E: np.sort(np.argsort(np.abs(E))[:4])[2:]
        else:
            kk = np.hypot(*kyz)
            sel = lambda E, kk=kk: np.sort(np.argsort(np.abs(E - CP * kk))[:2])
        psi0 = packet(X, kyz, sel)
        nst = 80
        ts = np.arange(nst + 1) * DT_HALF
        T = x_chain([expm(1j * n * X) for n in Nl], kyz, Nx)
        psi, xc = psi0.copy(), [com(psi0)]
        for _ in range(nst):
            psi = T @ psi
            xc.append(com(psi))
        Tm, Z = schur(x_chain([expm(1j * X)] * Nx, kyz, Nx), output='complex')
        E = -np.angle(np.diag(Tm)) / DT_HALF
        fw = 1 / (1 + np.exp((np.abs(E) - 1.2) / 0.05))
        Hs = (Z * (E * fw)) @ Z.conj().T
        Nd = np.repeat(Nl, 24)
        w, V = eigh(0.5 * (Nd[:, None] * Hs + Hs * Nd[None, :]))
        c0 = V.conj().T @ psi0
        xf = [com(V @ (np.exp(-1j * w * t) * c0)) for t in ts]
        del Tm, Z, Hs, V

        def Ek(kx, N=1.0):
            E_ = bands_C(np.array([kx, *kyz]), expm(1j * N * X))[0]
            return E_[sel(E_)].mean()
        kq = np.linspace(-3 / SIG, 3 / SIG, 41)
        wq = np.exp(-SIG ** 2 * kq ** 2)
        d2 = np.array([(Ek(k + 1e-3) - 2 * Ek(k) + Ek(k - 1e-3)) / 1e-6 for k in kq])
        Fc = np.array([(Ek(k, 1 + 1e-4) - Ek(k, 1 - 1e-4)) / 2e-4 for k in kq])
        Ff = np.array([Ek(k) for k in kq])
        r = (cname, accel(ts, np.array(xc)), -g * np.sum(wq * Fc * d2) / wq.sum(),
             accel(ts, np.array(xf)), -g * np.sum(wq * Ff * d2) / wq.sum())
        rowsL.append(r)
        print(f"   {cname:18s}: clock {r[1]:+.3e} (semicl. {r[2]:+.3e}); full {r[3]:+.3e} (semicl. {r[4]:+.3e}); "
              f"-g c'^2 = {-g*CP**2:+.3e}  [{time.time()-t0:.0f}s]")
    out['A_lapse'] = np.array(rowsL, dtype=object)
    out['A_g'] = g

    # ── B. Z2: spectral flow of the domain wall ────────────────────────────
    print("B. spectral flow of the mass-sign domain wall")
    MU, Nz, w_ = 0.05, 120, 3.0
    z = np.arange(Nz); z1, z2 = Nz / 4, 3 * Nz / 4
    prof = np.tanh((z - z1) / w_) * np.tanh((z2 - z) / w_)
    rng = np.random.default_rng(11)
    A = rng.normal(size=(24, 24)) + 1j * rng.normal(size=(24, 24))
    Vw = (A + A.conj().T) / 2
    Vw = (Vw + T_TRUE @ Vw.conj() @ T_TRUE.conj().T) / 2
    Vw /= np.abs(np.linalg.eigvalsh(Vw)).max()
    print(f"   random wall term: T-even {np.abs(T_TRUE @ Vw.conj() @ T_TRUE.conj().T - Vw).max():.0e}")
    kxs = np.array([0.0, 0.01, 0.02, 0.04, 0.07, 0.1, 0.14, 0.2])
    Eg = np.linspace(-0.8, 0.8, 33) * 2 * MU
    for mname, Gm, lv in (('2O* (one quartet)', G2 + lam * W90, lvf(lam)),
                          ('2O, lambda=-2 (two quartets)', G2 - 2 * W90, 0.0)):
        Ec = -EPS * lv / DT_HALF
        for delta in (0.0, 1.2 * MU):
            Vz = [MU * p * PI + (delta * Vw if abs(l - z1) <= 3 else 0) for l, p in zip(z, prof)]
            wall = []
            for kx in kxs:
                lamv, U = np.linalg.eig(slab_T(Gm, EPS, Vz, (kx, 0.0)))
                E = -np.angle(lamv) / DT_HALF - Ec
                s = np.abs(E) < 2 * MU
                wl = (np.abs(U[:, s]) ** 2).reshape(Nz, 24, -1).sum(1)
                wl /= wl.sum(0)
                wall.append(np.sort(E[s][wl[np.abs(z - z1) < 12].sum(0) > 0.6]))
            n0 = np.array([np.sum(wall[0] < e) for e in Eg])
            n1 = np.array([np.sum(wall[-1] < e) for e in Eg])
            flow = (n1 - n0) / 2
            key = f"{'single' if '2O*' in mname else 'double'}_{'pert' if delta else 'clean'}"
            out[f'B_{key}'] = np.array([np.pad(wv, (0, 12 - len(wv)), constant_values=np.nan)[:12] for wv in wall])
            out[f'B_{key}_flow'] = flow
            print(f"   {mname:28s} wall term {delta/MU:.1f}mu: wall-1 pairs at k=0: {len(wall[0])//2}; "
                  f"flow over the gap: {np.unique(flow).tolist()}; energies with zero flow {int((flow == 0).sum())}/{len(Eg)}  "
                  f"[{time.time()-t0:.0f}s]")
            print("      " + " | ".join(f"k={k}:" + ",".join(f"{e:+.3f}" for e in wv) for k, wv in zip(kxs, wall)))
    out['B_kx'], out['B_Eg'] = kxs, Eg

    # ── C. Landau levels ───────────────────────────────────────────────────
    print("C. Landau levels, A = (0, B x, 0), minimal SU(2), mu = 0")
    B, NxB = 0.05, 200
    xb = (np.arange(NxB) - NxB // 2) * LAT_A
    C = expm(1j * EPS * G)
    T0 = x_chain([C] * NxB, (0.0, 0.1), NxB, B=B, xs=xb, periodic=False)
    Hf = np.kron(np.eye(NxB), H)
    print(f"   [H, W_B] = {np.abs(Hf @ T0 - T0 @ Hf).max():.1e}")
    del T0, Hf
    wH, VH = np.linalg.eigh(H)
    lB = 1 / np.sqrt(B)
    nodes = (-6.928 * EPS, 2.309 * EPS)
    kzs = np.linspace(-0.2, 0.2, 9)
    ll = {}
    for s in (+1, -1):
        Ps = np.kron(np.eye(NxB), VH[:, np.isclose(wH, s)])
        for kz in kzs:
            Ts = Ps.conj().T @ x_chain([C] * NxB, (0.0, kz), NxB, B=B, xs=xb, periodic=False) @ Ps
            lamv, U = np.linalg.eig(Ts)
            E = -np.angle(lamv) / DT_HALF
            wl = (np.abs(U) ** 2).reshape(NxB, 12, -1).sum(1)
            cen = wl[np.abs(xb) < 3 * lB].sum(0) / wl.sum(0)
            good = (np.abs(lamv) > 0.999) & (cen > 0.8)
            for En in nodes:
                ll.setdefault((s, En), []).append(np.sort(E[good & (np.abs(E - En) < 0.12)] - En))
        print(f"   helicity {s:+d} done [{time.time()-t0:.0f}s]")
    for (s, En), rows in ll.items():
        lowest = []
        for kz, Es in zip(kzs, rows):
            lowest.append(Es[np.argmin(np.abs(Es))] if len(Es) else np.nan)
        lowest = np.array(lowest)
        ok = ~np.isnan(lowest) & (np.abs(kzs) <= 0.1)
        slope = np.polyfit(kzs[ok], lowest[ok], 1)[0] if ok.sum() > 2 else np.nan
        out[f'C_ll_{s:+d}_{En:+.3f}'] = lowest
        print(f"   helicity {s:+d}, node E = {En:+.3f}: branch nearest the node vs kz: "
              + " ".join(f"{v:+.3f}" for v in lowest) + f";  slope near kz=0 {slope:+.3f} (c/3 = {CP:.3f})")
    out['C_kz'] = kzs

    np.savez(data_path('phase_bit_s4_open3.npz'), **out)
    print(f"saved data/phase_bit_s4_open3.npz  [{time.time()-t0:.0f}s]")


def part_B2():
    """Fine k scan of the central wall branches (wall 1, T-even perturbation at wall 1).
    One quartet: the central pair is a Kramers crossing at k = 0 (protected).
    Two quartets: the two central branches meet away from k = 0, where nothing protects
    the crossing -> look for an avoided crossing (gap = Z2-trivial wall)."""
    t0 = time.time()
    models, _ = model_generators()
    cands = candidates()
    W90 = cands['90° transport, K+K†']
    G2 = models['2O register']
    lvf = lambda l: 2 * np.sqrt(2) + l * np.sqrt(2)
    lam = brentq(lambda l: alpha_eps(G2 + l * W90, EPS, lvf(l)), -6, -2.2, xtol=1e-11)
    MU, Nz, w_ = 0.05, 120, 3.0
    z = np.arange(Nz); z1, z2 = Nz / 4, 3 * Nz / 4
    prof = np.tanh((z - z1) / w_) * np.tanh((z2 - z) / w_)
    rng = np.random.default_rng(11)
    A = rng.normal(size=(24, 24)) + 1j * rng.normal(size=(24, 24))
    Vw = (A + A.conj().T) / 2
    Vw = (Vw + T_TRUE @ Vw.conj() @ T_TRUE.conj().T) / 2
    Vw /= np.abs(np.linalg.eigvalsh(Vw)).max()
    kf = np.linspace(0, 0.03, 13)
    res = {}
    print("B2. fine scan of the central wall branches (wall 1)")
    for tag, Gm, lv in (('single', G2 + lam * W90, lvf(lam)), ('double', G2 - 2 * W90, 0.0)):
        Ec = -EPS * lv / DT_HALF
        for delta in (1.2 * MU, 2.4 * MU):
            Vz = [MU * p * PI + (delta * Vw if abs(l - z1) <= 3 else 0) for l, p in zip(z, prof)]
            cen = []
            allw = []
            for kx in kf:
                lamv, U = np.linalg.eig(slab_T(Gm, EPS, Vz, (kx, 0.0)))
                E = -np.angle(lamv) / DT_HALF - Ec
                s = np.abs(E) < 2 * MU
                wl = (np.abs(U[:, s]) ** 2).reshape(Nz, 24, -1).sum(1)
                wl /= wl.sum(0)
                Ew = np.sort(E[s][wl[np.abs(z - z1) < 12].sum(0) > 0.6])
                allw.append(np.pad(Ew, (0, 8 - len(Ew)), constant_values=np.nan)[:8])
                mid = len(Ew) // 2
                cen.append(Ew[mid] - Ew[mid - 1] if len(Ew) >= 2 else np.nan)
            cen = np.array(cen)
            res[f'B2_{tag}_{delta/MU:.1f}'] = np.array(allw)
            res[f'B2_{tag}_{delta/MU:.1f}_cen'] = cen
            i = int(np.nanargmin(cen))
            print(f"   {tag:6s} wall term {delta/MU:.1f}mu: min spacing of the central branches {cen[i]:.2e} at k = {kf[i]:.4f}; "
                  f"spacing vs k: " + " ".join(f"{v:.4f}" for v in cen) + f"  [{time.time()-t0:.0f}s]")
    res['B2_k'] = kf
    path = data_path('phase_bit_s4_open3.npz')
    old = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    old.update(res)
    np.savez(path, **old)
    print("updated data/phase_bit_s4_open3.npz")


def part_C2():
    """theta stays unobservable in a magnetic field: [H, W_B] = 0, so exp(i theta H/2) still removes it."""
    models, _ = model_generators()
    c = candidates()
    G = models['minimal SU(2)']
    VODD, Wid = c['180° i u_d.s, K+K†'], c['90° identity, K+K†']
    Nx, B = 60, 0.05
    xb = (np.arange(Nx) - Nx // 2) * LAT_A
    res = {}
    print("C2. theta in a magnetic field (B = 0.05, open x-chain of 60 layers, ky = 0.1, kz = 0.13)")
    for extra, lab in ((0 * PI, 'sym'), (0.05 * Wid, 'brk')):
        ref, devs = None, []
        for th in np.linspace(0, np.pi, 5):
            C = expm(1j * (0.3 * G + 0.05 * (np.cos(th) * PI + np.sin(th) * VODD) + extra))
            lamv = np.linalg.eigvals(x_chain([C] * Nx, (0.1, 0.13), Nx, B=B, xs=xb, periodic=False))
            E, r = np.sort(np.angle(lamv)), np.sort(np.abs(lamv))
            ref = (E, r) if ref is None else ref
            devs.append(max(np.abs(E - ref[0]).max(), np.abs(r - ref[1]).max()))
        res[f'C2_{lab}'] = max(devs)
        print(f"   {'helicity-symmetric' if lab == 'sym' else '+ H-breaking mass (90° identity)':34s}: "
              f"max spectral change over theta in [0, pi] = {max(devs):.1e}")
    path = data_path('phase_bit_s4_open3.npz')
    old = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    old.update(res)
    np.savez(path, **old)


if __name__ == '__main__':
    parts = sys.argv[1:] or ['main', 'B2', 'C2']
    if 'main' in parts:
        main()
    if 'B2' in parts:
        part_B2()
    if 'C2' in parts:
        part_C2()
