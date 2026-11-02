#!/usr/bin/env python3
"""
Paper 3, second follow-up to Experiment S4 — the new open points
================================================================

A.  2O register with E <-> -E AND no k^2 background.  The 90 deg spin-transport
    coupling W90 is a 2O term, commutes with G_2O and only shifts its four
    levels.  The background of the Dirac quartet is level repulsion from the
    two 8-fold levels (velocity weights 2 and 6); moving them to opposite sides
    cancels it at lambda* ~ -5 (second order: exactly -5).  Single quartet, no
    doubling; residual O(eps^2) Floquet drift.  (At lambda = -2 the two quartets
    meet at E = 0; there only the flavour AVERAGE is background-free.)
B.  theta becomes observable once helicity is broken: with a spin-identity
    (H-breaking) mass nu W_id next to mu(cos th Pi + sin th Pi iH), the gap is
    4 mu |1 + e^{i th}| ("twisted mass").
C.  Mass domain walls in a z-slab (exact, Bloch in x, y): sign flip of the real
    mass (T kept everywhere) vs chiral rotation 0 -> pi (T broken inside the
    wall): count and disperse the in-gap wall states.
D.  Zitterbewegung amplitude with a well-defined initial spinor (helicity +1
    state of the Dirac space at k = 0): simulation vs band theory
    (projector formula) vs the effective 4x4 Dirac model.
"""

import functools
import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import MOVES_PHYS, DT_HALF, FCC_STEPS, LAT_A
from phase_bit_mass_s4 import PI, THETA_U, DIRV, SZ, model_generators, candidates, bands_C, group_velocity
from phase_bit_s4_open import (CP, T_TRUE, fourfold_levels, level_idx, alpha_eps, Patch, run_patch)
from paths import data_path

print = functools.partial(print, flush=True)
LAM_STAR = -0.40827                     # S4+ compensation of the interior Dirac point


def slab_T(G, eps, Vz, kpar):
    """Floquet operator of a periodic z-slab (layers of the FCC lattice), Bloch kpar."""
    Nz = len(Vz)
    T = np.zeros((24 * Nz, 24 * Nz), complex)
    ph = np.exp(-1j * (MOVES_PHYS[:, :2] @ np.asarray(kpar)))
    cache = {}
    for l in range(Nz):
        key = Vz[l].tobytes()
        if key not in cache:
            cache[key] = expm(1j * (eps * G + Vz[l]))
        C = cache[key]
        for d in range(12):
            l2 = (l + FCC_STEPS[d, 2]) % Nz
            T[24 * l2 + 2 * d:24 * l2 + 2 * d + 2, 24 * l:24 * l + 24] += ph[d] * C[2 * d:2 * d + 2, :]
    return T


def main():
    t0 = time.time()
    out = {}
    models, _ = model_generators()
    cands = candidates()
    H = cands['on-site u_d.s']
    VODD = cands['180° i u_d.s, K+K†']
    W90 = cands['90° transport, K+K†']
    G, G2 = models['minimal SU(2)'], models['2O register']
    rng = np.random.default_rng(0)
    kr = rng.normal(size=(20, 3))
    kr *= (rng.uniform(0.05, 1.5, 20) / np.linalg.norm(kr, axis=1))[:, None]

    # ── A. self-conjugate Dirac point in the 2O register ───────────────────
    print("A. 2O register: G_2O + lambda W90")
    print(f"   [G_2O, W90] = {np.abs(G2 @ W90 - W90 @ G2).max():.0e}; W90 keeps P, T, PT, H: "
          f"{max(np.abs(PI @ W90 @ PI - W90).max(), np.abs(T_TRUE @ W90.conj() @ T_TRUE.conj().T - W90).max(), np.abs(H @ W90 - W90 @ H).max()):.0e}")
    w, V = np.linalg.eigh(G2)
    top = V[:, np.isclose(w, w.max())]
    for lv in np.unique(np.round(w, 6)):
        P = V[:, np.isclose(w, lv, atol=1e-6)]
        print(f"   G_2O level {lv:+.4f} (x{P.shape[1]}): W90 = {np.round(np.linalg.eigvalsh(P.conj().T @ W90 @ P), 4)[0]:+.4f}; "
              f"|v|^2 coupling to top quartet {sum(np.linalg.norm(P.conj().T @ D @ top) ** 2 for D in DIRV):.3f}")
    wA = sum(np.linalg.norm(V[:, np.isclose(w, -np.sqrt(2))].conj().T @ D @ top) ** 2 for D in DIRV)
    wB = sum(np.linalg.norm(V[:, np.isclose(w, np.sqrt(2))].conj().T @ D @ top) ** 2 for D in DIRV)
    lvf = lambda l: 2 * np.sqrt(2) + l * np.sqrt(2)          # the top quartet stays an eigenspace
    # second order (BCH limit): alpha ~ wA/(G_D - G_A) + wB/(G_D - G_B), levels shift with W90 = -/+ sqrt2
    bch = lambda l: wA / (lvf(l) - (-np.sqrt(2) + l * np.sqrt(2))) + wB / (lvf(l) - (np.sqrt(2) - l * np.sqrt(2)))
    lam_bch = brentq(bch, -8, -2.5)
    lams = np.linspace(-8.0, 1.0, 91)
    al = np.array([alpha_eps(G2 + l * W90, 0.05, lvf(l)) if (abs(l + 0.5) > 0.06 and abs(l + 2) > 0.06) else np.nan
                   for l in lams])
    out['A_lams'], out['A_alpha'] = lams, al
    print("   alpha*eps of the top quartet vs lambda (eps = 0.05): "
          + ", ".join(f"{l:+.1f}: {a:+.4f}" for l, a in zip(lams[::10], al[::10])))
    f = lambda l, e: alpha_eps(G2 + l * W90, e, lvf(l))
    lam2 = brentq(lambda l: f(l, 0.02), -6, -4, xtol=1e-11)
    print(f"   zero: lambda* = {lam2:.4f} (eps -> 0), second-order estimate {lam_bch:.4f}; alpha*eps at lambda*: "
          + ", ".join(f"{f(lam2, e):+.1e} (eps={e})" for e in (0.02, 0.05, 0.1, 0.2, 0.3))
          + f";  without W90: {f(0.0, 0.1):.3f}")
    vals, cnt = np.unique(np.round(np.linalg.eigvalsh(G2 + lam2 * W90), 4), return_counts=True)
    print(f"   G_2O + lambda* W90: levels {vals.tolist()} x {cnt.tolist()}  (single Dirac quartets at +-{abs(lvf(lam2)):.3f}, no doubling)")
    # the doubled point lambda = -2 for comparison: only the flavour average is background-free
    GD = G2 - 2 * W90
    Ft = top @ top.conj().T
    C = expm(1j * (0.3 * GD + 0.15 * PI))
    E, U = bands_C(np.array([0.4, 0, 0]), C)
    U = U / np.linalg.norm(U, axis=0)
    i8 = np.argsort(np.abs(E))[:8]
    et = [E[i] for i in i8 if np.real(np.vdot(U[:, i], Ft @ U[:, i])) > 0.5]
    print(f"   lambda = -2 (top and bottom quartet meet at E=0, 8-fold): mean of all 8 bands 0 by E<->-E, but the "
          f"top-flavour bands at k=0.4 are {np.round(sorted(et), 4).tolist()} -> its own centre {np.mean(et):+.4f} (background survives per flavour)")
    rowsA = []
    lam_eps = {}
    for eps in (0.1, 0.3):
        lam_e = brentq(lambda l: f(l, eps), -6, -2.2, xtol=1e-11)
        lam_eps[eps] = lam_e
        Gm = G2 + lam_e * W90
        lv = lvf(lam_e)
        w2, V2 = np.linalg.eigh(Gm)
        P = V2[:, np.isclose(w2, lv, atol=1e-7)]
        Gam = [P.conj().T @ D @ P for D in DIRV]
        acg = max(np.abs(Gam[i] @ Gam[j] + Gam[j] @ Gam[i] - 2 * (i == j) * CP ** 2 * np.eye(4)).max()
                  for i in range(3) for j in range(3))
        for mu in (0.1 * eps, 0.5 * eps):
            C = expm(1j * (eps * Gm + mu * PI))
            Ec0 = -eps * lv / DT_HALF
            idx = level_idx(C, Ec0)
            E0 = bands_C(np.zeros(3), C)[0][idx]
            m = 0.5 * (E0[2] - E0[1]); ec = 0.5 * (E0[2] + E0[1])
            sym = max(np.abs(np.sort(bands_C(k, C)[0]) - np.sort(-bands_C(k, C)[0])).max() for k in kr)
            kram = max(np.abs(np.diff(bands_C(k, C)[0])[0::2]).max() for k in kr)
            h = 1e-3
            E1 = bands_C(np.array([h, 0, 0]), C)[0]
            E1 = np.sort(E1[np.argsort(np.abs(E1 - ec))[:4]])
            cu = 2 * (E1 - E0) / h ** 2
            dev = []
            for dvec in ([1, 0, 0], [1, 1, 1], [1, 2, 3]):
                v = np.array(dvec, float) / np.linalg.norm(dvec)
                for k in (0.1, 0.2, 0.4):
                    E = bands_C(k * v, C)[0]
                    q = np.sort(E[np.argsort(np.abs(E - ec))[:4]])
                    ref = np.hypot(m, CP * k)
                    dev.append(((q[3] - ec) / ref - 1, (ec - q[0]) / ref - 1))
            dev = np.array(dev)
            vmax = max(np.linalg.norm([group_velocity(u) for u in bands_C(k, C)[1].T], axis=1).max() for k in kr)
            rowsA.append((eps, lam_e, mu, (E0[2] - E0[1]) / mu, cu[0], cu[3], CP ** 2 / m, sym, kram, vmax))
            print(f"   eps={eps} lambda*(eps)={lam_e:.4f} mu={mu:.3f} (mu/eps={mu/eps:.1f}): Clifford c/3 {acg:.0e}; gap/mu {(E0[2]-E0[1])/mu:.5f}; "
                  f"E<->-E {sym:.0e}; Kramers {kram:.0e}; d2E/dk2 lower {cu[0]:+.3f} upper {cu[3]:+.3f} (Dirac +-{CP**2/m:.3f}); "
                  f"upper vs Dirac k=.1/.2/.4 [100] " + " ".join(f"{x:+.3f}" for x in dev[:3, 0])
                  + " [111] " + " ".join(f"{x:+.3f}" for x in dev[3:6, 0]) + " [123] " + " ".join(f"{x:+.3f}" for x in dev[6:, 0])
                  + f"; max|v_g| {vmax:.3f}")
    out['A_rows'] = np.array(rowsA)
    out['A_lam_star'], out['A_lam_bch'] = lam2, lam_bch
    kl = np.linspace(0, 0.6, 31)
    for tag, Gm, lv in (('2O', G2 + lam_eps[0.3] * W90, lvf(lam_eps[0.3])),
                        ('2O_plain', G2, max(fourfold_levels(G2))),
                        ('comp', G + LAM_STAR * W90, min(fourfold_levels(G + LAM_STAR * W90)))):
        C = expm(1j * (0.3 * Gm + 0.15 * PI))
        idx = level_idx(C, -0.3 * lv / DT_HALF)
        E0 = bands_C(np.zeros(3), C)[0][idx]
        Ec = 0.5 * (E0[1] + E0[2])
        cur = []
        for k in kl:
            E = bands_C(np.array([k, 0, 0]), C)[0]
            cur.append(np.sort(E[np.argsort(np.abs(E - Ec))[:4]]) - Ec)
        out[f'A_disp_{tag}'] = np.array(cur)
    out['A_kl'] = kl
    print(f"   [{time.time()-t0:.0f}s]")

    # ── B. theta with an H-breaking mass ───────────────────────────────────
    print("B. chiral angle theta next to an H-breaking (spin-identity) mass")
    th = np.linspace(0, 2 * np.pi, 25)
    gapsB = {}
    for mname, Gm in models.items():
        for wname in ('90° identity, K+K†', '60° identity, K+K†'):
            Wb = cands[wname]
            w, V = np.linalg.eigh(Gm)
            P = V[:, np.isclose(w, w.max())]
            a = np.real(np.trace(P.conj().T @ Wb @ P @ P.conj().T @ PI @ P)) / 4
            mu, nu = 0.05, 0.05 / abs(a)
            g = []
            for t in th:
                C = expm(1j * (0.3 * Gm + mu * (np.cos(t) * PI + np.sin(t) * VODD) + nu * np.sign(a) * Wb))
                E0 = bands_C(np.zeros(3), C)[0]
                g.append(E0[2] - E0[1])
            g = np.array(g)
            gapsB[(mname, wname)] = g
            print(f"   {mname:14s} + {wname:20s} ([H,W] {np.abs(H @ Wb - Wb @ H).max():.2f}, projected = {a:+.3f} Pi): "
                  f"max |gap - 4mu|1+e^(i th)|| / 4mu = {np.abs(g - 4 * mu * np.abs(1 + np.exp(1j * th))).max() / (4 * mu):.1e}")
    out['B_theta'] = th
    out['B_gap'] = gapsB[('2O register', '90° identity, K+K†')]
    print(f"   [{time.time()-t0:.0f}s]")

    # ── C. domain walls ────────────────────────────────────────────────────
    print("C. mass domain walls in a z-slab")
    eps, mu, w_, Nz = 0.3, 0.05, 3.0, 120
    z = np.arange(Nz); z1, z2 = Nz / 4, 3 * Nz / 4
    th_up = np.pi * 0.5 * (1 + np.tanh((z - z1) / w_)) + np.pi * 0.5 * (1 + np.tanh((z - z2) / w_))
    profiles = {
        'sign flip (T kept)': [mu * np.tanh((l - z1) / w_) * np.tanh((z2 - l) / w_) * PI for l in z],
        'chiral rotation 0→π→2π (T broken in wall)': [mu * (np.cos(t) * PI + np.sin(t) * VODD) for t in th_up],
    }
    GC = G + LAM_STAR * W90
    LVC = min(fourfold_levels(GC))
    Cb = expm(1j * (eps * GC + mu * PI))
    near = (np.abs(z - z1) < 12) | (np.abs(z - z2) < 12)
    kxs = np.array([0.0, 0.02, 0.04, 0.06])
    m = 2 * mu
    Ec = -eps * LVC / DT_HALF                     # gap centre at k = 0 (no background)
    for pname, Vz in profiles.items():
        rows = []
        for kx in kxs:
            lam, U = np.linalg.eig(slab_T(GC, eps, Vz, (kx, 0.0)))
            E = -np.angle(lam) / DT_HALF
            sel = np.abs(E - Ec) < 0.9 * m
            wl = (np.abs(U[:, sel]) ** 2).reshape(Nz, 24, -1).sum(1)
            loc = wl[near].sum(0) / wl.sum(0)
            Ew = np.sort(E[sel][loc > 0.6])
            rows.append((kx, -m, m, Ew - Ec))
            print(f"   {pname:42s} kx={kx:.2f}: states with |E-Ec| < 0.9m localised at the walls "
                  f"(weight > 0.6): {len(Ew)}: {np.round(Ew - Ec, 4).tolist()}  "
                  f"[{int((loc <= 0.6).sum())} delocalised in window]")
        key = 'flip' if 'sign' in pname else 'chiral'
        out[f'C_{key}_kx'] = kxs
        out[f'C_{key}_gap'] = np.array([[r[1], r[2]] for r in rows])
        allw = [r[3] for r in rows]
        nmax = max(len(a) for a in allw)
        out[f'C_{key}_wall'] = np.array([np.pad(a, (0, nmax - len(a)), constant_values=np.nan) for a in allw])
        print(f"   [{time.time()-t0:.0f}s]")

    # ── D. Zitterbewegung amplitude ────────────────────────────────────────
    print("D. Zitterbewegung amplitude, initial state = helicity +1, max S_z in the Dirac space")
    SZT = np.kron(np.eye(12), SZ)
    EPS, MU, SIG = 0.3, 0.03, 40.0
    for tag, Gm, lv in (('plain', G, max(fourfold_levels(G))), ('comp', GC, LVC)):
        C = expm(1j * (EPS * Gm + MU * PI))
        idx = level_idx(C, -EPS * lv / DT_HALF)
        w, V = np.linalg.eigh(Gm)
        P = V[:, np.isclose(w, lv, atol=1e-7)]
        hh, hv = np.linalg.eigh(P.conj().T @ H @ P)
        B = hv[:, hh > 0]
        s, sv = np.linalg.eigh(B.conj().T @ P.conj().T @ SZT @ P @ B)
        v = B @ sv[:, -1]
        u0 = P @ v
        Gam = [P.conj().T @ D @ P for D in DIRV]
        Mm = P.conj().T @ PI @ P
        E0 = bands_C(np.zeros(3), C)[0][idx]
        mm = 0.5 * (E0[2] - E0[1])
        Qu, _ = np.linalg.qr(bands_C(np.zeros(3), C)[1][:, idx[2:4]])
        piu = np.sign(np.real(np.trace(Qu.conj().T @ PI @ Qu)))
        Hd = lambda k: sum(k[i] * Gam[i] for i in range(3)) + mm * piu * Mm

        def latP(k):
            E, U = bands_C(k, C)
            ii = np.sort(np.argsort(np.abs(E - E0.mean()))[:4])
            Ql, _ = np.linalg.qr(U[:, ii[:2]]); Qu_, _ = np.linalg.qr(U[:, ii[2:]])
            return Ql @ Ql.conj().T, Qu_ @ Qu_.conj().T, E[ii]

        def dirP(k):
            e, vv = np.linalg.eigh(Hd(k))
            return vv[:, :2] @ vv[:, :2].conj().T, vv[:, 2:] @ vv[:, 2:].conj().T, e

        for k0x in (0.03,):
            k0 = np.array([k0x, 0, 0])
            n = 15
            q = np.linspace(-3 / SIG, 3 / SIG, n)
            QX, QY, QZ = np.meshgrid(q, q, q, indexing='ij')
            ks = k0 + np.stack([QX.ravel(), QY.ravel(), QZ.ravel()], 1)
            g2 = np.exp(-SIG ** 2 * ((ks - k0) ** 2).sum(1))
            h = 1e-5
            res = {}
            for nm, fP, vec in (('band', latP, u0), ('Dirac', dirP, v)):
                amp, om = [], []
                for k in ks:
                    Pl, Pu, E = fP(k)
                    dPu = (fP(k + [0, h, 0])[1] - fP(k - np.array([0, h, 0]))[1]) / (2 * h)
                    amp.append(1j * np.vdot(vec, Pl @ dPu @ vec))
                    om.append(E[2:].mean() - E[:2].mean())
                res[nm] = (np.array(amp), np.array(om))
            pt = Patch(k0, SIG, 40); pt.plain_state(u0)
            ts, r, _ = run_patch(pt, C, 600, 2)
            curves = {nm: 2 * np.real((g2[:, None] * a[:, None] * np.exp(-1j * o[:, None] * ts[None])).sum(0)) / g2.sum()
                      for nm, (a, o) in res.items()}
            y = r[:, 1]
            A = np.vstack([np.ones_like(ts), ts, ts ** 2]).T
            yo = y - A @ np.linalg.lstsq(A, y - curves['band'], rcond=None)[0]
            a_b = 2 * abs((g2 * res['band'][0]).sum()) / g2.sum()
            a_d = 2 * abs((g2 * res['Dirac'][0]).sum()) / g2.sum()
            print(f"   [{tag}] k0={k0x}: amplitude at t=0: band theory {a_b:.3f}, Dirac 4x4 {a_d:.3f}, simulation "
                  f"max|y_osc| {np.abs(yo).max():.3f}; rms(sim - band) {np.sqrt(np.mean((yo-curves['band'])**2)):.3f}, "
                  f"rms(sim - Dirac) {np.sqrt(np.mean((yo-curves['Dirac'])**2)):.3f} (signal rms {np.sqrt(np.mean(curves['band']**2)):.3f})"
                  f"  [{time.time()-t0:.0f}s]")
            out[f'D_{tag}_t'], out[f'D_{tag}_sim'] = ts, yo
            out[f'D_{tag}_band'], out[f'D_{tag}_dirac'] = curves['band'], curves['Dirac']

    np.savez(data_path('phase_bit_s4_open2.npz'), **out)
    print(f"saved data/phase_bit_s4_open2.npz  [{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()
