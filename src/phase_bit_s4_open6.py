#!/usr/bin/env python3
"""
Paper 3, sixth follow-up to Experiment S4 — GR source from the walker's load,
causal field equation, the lattice mirror for massive particles, Weyl-node census
==============================================================================

A.  GR-type source: s(x) = 2 e(x) - e_M(x), with the load e(x) of round 5 and the
    mass load e_M(x) = mu Re[i (U psi)^+(x) (dU/dmu psi)(x)] / dt (local).
    For a Dirac particle this is E + p.v = T00 + T_ii: E for rest mass,
    E(1 + v^2) for a moving body, 2E for light.  Two packets on the x-chain as in
    round 5 (1D Green function, Hartree without self-term).
B.  Causal gravity: Phi on the nodes obeys a local wave equation
    Phi_tt - c_g^2 Phi_xx = -c_g^2 kappa s  (c_g = c'), started at Phi = 0.
    The test particle feels the source only after t = D / c_g.
C.  The lattice mirror for a massive particle: falling from rest into N -> N_min,
    reflection at the band maximum, return height (energy conservation).
D.  Band structure of the helicity sector +1 over the whole zone (minimal SU(2),
    mu = 0, B = 0): band ranges, smallest gap between neighbouring bands, coverage of
    the quasi-energy circle.  Decides whether 'connection of the Landau branches'
    has a topological meaning beyond the net flow (det winding) of round 5.

Run parts: python phase_bit_s4_open6.py [A] [B] [C] [D]   (default: all)
"""

import functools
import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import MOVES_PHYS, DT_HALF, LAT_A
from phase_bit_mass_s4 import model_generators, candidates, bands_C
from phase_bit_s4_open import CP
from phase_bit_s4_open4 import f_dwell, a_of, model_2Ostar, EPS, N0
from phase_bit_s4_open5 import chain_T, Buffered, packet, on_branch
from paths import data_path

print = functools.partial(print, flush=True)
UPPER = lambda E: np.sort(np.argsort(np.abs(E))[:4])[2:]


# ═════════════════════════════════════════════════════════════════════════════
def make_runner(X, Nx, SIG, kap):
    """Two-packet Hartree on the buffered x-chain. source: 'load' (e) or 'gr' (2e - e_M);
    field: 'green' (instantaneous 1D Green function) or 'wave' (local wave equation)."""
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    absdx = np.abs(xs[:, None] - xs[None, :])
    aN0 = a_of(N0)
    dmu = 1e-4

    def run(specs, nst, kaps=None, source='load', field='green', cg=CP):
        kaps = kaps or [kap] * len(specs)
        st = []
        for mu, kyz, x0 in specs:
            C = expm(1j * X(mu))
            psi, Eb = packet(C, kyz, UPPER, Nx, xs, x0, SIG)
            T = chain_T(C, kyz, Nx)
            dT = chain_T((expm(1j * X(mu + dmu)) - expm(1j * X(mu - dmu))) / (2 * dmu), kyz, Nx)
            bt = Buffered(T, Nx)
            bt.set_a(np.full(Nx, aN0))
            m, b = on_branch(bt, psi, Eb * DT_HALF, np.full(Nx, aN0), aN0)
            w = (np.abs(m) ** 2 + np.abs(b) ** 2).reshape(Nx, 24).sum(1)
            st.append(dict(bt=bt, dT=dT, mu=mu, m=m, b=b, e=None, eM=None, Eb=Eb, xc=[], Q=[], QM=[], w0=w,
                           phi=np.zeros(Nx), phio=np.zeros(Nx)))
        def loads(s):
            bt = s['bt']
            m2, b2 = bt.tick(s['m'], s['b'])
            ld = (np.conj(s['m']) * 1j * (m2 - s['m']) + np.conj(s['b']) * 1j * (b2 - s['b'])).real / DT_HALF
            dm = s['dT'] @ s['m']
            lm = (1j * (np.conj(m2) * bt.ca * dm + np.conj(b2) * 1j * bt.sa * dm)).real * s['mu'] / DT_HALF
            return m2, b2, ld.reshape(Nx, 24).sum(1), lm.reshape(Nx, 24).sum(1)
        for s in st:   # initial loads from a probe tick
            _, _, s['e'], s['eM'] = loads(s)
        lam2 = (cg * DT_HALF / LAT_A) ** 2
        for it in range(nst + 1):
            srcs = [2 * s['e'] - s['eM'] if source == 'gr' else s['e'] for s in st]
            if field == 'green':
                phis = [0.5 * kj * absdx @ q for kj, q in zip(kaps, srcs)]
            else:
                phis = []
                for kj, q, s in zip(kaps, srcs, st):
                    p, po = s['phi'], s['phio']
                    lap = np.zeros(Nx)
                    lap[1:-1] = p[2:] - 2 * p[1:-1] + p[:-2]
                    pn = 2 * p - po + lam2 * (lap - kj * LAT_A * q)
                    s['phio'], s['phi'] = p, pn
                    phis.append(pn)
            for i, s in enumerate(st):
                Phi = sum(p for j, p in enumerate(phis) if j != i)
                w = (np.abs(s['m']) ** 2 + np.abs(s['b']) ** 2).reshape(Nx, 24).sum(1)
                s['xc'].append((w @ xs) / w.sum())
                s['Q'].append(s['e'].sum())
                s['QM'].append(s['eM'].sum())
                if it == nst:
                    continue
                Phi = Phi - np.interp(s['xc'][-1], xs, Phi)
                bt = s['bt']
                bt.set_a(a_of(np.clip(N0 * (1 + Phi), 1e-3, 1 - 1e-9)))
                s['m'], s['b'], s['e'], s['eM'] = loads(s)
        return st
    return run, xs


def part_A(out):
    print("A. GR-type source s = 2 e - e_M (two packets, 2O*, buffer lapse N0 = 0.75, 1D Green function)")
    t0 = time.time()
    X, lam = model_2Ostar()
    Nx, SIG, nst, D = 1000, 60.0, 200, 400
    kap = 2e-3 / (N0 * 0.06)
    run, xs = make_runner(X, Nx, SIG, kap)
    ts = np.arange(nst + 1) * DT_HALF
    acc = lambda s: 2 * np.polyfit(ts, np.array(s['xc']), 2)[0]
    rows = []
    for name, mu, ky in (('rest', 0.03, 0.0), ('k_y = 0.3', 0.03, 0.3), ('k_y = 0.6', 0.03, 0.6),
                         ('light k_y = 0.2', 0.0, 0.2), ('light k_y = 0.4', 0.0, 0.4)):
        Es = bands_C(np.array([0.0, ky, 0.0]), expm(1j * X(mu)))[0]
        Esrc = Es[UPPER(Es)].mean()
        ks = kap * 0.06 / Esrc / 2                 # keep the field slope small
        res = {}
        for source in ('load', 'gr'):
            st = run([(0.03, (0.0, 0.0), -D / 2 * LAT_A), (mu, (ky, 0.0), +D / 2 * LAT_A)], nst, [kap, ks], source)
            res[source] = (acc(st[0]) * kap / ks, np.mean(st[1]['Q']), np.mean(st[1]['QM']))
        m = 0.06
        v2 = 1 - (m / Esrc) ** 2 if mu > 0 else 1.0
        rows.append((ky, mu, Esrc, res['load'][0], res['gr'][0], res['gr'][1], res['gr'][2], Esrc * (1 + v2)))
        print(f"   source {name:16s}: E = {Esrc:.4f}; Q = {res['gr'][1]:.5f} (N0 E = {N0*Esrc:.5f}), "
              f"Q_M = {res['gr'][2]:.5f} (N0 m^2/E = {N0*(m**2/Esrc if mu > 0 else 0):.5f}); "
              f"a_test load {res['load'][0]:+.3e}, GR source {res['gr'][0]:+.3e}  [{time.time()-t0:.0f}s]")
    r = np.array(rows)
    ref_l, ref_g = r[0, 3], r[0, 4]
    print("   a_test / a_test(rest source):")
    for row in r:
        print(f"      E = {row[2]:.4f}: load source {row[3]/ref_l:.3f} (E/m = {row[2]/0.06:.3f}); "
              f"GR source {row[4]/ref_g:.3f} (E(1+v^2)/m = {row[7]/0.06:.3f})")
    out['A_rows'] = r


# ═════════════════════════════════════════════════════════════════════════════
def part_B(out):
    print("B. causal gravity: local wave equation for Phi (c_g = c'), started at Phi = 0")
    t0 = time.time()
    X, lam = model_2Ostar()
    Nx, SIG, nst, Dl = 1200, 60.0, 900, 260
    kap = 1e-3 / (N0 * 0.12)                 # slope 1e-3 from the mu = 0.12 source (Q = N0 E = 0.18)
    run, xs = make_runner(X, Nx, SIG, kap)
    ts = np.arange(nst + 1) * DT_HALF
    specs = [(0.06, (0.0, 0.0), -Dl / 2 * LAT_A), (0.12, (0.0, 0.0), +Dl / 2 * LAT_A)]
    D = Dl * LAT_A
    tr = {}
    for field in ('green', 'wave'):
        st = run(specs, nst, field=field)
        tr[field] = (np.array(st[0]['xc']), np.array(st[1]['xc']))
        w1, w2 = st[0]['w0'] / st[0]['w0'].sum(), st[1]['w0'] / st[1]['w0'].sum()
        print(f"   {field:5s}: test particle displacement at t = 100, 200, 300, 400: " +
              ", ".join(f"{tr[field][0][int(t/DT_HALF)] - tr[field][0][0]:+.4f}" for t in (100, 200, 300, 400)) +
              f"  [{time.time()-t0:.0f}s]")
    # fit the onset: x(t) = x0 + v0 t + a/2 max(0, t - t0)^2 with a from the instantaneous run
    xg, xw = tr['green'][0], tr['wave'][0]
    a_g = 2 * np.polyfit(ts, xg - xg[0], 2)[0]
    v0 = np.polyfit(ts, xg - xg[0], 2)[1]
    best = None
    for t_on in np.linspace(0, 400, 801):
        model = v0 * ts + 0.5 * a_g * np.clip(ts - t_on, 0, None) ** 2
        err = np.sum((xw - xw[0] - model) ** 2)
        if best is None or err < best[1]:
            best = (t_on, err)
    # same for the source (feels the test particle's field)
    xg2, xw2 = tr['green'][1], tr['wave'][1]
    a_g2 = 2 * np.polyfit(ts, xg2 - xg2[0], 2)[0]
    v02 = np.polyfit(ts, xg2 - xg2[0], 2)[1]
    best2 = None
    for t_on in np.linspace(0, 400, 801):
        model = v02 * ts + 0.5 * a_g2 * np.clip(ts - t_on, 0, None) ** 2
        err = np.sum((xw2 - xw2[0] - model) ** 2)
        if best2 is None or err < best2[1]:
            best2 = (t_on, err)
    # prediction with finite widths: a(t) = a_inst * F(t), F = P(|x_src - x_test| < c' t) from the initial densities
    dist = np.abs(xs[:, None] - xs[None, :]).ravel()
    pw = np.outer(w1, w2).ravel()
    o = np.argsort(dist)
    F = np.interp(CP * ts, dist[o], np.cumsum(pw[o]))
    acc_t = a_g * F
    vel = np.concatenate([[0], np.cumsum(0.5 * (acc_t[1:] + acc_t[:-1]) * DT_HALF)])
    xpred = v0 * ts + np.concatenate([[0], np.cumsum(0.5 * (vel[1:] + vel[:-1]) * DT_HALF)])
    print("   causal prediction from the initial densities (a_inst * F(t)): displacement at t = 100, 200, 300, 400: " +
          ", ".join(f"{xpred[int(t/DT_HALF)]:+.4f}" for t in (100, 200, 300, 400)) +
          f"; rms deviation sim - prediction over the run {np.sqrt(np.mean((xw - xw[0] - xpred)**2)):.4f}"
          f" (sim - instantaneous {np.sqrt(np.mean((xw - xg)**2)):.3f})")
    out['B_pred'] = xpred
    late = ts > 360
    a_late = 2 * np.polyfit(ts[late], xw[late] - xw[0], 2)[0]
    print(f"   instantaneous field: a_test = {a_g:+.3e}; causal field: onset t0 = {best[0]:.1f} "
          f"(D / c' = {D/CP:.1f}), late acceleration {a_late:+.3e} ({a_late/a_g:.3f} of instantaneous)")
    print(f"   the heavy source feels the light one: onset t0 = {best2[0]:.1f}")
    out['B_ts'] = ts
    out['B_green'] = np.array(tr['green'])
    out['B_wave'] = np.array(tr['wave'])
    out['B_fit'] = np.array([best[0], D / CP, a_g, a_late, best2[0], v0])


# ═════════════════════════════════════════════════════════════════════════════
def part_C(out):
    print("C. the lattice mirror for massive particles (fall from rest into N -> N_min)")
    X, lam = model_2Ostar()
    Nx = 360
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    Nmax, Nmin, x1, x2 = 0.9, 0.04, -40.0, 40.0
    Nprof = lambda x: np.clip(Nmin + (Nmax - Nmin) * (x - x1) / (x2 - x1), Nmin, Nmax)
    a = a_of(Nprof(xs))
    for mu, x0, nst in ((0.03, 20.0, 4000), (0.15, 20.0, 4000), (0.0, 60.0, 2400)):
        t0 = time.time()
        C = expm(1j * X(mu))
        if mu > 0:
            sel, kx0 = UPPER, 0.0
        else:
            sel, kx0 = (lambda E: np.sort(np.argsort(np.abs(E - CP * 0.3))[:2])), -0.3
        bt = Buffered(chain_T(C, (0.0, 0.0), Nx), Nx)
        bt.set_a(a)
        psi, Eb = packet(C, (0.0, 0.0), sel, Nx, xs, x0, 12.0, kx0=kx0)
        m, b = on_branch(bt, psi, Eb * DT_HALF, a, a_of(Nprof(x0)))
        # band along -x (tracked from the start), its maximum and the turning point
        kb = np.linspace(kx0 - 1e-3 if mu > 0 else -0.05, -2 * np.pi / LAT_A + 0.3, 3000)
        E0 = bands_C(np.array([kb[0], 0, 0]), C)[0]
        Ebd = [E0[sel(E0)].max()]
        for q in kb[1:]:
            E_ = bands_C(np.array([q, 0, 0]), C)[0]
            Ebd.append(E_[np.argmin(np.abs(E_ - Ebd[-1]))])
        Ebd = np.array(Ebd)
        Emax = Ebd.max()
        z = np.vdot(np.concatenate([m, b]), np.concatenate(bt.tick(m, b)))
        Ec = -np.angle(z) / DT_HALF
        Nturn = brentq(lambda N: f_dwell(Emax * DT_HALF, a_of(N)) / DT_HALF - Ec, 1e-4, 0.9999)
        xturn = x1 + (Nturn - Nmin) / (Nmax - Nmin) * (x2 - x1)
        rec = []
        for it in range(nst + 1):
            if it % 10 == 0:
                w = (np.abs(m) ** 2 + np.abs(b) ** 2).reshape(Nx, 24).sum(1)
                rec.append((it * DT_HALF, (w @ xs) / w.sum(), w[xs < xturn - 3].sum() / w.sum(),
                            w[xs > x0 - 10].sum() / w.sum()))
            m, b = bt.tick(m, b)
        rec = np.array(rec)
        jmin = np.argmin(rec[:, 1])
        after = rec[jmin:]
        jback = jmin + np.argmax(after[:, 1])
        name = f"mu = {mu}" if mu > 0 else "massless k = -0.3"
        print(f"   {name:18s}: E' = {Ec:.4f}, band maximum {Emax:.3f} at k = {kb[np.argmax(Ebd)]:.2f}; predicted turning "
              f"N = {Nturn:.4f} (x = {xturn:+.2f}); deepest <x> = {rec[jmin,1]:+.2f} at t = {rec[jmin,0]:.0f}; "
              f"then back up to <x> = {rec[jback,1]:+.2f} at t = {rec[jback,0]:.0f} (start {x0:+.0f}); "
              f"max weight beyond the turning point {rec[:,2].max():.3f}; weight back near the start "
              f"{rec[jback,3]:.3f}  [{time.time()-t0:.0f}s]")
        out[f'C_rec_{mu}'] = rec
        out[f'C_info_{mu}'] = np.array([Ec, Emax, Nturn, xturn, x0])
    out['C_prof'] = np.array([Nmax, Nmin, x1, x2])


# ═════════════════════════════════════════════════════════════════════════════
def part_D(out):
    print("D. band structure of the helicity sector +1 over the whole zone (minimal SU(2), mu = 0, B = 0)")
    t0 = time.time()
    models, _ = model_generators()
    H = candidates()['on-site u_d.s']
    C = expm(1j * EPS * models['minimal SU(2)'])
    wH, VH = np.linalg.eigh(H)
    P1 = VH[:, np.isclose(wH, 1)]
    CP1 = C @ P1
    Md = np.array([P1[2 * d:2 * d + 2].conj().T @ CP1[2 * d:2 * d + 2] for d in range(12)])
    PER = 4 * np.pi                                  # quasi-energy period (2 pi / dt)

    def spec(k):
        k = np.atleast_2d(k)
        U = np.einsum('nd,dij->nij', np.exp(-1j * k @ MOVES_PHYS.T), Md)
        E = np.sort(-np.angle(np.linalg.eigvals(U)) / DT_HALF, axis=-1)
        gaps = np.concatenate([np.diff(E, axis=-1), (E[..., :1] + PER - E[..., -1:])], axis=-1)
        return E, gaps                                # gap j between band j and j+1 (j = 11: across the zone edge)

    n = 48                                           # grid contains k = 0 and the zone-edge points
    g1 = np.arange(n) / n * 2 * np.pi / LAT_A - np.pi / LAT_A
    K = np.stack(np.meshgrid(g1, g1, g1, indexing='ij'), -1).reshape(-1, 3)
    E, gaps = spec(K)
    print("   bands at k = 0: " + ", ".join(f"{e:+.3f}" for e in spec(np.zeros(3))[0][0]))
    rng_ = np.array([(E[:, b].min(), E[:, b].max()) for b in range(12)])
    print("   band ranges over the zone: " + ", ".join(f"[{lo:+.2f}, {hi:+.2f}]" for lo, hi in rng_))
    mg = []
    for j in range(12):
        i = np.argmin(gaps[:, j])
        r = minimize(lambda kk: spec(kk)[1][0, j], K[i], method='Nelder-Mead',
                     options=dict(xatol=1e-9, fatol=1e-13, maxiter=4000))
        mg.append((gaps[i, j], r.fun))
    mg = np.array(mg)
    print("   smallest gap between neighbouring bands (grid -> refined): " +
          ", ".join(f"{j}|{(j+1)%12}: {g:.3f}->{r:.1e}" for j, (g, r) in enumerate(mg)))
    # coverage of the quasi-energy circle
    eg = np.linspace(-2 * np.pi, 2 * np.pi, 4001)[:-1]
    cover = np.array([((rng_[:, 0] <= e) & (rng_[:, 1] >= e)).sum() for e in eg])
    # the only candidate for a global gap: around the zone edge (top of band 11, bottom of band 0)
    i11, i0 = np.argmax(E[:, 11]), np.argmin(E[:, 0])
    r11 = minimize(lambda kk: -spec(kk)[0][0, 11], K[i11], method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-14))
    r0 = minimize(lambda kk: spec(kk)[0][0, 0], K[i0], method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-14))
    top, bot = -r11.fun, r0.fun + PER
    inner = cover[(eg > -2 * np.pi + 0.01) & (eg < 2 * np.pi - 0.01)].min()
    print(f"   away from the zone edge every quasi-energy is covered by at least {inner} band(s) (up to {cover.max()}); "
          f"at the zone edge: top of band 11 = {top:.5f}, bottom of band 0 + 4 pi = {bot:.5f} -> global gap "
          f"{bot - top:+.5f} centred at {0.5*(top+bot)-PER:+.5f} (= -2 pi = +2 pi)  [{time.time()-t0:.0f}s]")
    out['D_edge'] = np.array([top, bot])
    out['D_ranges'], out['D_mingap'], out['D_cover'] = rng_, mg, np.column_stack([eg, cover])


def main():
    parts = [p.upper() for p in sys.argv[1:]] or ['A', 'B', 'C', 'D']
    path = data_path('phase_bit_s4_open6.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in parts:
        {'A': part_A, 'B': part_B, 'C': part_C, 'D': part_D}[p](out)
        np.savez(path, **out)
    print("saved data/phase_bit_s4_open6.npz")


if __name__ == '__main__':
    main()
