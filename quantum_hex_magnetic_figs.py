#!/usr/bin/env python3
"""
Figures and numerical report for the magnetic turning-phase model.
Run:  python3 quantum_hex_magnetic_figs.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_hex_turning import (SQRT3, DT_HALF, C_LIGHT, N_D, MOVES_PHYS,
                                 coin, bands, pick_band, com_track)
from quantum_hex_magnetic import *

EPS = 0.5
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def _coarse(p):
    n = (p.shape[0] // 2) * 2
    return p[0:n:2, :] + p[1:n:2, :]


def peierls_landau(B, nx, ny, cx, cy):
    """Second gauge, for the gauge-invariance test:  A = B*(-y, 0)."""
    X, Y = phys_grid(nx, ny, cx, cy)
    ph = np.empty((nx, ny, N_D), dtype=complex)
    for d in range(N_D):
        dx, dy = MOVES_PHYS[d]
        ph[:, :, d] = np.exp(-1j * B * (Y + dy / 2.0) * dx)
    return ph


# ─── Fig 1: construction, flux, gauge invariance, loop families ──────────────

def fig_geometry(fname="magnetic_geometry.png"):
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.9))
    B = 0.05

    a = ax[0]
    for dirs, lab, col in [([2, 4, 0], "triangle, $w=+1$", 'C0'),
                           ([4, 2, 0], "triangle, $w=-1$", 'C3')]:
        x = y = 0.0
        xs, ys = [0.0], [0.0]
        for d in dirs:
            x += MOVES_PHYS[d, 0]; y += MOVES_PHYS[d, 1]
            xs.append(x); ys.append(y)
        f, area = loop_flux(B, dirs)
        a.plot(xs, ys, '-o', color=col, lw=2, ms=5,
               label=f"{lab}: $S={area:+.4f}$")
        a.fill(xs, ys, color=col, alpha=.15)
    a.set_aspect('equal'); a.grid(alpha=.3); a.legend(fontsize=7)
    a.set_title("Peierls phase $=\\frac{B}{2}(x\\,dy-y\\,dx)$\n"
                "sums to $B\\cdot S$ (shoelace)", fontsize=9)

    a = ax[1]
    tests = [("triangle", [2, 4, 0], AREA_TRI),
             ("hexagon", [1, 2, 3, 4, 5, 0], 6 * AREA_TRI),
             ("rev. triangle", [4, 2, 0], -AREA_TRI),
             ("rhombus", [1, 2, 4, 5], 2 * AREA_TRI)]
    names, err = [], []
    for nm, dirs, expect in tests:
        f, area = loop_flux(B, dirs)
        names.append(nm); err.append(abs(area - expect))
        print(f"  flux/B for {nm:14s} = {area:+.9f}   expected {expect:+.9f}")
    a.bar(range(len(names)), np.maximum(err, 1e-18))
    a.set_yscale('log'); a.set_xticks(range(len(names)))
    a.set_xticklabels(names, rotation=25, fontsize=7)
    a.set_ylabel("|measured $-$ exact| area")
    a.set_title("flux per plaquette is exact\n(machine precision)", fontsize=9)
    a.grid(alpha=.3)

    a = ax[2]
    nx, ny = 199, 133; cx, cy = nx // 2, ny // 2
    X, Y = phys_grid(nx, ny, cx, cy)
    psi = band_projected_packet(nx, ny, cx, cy, 0, 0, 6.0, 0.4, 0.0, EPS, 0.0)
    psi_L = psi * np.exp(-1j * B * X * Y / 2.0)[:, :, None]
    C = coin(EPS, 0.0, "unitary")
    ps, pl = peierls_field(B, nx, ny, cx, cy), peierls_landau(B, nx, ny, cx, cy)
    A_, B_ = psi.copy(), psi_L.copy()
    ts, dev = [], []
    for t in range(1, 41):
        A_ = step_B(A_, C, ps); B_ = step_B(B_, C, pl)
        if t % 5 == 0:
            pa = (np.abs(A_)**2).sum(-1); pb = (np.abs(B_)**2).sum(-1)
            ts.append(t); dev.append(np.abs(pa - pb).max() / pa.max())
    a.semilogy(ts, dev, '-o', ms=4)
    a.set_xlabel("step"); a.set_ylabel("max rel. density difference")
    a.set_title("gauge invariance:\nsymmetric vs Landau gauge", fontsize=9)
    a.grid(alpha=.3)
    print(f"  gauge invariance of the density: {max(dev):.2e} (relative)")

    a = ax[3]
    loops = enumerate_loops_area(7)
    Bs = np.linspace(0, 2.0, 400)
    argL, tot = [], []
    for b in Bs:
        fam = return_amplitude(loops, 0.0, b, EPS)
        argL.append(np.angle(fam["left"]))
        tot.append(abs(fam["left"] + fam["right"] + fam["eight"]))
    a.plot(Bs, np.array(argL), lw=2, label="$\\arg A_{left}$ ($=-\\arg A_{right}$)")
    a.plot(Bs, np.array(tot) / tot[0], lw=2, label="$|A_{total}|$ (normalised)")
    a.axhline(0, color='k', lw=.6)
    a.set_xlabel("$B$"); a.set_title(
        "left and right loop families acquire\nconjugate phases $e^{\\pm iBS}$", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    fig.suptitle("Magnetic field on the triangular lattice: construction and checks", y=1.03)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("wrote", fname)


# ─── Fig 2: cyclotron orbits and the chirality order parameter ───────────────

def fig_orbits(fname="magnetic_orbits.png"):
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(2, 4)

    k, Rt, sigma, alpha = 0.4, 25.0, 8.0, 0.0
    B = k / Rt
    _, _, vg = pick_band(k, 0.0, EPS, alpha)
    ns = int(2 * np.pi * Rt / np.linalg.norm(vg) / DT_HALF)

    runs = {}
    for sgn in (+1, -1):
        runs[sgn] = run_cyclotron(sgn * B, eps=EPS, alpha=alpha, k_mag=k,
                                  n_steps=ns, store_every=max(1, ns // 24),
                                  sigma=sigma)
    r = runs[+1]
    X, Y = phys_grid(r['nx'], r['ny'], r['cx'], r['cy'])
    idx = np.linspace(0, len(r['times']) - 1, 4).astype(int)
    for j, i in enumerate(idx):
        a = fig.add_subplot(gs[0, j])
        p = (np.abs(r['hist'][i])**2).sum(-1)
        pc = _coarse(p)
        a.imshow(np.sqrt(pc).T, origin='lower', cmap='inferno', aspect='equal',
                 extent=[X.min(), X.max(), Y.min(), Y.max()])
        th = np.linspace(0, 2 * np.pi, 200)
        a.plot(Rt * np.sin(th), -Rt + Rt * np.cos(th), 'c--', lw=.9)
        a.set_xlim(-1.6 * Rt, 1.6 * Rt); a.set_ylim(-2.4 * Rt, 1.0 * Rt)
        a.set_title(f"$t={r['times'][i]:.0f}$, norm {p.sum():.4f}", fontsize=8)
        a.set_xlabel("x")
        if j == 0:
            a.set_ylabel("y")

    a = fig.add_subplot(gs[1, 0])
    for sgn, col in ((+1, 'C0'), (-1, 'C3')):
        tr = com_track(runs[sgn]['hist'], runs[sgn]['cx'], runs[sgn]['cy'])
        a.plot(tr[:, 0], tr[:, 1], '-o', ms=2.5, color=col,
               label=f"$B={sgn*B:+.3f}$")
    a.set_aspect('equal'); a.grid(alpha=.3); a.legend(fontsize=7)
    a.set_xlabel("x"); a.set_ylabel("y")
    a.set_title("the two chiralities, separated\nby the sign of $B$", fontsize=9)

    a = fig.add_subplot(gs[1, 1])
    print("  cyclotron radius check:")
    kk, RR, RT = [], [], []
    for k_, Rt_ in [(0.3, 20.0), (0.4, 25.0), (0.4, 35.0), (0.5, 30.0), (0.6, 40.0)]:
        B_ = k_ / Rt_
        _, _, vg_ = pick_band(k_, 0.0, EPS, 0.0)
        ns_ = int(2 * np.pi * Rt_ / np.linalg.norm(vg_) / DT_HALF)
        rr = run_cyclotron(B_, eps=EPS, alpha=0.0, k_mag=k_, n_steps=ns_,
                           store_every=max(1, ns_ // 20), sigma=8.0)
        tr = com_track(rr['hist'], rr['cx'], rr['cy'])
        x, y = tr[:, 0], tr[:, 1]
        M = np.c_[x, y, np.ones_like(x)]
        xc, yc, c0 = np.linalg.lstsq(M, x**2 + y**2, rcond=None)[0]
        Rf = np.sqrt(c0 + (xc / 2)**2 + (yc / 2)**2)
        kk.append(k_); RR.append(Rf); RT.append(k_ / B_)
        print(f"    k={k_:.1f} B={B_:.4f}:  R_measured={Rf:6.2f}   k/B={k_/B_:6.2f}"
              f"   ratio {Rf/(k_/B_):.3f}   norm {tr[-1,2]:.6f}")
    a.plot(RT, RR, 'o', ms=7)
    lim = [0, max(RT) * 1.15]
    a.plot(lim, lim, 'k--', lw=1, label="$R = k/B$")
    a.set_xlabel("$k/B$"); a.set_ylabel("measured orbit radius")
    a.set_title("orbit radius follows $R=k/B$", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = fig.add_subplot(gs[1, 2:])
    nx, ny = 301, 201; cx, cy = nx // 2, ny // 2
    C = coin(EPS, 0.0, "unitary")
    print("  chirality order parameter L_z:")
    for Bv, col, st in [(0.0, 'k', ':'), (0.03, 'C0', '-'), (-0.03, 'C3', '-'),
                        (0.06, 'C2', '--')]:
        ph = peierls_field(Bv, nx, ny, cx, cy)
        psi = band_projected_packet(nx, ny, cx, cy, 0, 0, 6.0, 1e-6, 0.0, EPS, 0.0)
        ts, lz = [0.0], [angular_momentum(psi, C, ph, nx, ny, cx, cy)]
        for t in range(1, 121):
            psi = step_B(psi, C, ph)
            if t % 5 == 0:
                ts.append(t * DT_HALF)
                lz.append(angular_momentum(psi, C, ph, nx, ny, cx, cy))
        a.plot(ts, lz, st, color=col, lw=1.8, label=f"$B={Bv:+.2f}$, band-projected")
        print(f"    B={Bv:+.2f}  L_z(t=60) = {lz[-1]:+.4f}")
    ph = peierls_field(0.05, nx, ny, cx, cy)
    psi = excited_edge(nx, ny, cx, cy, 0)
    ts, lz = [0.0], [0.0]
    for t in range(1, 121):
        psi = step_B(psi, C, ph)
        if t % 5 == 0:
            ts.append(t * DT_HALF)
            lz.append(angular_momentum(psi, C, ph, nx, ny, cx, cy))
    a.plot(ts, lz, '-', color='C4', lw=2.4,
           label="$B=+0.05$, bare excited edge (all 6 bands)")
    print(f"    bare excited edge, B=+0.05:  L_z(t=60) = {lz[-1]:+.3e}")
    a.axhline(0, color='k', lw=.6)
    a.set_xlabel("t"); a.set_ylabel("$L_z$")
    a.set_title("chirality order parameter.  A bare excited edge populates all six\n"
                "bands equally (1/6 each) and its circulation cancels exactly.",
                fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    fig.suptitle("Cyclotron motion: the chirality families separate dynamically", y=1.0)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("wrote", fname)


# ─── Fig 3: Landau levels — which orbit sizes reinforce ──────────────────────

def landau_run(alpha, B=0.05, n_steps=600, k0=0.4, sigma=4.0,
               nx=361, ny=241, filters=None):
    cx, cy = nx // 2, ny // 2
    psi0 = band_projected_packet(nx, ny, cx, cy, 0, 0, sigma, k0, 0.0, EPS, alpha)
    C = coin(EPS, alpha, "unitary")
    ph = peierls_field(B, nx, ny, cx, cy)
    c, phi = autocorrelation(psi0, C, ph, n_steps, filters=filters)
    return c, phi, dict(nx=nx, ny=ny, cx=cx, cy=cy, C=C, ph=ph, psi0=psi0)


def fig_landau(fname="magnetic_landau.png"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    B = 0.05
    Eg = np.linspace(-0.5, 4.0, 4000)
    store = {}

    a = ax[0]
    for alpha, col in [(0.0, 'C0'), (0.5, 'C3')]:
        c, _, _ = landau_run(alpha, B)
        S = spectrum_from_autocorr(c, Eg)
        pk = find_peaks(Eg, S, 0.12)
        store[alpha] = pk
        a.plot(Eg, S / S.max(), color=col, lw=1.4, label=f"$\\alpha={alpha}$")
        a.plot(pk, np.interp(pk, Eg, S / S.max()), 'v', color=col, ms=5)
        print(f"  alpha={alpha}: Landau peaks " + " ".join(f"{p:.4f}" for p in pk))
        print(f"           spacings " + " ".join(f"{d:.4f}" for d in np.diff(pk)))
    a.set_xlim(1.5, 3.0)
    a.set_xlabel("$E$"); a.set_ylabel("$|\\hat c(E)|$ (normalised)")
    a.set_title(f"time-domain spectrum, $B={B}$\n"
                "only quantised orbits reinforce", fontsize=9)
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1]
    for alpha, col in [(0.0, 'C0'), (0.5, 'C3')]:
        pk = store[alpha]
        a.plot(np.arange(1, len(pk)), np.diff(pk), 'o-', color=col, ms=6,
               label=f"$\\alpha={alpha}$")
    a.set_xlabel("level index $n$"); a.set_ylabel("$E_n - E_{n-1}$")
    a.set_title("level spacings are NOT equal: the effective\n"
                "mass grows with energy (lattice band)", fontsize=9)
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[2]
    for alpha, col, g in [(0.0, 'C0', 0.5), (0.5, 'C3', 0.0)]:
        pk = store[alpha]
        idx = onsager_index(pk, B, EPS, alpha)
        n = np.arange(len(idx))
        sl, ic = np.polyfit(n, idx, 1)
        a.plot(n, idx, 'o', color=col, ms=7,
               label=f"$\\alpha={alpha}$: slope {sl:.3f}, $\\gamma={ic:.3f}$")
        a.plot(n, sl * n + ic, '-', color=col, lw=1)
        print(f"  alpha={alpha}: A_k/(2 pi B) = " + " ".join(f"{v:.3f}" for v in idx))
        print(f"           fit slope {sl:.4f} (expect 1), gamma {ic:.4f} "
              f"(expect {g})")
    a.set_xlabel("level index $n$")
    a.set_ylabel("$A_k(E_n)\\,/\\,2\\pi B$")
    a.set_title("Onsager quantisation $A_k=2\\pi B(n+\\gamma)$\n"
                "$\\gamma=1/2$ ordinary band, $\\gamma=0$ Dirac point", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    fig.suptitle("A standing wave on the orbit: only orbits enclosing $(n+\\gamma)$ "
                 "flux quanta are reinforced", y=1.02)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("wrote", fname)
    return store


# ─── Fig 4: standing waves and the direction that Fourier makes ──────────────

def fig_standing_waves(levels=None, fname="magnetic_standing_waves.png"):
    B, alpha = 0.05, 0.0
    if levels is None:
        c, _, _ = landau_run(alpha, B)
        levels = find_peaks(np.linspace(-0.5, 4.0, 4000),
                            spectrum_from_autocorr(c, np.linspace(-0.5, 4.0, 4000)),
                            0.12)
    levels = list(levels[:6])
    c, phi, env = landau_run(alpha, B, filters=levels)
    nx, ny, cx, cy = env['nx'], env['ny'], env['cx'], env['cy']
    C, ph = env['C'], env['ph']
    X, Y = phys_grid(nx, ny, cx, cy)
    R = np.hypot(X, Y)

    phi = [f / np.sqrt((np.abs(f)**2).sum()) for f in phi]

    fig, ax = plt.subplots(2, 5, figsize=(17, 7))
    rmean, stat = [], []
    for j, (E, f) in enumerate(zip(levels[:4], phi[:4])):
        p = (np.abs(f)**2).sum(-1)
        rmean.append((R * p).sum())
        g = f.copy()
        for _ in range(40):
            g = step_B(g, C, ph)
        pg = (np.abs(g)**2).sum(-1)
        stat.append(np.abs(pg - p).sum() / p.sum())
        a = ax[0, j]
        pc = _coarse(p)
        a.imshow(pc.T, origin='lower', cmap='magma', aspect='equal',
                 extent=[X.min(), X.max(), Y.min(), Y.max()])
        a.set_xlim(-40, 40); a.set_ylim(-40, 40)
        a.set_title(f"$n={j}$, $E={E:.3f}$\n$\\langle r\\rangle={rmean[-1]:.2f}$, "
                    f"drift {stat[-1]*100:.1f}%/40 steps", fontsize=8)
        a.set_xlabel("x")
        if j == 0:
            a.set_ylabel("y")
    for j in range(4, 5):
        ax[0, j].axis('off')
    a = ax[0, 4]
    a.axis('on')
    rall = []
    for f in phi:
        p = (np.abs(f)**2).sum(-1)
        rall.append((R * p).sum())
    n = np.arange(len(rall))
    a.plot(n, rall, 'o-', ms=6, label="measured $\\langle r\\rangle$")
    sc = rall[-1] / np.sqrt(n[-1] + 0.5)
    a.plot(n, sc * np.sqrt(n + 0.5), 'k--', lw=1, label="$\\propto\\sqrt{n+1/2}$")
    a.set_xlabel("level index $n$"); a.set_ylabel("$\\langle r\\rangle$")
    a.set_title("orbit size grows with energy", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)
    print("  ring states: <r> = " + " ".join(f"{v:.2f}" for v in rall))
    print("  stationarity (density drift over 40 steps): "
          + " ".join(f"{v*100:.1f}%" for v in stat))

    psu = sum(phi) / np.sqrt(len(phi))
    psu = psu / np.sqrt((np.abs(psu)**2).sum())
    track = []
    snaps, snapt = [], []
    want = [0, 20, 40, 60]
    for t in range(0, 161):
        p = (np.abs(psu)**2).sum(-1)
        xm = (X * p).sum(); ym = (Y * p).sum()
        track.append((t * DT_HALF, xm, ym))
        if t in want:
            snaps.append(p.copy()); snapt.append(t * DT_HALF)
        psu = step_B(psu, C, ph)
    track = np.array(track)

    for j, (p, t) in enumerate(zip(snaps, snapt)):
        a = ax[1, j]
        pc = _coarse(p)
        a.imshow(pc.T, origin='lower', cmap='inferno', aspect='equal',
                 extent=[X.min(), X.max(), Y.min(), Y.max()])
        a.set_xlim(-40, 40); a.set_ylim(-40, 40)
        a.set_title(f"superposition, $t={t:.0f}$", fontsize=8)
        a.set_xlabel("x")
        if j == 0:
            a.set_ylabel("y")
    a = ax[1, 4]
    a.plot(track[:, 1], track[:, 2], '-', lw=1.6)
    a.plot(track[0, 1], track[0, 2], 'go', ms=6, label="start")
    a.set_aspect('equal'); a.grid(alpha=.3); a.legend(fontsize=7)
    a.set_xlabel("$\\langle x\\rangle$"); a.set_ylabel("$\\langle y\\rangle$")
    a.set_title("the superposition orbits:\ndirection out of standing waves", fontsize=9)
    d = np.hypot(track[:, 1], track[:, 2])
    print(f"  superposition: |<r>| ranges {d.min():.2f} .. {d.max():.2f}, "
          f"returns near the start at t={track[np.argmin(d[20:])+20,0]:.1f}")

    fig.suptitle("Top: energy-filtered eigenstates — stationary rings, no direction.  "
                 "Bottom: their coherent sum — a localised packet running the orbit.",
                 y=1.01)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("wrote", fname)


if __name__ == "__main__":
    print("=" * 74)
    print("Magnetic field on the 2+1D turning-phase model")
    print("=" * 74)
    print("\n[1] construction, flux, gauge invariance, loop families")
    fig_geometry()
    print("\n[2] cyclotron orbits and chirality")
    fig_orbits()
    print("\n[3] Landau levels / Onsager quantisation")
    store = fig_landau()
    print("\n[4] standing waves and their superposition")
    fig_standing_waves(store[0.0])
    print("\ndone.")
