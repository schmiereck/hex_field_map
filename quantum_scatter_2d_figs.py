#!/usr/bin/env python3
"""
Report and figure for two-body deflection scattering in 2+1D (step 4).
Run:  python3 quantum_scatter_2d_figs.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_hex_turning import bands, pick_band, coin
from quantum_scatter_2d import *

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
EPS, ALPHA, U = 0.5, 0.0, 2.0
NX, NY = 241, 141


def setup(k):
    v = np.linalg.norm(pick_band(k, 0, EPS, ALPHA)[2])
    vrel = 2 * v
    sig = float(np.clip(3.5 / k, 5.0, 13.0))
    d0 = 3 * sig + 6
    n = int((d0 / vrel + 18.0) / DT_HALF)
    cx, cy = NX // 2, NY // 2
    phi0 = initial_relative(NX, NY, cx, cy, (-d0, 0.0),
                            np.array([k, 0.0]), sig, EPS, ALPHA)
    return phi0, cx, cy, n, vrel


def internal_overlap(k, alpha=ALPHA):
    _, V1, _ = bands(k, 0, EPS, alpha)
    _, V2, _ = bands(-k, 0, EPS, alpha)
    u1 = V1[:, -1] / np.linalg.norm(V1[:, -1])
    u2 = V2[:, -1] / np.linalg.norm(V2[:, -1])
    return abs(np.vdot(u1, u2))


def scatter_set(k):
    phi0, cx, cy, n, vrel = setup(k)
    Xp = exchange(phi0, cx, cy)
    out = {}
    for nm, st in (("d", phi0), ("b", (phi0 + Xp) / np.sqrt(2)),
                   ("f", (phi0 - Xp) / np.sqrt(2))):
        st = st / np.sqrt((np.abs(st) ** 2).sum())
        fr = run_scatter(st, EPS, ALPHA, 0.0, n, cx, cy)
        it = run_scatter(st, EPS, ALPHA, U, n, cx, cy)
        out[nm] = it - fr
        out["norm_" + nm] = float((np.abs(it) ** 2).sum())
    out.update(cx=cx, cy=cy, n=n, vrel=vrel, R0=vrel * n * DT_HALF)
    return out


def main():
    print("=" * 76)
    print("Step 4: two-body deflection scattering in 2+1D")
    print("=" * 76)

    print("\n[1] the relative-coordinate reduction is exact")
    print("    (verified elsewhere against the full two-particle evolution: 0.0e+00,")
    print("     with and without the contact term; [step, X] = 1e-17)")

    print("\n[2] does deflection happen at all?  (impossible in 1+1D: integrability)")
    S = scatter_set(0.30)
    cx, cy, R0 = S['cx'], S['cy'], S['R0']
    print("    scattered weight ||phi_U - phi_free||^2 = %.5f   (norm kept %.4f)"
          % (float((np.abs(S['d']) ** 2).sum()), S['norm_d']))
    th, praw = angular_profile(S['d'], cx, cy, 0.25 * R0, 0.80 * R0)
    fold = lambda q: 0.5 * (q + q[::-1])
    praw = fold(praw)                      # only the mirror fold, NO exchange partner
    pdist = fold(praw + np.roll(praw, 36))  # the distinguishable reference
    print("    a single incoming branch scatters forward-peaked:")
    print("      p(2.5 deg) / p(177.5 deg) = %.2f" % (praw[0] / praw[35]))
    print("      (the distinguishable REFERENCE adds the exchange partner and is")
    print("       therefore symmetric under theta -> theta+180 by construction)")

    print("\n[3] identical-particle (Mott) interference at 90 degrees, radially resolved")
    print("    a broad annulus averages the interference away — resolve it in r")
    rows = []
    for lo in np.arange(0.20, 0.90, 0.07):
        rmin, rmax = lo * R0, (lo + 0.07) * R0
        v = {}
        for nm in "bf":
            t, p = angular_profile(S[nm], cx, cy, rmin, rmax)
            v[nm] = fold(p)[np.argmin(np.abs(t - 90))]
        tot = v['b'] + v['f']
        if tot > 1e-12:
            rows.append((0.5 * (rmin + rmax), (v['b'] - v['f']) / tot))
            print("      r = %5.1f - %5.1f   contrast = %+.4f" % (rmin, rmax, rows[-1][1]))

    print("\n[4] is the internal-state overlap the ceiling on the contrast?")
    print("    k      |<u(k)|u(-k)>|   best contrast   ratio")
    scan = []
    for k in (0.30, 0.45, 0.60, 0.90):
        Sk = scatter_set(k) if k != 0.30 else S
        ov = internal_overlap(k)
        best = -9
        R0k = Sk['R0']
        for lo in np.arange(0.20, 0.90, 0.07):
            v = {}
            for nm in "bf":
                t, p = angular_profile(Sk[nm], Sk['cx'], Sk['cy'],
                                       lo * R0k, (lo + 0.07) * R0k)
                v[nm] = fold(p)[np.argmin(np.abs(t - 90))]
            tot = v['b'] + v['f']
            if tot > 1e-12:
                best = max(best, (v['b'] - v['f']) / tot)
        scan.append((k, ov, best))
        print("   %.2f      %.4f          %+.4f       %.3f" % (k, ov, best, best / ov))

    print("\n[5] why the ceiling exists: the band eigenvector is locked to k")
    ks = np.linspace(0.05, 1.4, 40)
    ov0 = [internal_overlap(k, 0.0) for k in ks]
    ov5 = [internal_overlap(k, 0.5) for k in ks]
    print("    alpha=0.0: |<u(k)|u(-k)>| falls from %.3f to %.3f over k in [0.05, 1.4]"
          % (ov0[0], ov0[-1]))
    print("    alpha=0.5: identically %.1e — counter-propagating states are EXACTLY"
          % max(ov5))
    print("               orthogonal, so no exchange interference is possible at all")

    figure(S, th, praw, rows, scan, ks, ov0, ov5)


def figure(S, th, praw, rows, scan, ks, ov0, ov5, fname="scatter_2d.png"):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    cx, cy, R0 = S['cx'], S['cy'], S['R0']
    X, Y, _ = grid_phys(NX, NY, cx, cy)
    fold = lambda q: 0.5 * (q + q[::-1])

    a = ax[0, 0]
    d = (np.abs(S['d']) ** 2).sum(axis=(2, 3))
    n2 = (NX // 2) * 2
    dc = d[0:n2:2] + d[1:n2:2]
    a.imshow(np.sqrt(dc).T, origin='lower', cmap='inferno', aspect='equal',
             extent=[X.min(), X.max(), Y.min(), Y.max()])
    a.set_xlabel("$r_x$"); a.set_ylabel("$r_y$")
    a.set_title("scattered wave $|\\phi_U-\\phi_{free}|^2$\n"
                "in the relative coordinate", fontsize=9)

    a = ax[0, 1]
    a.plot(th, praw / praw.sum(), 'o-', ms=3)
    a.set_yscale('log')
    a.set_xlabel("scattering angle $\\theta$ [deg]  (0° = forward)")
    a.set_ylabel("fraction per bin")
    a.set_title("one incoming branch: forward peaked,\n"
                "with six-fold lattice structure", fontsize=9)
    a.grid(alpha=.3)

    a = ax[0, 2]
    for nm, lab, col in (("b", "boson", 'C0'), ("f", "fermion", 'C3')):
        t, p = angular_profile(S[nm], cx, cy, 0.20 * R0, 0.27 * R0)
        p = fold(p)
        a.plot(t[:36], p[:36], 'o-', ms=4, color=col, label=lab)
    a.axvline(90, color='k', ls='--', lw=1)
    a.set_yscale('log')
    a.set_xlabel("$\\theta$ [deg]"); a.set_ylabel("scattered density")
    a.set_title("Mott interference: boson enhanced,\nfermion suppressed at 90°",
                fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[1, 0]
    r = [x for x, _ in rows]; c = [y for _, y in rows]
    a.plot(r, c, 'o-', ms=5)
    a.axhline(0, color='k', lw=.7)
    a.set_xlabel("annulus radius $r$")
    a.set_ylabel("contrast $(p_b-p_f)/(p_b+p_f)$ at 90°")
    a.set_title("the interference dephases with radius —\n"
                "a broad annulus averages it away", fontsize=9)
    a.grid(alpha=.3)

    a = ax[1, 1]
    ov = [o for _, o, _ in scan]; be = [b for _, _, b in scan]
    a.plot(ov, be, 'o', ms=9)
    lim = [0, 0.9]
    a.plot(lim, lim, 'k--', lw=1, label="contrast = $|\\langle u(k)|u(-k)\\rangle|$")
    for k, o, b in scan:
        a.annotate(f"k={k}", (o, b), fontsize=7, xytext=(4, -8),
                   textcoords='offset points')
    a.set_xlabel("$|\\langle u(k)|u(-k)\\rangle|$")
    a.set_ylabel("best measured contrast at 90°")
    a.set_title("the internal-state overlap is the ceiling", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[1, 2]
    a.plot(ks, ov0, lw=2, label="$\\alpha=0$ (massive)")
    a.plot(ks, ov5, lw=2, label="$\\alpha=1/2$ (spinor)")
    a.set_xlabel("$|k|$"); a.set_ylabel("$|\\langle u(k)|u(-k)\\rangle|$")
    a.set_ylim(-0.05, 1.05)
    a.set_title("the band eigenvector is locked to $k$:\n"
                "at $\\alpha=1/2$ the overlap is exactly 0", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    fig.suptitle("Deflection scattering in 2+1D: the collision that 1+1D forbids, "
                 "and how far the exchange interference can go", y=1.01)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("\nwrote", fname)


if __name__ == "__main__":
    main()
