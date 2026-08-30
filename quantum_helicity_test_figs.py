#!/usr/bin/env python3
"""
Figure for the helicity test.  Run:  python3 quantum_helicity_test_figs.py

The two scattering runs are expensive (about 8 minutes together), so their
results are cached in helicity_cache.npz; delete that file to recompute.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_helicity_test import (overlap_2d, overlap_fcc, dirac_reference)
from quantum_hex_turning import bands as h_bands, pick_band, group_velocity
from quantum_scatter_2d import (initial_relative, exchange, run_scatter,
                                angular_profile, DT_HALF)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
U = 2.0
CACHE = "helicity_cache.npz"


def band_turnover(eps, alpha=0.0, kmax=1.4):
    """Largest |k| for which the particle band still moves forward.

    Past the band maximum v_g changes sign, the state runs backwards, and
    'counter-propagating' stops being meaningful — so curves are cut there.
    """
    ks = np.linspace(0.02, kmax, 300)
    for k in ks:
        _, V, _ = h_bands(k, 0.0, eps, alpha)
        if group_velocity(V[:, -1])[0] <= 0:
            return k
    return kmax


def scatter_case(eps, k, sig, d0, n_steps, Nx, Ny):
    cx, cy = Nx // 2, Ny // 2
    ov = overlap_2d(k, eps)
    vrel = 2 * np.linalg.norm(pick_band(k, 0, eps, 0.0)[2])
    phi0 = initial_relative(Nx, Ny, cx, cy, (-d0, 0.0),
                            np.array([k, 0.0]), sig, eps, 0.0)
    Xp = exchange(phi0, cx, cy)
    res = {}
    for nm, st in (("b", (phi0 + Xp) / np.sqrt(2)), ("f", (phi0 - Xp) / np.sqrt(2))):
        st = st / np.sqrt((np.abs(st) ** 2).sum())
        fr = run_scatter(st, eps, 0.0, 0.0, n_steps, cx, cy)
        it = run_scatter(st, eps, 0.0, U, n_steps, cx, cy)
        res[nm] = it - fr
        res["n"] = float((np.abs(it) ** 2).sum())
    fold = lambda q: 0.5 * (q + q[::-1])
    R0 = vrel * n_steps * DT_HALF
    best, win = -9, None
    for lo in np.arange(0.15, 0.90, 0.07):
        vv = {}
        for nm in "bf":
            t, p = angular_profile(res[nm], cx, cy, lo * R0, (lo + 0.07) * R0)
            vv[nm] = fold(p)[np.argmin(np.abs(t - 90))]
        tot = vv["b"] + vv["f"]
        if tot > 1e-12 and (vv["b"] - vv["f"]) / tot > best:
            best, win = (vv["b"] - vv["f"]) / tot, (lo * R0, (lo + 0.07) * R0)
    prof = {}
    for nm in "bf":
        t, p = angular_profile(res[nm], cx, cy, *win)
        prof[nm] = fold(p)
    return ov, best, t, prof["b"], prof["f"], res["n"]


def main():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        print("using cached scattering results (delete %s to recompute)" % CACHE)
    else:
        print("running the two scattering cases (about 8 minutes) ...", flush=True)
        hi = scatter_case(1.5, 0.30, 9.0, 30.0, 260, 407, 235)
        print("  high overlap: ov=%.4f contrast=%.4f norm=%.3f" % (hi[0], hi[1], hi[5]),
              flush=True)
        lo = scatter_case(0.5, 0.30, 11.7, 41.0, 91, 241, 141)
        print("  low  overlap: ov=%.4f contrast=%.4f norm=%.3f" % (lo[0], lo[1], lo[5]),
              flush=True)
        np.savez(CACHE, hi_ov=hi[0], hi_c=hi[1], th=hi[2], hi_b=hi[3], hi_f=hi[4],
                 hi_n=hi[5], lo_ov=lo[0], lo_c=lo[1], lo_n=lo[5])
        z = np.load(CACHE)

    i90 = np.argmin(np.abs(z['th'] - 90))
    ratio = float(z['hi_f'][i90] / z['hi_b'][i90])
    print("\nSUMMARY")
    print("  high overlap %.4f -> contrast %.4f (ratio %.3f), norm %.3f"
          % (z['hi_ov'], z['hi_c'], z['hi_c'] / z['hi_ov'], z['hi_n']))
    print("  low  overlap %.4f -> contrast %.4f (ratio %.3f), norm %.3f"
          % (z['lo_ov'], z['lo_c'], z['lo_c'] / z['lo_ov'], z['lo_n']))
    print("  fermion/boson at 90 deg, high overlap: %.5f  (%.2f %%)"
          % (ratio, 100 * ratio))

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.1))

    a = ax[0]
    for eps, c in ((0.5, 'C0'), (1.5, 'C1')):
        kt = band_turnover(eps)
        ks = np.linspace(0.03, kt, 60)
        a.plot(ks, [overlap_2d(k, eps) for k in ks], color=c, lw=2,
               label=f"2D turning, $\\varepsilon$={eps}")
        a.plot([kt], [overlap_2d(kt, eps)], 'v', color=c, ms=6)
    ks = np.linspace(0.03, 1.2, 45)
    a.plot(ks, [overlap_fcc(k, 0.3, 0)[0][0] for k in ks], 'C2--', lw=2,
           label="FCC spin 0")
    a.plot(ks, [overlap_fcc(k, 0.3, 0.5)[0][0] for k in ks], 'C3-', lw=2,
           label="FCC spin ½")
    m = h_bands(0, 0, 0.5, 0.0)[0][-1]
    a.plot(ks, [dirac_reference(k, m) for k in ks], 'k:', lw=1.5,
           label="Dirac $m/E=1/\\gamma$")
    a.set_xlabel("$|k|$"); a.set_ylabel("$|\\langle u(k)|u(-k)\\rangle|$")
    a.set_title("all models $\\to 1$ as $k\\to0$; spin ½ is lowest.\n"
                "▽ = band turnover, curves cut there", fontsize=9)
    a.legend(fontsize=6); a.grid(alpha=.3); a.set_ylim(-0.03, 1.05)

    a = ax[1]
    kk = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])
    a.plot(kk, [overlap_fcc(k, 0.3, 0.5)[0][0] for k in kk], 'o-', ms=6,
           label="FCC spin ½, doublet singular value")
    a.plot(kk, [overlap_fcc(k, 0.3, 0.5)[1] for k in kk], 's--', ms=5,
           label="heading fidelity (spin traced out)")
    a.plot(kk, [overlap_fcc(k, 0.3, 0)[0][0] for k in kk], '^-', ms=5,
           label="FCC spin 0 (heading only)")
    a.set_xlabel("$|k|$"); a.set_ylabel("overlap")
    a.set_title("spin does not help: the doublet value stays\n"
                "BELOW the heading fidelity", fontsize=9)
    a.legend(fontsize=6); a.grid(alpha=.3)

    a = ax[2]
    pts = [(0.7943, 0.752), (0.6439, 0.568), (0.5160, 0.401), (0.3345, 0.374),
           (float(z['lo_ov']), float(z['lo_c'])), (float(z['hi_ov']), float(z['hi_c']))]
    a.plot([p[0] for p in pts], [p[1] for p in pts], 'o', ms=9)
    a.plot([0, 1], [0, 1], 'k--', lw=1, label="contrast = overlap")
    a.annotate("full Mott zero", (float(z['hi_ov']), float(z['hi_c'])), fontsize=8,
               xytext=(-95, -6), textcoords='offset points',
               arrowprops=dict(arrowstyle='->', lw=.8))
    a.set_xlabel("$|\\langle u(k)|u(-k)\\rangle|$")
    a.set_ylabel("contrast at 90°")
    a.set_title("the contrast tracks the overlap and reaches 1\n"
                "(a scaling guide, not a strict cap)", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)
    a.set_xlim(0, 1.05); a.set_ylim(0, 1.05)

    a = ax[3]
    a.plot(z['th'][:36], z['hi_b'][:36], 'o-', ms=4, color='C0', label="boson")
    a.plot(z['th'][:36], z['hi_f'][:36], 'o-', ms=4, color='C3', label="fermion")
    a.axvline(90, color='k', ls='--', lw=1)
    a.set_yscale('log'); a.set_xlabel("$\\theta$ [deg]")
    a.set_ylabel("scattered density")
    a.set_title("at overlap %.3f: fermion at 90° is\n%.2f %% of the boson"
                % (z['hi_ov'], 100 * ratio), fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    fig.suptitle("Does spin lift the ceiling?  No — but the non-relativistic limit does",
                 y=1.02)
    fig.tight_layout(); fig.savefig("helicity_test.png", bbox_inches='tight')
    plt.close(fig)
    print("\nwrote helicity_test.png")


if __name__ == "__main__":
    main()
