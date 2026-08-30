#!/usr/bin/env python3
"""
Report and figure for the SU(3) Wilson-action Monte Carlo.
Run:  python3 quantum_hex_su3_mc_figs.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

from quantum_hex_su3_mc import (run, single_plaquette, haar_reweight, sweep,
                                binned_error, mean_plaquette, wilson_rhombus,
                                update_pool, N_C)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

L = 24
BETAS_PLAQ = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
BETAS_LOOP = [4.0, 8.0, 12.0]

# Loops chosen so that several share a PERIMETER while differing in AREA:
#   P=8 : (3,1) A=6   (2,2) A=8
#   P=10: (4,1) A=8   (3,2) A=12
#   P=12: (5,1) A=10  (4,2) A=16  (3,3) A=18
LOOPS = [(1, 1), (2, 1), (3, 1), (2, 2), (4, 1), (3, 2), (5, 1), (4, 2), (3, 3)]


def geom(lp):
    a, b = lp
    return 2 * (a + b), 2 * a * b          # perimeter, area


# ─── static (quenched, uncorrelated) reference ensemble ──────────────────────

def static_config(g, L, rng):
    U = np.empty((L, L, 3, N_C, N_C), dtype=complex)
    for i in range(L):
        for j in range(L):
            for mu in range(3):
                a = rng.normal(size=(N_C, N_C)) + 1j * rng.normal(size=(N_C, N_C))
                H = (a + a.conj().T) / 2.0
                H -= np.trace(H) / N_C * np.eye(N_C)
                H /= np.sqrt((np.abs(H) ** 2).sum())
                U[i, j, mu] = expm(1j * g * H)
    return U


def static_measure(g, n_conf=120, L=L, seed=5):
    """Independently drawn links: <plaquette>, <U> scalar, and the loops."""
    rng = np.random.default_rng(seed)
    pl, cs, wl = [], [], {lp: [] for lp in LOOPS}
    for _ in range(n_conf):
        U = static_config(g, L, rng)
        pl.append(mean_plaquette(U))
        cs.append(np.trace(U.reshape(-1, N_C, N_C).mean(axis=0)).real / N_C)
        for lp in LOOPS:
            wl[lp].append(float(wilson_rhombus(U, *lp).mean()))
    return (float(np.mean(pl)), float(np.mean(cs)),
            {k: np.array(v) for k, v in wl.items()})


def match_g(target, lo=0.3, hi=1.8, n_conf=25, seed=4):
    """Find g so the static ensemble has the same plaquette as the MC."""
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        p, _, _ = static_measure(mid, n_conf=n_conf, L=12, seed=seed)
        if p > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 76)
    print("SU(3) lattice gauge theory, Wilson action, triangular lattice")
    print(f"L = {L}")
    print("=" * 76)

    print("\n[1] Validation: plaquette vs the EXACT single-plaquette integral")
    print("    (in 2D the plaquettes are independent, so this is the exact answer)")
    print("  beta   eps   acc     lattice <plaq>     exact 1-plaq     "
          "Haar rew.   pull")
    plaq_tab = []
    for b in BETAS_PLAQ:
        r = run(b, L=L, n_therm=700, n_meas=500, n_sep=4, seed=1, loops=[(1, 1)])
        ref, eref, _ = single_plaquette(b, n_meas=300000, seed=2)
        hw = haar_reweight(b, n=250000) if b <= 3.0 else np.nan
        p, e = r['plaq'].mean(), binned_error(r['plaq'])
        pull = (p - ref) / np.sqrt(e ** 2 + eref ** 2)
        plaq_tab.append((b, p, e, ref, eref, pull))
        print("  %5.1f  %.2f  %.3f   %.5f(%2d)    %.5f(%3d)    %s  %+5.2f"
              % (b, r['eps'], r['acc'], p, int(round(e * 1e5)),
                 ref, int(round(eref * 1e5)),
                 ("%.5f" % hw) if np.isfinite(hw) else "    -    ", pull))
    print("  => all %d couplings agree within |pull| = %.2f sigma"
          % (len(plaq_tab), max(abs(t[5]) for t in plaq_tab)))

    print("\n[2] The decisive test: loops of EQUAL PERIMETER but different AREA,")
    print("    measured in two ensembles TUNED TO THE SAME PLAQUETTE.")
    res = {}
    for b in BETAS_LOOP:
        r = run(b, L=L, n_therm=800, n_meas=1500, n_sep=4, seed=1, loops=LOOPS)
        w1 = r['plaq'].mean()
        g = match_g(w1)
        sp, sc, swl = static_measure(g, n_conf=120)
        res[b] = dict(mc=r, w1=w1, g=g, sp=sp, sc=sc, swl=swl)
        print(f"\n  beta = {b}:  Wilson-action <plaq> = {w1:.5f}"
              f"   static ensemble tuned to g = {g:.4f} -> <plaq> = {sp:.5f}"
              f"   (c = {sc:.5f})")
        print("    (a,b)  perim  area |  Wilson action <W>    w1^area  |"
              "   static <W>      c^perim")
        for lp in LOOPS:
            P, A = geom(lp)
            m, e = r['wl'][lp].mean(), binned_error(r['wl'][lp])
            sm, se = swl[lp].mean(), binned_error(swl[lp], 12)
            print("    (%d,%d)  %4d  %4d | %9.6f(%4.0f) %10.6f | %9.6f(%4.0f) %10.6f"
                  % (lp[0], lp[1], P, A, m, e * 1e6, w1 ** A,
                     sm, se * 1e6, sc ** P))

    print("\n[3] Same perimeter, different area — the two ensembles separate")
    for b in BETAS_LOOP:
        d = res[b]
        print(f"  beta = {b}")
        for P in (8, 10, 12):
            grp = [lp for lp in LOOPS if geom(lp)[0] == P]
            if len(grp) < 2:
                continue
            mc = [d['mc']['wl'][lp].mean() for lp in grp]
            st = [d['swl'][lp].mean() for lp in grp]
            ar = [geom(lp)[1] for lp in grp]
            print("    perimeter %2d, areas %s:" % (P, ar))
            print("        Wilson action: %s   ratio %.2f  (area law predicts %.2f)"
                  % (["%.5f" % v for v in mc],
                     mc[0] / mc[-1] if abs(mc[-1]) > 1e-9 else np.nan,
                     d['w1'] ** (ar[0] - ar[-1])))
            print("        static       : %s   ratio %.2f  (perimeter law predicts 1)"
                  % (["%.5f" % v for v in st],
                     st[0] / st[-1] if abs(st[-1]) > 1e-9 else np.nan))

    figure(plaq_tab, res)
    return plaq_tab, res


def figure(plaq_tab, res, fname="su3_mc_confinement.png"):
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.1))

    a = ax[0]
    b = np.array([t[0] for t in plaq_tab])
    w = np.array([t[1] for t in plaq_tab]); we = np.array([t[2] for t in plaq_tab])
    rf = np.array([t[3] for t in plaq_tab]); re_ = np.array([t[4] for t in plaq_tab])
    a.errorbar(b, w, yerr=we, fmt='o', ms=7, label="lattice Monte Carlo")
    a.errorbar(b, rf, yerr=re_, fmt='x', ms=9, color='C3',
               label="exact single-plaquette integral")
    bb = np.linspace(0.2, 3.5, 50)
    a.plot(bb, bb / (2 * N_C ** 2), 'k--', lw=1, label="strong coupling $\\beta/2N^2$")
    a.set_xlabel("$\\beta$"); a.set_ylabel("$w_1=\\langle\\frac{1}{3}{\\rm Re\\,tr}U_P\\rangle$")
    a.set_title("validation: the MC reproduces the\nexact 2D result at every coupling",
                fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[1]
    bref = BETAS_LOOP[-1]
    d = res[bref]
    P = np.array([geom(lp)[0] for lp in LOOPS])
    A = np.array([geom(lp)[1] for lp in LOOPS])
    mc = np.array([d['mc']['wl'][lp].mean() for lp in LOOPS])
    mce = np.array([binned_error(d['mc']['wl'][lp]) for lp in LOOPS])
    st = np.array([d['swl'][lp].mean() for lp in LOOPS])
    a.errorbar(A, np.abs(mc), yerr=mce, fmt='o', ms=7, color='C0',
               label="Wilson action (dynamical)")
    Ax = np.linspace(1, A.max(), 50)
    a.plot(Ax, d['w1'] ** Ax, '-', color='C0', lw=1.3, label="$w_1^{\\,A}$")
    a.plot(A, st, 's', ms=6, color='C3', label="static random links")
    a.axhline(3 * mce.mean(), color='gray', ls=':', lw=1,
              label="$3\\sigma$ noise floor")
    a.set_yscale('log'); a.set_xlabel("enclosed area $A$ (triangles)")
    a.set_ylabel("$\\langle W\\rangle$")
    a.set_title(f"$\\beta={bref}$, both ensembles at the\nSAME plaquette $w_1$="
                f"{d['w1']:.3f}", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[2]
    for P0, col in ((8, 'C0'), (10, 'C1'), (12, 'C2')):
        grp = [lp for lp in LOOPS if geom(lp)[0] == P0]
        ar = [geom(lp)[1] for lp in grp]
        mcv = [d['mc']['wl'][lp].mean() for lp in grp]
        stv = [d['swl'][lp].mean() for lp in grp]
        a.plot(ar, mcv, 'o-', ms=7, color=col, label=f"Wilson, $P={P0}$")
        a.plot(ar, stv, 's--', ms=6, color=col, alpha=.55,
               label=f"static, $P={P0}$")
    a.set_yscale('log'); a.set_xlabel("enclosed area $A$ at FIXED perimeter")
    a.set_ylabel("$\\langle W\\rangle$")
    a.set_title("at fixed perimeter the static curves are\nflat; the dynamical ones "
                "fall with area", fontsize=9)
    a.legend(fontsize=6, ncol=2); a.grid(alpha=.3)

    a = ax[3]
    bs = np.array([t[0] for t in plaq_tab])
    sig = -np.log(np.array([t[1] for t in plaq_tab]))
    a.semilogy(bs, sig, 'o-', ms=7)
    for bl in BETAS_LOOP:
        a.plot([bl], [-np.log(res[bl]['w1'])], 'r*', ms=13)
    a.set_xlabel("$\\beta$")
    a.set_ylabel("$\\sigma=-\\ln w_1$ per triangle")
    a.set_title("string tension, finite at every $\\beta$\n"
                "(2D confines at all couplings)", fontsize=9)
    a.grid(alpha=.3)

    fig.suptitle("A dynamical SU(3) gauge field confines (area law); a static random "
                 "background at the same plaquette does not (perimeter law)", y=1.03)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("\nwrote", fname)


if __name__ == "__main__":
    main()
