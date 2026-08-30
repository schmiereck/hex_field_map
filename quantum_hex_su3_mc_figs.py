#!/usr/bin/env python3
"""
Report and figure for the SU(3) Wilson-action Monte Carlo.
Run:  python3 quantum_hex_su3_mc_figs.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_hex_su3_mc import (run, single_plaquette, haar_reweight,
                                binned_error, N_C)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

BETAS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
LOOPS = [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4)]
L = 24


def jack_log(x, n_bin=20):
    """mean and error of log(<x>) by binned jackknife."""
    x = np.asarray(x)
    m = len(x) // n_bin * n_bin
    b = x[:m].reshape(n_bin, -1).mean(axis=1)
    tot = b.mean()
    jk = np.array([np.delete(b, i).mean() for i in range(n_bin)])
    lg = np.log(np.maximum(jk, 1e-12))
    return float(np.log(max(tot, 1e-12))), \
        float(np.sqrt((n_bin - 1) * np.var(lg, ddof=0)))


def main():
    print("=" * 74)
    print("SU(3) lattice gauge theory, Wilson action, triangular lattice")
    print(f"L = {L}, {len(BETAS)} couplings")
    print("=" * 74)

    res = {}
    print("\n[1] plaquette vs the exact single-plaquette integral")
    print("  beta   eps   acc     lattice <plaq>       exact 1-plaq      "
          "Haar reweight   sigma=-ln w1")
    for b in BETAS:
        r = run(b, L=L, n_therm=800, n_meas=600, n_sep=4, seed=1, loops=LOOPS)
        ref, eref, _ = single_plaquette(b, seed=2)
        hw = haar_reweight(b) if b <= 3.0 else np.nan
        p, e = r['plaq'].mean(), binned_error(r['plaq'])
        r['w1'], r['w1_err'], r['ref'], r['ref_err'] = p, e, ref, eref
        res[b] = r
        print("  %5.1f  %.2f  %.3f   %.5f(%2d)      %.5f(%2d)     %s     %.4f"
              % (b, r['eps'], r['acc'], p, int(round(e * 1e5)),
                 ref, int(round(eref * 1e5)),
                 ("%.5f" % hw) if np.isfinite(hw) else "     -   ",
                 -np.log(max(p, 1e-12))))

    print("\n[2] Wilson loops: is <W> = w1^(enclosed triangles)?")
    for b in BETAS:
        r = res[b]
        w1 = r['w1']
        print(f"  beta = {b}")
        print("     (a,b)  area   <W> measured         w1^area      ratio")
        for lp in LOOPS:
            v = r['wl'][lp]
            m, e = v.mean(), binned_error(v)
            A = 2 * lp[0] * lp[1]
            pred = w1 ** A
            print("     (%d,%d)   %3d   %10.6f(%s)  %10.6f   %6.3f"
                  % (lp[0], lp[1], A, m,
                     ("%.0f" % (e * 1e6)).rjust(4), pred,
                     m / pred if pred > 1e-12 else np.nan))

    print("\n[3] Creutz ratios chi(a,b) = -ln[W(a,b)W(a-1,b-1)/(W(a-1,b)W(a,b-1))]")
    print("    exact 2D area law predicts chi = 2*sigma = -2 ln w1 for every (a,b)")
    creutz = {}
    for b in BETAS:
        r = res[b]
        pred = -2 * np.log(max(r['w1'], 1e-12))
        rows = []
        for (a, bb) in [(2, 2), (3, 2), (3, 3), (4, 3), (4, 4)]:
            need = [(a, bb), (a - 1, bb - 1), (a - 1, bb), (a, bb - 1)]
            if not all(t in r['wl'] or tuple(reversed(t)) in r['wl'] for t in need):
                continue
            def get(t):
                return r['wl'][t] if t in r['wl'] else r['wl'][tuple(reversed(t))]
            l1, e1 = jack_log(get(need[0]))
            l2, e2 = jack_log(get(need[1]))
            l3, e3 = jack_log(get(need[2]))
            l4, e4 = jack_log(get(need[3]))
            chi = -(l1 + l2 - l3 - l4)
            err = np.sqrt(e1**2 + e2**2 + e3**2 + e4**2)
            rows.append(((a, bb), chi, err))
        creutz[b] = (rows, pred)
        print("  beta=%4.1f  prediction %.4f   measured: " % (b, pred)
              + "  ".join("%s %.4f(%.0f)" % (str(k), c, e * 1e4) for k, c, e in rows))

    figure(res, creutz)
    return res, creutz


def figure(res, creutz, fname="su3_mc_confinement.png"):
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8))

    a = ax[0, 0]
    for b in (1.0, 4.0, 12.0):
        th = res[b]['therm']
        a.plot(np.arange(len(th)) * 10, th, lw=1.5, label=f"$\\beta={b}$")
        a.axhline(res[b]['w1'], color='k', ls=':', lw=.7)
    a.set_xlabel("sweep"); a.set_ylabel("$\\langle\\frac{1}{3}{\\rm Re\\,tr}\\,U_P\\rangle$")
    a.set_title("thermalisation from a hot start\n(dotted: equilibrium value)", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[0, 1]
    bs = np.array(BETAS)
    w = np.array([res[b]['w1'] for b in BETAS])
    we = np.array([res[b]['w1_err'] for b in BETAS])
    rf = np.array([res[b]['ref'] for b in BETAS])
    a.errorbar(bs, w, yerr=we, fmt='o', ms=6, label="lattice MC")
    a.plot(bs, rf, 'x', ms=8, color='C3', label="exact single plaquette")
    bb = np.linspace(0.2, 3.5, 50)
    a.plot(bb, bb / (2 * N_C ** 2), 'k--', lw=1,
           label="strong coupling $\\beta/2N^2$")
    a.set_xlabel("$\\beta$"); a.set_ylabel("$w_1$")
    a.set_title("plaquette: lattice vs the exact\n2D single-plaquette integral", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[0, 2]
    for b, col in zip((2.0, 4.0, 8.0, 12.0), ('C0', 'C1', 'C2', 'C3')):
        r = res[b]
        A = np.array([2 * p * q for p, q in LOOPS])
        m = np.array([r['wl'][lp].mean() for lp in LOOPS])
        e = np.array([binned_error(r['wl'][lp]) for lp in LOOPS])
        o = np.argsort(A)
        a.errorbar(A[o], np.abs(m[o]), yerr=e[o], fmt='o', ms=5, color=col,
                   label=f"$\\beta={b}$")
        Ax = np.linspace(1, A.max(), 50)
        a.plot(Ax, r['w1'] ** Ax, '-', color=col, lw=1.2)
    a.set_yscale('log')
    a.set_xlabel("enclosed elementary triangles $A$")
    a.set_ylabel("$\\langle W\\rangle$")
    a.set_title("area law: points = measured,\nlines = $w_1^{A}$ (no free parameter)",
                fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[1, 0]
    for b, col in zip(BETAS, plt.cm.viridis(np.linspace(0, .9, len(BETAS)))):
        rows, pred = creutz[b]
        if not rows:
            continue
        x = np.arange(len(rows))
        a.errorbar(x, [c for _, c, _ in rows], yerr=[e for _, _, e in rows],
                   fmt='o-', ms=4, color=col, label=f"$\\beta={b}$")
        a.axhline(pred, color=col, ls=':', lw=1)
        if b == BETAS[0]:
            a.set_xticks(x); a.set_xticklabels([str(k) for k, _, _ in rows], fontsize=7)
    a.set_yscale('log')
    a.set_ylabel("Creutz ratio $\\chi(a,b)$")
    a.set_title("Creutz ratios are flat and equal\n$-2\\ln w_1$ (dotted)", fontsize=9)
    a.legend(fontsize=6, ncol=2); a.grid(alpha=.3)

    a = ax[1, 1]
    sig = np.array([-np.log(max(res[b]['w1'], 1e-12)) for b in BETAS])
    a.semilogy(bs, sig, 'o-', ms=6)
    a.set_xlabel("$\\beta$"); a.set_ylabel("$\\sigma = -\\ln w_1$  per triangle")
    a.set_title("string tension: finite for every $\\beta$\n"
                "(2D confines at all couplings)", fontsize=9)
    a.grid(alpha=.3)

    a = ax[1, 2]
    for b, col in zip((2.0, 4.0, 8.0), ('C0', 'C1', 'C2')):
        r = res[b]
        Rs, Vs = [], []
        for (R, T) in [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4)]:
            if (R, T) not in r['wl']:
                continue
            m = r['wl'][(R, T)].mean()
            if m <= 1e-9:
                continue
            Rs.append(R); Vs.append(-np.log(m) / T)
        a.plot(Rs, Vs, 'o', ms=6, color=col, label=f"$\\beta={b}$")
        x = np.linspace(0, max(Rs), 20)
        a.plot(x, 2 * (-np.log(r['w1'])) * x, '-', color=col, lw=1.1)
    a.set_xlabel("$R$"); a.set_ylabel("$V(R) = -\\ln W(R,T)/T$")
    a.set_title("static potential rises linearly:\n$V(R)=2\\sigma R$ (lines)", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    fig.suptitle("SU(3) with the Wilson action on the triangular lattice: "
                 "a dynamical gauge field confines (area law), "
                 "unlike the static random background (perimeter law)", y=1.01)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("\nwrote", fname)


if __name__ == "__main__":
    main()
