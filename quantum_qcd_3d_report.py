#!/usr/bin/env python3
"""
Report and overview figure for the two structural questions:
  (A) does the 2D turning-number theorem survive in 3D?   -> spin rigidity
  (B) what does the machinery need for QCD?               -> SU(3) colour
Run:  python3 quantum_qcd_3d_report.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_fcc_holonomy import (closed_holonomies, hex_2d_control,
                                  commutator_across_headings, FCC_UNIT)
from quantum_hex_su3 import (make_links, wilson, walk_loop, gauge_transform,
                             loop_average, rhombus_path, PLAQ_UP,
                             PLAQ_DOWN_PARTNER)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def part_A():
    print("[A] Does the turning-number theorem survive in 3D?")
    h2 = hex_2d_control(6)
    h3 = closed_holonomies(6)
    out = {}
    for nm, h in (("2D triangular", h2), ("3D FCC", h3)):
        phi = np.array([p for L in h.values() for _, p, _ in L])
        res = np.array([r for L in h.values() for _, _, r in L])
        frac = phi / (2 * np.pi)
        dev = np.abs(frac - np.round(frac)).max()
        print(f"  {nm:14s} {len(phi):5d} closed walks   rotation axis = initial "
              f"heading to {res.max():.1e}")
        print(f"  {'':14s} distinct phi/2pi values: {len(set(np.round(frac,6))):3d}"
              f"   max deviation from an integer: {dev:.6f}")
        out[nm] = frac
    mx, na, nb = commutator_across_headings(5)
    print(f"  holonomies based at DIFFERENT headings: max |QaQb - QbQa| = {mx:.4f}"
          f"  ({na} x {nb})")
    print("  => 2D: winding number w in Z, any alpha allowed (anyons).")
    print("     3D: no winding number, non-abelian holonomy; only the Z_2 centre")
    print("         of SU(2) survives -> alpha in {0, 1/2}: boson or fermion.\n")
    return out


def part_B(seed=7):
    print("[B] SU(3) colour: what changes")
    rng = np.random.default_rng(seed)
    L = 16
    U = make_links(L, 0.8, rng, "su3")
    V = gauge_transform(U, rng)
    g_tri = max(abs(wilson(U, (i, j), PLAQ_UP) - wilson(V, (i, j), PLAQ_UP))
                for i in range(L) for j in range(L))
    g_rho = max(abs(wilson(U, (i, j), rhombus_path(2, 3))
                    - wilson(V, (i, j), rhombus_path(2, 3)))
                for i in range(L) for j in range(L))
    print(f"  gauge invariance of tr W : triangle {g_tri:.2e}, 2x3 rhombus {g_rho:.2e}")

    U1 = make_links(L, 0.8, rng, "u1")
    a = np.angle(wilson(U1, (0, 0), PLAQ_UP)) + \
        np.angle(wilson(U1, (1, 0), PLAQ_DOWN_PARTNER))
    b = np.angle(wilson(U1, (0, 0), rhombus_path(1, 1)))
    add = abs(((a - b) + np.pi) % (2 * np.pi) - np.pi)
    A_, _ = walk_loop(U, (0, 0), PLAQ_UP)
    B_, _ = walk_loop(U, (0, 0), [(+1, 0)] + PLAQ_DOWN_PARTNER + [(-1, 0)])
    Bn, _ = walk_loop(U, (1, 0), PLAQ_DOWN_PARTNER)
    R_, _ = walk_loop(U, (0, 0), rhombus_path(1, 1))
    comp_ok = abs(np.trace(B_ @ A_) / 3 - np.trace(R_) / 3)
    comp_no = abs(np.trace(Bn @ A_) / 3 - np.trace(R_) / 3)
    noncom = np.abs(A_ @ Bn - Bn @ A_).max()
    print(f"  U(1) fluxes add exactly            : {add:.2e}")
    print(f"  SU(3) with a common base point     : {comp_ok:.2e}  (correct rule)")
    print(f"  SU(3) transporter omitted          : {comp_no:.4f}  (wrong)")
    print(f"  SU(3) plaquettes do not commute    : {noncom:.4f}")

    print("  Wilson loops in a STATIC random background:")
    print("    g    c=<U>/N   (a,b)  perim area   <W>       c^perim")
    data = {}
    for g in (0.6, 1.0):
        Ug = make_links(24, g, np.random.default_rng(11), "su3")
        c = np.trace(Ug.reshape(-1, 3, 3).mean(axis=0)).real / 3
        rows = []
        for (aa, bb) in [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]:
            m, _ = loop_average(Ug, rhombus_path(aa, bb))
            rows.append((2 * (aa + bb), 2 * aa * bb, m, c ** (2 * (aa + bb))))
            tag = f"    {g:.1f}   {c:.5f}  " if (aa, bb) == (1, 1) else " " * 18
            print(tag + f"({aa},{bb})   {rows[-1][0]:4d} {rows[-1][1]:4d} "
                        f"{m:9.5f}  {rows[-1][3]:9.5f}")
        data[g] = rows
    print("  => a static random background gives a PERIMETER law, i.e. NO")
    print("     confinement.  An area law needs the links correlated by the")
    print("     Wilson action (Monte Carlo).  See ROADMAP_QCD_3D_de.md.\n")
    return data


def figure(fracs, wl, fname="qcd_3d_overview.png"):
    fig, ax = plt.subplots(1, 4, figsize=(16.5, 3.9))

    a = ax[0]
    for nm, col, off in (("2D triangular", 'C0', -0.12), ("3D FCC", 'C3', 0.12)):
        v, n = np.unique(np.round(fracs[nm], 6), return_counts=True)
        a.bar(v + off * 0, n / n.sum(), width=0.02,
              color=col, alpha=.75, label=f"{nm} ({len(v)} values)")
    for x in (-1, 0, 1):
        a.axvline(x, color='k', ls=':', lw=.8)
    a.set_xlabel("$\\phi\\,/\\,2\\pi$ of the closed-walk holonomy")
    a.set_ylabel("relative frequency")
    a.set_title("2D: only integers (winding number)\n3D: not quantised", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[1]
    v = np.unique(np.round(fracs["3D FCC"], 6))
    a.plot(v, np.cos(np.pi * v), 'o', ms=6)
    for c, lab in [(1/3, "1/3"), (1/np.sqrt(3), "$1/\\sqrt{3}$"),
                   (np.sqrt(2/3), "$\\sqrt{2/3}$"), (2*np.sqrt(2)/3, "$2\\sqrt{2}/3$")]:
        a.axhline(c, color='gray', ls=':', lw=.8)
        a.axhline(-c, color='gray', ls=':', lw=.8)
        a.text(1.02, c, lab, fontsize=7, va='center')
    a.set_xlabel("$\\phi\\,/\\,2\\pi$"); a.set_ylabel("$\\cos(\\phi/2)$")
    a.set_title("FCC holonomies: algebraic but\nirrational multiples of $2\\pi$",
                fontsize=9)
    a.grid(alpha=.3)

    a = ax[2]
    a.axis('off')
    a.text(0.0, 0.97, "Why spin becomes rigid in 3D", fontsize=10, weight='bold',
           va='top')
    a.text(0.0, 0.80,
           "2D:  headings live on a circle $S^1$\n"
           "      $\\pi_1(S^1)=\\mathbb{Z}$  →  a flux can be threaded\n"
           "      loop phase $e^{i2\\pi\\alpha w}$, any $\\alpha$\n"
           "      → anyons; $\\alpha$ is a free knob\n\n"
           "3D:  headings live on a sphere $S^2$\n"
           "      $\\pi_1(S^2)=0$  →  no flux to thread\n"
           "      frames live in $SO(3)$, covered by $SU(2)$\n"
           "      $SU(2)$ is simple: no continuous character\n"
           "      only the centre $\\mathbb{Z}_2$ survives\n"
           "      → $\\alpha\\in\\{0,\\frac{1}{2}\\}$: boson or fermion",
           fontsize=8.5, va='top', family='monospace')

    a = ax[3]
    for g, col in ((0.6, 'C0'), (1.0, 'C3')):
        rows = wl[g]
        P = np.array([r[0] for r in rows]); A = np.array([r[1] for r in rows])
        W = np.array([r[2] for r in rows]); C = np.array([r[3] for r in rows])
        a.plot(P, W, 'o', color=col, ms=7, label=f"$g={g}$ measured")
        a.plot(P, C, '-', color=col, lw=1.4, label=f"$g={g}$: $c^{{\\rm perimeter}}$")
    a.set_yscale('log')
    a.set_xlabel("loop perimeter"); a.set_ylabel("$\\langle\\frac{1}{3}\\,{\\rm Re\\,tr}\\,W\\rangle$")
    a.set_title("static SU(3) background:\nperimeter law → no confinement", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    fig.suptitle("Two structural questions: spin needs 3D, and confinement needs a "
                 "dynamical gauge field", y=1.04)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("wrote", fname)


if __name__ == "__main__":
    print("=" * 74)
    print("Spin in 3D, and SU(3) colour: what the model already has")
    print("=" * 74 + "\n")
    fr = part_A()
    wl = part_B()
    figure(fr, wl)
