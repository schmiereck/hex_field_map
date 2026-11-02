#!/usr/bin/env python3
"""
Report and figure for the 3+1D FCC walker with an SU(2) coin.
Run:  python3 quantum_fcc_3d_figs.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

from quantum_fcc_3d import *

EPS = 0.1
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def belt_by_length(L_max=12, d0=0):
    QT = [[quat_transport(FCC_UNIT[i], FCC_UNIT[j]) for j in range(N_D)]
          for i in range(N_D)]
    allowed = [[j for j in range(N_D) if ADJ60[i, j]] for i in range(N_D)]
    out = {L: {"+1": 0, "-1": 0} for L in range(1, L_max + 1)}
    phis = []
    u0 = FCC_UNIT[d0]

    def rec(pos, d, q, depth):
        if depth > 0 and pos == (0, 0, 0) and d == d0:
            phi = 2.0 * np.arctan2(float(np.dot(np.array(q[1:]), u0)), q[0])
            f = phi / (2 * np.pi)
            phis.append(f)
            out[depth]["-1" if abs(round(f)) % 2 == 1 else "+1"] += 1
        if depth == L_max:
            return
        if np.sqrt(sum(p * p for p in pos)) / np.sqrt(2.) > (L_max - depth) + 1e-9:
            return
        for nd in allowed[d]:
            rec(tuple(p + s for p, s in zip(pos, FCC_STEPS[nd])),
                nd, qmul(QT[d][nd], q), depth + 1)

    rec((0, 0, 0), d0, (1., 0., 0., 0.), 0)
    return out, np.array(phis)


def main():
    print("=" * 76)
    print("3+1D FCC walker with an SU(2) coin")
    print("=" * 76)

    print("\n[1] geometry and coin")
    print("  12 headings; neighbours per heading: "
          + ", ".join("%d at %.0f deg" % (int(np.isclose(DOT[0], c).sum()),
                                          np.degrees(np.arccos(c)))
                      for c in (0.5, 0.0, -0.5, -1.0)))
    print("  the coin connects only the 60-deg pairs (cuboctahedron, degree 4),")
    print("  so the 180-deg reversal ambiguity of the 2D model never arises.")
    print("  |step| = sqrt(3)/2, dt = 1/2  =>  c = %.6f" % C_LIGHT)
    for spin in (0, 0.5):
        C = coin(EPS, spin)
        n = C.shape[0]
        print("  spin=%.1f: dim %2d, unitarity %.1e, G Hermitian %.1e"
              % (spin, n, np.abs(C @ C.conj().T - np.eye(n)).max(),
                 np.abs(generator(spin) - generator(spin).conj().T).max()))

    print("\n[2] rest spectrum: Kramers doubling is automatic for spin 1/2")
    spec = {}
    for spin in (0, 0.5):
        lam = np.linalg.eigvalsh(generator(spin))
        v, c = np.unique(np.round(lam, 8), return_counts=True)
        spec[spin] = (v, c, -EPS * v / DT_HALF)
        print("  spin=%.1f  " % spin
              + ", ".join("E=%+.4f x%d" % (e, ci)
                          for e, ci in zip(-EPS * v / DT_HALF, c)))
        print("           all multiplicities even? %s"
              % ("YES (Kramers)" if all(x % 2 == 0 for x in c) else "no"))

    print("\n[3] Kramers doubling holds at every k, not just k=0")
    rng = np.random.default_rng(1)
    ksplit = {0: [], 0.5: []}
    kk = np.linspace(0, 1.2, 13)
    for spin in (0, 0.5):
        for k in kk:
            E, _, _ = bands(k * np.array([1, 0.3, 0.7]) / np.linalg.norm([1, .3, .7]),
                            EPS, spin)
            ksplit[spin].append(abs(E[1] - E[0]))
        print("  spin=%.1f  splitting of the lowest pair over |k| in [0, 1.2]: "
              "max %.2e" % (spin, max(ksplit[spin])))

    print("\n[4] the belt trick on the lattice")
    bl, phis = belt_by_length(12)
    print("  closed walks (60-deg turns only), by length and SU(2) class:")
    print("    L     SU(2)=+1   SU(2)=-1")
    for L in sorted(bl):
        if bl[L]["+1"] or bl[L]["-1"]:
            print("   %3d   %8d   %8d" % (L, bl[L]["+1"], bl[L]["-1"]))
    print("  => the shortest closed heading loop (L=6) returns the spinor with -1;")
    print("     +1 first appears at L=12, i.e. two such loops.  That is 2pi vs 4pi.")
    print("  distinct phi/2pi values: %d (not quantised -> no winding number)"
          % len(set(np.round(phis, 6))))

    print("\n[5] mass, dispersion, causality")
    print("  eps      m(spin 0)   /eps      m(spin 1/2)   /eps")
    for eps in (0.02, 0.05, 0.1, 0.2, 0.4):
        m0 = abs(bands(np.zeros(3), eps, 0)[0][0])
        m1 = abs(bands(np.zeros(3), eps, 0.5)[0][0])
        print("  %.2f   %9.5f  %6.3f   %11.5f  %6.3f"
              % (eps, m0, m0 / eps, m1, m1 / eps))
    print("  => m = 8*eps (spin 0) and m = 4*sqrt(3)*eps = %.4f*eps (spin 1/2), exactly"
          % (4 * np.sqrt(3)))
    m = 4 * np.sqrt(3) * EPS
    print("  massive branch vs sqrt(c^2 k^2 + m^2), spin 1/2, eps=%.1f:" % EPS)
    for k in (0.05, 0.1, 0.2, 0.4, 0.8):
        E = abs(bands(np.array([k, 0, 0]), EPS, 0.5)[0][0])
        Er = np.sqrt(C_LIGHT**2 * k**2 + m**2)
        print("     k=%.2f  |E|=%.6f  ref %.6f  rel.dev %+.3f" % (k, E, Er, (E - Er) / Er))
    mx = 0.0
    for _ in range(300):
        k = rng.normal(size=3) * 1.5
        _, V, _ = bands(k, EPS, 0.5)
        for j in range(24):
            mx = max(mx, np.linalg.norm(group_velocity(V[:, j], 0.5)))
    print("  causality: max |v_g| over all bands = %.6f  <  c = %.6f" % (mx, C_LIGHT))

    print("\n[6] isotropy over 3D directions (top band)")
    iso = {}
    for spin in (0, 0.5):
        for kmag in (0.2, 0.5):
            vs = []
            for _ in range(200):
                n = rng.normal(size=3); n /= np.linalg.norm(n)
                _, V, _ = bands(kmag * n, EPS, spin)
                vs.append(np.linalg.norm(group_velocity(V[:, -1], spin)))
            vs = np.array(vs)
            iso[(spin, kmag)] = vs
            print("  spin=%.1f |k|=%.1f  <|v_g|>=%.4f  spread %.1f%%"
                  % (spin, kmag, vs.mean(), 100 * np.ptp(vs) / vs.mean()))

    print("\n[7] a packet moving in a generic 3D direction")
    tracks = []
    for kdir, sig, N in [((1, 0, 0), 4.0, 81), ((1, 1, 1), 4.0, 81),
                         ((1, 2, 3), 6.0, 91)]:
        n = np.array(kdir, float); n /= np.linalg.norm(n)
        r = run_packet3(N, 24, 0.5 * n, eps=EPS, spin=0.5, sigma=sig,
                        band=0, store_every=2)
        A = np.c_[r['t'], np.ones_like(r['t'])]
        v = np.array([np.linalg.lstsq(A, r['r'][:, j], rcond=None)[0][0]
                      for j in range(3)])
        ca = float(v @ r['vg'] / (np.linalg.norm(v) * np.linalg.norm(r['vg'])))
        ang = np.degrees(np.arccos(np.clip(ca, -1, 1)))
        tracks.append((kdir, sig, r, v, ang))
        print("  k || %-10s sigma=%.1f  |v_g(k0)|=%.4f  |v|meas=%.4f  angle %.2f°"
              "  norm %.5f"
              % (str(kdir), sig, np.linalg.norm(r['vg']), np.linalg.norm(v),
                 ang, r['norm'][-1]))
    print("  (the shortfall in |v| is the packet's k-average: it shrinks as sigma grows)")

    figure(spec, ksplit, kk, bl, iso, tracks)


def figure(spec, ksplit, kk, bl, iso, tracks, fname="fcc_3d_spin.png"):
    fig = plt.figure(figsize=(16.5, 8.4))

    a = fig.add_subplot(2, 3, 1, projection='3d')
    P = FCC_UNIT
    a.scatter(P[:, 0], P[:, 1], P[:, 2], s=45, c='C0', depthshade=False)
    for i in range(N_D):
        for j in range(i + 1, N_D):
            if ADJ60[i, j]:
                a.plot(*zip(P[i], P[j]), color='0.5', lw=1)
    a.set_title("12 FCC headings; the coin uses the\n60° edges (cuboctahedron)",
                fontsize=9)
    a.set_xticks([]); a.set_yticks([]); a.set_zticks([])

    a = fig.add_subplot(2, 3, 2)
    for spin, off, col in ((0, -0.18, 'C1'), (0.5, 0.18, 'C0')):
        v, c, E = spec[spin]
        for e, ci in zip(E, c):
            a.plot([off - 0.14, off + 0.14], [e, e], color=col, lw=2.2)
            a.text(off + 0.18, e, f"×{ci}", fontsize=7, va='center', color=col)
    a.set_xlim(-0.6, 0.75); a.set_xticks([-0.18, 0.18])
    a.set_xticklabels(["spin 0\n(boson)", "spin ½\n(fermion)"])
    a.set_ylabel("rest energy $E(k=0)$")
    a.set_title(f"rest spectrum, $\\varepsilon={EPS}$\nspin ½: every multiplicity even",
                fontsize=9)
    a.grid(alpha=.3, axis='y')

    a = fig.add_subplot(2, 3, 3)
    ks = np.linspace(-1.3, 1.3, 141)
    nhat = np.array([1, 0.3, 0.7]); nhat = nhat / np.linalg.norm(nhat)
    Eb = np.array([bands(k * nhat, EPS, 0.5)[0] for k in ks])
    for j in range(Eb.shape[1]):
        a.plot(ks, Eb[:, j], lw=1.0, color='0.65')
    a.plot(ks, Eb[:, 0], lw=2.2, color='C3', label="massive branch (Kramers pair)")
    m = 4 * np.sqrt(3) * EPS
    a.plot(ks, -np.sqrt(C_LIGHT**2 * ks**2 + m**2), 'k--', lw=1,
           label="$-\\sqrt{c^2k^2+m^2}$")
    a.set_xlabel("$|k|$ along a generic direction"); a.set_ylabel("$E$")
    a.set_title("24 bands, spin ½; the band bottom is\nmassive and nearly relativistic",
                fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = fig.add_subplot(2, 3, 4)
    a.semilogy(kk, np.maximum(ksplit[0.5], 1e-17), 'o-', ms=5,
               label="spin ½ (Kramers pair)")
    a.semilogy(kk, np.maximum(ksplit[0], 1e-17), 's-', ms=5, label="spin 0")
    a.set_xlabel("$|k|$"); a.set_ylabel("splitting of the lowest two bands")
    a.set_title("spin ½: the doublet stays degenerate at\nevery $k$ (machine precision)",
                fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = fig.add_subplot(2, 3, 5)
    Ls = [L for L in sorted(bl) if bl[L]["+1"] or bl[L]["-1"]]
    p = [bl[L]["+1"] for L in Ls]; m_ = [bl[L]["-1"] for L in Ls]
    w = 0.4
    a.bar(np.array(Ls) - w / 2, np.maximum(m_, 0.4), w, label="SU(2) $=-1$", color='C3')
    a.bar(np.array(Ls) + w / 2, np.maximum(p, 0.4), w, label="SU(2) $=+1$", color='C0')
    a.set_yscale('log'); a.set_xlabel("closed-walk length $L$")
    a.set_ylabel("number of walks")
    a.set_title("belt trick: the shortest loop ($L=6$) gives $-1$;\n"
                "$+1$ first appears at $L=12$ (two loops)", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = fig.add_subplot(2, 3, 6, projection='3d')
    for (kdir, sig, r, v, ang) in tracks:
        tr = r['r']
        a.plot(tr[:, 0], tr[:, 1], tr[:, 2], '-o', ms=3,
               label=f"$k\\parallel${kdir}, err {ang:.1f}°")
    a.set_title("centre-of-mass tracks in 3D\n(spin ½, massive branch)", fontsize=9)
    a.set_xlabel("x"); a.set_ylabel("y"); a.set_zlabel("z")
    a.legend(fontsize=6)

    fig.suptitle("3+1D FCC walker: spin is no longer a knob — the SU(2) transport "
                 "makes Kramers doubling automatic, and only spin 0 or spin ½ exist",
                 y=1.02)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("\nwrote", fname)


if __name__ == "__main__":
    main()
