#!/usr/bin/env python3
"""Figure for the gravitational hyperfine structure.  Reads data/phase_bit_s4_open16*.npz and the J = 0 runs of open15."""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import data_path, fig_path

S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3de'
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0, 'legend.frameon': False,
})
NAME = {'up': '2p3/2', 'dn': '2p1/2'}
COL = {'up': S1, 'dn': S2}


def title(ax, s):
    ax.set_title(s, loc='left', color=INK, fontsize=10)


def main():
    d = np.load(data_path('phase_bit_s4_open16.npz'), allow_pickle=True)
    A, C = d['A_rows'], d['C_rows']
    bj, rcbf = float(d['A_par'][4]), float(d['A_par'][5])
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) phase of the J run against the J = 0 run: slope = Lense-Thirring shift
    a = ax[0, 0]
    for spin in ('up', 'dn'):
        s0 = np.load(data_path(f'phase_bit_s4_open15_B_H2_{spin}.npz'))['sig']
        for rule, ls in (('norule', '-'), ('rule', '-')):
            p = data_path(f'phase_bit_s4_open16_B_{spin}_p_{rule}.npz')
            if not os.path.exists(p):
                continue
            s = np.load(p)['sig']
            n = min(len(s), len(s0))
            t = 0.5 * np.arange(n)
            ph = -np.unwrap(np.angle(s[:n] * np.conj(s0[:n])))
            a.plot(t, ph * 1e3, ls, color=COL[spin], alpha=1.0 if rule == 'rule' else 0.45, lw=2,
                   label=f"{NAME[spin]}, {'Bahn + Spin' if rule == 'rule' else 'nur Bahn'}")
            kap = 1.5 if spin == 'up' else -0.5
            row = C[(C[:, 0] == (rule == 'rule')) & (C[:, 1] == kap)]
            if len(row):
                ref_full = row[0, 3] + (row[0, 5] if row[0, 7] else 0)   # E(+J) - E(0) of the reference
                a.plot(t, ref_full * t * 1e3, '--', color=INK, lw=1)
    a.plot([], [], '--', color=INK, lw=1, label='Steigung der Referenz')
    a.set_xlabel('Zeit t (Takte × 0.5)')
    a.set_ylabel('Phase −arg(c_J(t) c_0(t)*)  (10⁻³ rad)')
    title(a, '(a) Phase des Laufs mit Drehimpuls J gegen J = 0\n(Steigung = Lense-Thirring-Verschiebung)')
    a.legend(loc='upper left', fontsize=8)

    # (b) shifts lattice vs reference
    b = ax[0, 1]
    labs, lat, ref, cols, alph = [], [], [], [], []
    for rl, rlab in ((0, 'nur Bahn'), (1, 'Bahn + Spin')):
        for spin, kap in (('up', 1.5), ('dn', -0.5)):
            row = C[(C[:, 0] == rl) & (C[:, 1] == kap)]
            if not len(row):
                continue
            labs.append(f'{NAME[spin]}\n{rlab}')
            lat.append(row[0, 2]); ref.append(row[0, 3]); cols.append(COL[spin]); alph.append(0.5 + 0.5 * rl)
    x = np.arange(len(labs))
    b.bar(x - 0.2, np.array(lat) * 1e5, 0.38, color=cols, label='Gitter (3D-Walker mit Mitführ-Hop)')
    b.bar(x + 0.2, np.array(ref) * 1e5, 0.38, color='none', edgecolor=cols, lw=1.8, label='Dirac in der Metrik + b-Terme')
    for xi, l_, r_ in zip(x, lat, ref):
        b.text(xi - 0.2, l_ * 1e5 + 0.12, f'{l_/r_:.3f}', ha='center', fontsize=8, color=INK)
    b.set_xticks(x); b.set_xticklabels(labs, fontsize=8)
    b.set_ylabel('Verschiebung durch J  (10⁻⁵)')
    title(b, f'(b) Lense-Thirring-Verschiebung der 2p-Niveaus (B_J = {bj}, r_c,b = {rcbf} a0)\n'
             '(Zahl über dem Balken: Gitter / Referenz)')
    b.legend(loc='upper left', fontsize=8)

    # (c) spin part = (orbit + spin) - orbit
    c = ax[1, 0]
    sl, sr, nm = [], [], []
    for spin, kap in (('up', 1.5), ('dn', -0.5)):
        r1 = C[(C[:, 0] == 1) & (C[:, 1] == kap)]
        r0 = C[(C[:, 0] == 0) & (C[:, 1] == kap)]
        if not (len(r1) and len(r0)):
            continue
        # bring every row to the odd part (E(+J) - E(-J))/2: rows without a -J run lose the reference even part
        odd = lambda q, col: q[0, col] - (0 if q[0, 7] else q[0, 5])
        l0, l1 = odd(r0, 2), odd(r1, 2)
        f0, f1 = odd(r0, 3), odd(r1, 3)
        sl.append(l1 - l0); sr.append(f1 - f0); nm.append(NAME[spin])
    fo = {1.5: A[0, 6], -0.5: A[1, 6]}
    if sl:
        x = np.arange(len(sl))
        c.bar(x - 0.2, np.array(sl) * 1e6, 0.38, color=[COL[s] for s in ('up', 'dn')][:len(sl)], label='Gitter (Spinregel)')
        c.bar(x + 0.2, np.array(sr) * 1e6, 0.38, color='none', edgecolor=[COL[s] for s in ('up', 'dn')][:len(sl)], lw=1.8,
              label='Referenz (exakt in B_J, ungerader Anteil)')
        c.plot(x + 0.2, np.array([fo[1.5], fo[-0.5]][:len(sl)]) * 1e6, 'D', color=INK, ms=7, label='Referenz, 1. Ordnung')
        for xi, l_, r_ in zip(x, sl, sr):
            c.text(xi - 0.2, l_ * 1e6 + (0.4 if l_ > 0 else -1.0), f'{l_/r_:.3f}', ha='center', fontsize=8, color=INK)
        c.set_xticks(x); c.set_xticklabels(nm)
        c.axhline(0, color=MUTED, lw=0.8)
        c.legend(loc='upper left', fontsize=8)
    c.set_ylabel('Spinanteil (Bahn + Spin) − Bahn  (10⁻⁶)')
    title(c, '(c) Spinanteil ½ (rot b)·J_int: Vorzeichen folgt s_z\n(gyrogravitomagnetischer Faktor 1)')

    # (d) radial profiles of the two couplings, scaled by rho^3 / B_J, with the 2p densities
    dd = ax[1, 1]
    import phase_bit_s4_open15 as o15
    import phase_bit_s4_open16 as o16
    from quantum_fcc_3d import LAT_A
    m, GM, a0, rc = o15.params()
    rcb = rcbf * a0
    r = np.linspace(0.02, 6 * a0, 800)
    dd.plot(r / a0, o16.b_phi(r, bj, rcb) * r ** 2 / bj, color=S3, label='Bahn: (b_φ/ρ)·ρ³/B_J  (Fernfeld 1)')
    dd.plot(r / a0, o16.curl_z(r, bj, rcb) / 2 * r ** 3 / bj, color=S4, label='Spin: ½ (rot b)_z·ρ³/B_J  (Fernfeld −½)')
    for yv in (1, -0.5):
        dd.axhline(yv, color=MUTED, lw=0.8, ls='--')
    dd.axhline(0, color=MUTED, lw=0.8)
    # 2p densities (large component) of the reference without b
    Nfun, Ffun, nexp, _ = o15.VARIANTS['H2']
    U = lambda x: GM / np.sqrt(x ** 2 + rc ** 2)
    Et = m - 0.42 * m * o15.AF ** 2 / 2
    d2 = dd.twinx()
    for kap, spin in ((1.5, 'up'), (-0.5, 'dn')):
        w, v, gP, gQ = o15.dirac_radial(kap, m, lambda x: Nfun(U(x)), lambda x: Ffun(U(x)), o15.NBOX / 2 * LAT_A,
                                        8000, Et, k=3, rb=o15.lattice_rb(o15.MU), nexp=nexp or 0)
        i0 = np.argmin(np.abs(w - Et))
        P = v[:len(gP), i0]
        d2.plot(gP / a0, P ** 2 / np.diff(gP).mean(), ':', color=COL[spin], lw=1.5, label=f'Dichte P² {NAME[spin]}')
    d2.set_ylim(bottom=0)
    d2.set_ylabel('radiale Dichte P²', color=MUTED)
    d2.grid(False)
    d2.spines['right'].set_visible(True)
    dd.set_xlim(0, 6)
    dd.set_ylim(-0.7, 1.15)
    dd.set_xlabel('ρ / a0')
    dd.set_ylabel('Kopplung × ρ³ / B_J')
    title(dd, f'(d) Radiale Profile (r_c,b = {rcbf} a0) und 2p-Dichten:\nim Kern sind beide unterdrückt, rot b kehrt das Vorzeichen um')
    h1, l1 = dd.get_legend_handles_labels(); h2, l2 = d2.get_legend_handles_labels()
    dd.legend(h1 + h2, l1 + l2, loc='lower right', fontsize=8)

    fig.suptitle('Gravitative Hyperfeinstruktur: Lense-Thirring-Kopplung des 2p-Atoms an den Drehimpuls J',
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open16.png'), dpi=120)
    print('saved figures/phase_bit_s4_open16.png')


if __name__ == '__main__':
    main()
