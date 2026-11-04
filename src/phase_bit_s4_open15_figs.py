#!/usr/bin/env python3
"""Figure for the gravitational spin-orbit coupling.  Reads data/phase_bit_s4_open15*.npz."""

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
COL = {'W': S2, 'H2': S1, 'lin': S3}
LAB = {'W': 'nur Warten (γ = 0)', 'H2': 'H2 (γ = β = 1)', 'lin': 'linearer Lapse (γ = 1, β = ½)'}


def title(ax, s):
    ax.set_title(s, loc='left', color=INK, fontsize=10)


def main():
    d = np.load(data_path('phase_bit_s4_open15.npz'), allow_pickle=True)
    m = 2 * float(d['A_par'][0])
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) phase difference of the two spin signals: slope = fine structure
    a = ax[0, 0]
    rows = d['C_rows'] if 'C_rows' in d else None
    for k, v in enumerate(('W', 'H2', 'lin')):
        pu, pd = (data_path(f'phase_bit_s4_open15_B_{v}_{s}.npz') for s in ('up', 'dn'))
        if not (os.path.exists(pu) and os.path.exists(pd)):
            continue
        su, sd = np.load(pu)['sig'], np.load(pd)['sig']
        t = 0.5 * np.arange(len(su))
        ph = -np.unwrap(np.angle(su * np.conj(sd)))
        a.plot(t, ph, color=COL[v], lw=2, label=f'{LAB[v]}: Gitter')
        if rows is not None:
            a.plot(t, rows[k, 6] * t, '--', color=INK, lw=1)
    a.plot([], [], '--', color=INK, lw=1, label='Steigung der Referenz (Dirac + Wilson-Term)')
    a.set_xlabel('Zeit t (Takte × 0.5)')
    a.set_ylabel('Phasendifferenz −arg(c↑(t) c↓(t)*)  (rad)')
    title(a, '(a) 2D-Gitteratom im 3D-Walker: Phasendifferenz der Spin-Läufe\n2p3/2 gegen 2p1/2, die Steigung ist die Feinstruktur')
    a.legend(loc='upper left', fontsize=8)

    # (b) fine structure lattice vs reference
    b = ax[0, 1]
    if 'C_rows' in d and len(d['C_rows']):
        r = d['C_rows']
        names = ['W', 'H2', 'lin'][:len(r)]
        x = np.arange(len(r))
        b.bar(x - 0.27, r[:, 5] * 1e4, 0.25, color=[COL[nm] for nm in names], label='Gitter (Walker)')
        b.bar(x, r[:, 6] * 1e4, 0.25, color=[COL[nm] for nm in names], alpha=0.45, label='Dirac in der Metrik + Wilson-Term')
        b.bar(x + 0.27, r[:, 7] * 1e4, 0.25, color='none', edgecolor=[COL[nm] for nm in names], lw=1.5,
              label='Dirac in der Metrik (rein)')
        for xi, row in zip(x, r):
            b.text(xi - 0.27, row[5] * 1e4 + 0.08, f'{row[5]/row[6]:.2f}', ha='center', fontsize=8, color=INK)
        b.set_xticks(x); b.set_xticklabels([LAB[nm] for nm in names], fontsize=8)
        b.legend(loc='upper left', fontsize=8)
    b.set_ylabel('Feinstruktur E(2p3/2) − E(2p1/2)  (10⁻⁴)')
    title(b, '(b) Feinstruktur: Gitter gegen Dirac-Gleichung in der Metrik\n(Zahl über dem Balken: Gitter / Referenz)')

    # (c) reference: ratio to W vs coupling, FW limit 1 : 3 : 3 (A = N: -1)
    c = ax[1, 0]
    R = d['A_ratios']                       # af, flat, W, H2, lin
    for j, (nm, col, lab) in enumerate((('flat', MUTED, 'A = N (γ = −1)'), ('W', S2, LAB['W']),
                                        ('H2', S1, LAB['H2']), ('lin', S3, LAB['lin']))):
        c.plot(R[:, 0], R[:, 1 + j] / R[:, 2], 'o-', color=col, ms=7, label=lab)
    for yv in (-1, 1, 3):
        c.axhline(yv, color=MUTED, lw=0.8, ls='--')
    if 'C_rows' in d and len(d['C_rows']) == 3:
        r = d['C_rows']
        af = float(d['A_par'][1])
        c.plot([af, af], [r[1, 5] / r[0, 5], r[2, 5] / r[0, 5]], '*', color=INK, ms=14, label='Gitter (Verhältnis zu W)')
    c.set_xlabel('Kopplung α_f = m GM / c′')
    c.set_ylabel('Feinstruktur relativ zu „nur Warten“')
    title(c, '(c) Dirac-Referenz: im schwachen Feld −1 : 1 : 3 : 3 = (2γ + 1),\nder de-Sitter-Faktor; starkes Feld hebt H2 und linear an')
    c.legend(loc='upper left', fontsize=8)

    # (d) absolute levels lattice vs reference
    dd = ax[1, 1]
    if 'C_rows' in d and len(d['C_rows']):
        r = d['C_rows']
        names = ['W', 'H2', 'lin'][:len(r)]
        for nm, row in zip(names, r):
            dd.plot(-row[3] * 1e3, -row[1] * 1e3, 'o', color=COL[nm], ms=9, label=f'{LAB[nm]}: 2p3/2')
            dd.plot(-row[4] * 1e3, -row[2] * 1e3, 's', color=COL[nm], ms=9, mfc='none', mew=2, label=f'{LAB[nm]}: 2p1/2')
        lo, hi = 3.4, 5.2
        dd.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls='--')
        dd.legend(loc='upper left', fontsize=7)
    dd.set_xlabel('Bindungsenergie, Referenz (10⁻³)')
    dd.set_ylabel('Bindungsenergie, Gitter (10⁻³)')
    title(dd, '(d) Absolute 2p-Niveaus: Gitter gegen Referenz')

    fig.suptitle('Gravitative Spin-Bahn-Kopplung: Feinstruktur des Gravitationsatoms', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open15.png'), dpi=120)
    print('saved figures/phase_bit_s4_open15.png')


if __name__ == '__main__':
    main()
