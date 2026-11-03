#!/usr/bin/env python3
"""Figure for H1 (antimatter source) and the proper time as mass load.  Reads data/phase_bit_s4_open7.npz."""

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
CP = np.sqrt(3.0) / 3
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0, 'legend.frameon': False,
})


def title(ax, s):
    ax.set_title(s, loc='left', color=INK, fontsize=10)


def main():
    d = np.load(data_path('phase_bit_s4_open7.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) loads of particle and antiparticle
    a = ax[0, 0]
    A1 = d['A1']
    lab = ['Teilchen\nk_y = 0', 'Teilchen\nk_y = 0.3', 'Antiteilchen\nk_y = 0', 'Antiteilchen\nk_y = 0.3']
    x = np.arange(4)
    a.bar(x - 0.27, A1[:, 3], width=0.25, color=S1, label='Last Q = Σe')
    a.bar(x, A1[:, 4], width=0.25, color=S3, label='Massen-Last Q_M = Σe_M')
    a.bar(x + 0.27, A1[:, 5], width=0.25, color=S2, label='Quelle 2Q − Q_M')
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=9)
    a.set_ylabel('Integral über die Kette')
    title(a, '(a) Im Einteilchen-Bild ist die Last des unteren Bandes\nnegativ: Antiteilchen hätten negative aktive Masse')
    a.legend(loc='lower left', fontsize=8)

    # (b) pair: centre of mass and separation
    b = ax[0, 1]
    ts = d['A2_ts']
    for key, col, nm in (('A2_sing', S2, 'Einteilchen-Bild'), ('A2_hole', S1, 'Loch-Bild (Dirac-See)'),
                         ('A2_pp', MUTED, 'zwei Teilchen (Referenz)')):
        xc = d[key]
        cm = 0.5 * (xc[0] + xc[1])
        sep = xc[1] - xc[0]
        lw = 1.2 if key == 'A2_pp' else 2.0
        b.plot(ts, cm - cm[0], color=col, lw=lw, label=f'{nm}: Schwerpunkt')
        b.plot(ts, sep - sep[0], color=col, lw=lw, ls='--', label=f'{nm}: Abstand')
    b.set_xlabel('Zeit t'); b.set_ylabel('Änderung')
    acc = d['A2_sing_acc']
    title(b, '(b) Paar Teilchen + Antiteilchen: im Einteilchen-Bild läuft der\nSchwerpunkt weg (Impuls nicht erhalten), im Loch-Bild ziehen sie sich an')
    b.legend(loc='lower left', fontsize=7.5)

    # (c) proper time: simulation vs lattice band vs Dirac
    c = ax[1, 0]
    r = d['B_rows']   # mu, N, A, ky, sim, dirac, band
    for mu, col, mk in ((0.03, S1, 'o'), (0.15, S2, 's')):
        s = r[:, 0] == mu
        c.plot(r[s, 6], r[s, 4], mk, color=col, ms=7, label=f'μ = {mu}: Simulation gegen Gitterband')
        c.plot(r[s, 5], r[s, 4], mk, color=col, ms=7, mfc='none', label=f'μ = {mu}: Simulation gegen Dirac N√(1 − v²)')
    c.plot([0, 1], [0, 1], color=MUTED, lw=1, ls='--')
    c.set_xlabel('Vorhersage dτ/dt'); c.set_ylabel('Q_M / m  (lokal gemessene Massen-Last)')
    title(c, '(c) Massen-Last = m · dτ/dt: Lapse N, räumliche Metrik A\nund Bewegung in einer Größe (N = 0.4…1, A = 0.9…1.1)')
    c.legend(loc='upper left', fontsize=7.5)

    # (d) budget split: clock part and motion part vs k
    dd = ax[1, 1]
    s = (r[:, 0] == 0.15) & (r[:, 2] == 1.0) & (r[:, 1] == 0.999)
    k = r[s, 3]
    kk = np.linspace(0, 0.65, 100)
    m = 0.3
    E = np.hypot(CP * kk, m)
    dd.plot(kk, (m / E) ** 2, color=S3, label='Uhr-Anteil m·dτ/dt / E = 1/γ² (Dirac)')
    dd.plot(kk, 1 - m ** 2 / E ** 2, color=S2, label='Bewegungs-Anteil p·v / E = v² (Dirac)')
    dd.plot(k, r[s, 7], 'o', color=S3, ms=8, label='Simulation: Q_M / Q (Uhr) und 1 − Q_M / Q (Bewegung)')
    dd.plot(k, 1 - r[s, 7], 'o', color=S2, ms=8)
    dd.set_xlabel('k_y  (μ = 0.15, m = 0.3, N ≈ 1)'); dd.set_ylabel('Anteil an der Rechenlast')
    title(dd, '(d) Bewegung frisst Rechenzeit: E = m·dτ/dt + p·v,\nder Uhr-Anteil sinkt mit der Geschwindigkeit')
    dd.legend(loc='center right', fontsize=8)

    fig.suptitle('H1: aktive schwere Masse der Antimaterie, und die Eigenzeit als Massenanteil der Rechenlast',
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open7.png'), dpi=120)
    print('saved figures/phase_bit_s4_open7.png')


if __name__ == '__main__':
    main()
