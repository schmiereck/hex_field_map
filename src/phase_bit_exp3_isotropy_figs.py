#!/usr/bin/env python3
"""Figure for Experiment 3 (isotropy of shells and of the walker).  Reads data/phase_bit_exp3_isotropy.npz."""

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
CP, C_LIGHT = np.sqrt(3) / 3, np.sqrt(3)
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0, 'legend.frameon': False,
})


def title(ax, s):
    ax.set_title(s, loc='left', color=INK, fontsize=10)


def main():
    d = np.load(data_path('phase_bit_exp3_isotropy.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) geometric shells
    a = ax[0, 0]
    A = d['A_rows']
    lim = d['A_limit']
    for j, (l, col) in enumerate(((4, S1), (6, S2), (8, S3))):
        a.plot(A[:, 0], A[:, 2 + j], 'o-', color=col, ms=6, lw=1.4, label=f'C_{l}, Schale n')
        a.axhline(lim[j], color=col, lw=0.9, ls='--')
    nn = np.logspace(0, np.log10(200), 50)
    a.plot(nn, 1 / (10 * nn ** 2 + 2), color=MUTED, lw=1.2, ls=':', label='zufällige Punkte gleicher Anzahl (≈ 1/N)')
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel('Schale n (Graphabstand)'); a.set_ylabel('Anisotropie-Leistung C_l')
    title(a, '(a) Kuboktaeder-Schalen als „Bloch-Kugel“: C_l sättigt ab n ≈ 8\n'
             'beim Polyeder-Grenzwert (gestrichelt) und wird nie isotrop')
    a.legend(loc='lower left', fontsize=8)

    # (b) walker k space: smooth sources
    b = ax[0, 1]
    r = d['B_rows']
    s0 = r[:, 0] == 0.03
    sig = r[s0, 1]
    b.plot(sig, r[s0, 2], 's-', color=S2, ms=7, label='alle 24 Moden: C_4')
    b.plot(sig, r[s0, 5], 'o-', color=S1, ms=7, label='Dirac-Quartett (Teilchen): C_4')
    b.plot(sig, r[s0, 6], 'o--', color=S1, ms=5, mfc='none', lw=1.0, label='Dirac-Quartett: C_6')
    ss = np.array([2.0, 32.0])
    b.plot(ss, r[s0, 5][1] * (ss / 2) ** -4, color=INK, lw=1.0, ls=':', label='∝ σ⁻⁴ (Amplitude ∝ (a/σ)²)')
    fl = d['B_floor_0.03']
    b.axhline(fl[1], color=MUTED, lw=0.8, ls='--', label='Grenze der Quadratur')
    b.set_xscale('log', base=2); b.set_yscale('log')
    b.set_xlabel('Breite σ der Quelle'); b.set_ylabel('C_l der Bewegungsrichtungen (Langzeitgrenze)')
    title(b, '(b) Walker, glatte Quelle der Breite σ: das Teilchen wird beliebig\nisotrop (C_4 ∝ σ⁻⁴), die übrigen Moden bleiben kubisch verzerrt')
    b.legend(loc='lower left', fontsize=8)

    # (c) real-space point source
    c = ax[1, 0]
    Cr = d['C_rows']
    pt = d['B_point_0.03']
    for j, (l, col) in enumerate(((4, S1), (6, S2), (8, S3))):
        c.plot(Cr[:, 0], Cr[:, 1 + j], 'o-', color=col, ms=7, label=f'Simulation C_{l}')
        c.axhline(pt[j], color=col, lw=0.9, ls='--')
    c.plot([], [], color=MUTED, lw=0.9, ls='--', label='Langzeitgrenze aus dem k-Raum')
    c.set_yscale('log')
    c.set_xlabel('Schritte n'); c.set_ylabel('C_l der gesamten Verteilung')
    title(c, '(c) 3D-Simulation, Punktquelle (alle inneren Zustände): die Richtungs-\nverteilung läuft gegen die Langzeitgrenze, eine feste kubische Form')
    c.legend(loc='lower right', fontsize=8)

    # (d) radial distribution vs the geometric light cone
    dd = ax[1, 1]
    edges = d['C_rad_edges']
    mid = 0.5 * (edges[1:] + edges[:-1])
    for n, col in ((16, S3), (32, S4), (64, S1)):
        h = d[f'C_rad_{n}']
        dd.plot(mid, h / (edges[1] - edges[0]), color=col, label=f'n = {n}')
    fr = d['D_front']
    dd.axvline(CP, color=MUTED, lw=1, ls='--')
    dd.text(CP + 0.02, dd.get_ylim()[1] * 0.9, "c′ (Dirac-Kegel)", color=MUTED, fontsize=8)
    dd.axvspan(fr[:, 0].min(), fr[:, 0].max(), color=S2, alpha=0.15, lw=0)
    dd.text(fr[:, 0].max() + 0.02, dd.get_ylim()[1] * 0.75, 'Front (99 %)\n[110] … [111]', color=S2, fontsize=8)
    dd.axvline(C_LIGHT, color=INK, lw=1.2)
    dd.text(C_LIGHT - 0.02, dd.get_ylim()[1] * 0.9, 'Schale n\n(Gitter-Lichtkegel c)', color=INK, fontsize=8, ha='right')
    dd.set_xlim(0, 1.9)
    dd.set_xlabel('r / t'); dd.set_ylabel('Wahrscheinlichkeitsdichte in r/t')
    title(dd, '(d) Wo die Wahrscheinlichkeit ist: Front bei 0.86–0.92, weit innerhalb\nder geometrischen Schale (c = 1.73); dort liegt praktisch nichts')
    dd.legend(loc='center right', fontsize=8)

    fig.suptitle('Experiment 3: Isotropie der Kuboktaeder-Schalen und der Lichtkegel-Front des Walkers (2O*, ε = 0.3)',
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_exp3_isotropy.png'), dpi=120)
    print('saved figures/phase_bit_exp3_isotropy.png')


if __name__ == '__main__':
    main()
