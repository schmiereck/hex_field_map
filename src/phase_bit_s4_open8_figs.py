#!/usr/bin/env python3
"""Figure for hypothesis H2 (one budget rule: wait vs stay).  Reads data/phase_bit_s4_open8.npz."""

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


def title(ax, s):
    ax.set_title(s, loc='left', color=INK, fontsize=10)


def main():
    d = np.load(data_path('phase_bit_s4_open8.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) wait vs stay: scaling of clock (gap) and hop (velocity)
    a = ax[0, 0]
    NN = np.linspace(0.25, 1, 100)
    a.plot(NN, NN / (2 - NN), color=MUTED, lw=1.5, label='zwei Warte-Puffer (Hüpfen und Uhr): beide 1/(2/N − 1)')
    A1 = d['A1']
    ww = (A1[:, 0] == 1) & (np.abs(A1[:, 1] - A1[:, 2]) < 1e-9)
    a.plot(A1[ww, 1], A1[ww, 3], 'o', color=MUTED, ms=8, mfc='none')
    a.plot(A1[ww, 1], A1[ww, 4], 'x', color=MUTED, ms=8)
    A2 = d['A2']
    a.plot(NN, NN, color=S1, lw=1.2, ls='--')
    a.plot(NN, NN ** 2, color=S2, lw=1.2, ls='--')
    a.plot(A2[:, 0], A2[:, 1], 'o', color=S1, ms=8, label='H2, Uhr: Lücke / m = N')
    a.plot(A2[:, 0], A2[:, 2], 's', color=S2, ms=8, label="H2, Hüpfen: v / c′ = N²")
    a.set_xlabel('N (freier Anteil eines Knotens)'); a.set_ylabel('Skalierung')
    gam = A2[:, 4]
    title(a, f'(a) Warten skaliert alles gleich (γ = 0); Bleiben trennt Uhr und Hüpfen:\n'
             f'A = 1/N, γ = {gam.min():.4f}…{gam.max():.4f} (Bandebene, N = 0.3…0.95)')
    a.legend(loc='upper left', fontsize=8)

    # (b) price: extra background of the stay gate
    b = ax[0, 1]
    A3 = d['A3']
    for mu, col, mk in ((0.0, S1, 'o'), (0.03, S3, 's'), (0.15, S2, 'D')):
        s = A3[:, 0] == mu
        b.plot(A3[s, 1], A3[s, 3], mk, color=col, ms=8, label=f'μ = {mu}')
    b.plot(NN, 0.245 * NN * (1 - NN), color=MUTED, lw=1.2, ls='--', label='0.245 · N_h (1 − N_h)')
    b.set_xlabel('N_h (Bleibe-Gatter)'); b.set_ylabel('zusätzlicher Untergrund α  (E ⊃ −α k²)')
    title(b, '(b) Preis des Bleibe-Gatters: ein Untergrund −α k² der Ordnung Δt,\nrelativ zur Kinetik ∝ m·Δt bzw. k·Δt (verschwindet im Kontinuum)')
    b.legend(loc='upper right', fontsize=8)

    # (c) x-chain: ratio (a_H2 / N0^2) / a_1b
    c = ax[1, 0]
    r = d['B_rows']
    names = [x.replace('massless', 'Licht') for x in r[:, 0]]
    y = np.arange(len(r))
    c.plot(r[:, 7].astype(float), y, 'o', color=S1, ms=9, label='Simulation')
    c.plot(r[:, 8].astype(float), y, '|', color=INK, ms=14, mew=2, label='Halbklassik mit exaktem Gitterband')
    c.plot(r[:, 9].astype(float), y, 'D', color=S3, ms=7, mfc='none', mew=1.5, label='ART, γ = 1 (ideale Dirac-Dispersion)')
    c.plot(r[:, 13].astype(float), y, 's', color=S2, ms=7, mfc='none', mew=1.5, label='ART, γ = 1 plus gemessener Untergrund α')
    c.set_yticks(y); c.set_yticklabels(names, fontsize=9); c.invert_yaxis()
    c.set_xlabel('Ablenkung H2 / Ablenkung nur Warten  (normiert mit N₀²)')
    title(c, '(c) x-Kette: Licht wird doppelt so stark abgelenkt wie mit Warten allein;\nRest erklärt γ = 1 plus Untergrund α')
    c.legend(loc='lower left', fontsize=8)

    # (d) light trajectories
    dd = ax[1, 1]
    ts = d['B_ts']
    N0 = float(d['B_N0'])
    for case, col in (('massless, k_y=0.2', S1), ('mu=0.03, k_y=0.2', S3)):
        x1 = d[f'B_traj_{case}|1b']
        x2 = d[f'B_traj_{case}|H2']
        lab = case.replace('massless', 'Licht')
        dd.plot(ts, x1 - x1[0], color=col, lw=1.4, ls='--', label=f'{lab}: nur Warten (1b)')
        dd.plot(ts, (x2 - x2[0]) / N0 ** 2, color=col, label=f'{lab}: H2 (Warten + Bleiben) / N₀²')
    dd.set_xlabel('Zeit t'); dd.set_ylabel('Verschiebung zu kleinerem N')
    title(dd, '(d) Querlaufende Pakete im Gradienten von N: H2 lenkt Licht doppelt ab\n(volle ART-Ablenkung), massiv um den Faktor ≈ 1 + v²')
    dd.legend(loc='lower left', fontsize=8)

    fig.suptitle('H2: eine Budget-Regel pro Knoten (jeder Prozess braucht jeden beteiligten Knoten frei) ergibt Lapse und räumliche Metrik',
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open8.png'), dpi=120)
    print('saved figures/phase_bit_s4_open8.png')


if __name__ == '__main__':
    main()
