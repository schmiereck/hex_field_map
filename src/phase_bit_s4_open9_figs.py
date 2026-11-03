#!/usr/bin/env python3
"""Figure for H2 continued: compensated stay gate and beta_PPN.  Reads data/phase_bit_s4_open9.npz."""

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
    d = np.load(data_path('phase_bit_s4_open9.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) background before / after compensation
    a = ax[0, 0]
    A1, own = d['A1'], d['A_own']
    N = A1[:, 0]
    a.plot(N, A1[:, 5], 'o--', color=S3, ms=7, lw=1.2, label='μ = 0.03, ohne Kompensation')
    a.plot(N, A1[:, 7], 'o-', color=S3, ms=7, label='μ = 0.03, mit λ(N_h)')
    a.plot(N, A1[:, 6], 'D--', color=S2, ms=6, lw=1.2, label='μ = 0.15, ohne Kompensation')
    a.plot(N, A1[:, 8], 'D-', color=S2, ms=6, label='μ = 0.15, mit λ(N_h)')
    a.axhline(own[1], color=S3, lw=0.8, ls=':')
    a.axhline(own[2], color=S2, lw=0.8, ls=':')
    a.set_xlabel('N_h'); a.set_ylabel('Untergrund α  (E ⊃ −α k²)')
    title(a, '(a) Kompensation: der Knoten stimmt λ (W₉₀) mit N_h nach;\nder Untergrund kehrt auf den Vakuumwert (punktiert) zurück')
    a.legend(loc='upper right', fontsize=8)

    # (b) x-chain with compensation
    b = ax[0, 1]
    r = d['B_rows']
    names = [x.replace('massless', 'Licht') for x in r[:, 0]]
    y = np.arange(len(r))
    b.plot(r[:, 5].astype(float), y, 'o', color=S1, ms=9, label='Simulation (H2 kompensiert)')
    b.plot(r[:, 6].astype(float), y, '|', color=INK, ms=14, mew=2, label='Halbklassik mit exaktem Gitterband')
    b.plot(r[:, 7].astype(float), y, 'D', color=S3, ms=7, mfc='none', mew=1.5, label='ART, γ = 1')
    b.set_yticks(y); b.set_yticklabels(names, fontsize=9); b.invert_yaxis()
    b.set_xlabel('Ablenkung H2 / Ablenkung nur Warten  (normiert mit N₀²)')
    title(b, '(b) Mit Kompensation: Licht γ = 1.005 / 0.998 / 1.007; μ = 0.15 behält\nden eigenen (massenabhängigen) Untergrund des Walkers')
    b.legend(loc='lower left', fontsize=8)

    # (c) beta from the composition of two loads
    c = ax[1, 0]
    C1 = d['C1']
    U = -np.log(C1[:, 3]) * -np.log(C1[:, 4])
    for law, col, mk, lab in ((0, S2, 's', 'ein Gatter, Φ linear überlagert'),
                              (1, S4, '^', 'parallele Warte-Register (Wartezeiten addieren sich)'),
                              (2, S1, 'o', 'verschachtelte Gatter (frei für beide Lasten: Produkt)')):
        s = C1[:, 2] == law
        c.plot(U[s], C1[s, 6], mk + '-', color=col, ms=8, lw=1.2, label=lab)
    c.axhline(1, color=MUTED, lw=0.8, ls='--')
    c.text(U.max(), 1.03, 'ART: β = 1', ha='right', color=MUTED, fontsize=9)
    c.set_xscale('log')
    c.set_xlabel('U₁·U₂ = ln N₁ · ln N₂  (Feldstärke)'); c.set_ylabel('β_PPN aus ln N₁₂ − ln N₁ − ln N₂ = 2(β − 1) U₁U₂')
    title(c, '(c) β_PPN: wie sich zwei Lasten an einem Knoten verbinden;\nnur „frei für beide“ (Produkt, wie die H2-Regel) gibt β = 1, auch im starken Feld')
    c.legend(loc='center left', fontsize=8)

    # (d) compensating coupling
    dd = ax[1, 1]
    fitc = d['A_fit']
    lam0 = A1[0, 1] - np.polyval(fitc, 1 - A1[0, 0])
    NN = np.linspace(0.4, 1, 100)
    dd.plot(1 - N, A1[:, 1] - lam0, 'o', color=S1, ms=8, label='Nullstelle von α_extra (μ = 0)')
    dd.plot(1 - NN, np.polyval(fitc, 1 - NN), color=S1, lw=1.2, label='Polynom 4. Grades (für die Kette)')
    dd.axhline(-2 - lam0, color=S2, lw=1, ls='--', label='λ = −2: die beiden Quartette treffen sich')
    dd.set_xlabel('1 − N_h'); dd.set_ylabel('λ(N_h) − λ*')
    title(dd, '(d) Nachstimmung der W₉₀-Kopplung: schwaches Feld Δλ ≈ 3.3 (1 − N_h);\nmöglich bis N_h ≈ 0.4')
    dd.legend(loc='upper left', fontsize=8)

    fig.suptitle('H2 fortgesetzt: Untergrund kompensiert (γ = 1 auch für Licht bei endlichem k), β_PPN = 1 aus der Produktregel',
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open9.png'), dpi=120)
    print('saved figures/phase_bit_s4_open9.png')


if __name__ == '__main__':
    main()
