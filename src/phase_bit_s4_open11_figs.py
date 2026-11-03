#!/usr/bin/env python3
"""Figure for causal g0i.  Reads data/phase_bit_s4_open11.npz."""

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
    d = np.load(data_path('phase_bit_s4_open11.npz'), allow_pickle=True)
    ts = d['ts']
    N0, Nx, SIG, D, gphi, nst, cg = d['par']
    tarr = D * 0.6123724356957945 / cg
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    names = {'light, parallel': 'Licht, parallel', 'light, antiparallel': 'Licht, antiparallel',
             'mu=0.03, parallel': 'μ = 0.03 (v = 0.75), parallel', 'mu=0.03, antiparallel': 'μ = 0.03 (v = 0.75), antiparallel'}
    for a, c in zip(ax.ravel(), names):
        kap = d[f'{c}|summary'][3]
        sc = 1e3 / kap
        for var, col, ls, lab in (('instant', MUTED, '-', 'beide Felder instantan'),
                                  ('nob', S3, '-', 'Φ kausal, ohne b'),
                                  ('mixed', S2, '-', 'Φ kausal, b instantan'),
                                  ('causal', S1, '-', 'Φ und b kausal (c_g)')):
            k = d[f'{c}|{var}|k']
            a.plot(ts, (k - k[0]) * sc, color=col, ls=ls, lw=1.4 if var == 'instant' else 2.0, label=lab)
            if var != 'instant':
                a.plot(ts, d[f'{c}|pred_{var}'] * sc, color=INK, lw=1.0, ls=':')
        a.plot([], [], color=INK, lw=1.0, ls=':', label='Vorhersage: statische Kraft × Lichtkegel-Anteil')
        a.axvline(tarr, color=MUTED, lw=0.8, ls='--')
        a.text(tarr + 4, a.get_ylim()[0] * 0.92 if 'anti' in c else a.get_ylim()[1] * 0.92, 'D / c_g', color=MUTED, fontsize=8)
        a.set_xlabel('Zeit t'); a.set_ylabel('⟨k_x⟩(t) − ⟨k_x⟩(0)  (10⁻³ κ)')
        if c == 'light, parallel':
            a.set_ylim(-2500, 3200)
        title(a, f'({"abcd"[list(names).index(c)]}) {names[c]}')
        a.legend(loc='upper left' if 'anti' not in c else 'lower left', fontsize=8)
    fig.suptitle('Kausales g₀ᵢ: Lapse Φ und Shift b als lokale Wellengleichungen mit c_g = c′N₀²\n'
                 'parallele Lichtstrahlen spüren nur dann zu keiner Zeit etwas, wenn beide Felder gemeinsam laufen',
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fig_path('phase_bit_s4_open11.png'), dpi=120)
    print('saved figures/phase_bit_s4_open11.png')


if __name__ == '__main__':
    main()
