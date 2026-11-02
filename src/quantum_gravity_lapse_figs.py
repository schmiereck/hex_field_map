#!/usr/bin/env python3
"""Figure for Experiment 1a (internal-clock lapse).  Reads data/gravity_lapse.npz."""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import data_path, fig_path

C2 = 3.0
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'   # validated order
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3de'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0, 'legend.frameon': False,
})


def main():
    d = np.load(data_path('gravity_lapse.npz'))
    g = float(d['g'])
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a) equivalence ratio
    a = ax[0, 0]
    e = d['eps_list']
    a.plot(e, d['R_alpha0.0'], 'o-', color=S1, ms=6, label='α = 0')
    a.plot(e, d['R_alpha0.25'], 's-', color=S2, ms=6, label='α = 1/4')
    a.axhline(1.0, color=MUTED, ls='--', lw=1.2)
    a.text(0.36, 1.06, 'GR / Dirac: R = 1', color=MUTED)
    a.set_xlabel('ε₀'); a.set_ylabel('R = (m₀/m*) / c²')
    a.set_ylim(0.8, 2.8)
    a.set_title('(a) Universalität des freien Falls', loc='left', color=INK)
    a.legend(loc='center right')

    # (b) free fall, simulation vs semiclassics
    b = ax[0, 1]
    for eps, col in zip(d['fall_eps'], (S1, S2, S3, S4)):
        t, x = d[f'fall_t_eps{eps}'], d[f'fall_x_eps{eps}']
        b.plot(d[f'sc_t_eps{eps}'], d[f'sc_x_eps{eps}'], color=col, lw=2,
               label=f'ε₀ = {eps}')
        b.plot(t[::4], x[::4], 'o', color=col, ms=5, mfc='white', mew=1.5)
    tt = d['sc_t_eps0.3']
    b.plot(tt, -0.5 * C2 * g * tt ** 2, color=MUTED, ls='--', lw=1.2,
           label='GR: −c²g t²/2')
    b.set_xlabel('t'); b.set_ylabel('⟨x⟩')
    b.set_title('(b) Fall im Lapse N = 1 + g·x\n(Linie: Halbklassik, ○: Gitter)',
                loc='left', color=INK)
    b.legend(loc='lower left')

    # (c) coupling of moving particles
    c = ax[1, 0]
    k = d['k_kappa']
    for eps, col in ((0.1, S1), (0.3, S2)):
        c.plot(k, d[f'kappa_eps{eps}'], color=col, label=f'Gitter ε₀ = {eps}')
        c.plot(k, d[f'kdirac_eps{eps}'], color=col, ls=':', lw=1.5,
               label=f'(m₀/E)² ε₀ = {eps}')
    c.axhline(1.0, color=MUTED, ls='--', lw=1.2)
    c.text(1.9, 0.93, 'voller Lapse: κ = 1', color=MUTED)
    c.set_xlabel('|k|'); c.set_ylabel('κ = ε ∂E/∂ε / E')
    c.set_ylim(0, 1.05)
    c.set_title('(c) Kopplung bewegter Teilchen an den Uhren-Lapse', loc='left',
                color=INK)
    c.legend(loc='upper right', bbox_to_anchor=(1.0, 0.88))

    # (d) transverse acceleration
    dd = ax[1, 1]
    kk = d['defl_k']
    dd.plot(kk, -d['ax_full'], 'o--', color=MUTED, ms=6, lw=1.2,
            label='voller Lapse (koppelt an E)')
    dd.plot(kk, -d['ax_sc'], '-', color=S1, label='Uhren-Lapse, Halbklassik')
    dd.plot(kk, -d['ax_sim'], 'o', color=S1, ms=7, mfc='white', mew=1.5,
            label='Uhren-Lapse, Gitter')
    dd.set_xlabel('|k| (Bewegung quer zum Gradienten)')
    dd.set_ylabel('|a_x|')
    dd.set_ylim(0, None)
    dd.set_title('(d) Querbeschleunigung, ε₀ = 0.3', loc='left', color=INK)
    dd.legend(loc='upper right')

    fig.suptitle('Experiment 1a — Gravitation als Rechenzeit: Lapse auf der inneren Uhr '
                 '(Turning-Phase-Modell)', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('gravity_lapse.png'), dpi=140)
    print('saved figures/gravity_lapse.png')


if __name__ == '__main__':
    main()
