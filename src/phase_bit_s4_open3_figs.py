#!/usr/bin/env python3
"""Figure for the third S4 follow-up.  Reads data/phase_bit_s4_open3.npz."""

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
    d = np.load(data_path('phase_bit_s4_open3.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))

    # (a) R'
    a = ax[0, 0]
    R = d['A_R']
    names = ['2O plain', 'SU2 comp', '2O*']
    labels = ['2O ohne\nKompensation', 'SU(2)\nkompensiert', '2O*\n(2O + λ*εW₉₀)']
    for i, n in enumerate(names):
        row = [r for r in R if r[0] == n and np.isclose(r[1], 0.03)][0]
        a.bar(i - 0.18, row[2], width=0.34, color=S1, label='Teilchen' if i == 0 else None)
        a.bar(i + 0.18, row[3], width=0.34, color=S2, label='Antiteilchen' if i == 0 else None)
    a.axhline(1, color=MUTED, ls='--', lw=1)
    a.set_xticks(range(3)); a.set_xticklabels(labels)
    a.set_ylabel("R′ = E₀ / (m* c′²)")
    title(a, "(a) Äquivalenzprinzip, μ/ε = 0.1, Dirac-Punkt bei E = 0\n(R′ = 1: fällt wie die ART es verlangt, c′ = c/3)")
    a.legend(loc='upper left', fontsize=8.5)

    # (b) lapse accelerations
    b = ax[0, 1]
    L = d['A_lapse']
    g = float(d['A_g'])
    ref = -g * CP ** 2
    xs = np.arange(len(L))
    b.bar(xs - 0.18, [r[1] / ref for r in L], width=0.34, color=S1, label='Uhren-Lapse (Simulation)')
    b.bar(xs + 0.18, [r[3] / ref for r in L], width=0.34, color=S2, label='voller Lapse (Simulation)')
    b.plot(xs - 0.18, [r[2] / ref for r in L], 'k_', ms=14, mew=2, label='Halbklassik')
    b.plot(xs + 0.18, [r[4] / ref for r in L], 'k_', ms=14, mew=2)
    b.axhline(1, color=MUTED, ls='--', lw=1); b.axhline(0, color=MUTED, lw=0.8)
    b.set_xticks(xs); b.set_xticklabels([r[0].replace(', ', '\n') for r in L], fontsize=8.5)
    b.set_ylabel("a_x / (−g c′²)")
    title(b, '(b) Querbeschleunigung im Lapse-Feld\nUhren-Lapse: Licht fällt nicht (Nordström); voll: alles fällt (Einstein 1911)')
    b.legend(loc='upper right', fontsize=8)

    # (c) kappa
    c = ax[0, 2]
    K = d['A_kappa']
    for mu, col in ((0.03, S1), (0.15, S2)):
        sel = np.isclose(K[:, 0], mu)
        c.plot(K[sel, 1], K[sel, 2], 'o-', color=col, label=f'κ(k), μ = {mu}')
        c.plot(K[sel, 1], K[sel, 3], '--', color=col, lw=1.2, label=f'(m/E)², μ = {mu}')
    c.axhline(0, color=MUTED, lw=0.8)
    c.set_xlabel('k'); c.set_ylabel('Kopplung an den Uhren-Lapse')
    title(c, '(c) Uhren-Lapse koppelt an (m/E)² = 1/γ²\n(für k ≳ ε zusätzlich schwächer)')
    c.legend(loc='upper right', fontsize=8)

    # (d) wall dispersion with a T-even perturbation
    kx = d['B_kx']
    dd = ax[1, 0]
    for tag, col, mk, lab in (('single', S1, 'o', 'ein Quartett (2O*)'), ('double', S2, 's', 'zwei Quartette (λ = −2)')):
        W = d[f'B_{tag}_pert']
        first = True
        for i, k in enumerate(kx):
            ys = W[i][~np.isnan(W[i])]
            if len(ys):
                dd.plot([k] * len(ys), ys, mk, color=col, ms=6, mfc=col if tag == 'single' else 'white', mew=1.3,
                        label=lab if first else None)
                first = False
    dd.axhspan(-0.1, 0.1, color=GRID, alpha=0.4, lw=0)
    dd.set_ylim(-0.1, 0.1)
    dd.set_xlabel('k_∥'); dd.set_ylabel('E − E_c  (Zustände an Wand 1)')
    title(dd, '(d) Domänenwand mit zufälliger T-gerader Störung (1.2μ):\nZweige laufen vom unteren ins obere Volumenband')
    dd.legend(loc='lower left', fontsize=8)

    # (e) central spacing: protected crossing vs avoided crossing
    e = ax[1, 1]
    kf = d['B2_k']
    for tag, col, lab in (('single', S1, 'ein Quartett'), ('double', S2, 'zwei Quartette')):
        for dl, ls in (('1.2', '-'), ('2.4', '--')):
            e.semilogy(kf, np.maximum(d[f'B2_{tag}_{dl}_cen'], 1e-16), marker='o', ms=4, color=col, ls=ls,
                       label=f'{lab}, Störung {dl}μ')
    e.set_xlabel('k_∥'); e.set_ylabel('Abstand der beiden mittleren Wandzweige')
    e.set_ylim(1e-16, 1e-1)
    title(e, '(e) Z₂ praktisch: ein Kramers-Paar → Kreuzung geschützt;\nzwei Paare → vermiedene Kreuzung, die Wand bekommt eine Lücke')
    e.legend(loc='center right', fontsize=8)

    # (f) Landau levels
    f = ax[1, 2]
    kz = d['C_kz']
    for key in d.files:
        if not key.startswith('C_ll_'):
            continue
        _, _, s, En = key.split('_')
        s, En = int(s), float(En)
        if En > 0:
            continue                  # interior node: LLL branch not separable from higher Landau levels
        col = S1 if s > 0 else S2
        ls = '-'
        f.plot(kz, d[key], marker='o', color=col, ls=ls, ms=4,
               label=f'Helizität {s:+d}, Knoten E = {En:+.2f}')
    f.plot(kz, CP * kz, color=MUTED, ls='--', lw=1); f.plot(kz, -CP * kz, color=MUTED, ls='--', lw=1)
    f.set_xlabel('k_z  (entlang B)'); f.set_ylabel('E − E_Knoten (unterstes Landau-Niveau)')
    title(f, '(f) Magnetfeld, unterster Weyl-Knoten: ein chiraler Zweig\nmit v = ∓c/3 je nach Helizität ([H, W_B] = 3·10⁻¹⁶)')
    f.legend(loc='upper center', fontsize=7.5)

    fig.suptitle('S4, dritte Runde: Gravitation auf dem sauberen Teilchen, Z₂ der Domänenwand, '
                 'Magnetfeld und Anomalie', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open3.png'), dpi=120)
    print('saved figures/phase_bit_s4_open3.png')


if __name__ == '__main__':
    main()
