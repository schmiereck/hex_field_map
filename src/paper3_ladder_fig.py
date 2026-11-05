#!/usr/bin/env python3
"""Overview figure for Paper 3: the ladder clock lapse -> wait buffer -> budget rule (-> product rule, drag hop).

All numbers are measured values quoted from the RESULTS documents (source given next to each):
  clock lapse vs global full lapse on the same x-chain ..... RESULTS_PhaseBit_S4_Open3_de.md, section 1
  local wait buffer vs global full lapse; Exp. 1c .......... RESULTS_PhaseBit_S4_Open4_de.md, sections 1-2
  budget rule H2 on the chain (raw / compensated) .......... RESULTS_H2_BudgetRule_de.md 4, RESULTS_H2b_Compensation_Beta_de.md 2
  3D walker, point mass, H2/W at equal time ................ RESULTS_Gravity3D_de.md B
  composition laws (beta), gamma of the budget rule ........ RESULTS_H2b_Compensation_Beta_de.md 3, RESULTS_H2_BudgetRule_de.md 2
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import fig_path

S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3de'
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0, 'legend.frameon': False,
})
C_CLOCK, C_WAIT, C_H2, C_EXT = S2, S4, S1, S3


def title(ax, s):
    ax.set_title(s, loc='left', color=INK, fontsize=10)


def node(ax, x, y, filled, col):
    ax.scatter([x], [y], s=230, facecolor=col if filled else 'white', edgecolor=col, linewidth=1.8, zorder=3)


def rung(ax, y, col, name, theory, rule, ham, clock_fill, hop_fill):
    """one rung: box with name, rule, Hamiltonian; right: mini diagram (filled node = process waits for it)."""
    h = 0.19
    ax.add_patch(FancyBboxPatch((0.01, y - h / 2), 0.98, h, boxstyle='round,pad=0.006,rounding_size=0.02',
                                facecolor='white', edgecolor=col, lw=1.8))
    ax.text(0.03, y + 0.058, name, color=INK, fontsize=10.5, fontweight='bold', va='center')
    ax.text(0.97, y + 0.058, theory, color=col, fontsize=9.5, fontweight='bold', ha='right', va='center')
    ax.text(0.03, y + 0.002, rule, color=INK, fontsize=8.4, va='center')
    ax.text(0.03, y - 0.052, ham, color=INK, fontsize=9.5, va='center')
    yd = y - 0.035
    cx, h1, h2 = 0.70, 0.80, 0.92
    node(ax, cx, yd, clock_fill, col)
    node(ax, h1, yd, hop_fill[0], col)
    node(ax, h2, yd, hop_fill[1], col)
    ax.annotate('', xy=(h2 - 0.03, yd), xytext=(h1 + 0.03, yd), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.4))
    ax.text(cx, yd - 0.042, 'clock', ha='center', va='center', fontsize=7.5, color=MUTED)
    ax.text((h1 + h2) / 2, yd - 0.042, 'hop', ha='center', va='center', fontsize=7.5, color=MUTED)


def main():
    fig = plt.figure(figsize=(16, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.05, 1], wspace=0.25)

    # (a) the ladder
    a = fig.add_subplot(gs[0])
    a.set_xlim(0, 1); a.set_ylim(0, 1.0); a.axis('off')
    ys = [0.11, 0.355, 0.60, 0.845]
    rung(a, ys[0], C_CLOCK, '1  Clock lapse', 'Nordström',
         'only the coin is slowed: C = exp(i N X)', 'H = K + N M   (light not bent)', True, (False, False))
    rung(a, ys[1], C_WAIT, '2  Wait buffer', 'Einstein 1911',
         'the node idles for the fraction 1 − N', 'H = N (K + M),   γ = 0', True, (True, False))
    rung(a, ys[2], C_H2, '3  Budget rule', 'GR: γ = 1',
         'every process needs each node it involves free', 'H = N² K + N M,   A = 1/N', True, (True, True))
    rung(a, ys[3], C_EXT, '4  + product rule, drag hop', 'GR: β = 1, g₀ᵢ',
         'loads nest: N = Π Nᵢ;  momentum current drags', 'β = 1,   H ⊃ b·p  (C = 4)', True, (True, True))
    for y0, y1 in zip(ys[:-1], ys[1:]):
        a.annotate('', xy=(0.45, y1 - 0.098), xytext=(0.45, y0 + 0.098),
                   arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.4))
    a.text(0.01, 0.985, 'filled node: the process waits for that node to be free (each one a factor N)',
           fontsize=8, color=MUTED, va='top')
    title(a, '(a) Gravity as computation time: what a loaded node slows down')

    # (b) light deflection relative to the wait buffer (Einstein 1911); GR = 2
    b = fig.add_subplot(gs[1])
    groups = [
        ('clock\nlapse', [+0.10 / -4.85, +0.22 / -5.86], C_CLOCK),           # clock / global full lapse
        ('wait\nbuffer', [0.72 / 0.73, 0.88 / 0.88], C_WAIT),                 # local buffer / global full lapse
        ('buffer\n+ A (1c)', [1.999, 1.989], MUTED),                        # Exp. 1c: lapse N/A, mass mu A
        ('budget\nrule', [1.986, 1.922, 1.860], C_H2),                        # H2 raw (k = 0.1, 0.2, 0.4)
        ('budget\n+ λ(N)', [2.005, 1.998, 2.007], C_H2),                      # H2b compensated
        ('budget\n3D walker', [2.071, 2.058], C_EXT),                        # 3D point mass, equal time
    ]
    for i, (lab, vals, col) in enumerate(groups):
        xs = i + np.linspace(-0.16, 0.16, len(vals))
        b.plot(xs, vals, 'o', color=col, ms=9, mec='white', mew=1.5, zorder=3)
    for yv, lab in ((0, 'Nordström'), (1, 'Einstein 1911'), (2, 'GR')):
        b.axhline(yv, color=MUTED, lw=0.9, ls='--', zorder=1)
        b.text(-0.45, yv + 0.04, lab, ha='left', va='bottom', fontsize=8, color=MUTED)
    b.set_xticks(range(len(groups))); b.set_xticklabels([g[0] for g in groups], fontsize=8)
    b.set_xlim(-0.5, len(groups) - 0.5); b.set_ylim(-0.25, 2.4)
    b.grid(axis='x', visible=False)
    b.set_ylabel('light deflection / deflection with the wait buffer')
    b.text(5, 2.2, 'equal time;\nper path 1.99', ha='center', fontsize=7.5, color=MUTED)
    b.text(3, 1.72, 'k = 0.1, 0.2, 0.4\n(background −αk²)', ha='center', fontsize=7.5, color=MUTED)
    title(b, '(b) Measured light deflection (massless packets):\nthe budget rule adds the missing half')

    # (c) gamma-beta plane with perihelion contours
    c = fig.add_subplot(gs[2])
    g = np.linspace(-1.3, 1.5, 200)
    for q, lab in ((1 / 3, '1/3'), (2 / 3, '2/3'), (1.0, '')):
        c.plot(g, 2 + 2 * g - 3 * q, color=MUTED if q == 1 else '#c9c8c2', lw=1.4 if q == 1 else 1.0, zorder=1)
        yl = 1.9
        gl = (yl - 2 + 3 * q) / 2
        if -1.25 < gl < 1.45:
            c.text(gl - 0.03, yl, lab, fontsize=7.5, color=MUTED, ha='right', va='center', rotation=0)
    c.axvline(-1, color=C_CLOCK, lw=1.6, ls='--')
    c.text(-0.96, 0.25, 'clock lapse:\nno light bending,\nγ = −1', color=C_CLOCK, fontsize=8, va='center')
    c.plot(0, 1, 's', color=C_WAIT, ms=11, mec='white', mew=1.2, zorder=4)
    c.plot(1, 0.495, 'v', color=MUTED, ms=11, mec='white', mew=1.2, zorder=4)
    c.plot(1, 1.495, '^', color=MUTED, ms=11, mec='white', mew=1.2, zorder=4)
    c.plot(1, 1.0001, '*', color=C_EXT, ms=19, mec='white', mew=1.2, zorder=5)
    kw = dict(fontsize=8, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    c.annotate('wait buffer (product rule)\nperihelion 1/3 of GR', (0, 1), xytext=(-0.62, 0.62), **kw)
    c.annotate('linear lapse: β = 1/2\nperihelion 7/6', (1, 0.495), xytext=(0.62, 0.03), **kw)
    c.annotate('queue (waits add): β = 3/2', (1, 1.495), xytext=(-0.35, 1.68), **kw)
    c.annotate('budget rule + nested gates\nγ = 1.0000, β = 1.0000', (1, 1.0), xytext=(0.08, 1.12), **kw)
    c.text(1.2, 1.3, 'perihelion\n= GR', fontsize=7.5, color=MUTED, va='center')
    c.set_xlim(-1.3, 1.5); c.set_ylim(-0.1, 2.0)
    c.set_xlabel('γ_PPN  (spatial curvature per unit mass)')
    c.set_ylabel('β_PPN  (nonlinearity of superposition)')
    title(c, '(c) PPN parameters of the rules; grey lines:\nperihelion advance (2 + 2γ − β)/3 relative to GR')

    fig.suptitle('Paper 3 overview: Nordström → Einstein 1911 → GR as successive rules for a busy node',
                 color=INK, fontsize=12, y=1.0)
    fig.savefig(fig_path('paper3_ladder.png'), dpi=130, bbox_inches='tight')
    print('saved figures/paper3_ladder.png')


if __name__ == '__main__':
    main()
