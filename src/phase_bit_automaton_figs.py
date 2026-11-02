#!/usr/bin/env python3
"""Figure for Experiment S1 (phase-bit automaton).  Reads data/phase_bit_automaton.npz."""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import data_path, fig_path
from quantum_hex_turning import MOVES_PHYS, DX_PHYS, DY_PHYS

S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3de'
PB_LABEL = ['+1', '+i', '−1', '−i']

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0, 'legend.frameon': False,
})


def hexagon_loop_panel(ax):
    """Six left turns: the walker returns after 360 deg with phase bit -1."""
    pts = [np.zeros(2)]
    for N in range(12):
        pts.append(pts[-1] + MOVES_PHYS[N % 6])
    pts = np.array(pts)
    center = pts[:6].mean(axis=0)
    for lap, (scale, col) in enumerate(((1.0, S1), (1.55, S2))):
        hx = center + scale * (pts[:7] - center)
        ax.plot(hx[:, 0], hx[:, 1], color=col, lw=1.5, alpha=0.5, zorder=1)
        for s in range(6):
            N = 6 * lap + s
            mid = center + scale * (0.5 * (pts[s] + pts[s + 1]) - center)
            ax.plot(*mid, 'o', ms=22, color='white', mec=col, mew=2, zorder=2)
            ax.text(*mid, PB_LABEL[N % 4], ha='center', va='center',
                    color=INK, fontsize=9, zorder=3)
    ax.text(*(center + 1.95 * (0.5 * (pts[0] + pts[1]) - center)),
            'Kante 7 = Kante 1,\naber Bit −1', ha='center', va='top',
            color=INK, fontsize=8)
    ax.set_xlim(center[0] - 1.9, center[0] + 1.9)
    ax.set_ylim(center[1] - 2.2, center[1] + 1.7)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('(a) Phasen-Bit auf 12 Schritten Linkskurve\n'
                 'innen 1. Umlauf (blau), außen 2. Umlauf (orange)',
                 loc='left', color=INK)


def vortex_panel(ax, j, title, col):
    """Mode e^{2 pi i j N/12} on the double cover, drawn as a two-lap spiral."""
    for N in range(12):
        th = 2 * np.pi * (N % 6) / 6
        r = 1.0 + 0.45 * (N // 6)
        pos = r * np.array([np.cos(th), np.sin(th)])
        ph = 2 * np.pi * j * N / 12
        ax.annotate('', xy=pos + 0.28 * np.array([np.cos(ph), np.sin(ph)]),
                    xytext=pos, arrowprops=dict(arrowstyle='-|>', color=col, lw=2))
        ax.plot(*pos, 'o', color=col if N < 6 else MUTED, ms=4)
    s = np.linspace(0, 4 * np.pi, 200)
    rr = 1.0 + 0.45 * s / (2 * np.pi)
    ax.plot(rr * np.cos(s), rr * np.sin(s), color=GRID, lw=1, zorder=0)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, loc='left', color=INK)


def main():
    d = np.load(data_path('phase_bit_automaton.npz'))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))

    hexagon_loop_panel(ax[0, 0])
    vortex_panel(ax[0, 1], 1, '(b) Spin-Wirbel j = 1 auf der Doppelüberlagerung\n'
                 'Pfeil dreht 30° pro Schritt: nach 1 Umlauf −1', S1)
    vortex_panel(ax[0, 2], 2, '(c) ganzzahliger Wirbel j = 2 (Vergleich)\n'
                 'Pfeil dreht 60° pro Schritt: nach 1 Umlauf +1', S3)

    # (d), (e): the same integer table, two readouts
    T = int(d['T_MAX'])
    for k, (a, name) in enumerate(((1, 'Spin-Auslesung (a = 1)'),
                                   (0, 'Boson-Auslesung (a = 0)'))):
        axx = ax[1, k]
        rho = d[f'rho_a{a}']
        nx, ny = rho.shape
        IX, IY = np.meshgrid(np.arange(nx) - nx // 2, np.arange(ny) - ny // 2,
                             indexing='ij')
        m = rho > 1e-30 * rho.max()
        rr = rho / rho.max()
        sc = axx.scatter(IX[m] * DX_PHYS, IY[m] * DY_PHYS, c=rr[m], s=7,
                         cmap='Blues', norm=LogNorm(vmin=1e-6, vmax=1))
        axx.set_aspect('equal')
        axx.set_xlabel('x'); axx.set_ylabel('y')
        axx.set_title(f'({"de"[k]}) {name}, t = {T}, ε = 0.3, b = 1\n'
                      'dieselbe Ganzzahl-Tabelle N[x,y,d,p,T]', loc='left', color=INK)
        fig.colorbar(sc, ax=axx, shrink=0.8, label='|A|² (relativ)')

    # (f) closed walks: sign of the phase bit vs winding enumeration
    axf = ax[1, 2]
    L = np.arange(1, len(d['enum_odd']) + 1)
    w = 0.38
    axf.bar(L - w / 2, d['loops_p2'][:len(L)], w, color=S2,
            label='Automat: Bit = −1 (p = 2)')
    axf.bar(L + w / 2, d['loops_p0'][:len(L)], w, color=S1,
            label='Automat: Bit = +1 (p = 0)')
    axf.plot(L - w / 2, d['enum_odd'], 'k_', ms=14, mew=2,
             label='Aufzählung: ungerade / gerade Windung')
    axf.plot(L + w / 2, d['enum_even'], 'k_', ms=14, mew=2)
    axf.set_yscale('log'); axf.set_ylim(1, 1e4)
    axf.set_xlabel('Schleifenlänge L'); axf.set_ylabel('Anzahl geschlossener Pfade')
    axf.set_title('(f) Bit = −1 genau bei ungerader Windung\n(exakte Übereinstimmung)',
                  loc='left', color=INK)
    axf.legend(loc='upper left', fontsize=8)
    axf.grid(True, axis='y', color=GRID, lw=0.6)

    fig.suptitle('Experiment S1 — Spin-Wirbel aus Phasen-Bits: ganzzahliger Automat '
                 'auf dem Dreiecksgitter', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_automaton.png'), dpi=130)
    print('saved figures/phase_bit_automaton.png')


if __name__ == '__main__':
    main()
