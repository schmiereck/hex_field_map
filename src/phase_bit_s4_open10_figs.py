#!/usr/bin/env python3
"""Figure for g0i (gravitomagnetism from the momentum current).  Reads data/phase_bit_s4_open10.npz."""

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
    d = np.load(data_path('phase_bit_s4_open10.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) band level: drag hop exact, heading-dependent stay gate not
    a = ax[0, 0]
    A2 = d['A2']
    ELL = 0.6123724356957945
    kk = np.linspace(-0.25, 0.25, 50)
    a.plot(kk, 0.02 * np.sin(kk * ELL) / ELL * 1e3, color=MUTED, lw=1.2, label='b·sin(k_y l)/l  (b = 0.02)')
    pred = A2[:, 4]
    s0 = A2[:, 0] == 0
    a.plot(A2[s0, 2], pred[s0] * 1e3, 'o', color=S1, ms=9, mfc='none', mew=1.5, label='Mitnahme, masselos (alle 4 Niveaus, Abw. ≤ 5e-15)')
    a.plot(A2[~s0, 2], pred[~s0] * 1e3, 'x', color=S1, ms=8, mew=1.5, label='Mitnahme, μ = 0.03')
    A1 = d['A1']
    a.plot(A1[:, 2], (A1[:, 4] + A1[:, 5] + A1[:, 6] + A1[:, 7]) / 4 * 1e3, 's', color=S2, ms=7,
           label='richtungsabh. Bleibe-Gatter (Mittel): Vakuum-Verschiebung')
    a.set_xlabel('k_y'); a.set_ylabel('Energieverschiebung ΔE  (10⁻³)')
    title(a, '(a) Bandebene: der richtungsunabhängige Mitnahme-Sprung ist exakt\nb·p für alle Zustände; das richtungsabhängige Gatter verschiebt das Vakuum')
    a.legend(loc='upper left', fontsize=8)

    # (b) on/off ratios
    b = ax[0, 1]
    r = d['B_rows']
    keep = [i for i in range(len(r)) if r[i, 1] != 0 and r[i, 3] != 0]
    lab = []
    for i in keep:
        kind = 'Licht' if r[i, 0] == 0 else 'μ = 0.03'
        rel = 'parallel' if r[i, 1] * r[i, 3] > 0 else 'antiparallel'
        lab.append(f'{kind}, k = {abs(r[i,1]):.1f}, {rel}')
    y = np.arange(len(keep))
    b.plot([r[i, 18] for i in keep], y, 'o', color=S1, ms=9, label='Simulation')
    b.plot([r[i, 17] / r[i, 16] for i in keep], y, '|', color=INK, ms=14, mew=2, label='ART für genau diese Pakete')
    b.plot([r[i, 19] for i in keep], y, 'D', color=S3, ms=7, mfc='none', mew=1.5,
           label='ART, ideale Strahlen: 1 − 4v_s v_t / ((1+v_s²)(1+v_t²))')
    b.axvline(1, color=MUTED, lw=0.8, ls='--')
    b.set_yticks(y); b.set_yticklabels(lab, fontsize=9); b.invert_yaxis()
    b.set_xlabel('Anziehung mit g₀ᵢ / ohne g₀ᵢ')
    title(b, '(b) Zwei bewegte Schichten: parallele Lichtstrahlen ziehen sich nicht an,\nantiparallele doppelt so stark')
    b.legend(loc='center right', fontsize=8)

    # (c) co-moving sheets: relativity principle
    c = ax[1, 0]
    cm = d['B_comove']
    vv = np.linspace(0, 0.8, 100)
    gg = 1 / np.sqrt(1 - vv ** 2)
    c.plot(vv, gg ** -3, color=S1, lw=1.5, label='ART / Relativitätsprinzip: 1/γ³')
    c.plot(vv, gg * (1 + vv ** 2) ** 2, color=S2, lw=1.5, ls='--', label='nur Lapse und A (ohne g₀ᵢ): γ(1+v²)²')
    c.plot(np.r_[0, cm[:, 1]], np.r_[1, cm[:, 2]], 'o', color=S1, ms=9, label='Simulation mit g₀ᵢ')
    c.plot(np.r_[0, cm[:, 1]], np.r_[1, cm[:, 3]], 's', color=S2, ms=8, mfc='none', mew=1.5, label='Simulation ohne g₀ᵢ')
    c.plot(cm[:, 1], cm[:, 6], '|', color=INK, ms=14, mew=2, label='ART für genau diese Pakete')
    c.set_xlabel('v (lokal, Einheiten von c)'); c.set_ylabel('a(mitbewegt) / a(beide in Ruhe)')
    title(c, '(c) Mitbewegte Schichten: ohne g₀ᵢ verrät die Anziehung die Bewegung\ngegen das Gitter; mit g₀ᵢ gilt das Relativitätsprinzip')
    c.legend(loc='upper left', fontsize=8)

    # (d) gravitomagnetic force itself
    dd = ax[1, 1]
    dFs = (r[keep, 11] - r[keep, 10]).astype(float) / r[keep, 9].astype(float)
    dFg = (r[keep, 17] - r[keep, 16]).astype(float) / r[keep, 9].astype(float)
    lim = np.abs(np.r_[dFs, dFg]).max() * 1.1e3
    dd.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls='--')
    for i, (x_, y_) in enumerate(zip(dFg * 1e3, dFs * 1e3)):
        if r[keep[i], 0] == 0:
            dd.plot(x_, y_, 'o', color=S2, ms=6)
        else:
            dd.plot(x_, y_, 'o', color=S1, ms=13, mfc='none', mew=1.8)
    dd.plot([], [], 'o', color=S1, ms=13, mfc='none', mew=1.8, label='μ = 0.03')
    dd.plot([], [], 'o', color=S2, ms=6, label='Licht')
    dd.set_xlabel('ART: ΔF = −⟨∂ₓb⟩·sin(k_t l)/l  (10⁻³ κ)'); dd.set_ylabel('Simulation: F(mit g₀ᵢ) − F(ohne)  (10⁻³ κ)')
    title(dd, '(d) Die gravitomagnetische Kraft selbst: Simulation gegen ART\n(Quelle: Impulsdichte k_s ρ_s der Rechenlast, Faktor 4)')
    dd.legend(loc='upper left', fontsize=8)

    fig.suptitle('g₀ᵢ: Gravitomagnetismus aus dem Impulsstrom der Rechenlast (richtungsunabhängiger Mitnahme-Sprung)',
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open10.png'), dpi=120)
    print('saved figures/phase_bit_s4_open10.png')


if __name__ == '__main__':
    main()
