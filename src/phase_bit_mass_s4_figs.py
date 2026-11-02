#!/usr/bin/env python3
"""Figure for Experiment S4 (mass for the 3D spin vortex).  Reads data/phase_bit_mass_s4.npz."""

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
C_LIGHT = np.sqrt(3.0)
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0, 'legend.frameon': False,
})


def main():
    d = np.load(data_path('phase_bit_mass_s4.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))

    # (a) Dirac quartet without / with mass
    a = ax[0, 0]
    k = d['kline']
    for mu, col in ((0.0, S1), (0.1, S2)):
        E = d[f'bands_mu{mu}'][:, :4]
        for j in range(4):
            a.plot(k, E[:, j], color=col, label=f'μ = {mu}' if j == 0 else None)
            a.plot(-k, E[:, j], color=col)
    a.set_xlabel('kₓ'); a.set_ylabel('E (unterstes Quartett)')
    a.set_title('(a) Dirac-Punkt (μ = 0) vs. Lücke mit μ·Π\nε = 0.3, minimale SU(2)',
                loc='left', color=INK)
    a.legend(loc='lower center')

    # (b) gap vs mu
    b = ax[0, 1]
    mu = d['gap_mu']
    b.plot(mu, d['gap'], 'o', color=S1, ms=8, label='minimale SU(2)')
    b.plot(mu, d['gap_2O'], 's', color=S2, ms=6, mfc='white', mew=1.5, label='2O-Register')
    b.plot(mu, 4 * mu, color=MUTED, ls='--', lw=1.2, label='4μ')
    b.set_xlabel('μ'); b.set_ylabel('Lücke bei k = 0')
    b.set_title('(b) Lücke exakt 4μ in beiden Modellen\n(Kramers bleibt, 2O bleibt E ↔ −E)',
                loc='left', color=INK)
    b.legend(loc='upper left')

    # (c) split vs Dirac, background
    c = ax[0, 2]
    rows = d['bg_rows']
    for m_, col in ((0.0, S1), (0.05, S2), (0.15, S3)):
        sel = np.isclose(rows[:, 0], m_)
        c.plot(rows[sel, 1], 100 * rows[sel, 3], 'o-', color=col, ms=5,
               label=f'μ = {m_}')
    c.axhline(0, color=MUTED, lw=0.8)
    c.set_xlabel('kₓ'); c.set_ylabel('Δ(k) / √(m² + (c/3)²k²) − 1  [%]')
    c.set_title('(c) Aufspaltung folgt Dirac mit c/3 bei kleinem k\n'
                '(dazu gemeinsamer Untergrund −αk², αε ≈ 0.125)', loc='left', color=INK)
    c.legend(loc='lower left')

    # (d) rms at rest
    dd = ax[1, 0]
    t = d['rest_t']
    dd.plot(t, d['rest_rms0'], color=S1, label='μ = 0 (masselos)')
    dd.plot(t, d['rest_rms1'], color=S2, label=f"μ = {float(d['MU'])} (massiv)")
    dd.set_xlabel('t'); dd.set_ylabel('rms-Radius')
    dd.set_title('(d) Ruhendes Paket (3+1D, 81³ Gitter)', loc='left', color=INK)
    dd.legend(loc='upper left')

    # (e) boosted packets
    e = ax[1, 1]
    e.plot(d['boost_k'], d['v_band'], 'o-', color=S2, ms=6, label='v_g des Bandes')
    e.plot(d['boost_k'], d['v_sim'], 's', color=S2, ms=8, mfc='white', mew=1.5,
           label='v des Schwerpunkts (Gitter)')
    e.axhline(0, color=MUTED, lw=0.8)
    e.set_xlabel('kₓ'); e.set_ylabel('vₓ')
    e.set_title('(e) Bewegter massiver Zustand, |v| ≪ c = 1.73\n'
                '(oberes Dublett läuft gegen k: Untergrund −αk²)', loc='left', color=INK)
    e.legend(loc='lower left')

    # (f) Zitterbewegung: transverse CoM
    f = ax[1, 2]
    t = d['zb_t']
    y = d['zb_y'] - np.polyval(np.polyfit(t, d['zb_y'], 2), t)
    f.plot(t, y, color=S1)
    f.set_xlabel('t'); f.set_ylabel('Schwerpunkt y (Trend entfernt)')
    f.set_title(f"(f) Zitterbewegung, k₀ = 0.3: ω_Fit = {float(d['zb_omega_y']):.3f}\n"
                f"Bandabstand {float(d['zb_omega_pred']):.3f}", loc='left', color=INK)

    fig.suptitle('Experiment S4 — Masse für den 3D-Spin-Wirbel: μ·Π (Umkehr ohne '
                 'Spindrehung) öffnet den Dirac-Punkt und erhält die Zeitumkehr',
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_mass_s4.png'), dpi=130)
    print('saved figures/phase_bit_mass_s4.png')


if __name__ == '__main__':
    main()
