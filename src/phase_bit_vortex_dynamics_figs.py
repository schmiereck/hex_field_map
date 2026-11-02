#!/usr/bin/env python3
"""Figure for Experiment S2 (vortex dynamics).  Reads data/phase_bit_vortex_dynamics.npz."""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import data_path, fig_path
from quantum_hex_turning import DX_PHYS, DY_PHYS, C_LIGHT

S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3de'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0, 'legend.frameon': False,
})


def density(ax, rho, title, vmax):
    nx, ny = rho.shape
    ext = [-(nx // 2) * DX_PHYS, (nx // 2) * DX_PHYS,
           -(ny // 2) * DY_PHYS, (ny // 2) * DY_PHYS]
    # sublattice: merge neighbouring columns for display
    img = rho + np.roll(rho, 1, axis=0)
    ax.imshow(img.T, origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=vmax,
              aspect='equal', interpolation='nearest')
    ax.set_xlim(-110, 110); ax.set_ylim(-110, 110)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(False)
    ax.set_title(title, loc='left', color=INK)


def main():
    d = np.load(data_path('phase_bit_vortex_dynamics.npz'))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))

    # (a) bands near the top doublet
    a = ax[0, 0]
    k = d['k_bands']
    for mu, col, ls in ((0.0, S1, '-'), (0.05, S2, '-')):
        Eb = d[f'bands_mu{mu}']
        a.plot(k, Eb[:, -1], color=col, ls=ls, label=f'μ = {mu}')
        a.plot(k, Eb[:, -2], color=col, ls=ls)
        a.plot(-k, Eb[:, -1], color=col, ls=ls)
        a.plot(-k, Eb[:, -2], color=col, ls=ls)
    a.set_xlabel('kₓ'); a.set_ylabel('E (Spin-Sektor, oberes Dublett)')
    a.set_title('(a) Kramers-Kegel (μ = 0) vs. Lücke (μ > 0)', loc='left', color=INK)
    a.legend(loc='upper center')
    a.text(0.03, 0.80, f"Aufspaltung bei k = 0: 8μ\n(j = {int(d['j_up'])} oben, "
           f"j = {int(d['j_down'])} unten)", transform=a.transAxes, color=INK, fontsize=9)

    vmax = max(d['ring_rho'].max(), d['rest_rho'].max()) * 1.2
    density(ax[0, 1], d['ring_rho'], '(b) μ = 0: ruhender Wirbel, t = 100\n'
            '→ Ring, der mit c/2 wächst', vmax)
    density(ax[0, 2], d['rest_rho'], '(c) μ = 0.05: massiver Wirbel, t = 100\n'
            '→ bleibt am Ort, zerfließt nur', vmax)

    # (d) rms radius
    dd = ax[1, 0]
    dd.plot(d['ring_t'], d['ring_rms'], color=S1, label='μ = 0 (masselos)')
    dd.plot(d['rest_t'], d['rest_rms'], color=S2, label='μ = 0.05 (massiv)')
    tt = d['ring_t']
    dd.plot(tt, 6 + C_LIGHT / 2 * tt, color=MUTED, ls='--', lw=1.2, label='Steigung c/2')
    dd.set_xlabel('t'); dd.set_ylabel('rms-Radius')
    dd.set_title('(d) Ausbreitung des Wirbels', loc='left', color=INK)
    dd.legend(loc='upper left')

    # (e) velocity of boosted massive vortex packets
    e = ax[1, 1]
    e.plot(d['boost_k'], d['v_band'], 'o-', color=S2, ms=6, label='v_g des Bandes')
    e.plot(d['boost_k'], d['v_sim'], 's', color=S2, ms=8, mfc='white', mew=1.5,
           label='v des Paket-Schwerpunkts (Gitter)')
    e.axhline(C_LIGHT, color=INK, ls='--', lw=1.2)
    e.text(0.1, C_LIGHT + 0.04, 'c = √3', color=INK)
    e.axhline(C_LIGHT / 2, color=MUTED, ls=':', lw=1.2)
    e.text(0.40, C_LIGHT / 2 + 0.04, 'c/2 (Kegelsteigung)', color=MUTED)
    e.set_ylim(0, 2.0)
    e.set_xlabel('kₓ'); e.set_ylabel('Geschwindigkeit')
    e.set_title('(e) Massiver Wirbel bewegt sich mit v < c', loc='left', color=INK)
    e.legend(loc='lower right')

    # (f) Zitterbewegung in heading space
    f = ax[1, 2]
    t = d['zb_t']
    ju, jd = int(d['j_up']), int(d['j_down'])
    f.plot(t, d['zb_jpop'][:, ju], color=S2, label=f'Anteil Wirbel j = {ju}')
    f.plot(t, d['zb_jpop'][:, jd], color=S1, label=f'Anteil Wirbel j = {jd}')
    f.set_xlim(0, 80)
    f.set_xlabel('t'); f.set_ylabel('Wirbel-Anteil')
    f.set_title(f"(f) Zitterbewegung im Richtungsraum, k₀ = 0.3\n"
                f"ω_Fit = {float(d['zb_omega_j']):.4f}, Bandabstand "
                f"{float(d['zb_omega_band']):.4f}", loc='left', color=INK)
    f.legend(loc='center right')

    fig.suptitle('Experiment S2 — Dynamik der Spin-Wirbel (Phasen-Bit-Walker auf Z₁₂, '
                 'exakt unitär)', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_vortex_dynamics.png'), dpi=130)
    print('saved figures/phase_bit_vortex_dynamics.png')


if __name__ == '__main__':
    main()
