#!/usr/bin/env python3
"""Figure for the S4 follow-up (open points).  Reads data/phase_bit_s4_open.npz."""

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
    d = np.load(data_path('phase_bit_s4_open.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 4, figsize=(20, 9.5))

    # (a) 2D Kramers protection
    a = ax[0, 0]
    ev = np.log10(np.maximum(d['split2d_even'], 1e-17))
    od = np.log10(d['split2d_odd'])
    bins = np.linspace(-17, 0, 52)
    a.hist(ev, bins=bins, color=S1, label='T-gerade Terme (größte Aufspaltung)')
    a.hist(od, bins=bins, color=S2, label='T-ungerade Terme (kleinste Aufspaltung)')
    a.set_xlabel('log₁₀ Aufspaltung der k=0-Paare'); a.set_ylabel('Anzahl (je 500 Zufallsterme)')
    title(a, '(a) 2D: der Dirac-Punkt ist ein Kramers-Paar\nT-gerade Terme können ihn nicht öffnen')
    a.legend(loc='upper center', fontsize=8.5)

    # (b) theta independence
    b = ax[0, 1]
    th = d['theta'] / np.pi
    dev = np.maximum(d['theta_dev_min'][:, 0], 1e-17)
    b.semilogy(th, dev, 'o-', color=S1, ms=7)
    b.axhline(1e-14, color=MUTED, ls='--', lw=1)
    b.set_ylim(1e-17, 1e-1)
    b.set_xlabel('θ / π'); b.set_ylabel('max |E(θ) − E(0)|, 20 k-Punkte')
    title(b, '(b) Masse μ(cos θ·Π + sin θ·Π iH): Spektrum\nunabhängig von θ (exakte Gitter-Chiralität H)')

    # (c) background alpha vs lambda
    c = ax[0, 2]
    lam = d['lam_scan']
    s90 = d['scan90']
    c.plot(lam, s90[:, 0], color=S1, label='unterster Dirac-Punkt (ε = 0.3)')
    c.plot(lam, s90[:, 1], color=S2, label='innerer Dirac-Punkt (ε = 0.3)')
    c.plot(lam, s90[:, 2], color=S2, ls=':', label='innerer Dirac-Punkt (ε = 0.1)')
    c.axhline(0, color=MUTED, lw=0.8)
    c.axvline(float(d['lam_star']), color=MUTED, ls='--', lw=1)
    c.set_ylim(-0.1, 0.35); c.set_xlim(-0.95, 0.36)
    c.set_xlabel('λ  (Zusatzterm λ·ε·W₉₀, Spin-Transport)'); c.set_ylabel('α·ε  (Untergrund −αk²)')
    title(c, f"(c) Untergrund kompensierbar am inneren Punkt\nλ* = {float(d['lam_star']):+.4f}, unabhängig von ε")
    c.legend(loc='upper left', fontsize=8.5)

    # (d) dispersion
    dd = ax[0, 3]
    k = d['disp_k']
    for g, col, lab in (('plain', S1, 'ohne Kompensation'), ('comp', S2, 'kompensiert')):
        cur = d[f'disp_{g}'][0]          # [100]
        for j in range(4):
            dd.plot(k, cur[:, j], color=col, label=lab if j == 0 else None)
    m = float(d['disp_comp_m'])
    dd.plot(k, np.hypot(m, CP * k), color=MUTED, ls='--', lw=1.2, label='Dirac ±√(m²+(c/3)²k²)')
    dd.plot(k, -np.hypot(m, CP * k), color=MUTED, ls='--', lw=1.2)
    dd.set_xlabel('kₓ'); dd.set_ylabel('E − E_c(0)  (Quartett)')
    title(dd, '(d) Massives Quartett, ε = 0.3, μ = 0.03')
    dd.legend(loc='lower left', fontsize=8.5)

    # (e) boosts
    e = ax[1, 0]
    for g, col, lab in (('plain', S1, 'ohne Kompensation'), ('comp', S2, 'kompensiert')):
        kb, vs, vg, vd = (d[f'dyn_{g}_boost_{j}'] for j in range(4))
        e.plot(kb, vs, 's', color=col, ms=8, mfc='white', mew=1.5, label=f'{lab}: Schwerpunkt')
        e.plot(kb, vg, '-', color=col, lw=1.2, label=f'{lab}: v_g des Bandes')
    kk = np.linspace(0, 0.2, 100)
    e.plot(kk, CP ** 2 * kk / np.hypot(float(d['disp_comp_m']), CP * kk), color=MUTED, ls='--', lw=1.2,
           label='Dirac c′²k/E')
    e.set_xlabel('kₓ'); e.set_ylabel('vₓ')
    title(e, '(e) Bewegtes Paket im Dirac-Regime (μ/ε = 0.1, σ = 40)')
    e.legend(loc='upper left', fontsize=8)

    # (f) rest spreading
    f = ax[1, 1]
    for g, col, lab in (('plain', S1, 'ohne Kompensation'), ('comp', S2, 'kompensiert')):
        t, rms, pred = (d[f'dyn_{g}_rest_{j}'] for j in range(3))
        f.plot(t, rms, color=col, label=lab)
    f.plot(t, pred, color=MUTED, ls='--', lw=1.2, label='Dirac (ballistisch)')
    f.set_xlabel('t'); f.set_ylabel('rms-Radius')
    title(f, '(f) Ruhendes Paket zerfließt wie ein Dirac-Teilchen')
    f.legend(loc='upper left', fontsize=8.5)

    # (g)/(h) Zitterbewegung
    for axx, g, col, lab in ((ax[1, 2], 'plain', S1, 'ohne Kompensation'),
                             (ax[1, 3], 'comp', S2, 'kompensiert')):
        t, y, om, omp = (d[f'dyn_{g}_zb_{j}'] for j in range(4))
        yy = y - np.polyval(np.polyfit(t, y, 2), t)
        axx.plot(t, yy, color=col)
        axx.set_xlabel('t'); axx.set_ylabel('Schwerpunkt y (Trend entfernt)')
        title(axx, f'({"g" if g == "plain" else "h"}) Zitterbewegung, {lab}, k₀ = 0.03\n'
                   f'ω_Fit = {float(om):.4f}, Bandabstand {float(omp):.4f}')

    fig.suptitle('S4, offene Punkte: Symmetrien der Masse, 2D-Kramers-Schutz, Kompensation des '
                 'Untergrunds, Dynamik im Dirac-Regime', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open.png'), dpi=120)
    print('saved figures/phase_bit_s4_open.png')


if __name__ == '__main__':
    main()
