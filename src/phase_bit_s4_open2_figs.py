#!/usr/bin/env python3
"""Figure for the second S4 follow-up.  Reads data/phase_bit_s4_open2.npz."""

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
    d = np.load(data_path('phase_bit_s4_open2.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))

    # (a) background of the 2O quartet vs lambda
    a = ax[0, 0]
    lam, al = d['A_lams'], d['A_alpha']
    a.plot(lam, al, 'o-', color=S1, ms=3)
    a.axhline(0, color=MUTED, lw=0.8)
    a.axvline(float(d['A_lam_star']), color=MUTED, ls='--', lw=1)
    a.set_ylim(-0.1, 0.25)
    a.set_xlabel('λ  (G₂O + λ·W₉₀, beide aus 2O)'); a.set_ylabel('α·ε des Dirac-Quartetts (ε = 0.05)')
    title(a, f"(a) 2O: Untergrund verschwindet bei λ* = {float(d['A_lam_star']):.3f} (ε→0)\n"
             f"(2. Ordnung: {float(d['A_lam_bch']):.3f}); E ↔ −E bleibt exakt")

    # (b) dispersion
    b = ax[0, 1]
    k = d['A_kl']
    for tag, col, ls, lab in (('2O_plain', MUTED, ':', '2O ohne Zusatzterm'),
                              ('2O', S1, '-', '2O + λ*(ε)·ε·W₉₀'),
                              ('comp', S2, '-.', 'SU(2) kompensiert (S4+)')):
        D = d[f'A_disp_{tag}'].astype(float).copy()
        jump = np.r_[False, (np.abs(np.diff(D, axis=0)) > 0.1).any(1)]
        D[np.cumsum(jump) > 0] = np.nan          # stop at the first band re-ordering
        for j in range(4):
            b.plot(k, D[:, j], color=col, ls=ls, lw=1.8, label=lab if j == 0 else None)
    m = 0.3
    b.plot(k, np.hypot(m, CP * k), color=INK, ls='--', lw=1, label='Dirac ±√(m²+(c/3)²k²)')
    b.plot(k, -np.hypot(m, CP * k), color=INK, ls='--', lw=1)
    b.set_xlabel('kₓ'); b.set_ylabel('E − E_c')
    title(b, '(b) Massives Quartett, ε = 0.3, μ = 0.15 (μ/ε = 0.5)')
    b.legend(loc='lower left', fontsize=8)

    # (c) theta with an H-breaking mass
    c = ax[0, 2]
    th = d['B_theta']
    g = d['B_gap']
    c.plot(th / np.pi, g / (4 * 0.05), 'o', color=S1, ms=6, label='Gitter (2O)')
    tt = np.linspace(0, 2 * np.pi, 200)
    c.plot(tt / np.pi, np.abs(1 + np.exp(1j * tt)), color=MUTED, ls='--', lw=1.2, label='|1 + e^{iθ}|')
    c.set_xlabel('θ / π'); c.set_ylabel('Lücke / 4μ')
    title(c, '(c) θ wird beobachtbar neben einer\nhelizitätsbrechenden Masse (Spin-Identität)')
    c.legend(loc='upper center', fontsize=8.5)

    # (d) domain walls
    dd = ax[1, 0]
    for key, col, mk, lab in (('flip', S1, 'o', 'Vorzeichenwechsel (T erhalten)'),
                              ('chiral', S2, 's', 'chirale Drehung 0→π (T gebrochen)')):
        kx = d[f'C_{key}_kx']
        W = d[f'C_{key}_wall']
        first = True
        for i, kv in enumerate(kx):
            ys = W[i][~np.isnan(W[i])] if W.ndim == 2 else np.array([])
            if len(ys):
                dd.plot([kv] * len(ys), ys, mk, color=col, ms=7, mfc='white' if key == 'chiral' else col,
                        mew=1.5, label=lab if first else None)
                first = False
        if first:
            dd.plot([], [], mk, color=col, label=lab + ': keine')
    kk = np.linspace(0, 0.07, 50)
    dd.plot(kk, CP * kk, color=MUTED, ls='--', lw=1.2, label='±(c/3)k')
    dd.plot(kk, -CP * kk, color=MUTED, ls='--', lw=1.2)
    dd.axhspan(-0.1, 0.1, color=GRID, alpha=0.4, lw=0)
    dd.set_ylim(-0.1, 0.1)
    dd.set_xlabel('k_∥'); dd.set_ylabel('E − E_c  (Wandzustände)')
    title(dd, '(d) Massen-Domänenwand: masseloser 2D-Kegel\nnur bei T-erhaltendem Vorzeichenwechsel')
    dd.legend(loc='lower left', fontsize=8)

    # (e)/(f) Zitterbewegung
    for axx, tag, lab in ((ax[1, 1], 'plain', 'ohne Kompensation'), (ax[1, 2], 'comp', 'kompensiert')):
        t = d[f'D_{tag}_t']
        axx.plot(t, d[f'D_{tag}_sim'], color=S1, lw=2.5, label='Simulation')
        axx.plot(t, d[f'D_{tag}_band'], color=S2, lw=1.2, ls='--', label='Bandtheorie (Projektoren)')
        axx.plot(t, d[f'D_{tag}_dirac'], color=S3, lw=1.2, ls=':', label='Dirac 4×4')
        axx.set_xlabel('t'); axx.set_ylabel('Schwerpunkt y (oszillierender Teil)')
        title(axx, f'({"e" if tag == "plain" else "f"}) Zitterbewegung, {lab}, k₀ = 0.03\n'
                   'Anfangszustand: Helizität +1')
        axx.legend(loc='upper right', fontsize=8.5)

    fig.suptitle('S4, neue offene Punkte: 2O ohne Untergrund (E ↔ −E erhalten), θ mit Chiralitätsbruch, '
                 'Massen-Domänenwände, Amplitude der Zitterbewegung', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open2.png'), dpi=120)
    print('saved figures/phase_bit_s4_open2.png')


if __name__ == '__main__':
    main()
