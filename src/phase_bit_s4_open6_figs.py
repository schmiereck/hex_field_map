#!/usr/bin/env python3
"""Figure for the sixth S4 follow-up.  Reads data/phase_bit_s4_open6.npz."""

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
    d = np.load(data_path('phase_bit_s4_open6.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))

    # rows: (ky, mu, E, a_load, a_gr, Q, Q_M, E(1+v^2))
    r = d['A_rows']
    m = 0.06
    v2 = np.where(r[:, 1] > 0, 1 - (m / r[:, 2]) ** 2, 1.0)

    # (a) GR source / load source = 1 + v^2
    a = ax[0, 0]
    vv = np.linspace(0, 1, 50)
    a.plot(vv, 1 + vv, color=MUTED, lw=1.2, ls='--', label='ART: (T₀₀ + T_ii)/T₀₀ = 1 + v²')
    mass = r[:, 1] > 0
    a.plot(v2[mass], (r[:, 4] / r[:, 3])[mass], 'o', color=S1, ms=9, label='massive Quelle (ruhend, k_y = 0.3, 0.6)')
    a.plot(v2[~mass], (r[:, 4] / r[:, 3])[~mass], 's', color=S2, ms=9, label='Licht (k_y = 0.2, 0.4)')
    a.set_xlabel('v² / c′² der Quelle (Dirac)'); a.set_ylabel('a_Test(Quelle 2e − e_M) / a_Test(Quelle e)')
    title(a, '(a) Quelle s = 2e − e_M: der Bewegungsanteil der Rechnung\nzählt doppelt → 1 + v², Licht 2×')
    a.legend(loc='upper left', fontsize=8)

    # (b) absolute: both sources against their predictions
    b = ax[0, 1]
    pl = r[:, 2] / m
    pg = r[:, 7] / m
    b.plot([0, 12], [0, 12], color=MUTED, lw=1, ls='--')
    b.plot(pl, r[:, 3] / r[0, 3], 'o', color=S3, ms=9, label='Quelle e (Runde 5): gegen E/m')
    b.plot(pg, r[:, 4] / r[0, 4], 'D', color=S1, ms=8, label='Quelle 2e − e_M: gegen E(1 + v²)/m')
    b.set_xlabel('Vorhersage (relativ zur ruhenden Quelle)'); b.set_ylabel('a_Test / a_Test(ruhende Quelle)')
    title(b, '(b) Testteilchen in Ruhe: Stärke der Anziehung\n(Abweichung ≤ 2.5 %, Ruhe-Normierung 1 %)')
    b.legend(loc='upper left', fontsize=8)

    # (c) causal gravity
    c = ax[0, 2]
    ts = d['B_ts']
    g, w, pr = d['B_green'], d['B_wave'], d['B_pred']
    fit = d['B_fit']
    c.plot(ts, g[0] - g[0][0], color=MUTED, lw=1.5, label='instantanes Feld (Green-Funktion)')
    c.plot(ts, w[0] - w[0][0], color=S1, label='lokale Wellengleichung, c_g = c′')
    c.plot(ts, pr, color=INK, lw=1.2, ls='--', label='Vorhersage a·P(|Δx| < c′t)')
    c.axvline(fit[1], color=S2, lw=1, label=f'D / c′ = {fit[1]:.0f}')
    c.set_ylim(-0.5, 6)
    c.set_xlabel('Zeit t'); c.set_ylabel('Verschiebung des Testteilchens')
    title(c, '(c) Kausale Gravitation: das Testteilchen spürt die Quelle\nerst, wenn das Feld mit c′ angekommen ist')
    c.legend(loc='upper left', fontsize=8)

    # (d) lattice mirror, massive
    dd = ax[1, 0]
    Nmax, Nmin, x1, x2 = d['C_prof']
    for key, col, lab in (('0.15', S1, 'μ = 0.15 (fällt aus der Ruhe)'), ('0.03', S3, 'μ = 0.03 (fällt aus der Ruhe)'),
                          ('0.0', S2, 'masselos, k = −0.3')):
        rec, info = d[f'C_rec_{key}'], d[f'C_info_{key}']
        dd.plot(rec[:, 0], rec[:, 1], color=col, label=lab)
        if info[2] > Nmin:
            dd.axhline(info[3], color=col, lw=0.9, ls=':')
    dd.axhspan(-62, x1, color=GRID, alpha=0.6, lw=0)
    dd.text(1980, x1 - 2, f'N = {Nmin}', ha='right', va='top', color=MUTED, fontsize=8)
    dd.set_xlim(0, 2000); dd.set_ylim(-62, 65)
    dd.set_xlabel('Zeit t'); dd.set_ylabel('⟨x⟩')
    title(dd, '(d) Spiegel für Masse: Umkehr bei f_N(E_max) = E′ (punktiert),\nRückkehr fast auf die Starthöhe; zu wenig Energie → ins Innere')
    dd.legend(loc='upper right', fontsize=8)

    # (e) band ranges of the helicity sector
    e = ax[1, 1]
    rg = d['D_ranges']
    for bnd, (lo, hi) in enumerate(rg):
        e.plot([lo, hi], [bnd, bnd], color=S1, lw=6, solid_capstyle='butt')
    k0 = [-2.078, -0.570, 0.693, 1.263]
    for E0, lab in zip(k0, ('unterer Knoten', '4-fach', 'innerer Knoten', '4-fach')):
        e.axvline(E0, color=S2 if 'Knoten' in lab else MUTED, lw=1, ls='--')
    e.axvline(-2 * np.pi, color=INK, lw=0.8); e.axvline(2 * np.pi, color=INK, lw=0.8)
    e.set_yticks(range(12)); e.set_ylabel('Band (sortiert)'); e.set_xlabel('Quasienergie E (Zone ±2π)')
    title(e, '(e) Helizitätssektor +1: die Bänder überdecken den ganzen\nQuasienergie-Kreis (gestrichelt: Niveaus bei k = 0)')

    # (f) coverage
    f = ax[1, 2]
    cv = d['D_cover']
    inn = np.abs(cv[:, 0]) < 2 * np.pi - 0.005          # grid end points sit exactly on the zone edge
    f.plot(cv[inn, 0], cv[inn, 1], color=S1, lw=1.5)
    f.set_xlabel('Quasienergie E'); f.set_ylabel('Anzahl Bänder bei dieser Energie')
    mg = d['D_mingap'][:, 1]
    title(f, f'(f) Keine globale Lücke (kleinste Lücke benachbarter Bänder\n≤ {mg.max():.1e}): nur der Netto-Fluss (Windung 0) ist topologisch')
    f.set_ylim(0, 8)

    fig.suptitle('S4, sechste Runde: Quelle T₀₀ + T_ii aus der Rechenlast, kausale Gravitation, Spiegel für Masse, '
                 'Bandstruktur des Helizitätssektors', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open6.png'), dpi=120)
    print('saved figures/phase_bit_s4_open6.png')


if __name__ == '__main__':
    main()
