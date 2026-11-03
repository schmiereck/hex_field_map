#!/usr/bin/env python3
"""Figure for the fifth S4 follow-up.  Reads data/phase_bit_s4_open5.npz."""

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
    d = np.load(data_path('phase_bit_s4_open5.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))

    # (a) slope = active weight: one buffer down to N = 0.02, random two-buffer schemes
    a = ax[0, 0]
    r = d['A1_rows']
    g = d['A1_general']
    a.plot([0, 1], [0, 1], color=MUTED, lw=1, ls='--', label='Steigung = aktiver Anteil')
    a.plot(1 - r[:, 2], r[:, 1], 'o', color=S1, ms=8, label='ein Puffer, N = 0.4 … 0.02')
    a.plot(g[:, 0], g[:, 1], 's', color=S2, ms=8, label='zwei Puffer, zufällige lokale Mischung')
    a.set_xlabel('aktiver Anteil des Vakuum-Fixvektors (1 − Pufferanteil)')
    a.set_ylabel("Steigung des Dirac-Kegels / c′")
    title(a, f"(a) Ein Puffer reicht für 0 < N ≤ 1 (Formel: {float(d['A1_err']):.0e});\n"
             "allgemein: Lapse = aktiver Anteil des Fixvektors")
    a.legend(loc='upper left', fontsize=8)

    # (b) horizon approach
    b = ax[0, 1]
    rec, ray = d['A2_rec'], d['A2_ray']
    Nmax, Nmin, x1, x2 = d['A2_prof']
    turn = d['A2_turn']
    b.plot(rec[:, 0], rec[:, 1], color=S1, label='Simulation ⟨x⟩')
    okr = ray[:, 3] > d['A2_band'][-1, 0] + 0.05          # inside the tabulated band
    b.plot(ray[okr, 0], ray[okr, 2], color=INK, lw=1.2, ls='--', label='Strahl, exakte Gitterdispersion')
    b.plot(ray[:, 0], ray[:, 1], color=MUTED, lw=1.2, ls=':', label="Strahl, Kegel c′N(x)")
    b.axhline(turn[1], color=S2, lw=1, label=f'Umkehrpunkt f_N(E_max) = E′ (N = {turn[0]:.3f})')
    b.axhspan(-60, x1, color=GRID, alpha=0.6, lw=0)
    b.text(rec[-1, 0] * 0.98, x1 - 3, f'N = {Nmin}', ha='right', va='top', color=MUTED, fontsize=8)
    b.set_ylim(-62, 70)
    b.set_xlabel('Zeit t'); b.set_ylabel('Position x')
    title(b, '(b) Photon läuft auf N → 0 zu: blauverschoben bis zur\nGittergrenze, dann Umkehr (kein Horizont auf dem Gitter)')
    b.legend(loc='upper right', fontsize=7.5)

    # (c) third law: E_i * displacement_i
    c = ax[0, 2]
    two = d['B_two']
    E1, E2 = two[4], two[5]
    x1c, x2c = d['B_xc1'], d['B_xc2']
    t = np.arange(len(x1c)) * 0.5
    c.plot(t, E1 * (x1c - x1c[0]), color=S1, label=f'E₁·Δx₁  (m = {2*0.03:.2f})')
    c.plot(t, -E2 * (x2c - x2c[0]), color=S2, ls='--', label=f'−E₂·Δx₂  (m = {2*0.06:.2f})')
    c.set_xlabel('Zeit t'); c.set_ylabel('E · Verschiebung')
    title(c, f'(c) Exp. 1d: Lapse aus der lokalen Last e(x), zwei Pakete\n'
             f'Sim./Halbkl.: a₁ {two[0]/two[11]:.3f}, a₂ {two[1]/two[12]:.3f}; drittes Gesetz bis {abs(two[6])*100:.1f} %')
    c.legend(loc='upper left', fontsize=8)

    # (d) active mass of a moving source
    dd = ax[1, 0]
    mv = d['B_moving']
    er = mv[:, 3] / mv[0, 3]
    ar = mv[:, 1] / mv[0, 1]
    m = 0.06
    v2 = 1 - (m / mv[:, 3]) ** 2
    xx = np.linspace(1, er.max() * 1.05, 50)
    dd.plot(xx, xx, color=MUTED, lw=1, ls='--', label='aktive Masse = E (Last)')
    vv = 1 - (1 / xx) ** 2
    dd.plot(xx, xx * (1 + vv), color=MUTED, lw=1, ls=':', label='ART (Testteilchen in Ruhe): E(1 + v²)')
    dd.plot(er, ar, 'o', color=S1, ms=9, label='Simulation')
    for k, x_, y_ in zip(mv[:, 0], er, ar):
        dd.annotate(f'k_y = {k}', (x_, y_), textcoords='offset points', xytext=(6, -12), fontsize=8, color=MUTED)
    dd.set_xlabel('E_Quelle / E_Quelle(Ruhe)'); dd.set_ylabel('a_Test / a_Test(ruhende Quelle)')
    title(dd, '(d) Quelle in Querbewegung: zieht mit ihrer Energie,\nnicht mit E(1 + v²) wie in der ART')
    dd.legend(loc='upper left', fontsize=8)

    # (e) anisotropic metric: waits vs coin couplings
    e = ax[1, 1]
    c1 = d['C1']
    pl = d['C2_plus']
    xs_ = np.arange(3)
    gr = (pl[1] - 1) * 100
    waits = (c1[:3] / 0.75 - 1) * 100
    coin = (pl[0] - 1) * 100
    e.bar(xs_ - 0.27, gr, width=0.25, color=MUTED, label='ART (g = 1 + 2h, h_plus = 0.05)')
    e.bar(xs_, waits, width=0.25, color=S1, label='Wartezeiten je Richtung (Puffer ∝ Kantenlänge)')
    e.bar(xs_ + 0.27, coin, width=0.25, color=S2, label='Coin-Kopplungen je Richtung')
    e.axhline(0, color=MUTED, lw=0.8)
    e.set_xticks(xs_); e.set_xticklabels(['x', 'y', 'z'])
    e.set_ylabel('Änderung der Lichtgeschwindigkeit (%)')
    resp = d['C2_resp']
    title(e, f'(e) Anisotrope Metrik: Warten mittelt sich weg,\nCoin-Kopplungen wirken (Antwort E_g {resp[0]:.2f}, T_2g {resp[1]:.2f})')
    e.set_ylim(-8, 7)
    e.legend(loc='lower right', fontsize=7.5)

    # (f) time-dependent lapse: leakage and phase error vs ramp time
    f = ax[1, 2]
    r3 = d['A3_rows']
    f.loglog(r3[:, 0], r3[:, 1], 'o-', color=S1, ms=7, label='Anteil außerhalb des Bandzustands')
    f.loglog(r3[:, 0], np.abs(r3[:, 3]), 's-', color=S2, ms=7, label='|Phase − Σ f_N(t)(θ)|')
    tt = np.array([4, 1000.0])
    f.loglog(tt, r3[0, 3] * 4 / tt, color=MUTED, lw=1, ls=':', label='∝ 1/T')
    f.loglog(tt[:1].tolist() + [300], [r3[0, 1], r3[0, 1] * (4 / 300) ** 3], color=MUTED, lw=1, ls='--', label='∝ 1/T³')
    f.set_xlabel('Rampendauer T (Takte), N: 0.9 → 0.3 gleichförmig'); f.set_ylabel('Abweichung')
    wind = (float(d['D_wind_168']), float(d['D_wind_336']))
    title(f, '(f) Zeitabhängiger Lapse: adiabatisch, Phase = Σ N(t)·E·Δt\n'
             f'(Teil D: Windung von det U(k_z) = {wind[0]:+.0f} / {wind[1]:+.0f} für B = 0.100 / 0.050)')
    f.legend(loc='lower left', fontsize=8)
    fig.suptitle('S4, fünfte Runde: starker und zeitabhängiger Lapse, Lapse aus der Last (Exp. 1d), '
                 'anisotrope Metrik, globale Landau-Zweige', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open5.png'), dpi=120)
    print('saved figures/phase_bit_s4_open5.png')


if __name__ == '__main__':
    main()
