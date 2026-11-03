#!/usr/bin/env python3
"""Figure for the fourth S4 follow-up.  Reads data/phase_bit_s4_open4.npz."""

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
DT = 0.5
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0, 'legend.frameon': False,
})


def title(ax, s):
    ax.set_title(s, loc='left', color=INK, fontsize=10)


def f_dwell(theta, a):
    return theta / 2 + np.arcsin(np.cos(a) * np.sin(theta / 2))


def main():
    d = np.load(data_path('phase_bit_s4_open4.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))

    # (a) eigenphase map
    a_ = ax[0, 0]
    th = np.linspace(-np.pi, np.pi, 400)
    for N, col in ((1.0, MUTED), (0.875, S1), (0.75, S2), (0.5, S3)):
        a = np.arccos(2 * N - 1)
        a_.plot(th / DT, f_dwell(th, a) / DT, color=col, label=f'N = {N}' + (' (ohne Puffer)' if N == 1 else ''))
    a_.plot(th / DT, 0.75 * th / DT, color=S2, ls=':', lw=1.2, label='N·E (N = 0.75)')
    a_.set_xlabel('E  (Walker ohne Puffer)'); a_.set_ylabel("E′  (mit Puffer)")
    title(a_, f"(a) Puffer-Lapse: E′ = f(E), Steigung N = (1+cos a)/2 bei E = 0\n"
              f"(Formel gegen exaktes 48×48-Spektrum: {float(d['A_formula_err']):.0e})")
    a_.legend(loc='upper left', fontsize=8)

    # (b) vacuum shift vs buffer phase (analytic curve, lattice points where unambiguous)
    b = ax[0, 1]
    cols = [S1, S2, S3, S4, MUTED]
    aa = np.linspace(1e-3, np.pi / 2, 200)
    for ph, v, col in zip(d['A_phis'], d['A_vac'], cols):
        phi = ph * np.pi
        ev = (-phi / 2 + np.arccos(np.cos(aa) * np.cos(phi / 2))) / DT
        b.plot((1 + np.cos(aa)) / 2, ev, color=col, label=f'Pufferphase {ph:.2f}π')
        evp = (-phi / 2 + np.arccos(np.cos(d['A_as']) * np.cos(phi / 2))) / DT
        ok = np.abs(v - evp) < 1e-6
        b.plot(((1 + np.cos(d['A_as'])) / 2)[ok], v[ok], 'o', color=col, ms=4)
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_xlabel('Lapse N'); b.set_ylabel('Energie des Dirac-Punkts nach dem Puffern')
    title(b, '(b) Nur die Pufferphase π lässt das Vakuum bei E = 0\n(sonst hängt seine Energie von N ab und gravitiert; Punkte: Gitter)')
    b.legend(loc='upper right', fontsize=8)

    # (c) R' vs residual offset
    c = ax[0, 2]
    R = d['A_offset'].astype(float)
    c.plot(R[:, 0], R[:, 1], 'o', color=S1, ms=8, label='Teilchen (Gitter)')
    c.plot(R[:, 0], R[:, 2], 's', color=S2, ms=8, label='Antiteilchen (Gitter)')
    c.plot(R[:, 0], R[:, 3], color=S1, lw=1.2, label='1 + δ/m')
    c.plot(R[:, 0], R[:, 4], color=S2, lw=1.2, label='1 − δ/m')
    c.axhline(1, color=MUTED, ls='--', lw=1); c.axvline(0, color=MUTED, lw=0.8)
    c.set_xlabel('Restenergie δ des Dirac-Punkts (Gegenterm verstimmt)'); c.set_ylabel("R′")
    title(c, '(c) Teilchen und Antiteilchen fallen nur bei δ = 0 gleich\n(μ = 0.03, m = 0.06): der Gegenterm ist erzwungen')
    c.legend(loc='upper center', fontsize=8)

    # (d) accelerations
    dd = ax[1, 0]
    rows = d['B_rows']
    g, N0 = float(d['B_g']), float(d['B_N0'])
    ref = -g * N0 ** 2 * CP ** 2
    xs = np.arange(len(rows))
    dd.bar(xs - 0.18, [r[2] / ref for r in rows], width=0.34, color=S1, label='Puffer-Lapse (voller Lapse, lokal)')
    dd.bar(xs + 0.18, [r[4] / ref for r in rows], width=0.34, color=S2, label='+ räumliche Metrik (Exp. 1c)')
    dd.plot(xs - 0.18, [r[3] / ref for r in rows], 'k_', ms=14, mew=2, label='Halbklassik')
    dd.plot(xs + 0.18, [r[5] / ref for r in rows], 'k_', ms=14, mew=2)
    dd.axhline(1, color=MUTED, ls='--', lw=1); dd.axhline(2, color=MUTED, ls=':', lw=1)
    dd.set_xticks(xs); dd.set_xticklabels([r[0].replace(', ', '\n') for r in rows], fontsize=8.5)
    dd.set_ylabel("a_x / (−g N₀² c′²)")
    title(dd, '(d) Querbeschleunigung, streng lokale unitäre Regeln\n(x-Kette, N = N₀(1 + g x), A = 1 − g x)')
    dd.legend(loc='upper left', fontsize=8)

    # (e) ratio vs GR
    e = ax[1, 1]
    def pce(name):                       # Dirac kinematics p c'/E, m = 2 mu = 0.3 for the massive cases
        if 'massless' in name:
            return 1.0
        k = 0.0 if 'rest' in name else float(name.split('=')[1])
        return CP * k / np.hypot(CP * k, 0.3)
    v = np.array([pce(r[0]) for r in rows])
    rs = np.array([r[6] for r in rows], float)
    rc = np.array([r[7] for r in rows], float)
    vv = np.linspace(0, 1.05, 50)
    e.plot(vv, 1 + vv ** 2, color=MUTED, ls='--', lw=1.2, label='ART: 1 + (p c′/E)²')
    e.plot(v, rs, 'o', color=S2, ms=8, label='Simulation')
    e.plot(v, rc, 'x', color=INK, ms=8, mew=2, label='Halbklassik')
    e.set_xlabel("p c′ / E  (Dirac-Geschwindigkeit / c′)"); e.set_ylabel('a(1c) / a(nur Lapse)')
    title(e, '(e) Faktor der räumlichen Metrik: 1 für ruhende Masse,\n2 für Licht (volle ART-Lichtablenkung)')
    e.legend(loc='upper left', fontsize=8)

    # (f) pumping
    f = ax[1, 2]
    for key, col in (('C_B0.05_E-2.078', S1), ('C_B0.05_E+0.693', S2),
                     ('C_B0.035_E-2.078', S1), ('C_B0.035_E+0.693', S2)):
        if key not in d.files:
            continue
        tr = d[key]
        B = key.split('_')[1][1:]
        node = 'unterer Knoten (χ = +1)' if '-2.078' in key else 'innerer Knoten (χ = −1)'
        f.plot(tr[:, 0], tr[:, 1], color=col, ls='-' if B == '0.05' else '--', label=f'{node}, B = {B}')
    kz = np.linspace(-0.25, 0.25, 3)
    f.plot(kz, -CP * kz, color=MUTED, lw=1, ls=':'); f.plot(kz, CP * kz, color=MUTED, lw=1, ls=':')
    f.set_xlabel('k_z(t)  (E ∥ B: k_z läuft linear in der Zeit)'); f.set_ylabel('⟨E⟩ − E_Knoten')
    title(f, '(f) Pumpen in der Zeit, Helizität +1: die beiden Knoten\nwerden in entgegengesetzte Richtung gepumpt (∓c′)')
    f.legend(loc='lower left', fontsize=7.5)

    fig.suptitle('S4, vierte Runde: lokaler voller Lapse über einen Warte-Puffer, räumliche Metrik (Exp. 1c), '
                 'Vakuum-Gegenterm, chiraler Zweig am inneren Knoten', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open4.png'), dpi=120)
    print('saved figures/phase_bit_s4_open4.png')


if __name__ == '__main__':
    main()
