#!/usr/bin/env python3
"""Figure for Michelson-Morley in the lattice, Newton/perihelion and Lense-Thirring.  Reads data/phase_bit_s4_open13.npz."""

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
CP = np.sqrt(3) / 3
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0, 'legend.frameon': False,
})


def title(ax, s):
    ax.set_title(s, loc='left', color=INK, fontsize=10)


def main():
    d = np.load(data_path('phase_bit_s4_open13.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    # (a) Michelson-Morley: direction spread and energy dependence of the light speed
    a = ax[0, 0]
    ks, spread = d['A_spread']
    mean_dev = []
    n = None
    for k in ks:
        v = d[f'A_v_{k}']
        if n is None:
            i = np.arange(len(v)) + 0.5
            z = 1 - 2 * i / len(v)
            az = np.pi * (1 + np.sqrt(5)) * i
            n = np.stack([np.sqrt(1 - z ** 2) * np.cos(az), np.sqrt(1 - z ** 2) * np.sin(az), z], 1)
        mean_dev.append(1 - np.mean(np.sum(v * n, 1)) / CP)
    a.loglog(ks, spread, 'o-', color=S1, ms=8, label='Richtungsabhängigkeit (Michelson–Morley): (max − min)/Mittel')
    a.loglog(ks, mean_dev, 's-', color=S2, ms=7, label='Energieabhängigkeit: 1 − ⟨v⟩/c′ (Laufzeit, ohne Drehen)')
    kk = np.array([0.02, 0.1])
    a.loglog(kk, spread[0] * (kk / 0.02) ** 2, color=INK, lw=1, ls=':', label='∝ k²')
    a.set_xlabel('Wellenzahl k des Lichts'); a.set_ylabel('relative Änderung der Lichtgeschwindigkeit')
    title(a, '(a) Was ein Beobachter im Gitter messen kann: Licht zweier Energien\nim gedrehten Aufbau; beides ∝ (ka)² für langwelliges Licht')
    a.legend(loc='upper left', fontsize=8)

    # (b) Newton: force on a packet at rest in the 3D walker
    b = ax[0, 1]
    r = d['B1_rows']
    rr = np.linspace(5, 28, 100)
    b.loglog(rr, 0.3 * 0.25 * rr / (rr ** 2 + 4) ** 1.5, color=MUTED, lw=1, ls='--', label='Newton m GM r/(r² + r_c²)^{3/2}')
    b.loglog(r[:, 0], -r[:, 1], 's', color=S2, ms=9, label='3D-Walker, nur Warten')
    b.loglog(r[:, 0], -r[:, 2], '_', color=INK, ms=16, mew=2, label='lokale Kraft der Metrik (aus Dichte und k-Breite)')
    b.loglog(r[:, 0], -r[:, 3], 'o', color=S1, ms=9, label='3D-Walker, H2')
    b.loglog(r[:, 0], -r[:, 4], '_', color=INK, ms=16, mew=2)
    b.set_xlabel('Abstand r von der Punktmasse'); b.set_ylabel('Kraft |F| = |d⟨k_x⟩/dt|')
    title(b, '(b) Kraft auf ein ruhendes massives Paket (μ = 0.15): Walker = lokale\nKraft auf ≤ 3 %; über Newton, weil die Impulsbreite mitgravitiert')
    b.legend(loc='lower left', fontsize=8)

    # (c) perihelion advance
    c = ax[0, 2]
    B2 = d['B2_rows']
    names = ['H2\nγ = β = 1', 'nur Warten\nγ = 0', 'linearer Lapse\nβ = 1/2']
    for j in range(3):
        for i, (col, lab) in enumerate(((S1, 'a = 400'), (S3, 'a = 1600'))):
            row = B2[2 * j + i]
            c.bar(j + (i - 0.5) * 0.35, row[2] / row[4], width=0.33, color=col, label=lab if j == 0 else None)
        c.plot([j - 0.4, j + 0.4], [B2[2 * j][3] / B2[2 * j][4]] * 2, color=INK, lw=2, label='PPN (2 + 2γ − β)/3' if j == 0 else None)
    c.set_xticks(range(3)); c.set_xticklabels(names)
    c.set_ylabel('Periheldrehung / ART-Wert 6πGM/p')
    title(c, '(c) Klassische Bahnen der Gitter-Metrik: H2 gibt\ndie volle ART-Periheldrehung (Rest ∝ 1/a)')
    c.legend(loc='upper right', fontsize=8)

    # (d) gravitomagnetic dipole of a rotating ball
    dd = ax[1, 0]
    f = d['C1_field']
    rr = np.linspace(7, 16, 50)
    J = f[0, 2] * f[0, 0] ** 2
    dd.plot(rr, J / rr ** 2 * 1e4, color=MUTED, lw=1.2, ls='--', label="ART-Dipol 2c′²(κ/4π) J/r² (aus dem Faktor 4)")
    dd.plot(f[:, 0], f[:, 1] * 1e4, 'o', color=S1, ms=9, label='3D-Feldgleichung für b (periodische Box)')
    lt = d['C1_orbit']
    dd.set_xlabel('Abstand r in der Äquatorebene'); dd.set_ylabel('Mitnahmefeld b (10⁻⁴)')
    title(dd, f'(d) Rotierende Kugel: Mitnahmefeld = ART-Dipol; klassische Bahn:\nKnotendrehung / Lense–Thirring = {lt[0]/lt[1]:.3f}')
    dd.legend(loc='upper right', fontsize=8)

    # (e) frame dragging in the walker: orbit angle and spin angle
    e = ax[1, 1]
    Om = d['C2_rates'][1, 0]
    for key, col, lab in (('C2_0', S2, 'nur skalare Mitnahme'), ('C2_1', S1, 'Mitnahme + innere Drehung ½∇×b·J')):
        ts, ang, sp = d[key]
        e.plot(ts, ang - ang[0], color=MUTED, lw=1.2)
        e.plot(ts, sp - sp[0], color=col, label=f'Spin-Winkel: {lab}')
    e.plot([], [], color=MUTED, lw=1.2, label='Bahnwinkel des Pakets (beide Läufe gleich)')
    e.plot(ts, Om * ts, color=INK, lw=1, ls=':', label='Ω t (mitgeführtes Bezugssystem)')
    e.set_xlabel('Zeit t'); e.set_ylabel('Winkel (rad)')
    title(e, '(e) 3D-Walker im starr rotierenden Mitnahmefeld: das Paket wird\nmitgeführt; der Spin nur mit der Kreisel-Regel')
    e.legend(loc='upper left', fontsize=8)

    # (f) spin rotation rates relative to Omega
    g = ax[1, 2]
    rates = d['C2_rates'][0]
    vals = [rates[1] / Om, -1 / 3, 1.0, rates[3] / Om]
    labs = ['skalare Mitnahme\n(gemessen)', 'Spin-Index drehen:\nTeilchen (g = −1/3)', 'Spin-Index drehen:\nAntiteilchen (g = 1)',
            'innere Drehung J\n(gemessen)']
    cols = [S2, MUTED, MUTED, S1]
    g.bar(range(4), vals, color=cols, width=0.6)
    g.axhline(1, color=INK, lw=1, ls=':')
    g.text(3.35, 1.03, 'ART', color=INK, fontsize=9)
    g.set_xticks(range(4)); g.set_xticklabels(labs, fontsize=8)
    g.set_ylabel('Spin-Drehrate / Ω')
    title(g, '(f) Kreisel: nur der Gitter-eigene Drehgenerator J = (2i/π) log R₉₀\ngibt für Teilchen und Antiteilchen g = 1')

    fig.suptitle('Beobachter im Gitter (Michelson–Morley), Newton und Periheldrehung, Lense–Thirring (2O*, ε = 0.3)',
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open13.png'), dpi=110)
    print('saved figures/phase_bit_s4_open13.png')


if __name__ == '__main__':
    main()
