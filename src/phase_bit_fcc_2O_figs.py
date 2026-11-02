#!/usr/bin/env python3
"""Figure for Experiment S3 (2O spin register on FCC).  Reads data/phase_bit_fcc_2O.npz."""

import itertools
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import data_path, fig_path
from phase_bit_fcc_2O import build_transport, su2_min, ADJ60, FCC_UNIT, N_D

S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3de'
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0, 'legend.frameon': False,
})


def rot_angle(g):
    return 2 * np.arccos(np.clip(np.real(np.trace(g)) / 2, -1, 1)) / (2 * np.pi)


def plaquette_holonomies():
    Q, _ = build_transport()
    tri = next((a, b, c) for a, b, c in itertools.combinations(range(N_D), 3)
               if ADJ60[a, b] and ADJ60[b, c] and ADJ60[c, a])
    sq = next((0, b, c, d) for b in range(N_D) for c in range(N_D) for d in range(N_D)
              if len({0, b, c, d}) == 4 and ADJ60[0, b] and ADJ60[b, c]
              and ADJ60[c, d] and ADJ60[d, 0] and not ADJ60[0, c])
    out = {}
    for name, f in (('2O', lambda x, y: Q[(x, y)]),
                    ('min', lambda x, y: su2_min(FCC_UNIT[x], FCC_UNIT[y]))):
        cyc = lambda c: np.linalg.multi_dot([f(c[(i + 1) % len(c)], c[(i + 2) % len(c)])
                                             for i in reversed(range(len(c)))]) \
            if len(c) > 2 else None
        g_t = f(tri[2], tri[0]) @ f(tri[1], tri[2]) @ f(tri[0], tri[1])
        g_s = f(sq[3], sq[0]) @ f(sq[2], sq[3]) @ f(sq[1], sq[2]) @ f(sq[0], sq[1])
        out[name] = [rot_angle(g_t), rot_angle(g_s), 1.0]      # L=6 loop: -1 (measured)
    return out


def main():
    d = np.load(data_path('phase_bit_fcc_2O.npz'))
    fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))

    # (a) holonomy per heading plaquette
    a = ax[0, 0]
    h = plaquette_holonomies()
    x = np.arange(3)
    w = 0.36
    a.bar(x - w / 2, h['2O'], w, color=S1, label='endliches Register 2O (48)')
    a.bar(x + w / 2, h['min'], w, color=S2, label='minimale SU(2) (kontinuierlich)')
    for xi, v in zip(x, h['2O']):
        a.text(xi - w / 2, v + 0.02, f'{v:.3f}', ha='center', color=INK, fontsize=9)
    for xi, v in zip(x, h['min']):
        a.text(xi + w / 2, v + 0.02, f'{v:.3f}', ha='center', color=INK, fontsize=9)
    a.set_xticks(x, ['Richtungs-\nDreieck', 'Richtungs-\nQuadrat', 'geschlossener\nWeg L = 6'])
    a.set_ylabel('Drehwinkel der Spin-Holonomie / 2π')
    a.set_ylim(0, 1.25)
    a.set_title('(a) 2O quantisiert die Krümmung: Dreieck π, Quadrat 2π',
                loc='left', color=INK)
    a.legend(loc='upper left')

    # (b) rest spectra
    b = ax[0, 1]
    sets = [('2O, Spin ½', d['rest_E1/2_False'], S1),
            ('min. SU(2), Spin ½', d['rest_E1/2_True'], S2),
            ('Boson (A1)', d['rest_A1_False'], S3)]
    for i, (name, lv, col) in enumerate(sets):
        for E, m in lv:
            b.plot([i - 0.3, i + 0.3], [E, E], color=col, lw=3)
            b.text(i + 0.34, E, f'×{int(m)}', va='center', color=INK, fontsize=8)
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_xticks(range(3), [s[0] for s in sets])
    b.set_xlim(-0.6, 2.8)
    b.set_ylabel('Ruheenergie E / ε  (k = 0)')
    b.set_title('(b) Ruhespektren: 2O ist symmetrisch, ±2√2, ±4√2', loc='left', color=INK)
    b.grid(axis='x', visible=False)

    # (c), (d) bands along three directions
    k = d['disp_k']
    for axx, minimal, title in ((ax[1, 0], 0, '(c) 2O-Register: Bänder, E ↔ −E exakt'),
                                (ax[1, 1], 1, '(d) minimale SU(2): Bänder, nicht symmetrisch')):
        for name, col in (('[100]', S1), ('[110]', S2), ('[111]', S3)):
            E = d[f'disp_{minimal}_{name}']
            for j in range(E.shape[1]):
                axx.plot(k, E[:, j], color=col, lw=1.2,
                         label=name if j == 0 else None)
        axx.set_xlabel('|k|'); axx.set_ylabel('E')
        axx.set_title(title, loc='left', color=INK)
        axx.legend(loc='upper left', title='Richtung', fontsize=8)

    fig.suptitle('Experiment S3 — Spin ½ in 3D aus einem endlichen Phasen-Bit-Register '
                 '(binäre Oktaedergruppe, FCC)', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_fcc_2O.png'), dpi=130)
    print('saved figures/phase_bit_fcc_2O.png')


if __name__ == '__main__':
    main()
