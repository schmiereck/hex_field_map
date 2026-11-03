#!/usr/bin/env python3
"""Figure for the gravitational atom.  Reads data/phase_bit_s4_open14.npz."""

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
    d = np.load(data_path('phase_bit_s4_open14.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) 1D lattice atom vs effective Hamiltonian
    a = ax[0, 0]
    for kind, col, mk in (('W', S2, 's'), ('H2', S1, 'o')):
        lat, ed, el = d[f'A_{kind}']
        lat = np.sort(lat[np.isfinite(lat)])[:3]
        el = np.sort(el[el < 0])[:3]
        a.plot(-el * 1e3, -lat * 1e3, mk, color=col, ms=10, label=f'{"nur Warten" if kind == "W" else "H2"}: Niveaus 0, 1, 2')
    lim = [0, 5.6]
    a.plot(lim, lim, color=MUTED, lw=1, ls='--')
    a.set_xlabel('Bindungsenergie, effektiver Hamiltonian (10⁻³)')
    a.set_ylabel('Bindungsenergie, Gitter (Zeitsignal, 10⁻³)')
    title(a, '(a) 1D-Gitteratom: die gebundenen Zustände des Walkers folgen dem\neffektiven Hamiltonian (0.2–3 %), auch die Verschiebung H2 − W')
    a.legend(loc='upper left', fontsize=8)

    # (b) 3D level scheme (GM = 0.1), H2 and W, binding relative to Bohr
    b = ax[0, 1]
    GM = 0.1
    L = d[f'B_levels_{GM}']           # rows: variant (H2, W, linear, Newton) x l (0..3); cols: radial index
    names = ['H2', 'W', 'linear', 'Newton']
    for vi, (vn, col) in enumerate(zip(names[:2] + ['Newton'], (S1, S2, MUTED))):
        row0 = names.index(vn) * 4
        for n in (2, 3, 4):
            bohr = -GM ** 2 / (2 * n ** 2)
            for l in range(min(n, 4)):
                E = L[row0 + l, n - l - 1]
                x0 = n + vi * 0.27 - 0.27
                b.plot([x0 - 0.1, x0 + 0.1], [(E - bohr) / abs(bohr) * 1e3] * 2, color=col, lw=2.5)
                if vn == 'H2':
                    b.text(x0 + 0.12, (E - bohr) / abs(bohr) * 1e3, f'l={l}', fontsize=7, color=col, va='center')
        b.plot([], [], color=col, lw=2.5, label={'H2': 'H2 (γ = β = 1)', 'W': 'nur Warten (γ = 0)', 'Newton': 'Newton (nur Abrundung)'}[vn])
    b.set_xticks([2, 3, 4]); b.set_xticklabels(['n = 2', 'n = 3', 'n = 4'])
    b.set_ylabel('(E − Bohr) / |Bohr|  (10⁻³)')
    title(b, '(b) 3D-Gravitationsatom (GM = 0.1): die Metrik hebt die l-Entartung auf,\nhöheres l ist schwächer gebunden (Quanten-Periheldrehung)')
    b.legend(loc='lower right', fontsize=8)

    # (c) ratios of the relativistic l-splitting between variants
    c = ax[1, 0]
    r = d['B_rows']           # GM, n, l, Newton, H2, W, linear
    for GMv, mk in ((0.1, 'o'), (0.05, 's')):
        s = r[:, 0] == GMv
        c.plot(r[s, 2] + 0.5 + (0.03 if GMv == 0.05 else -0.03), r[s, 5] / r[s, 4], mk, color=S2, ms=8,
               label=f'nur Warten / H2, GM = {GMv}')
        c.plot(r[s, 2] + 0.5 + (0.03 if GMv == 0.05 else -0.03), r[s, 6] / r[s, 4], mk, color=S3, ms=8,
               label=f'linearer Lapse / H2, GM = {GMv}')
    c.axhline(1 / 3, color=S2, lw=1, ls='--')
    c.axhline(7 / 6, color=S3, lw=1, ls='--')
    c.text(2.85, 1 / 3 + 0.03, 'klassisch 1/3', color=S2, fontsize=8, ha='right')
    c.text(2.85, 7 / 6 + 0.03, 'klassisch 7/6', color=S3, fontsize=8, ha='right')
    c.set_xticks([0.5, 1.5, 2.5]); c.set_xticklabels(['l = 0 → 1', '1 → 2', '2 → 3'])
    c.set_ylabel('relativistische l-Aufspaltung, relativ zu H2')
    title(c, '(c) Verhältnisse der Metrik-Varianten: gehen mit wachsendem l gegen\ndie klassischen Periheldrehungs-Faktoren')
    c.legend(loc='center left', fontsize=8)

    # (d) splitting vs classical precession frequency
    dd = ax[1, 1]
    for GMv, col in ((0.1, S1), (0.05, S4)):
        s = r[:, 0] == GMv
        wp = 3 * GMv ** 4 / ((r[s, 2] + 1.0) ** 2 * r[s, 1] ** 3)
        dd.loglog(wp, r[s, 4], 'o', color=col, ms=9, label=f'H2, GM = {GMv}')
    xx = np.array([1e-8, 1e-4])
    dd.loglog(xx, xx, color=MUTED, lw=1, ls='--', label='= klassische Präzessionsfrequenz')
    dd.set_xlabel('klassisch: ω_p = 3 GM⁴ / (L² n³), L = l + 1')
    dd.set_ylabel('Quanten: E(n, l+1) − E(n, l) (relativistischer Teil)')
    title(dd, '(d) Korrespondenz: die Aufspaltung ist die Präzessionsfrequenz der Bahn\n(Faktor 1.1–1.25 bei n ≤ 4; skaliert wie GM⁴)')
    dd.legend(loc='upper left', fontsize=8)

    fig.suptitle('Das Gravitationsatom: die Quantenversion der Periheldrehung', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open14.png'), dpi=120)
    print('saved figures/phase_bit_s4_open14.png')


if __name__ == '__main__':
    main()
