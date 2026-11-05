#!/usr/bin/env python3
"""Figure for gravity in 3D (1/r field, light deflection).  Reads data/phase_bit_s4_open12.npz."""

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
    d = np.load(data_path('phase_bit_s4_open12.npz'), allow_pickle=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) static 1/r
    a = ax[0, 0]
    st = d['A_static']
    rr = np.linspace(3, 21, 100)
    a.plot(rr, 1 / (4 * np.pi * rr), color=MUTED, lw=1.2, ls='--', label='κQ / (4πr)')
    a.plot(st[:, 0], -st[:, 1], 'o', color=S1, ms=8, label='3D field equation (lattice, periodic box)')
    a.set_xlabel('distance r'); a.set_ylabel('−Φ(r)')
    title(a, '(a) static: the local field equation gives the 1/r potential in 3D\n(deviation at large r: image sources of the periodic box)')
    a.legend(loc='upper right', fontsize=8)

    # (b) causal: gradient at three radii and the centre value
    b = ax[0, 1]
    w = d['A_wave']
    for j, (p, col) in enumerate(((6, S1), (10, S3), (16, S2))):
        b.plot(w[:, 0], w[:, 1 + j] * 4 * np.pi * p ** 2, color=col, label=f'r = {p}: 4πr² ∂Φ/∂r')
        b.axvline(p / CP, color=col, lw=0.8, ls=':')
    b.axhline(1, color=MUTED, lw=0.8, ls='--')
    b.set_xlabel('time t (source switched on at t = 0)'); b.set_ylabel('field strength / static 1/r² value')
    title(b, '(b) causal: the front of the switched-on mass passes at c_g (dotted: r/c_g)\nas a pulse, then the static value stays; nothing grows')
    b.legend(loc='lower right', fontsize=8)

    # (c) deflection vs time, b = 12
    c = ax[1, 0]
    for kind, col, ls in (('W', S2, '-'), ('H2', S1, '-')):
        for bi, lw in ((12, 2.0), (20, 1.2)):
            key = f'B_{bi}_{kind}'
            if key not in d:
                continue
            ts, dky, pl, xc, yc = d[key]
            lab = f'{"wait only (γ = 0)" if kind == "W" else "H2 (γ = 1)"}, b = {bi}'
            c.plot(xc, dky, color=col, lw=lw, label=lab)
            c.plot(xc, pl, color=INK, lw=0.9, ls=':')
    c.plot([], [], color=INK, lw=0.9, ls=':', label='local force n_s N^n_s c′k ∂U/∂y (from the density)')
    c.axvline(0, color=MUTED, lw=0.8)
    c.set_xlabel('packet centre x (mass at x = 0)'); c.set_ylabel('Δ⟨k_y⟩ (towards the mass < 0)')
    title(c, '(c) light passing a point mass: with the spatial metric\n(H2) twice the deflection of waiting alone')
    c.legend(loc='lower left', fontsize=8)

    # (d) summary from the stored runs: deflection over the travelled path, sim vs predictions
    dd = ax[1, 1]
    k0, sig, GM, rc, nst = d['B_par']
    LA = 0.6123724356957945
    xpos = {'W': -0.17, 'H2': 0.17}
    for j, bi in enumerate((12, 20)):
        bb = bi * LA
        ray = lambda xx: -GM * bb * xx / ((bb ** 2 + rc ** 2) * np.sqrt(xx ** 2 + bb ** 2 + rc ** 2))
        for kind, col in (('W', S2), ('H2', S1)):
            ts, dky, pl, xc, yc = d[f'B_{bi}_{kind}']
            ns = 1 if kind == 'W' else 2
            pr = ns * k0 * (ray(xc[-1]) - ray(xc[0]))
            x_ = j + xpos[kind]
            dd.bar(x_, -dky[-1] / k0, width=0.3, color=col, label=('wait only (γ = 0)' if kind == 'W' else 'H2 (γ = 1)') if j == 0 else None)
            dd.plot([x_ - 0.15, x_ + 0.15], [-pl[-1] / k0] * 2, color=INK, lw=2, label='local force (from the density)' if (j == 0 and kind == 'W') else None)
            dd.plot([x_ - 0.15, x_ + 0.15], [-pr / k0] * 2, color=MUTED, lw=1.5, ls='--', label='thin straight ray' if (j == 0 and kind == 'W') else None)
        r = d[f'B_{bi}_H2'][1][-1] / d[f'B_{bi}_W'][1][-1]
        dd.text(j, -d[f'B_{bi}_H2'][1][-1] / k0 * 1.04 + 0.004, f'H2 / W = {r:.3f}', ha='center', color=INK, fontsize=9)
    dd.set_xticks([0, 1]); dd.set_xticklabels([f'b = {12*LA:.1f}', f'b = {20*LA:.1f}'])
    dd.set_ylabel('deflection angle Δk_y / k₀ (rad) over the path travelled')
    title(dd, '(d) H2 deflects light twice as strongly as waiting alone (GR: 4GM/b vs\n2GM/b); at b ≈ σ the packet is wider than a thin ray')
    dd.legend(loc='upper right', fontsize=8)

    fig.suptitle('Gravity in 3D: 1/r field from the local field equation and light deflection by a point mass (2O*, ε = 0.3)',
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path('phase_bit_s4_open12.png'), dpi=120)
    print('saved figures/phase_bit_s4_open12.png')


if __name__ == '__main__':
    main()
