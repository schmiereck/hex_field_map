#!/usr/bin/env python3
"""Lead figure for Paper 3: light passing a point mass on the phase-bit lattice.

Same setup as phase_bit_s4_open12.py part B (box 112x88x80, 2O*, eps 0.3, massless packet k0 0.6, sigma 5,
U = GM/sqrt(r^2 + rc^2), GM 0.25, rc 2, N = exp(-U), impact parameter b = 12 a), one Kramers member, but with
z-projected density snapshots saved for the picture.  The deflection numbers themselves are those of open12 B.

    python paper3_lead_fig.py run     # ~10 min, ~2 GB: writes data/paper3_lead.npz
    python paper3_lead_fig.py         # figure only: figures/paper3_lead.png
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import data_path, fig_path

SHAPE = (112, 88, 80)
K0, SIG, GM, RC, B_IDX, NST, EVERY = 0.6, 5.0, 0.25, 2.0, 12, 128, 16


def run():
    import phase_bit_s4_open12 as o12
    LAT_A = o12.LAT_A
    xs = [(np.arange(n) - n // 2) * LAT_A for n in SHAPE]
    X, Y, Z = np.meshgrid(*xs, indexing='ij')
    ymass = -14 * LAT_A
    R = np.sqrt(X ** 2 + (Y - ymass) ** 2 + Z ** 2)
    Nf = np.exp(-GM / np.sqrt(R ** 2 + RC ** 2))
    x0 = np.array([-30 * LAT_A, ymass + B_IDX * LAT_A, 0.0])
    out = {'par': np.array([K0, SIG, GM, RC, B_IDX, LAT_A, ymass]), 'N_z0': Nf[:, :, SHAPE[2] // 2]}
    t0 = time.time()
    for kind, ns in (('W', 1), ('H2', 2)):
        wk = o12.Walker3D(SHAPE, 'W' if kind == 'W' else 'WH', Nf)
        nreg = 2 if kind == 'W' else 4
        psi = o12.packet3d(SHAPE, (K0, 0.0, 0.0), SIG, x0, 0)
        st = [psi] + [np.zeros_like(psi) for _ in range(nreg - 1)]
        i0 = np.argmin(np.abs(X.ravel() - x0[0]) + np.abs(Y.ravel() - x0[1]) + np.abs(Z.ravel()))
        ph = np.exp(1j * o12.CP * K0 * Nf.ravel()[i0] ** ns * o12.DT_HALF)
        for _ in range(3):                                    # same filter against the pi branches as open12
            st2 = wk.tick(st)
            st = [(a + ph * c) / 2 for a, c in zip(st, st2)]
            del st2
        nrm = np.sqrt(sum(np.vdot(a, a).real for a in st))
        st = [a / nrm for a in st]
        snaps, track = [], []
        for it in range(NST + 1):
            if it % 4 == 0:
                rho = sum((np.abs(a) ** 2).sum(1) for a in st)
                track.append((it * o12.DT_HALF, rho @ X.ravel() / rho.sum(), rho @ Y.ravel() / rho.sum()))
                if it % EVERY == 0:
                    snaps.append(rho.reshape(SHAPE).sum(2).astype(np.float32))
            if it < NST:
                st = wk.tick(st)
        out[f'{kind}_snaps'] = np.array(snaps)
        out[f'{kind}_track'] = np.array(track)
        print(f"   {kind}: y {track[0][2]:+.3f} -> {track[-1][2]:+.3f}  [{time.time()-t0:.0f}s]")
        del st, wk
    np.savez_compressed(data_path('paper3_lead.npz'), **out)
    print('saved data/paper3_lead.npz')


S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED = '#1f1f1e', '#6b6a63'


def figure():
    d = np.load(data_path('paper3_lead.npz'))
    k0, sig, gm, rc, bidx, a, ymass = d['par']
    N = d['N_z0']
    nx, ny = N.shape
    xs = (np.arange(nx) - nx // 2) * a
    ys = (np.arange(ny) - ny // 2) * a
    xlim, ylim = (-27.0, 23.0), (ymass - 9.5, ymass + 15.5)

    fig = plt.figure(figsize=(12, 6.2))
    ax = fig.add_axes([0.02, 0.04, 0.70, 0.88])
    ax.set_facecolor('white')
    # lattice nodes of the z = 0 plane (FCC: i + j even), shaded by the busy fraction 1 - N
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    sel = ((I + J - nx // 2 - ny // 2) % 2 == 0) & (xs[I] > xlim[0] - 1) & (xs[I] < xlim[1] + 1) \
        & (ys[J] > ylim[0] - 1) & (ys[J] < ylim[1] + 1)
    busy = 1 - N[sel]
    busy_cmap = LinearSegmentedColormap.from_list('busy', ['#ecebe6', '#8a5a2b', '#3b2412'])
    sc = ax.scatter(xs[I[sel]], ys[J[sel]], s=7 + 60 * busy / busy.max(), c=busy, cmap=busy_cmap,
                    vmin=0, vmax=busy.max(), linewidths=0, zorder=1)
    # the light packet (H2 run): z-projected density at successive times
    snaps = d['H2_snaps']
    trH, trW = d['H2_track'], d['W_track']
    pick = [0, len(snaps) // 2, len(snaps) - 1]
    Xg, Yg = np.meshgrid(xs, ys, indexing='ij')
    soft = sum(snaps[k] / snaps[k].max() for k in pick)
    pk = LinearSegmentedColormap.from_list('pk', [(1, 1, 1, 0), (0.16, 0.47, 0.84, 0.30)])
    ax.imshow(np.clip(soft, 0, 1).T, origin='lower', extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=pk,
              interpolation='bicubic', zorder=0)
    for j, k in enumerate(pick):
        s = snaps[k] / snaps[k].max()
        ax.contour(Xg, Yg, s, levels=[0.25, 0.5, 0.75], colors=[S1], linewidths=[0.8, 1.1, 1.4],
                   alpha=0.45 + 0.55 * j / 2, zorder=2)
        t = trH[min(k * 4, len(trH) - 1), 0]
        ax.text(trH[min(k * 4, len(trH) - 1), 1], ylim[1] - 1.2, f't = {t:.0f}', ha='center', fontsize=9, color=S1)
    ax.plot(trH[:, 1], trH[:, 2], color=S1, lw=2, zorder=3)
    ax.plot([trH[0, 1], xlim[1]], [trH[0, 2]] * 2, color=MUTED, lw=1, ls='--', zorder=3)
    ax.plot(0, ymass, marker='o', ms=12, color=INK, zorder=4)
    ax.text(0, ymass - 1.6, 'mass', ha='center', va='top', fontsize=10, color=INK,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=5)
    ax.text(xlim[0] + 0.5, ylim[0] + 0.6, 'z-projected density of a massless packet in the full 3D walker (budget rule);\n'
            'line: packet centre, dashed: straight line', fontsize=8.5, color=MUTED, va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', pad=2), zorder=5)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal'); ax.axis('off')
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01, shrink=0.55, location='left')
    cb.set_label('busy fraction of a node, 1 − N', color=INK, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_title('Every node is a processor; near a mass it is busy part of the time, and a light packet\n'
                 'that must wait for free nodes is bent towards the mass', loc='left', fontsize=11, color=INK)

    # right: the measured deflection, y exaggerated
    b = fig.add_axes([0.78, 0.17, 0.20, 0.60])
    for tr, col, lab in ((trW, S4, 'wait buffer only\n(Einstein 1911, γ = 0)'), (trH, S1, 'budget rule\n(GR, γ = 1)')):
        b.plot(tr[:, 1], tr[:, 2] - tr[0, 2], color=col, lw=2.2, label=lab)
    b.axhline(0, color=MUTED, lw=1, ls='--')
    b.set_xlabel('x', color=INK)
    b.set_ylabel('transverse shift of the packet centre', color=INK, fontsize=9)
    for s in ('top', 'right'):
        b.spines[s].set_visible(False)
    b.tick_params(labelsize=8, colors=MUTED)
    b.legend(loc='lower left', fontsize=8.5, frameon=False)
    b.set_title('measured: twice the bending\n(H2 / W = 2.07, y stretched)', loc='left', fontsize=9.5, color=INK)
    fig.savefig(fig_path('paper3_lead.png'), dpi=150, bbox_inches='tight')
    print('saved figures/paper3_lead.png')


if __name__ == '__main__':
    if 'run' in sys.argv[1:]:
        run()
    figure()
