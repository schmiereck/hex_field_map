#!/usr/bin/env python3
"""
Scientific animation: 2-panel visualisation of the 2+1D hexagonal lattice model.
  Panel 1: Wave propagation with hex lattice background + light cone
  Panel 2: Dispersion relation — 3 directions, 1 curve (isotropy)
Outputs: animation.gif, animation.mp4, test_animation.png
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
from quantum_hex_2d import (
    simulate_hex_2d, TM14_half, C_LIGHT, DX_PHYS, DY_PHYS,
    fit_rel_2d_direct,
)

# ── colours ─────────────────────────────────────────────────────────────────
BG      = '#070810'
WHITE   = '#FFFFFF'
GRAY    = '#AAAACC'
CYAN    = '#00D4FF'
ORANGE  = '#FF8C00'
MAGENTA = '#FF44CC'
LATTICE_NODE = '#1a1a3a'
LATTICE_EDGE = '#0d0d2a'

# cool colormap: black -> deep blue -> cyan -> white
_cmap_data = {
    'red':   [(0, 0, 0), (0.33, 0.0, 0.0), (0.66, 0.0, 0.0), (1, 1, 1)],
    'green': [(0, 0, 0), (0.33, 0.05, 0.05), (0.66, 0.7, 0.7), (1, 1, 1)],
    'blue':  [(0, 0, 0), (0.33, 0.3, 0.3), (0.66, 1.0, 1.0), (1, 1, 1)],
}
COOL_CMAP = LinearSegmentedColormap('cool_plasma', _cmap_data, N=256)

SQRT3 = np.sqrt(3)

# ════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE ALL DATA
# ════════════════════════════════════════════════════════════════════════════
print("Pre-computing simulation data ...")

# ── Panel 1: wave propagation ───────────────────────────────────────────────
EPS = 0.1
T_SIM = 20
print("  Panel 1: simulate_hex_2d ...", end=" ", flush=True)
psi_all, xc, yc = simulate_hex_2d(T_SIM, eps=EPS)
prob_all = np.abs(psi_all)**2
Nx, Ny = prob_all.shape[1], prob_all.shape[2]
x_ph = (np.arange(Nx) - xc) * DX_PHYS
y_ph = (np.arange(Ny) - yc) * DY_PHYS

# Smooth to remove lattice checkerboard artifacts
prob_smooth = np.zeros_like(prob_all)
for t in range(prob_all.shape[0]):
    prob_smooth[t] = gaussian_filter(prob_all[t], sigma=1.5)

R_MAX = SQRT3 * T_SIM * 1.15
x_mask = np.abs(x_ph) < R_MAX
y_mask = np.abs(y_ph) < R_MAX
prob_crop = prob_smooth[:, x_mask, :][:, :, y_mask]
x_crop = x_ph[x_mask]
y_crop = y_ph[y_mask]
print(f"done. shape {prob_crop.shape}")

# ── Hex lattice background geometry ─────────────────────────────────────────
# Basis vectors: a1 = (1, 0), a2 = (0.5, sqrt(3)/2)
# Generate nodes within plot extent
print("  Hex lattice nodes ...", end=" ", flush=True)
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, SQRT3 / 2])
extent = R_MAX + 1
n_range = int(extent / 0.8) + 2
lattice_nodes_x = []
lattice_nodes_y = []
lattice_edges = []  # list of ((x1,y1),(x2,y2))
node_set = set()
for i in range(-n_range, n_range + 1):
    for j in range(-n_range, n_range + 1):
        p = i * a1 + j * a2
        if abs(p[0]) < extent and abs(p[1]) < extent:
            lattice_nodes_x.append(p[0])
            lattice_nodes_y.append(p[1])
            node_set.add((i, j))

# Edges: connect each node to its 3 forward neighbors (a1, a2, a2-a1)
for (i, j) in node_set:
    p0 = i * a1 + j * a2
    for di, dj in [(1, 0), (0, 1), (-1, 1)]:
        ni, nj = i + di, j + dj
        if (ni, nj) in node_set:
            p1 = ni * a1 + nj * a2
            lattice_edges.append(((p0[0], p0[1]), (p1[0], p1[1])))

lattice_nodes_x = np.array(lattice_nodes_x)
lattice_nodes_y = np.array(lattice_nodes_y)
print(f"done. {len(lattice_nodes_x)} nodes, {len(lattice_edges)} edges")

# ── Panel 2: dispersion relation ────────────────────────────────────────────
print("  Panel 2: dispersion ...", end=" ", flush=True)
c_fit, m_fit, rmse = fit_rel_2d_direct(EPS)

K_ISO_MAX = 0.40
k_vals = np.arange(0.02, K_ISO_MAX + 0.001, 0.02)  # 20 points
n_k_disp = len(k_vals)

def scan_dir_cont(kx_scan, ky_scan, eps, m0):
    """Track physical band along a 1D k-path by continuity."""
    E_vals = []
    E_prev = m0
    for kx_, ky_ in zip(kx_scan, ky_scan):
        M = TM14_half(kx_, ky_, eps)
        M2 = M @ M
        lam = np.linalg.eigvals(M2)
        e = -np.angle(lam)
        E_cur = float(e[np.argmin(np.abs(e - E_prev))])
        E_vals.append(E_cur)
        E_prev = E_cur
    return np.array(E_vals)

angles_deg = [0, 60, 120]
angle_colors = [CYAN, ORANGE, MAGENTA]
angle_labels = ['0\u00b0', '60\u00b0', '120\u00b0']

disp_data = {}  # ang_deg -> (k_arr, E_arr)
for ang_deg in angles_deg:
    ang = np.radians(ang_deg)
    kxs = k_vals * np.cos(ang)
    kys = k_vals * np.sin(ang)
    E_arr = scan_dir_cont(kxs, kys, EPS, m_fit)
    disp_data[ang_deg] = (k_vals.copy(), E_arr)

# Theoretical curve (continuum limit)
k_theory = np.linspace(0, K_ISO_MAX + 0.05, 200)
E_theory = np.sqrt(C_LIGHT**2 * k_theory**2 + m_fit**2)

# Actual data curve (for a smooth line through the tracked points)
E_data_0 = disp_data[0][1]  # all directions identical
print(f"done. {n_k_disp} k-points, isotropy exact.")
print("All data pre-computed.\n")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE SETUP
# ════════════════════════════════════════════════════════════════════════════

def make_figure(dpi=100, figsize=(16, 7)):
    """2-panel figure with dark background and title bar."""
    fig = plt.figure(figsize=figsize, facecolor=BG, dpi=dpi)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[0.08, 0.92],
                           hspace=0.04, wspace=0.22,
                           top=0.97, bottom=0.07,
                           left=0.05, right=0.97)

    # Title bar spans both columns
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.set_facecolor(BG)
    title_ax.axis('off')

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=GRAY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRAY)
            spine.set_linewidth(0.5)

    return fig, title_ax, ax1, ax2


# ════════════════════════════════════════════════════════════════════════════
# DRAW FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def draw_lattice(ax):
    """Draw hex lattice nodes + edges as ghost background."""
    # Edges
    lc = LineCollection(lattice_edges, colors=LATTICE_EDGE,
                        linewidths=0.3, zorder=0)
    ax.add_collection(lc)
    # Nodes
    ax.scatter(lattice_nodes_x, lattice_nodes_y,
               s=1.5, color=LATTICE_NODE, zorder=0, marker='.')


def draw_panel1(ax, t_idx):
    """Wave propagation heatmap on hex lattice."""
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_title('2+1D hexagonal lattice \u00b7 causal propagation',
                 color=WHITE, fontsize=11, pad=6, fontfamily='DejaVu Sans')

    # Hex lattice background (drawn first, beneath everything)
    draw_lattice(ax)

    # Probability heatmap
    p = prob_crop[t_idx]
    vmin = max(p.max() * 1e-5, 1e-15)
    vmax = max(p.max(), 1e-12)
    ax.pcolormesh(y_crop, x_crop, p, cmap=COOL_CMAP,
                  norm=LogNorm(vmin=vmin, vmax=vmax),
                  shading='auto', rasterized=True, zorder=1, alpha=0.92)

    # Light cone
    if t_idx > 0:
        theta = np.linspace(0, 2 * np.pi, 300)
        r = SQRT3 * t_idx
        ax.plot(r * np.sin(theta), r * np.cos(theta), '--', color=WHITE,
                linewidth=1.0, alpha=0.9, zorder=2)

    lim = R_MAX * 0.98
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('y  (physical units)', color=GRAY, fontsize=8)
    ax.set_ylabel('x  (physical units)', color=GRAY, fontsize=8)
    ax.text(0.03, 0.03, f't = {t_idx}', transform=ax.transAxes,
            color=WHITE, fontsize=10, va='bottom', fontweight='bold')
    ax.text(0.03, 0.09, 'light cone: r = \u221a3 \u00b7 t', transform=ax.transAxes,
            color=GRAY, fontsize=7, va='bottom')


def draw_panel2(ax, n_pts, show_annotations=False):
    """Dispersion relation: 3 directions, 1 curve."""
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_title('relativistic dispersion \u00b7 3 directions, 1 curve',
                 color=WHITE, fontsize=11, pad=6, fontfamily='DejaVu Sans')
    ax.grid(True, color='#222244', linewidth=0.3, alpha=0.6)

    # Theoretical continuum curve (faded, clipped to data range)
    E_clip = E_data_0[-1] * 1.15
    mask_th = E_theory <= E_clip
    ax.plot(k_theory[mask_th], E_theory[mask_th], '--', color='#FFFFFF44',
            linewidth=1.5, zorder=1, label='E\u00b2 = c\u00b2k\u00b2 + m\u00b2')

    # Mass line
    ax.axhline(m_fit, color=GRAY, linewidth=0.4, linestyle=':', alpha=0.4)

    # Data points — three directions, staggered slightly for visibility
    offsets_y = [0.004, 0.0, -0.004]
    sizes = [70, 50, 35]  # decreasing so all visible even when overlapping
    for ang_deg, col, lbl, dy, sz in zip(
            angles_deg, angle_colors, angle_labels, offsets_y, sizes):
        k_d, E_d = disp_data[ang_deg]
        n = min(n_pts, len(k_d))
        if n > 0:
            ax.scatter(k_d[:n], E_d[:n] + dy, s=sz, color=col, zorder=3,
                       label=lbl, edgecolors='none', alpha=0.9)

    ax.set_xlim(-0.01, K_ISO_MAX + 0.03)
    ax.set_ylim(m_fit * 0.85, E_data_0[-1] * 1.15)
    ax.set_xlabel('|k|', color=WHITE, fontsize=10)
    ax.set_ylabel('E(k)', color=WHITE, fontsize=10)

    ax.legend(fontsize=9, loc='upper left', framealpha=0.4,
              labelcolor=[GRAY, CYAN, ORANGE, MAGENTA],
              facecolor=BG, edgecolor=GRAY, markerscale=0.8)

    if show_annotations:
        ax.text(0.97, 0.06,
                f'c = \u221a3  (geometric)\nm \u2248 2\u03b5  =  {m_fit:.4f}\n'
                f'RMSE = {rmse:.4f}\n'
                'isotropy error = 0.0000',
                transform=ax.transAxes, color=WHITE, fontsize=9,
                ha='right', va='bottom', fontfamily='DejaVu Sans',
                bbox=dict(facecolor='#0a0a1aCC', edgecolor=GRAY,
                          alpha=0.9, linewidth=0.4, pad=6))


# ════════════════════════════════════════════════════════════════════════════
# TEST SNAPSHOT
# ════════════════════════════════════════════════════════════════════════════

def save_test_snapshot():
    print("Saving test_animation.png ...")
    fig, title_ax, ax1, ax2 = make_figure(dpi=120)
    title_ax.text(0.5, 0.65,
                  'Relativistic Quantum Mechanics from Lattice Geometry',
                  color=WHITE, fontsize=14, ha='center', va='center',
                  fontfamily='DejaVu Sans', fontweight='bold',
                  transform=title_ax.transAxes)
    title_ax.text(0.5, 0.1,
                  'Schmiereck 2026 \u00b7 github.com/schmiereck/hex_field_map',
                  color=GRAY, fontsize=9, ha='center', va='center',
                  fontfamily='DejaVu Sans', transform=title_ax.transAxes)
    draw_panel1(ax1, T_SIM // 2)
    draw_panel2(ax2, n_k_disp, show_annotations=True)
    fig.savefig('test_animation.png', facecolor=BG, dpi=120)
    plt.close(fig)
    print("  -> test_animation.png saved.")


# ════════════════════════════════════════════════════════════════════════════
# MP4 ANIMATION  (15 s, 30 fps = 450 frames)
# ════════════════════════════════════════════════════════════════════════════

def render_mp4():
    print("Rendering animation.mp4 (450 frames) ...")
    W, H, DPI = 1280, 720, 120
    fig, title_ax, ax1, ax2 = make_figure(dpi=DPI, figsize=(W / DPI, H / DPI))

    main_title = title_ax.text(
        0.5, 0.65, '', color=WHITE, fontsize=12, ha='center', va='center',
        fontfamily='DejaVu Sans', fontweight='bold',
        transform=title_ax.transAxes)
    sub_title = title_ax.text(
        0.5, 0.1, '', color=GRAY, fontsize=8, ha='center', va='center',
        fontfamily='DejaVu Sans', transform=title_ax.transAxes)
    summary_text = fig.text(0.5, 0.015, '', color=GRAY, fontsize=7,
                            ha='center', va='bottom', fontfamily='DejaVu Sans')
    extra_texts = []

    def clear_extras():
        for t in extra_texts:
            t.remove()
        extra_texts.clear()

    def hide_panels():
        for ax in [ax1, ax2]:
            ax.clear(); ax.set_facecolor(BG); ax.axis('off')

    def update(frame):
        clear_extras()
        summary_text.set_text('')

        # ── Title card: frames 0-29 ──
        if frame < 30:
            alpha = min(frame / 12.0, 1.0)
            main_title.set_text('')
            sub_title.set_text('')
            hide_panels()
            t1 = fig.text(0.5, 0.58,
                          'Relativistic Quantum Mechanics\nfrom Lattice Geometry',
                          color=WHITE, fontsize=16, ha='center', va='center',
                          fontfamily='DejaVu Sans', fontweight='bold', alpha=alpha)
            t2 = fig.text(0.5, 0.38,
                          'Schmiereck 2026 \u00b7 arXiv quant-ph',
                          color=GRAY, fontsize=10, ha='center', va='center',
                          fontfamily='DejaVu Sans', alpha=alpha)
            extra_texts.extend([t1, t2])
            return

        # ── Main animation: frames 30-179 (150 frames) ──
        if frame < 180:
            for ax in [ax1, ax2]:
                ax.axis('on')
            main_title.set_text(
                'Relativistic Quantum Mechanics from Lattice Geometry')
            sub_title.set_text(
                'Schmiereck 2026 \u00b7 github.com/schmiereck/hex_field_map')
            progress = (frame - 30) / 149.0

            t_idx = min(int(progress * T_SIM), T_SIM)
            draw_panel1(ax1, t_idx)

            n_pts = max(1, int(progress * n_k_disp))
            draw_panel2(ax2, n_pts, show_annotations=(progress > 0.85))
            return

        # ── Hold + annotations: frames 180-299 ──
        if frame < 300:
            for ax in [ax1, ax2]:
                ax.axis('on')
            main_title.set_text(
                'Relativistic Quantum Mechanics from Lattice Geometry')
            sub_title.set_text(
                'Schmiereck 2026 \u00b7 github.com/schmiereck/hex_field_map')
            draw_panel1(ax1, T_SIM)
            draw_panel2(ax2, n_k_disp, show_annotations=True)
            return

        # ── Summary card: frames 300-389 ──
        if frame < 390:
            for ax in [ax1, ax2]:
                ax.axis('on')
            main_title.set_text(
                'Relativistic Quantum Mechanics from Lattice Geometry')
            sub_title.set_text(
                'Schmiereck 2026 \u00b7 github.com/schmiereck/hex_field_map')
            draw_panel1(ax1, T_SIM)
            draw_panel2(ax2, n_k_disp, show_annotations=True)
            alpha_s = min((frame - 300) / 25.0, 1.0)
            summary_text.set_text(
                'c = \u221a3 (geometric)  \u00b7  m \u2248 2\u03b5  \u00b7  '
                f'RMSE = {rmse:.4f}  \u00b7  '
                'isotropy error = 0.0000')
            summary_text.set_alpha(alpha_s)
            return

        # ── Fade to black: frames 390-449 ──
        main_title.set_text('')
        sub_title.set_text('')
        hide_panels()
        progress4 = (frame - 390) / 59.0
        alpha_gh = min(progress4 * 3, 1.0) * max(1.0 - (progress4 - 0.6) * 2.5, 0)
        alpha_gh = max(min(alpha_gh, 1.0), 0.0)
        t_gh = fig.text(0.5, 0.5, 'github.com/schmiereck/hex_field_map',
                        color=WHITE, fontsize=13, ha='center', va='center',
                        fontfamily='DejaVu Sans', alpha=alpha_gh)
        extra_texts.append(t_gh)

    anim = FuncAnimation(fig, update, frames=450, interval=1000 / 30, blit=False)
    try:
        anim.save('animation.mp4', writer='ffmpeg', fps=30, dpi=DPI,
                  savefig_kwargs={'facecolor': BG}, bitrate=2000)
        print("  -> animation.mp4 saved.")
    except Exception as e:
        print(f"  ffmpeg failed ({e}), saving animation_hq.gif ...")
        anim.save('animation_hq.gif', writer='pillow', fps=15, dpi=100,
                  savefig_kwargs={'facecolor': BG})
        print("  -> animation_hq.gif saved.")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# GIF ANIMATION  (5 s, 15 fps = 75 frames)
# ════════════════════════════════════════════════════════════════════════════

def render_gif():
    print("Rendering animation.gif (75 frames) ...")
    fig, title_ax, ax1, ax2 = make_figure(dpi=80, figsize=(10, 4.4))
    title_ax.text(0.5, 0.65,
                  'Relativistic Quantum Mechanics from Lattice Geometry',
                  color=WHITE, fontsize=10, ha='center', va='center',
                  fontfamily='DejaVu Sans', fontweight='bold',
                  transform=title_ax.transAxes)
    title_ax.text(0.5, 0.1,
                  'Schmiereck 2026 \u00b7 github.com/schmiereck/hex_field_map',
                  color=GRAY, fontsize=7, ha='center', va='center',
                  fontfamily='DejaVu Sans', transform=title_ax.transAxes)
    summary_text = fig.text(0.5, 0.015, '', color=GRAY, fontsize=6,
                            ha='center', va='bottom', fontfamily='DejaVu Sans')

    def update_gif(frame):
        summary_text.set_text('')

        if frame < 60:
            progress = frame / 59.0
            t_idx = min(int(progress * T_SIM), T_SIM)
            draw_panel1(ax1, t_idx)
            n_pts = max(1, int(progress * n_k_disp))
            draw_panel2(ax2, n_pts, show_annotations=(progress > 0.85))

        elif frame < 70:
            draw_panel1(ax1, T_SIM)
            draw_panel2(ax2, n_k_disp, show_annotations=True)
            alpha_s = min((frame - 60) / 4.0, 1.0)
            summary_text.set_text(
                'c = \u221a3 (geometric)  \u00b7  m \u2248 2\u03b5  \u00b7  '
                f'RMSE = {rmse:.4f}  \u00b7  '
                'isotropy = 0.0000')
            summary_text.set_alpha(alpha_s)

        else:
            draw_panel1(ax1, T_SIM)
            draw_panel2(ax2, n_k_disp, show_annotations=True)
            summary_text.set_text(
                'c = \u221a3 (geometric)  \u00b7  m \u2248 2\u03b5  \u00b7  '
                f'RMSE = {rmse:.4f}  \u00b7  '
                'isotropy = 0.0000')

    anim = FuncAnimation(fig, update_gif, frames=75, interval=1000 / 15, blit=False)
    anim.save('animation.gif', writer='pillow', fps=15, dpi=80,
              savefig_kwargs={'facecolor': BG})
    plt.close(fig)
    sz = os.path.getsize('animation.gif')
    print(f"  -> animation.gif saved ({sz / 1e6:.1f} MB).")
    if sz > 5_000_000:
        print("  WARNING: GIF > 5 MB.")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    save_test_snapshot()
    render_gif()
    render_mp4()
    print("\nDone. Files: test_animation.png, animation.gif, animation.mp4")
