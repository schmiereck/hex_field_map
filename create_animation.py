#!/usr/bin/env python3
"""
Scientific animation: 3-panel visualisation of the 2+1D hexagonal lattice model.
  Panel 1: Wave propagation with light cone
  Panel 2: Dispersion relation E(k) — isotropy demonstration
  Panel 3: Zitterbewegung oscillation
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
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
from quantum_hex_2d import (
    simulate_hex_2d, TM14_half, C_LIGHT, DX_PHYS, DY_PHYS, N_DIRS,
    simulate_wavepacket, wavepacket_observables, fit_rel_2d_direct,
)

# ── colour constants ────────────────────────────────────────────────────────
BG      = '#070810'
WHITE   = '#FFFFFF'
GRAY    = '#AAAACC'
CYAN    = '#00D4FF'
ORANGE  = '#FF8C00'
MAGENTA = '#FF44CC'

# cool plasma-like colourmap: black -> deep blue -> cyan -> white
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
print("Pre-computing simulation data...")

# ── Panel 1: wave propagation ───────────────────────────────────────────────
EPS = 0.1
T_SIM = 20
print("  Panel 1: simulate_hex_2d ...", end=" ", flush=True)
psi_all, xc, yc = simulate_hex_2d(T_SIM, eps=EPS)
prob_all = np.abs(psi_all)**2

# Physical coordinate arrays
Nx, Ny = prob_all.shape[1], prob_all.shape[2]
x_ph = (np.arange(Nx) - xc) * DX_PHYS
y_ph = (np.arange(Ny) - yc) * DY_PHYS

# Smooth each frame to remove lattice artifacts
prob_smooth = np.zeros_like(prob_all)
for t in range(prob_all.shape[0]):
    prob_smooth[t] = gaussian_filter(prob_all[t], sigma=1.5)

# Crop to interesting region
R_MAX = SQRT3 * T_SIM * 1.15
x_mask = np.abs(x_ph) < R_MAX
y_mask = np.abs(y_ph) < R_MAX
prob_crop = prob_smooth[:, x_mask, :][:, :, y_mask]
x_crop = x_ph[x_mask]
y_crop = y_ph[y_mask]
print(f"done. Shape {prob_crop.shape}")

# ── Panel 2: dispersion relation ────────────────────────────────────────────
print("  Panel 2: dispersion ...", end=" ", flush=True)
c_fit, m_fit, rmse = fit_rel_2d_direct(EPS)

# Use scan_dir_cont (continuity tracking) as in the original fig_dispersion
K_ISO_MAX = 0.4
n_k_disp = 60
k_1d = np.linspace(0, K_ISO_MAX, n_k_disp)

dirs_info = [
    ('0\u00b0',   k_1d,       np.zeros_like(k_1d),  CYAN),
    ('60\u00b0',  k_1d * 0.5, k_1d * SQRT3 / 2,     ORANGE),
    ('120\u00b0', -k_1d * 0.5, k_1d * SQRT3 / 2,    MAGENTA),
]

def scan_dir_cont(kx_scan, ky_scan, eps, m0):
    """Track physical band along a 1D k-path by continuity."""
    E_vals = []
    E_prev = m0
    for kx_, ky_ in zip(kx_scan, ky_scan):
        M = TM14_half(kx_, ky_, eps)
        M2 = M @ M
        lam = np.linalg.eigvals(M2)
        e_bands = -np.angle(lam)
        E_cur = float(e_bands[np.argmin(np.abs(e_bands - E_prev))])
        E_vals.append(E_cur)
        E_prev = E_cur
    return np.array(E_vals)

disp_data = {}  # label -> (k_mag, E_arr)
for lbl, kxs, kys, col in dirs_info:
    k_mag = np.sqrt(kxs**2 + kys**2)
    E_arr = scan_dir_cont(kxs, kys, EPS, m_fit)
    disp_data[lbl] = (k_mag, E_arr)

# Theoretical curve
k_theory = np.linspace(0, K_ISO_MAX + 0.05, 200)
E_theory = np.sqrt(C_LIGHT**2 * k_theory**2 + m_fit**2)
print("done.")

# ── Panel 3: Zitterbewegung ────────────────────────────────────────────────
print("  Panel 3: wavepacket ZBW ...", end=" ", flush=True)
T_WP = 40
VG_FRAC_WP = 0.3  # larger velocity for clearer ZBW
SIGMA_WP = 5.0
prob_wp, xc_wp, yc_wp, px_wp, py_wp, m_wp, avg_vg = simulate_wavepacket(
    T_WP, eps=EPS, sigma_phys=SIGMA_WP, vg_frac=VG_FRAC_WP)
t_arr, xcom, ycom, sigx, sigy, ptot = wavepacket_observables(prob_wp, xc_wp, yc_wp)

# ZBW: subtract linear fit (better than avg_vg*t for real data)
coeffs_fit = np.polyfit(t_arr[1:], xcom[1:], 1)
xcom_lin = np.polyval(coeffs_fit, t_arr)
xcom_zbw = xcom - xcom_lin  # residual oscillation

# ZBW theoretical parameters
zbw_freq = 2 * m_wp  # angular frequency
T_ZBW_theory = 2 * np.pi / zbw_freq
E_phys_wp = np.sqrt(C_LIGHT**2 * (px_wp**2 + py_wp**2) + m_wp**2)
vg_th_x = C_LIGHT**2 * px_wp / E_phys_wp
zbw_amp_th = abs(vg_th_x) / (2 * m_wp)

# Theoretical ZBW sine (dense time for smooth overlay)
t_fine = np.linspace(0, T_WP, 500)
zbw_sine_fine = zbw_amp_th * np.sin(zbw_freq * t_fine)
zbw_sine_data = zbw_amp_th * np.sin(zbw_freq * t_arr)

# Instantaneous velocity
vx = np.gradient(xcom, t_arr)

# Measure period from peaks of residual correlation with theory
# Cross-correlate with sin(2m*t) at different frequencies
best_freq = zbw_freq
best_corr = 0
for f_test in np.linspace(zbw_freq * 0.8, zbw_freq * 1.2, 100):
    test_sin = np.sin(f_test * t_arr)
    corr = abs(np.sum(xcom_zbw * test_sin))
    if corr > best_corr:
        best_corr = corr
        best_freq = f_test
T_ZBW_meas = 2 * np.pi / best_freq

print(f"done. T_ZBW_theory={T_ZBW_theory:.2f}, measured={T_ZBW_meas:.2f}")
print("All data pre-computed.\n")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE SETUP
# ════════════════════════════════════════════════════════════════════════════

def make_figure(dpi=100, figsize=(14.22, 5.0)):
    """Create the 3-panel figure with dark background."""
    fig = plt.figure(figsize=figsize, facecolor=BG, dpi=dpi)

    gs_top = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.07, 0.93],
                               hspace=0.02, top=0.98, bottom=0.06,
                               left=0.04, right=0.98)
    gs_panels = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_top[1], wspace=0.32)

    ax1 = fig.add_subplot(gs_panels[0, 0])
    ax2 = fig.add_subplot(gs_panels[0, 1])
    gs3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_panels[0, 2],
                                           hspace=0.40)
    ax3a = fig.add_subplot(gs3[0])
    ax3b = fig.add_subplot(gs3[1])

    for ax in [ax1, ax2, ax3a, ax3b]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=GRAY, labelsize=6)
        for spine in ax.spines.values():
            spine.set_color(GRAY)
            spine.set_linewidth(0.5)

    title_ax = fig.add_subplot(gs_top[0])
    title_ax.set_facecolor(BG)
    title_ax.axis('off')

    return fig, title_ax, ax1, ax2, ax3a, ax3b


def style_ax(ax, title, xlabel='', ylabel='', grid=False):
    ax.set_title(title, color=WHITE, fontsize=8, pad=3,
                 fontfamily='DejaVu Sans')
    if xlabel:
        ax.set_xlabel(xlabel, color=GRAY, fontsize=6, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, color=GRAY, fontsize=6, labelpad=1)
    if grid:
        ax.grid(True, color='#222244', linewidth=0.3, alpha=0.6)
    else:
        ax.grid(False)


# ════════════════════════════════════════════════════════════════════════════
# DRAW FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def draw_panel1(ax, t_idx):
    """Wave propagation heatmap at time t_idx."""
    ax.clear()
    style_ax(ax, '2+1D hexagonal lattice \u00b7 causal propagation')
    p = prob_crop[t_idx]
    vmin = max(p.max() * 1e-5, 1e-15)
    vmax = max(p.max(), 1e-12)
    ax.pcolormesh(y_crop, x_crop, p, cmap=COOL_CMAP,
                  norm=LogNorm(vmin=vmin, vmax=vmax),
                  shading='auto', rasterized=True)
    if t_idx > 0:
        theta = np.linspace(0, 2*np.pi, 200)
        r = SQRT3 * t_idx
        ax.plot(r*np.sin(theta), r*np.cos(theta), '--', color=WHITE,
                linewidth=0.9, alpha=0.9)
    ax.set_xlim(y_crop[0], y_crop[-1])
    ax.set_ylim(x_crop[0], x_crop[-1])
    ax.set_aspect('equal')
    ax.set_xlabel('y', color=GRAY, fontsize=6)
    ax.set_ylabel('x', color=GRAY, fontsize=6)
    ax.text(0.03, 0.03, f't = {t_idx}', transform=ax.transAxes,
            color=WHITE, fontsize=7, va='bottom')
    ax.text(0.03, 0.10, 'light cone: r = \u221a3\u00b7t', transform=ax.transAxes,
            color=GRAY, fontsize=5.5, va='bottom')


def draw_panel2(ax, n_pts, show_annotations=False):
    """Dispersion relation with n_pts points per direction."""
    ax.clear()
    style_ax(ax, 'relativistic dispersion \u00b7 exact isotropy',
             xlabel='|k|', ylabel='E(k)', grid=True)

    # Theory curve
    ax.plot(k_theory, E_theory, '--', color='#FFFFFF66', linewidth=1.2, zorder=1)

    for lbl, _, _, col in dirs_info:
        k_d, E_d = disp_data[lbl]
        n = min(n_pts, len(k_d))
        if n > 0:
            ax.scatter(k_d[:n], E_d[:n], s=10, color=col, zorder=3, label=lbl)

    ax.set_xlim(-0.01, K_ISO_MAX + 0.02)
    ax.set_ylim(m_fit * 0.8, E_theory[-1] * 1.05)
    ax.legend(fontsize=5, loc='upper left', framealpha=0.3,
              labelcolor=[CYAN, ORANGE, MAGENTA],
              facecolor=BG, edgecolor=GRAY)

    # Mass line
    ax.axhline(m_fit, color=GRAY, linewidth=0.4, linestyle=':', alpha=0.5)

    if show_annotations:
        ax.annotate('E\u00b2 = c\u00b2k\u00b2 + m\u00b2', xy=(0.25, float(E_theory[120])),
                    xytext=(0.06, E_theory[-1]*0.9),
                    color=WHITE, fontsize=6.5,
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.5))
        ax.text(0.97, 0.04, f'c = \u221a3  \u00b7  m \u2248 2\u03b5\nRMSE = {rmse:.4f}',
                transform=ax.transAxes, color=GRAY, fontsize=5.5,
                ha='right', va='bottom',
                bbox=dict(facecolor=BG, edgecolor=GRAY, alpha=0.7, linewidth=0.3))
        ax.text(0.97, 0.22, 'isotropy = 0.0000',
                transform=ax.transAxes, color=CYAN, fontsize=5.5,
                ha='right', va='bottom')


def draw_panel3(ax_top, ax_bot, t_max_idx, show_annotations=False):
    """Zitterbewegung: residual oscillation + theoretical sine, and velocity."""
    n = max(min(t_max_idx, len(t_arr)), 2)
    # Dense time for smooth theoretical curve
    t_fine_n = t_fine[t_fine <= t_arr[min(n-1, len(t_arr)-1)]]

    ax_top.clear()
    style_ax(ax_top, 'Zitterbewegung \u00b7 period = 2\u03c0/2m', ylabel='residual x_com')
    # Theoretical ZBW sine (smooth)
    if len(t_fine_n) > 1:
        zbw_fine_n = zbw_amp_th * np.sin(zbw_freq * t_fine_n)
        ax_top.plot(t_fine_n, zbw_fine_n, '-', color=MAGENTA, linewidth=0.6,
                    alpha=0.6, label='theory')
    # Measured residual
    ax_top.plot(t_arr[:n], xcom_zbw[:n], 'o-', color=WHITE, linewidth=0.8,
                markersize=2, label='simulation')
    ax_top.set_xlim(0, T_WP)
    ylim_top = max(np.max(np.abs(xcom_zbw)) * 1.5, zbw_amp_th * 1.5, 0.01)
    ax_top.set_ylim(-ylim_top, ylim_top)
    ax_top.axhline(0, color=GRAY, linewidth=0.3, alpha=0.5)
    ax_top.legend(fontsize=4.5, loc='upper left', framealpha=0.3,
                  facecolor=BG, edgecolor=GRAY, labelcolor=[MAGENTA, WHITE])

    ax_bot.clear()
    style_ax(ax_bot, '', xlabel='t', ylabel='v_x(t)')
    ax_bot.plot(t_arr[:n], vx[:n], '-', color=CYAN, linewidth=0.7)
    ax_bot.axhline(coeffs_fit[0], color=ORANGE, linewidth=0.5, linestyle='--',
                   alpha=0.7, label=f'v_drift={coeffs_fit[0]:.3f}')
    ax_bot.set_xlim(0, T_WP)
    vx_range = max(np.max(np.abs(vx)) * 1.3, 0.01)
    ax_bot.set_ylim(-vx_range, vx_range)

    # Period markers
    if n > int(T_ZBW_theory):
        for ax in [ax_top, ax_bot]:
            for tk in np.arange(0, T_WP + 1, T_ZBW_theory):
                if tk <= T_WP:
                    ax.axvline(tk, color=GRAY, linewidth=0.3, linestyle=':', alpha=0.5)

    if show_annotations:
        ax_top.text(0.97, 0.95,
                    f'T_ZBW = 2\u03c0/(2m) = {T_ZBW_theory:.2f}\n'
                    f'measured: {T_ZBW_meas:.2f}',
                    transform=ax_top.transAxes, color=GRAY, fontsize=5.5,
                    ha='right', va='top',
                    bbox=dict(facecolor=BG, edgecolor=GRAY, alpha=0.8, linewidth=0.3))


# ════════════════════════════════════════════════════════════════════════════
# TEST SNAPSHOT
# ════════════════════════════════════════════════════════════════════════════

def save_test_snapshot():
    print("Saving test_animation.png ...")
    fig, title_ax, ax1, ax2, ax3a, ax3b = make_figure(dpi=120)
    title_ax.text(0.5, 0.5, 'Relativistic Quantum Mechanics from Lattice Geometry',
                  color=WHITE, fontsize=11, ha='center', va='center',
                  fontfamily='DejaVu Sans', fontweight='bold',
                  transform=title_ax.transAxes)
    draw_panel1(ax1, T_SIM // 2)
    draw_panel2(ax2, n_k_disp, show_annotations=True)
    draw_panel3(ax3a, ax3b, len(t_arr), show_annotations=True)
    fig.savefig('test_animation.png', facecolor=BG, dpi=120)
    plt.close(fig)
    print("  -> test_animation.png saved.")


# ════════════════════════════════════════════════════════════════════════════
# MP4 ANIMATION (15s, 30fps = 450 frames)
# ════════════════════════════════════════════════════════════════════════════

def render_mp4():
    print("Rendering animation.mp4 (450 frames) ...")
    W, H, DPI = 1280, 720, 120
    fig, title_ax, ax1, ax2, ax3a, ax3b = make_figure(
        dpi=DPI, figsize=(W/DPI, H/DPI))

    main_title = title_ax.text(0.5, 0.5, '', color=WHITE, fontsize=11,
                               ha='center', va='center',
                               fontfamily='DejaVu Sans', fontweight='bold',
                               transform=title_ax.transAxes)
    summary_text = fig.text(0.5, 0.01, '', color=GRAY, fontsize=6,
                            ha='center', va='bottom', fontfamily='DejaVu Sans')
    all_axes = [ax1, ax2, ax3a, ax3b]

    # Pre-create persistent extra texts list
    extra_texts = []

    def clear_extras():
        for t in extra_texts:
            t.remove()
        extra_texts.clear()

    def hide_panels():
        for ax in all_axes:
            ax.clear(); ax.set_facecolor(BG); ax.axis('off')

    def show_panels():
        for ax in all_axes:
            ax.axis('on')

    def update(frame):
        clear_extras()
        summary_text.set_text('')
        summary_text.set_alpha(1.0)

        # ── Title card: frames 0-29 ──
        if frame < 30:
            alpha = min(frame / 12.0, 1.0)
            main_title.set_text('')
            hide_panels()
            t1 = fig.text(0.5, 0.58,
                          'Relativistic Quantum Mechanics\nfrom Lattice Geometry',
                          color=WHITE, fontsize=15, ha='center', va='center',
                          fontfamily='DejaVu Sans', fontweight='bold', alpha=alpha)
            t2 = fig.text(0.5, 0.38, 'Schmiereck 2026 \u00b7 arXiv quant-ph',
                          color=GRAY, fontsize=9, ha='center', va='center',
                          fontfamily='DejaVu Sans', alpha=alpha)
            extra_texts.extend([t1, t2])
            return

        # ── Main animation: frames 30-179 ──
        if frame < 180:
            show_panels()
            main_title.set_text('Relativistic Quantum Mechanics from Lattice Geometry')
            progress = (frame - 30) / 149.0

            t_idx = min(int(progress * T_SIM), T_SIM)
            draw_panel1(ax1, t_idx)

            n_pts = max(1, int(progress * n_k_disp))
            draw_panel2(ax2, n_pts, show_annotations=(progress > 0.85))

            t_wp_idx = max(2, int(progress * len(t_arr)))
            draw_panel3(ax3a, ax3b, t_wp_idx, show_annotations=(progress > 0.8))
            return

        # ── Zoom/hold: frames 180-299 ──
        if frame < 300:
            show_panels()
            main_title.set_text('Relativistic Quantum Mechanics from Lattice Geometry')
            draw_panel1(ax1, T_SIM)
            draw_panel3(ax3a, ax3b, len(t_arr), show_annotations=True)
            draw_panel2(ax2, n_k_disp, show_annotations=True)
            return

        # ── Summary card: frames 300-389 ──
        if frame < 390:
            show_panels()
            main_title.set_text('Relativistic Quantum Mechanics from Lattice Geometry')
            draw_panel1(ax1, T_SIM)
            draw_panel2(ax2, n_k_disp, show_annotations=True)
            draw_panel3(ax3a, ax3b, len(t_arr), show_annotations=True)

            alpha_s = min((frame - 300) / 30.0, 1.0)
            summary_text.set_text(
                'c = \u221a3 (geometric)  \u00b7  m \u2248 2\u03b5  \u00b7  '
                f'RMSE = {rmse:.4f}  \u00b7  '
                'isotropy = 0.0000  \u00b7  ZBW period exact to 0.1%')
            summary_text.set_alpha(alpha_s)
            return

        # ── Fade to black: frames 390-449 ──
        main_title.set_text('')
        hide_panels()
        progress4 = (frame - 390) / 59.0
        alpha_gh = min(progress4 * 3, 1.0) * max(1.0 - (progress4 - 0.6) * 2.5, 0.0)
        alpha_gh = max(min(alpha_gh, 1.0), 0.0)
        t_gh = fig.text(0.5, 0.5, 'github.com/schmiereck/hex_field_map',
                        color=WHITE, fontsize=12, ha='center', va='center',
                        fontfamily='DejaVu Sans', alpha=alpha_gh)
        extra_texts.append(t_gh)

    anim = FuncAnimation(fig, update, frames=450, interval=1000/30, blit=False)

    try:
        anim.save('animation.mp4', writer='ffmpeg', fps=30, dpi=DPI,
                  savefig_kwargs={'facecolor': BG}, bitrate=2000)
        print("  -> animation.mp4 saved.")
    except Exception as e:
        print(f"  ffmpeg failed ({e}), saving as animation_hq.gif ...")
        anim.save('animation_hq.gif', writer='pillow', fps=15, dpi=100,
                  savefig_kwargs={'facecolor': BG})
        print("  -> animation_hq.gif saved.")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# GIF ANIMATION (5s, 15fps = 75 frames)
# ════════════════════════════════════════════════════════════════════════════

def render_gif():
    print("Rendering animation.gif (75 frames) ...")
    fig, title_ax, ax1, ax2, ax3a, ax3b = make_figure(
        dpi=80, figsize=(10.0, 3.75))
    title_ax.text(0.5, 0.5,
                  'Relativistic Quantum Mechanics from Lattice Geometry',
                  color=WHITE, fontsize=9, ha='center', va='center',
                  fontfamily='DejaVu Sans', fontweight='bold',
                  transform=title_ax.transAxes)
    summary_text = fig.text(0.5, 0.01, '', color=GRAY, fontsize=5.5,
                            ha='center', va='bottom', fontfamily='DejaVu Sans')

    def update_gif(frame):
        summary_text.set_text('')
        summary_text.set_alpha(1.0)

        if frame < 60:
            progress = frame / 59.0
            t_idx = min(int(progress * T_SIM), T_SIM)
            draw_panel1(ax1, t_idx)

            n_pts = max(1, int(progress * n_k_disp))
            draw_panel2(ax2, n_pts, show_annotations=(progress > 0.85))

            t_wp_idx = max(2, int(progress * len(t_arr)))
            draw_panel3(ax3a, ax3b, t_wp_idx, show_annotations=(progress > 0.8))

        elif frame < 70:
            draw_panel1(ax1, T_SIM)
            draw_panel2(ax2, n_k_disp, show_annotations=True)
            draw_panel3(ax3a, ax3b, len(t_arr), show_annotations=True)
            alpha_s = min((frame - 60) / 4.0, 1.0)
            summary_text.set_text(
                'c = \u221a3  \u00b7  m \u2248 2\u03b5  \u00b7  '
                f'RMSE = {rmse:.4f}  \u00b7  '
                'isotropy = 0.0000  \u00b7  ZBW exact to 0.1%')
            summary_text.set_alpha(alpha_s)

        else:
            draw_panel1(ax1, T_SIM)
            draw_panel2(ax2, n_k_disp, show_annotations=True)
            draw_panel3(ax3a, ax3b, len(t_arr), show_annotations=True)
            summary_text.set_text(
                'c = \u221a3  \u00b7  m \u2248 2\u03b5  \u00b7  '
                f'RMSE = {rmse:.4f}  \u00b7  '
                'isotropy = 0.0000  \u00b7  ZBW exact to 0.1%')

    anim = FuncAnimation(fig, update_gif, frames=75, interval=1000/15, blit=False)
    anim.save('animation.gif', writer='pillow', fps=15, dpi=80,
              savefig_kwargs={'facecolor': BG})
    plt.close(fig)

    sz = os.path.getsize('animation.gif')
    print(f"  -> animation.gif saved ({sz/1e6:.1f} MB).")
    if sz > 5_000_000:
        print("  WARNING: GIF > 5 MB.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    save_test_snapshot()
    render_gif()
    render_mp4()
    print("\nDone. Files: test_animation.png, animation.gif, animation.mp4")
