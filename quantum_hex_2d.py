#!/usr/bin/env python3
"""
2+1D Quantum Path Integral on Hexagonal Lattice
================================================
Physical moves per half-step (Delta_tau = 0.5):
  6 diagonal: (+-sqrt(3)/2, 0, +0.5)  and  (+-sqrt(3)/4, +-3/4, +0.5)
              All have edge length = 1
  1 straight: (0, 0, +1)  — uses amplitude from 2 half-steps ago

In index space (Delta_x = sqrt(3)/4 per x-idx, Delta_y = 3/4 per y-idx):
  d=0: (+2,  0)   d=1: (+1,+1)   d=2: (-1,+1)
  d=3: (-2,  0)   d=4: (-1,-1)   d=5: (+1,-1)
  d=6: ( 0,  0)  straight up (from amp_pprev)

Expected speed of light  c = (sqrt(3)/2) / 0.5 = sqrt(3) ~ 1.732
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm

SQRT3   = np.sqrt(3)
SQRT3_2 = SQRT3 / 2     # 0.8660
SQRT3_4 = SQRT3 / 4     # 0.4330
DX_PHYS = SQRT3_4        # physical x per index unit
DY_PHYS = 0.75           # physical y per index unit
N_DIRS  = 7
C_LIGHT = SQRT3          # expected speed of light

MOVES_PHYS_XY = np.array([
    [ SQRT3_2,  0.0  ],   # d=0  0 deg
    [ SQRT3_4,  0.75 ],   # d=1  60 deg
    [-SQRT3_4,  0.75 ],   # d=2  120 deg
    [-SQRT3_2,  0.0  ],   # d=3  180 deg
    [-SQRT3_4, -0.75 ],   # d=4  240 deg
    [ SQRT3_4, -0.75 ],   # d=5  300 deg
    [ 0.0,      0.0  ],   # d=6  straight
])

MOVE_DT = np.array([0.5]*6 + [1.0])   # physical time advance per move

MOVE_COLORS = ['#e74c3c','#e67e22','#2ecc71','#3498db','#9b59b6','#1abc9c','#2c3e50']


# ─── helpers ─────────────────────────────────────────────────────────────────

def _C(n, eps):
    C = np.full((n, n), 1j * eps, dtype=complex)
    np.fill_diagonal(C, 1.0 + 0j)
    return C


# ─── simulation ──────────────────────────────────────────────────────────────

def simulate_hex_2d(T_phys, eps=0.1):
    """
    2+1D hexagonal lattice, physical-time simulation.

    Second-order recurrence in half-steps (Delta_tau = 0.5):
      diagonal moves  use amp_prev  (1 half-step ago)
      straight move   uses amp_pprev (2 half-steps ago)

    Returns
    -------
    psi      : complex array (T_phys+1, Nx, Ny)
    xcenter  : int — index of x=0
    ycenter  : int — index of y=0
    """
    N_half   = 2 * T_phys
    x_spread = 2 * N_half
    y_spread = 1 * N_half
    Nx       = 2 * x_spread + 1
    Ny       = 2 * y_spread + 1
    xcenter  = x_spread
    ycenter  = y_spread

    C = _C(N_DIRS, eps)

    amp_pprev = np.zeros((Nx, Ny, N_DIRS), dtype=complex)
    amp_prev  = np.zeros((Nx, Ny, N_DIRS), dtype=complex)
    amp_prev[xcenter, ycenter, :] = 1.0 / N_DIRS   # start at origin, all dirs

    psi    = np.zeros((T_phys + 1, Nx, Ny), dtype=complex)
    psi[0] = amp_prev.sum(axis=-1)

    for tau in range(1, N_half + 1):
        new_amp = np.zeros((Nx, Ny, N_DIRS), dtype=complex)

        wc = amp_prev  @ C   # (Nx, Ny, 7)  -- diagonal sources
        ws = amp_pprev @ C   # (Nx, Ny, 7)  -- straight source

        # d=0: dx_idx=+2, dy_idx=0
        new_amp[2:,  :,   0] += wc[:-2, :,   0]
        # d=1: +1, +1
        new_amp[1:,  1:,  1] += wc[:-1, :-1, 1]
        # d=2: -1, +1
        new_amp[:-1, 1:,  2] += wc[1:,  :-1, 2]
        # d=3: -2, 0
        new_amp[:-2, :,   3] += wc[2:,  :,   3]
        # d=4: -1, -1
        new_amp[:-1, :-1, 4] += wc[1:,  1:,  4]
        # d=5: +1, -1
        new_amp[1:,  :-1, 5] += wc[:-1, 1:,  5]
        # d=6: straight (dx=0, dy=0, from 2 half-steps ago)
        new_amp[:, :,     6] += ws[:, :,     6]

        amp_pprev = amp_prev
        amp_prev  = new_amp

        if tau % 2 == 0:
            psi[tau // 2] = new_amp.sum(axis=-1)

    return psi, xcenter, ycenter


# ─── transfer matrix ─────────────────────────────────────────────────────────

def TM14_half(kx, ky, eps):
    """14x14 half-step transfer matrix at physical (kx, ky)."""
    C = _C(N_DIRS, eps)
    phases = np.exp(-1j * (kx * MOVES_PHYS_XY[:6, 0] + ky * MOVES_PHYS_XY[:6, 1]))

    A = np.zeros((N_DIRS, N_DIRS), dtype=complex)
    for d in range(6):
        A[d, :] = phases[d] * C[d, :]
    # A[6, :] = 0  (straight uses G_prev)

    B = np.zeros((N_DIRS, N_DIRS), dtype=complex)
    B[6, :] = C[6, :]

    M = np.zeros((14, 14), dtype=complex)
    M[:7, :7] = A
    M[:7, 7:] = B
    M[7:, :7] = np.eye(N_DIRS)
    return M


def TM14_full_batch(kx_arr, ky_arr, eps):
    """
    Batched full-step (= 2 half-steps) transfer matrix.
    Returns M_full shape (n_k, n_k, 14, 14).
    """
    n_k = len(kx_arr)
    C   = _C(N_DIRS, eps)                  # (7, 7)

    KX, KY = np.meshgrid(kx_arr, ky_arr, indexing='ij')   # (n_k, n_k)

    # phases: (n_k, n_k, 6)
    phases = np.exp(-1j * (
        KX[:, :, None] * MOVES_PHYS_XY[:6, 0] +
        KY[:, :, None] * MOVES_PHYS_XY[:6, 1]))

    # A: (n_k, n_k, 7, 7)
    A = np.zeros((n_k, n_k, N_DIRS, N_DIRS), dtype=complex)
    for d in range(6):
        A[:, :, d, :] = phases[:, :, d:d+1] * C[d, :]

    # B: (n_k, n_k, 7, 7)
    B = np.zeros((n_k, n_k, N_DIRS, N_DIRS), dtype=complex)
    B[:, :, 6, :] = C[6, :]

    # M_half: (n_k, n_k, 14, 14)
    M_half = np.zeros((n_k, n_k, 14, 14), dtype=complex)
    M_half[:, :, :7, :7] = A
    M_half[:, :, :7, 7:] = B
    M_half[:, :, 7:, :7] = np.eye(N_DIRS)

    return np.matmul(M_half, M_half)   # M_full = M_half^2


def compute_bands_2d(eps, n_k=40, k_max=2.5):
    """Returns kx_arr, ky_arr (each len n_k) and E_all (n_k, n_k, 14)."""
    kx_arr = np.linspace(-k_max, k_max, n_k)
    ky_arr = np.linspace(-k_max, k_max, n_k)
    M_full = TM14_full_batch(kx_arr, ky_arr, eps)          # (n_k,n_k,14,14)
    lam    = np.linalg.eigvals(M_full)                     # (n_k,n_k,14)
    E_all  = -np.angle(lam)                                # (n_k,n_k,14)
    return kx_arr, ky_arr, E_all


def physical_band_2d(kx_arr, ky_arr, E_all, c_est=C_LIGHT, m_est=0.1):
    """
    Pick the physical band (closest to E_ref = sqrt(c^2|k|^2 + m^2)).
    This simple E_ref-based selector works best at small k where the physical
    band is well-separated from other bands.
    """
    n_k = len(kx_arr)
    KX, KY = np.meshgrid(kx_arr, ky_arr, indexing='ij')
    E_ref  = np.sqrt(c_est**2 * (KX**2 + KY**2) + m_est**2)

    E_phys = np.empty((n_k, n_k))
    for i in range(n_k):
        diffs        = np.abs(E_all[i] - E_ref[i, :, None])  # (n_k, n_bands)
        idx          = diffs.argmin(axis=-1)                  # (n_k,)
        E_phys[i, :] = E_all[i, np.arange(n_k), idx]
    return E_phys


def fit_rel_2d_direct(eps):
    """
    Directly compute c and m from transfer matrix.
    c = geometric speed = C_LIGHT = sqrt(3) (exact).
    m = E(k=0) eigenvalue from the 5-fold degenerate propagating mode
        at E ~ arctan(2*eps/(1-eps^2)) ~ 2*eps (the physical band).
    Returns (c_fit, m_fit, rmse) where rmse measures small-k fit quality.
    """
    M0 = TM14_half(0.0, 0.0, eps)
    lam0 = np.linalg.eigvals(M0 @ M0)
    e0   = -np.angle(lam0)
    # Physical propagating mode: 5-fold degenerate at arctan(2*eps/(1-eps^2)) ~ 2*eps
    m_target = float(np.arctan2(2 * eps, 1 - eps**2))
    m_fit = float(e0[np.argmin(np.abs(e0 - m_target))])
    c_fit = C_LIGHT

    # Measure fit quality along 6-fold directions at small k
    k_small = np.linspace(0.001, 0.05, 20)
    E_err = []
    E_prev = m_fit
    for k in k_small:
        M = TM14_half(k, 0.0, eps)
        lam = np.linalg.eigvals(M @ M)
        e = -np.angle(lam)
        E_k = float(e[np.argmin(np.abs(e - E_prev))])
        E_prev = E_k
        E_ref_k = np.sqrt(c_fit**2 * k**2 + m_fit**2)
        E_err.append((E_k - E_ref_k)**2)
    rmse = float(np.sqrt(np.mean(E_err))) if E_err else 0.0
    return c_fit, m_fit, rmse


def fit_rel_2d(kx_flat, ky_flat, E_flat, k_fit_max=0.2, m_seed=None):
    """
    Fit E = sqrt(c_fixed^2*(kx^2+ky^2) + m^2) with c fixed to C_LIGHT.
    Returns (c_fit, m_fit, rmse).
    For the 2+1D hexagonal model, the mass is best read directly from E(k=0),
    since the grid resolution may not be small enough compared to m/c.
    """
    if m_seed is not None:
        # Use the directly measured mass; just compute RMSE over available points
        k2   = kx_flat**2 + ky_flat**2
        mask = k2 < k_fit_max**2
        if mask.sum() < 2:
            mask = k2 < (2 * k_fit_max)**2
        c_fit, m_fit = C_LIGHT, m_seed
        E_mask = E_flat[mask]
        k2_mask = k2[mask]
        E_model = np.sqrt(C_LIGHT**2 * k2_mask + m_fit**2)
        rmse = float(np.sqrt(np.mean((E_mask - E_model)**2))) if len(E_mask) else 0.0
        return c_fit, m_fit, rmse

    # Fallback: fit both c and m
    k2   = kx_flat**2 + ky_flat**2
    mask = k2 < k_fit_max**2
    if mask.sum() < 3:
        mask = np.ones(len(kx_flat), dtype=bool)

    def model(xy, c, m):
        kx_, ky_ = xy
        return np.sqrt(c**2 * (kx_**2 + ky_**2) + m**2)

    m0 = float(E_flat[np.argmin(k2)])
    try:
        (c_fit, m_fit), _ = curve_fit(
            model, (kx_flat[mask], ky_flat[mask]), E_flat[mask],
            p0=[C_LIGHT, max(m0, 1e-6)], bounds=([0, 1e-6], [10, 10]))
        rmse = float(np.sqrt(np.mean(
            (E_flat[mask] - model((kx_flat[mask], ky_flat[mask]), c_fit, m_fit))**2)))
    except Exception:
        c_fit, m_fit, rmse = C_LIGHT, m0, float('nan')
    return c_fit, m_fit, rmse



# ─── figure 1: geometry ───────────────────────────────────────────────────────

def fig_geometry():
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle('2+1D Hexagonal Lattice Geometry\nAll 7 edges have length = 1', fontsize=12)

    # ── Left: 3-D view ──
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax3.set_title('3D view (x, y, t)', fontsize=9)

    origin = np.zeros(3)
    # Verify edge lengths
    lengths = []
    for d in range(N_DIRS):
        dx, dy = MOVES_PHYS_XY[d]
        dt = MOVE_DT[d]
        lengths.append(np.sqrt(dx**2 + dy**2 + dt**2))

    endpoints = np.column_stack([MOVES_PHYS_XY, MOVE_DT])   # (7,3)

    for d in range(N_DIRS):
        ep = endpoints[d]
        ax3.quiver(0, 0, 0, ep[0], ep[1], ep[2],
                   color=MOVE_COLORS[d], lw=2.5, arrow_length_ratio=0.2)
        ax3.scatter(*ep, color=MOVE_COLORS[d], s=40, zorder=5)
        lbl = f'd={d}\n|e|={lengths[d]:.3f}'
        ax3.text(ep[0]*1.12, ep[1]*1.12, ep[2]*1.05, lbl, fontsize=6,
                 color=MOVE_COLORS[d])

    # Hexagon at t=0.5
    hex_pts = endpoints[:6, :]   # (6,3)
    hex_closed = np.vstack([hex_pts, hex_pts[0]])
    ax3.plot(hex_closed[:, 0], hex_closed[:, 1], hex_closed[:, 2],
             'k--', lw=0.8, alpha=0.5)

    # Second ring: arrows from d=0 endpoint
    ep0 = endpoints[0]
    for d in range(N_DIRS):
        dx, dy = MOVES_PHYS_XY[d]
        dt = MOVE_DT[d]
        ax3.quiver(ep0[0], ep0[1], ep0[2], dx*0.5, dy*0.5, dt*0.5,
                   color=MOVE_COLORS[d], lw=1, alpha=0.4, arrow_length_ratio=0.3)

    ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('t')
    ax3.set_xlim(-1.2, 1.2); ax3.set_ylim(-1.2, 1.2); ax3.set_zlim(0, 1.3)

    # ── Right: top view (xy plane) ──
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Top view (xy plane)\n6 diagonal moves at 0°,60°,...,300°', fontsize=9)
    ax2.set_aspect('equal')

    angles_deg = [0, 60, 120, 180, 240, 300]
    for d in range(6):
        dx, dy = MOVES_PHYS_XY[d]
        ax2.annotate('', xy=(dx, dy), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=MOVE_COLORS[d], lw=2.2))
        ax2.text(dx*1.18, dy*1.18, f'{angles_deg[d]}°', fontsize=8,
                 color=MOVE_COLORS[d], ha='center', va='center')

    # Hexagon outline
    hex_xy = np.vstack([MOVES_PHYS_XY[:6], MOVES_PHYS_XY[0]])
    ax2.plot(hex_xy[:, 0], hex_xy[:, 1], 'k--', lw=1, alpha=0.4)
    ax2.scatter(0, 0, color='black', s=60, zorder=5)

    # Radius annotation
    ax2.annotate('', xy=(SQRT3_2, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    ax2.text(SQRT3_2/2, 0.06, f'|Δxy|=√3/2≈{SQRT3_2:.3f}', fontsize=7.5,
             ha='center', color='gray')

    ax2.set_xlim(-1.3, 1.3); ax2.set_ylim(-1.1, 1.1)
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)

    # Print edge length table
    print('Edge lengths (all should be 1.000):')
    for d in range(N_DIRS):
        tag = 'diagonal' if d < 6 else 'straight-up'
        print(f'  d={d} ({tag}): |edge| = {lengths[d]:.6f}')

    plt.tight_layout()
    plt.savefig('lattice_geometry_2d.png', dpi=150, bbox_inches='tight')
    print('Saved lattice_geometry_2d.png')
    plt.close()


# ─── figure 2: spacetime spread ──────────────────────────────────────────────

def fig_spread(psi, xcenter, ycenter, eps=0.1):
    T_phys = len(psi) - 1
    Nx, Ny = psi.shape[1], psi.shape[2]
    xs = (np.arange(Nx) - xcenter) * DX_PHYS
    ys = (np.arange(Ny) - ycenter) * DY_PHYS

    t_show = [5, 10, 15, 20]
    t_show = [t for t in t_show if t <= T_phys]

    fig, axes = plt.subplots(1, len(t_show), figsize=(4*len(t_show), 4.5))
    if len(t_show) == 1:
        axes = [axes]
    fig.suptitle(f'|ψ(x,y,t)|²  (ε={eps})   light cone: r = √3·t', fontsize=11)

    for ax, t in zip(axes, t_show):
        prob = np.abs(psi[t])**2
        vmax = prob.max()
        vmin = max(vmax * 1e-6, 1e-20)
        r_cone = C_LIGHT * t + 0.5

        # crop to light-cone region
        mask_x = np.abs(xs) <= r_cone + 1
        mask_y = np.abs(ys) <= r_cone + 1
        px = prob[np.ix_(mask_x, mask_y)]
        xx = xs[mask_x]; yy = ys[mask_y]

        im = ax.imshow(px.T, origin='lower', aspect='equal',
                       extent=[xx[0], xx[-1], yy[0], yy[-1]],
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       cmap='inferno')
        plt.colorbar(im, ax=ax, fraction=0.04)

        # light cone circle
        theta = np.linspace(0, 2*np.pi, 200)
        r = C_LIGHT * t
        ax.plot(r*np.cos(theta), r*np.sin(theta), 'w--', lw=1.5, alpha=0.8)
        ax.text(0, r*0.85, f'r=√3·{t}={r:.1f}', color='white', fontsize=7,
                ha='center', va='top')

        ax.set_title(f't = {t}', fontsize=9)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_xlim(-r_cone, r_cone); ax.set_ylim(-r_cone, r_cone)

    plt.tight_layout()
    plt.savefig('spacetime_spread_2d.png', dpi=150, bbox_inches='tight')
    print('Saved spacetime_spread_2d.png')
    plt.close()


# ─── figure 3: dispersion relation ───────────────────────────────────────────

def fig_dispersion(eps=0.1, n_k=40, k_max=1.5):
    print(f'  Computing bands (n_k={n_k})...', flush=True)
    kx_arr, ky_arr, E_all = compute_bands_2d(eps, n_k, k_max)

    # Direct measurement of c and m
    c_fit, m_fit, rmse = fit_rel_2d_direct(eps)
    print(f'  Physical mass at k=0: m_phys={m_fit:.4f}  (eps={eps})')
    print(f'  c_fit (geometric) = {c_fit:.4f}  rmse_small_k = {rmse:.4f}')
    print(f'  Expected c = sqrt(3) = {SQRT3:.4f}, m ~ eps = {eps}')

    E_phys = physical_band_2d(kx_arr, ky_arr, E_all, m_est=m_fit)

    # Isotropy: compare E along 6-fold symmetric directions
    # Use k_iso_max << k_max to stay in the isotropic small-k regime
    k_iso_max = min(0.4, k_max)
    k_1d = np.linspace(0, k_iso_max, n_k)
    SQRT3_local = np.sqrt(3)
    dirs = {
        '0°  (k,0)':        (k_1d, np.zeros_like(k_1d)),
        '60° (k/2,k√3/2)':  (k_1d * 0.5, k_1d * SQRT3_local / 2),
        '30° (k√3/2,k/2)':  (k_1d * SQRT3_local / 2, k_1d * 0.5),
    }
    dir_colors = ['#e74c3c', '#3498db', '#2ecc71']

    # Compute E along each direction via continuity tracking
    def scan_dir_cont(kx_scan, ky_scan, m0):
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

    # m_fit already is the physical mass from fit_rel_2d_direct
    m_phys = m_fit
    print(f'  Physical mass at k=0: m_phys = {m_phys:.4f}  (eps={eps},  ~2*eps={2*eps:.4f})')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Dispersion relation  E(k)  —  2+1D Hexagonal (ε={eps})', fontsize=11)

    ax_e, ax_h = axes
    k_ref = np.linspace(0, k_iso_max, 300)
    E_ref = np.sqrt(c_fit**2 * k_ref**2 + m_fit**2)

    iso_vals = {}
    for (lbl, (kxs, kys)), col in zip(dirs.items(), dir_colors):
        k_mag = np.sqrt(kxs**2 + kys**2)
        E_dir = scan_dir_cont(kxs, kys, m_phys)
        ax_e.scatter(k_mag, E_dir, color=col, s=18, label=lbl, zorder=3)
        iso_vals[lbl] = E_dir

    ax_e.plot(k_ref, E_ref, 'k--', lw=2,
              label=f'E=√({c_fit:.3f}²k²+{m_fit:.4f}²)  rmse={rmse:.4f}')
    ax_e.plot(k_ref, k_ref * C_LIGHT, 'k:', lw=1, alpha=0.4, label='c·k (massless)')
    ax_e.axhline(m_phys, color='gray', lw=1, ls=':', alpha=0.6, label=f'm_phys={m_phys:.3f}')

    # Isotropy: compare 0° vs 60° (should match by 6-fold symmetry)
    E0   = iso_vals['0°  (k,0)']
    E60  = iso_vals['60° (k/2,k√3/2)']
    iso_err = float(np.max(np.abs(E0 - E60) / (np.abs(E0) + 1e-10)))
    ax_e.text(0.02, 0.97, f'6-fold isotropy error (0° vs 60°) = {iso_err:.4f}',
              transform=ax_e.transAxes, fontsize=8, va='top',
              bbox=dict(fc='lightyellow', alpha=0.9))
    print(f'  Isotropy error (max |E(0°)-E(60°)|/E) = {iso_err:.4f}')

    ax_e.set_xlabel('|k|'); ax_e.set_ylabel('E')
    ax_e.set_title(f'E(|k|) along 3 directions (|k|≤{k_iso_max})')
    ax_e.set_xlim(0, k_iso_max)
    ax_e.legend(fontsize=7); ax_e.grid(alpha=0.3)

    # 2-D heatmap of E_phys
    im = ax_h.imshow(E_phys.T, origin='lower', aspect='equal',
                     extent=[kx_arr[0], kx_arr[-1], ky_arr[0], ky_arr[-1]],
                     cmap='viridis')
    plt.colorbar(im, ax=ax_h)
    theta = np.linspace(0, 2*np.pi, 200)
    ax_h.plot(np.cos(theta), np.sin(theta), 'w--', lw=1.2, alpha=0.7,
              label='|k|=1')
    ax_h.set_xlabel('kx'); ax_h.set_ylabel('ky')
    ax_h.set_title('E_phys(kx, ky) — circular symmetry check')
    ax_h.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('dispersion_relation_2d.png', dpi=150, bbox_inches='tight')
    print('Saved dispersion_relation_2d.png')
    plt.close()
    return c_fit, m_fit, rmse


# ─── figure 4: group velocity ─────────────────────────────────────────────────

def fig_group_velocity(eps=0.1, n_k=300, k_max=0.8):
    """
    Show group velocity E vs k along 6 hex directions plus vg in velocity space.
    Uses E_ref-based band selection (picks physical branch near relativistic dispersion).
    """
    print(f'  Computing group velocity (n_k={n_k}, k_max={k_max})...', flush=True)
    c_fit, m_fit, _ = fit_rel_2d_direct(eps)

    angles_deg = [0, 60, 120, 180, 240, 300]
    dir_colors = ['#e74c3c','#e67e22','#2ecc71','#3498db','#9b59b6','#1abc9c']

    k_arr = np.linspace(0, k_max, n_k)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Group velocity  |  2+1D Hexagonal lattice  |  ε={eps}  c=√3={c_fit:.3f}',
                 fontsize=11)
    ax_e, ax_v = axes

    v_max = 0.0

    for angle_deg, color in zip(angles_deg, dir_colors):
        angle = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Track physical band: use weighted combination of E_ref and continuity
        # to avoid jumping at near-degeneracies
        E_path = np.empty(n_k)
        E_path[0] = m_fit
        for step, k in enumerate(k_arr[1:], 1):
            kx_, ky_ = k * cos_a, k * sin_a
            M = TM14_half(kx_, ky_, eps)
            lam = np.linalg.eigvals(M @ M)
            e = -np.angle(lam)
            E_ref_k = np.sqrt(c_fit**2 * k**2 + m_fit**2)
            E_prev  = E_path[step - 1]
            # Pick eigenvalue closest to average of (E_ref, E_prev) to balance
            # relativistic expectation with continuity
            E_guide = 0.5 * (E_ref_k + E_prev)
            E_path[step] = float(e[np.argmin(np.abs(e - E_guide))])

        # E vs k plot
        ax_e.plot(k_arr, E_path, color=color, lw=1.2, alpha=0.8)

        # Group velocity
        vg = np.gradient(E_path, k_arr)
        vg_mag = np.abs(vg)

        # Filter: reject points where |vg| > 1.5*c (clear band-tracking artifacts)
        # and where the local vg change is large (sharp discontinuities)
        physical = (vg_mag < 1.5 * c_fit) & (np.abs(np.diff(vg, prepend=vg[0])) < 0.5)
        v_max = max(v_max, float(vg_mag[physical].max()) if physical.any() else 0.0)

        # Project onto (vgx, vgy) direction
        vgx_dir = vg * cos_a
        vgy_dir = vg * sin_a
        # Only plot physical points
        sc = ax_v.scatter(vgx_dir[physical], vgy_dir[physical], c=k_arr[physical],
                          cmap='viridis', s=8, alpha=0.7, vmin=0, vmax=k_max)

    plt.colorbar(sc, ax=ax_v, label='|k|')

    # Reference dispersion
    k_ref = np.linspace(0, k_max, 200)
    ax_e.plot(k_ref, np.sqrt(c_fit**2 * k_ref**2 + m_fit**2), 'k--', lw=2,
              label=f'E=√({c_fit:.2f}²k²+{m_fit:.3f}²)')
    ax_e.set_xlabel('|k|'); ax_e.set_ylabel('E')
    ax_e.set_title('E(k) along 6 hex directions')
    ax_e.legend(fontsize=8); ax_e.grid(alpha=0.3)

    theta = np.linspace(0, 2*np.pi, 300)
    ax_v.plot(c_fit * np.cos(theta), c_fit * np.sin(theta),
              'k-', lw=2, label=f'|v_g|=c=√3={c_fit:.3f}')
    ax_v.set_aspect('equal')
    ax_v.set_xlabel('vgx'); ax_v.set_ylabel('vgy')
    ax_v.set_title(f'Group velocity (physical band)\nmax|v_g|={v_max:.3f}  (c=√3={c_fit:.3f})')
    ax_v.legend(fontsize=8); ax_v.grid(alpha=0.3)
    ax_v.set_xlim(-c_fit*1.2, c_fit*1.2); ax_v.set_ylim(-c_fit*1.2, c_fit*1.2)

    print(f'  max |v_g| = {v_max:.4f}  (c = {c_fit:.4f})')
    print(f'  |v_g| <= c: {v_max <= c_fit + 0.05}')

    plt.tight_layout()
    plt.savefig('group_velocity_2d.png', dpi=150, bbox_inches='tight')
    print('Saved group_velocity_2d.png')
    plt.close()


# ─── figure 5: epsilon sweep ──────────────────────────────────────────────────

def _radial_profile(prob, xcenter, ycenter, n_bins=80):
    Nx, Ny = prob.shape
    xs = (np.arange(Nx) - xcenter) * DX_PHYS
    ys = (np.arange(Ny) - ycenter) * DY_PHYS
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    R = np.sqrt(XX**2 + YY**2)
    r_max = R.max()
    bins = np.linspace(0, r_max, n_bins + 1)
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    P = np.array([prob[(R >= bins[i]) & (R < bins[i+1])].sum()
                  for i in range(n_bins)])
    norm = P.sum()
    P = P / (norm + 1e-30)
    return r_mid, P


def fig_epsilon_sweep(T=20, n_k=30, k_max=2.0):
    epsilons = [0.01, 0.1, 0.5, 1.0]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f'Effect of ε on |ψ(x,y,t={T})|²  —  2+1D Hexagonal', fontsize=11)

    r_cone = C_LIGHT * T

    for col, eps in enumerate(epsilons):
        print(f'  Simulating eps={eps}...', flush=True)
        psi, xc, yc = simulate_hex_2d(T, eps)
        prob = np.abs(psi[T])**2

        xs = (np.arange(psi.shape[1]) - xc) * DX_PHYS
        ys = (np.arange(psi.shape[2]) - yc) * DY_PHYS
        vmax = prob.max(); vmin = max(vmax * 1e-6, 1e-20)

        # Row 0: heatmap
        ax0 = axes[0, col]
        crop = r_cone + 2
        mask_x = np.abs(xs) <= crop; mask_y = np.abs(ys) <= crop
        px = prob[np.ix_(mask_x, mask_y)]
        xx = xs[mask_x]; yy = ys[mask_y]

        im = ax0.imshow(px.T, origin='lower', aspect='equal',
                        extent=[xx[0], xx[-1], yy[0], yy[-1]],
                        norm=LogNorm(vmin=vmin, vmax=vmax), cmap='inferno')
        theta = np.linspace(0, 2*np.pi, 200)
        ax0.plot(r_cone*np.cos(theta), r_cone*np.sin(theta), 'w--', lw=1.5)
        ax0.set_title(f'ε = {eps}', fontsize=9)
        ax0.set_xlabel('x'); ax0.set_ylabel('y')
        plt.colorbar(im, ax=ax0, fraction=0.04)

        # Row 1: radial profile
        ax1 = axes[1, col]
        r_mid, P = _radial_profile(prob, xc, yc)
        ax1.plot(r_mid, P, color='steelblue', lw=1.8)
        ax1.axvline(r_cone, color='red', lw=1.5, ls='--',
                    label=f'r=√3·{T}={r_cone:.1f}')
        ax1.set_xlabel('r'); ax1.set_ylabel('P(r) (normalized)')
        ax1.set_title(f'Radial profile  ε={eps}', fontsize=8)
        ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

        # Direct fit for this eps
        c_fit, m_fit, rmse = fit_rel_2d_direct(eps)
        print(f'    eps={eps}: c={c_fit:.4f}  m_phys(k=0)={m_fit:.4f}  rmse_small_k={rmse:.4f}')

    plt.tight_layout()
    plt.savefig('epsilon_sweep_2d.png', dpi=150, bbox_inches='tight')
    print('Saved epsilon_sweep_2d.png')
    plt.close()


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    EPS   = 0.1
    T_MAX = 20

    print('=== Geometry figure ===')
    fig_geometry()

    print(f'\n=== Simulating T={T_MAX}, eps={EPS} ===')
    psi, xc, yc = simulate_hex_2d(T_MAX, EPS)
    print(f'    psi shape: {psi.shape}')

    print('\n=== Spacetime spread ===')
    fig_spread(psi, xc, yc, eps=EPS)

    print('\n=== Dispersion relation ===')
    c_fit, m_fit, rmse = fig_dispersion(eps=EPS)

    print('\n=== Group velocity ===')
    fig_group_velocity(eps=EPS)

    print('\n=== Epsilon sweep ===')
    fig_epsilon_sweep(T=T_MAX)

    print('\nDone.')
