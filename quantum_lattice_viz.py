"""
Three lattice models for the discrete path integral — compared geometrically
and dynamically.

1. Feynman Checkerboard     square lattice,   2 moves: (±1, +1)
2. Equilateral Triangular   rotated 30° grid, 3 moves: (±√3/2, +1/2) and (0, +1)
                            → all edges length 1, one edge points straight up
3. Square + Rest            rectangular grid, 3 moves: (-1,+1), (0,+1), (+1,+1)

Amplitude rule: same direction → ×1, direction change → ×(iε).

Key insight (Equilateral vs Square+Rest):
  The 3-move equilateral lattice has the same DP graph structure as Square+Rest.
  Using step count as 'time', their probability distributions are identical up to
  x-axis rescaling by √3/2. The only physical difference is that on the equilateral
  lattice, diagonal steps advance physical time by 1/2 while the straight-up step
  advances it by 1 — an inherent time asymmetry.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SQRT3_2 = np.sqrt(3) / 2   # ≈ 0.866


# ═══════════════════════════════════════════════════════════════════════════
#  Simulators
# ═══════════════════════════════════════════════════════════════════════════

def _change_matrix(n_dirs, epsilon):
    """n × n: diagonal = 1 (same dir), off-diagonal = iε (direction change)."""
    C = np.full((n_dirs, n_dirs), 1j * epsilon, dtype=complex)
    np.fill_diagonal(C, 1.0 + 0j)
    return C


def simulate_checkerboard(T, epsilon=0.1):
    """
    Feynman Checkerboard: 2 moves (dx,dt) = (±1, +1).
    Returns psi[T+1, N], N=2T+1.  Physical x[i] = i - T.
    """
    N, center = 2 * T + 1, T
    C = _change_matrix(2, epsilon)
    amp = np.zeros((N, 2), dtype=complex)
    amp[center, :] = 0.5
    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0] = amp.sum(axis=1)
    for step in range(1, T + 1):
        new_amp = np.zeros((N, 2), dtype=complex)
        w = amp @ C[0];  new_amp[:-1, 0] += w[1:]    # dx=-1: shift left
        w = amp @ C[1];  new_amp[1:,  1] += w[:-1]   # dx=+1: shift right
        amp = new_amp
        psi[step] = amp.sum(axis=1)
    return psi    # x[i] = i - T,  Δx = 1


def simulate_equilateral(T, epsilon=0.1):
    """
    Equilateral triangular lattice, rotated 30°.

    Three forward moves in physical (x, t_phys):
      d=0  left-diagonal:  (dx, dt_phys) = (-√3/2, +1/2)
      d=1  straight up:    (dx, dt_phys) = (   0,  +1  )
      d=2  right-diagonal: (dx, dt_phys) = (+√3/2, +1/2)
    All edges have length 1: ||(±√3/2, 1/2)|| = ||(0, 1)|| = 1.

    Simulation uses step count as 'time' (one step = one lattice edge).
    Under this convention the DP is structurally identical to Square+Rest;
    x[i] = (i - T) * √3/2  (spacing √3/2 ≈ 0.866 instead of 1).
    """
    N, center = 2 * T + 1, T
    C = _change_matrix(3, epsilon)
    amp = np.zeros((N, 3), dtype=complex)
    amp[center, :] = 1.0 / 3.0
    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0] = amp.sum(axis=1)
    for step in range(1, T + 1):
        new_amp = np.zeros((N, 3), dtype=complex)
        w = amp @ C[0];  new_amp[:-1, 0] += w[1:]    # d=0: dx=-√3/2 → shift left
        w = amp @ C[1];  new_amp[:, 1]   += w         # d=1: dx=0     → no shift
        w = amp @ C[2];  new_amp[1:,  2] += w[:-1]   # d=2: dx=+√3/2 → shift right
        amp = new_amp
        psi[step] = amp.sum(axis=1)
    return psi    # x[i] = (i - T) * √3/2,  Δx = √3/2


def simulate_square_rest(T, epsilon=0.1):
    """
    Square lattice + rest: 3 moves (dx,dt) ∈ {(-1,+1),(0,+1),(+1,+1)}.
    Returns psi[T+1, N], N=2T+1.  Physical x[i] = i - T.
    """
    N, center = 2 * T + 1, T
    C = _change_matrix(3, epsilon)
    amp = np.zeros((N, 3), dtype=complex)
    amp[center, :] = 1.0 / 3.0
    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0] = amp.sum(axis=1)
    for step in range(1, T + 1):
        new_amp = np.zeros((N, 3), dtype=complex)
        w = amp @ C[0];  new_amp[:-1, 0] += w[1:]
        w = amp @ C[1];  new_amp[:, 1]   += w
        w = amp @ C[2];  new_amp[1:,  2] += w[:-1]
        amp = new_amp
        psi[step] = amp.sum(axis=1)
    return psi    # x[i] = i - T,  Δx = 1


# ═══════════════════════════════════════════════════════════════════════════
#  Physical x-axis helpers
# ═══════════════════════════════════════════════════════════════════════════

def xs_checkerboard(T):
    return (np.arange(2 * T + 1) - T).astype(float)       # Δx = 1

def xs_equilateral(T):
    return (np.arange(2 * T + 1) - T) * SQRT3_2           # Δx = √3/2

def xs_square_rest(T):
    return (np.arange(2 * T + 1) - T).astype(float)       # Δx = 1


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1 — Lattice geometry (correct visual proportions)
# ═══════════════════════════════════════════════════════════════════════════

def _arrow(ax, x0, y0, x1, y1, color, lw=1.4):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, shrinkA=4, shrinkB=4))


def draw_checkerboard_panel(ax, T_draw=5):
    """Feynman Checkerboard — square grid, diagonal moves only."""
    ax.set_title('Feynman Checkerboard\n2 moves: (±1, +1)  |edge| = √2', fontsize=9)
    for t in range(T_draw + 1):
        for x in range(-T_draw, T_draw + 1):
            ax.plot(x, t, 'o', color='#aaaaaa', ms=3, zorder=2)
    ax.plot(0, 0, 'ko', ms=6, zorder=3)
    for (x0, t0) in [(0,0), (-1,1), (1,1), (0,2), (-1,3), (1,3)]:
        _arrow(ax, x0, t0, x0-1, t0+1, '#e74c3c')
        _arrow(ax, x0, t0, x0+1, t0+1, '#3498db')
    ax.legend(handles=[mpatches.Patch(color='#e74c3c', label='(-1,+1)'),
                       mpatches.Patch(color='#3498db', label='(+1,+1)')],
              fontsize=7, loc='upper right')
    ax.set_xlim(-T_draw-0.5, T_draw+0.5);  ax.set_ylim(-0.5, T_draw+0.4)
    ax.set_xlabel('x');  ax.set_ylabel('t');  ax.set_aspect('equal')
    ax.text( 0.6,  0.52, '√2', fontsize=7.5, color='#3498db', rotation=45)
    ax.text(-1.05, 0.52, '√2', fontsize=7.5, color='#e74c3c', rotation=-45)


def draw_equilateral_panel(ax, T_draw=5):
    """
    Equilateral triangular lattice, rotated 30°.
    Nodes at (n·√3/2, m/2) for integers n,m with n+m even.
    Three forward moves: (±√3/2, +1/2) and (0, +1) — all length 1.
    One edge points exactly straight up.
    """
    ax.set_title('Equilateral Triangular (rotated 30°)\n'
                 '3 moves: (±√3/2, +½), (0, +1)  |edge| = 1', fontsize=9)

    # Build node set: (n, m) with n+m even, m in [0, 2*T_draw]
    T2 = 2 * T_draw
    node_set = set()
    for m in range(0, T2 + 1):
        for n in range(-T2, T2 + 1):
            if (n + m) % 2 == 0:
                node_set.add((n, m))

    def pos(n, m):
        return n * SQRT3_2, m * 0.5

    # Draw edges first (thin gray)
    for (n, m) in node_set:
        for dn, dm in [(1, 1), (-1, 1), (0, 2)]:   # 3 forward directions
            nb = (n + dn, m + dm)
            if nb in node_set:
                x0, y0 = pos(n, m)
                x1, y1 = pos(*nb)
                ax.plot([x0, x1], [y0, y1], '-', color='#dddddd', lw=0.7, zorder=1)

    # Draw nodes
    for (n, m) in node_set:
        x, y = pos(n, m)
        ax.plot(x, y, 'o', color='#aaaaaa', ms=3, zorder=2)
    ax.plot(0, 0, 'ko', ms=6, zorder=3)

    # Arrows from selected nodes
    sample_nm = [(0, 0), (0, 2), (0, 4), (-2, 2), (2, 2), (-1, 1), (1, 1)]
    for (n0, m0) in sample_nm:
        if (n0, m0) not in node_set:
            continue
        x0, y0 = pos(n0, m0)
        _arrow(ax, x0, y0, x0 - SQRT3_2, y0 + 0.5, '#e74c3c')  # left
        _arrow(ax, x0, y0, x0,           y0 + 1.0, '#2ecc71')  # straight up
        _arrow(ax, x0, y0, x0 + SQRT3_2, y0 + 0.5, '#3498db')  # right

    ax.legend(handles=[mpatches.Patch(color='#e74c3c', label='(-√3/2, +½)'),
                       mpatches.Patch(color='#2ecc71', label='(0, +1)'),
                       mpatches.Patch(color='#3498db', label='(+√3/2, +½)')],
              fontsize=7, loc='upper right')

    xlim = (T_draw + 1) * SQRT3_2
    ax.set_xlim(-xlim, xlim);  ax.set_ylim(-0.5, T_draw + 0.4)
    ax.set_xlabel('x');  ax.set_ylabel('t_phys');  ax.set_aspect('equal')

    # annotate edge lengths
    ax.text(0.12,  0.65, '1', fontsize=8, color='#2ecc71')
    ax.text(0.5,   0.18, '1', fontsize=8, color='#3498db', rotation=30)
    ax.text(-0.72, 0.18, '1', fontsize=8, color='#e74c3c', rotation=-30)


def draw_square_rest_panel(ax, T_draw=5):
    """Square + Rest — rectangular grid, 3 moves including straight up."""
    ax.set_title('Square + Rest move\n3 moves: (-1,+1), (0,+1), (+1,+1)', fontsize=9)
    for t in range(T_draw + 1):
        for x in range(-T_draw, T_draw + 1):
            ax.plot(x, t, 'o', color='#aaaaaa', ms=3, zorder=2)
    ax.plot(0, 0, 'ko', ms=6, zorder=3)
    for (x0, t0) in [(0,0), (-1,1), (1,1), (0,2), (-1,3), (1,3)]:
        _arrow(ax, x0, t0, x0-1, t0+1, '#e74c3c')
        _arrow(ax, x0, t0, x0,   t0+1, '#2ecc71')
        _arrow(ax, x0, t0, x0+1, t0+1, '#3498db')
    ax.legend(handles=[mpatches.Patch(color='#e74c3c', label='(-1,+1)'),
                       mpatches.Patch(color='#2ecc71', label='( 0,+1)'),
                       mpatches.Patch(color='#3498db', label='(+1,+1)')],
              fontsize=7, loc='upper right')
    ax.set_xlim(-T_draw-0.5, T_draw+0.5);  ax.set_ylim(-0.5, T_draw+0.4)
    ax.set_xlabel('x');  ax.set_ylabel('t');  ax.set_aspect('equal')
    ax.text( 0.1,  0.62, '1',  fontsize=7.5, color='#2ecc71')
    ax.text( 0.55, 0.48, '√2', fontsize=7.5, color='#3498db', rotation=45)
    ax.text(-1.05, 0.48, '√2', fontsize=7.5, color='#e74c3c', rotation=-45)


def fig_geometry():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        'Gittergeometrie: Kantenlängen und Bewegungsrichtungen\n'
        'Nur das gleichseitige Dreiecksgitter hat alle Kanten gleich lang '
        'und einen geraden Aufwärts-Move',
        fontsize=11)
    draw_checkerboard_panel(axes[0])
    draw_equilateral_panel(axes[1])
    draw_square_rest_panel(axes[2])
    plt.tight_layout()
    plt.savefig('lattice_geometry.png', dpi=150, bbox_inches='tight')
    print('Saved lattice_geometry.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 2 — Probability heatmaps + 1-D slices
# ═══════════════════════════════════════════════════════════════════════════

def fig_comparison(T=40, epsilon=0.1):
    print(f'Simulating (T={T}, eps={epsilon})...', flush=True)
    psi_C = simulate_checkerboard(T, epsilon)
    psi_E = simulate_equilateral(T, epsilon)
    psi_S = simulate_square_rest(T, epsilon)

    prob_C = np.abs(psi_C) ** 2
    prob_E = np.abs(psi_E) ** 2
    prob_S = np.abs(psi_S) ** 2

    xC = xs_checkerboard(T)
    xE = xs_equilateral(T)
    xS = xs_square_rest(T)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f'Drei Modelle — |ψ(x,t)|²   (T={T}, ε={epsilon})\n'
        'Equilateral und Square+Rest haben gleiche DP-Struktur '
        '(identische Verteilungen bis auf x-Skalierung ×√3/2)',
        fontsize=11)

    titles = [
        'Feynman Checkerboard\n(±1, +1)',
        'Equilateral (rotiert 30°)\n(±√3/2, +½) und (0, +1)',
        'Square + Rest\n(-1/0/+1, +1)'
    ]
    kw = dict(aspect='auto', origin='lower', cmap='inferno')

    for col, (prob, xs, ttl) in enumerate(
            zip([prob_C, prob_E, prob_S], [xC, xE, xS], titles)):
        ext = [xs[0]-abs(xs[1]-xs[0])/2, xs[-1]+abs(xs[1]-xs[0])/2, -0.5, T+0.5]
        im = axes[0, col].imshow(prob, extent=ext, **kw)
        axes[0, col].set_title(ttl);  axes[0, col].set_xlabel('x')
        axes[0, col].set_ylabel('t (step count)')
        plt.colorbar(im, ax=axes[0, col])

    slices_t = [T // 4, T // 2, 3 * T // 4, T]
    colors_sl = plt.cm.plasma(np.linspace(0.2, 0.9, len(slices_t)))
    for t_idx, col in zip(slices_t, colors_sl):
        axes[1, 0].plot(xC, prob_C[t_idx], color=col, label=f't={t_idx}')
        axes[1, 1].plot(xE, prob_E[t_idx], color=col, label=f't={t_idx}')
        axes[1, 2].plot(xS, prob_S[t_idx], color=col, label=f't={t_idx}')

    for ax_col, ttl in zip(axes[1], titles):
        ax_col.set_xlabel('x (physikalische Einheiten)')
        ax_col.set_ylabel('|ψ|²')
        ax_col.set_title(ttl)
        ax_col.legend(fontsize=7)
        ax_col.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print('Saved model_comparison.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 3 — Spread σ(t) for all three models
# ═══════════════════════════════════════════════════════════════════════════

def fig_spread(T=50, epsilon=0.1):
    print(f'Simulating spread (T={T}, eps={epsilon})...', flush=True)
    psi_C = simulate_checkerboard(T, epsilon)
    psi_E = simulate_equilateral(T, epsilon)
    psi_S = simulate_square_rest(T, epsilon)

    t_arr = np.arange(T + 1)

    def spread(psi, xs):
        sigmas = []
        for row in psi:
            p = np.abs(row) ** 2
            xs_row = xs[:len(p)]
            norm = p.sum()
            if norm < 1e-30:
                sigmas.append(0.0)
            else:
                mu  = np.dot(p, xs_row) / norm
                mu2 = np.dot(p, xs_row ** 2) / norm
                sigmas.append(np.sqrt(max(mu2 - mu**2, 0)))
        return np.array(sigmas)

    sig_C = spread(psi_C, xs_checkerboard(T))
    sig_E = spread(psi_E, xs_equilateral(T))
    sig_S = spread(psi_S, xs_square_rest(T))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f'Ausbreitung σ(t) — alle drei Modelle  (ε={epsilon})', fontsize=11)
    ax.plot(t_arr, sig_C, color='#3498db', lw=2,
            label='Feynman Checkerboard (±1)')
    ax.plot(t_arr, sig_E, color='#e67e22', lw=2, ls='--',
            label='Equilateral rotiert (±√3/2, 0)  [×√3/2 ≈ 0.87 vs Sq+R]')
    ax.plot(t_arr, sig_S, color='#2ecc71', lw=2,
            label='Square + Rest (-1/0/+1)')
    # ballistic reference
    ref = t_arr * (sig_C[-1] / (T + 1e-9))
    ax.plot(t_arr, ref, 'k--', alpha=0.35, label='∝ t (ballistisch)')
    ax.set_xlabel('t (Schrittzahl)');  ax.set_ylabel('σ(x)')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # ratio annotation
    ratio = sig_E[-1] / (sig_S[-1] + 1e-15)
    ax.text(0.02, 0.97,
            f'σ_equilateral / σ_square+rest ≈ {ratio:.4f}  (erwartet: √3/2 = {SQRT3_2:.4f})',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig('lattice_spread.png', dpi=150, bbox_inches='tight')
    print('Saved lattice_spread.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=== Gittergeometrie ===')
    fig_geometry()

    print('\n=== Wahrscheinlichkeitsvergleich (T=40) ===')
    fig_comparison(T=40, epsilon=0.1)

    print('\n=== Ausbreitung (T=50) ===')
    fig_spread(T=50, epsilon=0.1)

    print('\nFertig.')
