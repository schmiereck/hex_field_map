"""
Three lattice models for the discrete path integral — compared geometrically
and dynamically.

1. Feynman Checkerboard  (square lattice, 2 moves: ±1 in x, +1 in t)
2. Equilateral Triangular (staggered grid,  2 moves: ±½ in x, +1 in t, |edge|=1)
3. Square + Rest          (rectangular grid, 3 moves: -1/0/+1 in x, +1 in t)

Amplitude rule everywhere: same direction → ×1, direction change → ×(iε).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SQRT3_2 = np.sqrt(3) / 2   # vertical spacing for equilateral triangles


# ═══════════════════════════════════════════════════════════════════════════
#  Simulators
# ═══════════════════════════════════════════════════════════════════════════

def _change_matrix(n_dirs, epsilon):
    """n_dirs × n_dirs matrix: 1 on diagonal, iε off-diagonal."""
    C = np.full((n_dirs, n_dirs), 1j * epsilon, dtype=complex)
    np.fill_diagonal(C, 1.0 + 0j)
    return C


def simulate_checkerboard(T, epsilon=0.1):
    """
    Feynman Checkerboard: moves (dx,dt) ∈ {(-1,+1),(+1,+1)}.
    Returns psi[T+1, N], N=2T+1.  x[i] = i - T.
    """
    N = 2 * T + 1
    center = T
    C = _change_matrix(2, epsilon)   # 2×2

    amp = np.zeros((N, 2), dtype=complex)
    amp[center, :] = 0.5

    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0] = amp.sum(axis=1)

    for _ in range(1, T + 1):
        new_amp = np.zeros((N, 2), dtype=complex)
        # d=0: dx=-1 → shift left
        w = amp @ C[0]
        new_amp[:-1, 0] += w[1:]
        # d=1: dx=+1 → shift right
        w = amp @ C[1]
        new_amp[1:, 1] += w[:-1]
        amp = new_amp
        psi[_] = amp.sum(axis=1)

    return psi          # x[i] = i - center,  Δx = 1


def simulate_equilateral(T, epsilon=0.1):
    """
    Equilateral triangular lattice: staggered grid.
    Moves: (-½,+1) and (+½,+1)  — all edges length 1 when Δt = √3/2.

    Indexing: amp[i, d] at time t.
      t even : physical x = (i - center)          (integer)
      t odd  : physical x = (i - center) + ½      (half-integer)

    Shift rule:
      t even → t+1 odd:  left d=0 shifts i-1,  right d=1 keeps i
      t odd  → t+2 even: left d=0 keeps i,      right d=1 shifts i+1
    """
    N = 2 * T + 3      # small margin
    center = T + 1
    C = _change_matrix(2, epsilon)

    amp = np.zeros((N, 2), dtype=complex)
    amp[center, :] = 0.5

    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0] = amp.sum(axis=1)

    for step in range(1, T + 1):
        new_amp = np.zeros((N, 2), dtype=complex)
        t_parity = (step - 1) % 2   # parity of the time slice we're leaving

        if t_parity == 0:            # even → odd
            w0 = amp @ C[0];  new_amp[:-1, 0] += w0[1:]   # left: i→i-1
            w1 = amp @ C[1];  new_amp[:, 1]   += w1        # right: i→i
        else:                        # odd → even
            w0 = amp @ C[0];  new_amp[:, 0]   += w0        # left: i→i
            w1 = amp @ C[1];  new_amp[1:, 1]  += w1[:-1]  # right: i→i+1

        amp = new_amp
        psi[step] = amp.sum(axis=1)

    return psi          # x[i] = (i - center) + (t%2)*0.5,  Δx = 0.5


def simulate_square_rest(T, epsilon=0.1):
    """
    Square lattice + rest move: 3 moves (dx,dt) ∈ {(-1,+1),(0,+1),(+1,+1)}.
    Returns psi[T+1, N], N=2T+1.  x[i] = i - T.
    """
    N = 2 * T + 1
    center = T
    C = _change_matrix(3, epsilon)

    amp = np.zeros((N, 3), dtype=complex)
    amp[center, :] = 1.0 / 3.0

    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0] = amp.sum(axis=1)

    for step in range(1, T + 1):
        new_amp = np.zeros((N, 3), dtype=complex)
        # d=0: dx=-1
        w = amp @ C[0];  new_amp[:-1, 0] += w[1:]
        # d=1: dx= 0
        w = amp @ C[1];  new_amp[:, 1]   += w
        # d=2: dx=+1
        w = amp @ C[2];  new_amp[1:, 2]  += w[:-1]
        amp = new_amp
        psi[step] = amp.sum(axis=1)

    return psi          # x[i] = i - center,  Δx = 1


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: x-axis arrays in physical units
# ═══════════════════════════════════════════════════════════════════════════

def xs_checkerboard(T):
    return np.arange(2 * T + 1) - T           # integers, Δx=1

def xs_equilateral(T, t):
    """Physical x at time step t for equilateral model."""
    N = 2 * T + 3
    center = T + 1
    offset = 0.5 * (t % 2)
    return (np.arange(N) - center + offset) * 0.5   # half-integers or integers

def xs_square_rest(T):
    return np.arange(2 * T + 1) - T           # integers, Δx=1


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1 — Lattice geometry (correct visual proportions)
# ═══════════════════════════════════════════════════════════════════════════

def _arrow(ax, x0, y0, x1, y1, color, lw=1.3):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, shrinkA=4, shrinkB=4))


def draw_checkerboard_panel(ax, T_draw=5):
    """Square lattice, moves (±1,+1). All edges length √2."""
    ax.set_title('Feynman Checkerboard\n2 moves: (±1,+1), Δx=Δt=1', fontsize=9)
    for t in range(T_draw + 1):
        for x in range(-T_draw, T_draw + 1):
            ax.plot(x, t, 'o', color='#aaaaaa', ms=3, zorder=2)
    ax.plot(0, 0, 'ko', ms=6, zorder=3)

    samples = [(0,0),(-1,1),(1,1),(0,2),(-1,3),(1,3)]
    for (x0, t0) in samples:
        _arrow(ax, x0, t0, x0-1, t0+1, '#e74c3c')
        _arrow(ax, x0, t0, x0+1, t0+1, '#3498db')

    patches = [mpatches.Patch(color='#e74c3c', label='(-1,+1)'),
               mpatches.Patch(color='#3498db', label='(+1,+1)')]
    ax.legend(handles=patches, fontsize=7, loc='upper right')
    ax.set_xlim(-T_draw-0.5, T_draw+0.5)
    ax.set_ylim(-0.3, T_draw+0.3)
    ax.set_xlabel('x');  ax.set_ylabel('t')
    ax.set_aspect('equal')

    # annotate edge lengths
    ax.text(0.6, 0.55, '√2', fontsize=7.5, color='#3498db', rotation=45)
    ax.text(-0.95, 0.55, '√2', fontsize=7.5, color='#e74c3c', rotation=-45)


def draw_equilateral_panel(ax, T_draw=5):
    """Staggered grid, moves (±½,+1) drawn at Δt=√3/2 so all edges = 1."""
    ax.set_title('Equilateral Triangular\n2 moves: (±½,+1), |edge|=1', fontsize=9)
    dt = SQRT3_2   # visual row spacing

    for t in range(T_draw + 1):
        offset = 0.5 if t % 2 else 0.0
        x_range = np.arange(-T_draw, T_draw + 1) + offset
        for x in x_range:
            ax.plot(x, t * dt, 'o', color='#aaaaaa', ms=3, zorder=2)
        # draw horizontal edges within the row
        for x in x_range[:-1]:
            ax.plot([x, x+1], [t*dt, t*dt], '-', color='#dddddd', lw=0.5, zorder=1)

    # draw diagonal edges (all edges)
    for t in range(T_draw):
        offset = 0.5 if t % 2 else 0.0
        x_range = np.arange(-T_draw, T_draw + 1) + offset
        for x in x_range:
            # left move
            ax.plot([x, x-0.5], [t*dt, (t+1)*dt], '-', color='#dddddd', lw=0.5, zorder=1)
            # right move
            ax.plot([x, x+0.5], [t*dt, (t+1)*dt], '-', color='#dddddd', lw=0.5, zorder=1)

    # origin
    ax.plot(0, 0, 'ko', ms=6, zorder=3)

    # arrows from a few nodes
    samples_even = [(0,0),(-1,2),(1,2)]
    samples_odd  = [(-0.5,1),(0.5,1),(-0.5,3),(0.5,3)]
    for (x0, t0) in samples_even + samples_odd:
        _arrow(ax, x0, t0*dt, x0-0.5, (t0+1)*dt, '#e74c3c')
        _arrow(ax, x0, t0*dt, x0+0.5, (t0+1)*dt, '#3498db')

    patches = [mpatches.Patch(color='#e74c3c', label='(-½,+1)'),
               mpatches.Patch(color='#3498db', label='(+½,+1)')]
    ax.legend(handles=patches, fontsize=7, loc='upper right')
    ax.set_xlim(-T_draw-0.5, T_draw+0.5)
    ax.set_ylim(-0.3, T_draw*dt+0.3)
    ax.set_xlabel('x');  ax.set_ylabel('t  (unit: √3/2)')
    ax.set_aspect('equal')

    # annotate one edge
    ax.text(0.35, dt*0.45, '1', fontsize=8, color='#3498db', rotation=60)
    ax.text(-0.55, dt*0.45, '1', fontsize=8, color='#e74c3c', rotation=-60)


def draw_square_rest_panel(ax, T_draw=5):
    """Rectangular grid, 3 moves: (-1,+1),(0,+1),(+1,+1)."""
    ax.set_title('Square + Rest move\n3 moves: (-1,+1),(0,+1),(+1,+1)', fontsize=9)
    for t in range(T_draw + 1):
        for x in range(-T_draw, T_draw + 1):
            ax.plot(x, t, 'o', color='#aaaaaa', ms=3, zorder=2)
    ax.plot(0, 0, 'ko', ms=6, zorder=3)

    samples = [(0,0),(-1,1),(1,1),(0,2),(-1,3),(1,3)]
    for (x0, t0) in samples:
        _arrow(ax, x0, t0, x0-1, t0+1, '#e74c3c')
        _arrow(ax, x0, t0, x0,   t0+1, '#2ecc71')
        _arrow(ax, x0, t0, x0+1, t0+1, '#3498db')

    patches = [mpatches.Patch(color='#e74c3c', label='(-1,+1)'),
               mpatches.Patch(color='#2ecc71', label='( 0,+1)'),
               mpatches.Patch(color='#3498db', label='(+1,+1)')]
    ax.legend(handles=patches, fontsize=7, loc='upper right')
    ax.set_xlim(-T_draw-0.5, T_draw+0.5)
    ax.set_ylim(-0.3, T_draw+0.3)
    ax.set_xlabel('x');  ax.set_ylabel('t')
    ax.set_aspect('equal')

    ax.text(0.1, 0.6, '1', fontsize=7.5, color='#2ecc71')
    ax.text(0.55, 0.48, '√2', fontsize=7.5, color='#3498db', rotation=45)
    ax.text(-1.0, 0.48, '√2', fontsize=7.5, color='#e74c3c', rotation=-45)


def fig_geometry():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        'Lattice geometry: edge lengths and move structure\n'
        'Only the equilateral triangular lattice has all edges equal',
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
    print(f'Simulating (T={T}, ε={epsilon})…', flush=True)
    psi_C = simulate_checkerboard(T, epsilon)
    psi_E = simulate_equilateral(T, epsilon)
    psi_S = simulate_square_rest(T, epsilon)

    prob_C = np.abs(psi_C) ** 2
    prob_E = np.abs(psi_E) ** 2
    prob_S = np.abs(psi_S) ** 2

    # x-axes for 1-D slices
    xC = xs_checkerboard(T)                        # integers
    xS = xs_square_rest(T)
    # for equilateral, use x at the last time step (t=T)
    xE = xs_equilateral(T, T)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f'Three models  —  |ψ(x,t)|²    (T={T}, ε={epsilon})',
        fontsize=12)

    titles = ['Feynman Checkerboard\n(±1, +1)',
              'Equilateral Triangular\n(±½, +1)',
              'Square + Rest\n(-1/0/+1, +1)']

    # ── Row 0: heatmaps ────────────────────────────────────────────────────
    kw = dict(aspect='auto', origin='lower', cmap='inferno')

    # Checkerboard
    extC = [xC[0]-0.5, xC[-1]+0.5, -0.5, T+0.5]
    im0 = axes[0, 0].imshow(prob_C, extent=extC, **kw)
    axes[0, 0].set_title(titles[0]);  axes[0, 0].set_xlabel('x');  axes[0, 0].set_ylabel('t')
    plt.colorbar(im0, ax=axes[0, 0])

    # Equilateral — x-axis covers (i-center)*0.5 range
    xE_all = xs_equilateral(T, T)   # use last row for extent
    extE = [xE_all[0]-0.25, xE_all[-1]+0.25, -0.5, T+0.5]
    im1 = axes[0, 1].imshow(prob_E, extent=extE, **kw)
    axes[0, 1].set_title(titles[1]);  axes[0, 1].set_xlabel('x (units: ½)');  axes[0, 1].set_ylabel('t')
    plt.colorbar(im1, ax=axes[0, 1])

    # Square + Rest
    extS = [xS[0]-0.5, xS[-1]+0.5, -0.5, T+0.5]
    im2 = axes[0, 2].imshow(prob_S, extent=extS, **kw)
    axes[0, 2].set_title(titles[2]);  axes[0, 2].set_xlabel('x');  axes[0, 2].set_ylabel('t')
    plt.colorbar(im2, ax=axes[0, 2])

    # ── Row 1: 1-D slices ──────────────────────────────────────────────────
    slices_t = [T // 4, T // 2, 3 * T // 4, T]
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(slices_t)))

    for t_idx, col in zip(slices_t, colors):
        xE_t = xs_equilateral(T, t_idx)
        axes[1, 0].plot(xC, prob_C[t_idx], color=col, label=f't={t_idx}')
        axes[1, 1].plot(xE_t, prob_E[t_idx], color=col, label=f't={t_idx}')
        axes[1, 2].plot(xS, prob_S[t_idx], color=col, label=f't={t_idx}')

    for ax, ttl in zip(axes[1], titles):
        ax.set_xlabel('x');  ax.set_ylabel('|ψ|²')
        ax.set_title(ttl);  ax.legend(fontsize=7);  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print('Saved model_comparison.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 3 — Spread σ(t) for all three models
# ═══════════════════════════════════════════════════════════════════════════

def fig_spread(T=50, epsilon=0.1):
    print(f'Simulating spread (T={T}, ε={epsilon})…', flush=True)
    psi_C = simulate_checkerboard(T, epsilon)
    psi_E = simulate_equilateral(T, epsilon)
    psi_S = simulate_square_rest(T, epsilon)

    t_arr = np.arange(T + 1)

    def spread(psi, x_arr_fn):
        sigmas = []
        for t_idx, row in enumerate(psi):
            p = np.abs(row) ** 2
            xs = x_arr_fn(t_idx)
            # pad/trim xs to match row length
            xs_aligned = xs[:len(p)] if len(xs) >= len(p) else np.pad(xs, (0, len(p)-len(xs)))
            norm = p.sum()
            if norm < 1e-30:
                sigmas.append(0.0)
                continue
            mu  = np.dot(p, xs_aligned) / norm
            mu2 = np.dot(p, xs_aligned**2) / norm
            sigmas.append(np.sqrt(max(mu2 - mu**2, 0)))
        return np.array(sigmas)

    xC = lambda t: xs_checkerboard(T)
    xE = lambda t: xs_equilateral(T, t)
    xS = lambda t: xs_square_rest(T)

    sig_C = spread(psi_C, xC)
    sig_E = spread(psi_E, xE)
    sig_S = spread(psi_S, xS)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f'Spread σ(t)  —  all three models  (ε={epsilon})', fontsize=11)

    ax.plot(t_arr, sig_C, color='#3498db', lw=2, label='Feynman Checkerboard (±1)')
    ax.plot(t_arr, sig_E, color='#e67e22', lw=2, label='Equilateral Triangular (±½)')
    ax.plot(t_arr, sig_S, color='#2ecc71', lw=2, label='Square + Rest (-1/0/+1)')

    # reference: ballistic σ ∝ t
    ref = t_arr * (sig_C[-1] / (T + 1e-9))
    ax.plot(t_arr, ref, 'k--', alpha=0.4, label='∝ t (ballistic)')

    ax.set_xlabel('t');  ax.set_ylabel('σ(x)')
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

    # note: equilateral x is in units of ½, so its σ will be half of checkerboard
    ax.text(0.02, 0.97,
            'Note: equilateral σ in units of ½ (physical σ is ×2)',
            transform=ax.transAxes, fontsize=8, va='top', color='#e67e22')

    plt.tight_layout()
    plt.savefig('lattice_spread.png', dpi=150, bbox_inches='tight')
    print('Saved lattice_spread.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=== Lattice geometry ===')
    fig_geometry()

    print('\n=== Probability comparison (T=40) ===')
    fig_comparison(T=40, epsilon=0.1)

    print('\n=== Spread analysis (T=50) ===')
    fig_spread(T=50, epsilon=0.1)

    print('\nDone.')
