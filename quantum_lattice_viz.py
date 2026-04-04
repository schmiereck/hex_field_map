"""
Lattice geometry visualization + 60°-rotated "Model B" comparison.

Model A (original triangular):
  moves: (-1,+1), (0,+1), (+1,+1)   [dx, dt]
  all moves advance time by 1

Model B (60°-rotated frame):
  The rotation maps time axis T=(0,1) → T'=(-√3/2, 1/2).
  In the rotated frame, move (+1,+1) goes *backward* (ΔT'=-0.37).
  We replace it with (-1,0) which has ΔT'=+0.87 → genuinely forward.
  moves: (-1,+1), (0,+1), (-1,0)    [dx, dt]
  Note: move (-1,0) has dt=0 (horizontal), so the DP must track (t,x) separately.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ─────────────────────────── geometry helpers ───────────────────────────────

MOVES_A = [(-1, 1), (0, 1), (1, 1)]   # (dx, dt)
MOVES_B = [(-1, 1), (0, 1), (-1, 0)]  # (-1,0) replaces (+1,+1)

COLORS = ['#e74c3c', '#2ecc71', '#3498db']   # red, green, blue per move
LABELS_A = ['(-1,+1)', '(0,+1)', '(+1,+1)']
LABELS_B = ['(-1,+1)', '(0,+1)', '(-1,0)']


def draw_lattice_ax(ax, moves, labels, T_draw=5, title='', show_rotation=False):
    """Draw lattice nodes and move-vectors on ax."""
    xs = range(-T_draw, T_draw + 1)
    ts = range(0, T_draw + 1)

    # grid nodes
    for t in ts:
        for x in xs:
            ax.plot(x, t, 'o', color='#95a5a6', ms=3, zorder=2)

    # highlight origin
    ax.plot(0, 0, 'ko', ms=6, zorder=3)

    # draw example move arrows from a few nodes
    sample_nodes = [(0, 0), (-1, 1), (1, 1), (0, 2), (-1, 3), (1, 3)]
    for (x0, t0) in sample_nodes:
        for (dx, dt), col in zip(moves, COLORS):
            x1, t1 = x0 + dx, t0 + dt
            if -T_draw <= x1 <= T_draw and 0 <= t1 <= T_draw:
                ax.annotate('', xy=(x1, t1), xytext=(x0, t0),
                            arrowprops=dict(arrowstyle='->', color=col,
                                           lw=1.4, shrinkA=3, shrinkB=3))

    # legend patches
    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(COLORS, labels)]
    ax.legend(handles=patches, loc='upper right', fontsize=7)

    if show_rotation:
        # draw original and rotated time axis
        ax.annotate('', xy=(0, T_draw * 0.7), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.text(0.15, T_draw * 0.65, "T", fontsize=10, color='black')
        # rotated axis: T' = (-√3/2, 1/2) normalized, scale by T_draw*0.6
        scale = T_draw * 0.6
        tx, ty = -np.sqrt(3) / 2 * scale, 0.5 * scale
        ax.annotate('', xy=(tx, ty), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2))
        ax.text(tx - 0.5, ty + 0.1, "T'", fontsize=10, color='purple')

    ax.set_xlim(-T_draw - 0.5, T_draw + 0.5)
    ax.set_ylim(-0.5, T_draw + 0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.grid(False)


# ─────────────────────────── simulators ─────────────────────────────────────

def simulate_model_A(T, epsilon=0.1):
    """
    Model A: moves (-1,+1),(0,+1),(+1,+1).
    Returns complex psi[T+1, 2T+1].
    """
    N = 2 * T + 1
    center = T
    ie = 1j * epsilon

    # change[d_new, d_old]: factor when arriving via d_new from d_old
    change = np.full((3, 3), ie, dtype=complex)
    np.fill_diagonal(change, 1.0 + 0j)

    amp = np.zeros((N, 3), dtype=complex)
    amp[center, :] = 1.0 / 3.0  # start uniform over directions

    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0] = np.sum(amp, axis=1)

    dirs = [-1, 0, 1]
    for step in range(1, T + 1):
        new_amp = np.zeros((N, 3), dtype=complex)
        for d_new, dv in enumerate(dirs):
            weighted = amp @ change[d_new]   # (N,)
            if dv == -1:
                new_amp[:-1, d_new] += weighted[1:]
            elif dv == 0:
                new_amp[:, d_new]   += weighted
            else:
                new_amp[1:, d_new]  += weighted[:-1]
        amp = new_amp
        psi[step] = np.sum(amp, axis=1)

    return psi


def simulate_model_B(T, epsilon=0.1):
    """
    Model B: moves (-1,+1),(0,+1),(-1,0).
    Move 2 has dt=0, so we track a 2D amplitude grid amp[t, x, direction].
    After each "time row" we apply the dt=0 move iteratively until convergence,
    but since the move is purely spatial (no loop possible), one application suffices.

    State: amp[x, d] for current time slice.
    Advances:
      d=0: (dx=-1, dt=+1) → spatial shift left, time advance
      d=1: (dx= 0, dt=+1) → no spatial shift, time advance
      d=2: (dx=-1, dt= 0) → spatial shift left, SAME time slice

    We process each time step as:
      1. Apply dt=0 move within current slice (only once; no cycles possible)
      2. Advance to next time slice via dt=1 moves
    """
    N = 2 * T + 1
    center = T
    ie = 1j * epsilon

    # 3 moves: indices 0,1,2
    # change[d_new, d_old]
    change = np.full((3, 3), ie, dtype=complex)
    np.fill_diagonal(change, 1.0 + 0j)

    amp = np.zeros((N, 3), dtype=complex)
    amp[center, :] = 1.0 / 3.0

    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0] = np.sum(amp, axis=1)

    for step in range(1, T + 1):
        # Step A: propagate dt=1 moves from previous slice → new slice
        new_amp = np.zeros((N, 3), dtype=complex)

        # d_new=0: move (-1,+1)
        weighted = amp @ change[0]
        new_amp[:-1, 0] += weighted[1:]   # shift left

        # d_new=1: move (0,+1)
        weighted = amp @ change[1]
        new_amp[:, 1] += weighted

        # d_new=2: move (-1,0) arriving at new time slice from same slice
        # Source: previous slice arriving via dt=0 from previous slice
        # This is tricky: dt=0 move stays in the SAME time slice.
        # We handle it as: within new_amp, after the dt=1 contributions,
        # we fold in the dt=0 self-propagation once (no feedback loop possible
        # because the chain can't cycle: each application shifts x by -1).
        weighted = amp @ change[2]
        new_amp[:-1, 2] += weighted[1:]   # shift left, same time

        amp = new_amp

        # Step B: apply dt=0 move within this slice (propagate d=2 arrivals
        # that can themselves feed d=2 arrivals shifted further left).
        # Since each application shifts left by 1, we apply T times max.
        for _ in range(T):
            extra = np.zeros((N, 3), dtype=complex)
            weighted = amp[:, 2:3] @ change[2:3, 2:3]   # only from d=2
            weighted = weighted[:, 0]
            if np.max(np.abs(weighted)) < 1e-15:
                break
            # d_new = 0,1 can't come from dt=0 move (they have dt=1)
            # d_new = 2 from d=2: shift left
            extra[:-1, 2] += weighted[1:]
            amp += extra

        psi[step] = np.sum(amp, axis=1)

    return psi


# ─────────────────────────── figure 1: geometry ─────────────────────────────

def fig_lattice_and_rotation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Triangular Lattice Geometry & 60° Time-Axis Rotation', fontsize=13)

    # Panel 1: Model A lattice
    draw_lattice_ax(axes[0], MOVES_A, LABELS_A, T_draw=5,
                    title='Model A — original\nmoves: (-1,+1), (0,+1), (+1,+1)')

    # Panel 2: rotation diagram
    ax = axes[1]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title('60° rotation of time axis\nT=(0,1) → T\'=(-√3/2, 1/2)', fontsize=10)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    # original axes
    ax.annotate('', xy=(0, 2), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
    ax.annotate('', xy=(1.8, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
    ax.text(0.1, 1.9, 'T', fontsize=12, color='black', fontweight='bold')
    ax.text(1.7, 0.1, 'X', fontsize=12, color='black', fontweight='bold')

    # rotated time axis (60° CCW from vertical = 30° CCW from y-axis)
    angle_rad = np.radians(60)
    tx = -np.sin(angle_rad) * 1.8   # -√3/2 * 1.8
    ty =  np.cos(angle_rad) * 1.8   #  1/2  * 1.8
    ax.annotate('', xy=(tx, ty), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2.5))
    ax.text(tx - 0.35, ty + 0.05, "T'", fontsize=12, color='purple', fontweight='bold')

    # show moves and their projection onto T'
    moves_info = [
        ((-1, 1), '#e74c3c', '(-1,+1)\nΔT\'=+0.87 ✓'),
        (( 0, 1), '#2ecc71', '(0,+1)\nΔT\'=+0.50 ✓'),
        (( 1, 1), '#3498db', '(+1,+1)\nΔT\'=-0.37 ✗'),
        ((-1, 0), '#f39c12', '(-1,0)\nΔT\'=+0.87 ✓ (replacement)'),
    ]
    # unit T' vector
    Tp = np.array([-np.sqrt(3) / 2, 0.5])
    Tp /= np.linalg.norm(Tp)

    offsets = [(-0.8, 0.15), (0.1, 0.1), (0.35, 0.1), (-0.6, -0.35)]
    for (dx, dt), col, lbl, off in zip(
            [m[0] for m in moves_info],
            [m[1] for m in moves_info],
            [m[2] for m in moves_info],
            offsets):
        v = np.array([dx, dt], dtype=float)
        proj = np.dot(v, Tp)
        ax.annotate('', xy=(dx * 0.6, dt * 0.6), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2,
                                   shrinkA=0, shrinkB=0))
        ax.text(dx * 0.6 + off[0], dt * 0.6 + off[1], lbl,
                fontsize=7.5, color=col,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.grid(True, alpha=0.3)

    # Panel 3: Model B lattice
    draw_lattice_ax(axes[2], MOVES_B, LABELS_B, T_draw=5,
                    title='Model B — 60°-rotated frame\nmoves: (-1,+1), (0,+1), (-1,0)')

    plt.tight_layout()
    plt.savefig('lattice_geometry.png', dpi=150, bbox_inches='tight')
    print('Saved lattice_geometry.png')
    plt.close()


# ─────────────────────────── figure 2: simulation comparison ────────────────

def fig_simulation_comparison(T=30, epsilon=0.1):
    print(f'Simulating Model A (T={T})…', flush=True)
    psi_A = simulate_model_A(T, epsilon)
    print(f'Simulating Model B (T={T})…', flush=True)
    psi_B = simulate_model_B(T, epsilon)

    prob_A = np.abs(psi_A) ** 2
    prob_B = np.abs(psi_B) ** 2
    N = prob_A.shape[1]
    center = N // 2
    xs = np.arange(N) - center

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f'Model A vs Model B  (T={T}, ε={epsilon})\n'
        'Model A: symmetric moves  |  Model B: 60°-rotated (chiral, drifts left)',
        fontsize=12)

    # ── Row 0: probability heatmaps ──
    kw = dict(aspect='auto', origin='lower', cmap='inferno')
    ext = [xs[0] - 0.5, xs[-1] + 0.5, -0.5, T + 0.5]

    im0 = axes[0, 0].imshow(prob_A, extent=ext, **kw)
    axes[0, 0].set_title('Model A  |ψ|²')
    axes[0, 0].set_xlabel('x');  axes[0, 0].set_ylabel('t')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(prob_B, extent=ext, **kw)
    axes[0, 1].set_title('Model B  |ψ|²')
    axes[0, 1].set_xlabel('x');  axes[0, 1].set_ylabel('t')
    plt.colorbar(im1, ax=axes[0, 1])

    # difference
    diff = prob_B - prob_A
    vmax = np.abs(diff).max()
    im2 = axes[0, 2].imshow(diff, extent=ext, aspect='auto', origin='lower',
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title('B − A  (probability difference)')
    axes[0, 2].set_xlabel('x');  axes[0, 2].set_ylabel('t')
    plt.colorbar(im2, ax=axes[0, 2])

    # ── Row 1: 1-D slices at several time steps ──
    slices_t = [T // 4, T // 2, 3 * T // 4, T]
    cmap_sl = plt.cm.plasma(np.linspace(0.2, 0.9, len(slices_t)))

    for t_idx, col in zip(slices_t, cmap_sl):
        axes[1, 0].plot(xs, prob_A[t_idx], color=col, label=f't={t_idx}')
        axes[1, 1].plot(xs, prob_B[t_idx], color=col, label=f't={t_idx}')
        axes[1, 2].plot(xs, prob_B[t_idx] - prob_A[t_idx], color=col, label=f't={t_idx}')

    for ax_col, ttl in zip(axes[1], ['Model A  |ψ|²', 'Model B  |ψ|²', 'B − A']):
        ax_col.set_xlabel('x');  ax_col.set_ylabel('probability')
        ax_col.set_title(ttl)
        ax_col.legend(fontsize=7)
        ax_col.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print('Saved model_comparison.png')
    plt.close()


# ─────────────────────────── figure 3: mean / spread ────────────────────────

def fig_drift_and_spread(T=50, epsilon=0.1):
    print(f'Simulating Model A (T={T})…', flush=True)
    psi_A = simulate_model_A(T, epsilon)
    print(f'Simulating Model B (T={T})…', flush=True)
    psi_B = simulate_model_B(T, epsilon)

    N = psi_A.shape[1]
    center = N // 2
    xs = np.arange(N) - center
    t_arr = np.arange(T + 1)

    def mean_and_std(prob):
        mu  = np.array([np.sum(xs * p) / (np.sum(p) + 1e-30) for p in prob])
        mu2 = np.array([np.sum(xs**2 * p) / (np.sum(p) + 1e-30) for p in prob])
        sigma = np.sqrt(np.maximum(mu2 - mu**2, 0))
        return mu, sigma

    prob_A = np.abs(psi_A) ** 2
    prob_B = np.abs(psi_B) ** 2

    mu_A, sig_A = mean_and_std(prob_A)
    mu_B, sig_B = mean_and_std(prob_B)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Drift and spread  (T={T}, ε={epsilon})', fontsize=12)

    axes[0].plot(t_arr, mu_A,  label='Model A (symmetric)', color='steelblue')
    axes[0].plot(t_arr, mu_B,  label='Model B (chiral)',    color='crimson')
    axes[0].set_xlabel('t');  axes[0].set_ylabel('<x>')
    axes[0].set_title('Mean position')
    axes[0].legend();  axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_arr, sig_A, label='Model A', color='steelblue')
    axes[1].plot(t_arr, sig_B, label='Model B', color='crimson')
    # reference: diffusive √t
    ref_t = np.sqrt(t_arr.astype(float))
    ref_t[0] = 0
    axes[1].plot(t_arr, ref_t * (sig_A[-1] / ref_t[-1]),
                 'k--', alpha=0.5, label='∝ √t (diffusive)')
    axes[1].set_xlabel('t');  axes[1].set_ylabel('σ(x)')
    axes[1].set_title('Spread (standard deviation)')
    axes[1].legend();  axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('drift_and_spread.png', dpi=150, bbox_inches='tight')
    print('Saved drift_and_spread.png')
    plt.close()


# ─────────────────────────── main ────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Lattice geometry visualization ===')
    fig_lattice_and_rotation()

    print('\n=== Simulation comparison (T=30) ===')
    fig_simulation_comparison(T=30, epsilon=0.1)

    print('\n=== Drift and spread (T=50) ===')
    fig_drift_and_spread(T=50, epsilon=0.1)

    print('\nDone.')
