#!/usr/bin/env python3
"""
Quantum Path Integral Simulator
================================
Implements the Feynman Checkerboard model on two lattice geometries:

  1. Triangular lattice — 3 moves per timestep:
        left-diagonal  (-1, +1)
        forward-only   ( 0, +1)   (no lateral movement)
        right-diagonal (+1, +1)

  2. Square lattice — 2 moves per timestep (classic Feynman Checkerboard):
        left   (-1, +1)
        right  (+1, +1)

Amplitude rule (per step):
  · Same direction as previous step  →  factor = 1
  · Direction change                 →  factor = i·ε   (ε = mass parameter)

The total path amplitude is the product of all per-step factors.
Probability at a node is  P = |Σ_paths amplitude|².

Algorithm: dynamic programming over states (x, last_direction).
  amp[x_idx, d_idx] = total complex amplitude arriving at position x
                      via last move-direction d_idx.

Performance: O(T · N · D²) with numpy vectorisation — T=50 runs in < 1 s.

Usage:
  python quantum_path_integral.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from time import time


# ══════════════════════════════════════════════════════════════════════════════
#  Core simulators
# ══════════════════════════════════════════════════════════════════════════════

def simulate_triangular(T: int, epsilon: float) -> np.ndarray:
    """
    Path integral on the triangular lattice.

    State space : (x, last_dir) where last_dir ∈ {left(−1), straight(0), right(+1)}
    Directions  : dirs[0]=−1, dirs[1]=0, dirs[2]=+1

    Parameters
    ----------
    T       : number of timesteps
    epsilon : mass parameter; direction-change amplitude = i·epsilon

    Returns
    -------
    probs : ndarray, shape (T+1, 2T+1)
        |amplitude|² at each (t, x), unnormalised.
    """
    N      = 2 * T + 1           # spatial grid size; x ∈ {−T, …, +T}
    offset = T                   # index mapping: xi = x + offset
    dirs   = [-1, 0, 1]          # left, straight, right
    ie     = 1j * epsilon        # direction-change amplitude

    # amp[xi, d] = complex amplitude arriving at xi via last direction d
    amp   = np.zeros((N, 3), dtype=complex)
    probs = np.zeros((T + 1, N))
    probs[0, offset] = 1.0       # particle starts at origin

    # ── First step (t=0→1): no previous direction, no change penalty ──────────
    for d, dv in enumerate(dirs):
        xi = offset + dv
        if 0 <= xi < N:
            amp[xi, d] = 1.0

    if T >= 1:
        probs[1] = np.abs(np.sum(amp, axis=1)) ** 2

    # Transition matrix: change[d_new, d_old]
    #   = 1   if d_new == d_old  (same direction, no penalty)
    #   = i·ε if d_new != d_old  (direction change)
    change = np.full((3, 3), ie, dtype=complex)
    np.fill_diagonal(change, 1.0 + 0j)

    # ── Main time-evolution loop ──────────────────────────────────────────────
    for _ in range(1, T):
        new_amp = np.zeros((N, 3), dtype=complex)

        for d_new, dv_new in enumerate(dirs):
            # Weighted superposition of all incoming amplitudes for this new direction.
            # weighted[xi] = Σ_d_old  amp[xi, d_old] * change[d_new, d_old]
            weighted = amp @ change[d_new]          # shape (N,)

            # Propagate: new position = old position + dv_new
            #   ⇒ old xi = new xi − dv_new
            if dv_new == -1:        # moved left:   old_xi = new_xi + 1
                new_amp[:-1, d_new] += weighted[1:]
            elif dv_new == 0:       # stayed put:   old_xi = new_xi
                new_amp[:, d_new]   += weighted
            else:                   # moved right:  old_xi = new_xi − 1
                new_amp[1:, d_new]  += weighted[:-1]

        amp = new_amp
        probs[_ + 1] = np.abs(np.sum(amp, axis=1)) ** 2

    return probs


def simulate_square(T: int, epsilon: float) -> np.ndarray:
    """
    Path integral on the square lattice — classic Feynman Checkerboard.

    State space : (x, last_dir) where last_dir ∈ {left(−1), right(+1)}
    The particle always moves; no "staying in place" option.

    Parameters
    ----------
    T       : number of timesteps
    epsilon : mass parameter

    Returns
    -------
    probs : ndarray, shape (T+1, 2T+1)
    """
    N      = 2 * T + 1
    offset = T
    ie     = 1j * epsilon

    # amp[xi, d] where d: 0=left(−1), 1=right(+1)
    amp   = np.zeros((N, 2), dtype=complex)
    probs = np.zeros((T + 1, N))
    probs[0, offset] = 1.0

    # ── First step ────────────────────────────────────────────────────────────
    if offset - 1 >= 0:
        amp[offset - 1, 0] = 1.0   # moved left
    if offset + 1 < N:
        amp[offset + 1, 1] = 1.0   # moved right

    if T >= 1:
        probs[1] = np.abs(np.sum(amp, axis=1)) ** 2

    # Transition matrix [d_new, d_old]: 1 if same, ie if different
    change = np.array([[1.0 + 0j, ie],
                       [ie,  1.0 + 0j]])

    # ── Main loop ─────────────────────────────────────────────────────────────
    for t in range(1, T):
        new_amp = np.zeros((N, 2), dtype=complex)

        # d_new=0 (left, dv=−1): old_xi = new_xi + 1
        w0 = amp @ change[0]           # weighted sum for left arrivals
        new_amp[:-1, 0] += w0[1:]

        # d_new=1 (right, dv=+1): old_xi = new_xi − 1
        w1 = amp @ change[1]           # weighted sum for right arrivals
        new_amp[1:, 1] += w1[:-1]

        amp = new_amp
        probs[t + 1] = np.abs(np.sum(amp, axis=1)) ** 2

    return probs


# ══════════════════════════════════════════════════════════════════════════════
#  Analysis helpers
# ══════════════════════════════════════════════════════════════════════════════

def distribution_stats(probs: np.ndarray, T: int) -> dict:
    """
    Compute summary statistics for the probability distribution at t = T.

    Returns dict with keys: total, mean_x, std_x, spread_fraction, peak_x.
    """
    x   = np.arange(-T, T + 1, dtype=float)
    row = probs[T]
    tot = row.sum()

    if tot < 1e-30:
        return dict(total=0, mean_x=float('nan'), std_x=float('nan'),
                    spread_fraction=float('nan'), peak_x=float('nan'))

    p     = row / tot
    mean  = float(np.dot(p, x))
    std   = float(np.sqrt(np.dot(p, (x - mean) ** 2)))
    peak  = float(x[np.argmax(row)])

    return dict(total=tot, mean_x=mean, std_x=std,
                spread_fraction=std / T, peak_x=peak)


def light_cone_fraction(probs: np.ndarray, T: int) -> float:
    """
    Fraction of total probability within the light cone |x| ≤ t at each t,
    averaged over all t > 0.
    """
    fracs = []
    for t in range(1, T + 1):
        row  = probs[t]
        tot  = row.sum()
        if tot < 1e-30:
            continue
        xs   = np.arange(-T, T + 1)
        mask = np.abs(xs) <= t
        fracs.append(row[mask].sum() / tot)
    return float(np.mean(fracs)) if fracs else float('nan')


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

CMAP = 'inferno'


def _add_lightcone(ax, T: int, color='cyan', lw=1.4, alpha=0.80):
    t_arr = np.array([0.0, T])
    ax.plot( t_arr, t_arr, '--', color=color, lw=lw, alpha=alpha)
    ax.plot(-t_arr, t_arr, '--', color=color, lw=lw, alpha=alpha,
            label='Light cone')


def plot_heatmap(ax, probs: np.ndarray, title: str, T: int,
                 add_legend: bool = False) -> None:
    """Render a normalised probability heatmap with light-cone overlay."""
    vmax    = probs.max()
    display = probs / max(vmax, 1e-30)

    ax.imshow(display,
              origin='lower', aspect='auto',
              extent=[-T - 0.5, T + 0.5, -0.5, T + 0.5],
              cmap=CMAP, vmin=0, vmax=1)

    _add_lightcone(ax, T)
    ax.set_xlim(-T, T)
    ax.set_ylim(0, T)
    ax.set_xlabel('Space (x)', fontsize=8)
    ax.set_ylabel('Time  (t)', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    if add_legend:
        ax.legend(loc='upper right', fontsize=7, framealpha=0.6)


def plot_marginal(ax, probs: np.ndarray, T: int, color: str,
                  label: str = '') -> None:
    """Plot normalised distribution at final timestep."""
    x    = np.arange(-T, T + 1)
    row  = probs[T]
    vmax = row.max()
    norm = row / max(vmax, 1e-30)

    ax.plot(x, norm, '-', color=color, lw=1.6, label=label)
    ax.fill_between(x, norm, alpha=0.18, color=color)
    ax.axvline(-T, color='cyan', ls='--', lw=0.9, alpha=0.5)
    ax.axvline( T, color='cyan', ls='--', lw=0.9, alpha=0.5)
    ax.set_xlabel('Space (x)', fontsize=8)
    ax.set_ylabel('P  (norm.)',  fontsize=8)
    ax.set_xlim(-T, T)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_locator(MaxNLocator(4))


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Main lattice comparison (T=20 and T=50)
# ══════════════════════════════════════════════════════════════════════════════

def fig_main_comparison(results: dict, T_values: list, epsilon: float) -> None:
    """
    4-column layout:
      col 0 — Triangular T=20
      col 1 — Square     T=20
      col 2 — Triangular T=50
      col 3 — Square     T=50

    Each column has:
      rows 0-1 — heatmap
      row  2   — final distribution
    """
    fig = plt.figure(figsize=(22, 11))
    fig.suptitle(
        f'Quantum Path Integral  ·  Triangular vs Square Lattice  (ε = {epsilon})\n'
        'Colormap: normalised |amplitude|²  ·  Cyan dashed: light cone',
        fontsize=13, fontweight='bold', y=1.00)

    outer = gridspec.GridSpec(1, len(T_values), figure=fig,
                              wspace=0.22)

    colors = {'triangular': '#E8A838', 'square': '#4FC3F7'}

    for t_idx, T in enumerate(T_values):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 2, subplot_spec=outer[t_idx],
            height_ratios=[3, 3, 1.4],
            hspace=0.45, wspace=0.30)

        for col, (key, label, short) in enumerate([
            ('triangular', 'Triangular Lattice\n(3 moves/step)', 'Triangular'),
            ('square',     'Square Lattice\n(Feynman Checkerboard)', 'Square'),
        ]):
            probs = results[(T, key)]

            # Heatmap spans top two rows of inner grid
            ax_h = fig.add_subplot(inner[:2, col])
            plot_heatmap(ax_h, probs,
                         f'{label}  T = {T}', T,
                         add_legend=(t_idx == 0 and col == 0))

            # Final-time marginal
            ax_m = fig.add_subplot(inner[2, col])
            plot_marginal(ax_m, probs, T,
                          color=colors[key],
                          label=f't = {T}')
            ax_m.set_title(f'Distribution at t = {T}', fontsize=8)

    out = 'path_integral_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  → saved  {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Effect of mass parameter ε
# ══════════════════════════════════════════════════════════════════════════════

def fig_epsilon_effect(epsilons: list, T: int = 30) -> None:
    """
    Grid: rows = lattice type, cols = epsilon value.
    Shows how ε controls the spread / localisation of the wavepacket.
    """
    n_eps = len(epsilons)
    fig, axes = plt.subplots(2, n_eps, figsize=(4.5 * n_eps, 9))
    fig.suptitle(
        f'Effect of Mass Parameter ε  (T = {T})\n'
        'Top: Triangular lattice  ·  Bottom: Feynman Checkerboard (Square)',
        fontsize=12, fontweight='bold')

    for col, eps in enumerate(epsilons):
        for row, (fn, lattice_name) in enumerate([
            (simulate_triangular, 'Triangular'),
            (simulate_square,     'Square'),
        ]):
            print(f'    ε={eps:<5}  {lattice_name:<12}  T={T}…', end='', flush=True)
            t0    = time()
            probs = fn(T, eps)
            print(f' {time()-t0:.2f}s')

            ax = axes[row, col]
            plot_heatmap(ax, probs, f'ε = {eps}', T,
                         add_legend=(row == 0 and col == 0))
            if col == 0:
                ax.set_ylabel(f'{lattice_name}\nTime (t)', fontsize=9)

    plt.tight_layout()
    out = f'epsilon_effect_T{T}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  → saved  {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Spread over time (std-dev vs t)
# ══════════════════════════════════════════════════════════════════════════════

def fig_spread_vs_time(results: dict, T_values: list, epsilon: float) -> None:
    """
    Plot σ(t) = std-dev of |amplitude|² vs time for both lattices.
    Compares against ballistic (σ ~ t) and diffusive (σ ~ √t) scalings.
    """
    fig, axes = plt.subplots(1, len(T_values), figsize=(12, 4.5), sharey=False)
    fig.suptitle(f'Spatial Spread σ(t) vs Time  (ε = {epsilon})',
                 fontsize=12, fontweight='bold')

    configs = [
        ('triangular', '#E8A838', 'Triangular (3 moves)'),
        ('square',     '#4FC3F7', 'Square / Checkerboard'),
    ]

    for ax, T in zip(axes, T_values):
        t_arr = np.arange(1, T + 1, dtype=float)

        for key, color, label in configs:
            probs  = results[(T, key)]
            x_axis = np.arange(-T, T + 1, dtype=float)
            stds   = []

            for t in range(1, T + 1):
                row = probs[t]
                tot = row.sum()
                if tot < 1e-30:
                    stds.append(0.0)
                    continue
                p   = row / tot
                mu  = np.dot(p, x_axis)
                stds.append(float(np.sqrt(np.dot(p, (x_axis - mu) ** 2))))

            ax.plot(t_arr, stds, '-', color=color, lw=2.0, label=label)

        # Reference lines
        ax.plot(t_arr, t_arr,          'k--', lw=0.8, alpha=0.4, label='Ballistic  σ ~ t')
        ax.plot(t_arr, np.sqrt(t_arr), 'k:',  lw=0.8, alpha=0.4, label='Diffusive  σ ~ √t')

        ax.set_xlabel('Time  t',       fontsize=9)
        ax.set_ylabel('σ(t)',          fontsize=9)
        ax.set_title(f'T = {T}',       fontsize=10)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = 'spread_vs_time.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  → saved  {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print('Quantum Path Integral Simulator')
    print('=' * 55)
    print('Model  : Feynman Checkerboard — Triangular & Square lattice')
    print()

    epsilon  = 0.1
    T_values = [20, 50]

    # ── Run primary simulations ───────────────────────────────────────────────
    print(f'Running main simulations  (ε = {epsilon}):')
    results: dict = {}

    for T in T_values:
        for key, fn in [('triangular', simulate_triangular),
                        ('square',     simulate_square)]:
            print(f'  {key:<12}  T={T:>2}…', end='', flush=True)
            t0    = time()
            probs = fn(T, epsilon)
            dt    = time() - t0
            print(f' {dt:.3f}s')
            results[(T, key)] = probs

    # ── Print statistics ──────────────────────────────────────────────────────
    print()
    print('Distribution statistics at t = T:')
    print(f'  {"Lattice":<14} {"T":>4}   {"σ(T)":>8}  {"σ/T":>7}  '
          f'{"<|x|>":>7}  {"LC frac":>8}')
    print('  ' + '-' * 56)

    for T in T_values:
        for key in ('triangular', 'square'):
            probs = results[(T, key)]
            s     = distribution_stats(probs, T)
            lcf   = light_cone_fraction(probs, T)
            print(f'  {key:<14} {T:>4}   {s["std_x"]:>8.3f}  '
                  f'{s["spread_fraction"]:>7.4f}  '
                  f'{abs(s["mean_x"]):>7.4f}  {lcf:>8.4f}')

    # ── Visual answers to the three questions ─────────────────────────────────
    print()
    print('Visual questions (answered by the plots):')
    for T in T_values:
        print(f'\n  T = {T}:')
        for key in ('triangular', 'square'):
            s   = distribution_stats(results[(T, key)], T)
            lcf = light_cone_fraction(results[(T, key)], T)
            # Heuristic: spread_fraction > 0.3 → approaches ballistic / light-cone
            lc_msg = 'YES — approaching light cone' if s['spread_fraction'] > 0.3 \
                     else 'PARTIAL — spread below light cone'
            print(f'    {key:<14} σ/T={s["spread_fraction"]:.3f}  '
                  f'LC-frac={lcf:.3f}  → {lc_msg}')

    # ── Generate figures ──────────────────────────────────────────────────────
    print()
    print('Generating Figure 1 — main lattice comparison…')
    fig_main_comparison(results, T_values, epsilon)

    print()
    print('Generating Figure 2 — epsilon effect (T=30)…')
    fig_epsilon_effect(epsilons=[0.01, 0.1, 0.5, 1.0], T=30)

    print()
    print('Generating Figure 3 — spread vs time…')
    fig_spread_vs_time(results, T_values, epsilon)

    print()
    print('All done.')
    print()
    print('Saved files:')
    print('  path_integral_comparison.png')
    print('  epsilon_effect_T30.png')
    print('  spread_vs_time.png')


if __name__ == '__main__':
    main()
