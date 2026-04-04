#!/usr/bin/env python3
"""
Schritt 1 — Komplexe Amplituden & Interferenzmuster
=====================================================
Erweitert den Path-Integral-Simulator um die volle komplexe Wellenfunktion
ψ(x,t) statt nur |ψ|².

Neue Simulatoren geben ψ[T+1, N] (komplex) zurück.
Validierung: np.allclose(|ψ|², probs) muss passen.

Ausgabe-Plots:
  phase_components_T<N>.png   — Re(ψ), Im(ψ), |ψ|², arg(ψ) für beide Gitter
  phase_slices_T<N>.png       — 1D-Schnitte bei festen Zeiten (Wellenmuster)
  phase_comparison.png        — Direkter Phasenvergleich Dreieck vs. Quadrat
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from time import time

# Importiere bestehende Simulatoren für Kreuzvalidierung
from quantum_path_integral import simulate_triangular, simulate_square


# ══════════════════════════════════════════════════════════════════════════════
#  Komplexe Simulatoren
#  (identische Dynamik wie simulate_triangular/square, aber ψ statt |ψ|²)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_triangular_complex(T: int, epsilon: float) -> np.ndarray:
    """
    Dreiecksgitter-Path-Integral mit voller komplexer Wellenfunktion.

    Returns
    -------
    psi : ndarray, shape (T+1, 2T+1), dtype complex
        ψ(t, x) = Σ_d  amp[x, d]   —  Gesamtamplitude an jedem Knoten.
        Wahrscheinlichkeit: |ψ|²  (identisch mit simulate_triangular).
    """
    N      = 2 * T + 1
    offset = T
    dirs   = [-1, 0, 1]
    ie     = 1j * epsilon

    amp = np.zeros((N, 3), dtype=complex)
    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0, offset] = 1.0                    # Punktquelle am Ursprung

    # Erster Schritt ohne Vorzugsrichtung — kein Änderungsterm
    for d, dv in enumerate(dirs):
        xi = offset + dv
        if 0 <= xi < N:
            amp[xi, d] = 1.0

    if T >= 1:
        psi[1] = np.sum(amp, axis=1)        # komplexe Summe, kein |·|²

    # Übergangsmatrix: change[d_new, d_old]
    change = np.full((3, 3), ie, dtype=complex)
    np.fill_diagonal(change, 1.0 + 0j)

    for t in range(1, T):
        new_amp = np.zeros((N, 3), dtype=complex)
        for d_new, dv_new in enumerate(dirs):
            weighted = amp @ change[d_new]      # (N,)
            if dv_new == -1:
                new_amp[:-1, d_new] += weighted[1:]
            elif dv_new == 0:
                new_amp[:, d_new]   += weighted
            else:
                new_amp[1:, d_new]  += weighted[:-1]
        amp = new_amp
        psi[t + 1] = np.sum(amp, axis=1)

    return psi


def simulate_square_complex(T: int, epsilon: float) -> np.ndarray:
    """
    Quadratgitter-Path-Integral (Feynman-Checkerboard) mit komplexer ψ.

    Returns
    -------
    psi : ndarray, shape (T+1, 2T+1), dtype complex
    """
    N      = 2 * T + 1
    offset = T
    ie     = 1j * epsilon

    amp = np.zeros((N, 2), dtype=complex)
    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0, offset] = 1.0

    if offset - 1 >= 0:
        amp[offset - 1, 0] = 1.0
    if offset + 1 < N:
        amp[offset + 1, 1] = 1.0

    if T >= 1:
        psi[1] = np.sum(amp, axis=1)

    change = np.array([[1.0 + 0j, ie],
                       [ie,  1.0 + 0j]])

    for t in range(1, T):
        new_amp = np.zeros((N, 2), dtype=complex)
        w0 = amp @ change[0]
        new_amp[:-1, 0] += w0[1:]
        w1 = amp @ change[1]
        new_amp[1:, 1]  += w1[:-1]
        amp = new_amp
        psi[t + 1] = np.sum(amp, axis=1)

    return psi


# ══════════════════════════════════════════════════════════════════════════════
#  Validierung
# ══════════════════════════════════════════════════════════════════════════════

def validate(T: int = 20, epsilon: float = 0.1) -> bool:
    """
    Prüft: |ψ|² aus komplexem Simulator == probs aus originalen Simulator.
    """
    ok = True
    for name, fn_c, fn_r in [
        ('triangular', simulate_triangular_complex, simulate_triangular),
        ('square',     simulate_square_complex,     simulate_square),
    ]:
        psi   = fn_c(T, epsilon)
        probs = fn_r(T, epsilon)
        diff  = np.max(np.abs(np.abs(psi) ** 2 - probs))
        status = '✓' if diff < 1e-10 else '✗'
        print(f'  {status}  {name:<14}  max |  |ψ|² − probs  | = {diff:.2e}')
        ok = ok and (diff < 1e-10)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisierungs-Helfer
# ══════════════════════════════════════════════════════════════════════════════

# Divergierendes Colormap (rot-weiß-blau) für Re/Im
CMAP_DIV  = 'RdBu_r'
# Zyklisches Colormap für Phase arg(ψ)
CMAP_CYCL = 'hsv'
# Wie gehabt für |ψ|²
CMAP_PROB = 'inferno'

# Maske für schwache Amplituden bei der Phasendarstellung
PHASE_THRESH = 0.02   # Anteil des globalen Maximums von |ψ|


def _extent(T: int):
    return [-T - 0.5, T + 0.5, -0.5, T + 0.5]


def _lightcone(ax, T, color='cyan', lw=1.2, alpha=0.7):
    t = np.array([0.0, T])
    ax.plot( t, t, '--', color=color, lw=lw, alpha=alpha)
    ax.plot(-t, t, '--', color=color, lw=lw, alpha=alpha)


def _heatmap(ax, data, title, T, cmap, vmin=None, vmax=None,
             symm=False, label='', lightcone=True):
    """Generisches Heatmap-Panel."""
    if symm:
        # Symmetrisch um 0 mit Weiß in der Mitte
        vmax_ = np.max(np.abs(data)) if vmax is None else vmax
        vmin_ = -vmax_
        im = ax.imshow(data, origin='lower', aspect='auto',
                       extent=_extent(T), cmap=cmap,
                       vmin=vmin_, vmax=vmax_)
    else:
        im = ax.imshow(data, origin='lower', aspect='auto',
                       extent=_extent(T), cmap=cmap,
                       vmin=vmin, vmax=vmax)

    if lightcone:
        _lightcone(ax, T)

    ax.set_xlim(-T, T)
    ax.set_ylim(0, T)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('t', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    if label:
        cb.set_label(label, fontsize=7)

    return im


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Vier Komponenten: Re, Im, |ψ|², arg(ψ)
# ══════════════════════════════════════════════════════════════════════════════

def fig_components(psi_tri: np.ndarray, psi_sq: np.ndarray, T: int,
                   epsilon: float, save_prefix: str = 'phase_components') -> None:
    """
    2×4 Grid: Zeilen = Gittertyp (Dreieck / Quadrat),
              Spalten = Re(ψ) | Im(ψ) | |ψ|² | arg(ψ)
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        f'Komplexe Wellenfunktion ψ(x,t)  ·  T={T}  ·  ε={epsilon}\n'
        'Cyan gestrichelt: Lichtkegel',
        fontsize=12, fontweight='bold')

    labels_row = ['Dreiecksgitter  (3 Züge/Schritt)',
                  'Quadratgitter  (Feynman-Checkerboard)']
    col_titles = ['Re(ψ)', 'Im(ψ)', '|ψ|²', 'arg(ψ)  [rad]']

    for row, (psi, row_label) in enumerate([(psi_tri, labels_row[0]),
                                             (psi_sq,  labels_row[1])]):
        prob     = np.abs(psi) ** 2
        amp_max  = np.abs(psi).max()
        phase_mask = np.abs(psi) < PHASE_THRESH * amp_max

        # arg(ψ): maskiere schwache Amplituden mit NaN (undefinierte Phase)
        phase = np.angle(psi).astype(float)
        phase[phase_mask] = np.nan

        # Normierung Re/Im auf gemeinsame Skala
        re_im_scale = np.abs(np.real(psi)).max()

        panels = [
            (np.real(psi),    col_titles[0], CMAP_DIV,  True,  False, 'Re'),
            (np.imag(psi),    col_titles[1], CMAP_DIV,  True,  False, 'Im'),
            (prob / max(prob.max(), 1e-30),
                              col_titles[2], CMAP_PROB, False, False, '|ψ|²/max'),
            (phase,           col_titles[3], CMAP_CYCL, False, True,  'arg / rad'),
        ]

        for col, (data, title, cmap, symm, is_phase, cb_lbl) in enumerate(panels):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(f'{title}', fontsize=10, fontweight='bold')

            if is_phase:
                # Feste Grenzen [−π, π] für zyklisches Colormap
                _heatmap(ax, data, '', T, cmap,
                         vmin=-np.pi, vmax=np.pi, label=cb_lbl)
            else:
                _heatmap(ax, data, '', T, cmap,
                         symm=symm, label=cb_lbl)

            # Gittername links
            if col == 0:
                ax.set_ylabel(f'{row_label}\nt', fontsize=8)

    plt.tight_layout()
    out = f'{save_prefix}_T{T}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  → gespeichert  {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — 1D-Schnitte: Wellenmuster bei festen Zeitpunkten
# ══════════════════════════════════════════════════════════════════════════════

def fig_slices(psi_tri: np.ndarray, psi_sq: np.ndarray, T: int,
               epsilon: float, n_slices: int = 4,
               save_prefix: str = 'phase_slices') -> None:
    """
    Zeigt Re(ψ) und Im(ψ) als 1D-Kurven bei n_slices äquidistanten Zeitpunkten.
    Beide Gitter überlagert zum direkten Vergleich.
    """
    t_steps = np.linspace(T // (n_slices), T, n_slices, dtype=int)
    x       = np.arange(-T, T + 1)

    fig, axes = plt.subplots(n_slices, 2, figsize=(14, 3.2 * n_slices),
                              sharex=True)
    fig.suptitle(
        f'1D-Schnitte ψ(x) bei festen Zeitpunkten  ·  T={T}  ·  ε={epsilon}',
        fontsize=12, fontweight='bold')

    col_labels = ['Re(ψ)', 'Im(ψ)']
    c_tri = '#E8A838'
    c_sq  = '#4FC3F7'

    for i, t_step in enumerate(t_steps):
        for j, (fn, label) in enumerate([(np.real, col_labels[0]),
                                          (np.imag, col_labels[1])]):
            ax = axes[i, j]

            y_tri = fn(psi_tri[t_step])
            y_sq  = fn(psi_sq[t_step])

            # Auf gemeinsames Maximum normieren
            scale = max(np.abs(y_tri).max(), np.abs(y_sq).max(), 1e-30)

            ax.plot(x, y_tri / scale, color=c_tri, lw=1.6,
                    label='Dreieck')
            ax.plot(x, y_sq  / scale, color=c_sq,  lw=1.6, ls='--',
                    label='Quadrat')
            ax.fill_between(x, y_tri / scale, alpha=0.12, color=c_tri)

            # Lichtkegel-Grenzen bei ±t
            ax.axvline(-t_step, color='cyan', ls=':', lw=0.9, alpha=0.6)
            ax.axvline( t_step, color='cyan', ls=':', lw=0.9, alpha=0.6)
            ax.axhline(0, color='gray', lw=0.5, alpha=0.5)

            ax.set_xlim(-T, T)
            ax.set_ylim(-1.2, 1.2)
            ax.set_ylabel(f't = {t_step}\n{label}', fontsize=8)
            ax.tick_params(labelsize=7)

            if i == 0:
                ax.set_title(label, fontsize=10, fontweight='bold')
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='upper right', framealpha=0.7)
            if i == n_slices - 1:
                ax.set_xlabel('x', fontsize=9)

    plt.tight_layout()
    out = f'{save_prefix}_T{T}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  → gespeichert  {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Direkter Phasenvergleich Dreieck vs. Quadrat
# ══════════════════════════════════════════════════════════════════════════════

def fig_phase_comparison(psi_tri: np.ndarray, psi_sq: np.ndarray, T: int,
                         epsilon: float) -> None:
    """
    Zeigt arg(ψ) für beide Gitter nebeneinander.
    Hebt Unterschiede in der Phasensymmetrie hervor.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Phasenmuster  arg(ψ)  ·  T={T}  ·  ε={epsilon}  '
        '·  Weiße Bereiche: |ψ| < Schwellenwert',
        fontsize=11, fontweight='bold')

    titles = ['Dreiecksgitter', 'Quadratgitter', 'Phasendifferenz  Δφ']
    amp_max_tri = np.abs(psi_tri).max()
    amp_max_sq  = np.abs(psi_sq).max()

    phase_tri = np.angle(psi_tri).astype(float)
    phase_sq  = np.angle(psi_sq).astype(float)

    phase_tri[np.abs(psi_tri) < PHASE_THRESH * amp_max_tri] = np.nan
    phase_sq [np.abs(psi_sq)  < PHASE_THRESH * amp_max_sq ] = np.nan

    # Phasendifferenz nur wo beide Gitter über Schwelle
    both_valid = (~np.isnan(phase_tri)) & (~np.isnan(phase_sq))
    delta = np.full_like(phase_tri, np.nan)
    # Zyklische Differenz auf [−π, π]
    delta[both_valid] = np.angle(
        np.exp(1j * (phase_tri[both_valid] - phase_sq[both_valid])))

    datasets = [
        (phase_tri, CMAP_CYCL, -np.pi, np.pi, 'arg(ψ) / rad'),
        (phase_sq,  CMAP_CYCL, -np.pi, np.pi, 'arg(ψ) / rad'),
        (delta,     'PiYG',    -np.pi, np.pi, 'Δφ / rad'),
    ]

    for ax, (data, cmap, vmin, vmax, cb_lbl), title in zip(axes, datasets, titles):
        im = ax.imshow(data, origin='lower', aspect='auto',
                       extent=_extent(T), cmap=cmap,
                       vmin=vmin, vmax=vmax)
        _lightcone(ax, T)
        ax.set_xlim(-T, T)
        ax.set_ylim(0, T)
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('t', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cb_lbl, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = 'phase_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  → gespeichert  {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print('Schritt 1 — Komplexe Amplituden & Interferenzmuster')
    print('=' * 55)

    # ── Validierung ───────────────────────────────────────────────────────────
    print('\nKreuzvalidierung  |ψ|² == probs  (T=20, ε=0.1):')
    ok = validate(T=20, epsilon=0.1)
    if not ok:
        raise RuntimeError('Validierung fehlgeschlagen!')

    # ── Simulationen ──────────────────────────────────────────────────────────
    params = [
        (20,  0.1),
        (50,  0.1),
    ]

    for T, eps in params:
        print(f'\nSimuliere  T={T}  ε={eps}:')

        print(f'  Dreiecksgitter…', end='', flush=True)
        t0      = time()
        psi_tri = simulate_triangular_complex(T, eps)
        print(f' {time()-t0:.3f}s')

        print(f'  Quadratgitter…',  end='', flush=True)
        t0     = time()
        psi_sq = simulate_square_complex(T, eps)
        print(f' {time()-t0:.3f}s')

        print('\nGeneriere Figur 1 — Komponenten (Re, Im, |ψ|², arg)…')
        fig_components(psi_tri, psi_sq, T, eps)

        print('Generiere Figur 2 — 1D-Schnitte (Wellenmuster)…')
        fig_slices(psi_tri, psi_sq, T, eps)

    # Phasenvergleich nur für T=50 (aussagekräftiger)
    print('\nGeneriere Figur 3 — Phasenvergleich Dreieck vs. Quadrat (T=50)…')
    psi_tri50 = simulate_triangular_complex(50, 0.1)
    psi_sq50  = simulate_square_complex(50, 0.1)
    fig_phase_comparison(psi_tri50, psi_sq50, 50, 0.1)

    print('\nFertig. Gespeicherte Dateien:')
    for name in ['phase_components_T20.png', 'phase_components_T50.png',
                 'phase_slices_T20.png',      'phase_slices_T50.png',
                 'phase_comparison.png']:
        print(f'  {name}')


if __name__ == '__main__':
    main()
