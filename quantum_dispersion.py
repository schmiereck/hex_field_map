#!/usr/bin/env python3
"""
Schritt 2 — Dispersionsrelation: skaliert E²(p) = p² + m²?
============================================================
Test, ob beide Gitter die relativistische Dispersionsrelation reproduzieren.

Methode: Transfermatrix-Analyse
  Für eine ebene Welle ψ(x,t) = A(d)·e^{i(px−Et)} muss gelten:

      M(p) · A = λ · A   mit  λ = e^{−iE}

  M(p) hat Dimension n_dirs × n_dirs.  Eigenvalue → E = −arg(λ).

Transfermatrix (Dreiecksgitter, dirs = [−1, 0, +1]):
  M[d_new, d_old] = exp(−i·p·dirs[d_new]) · change[d_new, d_old]

Analytisch (Quadratgitter):
  λ = cos(p) ± i·√(sin²p + ε²)
  → E ≈ √(p²+ε²) für kleine p   (relativistisch korrekt mit m = ε)

Ausgabe-Plots:
  dispersion_curves.png    — alle Bänder E(p) beider Gitter vs. √(p²+m²)
  dispersion_residuals.png — E²(p)−p² in Abhängigkeit von p
  mass_scaling.png         — m_eff(ε) für beide Gitter (sollte linear sein)
  group_velocity.png       — vg = dE/dp ≤ 1 überall?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from time import time


# ══════════════════════════════════════════════════════════════════════════════
#  Transfermatrix
# ══════════════════════════════════════════════════════════════════════════════

def transfer_matrix_triangular(p: float, epsilon: float) -> np.ndarray:
    """
    3×3 Transfermatrix für das Dreiecksgitter bei Impuls p.

    M[d_new, d_old] = exp(−i·p·dirs[d_new]) · change[d_new, d_old]
    """
    dirs   = [-1, 0, 1]
    ie     = 1j * epsilon
    change = np.full((3, 3), ie, dtype=complex)
    np.fill_diagonal(change, 1.0 + 0j)

    M = np.empty((3, 3), dtype=complex)
    for d_new, dv in enumerate(dirs):
        M[d_new, :] = np.exp(-1j * p * dv) * change[d_new, :]
    return M


def transfer_matrix_square(p: float, epsilon: float) -> np.ndarray:
    """
    2×2 Transfermatrix für das Quadratgitter (Feynman-Checkerboard) bei p.

    Analytisch:  λ = cos(p) ± i·√(sin²p + ε²)
    """
    ie     = 1j * epsilon
    change = np.array([[1.0 + 0j, ie],
                       [ie,  1.0 + 0j]])
    # dirs = [−1, +1]
    M = np.empty((2, 2), dtype=complex)
    M[0, :] = np.exp( 1j * p) * change[0, :]   # d_new=0 (links,  dv=−1)
    M[1, :] = np.exp(-1j * p) * change[1, :]   # d_new=1 (rechts, dv=+1)
    return M


def transfer_matrix_square_analytical(p: np.ndarray, epsilon: float
                                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytische Eigenvalues des Quadratgitters (Formel aus Docstring oben).
    Gibt (λ_plus, λ_minus) zurück, je Form (n_p,).
    """
    lam_plus  = np.cos(p) + 1j * np.sqrt(np.sin(p)**2 + epsilon**2)
    lam_minus = np.cos(p) - 1j * np.sqrt(np.sin(p)**2 + epsilon**2)
    return lam_plus, lam_minus


# ══════════════════════════════════════════════════════════════════════════════
#  Eigenvalue → Energiebänder
# ══════════════════════════════════════════════════════════════════════════════

def compute_bands(epsilon: float, lattice: str,
                  n_p: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet alle Energiebänder E_k(p) = −arg(λ_k) für ein Gitter.

    Returns
    -------
    p_arr   : (n_p,)             Impulse in [−π, π]
    E_bands : (n_p, n_bands)     Energien aller Bänder, sortiert aufsteigend
    g_bands : (n_p, n_bands)     Amplitudenwachstum log|λ_k|
    """
    p_arr  = np.linspace(-np.pi, np.pi, n_p, endpoint=False)
    fn     = (transfer_matrix_triangular if lattice == 'triangular'
              else transfer_matrix_square)
    n_b    = 3 if lattice == 'triangular' else 2

    E_out = np.empty((n_p, n_b))
    g_out = np.empty((n_p, n_b))

    for i, p in enumerate(p_arr):
        evals     = np.linalg.eigvals(fn(p, epsilon))
        E_raw     = -np.angle(evals)          # E = −arg(λ)
        idx       = np.argsort(E_raw)
        E_out[i]  = E_raw[idx]
        g_out[i]  = np.log(np.abs(evals[idx]))

    return p_arr, E_out, g_out


def physical_band(p_arr: np.ndarray, E_bands: np.ndarray,
                  m_est: float = None) -> np.ndarray:
    """
    Identifiziert das physikalische positive Energieband — dasjenige, das
    der relativistischen Dispersion E_rel = √(p²+m²) am nächsten liegt.

    Beim Dreiecksgitter gibt es eine zusätzliche "Geradeaus"-Mode nahe E≈0,
    die vom physikalischen Band unterschieden werden muss.

    Algorithmus:
      1. Schätze m_est = kleinste positive E(p=0) aller Bänder.
      2. Berechne Referenzkurve E_ref = √(p²+m_est²).
      3. Wähle für jeden p-Wert das Band, dessen Energie E_ref am nächsten ist.
    """
    p0_idx = np.argmin(np.abs(p_arr))
    e_at_0 = E_bands[p0_idx, :]
    pos    = e_at_0[e_at_0 > 0]
    if len(pos) == 0:
        return E_bands[:, -1]

    if m_est is None:
        m_est = pos.min()                      # konservative Schätzung

    E_ref  = np.sqrt(p_arr**2 + m_est**2)     # relativistische Referenz

    # Für jeden p den nächstliegenden Band-Wert wählen
    result = np.empty(len(p_arr))
    for i in range(len(p_arr)):
        dists     = np.abs(E_bands[i, :] - E_ref[i])
        result[i] = E_bands[i, np.argmin(dists)]
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Effektive Masse & Fit
# ══════════════════════════════════════════════════════════════════════════════

def effective_mass(p_arr: np.ndarray, E_phys: np.ndarray) -> float:
    """m_eff = E(p≈0) des physikalischen Bandes."""
    p0 = np.argmin(np.abs(p_arr))
    return float(np.abs(E_phys[p0]))


def relativistic_dispersion(p: np.ndarray, m: float) -> np.ndarray:
    """E_rel(p) = √(p² + m²)"""
    return np.sqrt(p**2 + m**2)


def fit_mass(p_arr: np.ndarray, E_phys: np.ndarray,
             p_max: float = np.pi / 4) -> tuple[float, float]:
    """
    Passt E = √(p²+m²) an den Bereich |p| < p_max an.
    Gibt (m_fit, rmse) zurück.
    """
    mask = np.abs(p_arr) < p_max
    try:
        (m_fit,), _ = curve_fit(relativistic_dispersion,
                                p_arr[mask], E_phys[mask],
                                p0=[effective_mass(p_arr, E_phys)],
                                bounds=(0, np.inf))
        rmse = float(np.sqrt(np.mean(
            (E_phys[mask] - relativistic_dispersion(p_arr[mask], m_fit))**2)))
    except Exception:
        m_fit = effective_mass(p_arr, E_phys)
        rmse  = float('nan')
    return m_fit, rmse


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Dispersionsrelation E(p) für mehrere ε
# ══════════════════════════════════════════════════════════════════════════════

def fig_dispersion_curves(epsilons: list, n_p: int = 600) -> None:
    """
    Grid: Zeilen = ε-Werte, Spalten = Dreieck | Quadrat.
    Jeder Panel zeigt alle Bänder, das relativistische Limit und den Lichtkegel.
    """
    n_eps = len(epsilons)
    fig, axes = plt.subplots(n_eps, 2, figsize=(13, 4 * n_eps), sharey='row')
    fig.suptitle(
        'Dispersionsrelation  E(p)  beider Gitter\n'
        'Gestrichelt: relativistisches Limit  E = √(p²+m²)  ·  '
        'Grau gepunktet: Lichtkegel  |E|=|p|',
        fontsize=12, fontweight='bold')

    configs = [('triangular', '#E8A838', 'Dreiecksgitter (3 Züge/Schritt)'),
               ('square',     '#4FC3F7', 'Quadratgitter (Feynman-Checkerboard)')]

    for row, eps in enumerate(epsilons):
        for col, (lattice, color, title) in enumerate(configs):
            ax = axes[row, col] if n_eps > 1 else axes[col]

            p_arr, E_bands, g_bands = compute_bands(eps, lattice, n_p)
            E_phys = physical_band(p_arr, E_bands)
            m_fit, rmse = fit_mass(p_arr, E_phys)
            E_rel = relativistic_dispersion(p_arr, m_fit)

            n_b = E_bands.shape[1]

            # Alle Bänder (hell)
            for b in range(n_b):
                alpha = 0.35 if n_b > 1 else 0.9
                ax.plot(p_arr, E_bands[:, b], '-', color=color,
                        lw=1.0, alpha=alpha)

            # Physikalisches Band (kräftig)
            ax.plot(p_arr, E_phys, '-', color=color, lw=2.2,
                    label=f'E_phys  (m_eff={m_fit:.4f})')

            # Relativistisches Limit
            ax.plot(p_arr,  E_rel, 'k--', lw=1.5, alpha=0.75,
                    label=f'√(p²+{m_fit:.4f}²)  rmse={rmse:.4f}')
            ax.plot(p_arr, -E_rel, 'k--', lw=1.5, alpha=0.75)

            # Lichtkegel
            ax.plot(p_arr,  p_arr, ':', color='gray', lw=0.8, alpha=0.6,
                    label='Lichtkegel |E|=|p|')
            ax.plot(p_arr, -p_arr, ':', color='gray', lw=0.8, alpha=0.6)

            ax.axhline(0, color='black', lw=0.5, alpha=0.3)
            ax.axvline(0, color='black', lw=0.5, alpha=0.3)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xlabel('Impuls  p', fontsize=9)
            ax.set_ylabel('Energie  E', fontsize=9)
            ax.set_title(f'{title}  ε={eps}', fontsize=9)
            ax.legend(fontsize=7, framealpha=0.7)
            ax.tick_params(labelsize=7)
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    plt.tight_layout()
    out = 'dispersion_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  → gespeichert  {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Residuen E²(p) − p² − m²
# ══════════════════════════════════════════════════════════════════════════════

def fig_residuals(epsilon: float = 0.1, n_p: int = 600) -> None:
    """
    Zeigt E²(p) − p² für das physikalische Band beider Gitter.
    Für ein perfekt relativistisches Gitter wäre das eine Horizontale bei m².
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    fig.suptitle(
        f'Abweichung von der relativistischen Dispersionsrelation  (ε={epsilon})\n'
        'Blau: E²(p)−p²  ·  Gestrichelt: m_eff²  ·  '
        'Perfekt relativistisch ⟺ flache Kurve',
        fontsize=11, fontweight='bold')

    configs = [('triangular', '#E8A838', 'Dreiecksgitter'),
               ('square',     '#4FC3F7', 'Quadratgitter')]

    for ax, (lattice, color, title) in zip(axes, configs):
        p_arr, E_bands, _ = compute_bands(epsilon, lattice, n_p)
        E_phys = physical_band(p_arr, E_bands)
        m_eff  = effective_mass(p_arr, E_phys)
        m_fit, _ = fit_mass(p_arr, E_phys)

        residual = E_phys**2 - p_arr**2     # sollte ≈ m² sein

        ax.plot(p_arr, residual, '-', color=color, lw=2.0,
                label='E²(p) − p²')
        ax.axhline(m_eff**2,  color='gray',  ls='--', lw=1.2,
                   label=f'm_eff²  = {m_eff**2:.5f}')
        ax.axhline(m_fit**2, color='black', ls=':', lw=1.2,
                   label=f'm_fit²  = {m_fit**2:.5f}')
        ax.axhline(epsilon**2, color='red', ls='-.', lw=0.9, alpha=0.6,
                   label=f'ε²       = {epsilon**2:.5f}')

        # Lichtkegel-Grenze eintragen
        ax.axvline(-np.pi/2, color='silver', ls=':', lw=1)
        ax.axvline( np.pi/2, color='silver', ls=':', lw=1)
        ax.text( np.pi/2 + 0.05, ax.get_ylim()[0] if ax.get_ylim()[0] else 0,
                 '±π/2', fontsize=7, color='silver', va='bottom')

        ax.set_xlim(-np.pi, np.pi)
        ax.set_xlabel('Impuls  p', fontsize=9)
        ax.set_ylabel('E²(p) − p²', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.tick_params(labelsize=8)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    plt.tight_layout()
    out = 'dispersion_residuals.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  → gespeichert  {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Effektive Masse m_eff(ε)
# ══════════════════════════════════════════════════════════════════════════════

def fig_mass_scaling(n_p: int = 300) -> None:
    """
    Zeigt m_eff(ε) für beide Gitter.
    Erwartet: m_eff ≈ c·ε (linear) für kleine ε.
    """
    epsilons = np.logspace(-2, 0, 40)           # 0.01 … 1.0
    m_tri    = np.empty_like(epsilons)
    m_sq     = np.empty_like(epsilons)

    print('  Berechne m_eff(ε)…', end='', flush=True)
    for i, eps in enumerate(epsilons):
        for j, (lattice, arr) in enumerate(
                [('triangular', m_tri), ('square', m_sq)]):
            p_arr, E_bands, _ = compute_bands(eps, lattice, n_p)
            E_phys = physical_band(p_arr, E_bands)
            arr[i] = effective_mass(p_arr, E_phys)
    print(' fertig')

    # Lineare Fits für kleine ε (< 0.3)
    mask = epsilons < 0.3
    def lin(x, a): return a * x
    (a_tri,), _ = curve_fit(lin, epsilons[mask], m_tri[mask])
    (a_sq,),  _ = curve_fit(lin, epsilons[mask], m_sq[mask])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Effektive Masse  m_eff(ε)  —  Skalierung mit dem Masseparameter',
                 fontsize=12, fontweight='bold')

    # Lineare Achse
    ax = axes[0]
    ax.plot(epsilons, m_tri, 'o-', color='#E8A838', ms=3, lw=1.5,
            label=f'Dreieck   slope={a_tri:.4f}')
    ax.plot(epsilons, m_sq,  's-', color='#4FC3F7', ms=3, lw=1.5,
            label=f'Quadrat   slope={a_sq:.4f}')
    ax.plot(epsilons, epsilons, 'k--', lw=0.9, alpha=0.5,
            label='m = ε  (Referenz)')
    ax.plot(epsilons, a_tri * epsilons, '--', color='#E8A838', lw=0.8, alpha=0.6)
    ax.plot(epsilons, a_sq  * epsilons, '--', color='#4FC3F7', lw=0.8, alpha=0.6)
    ax.set_xlabel('ε  (Masseparameter)', fontsize=9)
    ax.set_ylabel('m_eff = E(p=0)',       fontsize=9)
    ax.set_title('Lineare Skala',          fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # Log-Log-Achse (zeigt Potenzgesetz klar)
    ax = axes[1]
    ax.loglog(epsilons, m_tri, 'o-', color='#E8A838', ms=3, lw=1.5,
              label=f'Dreieck   slope={a_tri:.4f}')
    ax.loglog(epsilons, m_sq,  's-', color='#4FC3F7', ms=3, lw=1.5,
              label=f'Quadrat   slope={a_sq:.4f}')
    ax.loglog(epsilons, epsilons, 'k--', lw=0.9, alpha=0.5,
              label='m = ε  (Referenz, slope=1)')
    ax.set_xlabel('ε  (Masseparameter)', fontsize=9)
    ax.set_ylabel('m_eff = E(p=0)',       fontsize=9)
    ax.set_title('Log-Log (zeigt Potenzgesetz)',  fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = 'mass_scaling.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  → gespeichert  {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 4 — Gruppengeschwindigkeit vg = dE/dp
# ══════════════════════════════════════════════════════════════════════════════

def fig_group_velocity(epsilon: float = 0.1, n_p: int = 600) -> None:
    """
    vg(p) = dE/dp darf max 1 sein (Kausalität).
    Relativistisch: vg = p / √(p²+m²) < 1.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    fig.suptitle(
        f'Gruppengeschwindigkeit  vg = dE/dp  (ε={epsilon})\n'
        'Kausalitätsgrenze: |vg| ≤ 1  ·  '
        'Relativistisch: vg = p/√(p²+m²)',
        fontsize=11, fontweight='bold')

    configs = [('triangular', '#E8A838', 'Dreiecksgitter'),
               ('square',     '#4FC3F7', 'Quadratgitter')]

    for ax, (lattice, color, title) in zip(axes, configs):
        p_arr, E_bands, _ = compute_bands(epsilon, lattice, n_p)
        E_phys = physical_band(p_arr, E_bands)
        m_eff  = effective_mass(p_arr, E_phys)

        dp   = p_arr[1] - p_arr[0]
        vg   = np.gradient(E_phys, dp)

        E_rel = relativistic_dispersion(p_arr, m_eff)
        vg_rel = p_arr / np.maximum(E_rel, 1e-10)

        ax.plot(p_arr, vg,     '-', color=color,   lw=2.0, label='vg(p) Gitter')
        ax.plot(p_arr, vg_rel, 'k--', lw=1.3, alpha=0.7,
                label='vg = p/√(p²+m²)  (relativistisch)')
        ax.axhline( 1, color='red', ls='--', lw=0.9, alpha=0.7,
                   label='Kausalitätsgrenze |vg|=1')
        ax.axhline(-1, color='red', ls='--', lw=0.9, alpha=0.7)
        ax.axhline( 0, color='black', lw=0.4, alpha=0.3)
        ax.axvline( 0, color='black', lw=0.4, alpha=0.3)

        vg_max = float(np.max(np.abs(vg[10:-10])))   # Randeffekte abschneiden
        ax.text(0.05, 0.95, f'max|vg| = {vg_max:.4f}',
                transform=ax.transAxes, fontsize=9,
                va='top', bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'))

        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel('Impuls  p', fontsize=9)
        ax.set_ylabel('Gruppengeschwindigkeit  vg', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.tick_params(labelsize=8)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    plt.tight_layout()
    out = 'group_velocity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  → gespeichert  {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Numerische Zusammenfassung
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(n_p: int = 600) -> None:
    """Tabellarische Zusammenfassung für mehrere ε-Werte."""
    epsilons = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

    # Analytische Formel (Quadratgitter): E(p) = arctan(√(sin²p+ε²)/cos p)
    # → m_eff_analytic = arctan(ε)  für p=0
    print()
    print('Zusammenfassung Dispersionsrelation')
    print('=' * 80)
    print(f'  {"ε":>6}  {"m_tri":>8}  {"m_sq":>8}  '
          f'{"m_analytic":>10}  {"slope_tri":>10}  {"rmse_tri":>9}  '
          f'{"slope_sq":>9}  {"rmse_sq":>8}')
    print('  ' + '-' * 78)

    for eps in epsilons:
        m_anal = np.arctan(eps)   # analytisch für Quadratgitter
        results = {}
        for lattice in ('triangular', 'square'):
            p_arr, E_bands, _ = compute_bands(eps, lattice, n_p)
            E_phys = physical_band(p_arr, E_bands)
            m_eff  = effective_mass(p_arr, E_phys)
            m_fit, rmse = fit_mass(p_arr, E_phys, p_max=np.pi/4)
            results[lattice] = (m_eff, m_fit, rmse)

        m_tri, mfit_tri, rmse_tri = results['triangular']
        m_sq,  mfit_sq,  rmse_sq  = results['square']

        print(f'  {eps:>6.3f}  {m_tri:>8.5f}  {m_sq:>8.5f}  '
              f'{m_anal:>10.5f}  {mfit_tri/eps:>10.5f}  {rmse_tri:>9.6f}  '
              f'{mfit_sq/eps:>9.5f}  {rmse_sq:>8.6f}')

    print()
    print('Legende:')
    print('  m_eff     = E(p=0)  direkt aus Band')
    print('  m_analytic= arctan(ε)  (exakt für Quadratgitter)')
    print('  slope     = m_fit/ε  (sollte → 1 für kleines ε)')
    print('  rmse      = Fit-Fehler für |p| < π/4')


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print('Schritt 2 — Dispersionsrelation: E²(p) = p² + m²?')
    print('=' * 55)
    print()

    # ── Numerische Zusammenfassung ────────────────────────────────────────────
    print_summary()

    # ── Figure 1: Dispersionsrelation für mehrere ε ───────────────────────────
    print('Generiere Figure 1 — Dispersionsrelation E(p)…')
    fig_dispersion_curves(epsilons=[0.05, 0.1, 0.5])

    # ── Figure 2: Residuen ────────────────────────────────────────────────────
    print('Generiere Figure 2 — Residuen E²(p)−p²…')
    fig_residuals(epsilon=0.1)

    # ── Figure 3: Masse-Skalierung ────────────────────────────────────────────
    print('Generiere Figure 3 — Massenparameter-Skalierung…')
    fig_mass_scaling()

    # ── Figure 4: Gruppengeschwindigkeit ─────────────────────────────────────
    print('Generiere Figure 4 — Gruppengeschwindigkeit vg(p)…')
    fig_group_velocity(epsilon=0.1)

    print()
    print('Fertig. Gespeicherte Dateien:')
    for name in ['dispersion_curves.png', 'dispersion_residuals.png',
                 'mass_scaling.png',       'group_velocity.png']:
        print(f'  {name}')


if __name__ == '__main__':
    main()
