#!/usr/bin/env python3
"""
Physically-correct dispersion relation analysis for all three lattice models.

Physical time is used as the evolution variable for the equilateral lattice.
Diagonal moves advance t_phys by 0.5; straight-up move advances by 1.0.

Transfer-matrix derivation
--------------------------
For each model we build M(k) such that ψ(t+Δt) = M(k) ψ(t).
Eigenvalues λ → E = -arg(λ) / Δt   [energy per physical time unit].

Models and time units
---------------------
1. Feynman Checkerboard (CB)   2×2   Δt=1   c_expected=1
2. Square + Rest        (SR)   3×3   Δt=1   c_expected≈1
3. Equilateral 3-move   (EQ)   6×6   Δτ=½   c_expected=√3

For EQ, one half-step transfers the state.  To get E per *physical* time unit
(Δt=1 = 2 half-steps) we square the half-step matrix:
   M_full = M_half²   →   λ_full = λ_half²   →   E = -arg(λ_full) = -2·arg(λ_half).

Physical x-spacing
------------------
CB / SR : Δx = 1   → physical k = p_idx
EQ      : Δx = √3/2 → physical k = p_idx / (√3/2)

Expected speed of light (massless, continuum limit)
----------------------------------------------------
CB diagonal: Δx=1, Δt=1   → c = 1
EQ diagonal: Δx=√3/2, Δt=0.5 → c_half = (√3/2)/0.5 = √3  per half-step
             in full-step units: c = 2·(√3/2)/1 = √3  (same value)
So EQ satisfies E² = 3k² + m² (not k²+m²).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SQRT3_2 = np.sqrt(3) / 2   # ≈ 0.866

# ─────────────────────────── helpers ────────────────────────────────────────

def _C(n, eps):
    """n×n change matrix: 1 on diagonal, iε off-diagonal."""
    C = np.full((n, n), 1j * eps, dtype=complex)
    np.fill_diagonal(C, 1.0 + 0j)
    return C


# ─────────────────────────── transfer matrices ──────────────────────────────

def TM_CB(k, eps):
    """2×2 Feynman Checkerboard.  k = physical momentum (Δx=1)."""
    C = _C(2, eps)
    M = np.empty((2, 2), dtype=complex)
    M[0, :] = np.exp( 1j * k) * C[0, :]   # dx=-1
    M[1, :] = np.exp(-1j * k) * C[1, :]   # dx=+1
    return M


def TM_SR(k, eps):
    """3×3 Square+Rest.  k = physical momentum (Δx=1)."""
    C = _C(3, eps)
    M = np.empty((3, 3), dtype=complex)
    M[0, :] = np.exp( 1j * k) * C[0, :]   # dx=-1
    M[1, :] = C[1, :]                       # dx=0
    M[2, :] = np.exp(-1j * k) * C[2, :]   # dx=+1
    return M


def TM_EQ_half(k, eps):
    """
    6×6 half-step transfer matrix for the equilateral lattice.

    Physical x-spacing Δx = √3/2  →  p_idx = k * √3/2.
    State vector: (G_curr[3], G_prev[3]).

    Update per half-step τ:
      G[τ+1, d=0] = exp(+ip) · C[0,:] @ G[τ]     (left-diag,  τ-dep)
      G[τ+1, d=1] = C[1,:] @ G[τ-1]               (straight,   τ-1 dep)
      G[τ+1, d=2] = exp(-ip) · C[2,:] @ G[τ]     (right-diag, τ-dep)
    """
    C = _C(3, eps)
    p = k * SQRT3_2   # index-space momentum

    A = np.zeros((3, 3), dtype=complex)
    A[0, :] = np.exp( 1j * p) * C[0, :]
    # A[1,:] = 0  -- straight uses G_prev
    A[2, :] = np.exp(-1j * p) * C[2, :]

    B = np.zeros((3, 3), dtype=complex)
    B[1, :] = C[1, :]                      # straight from G_prev

    M6 = np.zeros((6, 6), dtype=complex)
    M6[:3, :3] = A
    M6[:3, 3:] = B
    M6[3:, :3] = np.eye(3)                 # G_curr → G_prev
    return M6


def TM_EQ_full(k, eps):
    """6×6 full-step (=2 half-steps) EQ matrix: M_full = M_half²."""
    H = TM_EQ_half(k, eps)
    return H @ H


# ─────────────────────────── band computation ───────────────────────────────

def bands_from_TM(TM_fn, k_arr, eps):
    """
    Compute E = -arg(λ) bands at each k.
    Returns E_bands (n_k, n_bands), sorted per row.
    """
    evals0 = np.linalg.eigvals(TM_fn(k_arr[0], eps))
    n_b = len(evals0)
    E_bands = np.empty((len(k_arr), n_b))
    for i, k in enumerate(k_arr):
        lam = np.linalg.eigvals(TM_fn(k, eps))
        E_bands[i] = np.sort(-np.angle(lam))
    return E_bands


def physical_band(k_arr, E_bands, c_est=1.0, m_est=None):
    """
    Track the physical band closest to E_ref = sqrt(c²k² + m²).
    """
    k0 = np.argmin(np.abs(k_arr))
    pos = E_bands[k0][E_bands[k0] > 0]
    if m_est is None:
        m_est = pos.min() if len(pos) else 0.01
    E_ref = np.sqrt(c_est**2 * k_arr**2 + m_est**2)
    out = np.empty(len(k_arr))
    for i in range(len(k_arr)):
        diffs = np.abs(E_bands[i] - E_ref[i])
        out[i] = E_bands[i, np.argmin(diffs)]
    return out


def fit_rel(k_arr, E_phys, k_max=np.pi/4):
    """Fit E = sqrt(c²k² + m²).  Returns (c_fit, m_fit, rmse)."""
    mask = np.abs(k_arr) < k_max
    def model(k, c, m): return np.sqrt(c**2 * k**2 + m**2)
    k0 = np.argmin(np.abs(k_arr))
    m0 = max(E_phys[k0], 1e-4)
    try:
        (c_fit, m_fit), _ = curve_fit(
            model, k_arr[mask], E_phys[mask],
            p0=[1.0, m0], bounds=([0, 0], [10, 10]))
        rmse = float(np.sqrt(np.mean(
            (E_phys[mask] - model(k_arr[mask], c_fit, m_fit))**2)))
    except Exception:
        c_fit, m_fit, rmse = 1.0, m0, float('nan')
    return c_fit, m_fit, rmse


# ─────────────────────────── physical-time simulation ───────────────────────

def simulate_EQ_phys(T_phys, eps=0.1):
    """
    Equilateral lattice, physical time as evolution variable.

    Uses half-step resolution (Δτ = 0.5).  Stores psi at every integer
    physical time (τ even).

    Second-order recurrence:
      amp_next[i, 0] = Σ_d C[0,d] · amp_curr[i+1, d]   (left-diag)
      amp_next[i, 1] = Σ_d C[1,d] · amp_prev[i,   d]   (straight, 2τ ago)
      amp_next[i, 2] = Σ_d C[2,d] · amp_curr[i-1, d]   (right-diag)

    Returns psi[T_phys+1, N], N=2T_phys+1.
    Physical x[i] = (i - T_phys) * √3/2.
    """
    N      = 2 * T_phys + 1
    center = T_phys
    C      = _C(3, eps)

    # amp_pprev = τ-2, amp_prev = τ-1 (will be updated)
    amp_pprev = np.zeros((N, 3), dtype=complex)
    amp_prev  = np.zeros((N, 3), dtype=complex)
    amp_prev[center, :] = 1.0 / 3.0   # t_phys = 0

    N_half = 2 * T_phys   # number of half-steps to run
    psi    = np.zeros((T_phys + 1, N), dtype=complex)
    psi[0] = amp_prev.sum(axis=1)

    for tau in range(1, N_half + 1):
        new_amp = np.zeros((N, 3), dtype=complex)

        # left-diagonal: arrives from i+1 (one half-step ago)
        w = amp_prev @ C[0]
        new_amp[:-1, 0] += w[1:]

        # straight-up: arrives from i (two half-steps ago)
        w = amp_pprev @ C[1]
        new_amp[:, 1] += w

        # right-diagonal: arrives from i-1 (one half-step ago)
        w = amp_prev @ C[2]
        new_amp[1:, 2] += w[:-1]

        amp_pprev = amp_prev
        amp_prev  = new_amp

        if tau % 2 == 0:   # integer physical time step
            t_idx = tau // 2
            psi[t_idx] = new_amp.sum(axis=1)

    return psi    # x[i] = (i - T_phys) * √3/2


def simulate_CB_phys(T, eps=0.1):
    """Feynman Checkerboard (for comparison). x[i] = i - T."""
    N, center = 2 * T + 1, T
    C = _C(2, eps)
    amp = np.zeros((N, 2), dtype=complex)
    amp[center, :] = 0.5
    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0] = amp.sum(axis=1)
    for step in range(1, T + 1):
        new_amp = np.zeros((N, 2), dtype=complex)
        w = amp @ C[0]; new_amp[:-1, 0] += w[1:]
        w = amp @ C[1]; new_amp[1:,  1] += w[:-1]
        amp = new_amp
        psi[step] = amp.sum(axis=1)
    return psi


def simulate_SR_phys(T, eps=0.1):
    """Square+Rest (for comparison). x[i] = i - T."""
    N, center = 2 * T + 1, T
    C = _C(3, eps)
    amp = np.zeros((N, 3), dtype=complex)
    amp[center, :] = 1.0 / 3.0
    psi = np.zeros((T + 1, N), dtype=complex)
    psi[0] = amp.sum(axis=1)
    for step in range(1, T + 1):
        new_amp = np.zeros((N, 3), dtype=complex)
        w = amp @ C[0]; new_amp[:-1, 0] += w[1:]
        w = amp @ C[1]; new_amp[:, 1]   += w
        w = amp @ C[2]; new_amp[1:,  2] += w[:-1]
        amp = new_amp
        psi[step] = amp.sum(axis=1)
    return psi


# ─────────────────────────── figures ────────────────────────────────────────

MODELS = [
    ('CB',  TM_CB,      1.0,   '#4FC3F7', 'Feynman Checkerboard\n(2 moves, Δx=1)'),
    ('SR',  TM_SR,      1.0,   '#E8A838', 'Square + Rest\n(3 moves, Δx=1)'),
    ('EQ',  TM_EQ_full, SQRT3_2, '#7ED957', 'Equilateral (phys. time)\n(3 moves, Δx=√3/2)'),
]


def fig_dispersion(eps=0.1, n_p=600):
    """
    Two-panel figure:
    Left : E(k) bands for all models + relativistic fit sqrt(c²k²+m²)
    Right: residual E²(k) - (c²k² + m²) / m² (normalised)
    """
    k_arr = np.linspace(0, np.pi, n_p)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'Dispersionsrelation – physikalische Zeit  (ε={eps})\n'
        'EQ full-step = M_half²  →  Energie pro physikalischer Zeiteinheit',
        fontsize=11)

    ax_e, ax_r = axes

    for name, TM_fn, c_est, col, label in MODELS:
        E_bands = bands_from_TM(TM_fn, k_arr, eps)
        E_phys  = physical_band(k_arr, E_bands, c_est=c_est)
        c_fit, m_fit, rmse = fit_rel(k_arr, E_phys, k_max=np.pi/4)
        E_ref = np.sqrt(c_fit**2 * k_arr**2 + m_fit**2)

        # all bands (faint)
        for b in range(E_bands.shape[1]):
            ax_e.plot(k_arr, E_bands[:, b], '-', color=col, lw=0.8, alpha=0.25)

        ax_e.plot(k_arr, E_phys, '-', color=col, lw=2.2,
                  label=f'{name}  c={c_fit:.3f}  m={m_fit:.4f}  rmse={rmse:.4f}')
        ax_e.plot(k_arr, E_ref, '--', color=col, lw=1.2, alpha=0.7)

        # residual: (E²  - E_ref²) / m²
        resid = (E_phys**2 - E_ref**2) / (m_fit**2 + 1e-30)
        ax_r.plot(k_arr, resid, '-', color=col, lw=1.8,
                  label=f'{name}  (c={c_fit:.3f})')

    # light-cone reference
    ax_e.plot(k_arr, k_arr, 'k:', lw=1, alpha=0.4, label='|E|=k (c=1)')
    ax_e.set_xlim(0, np.pi); ax_e.set_ylim(0, np.pi)
    ax_e.set_xlabel('k  (physical momentum)'); ax_e.set_ylabel('E  (per time unit)')
    ax_e.set_title('Energy bands E(k)'); ax_e.legend(fontsize=7); ax_e.grid(alpha=0.3)

    ax_r.axhline(0, color='k', lw=0.8, ls='--')
    ax_r.set_xlim(0, np.pi)
    ax_r.set_xlabel('k'); ax_r.set_ylabel('(E² - c²k² - m²) / m²')
    ax_r.set_title('Relativistic residual (=0 means perfect E²=c²k²+m²)')
    ax_r.legend(fontsize=7); ax_r.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('dispersion_phys.png', dpi=150, bbox_inches='tight')
    print('Saved dispersion_phys.png')
    plt.close()


def fig_c_and_m_scaling(n_p=400):
    """
    c_fit and m_fit as function of ε for all three models.
    Checks: m ∝ ε  and  c ≈ const (model-dependent).
    """
    epsilons = np.linspace(0.02, 0.5, 20)
    k_arr = np.linspace(0, np.pi, n_p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Speed of light c  and  mass m  vs  ε', fontsize=11)
    ax_c, ax_m = axes

    for name, TM_fn, c_est, col, label in MODELS:
        cs, ms = [], []
        for eps in epsilons:
            E_bands = bands_from_TM(TM_fn, k_arr, eps)
            E_phys  = physical_band(k_arr, E_bands, c_est=c_est)
            c_fit, m_fit, _ = fit_rel(k_arr, E_phys, k_max=np.pi/4)
            cs.append(c_fit); ms.append(m_fit)
        ax_c.plot(epsilons, cs, 'o-', color=col, lw=1.8, ms=4, label=name)
        ax_m.plot(epsilons, ms, 'o-', color=col, lw=1.8, ms=4, label=name)

    # reference m = ε
    ax_m.plot(epsilons, epsilons, 'k--', lw=1, alpha=0.5, label='m = ε')

    ax_c.set_xlabel('ε'); ax_c.set_ylabel('c_fit')
    ax_c.set_title('Speed of light  (should be const w.r.t. ε)')
    ax_c.legend(fontsize=8); ax_c.grid(alpha=0.3)
    # annotation for expected c
    ax_c.axhline(1.0,    color='#4FC3F7', lw=0.8, ls=':')
    ax_c.axhline(SQRT3_2*2, color='#7ED957', lw=0.8, ls=':')   # √3

    ax_m.set_xlabel('ε'); ax_m.set_ylabel('m_fit')
    ax_m.set_title('Effective mass  (should ∝ ε)')
    ax_m.legend(fontsize=8); ax_m.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('dispersion_cm_scaling.png', dpi=150, bbox_inches='tight')
    print('Saved dispersion_cm_scaling.png')
    plt.close()


def fig_probability_comparison(T=30, eps=0.1):
    """
    |ψ(x,t)|² for CB, SR, EQ (physical-time simulations) side by side.
    EQ x-axis in physical units (×√3/2).
    """
    print(f'  simulating CB (T={T})...', flush=True)
    psi_CB = simulate_CB_phys(T, eps)
    print(f'  simulating SR (T={T})...', flush=True)
    psi_SR = simulate_SR_phys(T, eps)
    print(f'  simulating EQ (T={T}, phys-time)...', flush=True)
    psi_EQ = simulate_EQ_phys(T, eps)

    prob_CB = np.abs(psi_CB)**2
    prob_SR = np.abs(psi_SR)**2
    prob_EQ = np.abs(psi_EQ)**2

    xCB = np.arange(2*T+1) - T
    xSR = xCB.copy()
    xEQ = (np.arange(2*T+1) - T) * SQRT3_2

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f'Physical-time simulation  |ψ(x,t)|²   (T={T}, ε={eps})\n'
        f'EQ: half-step τ, stored at integer t_phys.  x_EQ in units of √3/2',
        fontsize=11)

    kw = dict(aspect='auto', origin='lower', cmap='inferno')
    data = [
        (prob_CB, xCB, 'Feynman CB\n(±1,+1)'),
        (prob_SR, xSR, 'Square+Rest\n(-1/0/+1,+1)'),
        (prob_EQ, xEQ, 'Equilateral (phys-time)\n(±√3/2,½) & (0,1)'),
    ]
    for col, (prob, xs, ttl) in enumerate(data):
        dx = abs(xs[1] - xs[0]) if len(xs) > 1 else 1
        ext = [xs[0]-dx/2, xs[-1]+dx/2, -0.5, T+0.5]
        im = axes[0, col].imshow(prob, extent=ext, **kw)
        axes[0, col].set_title(ttl)
        axes[0, col].set_xlabel('x (physical)'); axes[0, col].set_ylabel('t_phys')
        plt.colorbar(im, ax=axes[0, col])

    slices_t = [T//4, T//2, 3*T//4, T]
    colors_sl = plt.cm.plasma(np.linspace(0.2, 0.9, 4))
    for t_idx, col in zip(slices_t, colors_sl):
        axes[1, 0].plot(xCB, prob_CB[t_idx], color=col, label=f't={t_idx}')
        axes[1, 1].plot(xSR, prob_SR[t_idx], color=col, label=f't={t_idx}')
        axes[1, 2].plot(xEQ, prob_EQ[t_idx], color=col, label=f't={t_idx}')

    for ax_c, ttl in zip(axes[1], [d[2] for d in data]):
        ax_c.set_xlabel('x'); ax_c.set_ylabel('|ψ|²')
        ax_c.set_title(ttl); ax_c.legend(fontsize=7); ax_c.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulation_phys_comparison.png', dpi=150, bbox_inches='tight')
    print('Saved simulation_phys_comparison.png')
    plt.close()


def fig_group_velocity(eps=0.1, n_p=600):
    """Group velocity v_g = dE/dk for all models.  Must satisfy |v_g| ≤ c."""
    k_arr = np.linspace(0, np.pi, n_p)
    dk = k_arr[1] - k_arr[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f'Group velocity  v_g = dE/dk   (ε={eps})', fontsize=11)

    for name, TM_fn, c_est, col, label in MODELS:
        E_bands = bands_from_TM(TM_fn, k_arr, eps)
        E_phys  = physical_band(k_arr, E_bands, c_est=c_est)
        c_fit, m_fit, _ = fit_rel(k_arr, E_phys, k_max=np.pi/4)
        vg = np.gradient(E_phys, dk)
        ax.plot(k_arr, vg, '-', color=col, lw=2, label=f'{name}  (c={c_fit:.3f})')
        ax.axhline( c_fit, color=col, lw=0.8, ls=':', alpha=0.6)
        ax.axhline(-c_fit, color=col, lw=0.8, ls=':', alpha=0.6)

    ax.set_xlabel('k'); ax.set_ylabel('v_g = dE/dk')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('group_velocity_phys.png', dpi=150, bbox_inches='tight')
    print('Saved group_velocity_phys.png')
    plt.close()


# ─────────────────────────── main ───────────────────────────────────────────

if __name__ == '__main__':
    EPS = 0.1

    print('=== Transfer-matrix bands + dispersion fit ===')
    fig_dispersion(eps=EPS)

    print('\n=== c and m scaling with ε ===')
    fig_c_and_m_scaling()

    print('\n=== Physical-time probability simulation (T=30) ===')
    fig_probability_comparison(T=30, eps=EPS)

    print('\n=== Group velocity ===')
    fig_group_velocity(eps=EPS)

    print('\nFertig.')
