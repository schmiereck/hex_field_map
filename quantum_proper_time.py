#!/usr/bin/env python3
"""
Proper Time (Eigenzeit) Investigation — 1+1D Equilateral Triangular Lattice
============================================================================
Lattice moves (per half-step Δτ=0.5 physical time):
  d=0  left-diagonal   Δx = -√3/2,  Δt = 0.5,  dτ_proper = 0  (lightlike)
  d=1  straight-up     Δx =  0,     Δt = 1.0,  dτ_proper = 1  (timelike)
  d=2  right-diagonal  Δx = +√3/2,  Δt = 0.5,  dτ_proper = 0  (lightlike)

Speed of light: c = (√3/2)/0.5 = √3 ≈ 1.7321
Proper time along a path:  τ = Σ (Δτ per straight step) = n_straight

Second-order recurrence (matching quantum_dispersion_phys.simulate_EQ_phys):
  new_amp[i,   0] ← amp_prev[i+1, :] @ C[0]   (left-diag,  1 half-step ago)
  new_amp[i,   1] ← amp_pprev[i, :]  @ C[1]   (straight,   2 half-steps ago)
  new_amp[i,   2] ← amp_prev[i-1, :] @ C[2]   (right-diag, 1 half-step ago)

Physical mass (ε=0.1): m_phys ≈ ε  (smallest positive eigenvalue of TM_full at k=0)

Source files referenced:
  quantum_dispersion_phys.py  — TM_EQ_half / TM_EQ_full definitions
  quantum_path_integral.py    — simulate_triangular baseline
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── constants ─────────────────────────────────────────────────────────────────
SQRT3   = np.sqrt(3)
SQRT3_2 = SQRT3 / 2      # physical Δx per diagonal half-step  ≈ 0.8660
C_LIGHT = SQRT3          # speed of light
EPS     = 0.1            # default mass parameter


# ── transfer matrix helpers ───────────────────────────────────────────────────

def _C(eps):
    """3×3 coupling matrix: 1 on diagonal, iε off-diagonal."""
    M = np.full((3, 3), 1j * eps, dtype=complex)
    np.fill_diagonal(M, 1.0 + 0j)
    return M


def TM_half(k_phys, eps):
    """
    6×6 half-step transfer matrix.
    k_phys = physical momentum.  Grid momentum p = k_phys * SQRT3_2.

    State vector (G_curr[3], G_prev[3]).
    d=0 (left,  exp+ip): A[0,:] = exp(+ip)*C[0,:]
    d=1 (straight):      B[1,:] = C[1,:]  (uses G_prev)
    d=2 (right, exp-ip): A[2,:] = exp(-ip)*C[2,:]
    """
    C   = _C(eps)
    p   = k_phys * SQRT3_2
    A   = np.zeros((3, 3), dtype=complex)
    A[0, :] = np.exp(+1j * p) * C[0, :]
    A[2, :] = np.exp(-1j * p) * C[2, :]
    B   = np.zeros((3, 3), dtype=complex)
    B[1, :] = C[1, :]
    M   = np.zeros((6, 6), dtype=complex)
    M[:3, :3] = A;  M[:3, 3:] = B
    M[3:, :3] = np.eye(3)
    return M


def TM_full(k_phys, eps):
    H = TM_half(k_phys, eps)
    return H @ H


def m_phys_eq(eps):
    """
    Physical mass of the 1+1D EQ lattice.

    The propagating band has m ≈ 2ε (same scaling as the 2+1D hexagonal).
    We select the positive eigenvalue E = -arg(λ) of TM_full at k=0 that is
    closest to 2*eps.
    """
    lam   = np.linalg.eigvals(TM_full(0.0, eps))
    E_all = -np.angle(lam)
    pos   = E_all[(E_all > 1e-4) & (E_all < np.pi)]
    if len(pos) == 0:
        return float(2.0 * eps)
    idx = np.argmin(np.abs(pos - 2.0 * eps))
    return float(pos[idx])


def phys_eigvec_full(k_phys, eps, m_phys):
    """
    Physical eigenvector of TM_full at k_phys.
    Selects eigenvalue closest to exp(-i*E_ref) where E_ref = sqrt(c²k²+m²).
    Returns (v_curr[3], v_prev[3]), ||v_curr|| normalised to 1.
    """
    M           = TM_full(k_phys, eps)
    lam, vecs   = np.linalg.eig(M)
    E_ref       = float(np.sqrt(C_LIGHT**2 * k_phys**2 + m_phys**2))
    idx         = np.argmin(np.abs(-np.angle(lam) - E_ref))
    v           = vecs[:, idx]
    vc, vp      = v[:3].copy(), v[3:].copy()
    norm        = np.linalg.norm(vc) + 1e-30
    return vc / norm, vp / norm


def momentum_for_vfrac(v_frac, m_phys):
    """
    Physical momentum p for velocity v = v_frac * c.
    From E²=c²p²+m², v=c²p/E  →  p = m*v_frac / (√3*√(1-v_frac²))
    """
    if abs(v_frac) < 1e-12:
        return 0.0
    vf = min(abs(v_frac), 0.9999)
    return m_phys * vf / (SQRT3 * np.sqrt(1.0 - vf**2))


def _grid_size(T_phys, sigma, v_frac):
    """Adaptive grid: 3σ envelope + v·c·T drift + margin, minimum 4T+1."""
    x_drift = abs(v_frac) * C_LIGHT * T_phys           # physical drift
    x_gauss = 3.0 * sigma                               # 3σ envelope width
    xi_need = (x_drift + x_gauss) / SQRT3_2 + 10        # grid cells needed from centre
    xi_min  = 2 * T_phys                                # causal minimum
    xcenter = int(max(xi_need, xi_min)) + 1
    Nx      = 2 * xcenter + 1
    return Nx, xcenter


# ── wave-packet simulation ────────────────────────────────────────────────────

def simulate_eq_wp(T_phys, eps, sigma, v_frac):
    """
    Gaussian wave packet on the 1+1D EQ lattice.

    Observable: psi_proj[t, xi] = Σ_d conj(v_curr_ref[d]) * amp[xi,d,t]
    (projects onto the physical eigenvector, suppressing fast modes).

    Returns
    -------
    psi_proj : complex (T_phys+1, Nx)
    xcenter  : int  — index where x=0
    m_phys   : float
    k_phys   : float  — physical momentum
    E_phys   : float  — physical energy
    v_grid   : float  — CoM velocity in grid steps per full time step
    """
    m_phys  = m_phys_eq(eps)
    k_phys  = momentum_for_vfrac(v_frac, m_phys)
    E_phys  = float(np.sqrt(C_LIGHT**2 * k_phys**2 + m_phys**2))
    v_phys  = C_LIGHT**2 * k_phys / E_phys          # physical velocity
    v_grid  = v_phys / SQRT3_2                       # grid steps per phys-time

    vc_ref, vp_ref = phys_eigvec_full(k_phys, eps, m_phys)

    N_half  = 2 * T_phys
    Nx, xcenter = _grid_size(T_phys, sigma, v_frac)
    xi_arr  = np.arange(Nx) - xcenter                # grid indices relative to 0
    x_phys  = xi_arr * SQRT3_2                       # physical positions

    envelope = np.exp(-x_phys**2 / (2.0 * sigma**2)
                      + 1j * k_phys * x_phys)         # Gaussian * plane-wave

    C_mat   = _C(eps)

    # Initialise with physical eigenvector spinor
    amp_prev  = envelope[:, None] * vc_ref[None, :]  # (Nx, 3)
    amp_pprev = envelope[:, None] * vp_ref[None, :]  # (Nx, 3)  (≈ e^{+iE}·vc component)

    # Normalise so Σ|psi_proj(t=0)|² = 1
    psi0 = (amp_prev * np.conj(vc_ref)[None, :]).sum(axis=-1)
    p0   = float((np.abs(psi0)**2).sum())
    if p0 > 1e-30:
        s = 1.0 / np.sqrt(p0)
        amp_prev  *= s
        amp_pprev *= s

    psi_proj        = np.zeros((T_phys + 1, Nx), dtype=complex)
    psi_proj[0]     = (amp_prev * np.conj(vc_ref)[None, :]).sum(axis=-1)

    for tau in range(1, N_half + 1):
        new_amp = np.zeros((Nx, 3), dtype=complex)
        wc = amp_prev  @ C_mat          # diagonal uses 1 half-step ago
        ws = amp_pprev @ C_mat          # straight  uses 2 half-steps ago

        new_amp[:-1, 0] += wc[1:,  0]  # left-diag:  i+1 → i
        new_amp[:,   1] += ws[:,   1]  # straight:   i   → i  (from pprev)
        new_amp[1:,  2] += wc[:-1, 2]  # right-diag: i-1 → i

        amp_pprev, amp_prev = amp_prev, new_amp

        if tau % 2 == 0:
            t = tau // 2
            psi_proj[t] = (amp_prev * np.conj(vc_ref)[None, :]).sum(axis=-1)

    return psi_proj, xcenter, m_phys, k_phys, E_phys, v_grid


# ── proper-time distribution ──────────────────────────────────────────────────

def simulate_eq_tau(T_phys, eps, sigma, v_frac):
    """
    Tracks proper-time accumulator τ_acc = n_straight_triggers × 0.5.

    IMPORTANT: The physical eigenvector at E≈2ε (k=0) has zero straight
    component — it is a purely diagonal (lightlike) standing wave.  To
    populate the 'timelike' mode (E≈ε, |vc_straight|≈0.75) we use
    **uniform initialisation** (equal weight over all three directions).
    This ensures paths with straight moves contribute to τ_acc.

    Observable: prob_tau[ns] = Σ_xi |Σ_d amp_tau[xi,d,ns]|²
    (total amplitude squared, not projected — includes all modes).

    Returns
    -------
    prob_tau : real (N_half+1,)  — probability per τ_acc bin at t = T_phys
    tau_arr  : real (N_half+1,) — τ_acc values 0, 0.5, 1.0, ..., T_phys
    m_phys, k_phys : floats
    """
    m_phys  = m_phys_eq(eps)
    k_phys  = momentum_for_vfrac(v_frac, m_phys)

    N_half  = 2 * T_phys
    N_ns    = N_half + 1
    Nx, xcenter = _grid_size(T_phys, sigma, v_frac)
    xi_arr  = np.arange(Nx) - xcenter
    x_phys  = xi_arr * SQRT3_2
    tau_arr = np.arange(N_ns) * 0.5

    envelope = np.exp(-x_phys**2 / (2.0 * sigma**2)
                      + 1j * k_phys * x_phys)

    C_mat = _C(eps)

    # Uniform initialisation: equal weight 1/√3 over all three directions.
    # This activates the timelike mode (E≈ε, straight-dominated) which the
    # physical eigenvector misses due to its zero straight component.
    uniform = np.ones(3, dtype=complex) / np.sqrt(3.0)
    amp_curr = np.zeros((Nx, 3, N_ns), dtype=complex)
    amp_prev = np.zeros((Nx, 3, N_ns), dtype=complex)
    amp_curr[:, :, 0] = envelope[:, None] * uniform[None, :]
    amp_prev[:, :, 0] = envelope[:, None] * uniform[None, :]

    # Normalise so total prob(t=0) = 1
    psi0 = amp_curr[:, :, 0].sum(axis=1)        # Σ_d amp[xi, d, 0]
    p0   = float((np.abs(psi0)**2).sum())
    if p0 > 1e-30:
        s = 1.0 / np.sqrt(p0)
        amp_curr *= s;  amp_prev *= s

    for _tau in range(1, N_half + 1):
        new_tau = np.zeros((Nx, 3, N_ns), dtype=complex)
        wc = np.einsum('dj,xjn->dxn', C_mat, amp_curr)   # (3, Nx, N_ns)
        ws = np.einsum('dj,xjn->dxn', C_mat, amp_prev)   # (3, Nx, N_ns)

        new_tau[:-1, 0, :] += wc[0, 1:,  :]           # left-diag
        new_tau[:,   1, 1:] += ws[1, :,  :-1]          # straight → ns+1
        new_tau[1:,  2, :] += wc[2, :-1, :]            # right-diag

        amp_prev, amp_curr = amp_curr, new_tau

    # Total amplitude observable
    psi_total = amp_curr.sum(axis=1)             # (Nx, N_ns): Σ_d amp[xi,d,ns]
    prob_tau  = (np.abs(psi_total)**2).sum(axis=0)   # (N_ns,)
    return prob_tau, tau_arr, m_phys, k_phys


# ── observables from psi_proj ─────────────────────────────────────────────────

def packet_observables(psi_proj, xcenter, SQRT3_2_=SQRT3_2):
    """
    Returns t_arr, xcom, prob_total, phase_com.

    phase_com[t] = arg(psi_proj[t, xi_com(t)]) — phase at CoM grid index.
    This equals -(m/γ)*t  for a pure momentum eigenstate.
    """
    Nx     = psi_proj.shape[1]
    prob   = np.abs(psi_proj)**2
    xi_arr = np.arange(Nx) - xcenter
    x_phys = xi_arr * SQRT3_2_
    ptot   = prob.sum(axis=1) + 1e-30
    xcom   = (prob * x_phys[None, :]).sum(axis=1) / ptot

    # Phase at integer-rounded CoM grid index
    xi_com_f = xcom / SQRT3_2_ + xcenter          # float grid index of CoM
    xi_com   = np.clip(np.round(xi_com_f).astype(int), 0, Nx - 1)
    phase_com = np.angle(psi_proj[np.arange(len(xi_com)), xi_com])
    phase_com = np.unwrap(phase_com)

    t_arr = np.arange(psi_proj.shape[0], dtype=float)
    return t_arr, xcom, ptot - 1e-30, phase_com


# ── velocity sweep ────────────────────────────────────────────────────────────

def velocity_sweep(T_phys, eps, sigma, v_fracs):
    """
    For each v_frac compute:
      tau_th_classical: T·√(1-v²/c²)
      tau_th_quantum:   T·⟨m/E(k)⟩_Gaussian  (correct QM prediction for packet)
      tau_phase:        T·|d(phase_CoM)/dt| / m_phys  (phase-slope measurement)
      tau_dist:         ⟨τ_acc⟩ from straight-step distribution (timelike mode)

    Returns list of dicts.
    """
    results = []
    m_phys  = m_phys_eq(eps)
    print(f'  m_phys = {m_phys:.5f}  (eps={eps})')

    for vf in v_fracs:
        k_phys = momentum_for_vfrac(vf, m_phys)
        E_phys = float(np.sqrt(C_LIGHT**2 * k_phys**2 + m_phys**2))

        # Classical proper time
        tau_cl = T_phys * np.sqrt(max(1.0 - vf**2, 0.0))

        # Quantum proper time: Gaussian-weighted average ⟨m/E(k)⟩
        sigma_k = 1.0 / sigma
        k_fine  = np.linspace(k_phys - 4*sigma_k, k_phys + 4*sigma_k, 4000)
        G_k     = np.exp(-(k_fine - k_phys)**2 / (2.0 * sigma_k**2))
        E_fine  = np.sqrt(C_LIGHT**2 * k_fine**2 + m_phys**2)
        tau_qt  = float(T_phys * m_phys * (G_k/E_fine).sum() / (G_k.sum()+1e-30))

        # Phase-based τ measurement from simulate_eq_wp.
        # The phase at x_com(t) evolves as: φ(t) ≈ k0·x_com - ⟨E(k)⟩·t
        # so d(phase)/dt = k0·v_g_eff - ⟨E⟩.
        # For a narrow k-packet at k0: this → -m/γ (correct relativistic rate).
        # For a broad packet (σ_k ≳ m): ⟨E⟩ significantly exceeds E(k0),
        # causing tau_phase to OVERESTIMATE the proper time.
        # We report it as-is — the TREND (decreasing with v) is physically correct.
        psi, xc, mp, kp, Ep, vg = simulate_eq_wp(T_phys, eps, sigma, vf)
        t_arr, xcom, _, phase_com = packet_observables(psi, xc)
        # Fit phase slope on t ≥ 2 (skip initialisation transient)
        t_fit = t_arr[2:]; ph_fit = phase_com[2:]
        slope = float(np.polyfit(t_fit, ph_fit, 1)[0]) if len(t_fit) > 2 else 0.0
        tau_phase = float(T_phys * abs(slope) / m_phys)

        # Straight-step τ distribution (uniform init → timelike mode active)
        prob_tau, tau_arr, _, _ = simulate_eq_tau(T_phys, eps, sigma, vf)
        norm      = prob_tau.sum() + 1e-30
        tau_dist  = float((tau_arr * prob_tau).sum() / norm)

        results.append(dict(
            v_frac           = vf,
            tau_th_classical = tau_cl,
            tau_th_quantum   = tau_qt,
            tau_phase        = tau_phase,
            tau_dist         = tau_dist,
            tau_arr          = tau_arr,
            prob_tau         = prob_tau,
        ))
        print(f'  v={vf:.1f}c  τ_cl={tau_cl:.3f}  τ_qt={tau_qt:.3f}  '
              f'τ_phase={tau_phase:.3f}  τ_dist={tau_dist:.3f}')

    return results



# ── Figure 1: worldlines + heatmap ────────────────────────────────────────────

def fig_worldlines(T_phys, eps, sigma):
    """
    Spacetime heatmap |ψ|²(x,t) for v=0 and v=0.5c side by side.
    Worldlines colored by accumulated proper time (analytical).
    """
    v_pairs = [(0.0, 'rest  (v=0)'), (0.5, 'v = 0.5c')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f'Spacetime spread  |ψ(x,t)|²  —  EQ Triangular Lattice\n'
        f'(ε={eps}, σ={sigma})',
        fontsize=11)

    m_phys = m_phys_eq(eps)

    for ax, (vf, label) in zip(axes, v_pairs):
        psi, xc, mp, kp, Ep, vg = simulate_eq_wp(T_phys, eps, sigma, vf)
        prob = np.abs(psi)**2           # (T+1, Nx)
        Nx   = prob.shape[1]
        xi   = np.arange(Nx) - xc
        x_ph = xi * SQRT3_2
        t_ph = np.arange(T_phys + 1, dtype=float)

        # Heatmap
        vmax = prob.max(); vmin = max(vmax * 1e-6, 1e-30)
        im = ax.imshow(
            prob.T, origin='lower', aspect='auto',
            extent=[t_ph[0] - 0.5, t_ph[-1] + 0.5,
                    x_ph[0],       x_ph[-1]],
            vmin=vmin, vmax=vmax, cmap='hot')
        plt.colorbar(im, ax=ax, fraction=0.04, label='|ψ|²')

        # Classical worldline at velocity vf*c, colored by τ_acc
        t_wl = np.linspace(0, T_phys, 300)
        x_wl = vf * C_LIGHT * t_wl
        tau_wl = t_wl * np.sqrt(max(1.0 - vf**2, 0.0))
        pts = ax.scatter(t_wl, x_wl, c=tau_wl, cmap='cool',
                         s=6, zorder=3, label='worldline (colored by τ)')
        cb2 = plt.colorbar(pts, ax=ax, fraction=0.04, label='τ_acc')

        # Light-cone lines
        ax.plot(t_ph,  C_LIGHT * t_ph, 'w--', lw=0.8, alpha=0.5, label='light cone')
        ax.plot(t_ph, -C_LIGHT * t_ph, 'w--', lw=0.8, alpha=0.5)

        tau_th = T_phys * np.sqrt(max(1.0 - vf**2, 0.0))
        ax.set_title(
            f'{label}\np={kp:.4f}  E={Ep:.4f}  '
            f'τ_theory={tau_th:.2f}  (T={T_phys})',
            fontsize=9)
        ax.set_xlabel('Coordinate time  t'); ax.set_ylabel('Physical position  x')
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig('worldlines_proper_time.png', dpi=150, bbox_inches='tight')
    print('Saved worldlines_proper_time.png')
    plt.close()


# ── Figure 2: phase vs time (oscillation frequency ∝ m/γ) ────────────────────

def fig_phase_vs_time(T_phys, eps, sigma, v_fracs_show=(0.0, 0.5, 0.9)):
    """
    Im(ψ_center(t)) for several velocities.
    Phase oscillates at frequency m/γ = m·√(1−v²/c²).
    """
    m_phys  = m_phys_eq(eps)
    fig, axes = plt.subplots(len(v_fracs_show), 1,
                             figsize=(11, 3.5 * len(v_fracs_show)), sharex=True)
    if len(v_fracs_show) == 1:
        axes = [axes]
    fig.suptitle(
        f'Phase at CoM  arg(ψ_center(t))  —  EQ Triangular Lattice\n'
        f'(ε={eps}, σ={sigma})   Phase rate = m/γ  →  τ = T·(phase rate)/m',
        fontsize=10)

    for ax, vf in zip(axes, v_fracs_show):
        psi, xc, mp, kp, Ep, vg = simulate_eq_wp(T_phys, eps, sigma, vf)
        t_arr, xcom, ptot, phase = packet_observables(psi, xc)

        # Linear fit on phase (skip first 2 steps — initialization transient)
        t_fit  = t_arr[2:]
        ph_fit = phase[2:]
        if len(t_fit) > 2:
            slope, intercept = np.polyfit(t_fit, ph_fit, 1)
        else:
            slope = 0.0; intercept = 0.0
        omega_eff  = abs(slope)               # measured |d(phase)/dt|
        tau_meas_p = T_phys * omega_eff / mp  # proper time from phase slope
        tau_theory = T_phys * np.sqrt(max(1.0 - vf**2, 0.0))
        gamma_th   = 1.0 / np.sqrt(max(1.0 - vf**2, 1e-9))

        # Plot Im(psi_center)
        xi_com = np.clip(
            np.round(xcom / SQRT3_2 + xc).astype(int), 0, psi.shape[1] - 1)
        psi_cen = psi[np.arange(len(xi_com)), xi_com]

        ax.plot(t_arr, np.imag(psi_cen), 'b-', lw=1.2, alpha=0.8,
                label='Im(ψ_center)')
        ax.plot(t_arr, np.real(psi_cen), 'r-', lw=0.8, alpha=0.5,
                label='Re(ψ_center)')

        # Phase overlay (right axis)
        ax2 = ax.twinx()
        ax2.plot(t_arr, phase, 'm-', lw=1.3, alpha=0.7, label='phase (unwrapped)')
        ax2.plot(t_arr, slope * t_arr + intercept, 'm--', lw=1.2,
                 label=f'fit: slope={slope:.5f}')
        ax2.set_ylabel('Phase  arg(ψ_center)  [rad]', color='m', fontsize=8)
        ax2.tick_params(axis='y', colors='m')
        ax2.legend(fontsize=7, loc='lower left')

        ax.set_title(
            f'v = {vf}c   γ={gamma_th:.3f}   '
            f'|slope|={omega_eff:.5f}   '
            f'τ_phase={tau_meas_p:.3f}   τ_theory={tau_theory:.3f}   '
            f'ratio={tau_meas_p/tau_theory:.4f}' if tau_theory > 1e-6 else
            f'v = {vf}c  (at rest)  |slope|={omega_eff:.5f}  ≈ m={mp:.5f}',
            fontsize=9)
        ax.set_ylabel('ψ_center amplitude')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Coordinate time  t')
    plt.tight_layout()
    plt.savefig('phase_vs_time.png', dpi=150, bbox_inches='tight')
    print('Saved phase_vs_time.png')
    plt.close()


# ── Figure 3: time-dilation curve τ(v) ───────────────────────────────────────

def fig_dilation_curve(results, T_phys, eps):
    """
    Main panel: τ/T vs v/c.
    τ_quantum = T·m·⟨1/E(k)⟩_G  (correct QM prediction for a Gaussian packet)
    τ_phase   = T·|d(phase_CoM)/dt|/m  (phase-slope simulation; overestimates
                for broad packets because it measures ⟨E⟩·T/m ≥ τ_classical)
    τ_dist    = ⟨n_straight⟩×0.5  (timelike mode, uniform init; ≈0 for physical
                eigenvector since E≈2ε mode has |vc_straight|=0 — purely lightlike)
    Lines: classical √(1-v²/c²) and quantum ⟨m/E(k)⟩_Gaussian.
    Residual panel: (τ_phase - τ_quantum) / τ_quantum in percent.
    """
    v_fracs   = np.array([r['v_frac']           for r in results])
    tau_cl    = np.array([r['tau_th_classical']  for r in results])
    tau_qt    = np.array([r['tau_th_quantum']    for r in results])
    tau_ms    = np.array([r['tau_phase']         for r in results])
    tau_dist  = np.array([r['tau_dist']          for r in results])

    v_fine = np.linspace(0, 0.95, 200)
    tau_cl_fine = T_phys * np.sqrt(1.0 - v_fine**2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(
        f'Relativistic Time Dilation  τ/T  vs  v/c\n'
        f'EQ Triangular Lattice  (ε={eps},  T={T_phys},  σ={8})',
        fontsize=11)

    # Main panel
    ax1.plot(v_fine, tau_cl_fine / T_phys, 'k-', lw=2,
             label='Classical SR:  √(1−v²/c²)')
    ax1.plot(v_fracs, tau_qt / T_phys, 'b--o', lw=1.5, ms=7,
             label='Quantum ⟨m/E(k)⟩_Gaussian  (correct QM prediction)')
    ax1.plot(v_fracs, tau_ms / T_phys, 'rs', ms=10, zorder=5,
             label='τ_phase: T·⟨E⟩/m  (phase slope at CoM; overestimates for broad σ_k)')
    ax1.plot(v_fracs, tau_dist / T_phys, 'g^', ms=8, zorder=4,
             label='τ_dist: ⟨n_straight⟩×0.5  (timelike mode, ≈0 for physical eigenvec)')

    for vf, tms, tqt in zip(v_fracs, tau_ms, tau_qt):
        ax1.annotate(f'{tms/T_phys:.3f}',
                     xy=(vf, tms / T_phys), xytext=(5, 6),
                     textcoords='offset points', fontsize=7, color='darkred')

    ax1.set_ylabel('τ / T  (proper time / coordinate time)')
    ax1.set_xlim(-0.02, 1.0); ax1.set_ylim(-0.02, 1.3)
    ax1.axhline(1, color='gray', ls=':', lw=1)
    ax1.legend(fontsize=8, loc='upper right'); ax1.grid(alpha=0.3)

    # Explanatory note
    note = (
        'Key finding: Physical eigenvector (E≈2ε) has zero straight component\n'
        '→ purely lightlike mode → τ_dist ≈ 0.\n'
        'τ_phase overestimates τ when σ_k ≳ m (broad packet):\n'
        '  phase slope = k₀·v_g − ⟨E(k)⟩, not −m/γ.\n'
        'τ_quantum = T·m·⟨1/E(k)⟩_G is the correct QM prediction.'
    )
    ax1.text(0.02, 1.22, note, fontsize=7, va='top',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

    # Residual panel
    res_pct = 100.0 * (tau_ms - tau_qt) / (tau_qt + 1e-30)
    ax2.axhline(0, color='k', lw=1)
    ax2.bar(v_fracs, res_pct, width=0.04, color=['green' if abs(r) < 5 else 'orange'
                                                   for r in res_pct],
            edgecolor='k', linewidth=0.7)
    for vf, r in zip(v_fracs, res_pct):
        ax2.text(vf, r + np.sign(r) * 0.3, f'{r:+.1f}%',
                 ha='center', va='bottom' if r >= 0 else 'top', fontsize=7)

    ax2.set_xlabel('v / c'); ax2.set_ylabel('(τ_phase − τ_quantum) / τ_quantum  [%]')
    ax2.set_title('Residuals: τ_phase vs quantum ⟨m/E⟩ prediction', fontsize=9)
    ax2.set_xlim(-0.02, 1.0); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('dilation_curve.png', dpi=150, bbox_inches='tight')
    print('Saved dilation_curve.png')
    plt.close()


# ── Figure 4: proper-time distribution histograms ────────────────────────────

def fig_proper_time_distribution(results_subset, T_phys, eps):
    """
    Histogram of prob(τ_acc) at T_phys for selected velocities.
    Shows quantum spread around classical and quantum-averaged proper times.
    """
    n = len(results_subset)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f'Proper-Time Distribution  P(τ_acc)  at  t={T_phys}  [timelike mode, uniform init]\n'
        f'EQ Triangular Lattice  (ε={eps})',
        fontsize=11)

    for ax, r in zip(axes, results_subset):
        vf      = r['v_frac']
        prob    = r['prob_tau']
        tau_arr = r['tau_arr']
        tau_cl  = r['tau_th_classical']
        tau_qt  = r['tau_th_quantum']
        tau_ms  = r['tau_dist']
        norm    = prob.sum() + 1e-30

        # Bar chart
        ax.bar(tau_arr, prob / norm, width=0.45,
               color='steelblue', alpha=0.75, edgecolor='navy', lw=0.5,
               label='P(τ_acc)')
        ax.axvline(tau_cl, color='red',   lw=2.0, ls='-',
                   label=f'τ_classical = {tau_cl:.2f}')
        ax.axvline(tau_qt, color='orange', lw=2.0, ls='--',
                   label=f'τ_quantum = {tau_qt:.2f}')
        ax.axvline(tau_ms, color='green',  lw=2.0, ls=':',
                   label=f'⟨τ⟩_sim = {tau_ms:.2f}')

        gamma_th = 1.0 / np.sqrt(max(1.0 - vf**2, 1e-9))
        sigma_tau = np.sqrt(max((tau_arr**2 * prob/norm).sum() - (tau_arr*prob/norm).sum()**2, 0))
        ax.set_title(
            f'v = {vf}c   (γ = {gamma_th:.3f})\n'
            f'⟨τ⟩_sim={tau_ms:.2f}  σ_τ={sigma_tau:.2f}\n'
            f'[Note: physical E≈2ε mode is lightlike; ⟨τ⟩≈0 for that mode]',
            fontsize=8)
        ax.set_xlabel('Proper time  τ_acc')
        ax.set_ylabel('Probability  P(τ_acc)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('proper_time_distribution.png', dpi=150, bbox_inches='tight')
    print('Saved proper_time_distribution.png')
    plt.close()



def _print_table(results, label=''):
    """Print a summary table for a set of velocity-sweep results."""
    if label:
        print(f'\n  {label}')
    print(f'  {"v/c":>6} | {"τ_classical":>12} | {"τ_quantum":>10} | '
          f'{"τ_phase":>10} | {"τ_dist":>8} | {"qt_vs_cl%":>10}')
    print('  ' + '-' * 72)
    for r in results:
        dev = 100.0 * (r['tau_th_quantum'] - r['tau_th_classical']) / (r['tau_th_classical'] + 1e-30)
        print(f'  {r["v_frac"]:>6.1f} | {r["tau_th_classical"]:>12.4f} | '
              f'{r["tau_th_quantum"]:>10.4f} | {r["tau_phase"]:>10.4f} | '
              f'{r["tau_dist"]:>8.4f} | {dev:>10.2f}%')


# ── Figure 5: σ comparison dilation curve ────────────────────────────────────

def fig_sigma_comparison(all_sigma_results, T_phys, eps):
    """
    τ_quantum/T vs v/c for multiple σ values — shows convergence to classical SR.
    """
    v_fine = np.linspace(0, 0.95, 200)
    tau_cl_fine = np.sqrt(1.0 - v_fine**2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(
        f'σ Convergence:  τ_quantum/T  →  √(1−v²/c²)  as  σ_k/m → 0\n'
        f'EQ Triangular Lattice  (ε={eps},  T={T_phys})',
        fontsize=11)

    ax1.plot(v_fine, tau_cl_fine, 'k-', lw=2.5,
             label='Classical SR:  √(1−v²/c²)')

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    markers = ['o', 's', 'D', '^']
    for i, (sigma, results) in enumerate(all_sigma_results):
        v   = np.array([r['v_frac'] for r in results])
        tqt = np.array([r['tau_th_quantum'] for r in results])
        tcl = np.array([r['tau_th_classical'] for r in results])
        sk  = 1.0 / sigma
        m   = m_phys_eq(eps)
        c   = colors[i % len(colors)]
        mk  = markers[i % len(markers)]
        ax1.plot(v, tqt / T_phys, f'{mk}-', color=c, ms=8, lw=1.3,
                 label=f'σ={sigma}  (σ_k/m={sk/m:.3f})')

        # Residuals: (τ_quantum − τ_classical) / τ_classical
        res = 100.0 * (tqt - tcl) / (tcl + 1e-30)
        ax2.plot(v, res, f'{mk}-', color=c, ms=6, lw=1.2,
                 label=f'σ={sigma}')

    ax1.set_ylabel('τ / T')
    ax1.set_xlim(-0.02, 1.0); ax1.set_ylim(-0.02, 1.1)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2.axhline(0, color='k', lw=1)
    ax2.set_xlabel('v / c')
    ax2.set_ylabel('(τ_quantum − τ_classical) / τ_classical  [%]')
    ax2.set_title('Residuals: quantum vs classical proper time', fontsize=9)
    ax2.set_xlim(-0.02, 1.0); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('sigma_convergence.png', dpi=150, bbox_inches='tight')
    print('Saved sigma_convergence.png')
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    EPS     = 0.1
    T       = 20
    SIGMA   = 8.0
    SIGMAS  = [8.0, 20.0, 30.0]
    V_FRACS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

    m_phys = m_phys_eq(EPS)
    print(f'=== EQ Triangular Lattice — Proper Time Investigation ===')
    print(f'  ε={EPS}  T={T}  c=√3={C_LIGHT:.4f}')
    print(f'  m_phys = {m_phys:.5f}  (≈ 2ε = {2*EPS:.3f},  NOT ε — same scaling as 2+1D hexagonal)')
    print()
    print('  Physical eigenvector (E≈2ε, k=0) has |vc_straight|=0 → purely lightlike mode.')
    print('  τ_dist ≈ 2 is from the TIMELIKE mode (E≈ε) via uniform initialisation.')
    print('  τ_quantum = T·m·⟨1/E(k)⟩_G is the correct QM proper-time prediction.')

    # ── Figure 1: worldlines ──────────────────────────────────────────────────
    print(f'\n=== Fig 1: Worldlines heatmap (v=0 and v=0.5c)  [σ={SIGMA}] ===')
    fig_worldlines(T, EPS, SIGMA)

    # ── Figure 2: phase vs time ───────────────────────────────────────────────
    print(f'\n=== Fig 2: Phase vs time  [σ={SIGMA}] ===')
    fig_phase_vs_time(T, EPS, SIGMA, v_fracs_show=(0.0, 0.5, 0.9))

    # ── σ sweep ──────────────────────────────────────────────────────────────
    all_sigma_results = []
    for sig in SIGMAS:
        sk = 1.0 / sig
        print(f'\n=== Velocity sweep  σ={sig}  (σ_k/m={sk/m_phys:.3f}) ===')
        res = velocity_sweep(T, EPS, sig, V_FRACS)
        all_sigma_results.append((sig, res))
        _print_table(res, label=f'σ={sig}  σ_k/m={sk/m_phys:.3f}')

    # Use σ=8 for backward-compatible figures 3 & 4
    results_s8 = all_sigma_results[0][1]

    # ── Figure 3: dilation curve (σ=8, original) ─────────────────────────────
    print(f'\n=== Fig 3: Dilation curve  [σ={SIGMA}] ===')
    fig_dilation_curve(results_s8, T, EPS)

    # ── Figure 4: proper-time histograms (σ=8) ──────────────────────────────
    print(f'\n=== Fig 4: Proper-time distributions  [σ={SIGMA}] ===')
    subset = [r for r in results_s8 if r['v_frac'] in (0.0, 0.3, 0.7, 0.9)]
    fig_proper_time_distribution(subset, T, EPS)

    # ── Figure 5: σ convergence ──────────────────────────────────────────────
    print(f'\n=== Fig 5: σ convergence (σ={SIGMAS}) ===')
    fig_sigma_comparison(all_sigma_results, T, EPS)

    print('\nDone.')

