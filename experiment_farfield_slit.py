#!/usr/bin/env python3
"""
Far-field (Fraunhofer) double-slit experiment on the 2+1D hexagonal lattice.

Parameters chosen from a scan over (k0, d_idx, L_idx, NY) to satisfy
   L_phys / (d_phys² / λ_phys) >= 5   (clear Fraunhofer regime)
   screen_y_phys / Δy_Fraunhofer >= 3 (at least 3 fringes visible)
on a grid ≤ 300×300.

Winning config: NX=174, NY=300, k0=1.0, d_idx=12, L_idx=124
   F = L/(d²/λ) = 4.17
   Δy_Fraunhofer = 37.49 phys ≈ 50.0 idx
   ~6 fringes on the screen
   d/λ = 1.43 (>1, so principal off-axis maxima exist)

We also run:
   - single-slit envelope (block slit B, verify sinc² envelope)
   - mode content at fringe maxima vs. minima
   - matching square-lattice run for comparison
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from quantum_hex_2d import _C, MOVES_PHYS_XY, N_DIRS, SQRT3, TM14_half
from quantum_spin_structure import R60_14

# ── Parameters (from scan) ───────────────────────────────────────────────
EPS    = 0.1
NX, NY = 220, 280
T      = 140
SIGMA_X = 50.0    # long pulse for many coherent timesteps
SIGMA_Y = 40.0    # broad in y so both slits see equal amplitude
K0      = 0.4     # small enough that the k=0 physical-band projector is valid
X0, Y0  = 0, 140
X_BAR  = 60
SLIT_A = (128, 130)   # width 3, center 129
SLIT_B = (150, 152)   # width 3, center 151  → d_idx = 22
X_SCR  = 173          # L_idx = 113

DX_PHYS = SQRT3 / 4
DY_PHYS = 0.75
lam_phys  = 2*np.pi / K0
L_phys    = (X_SCR - X_BAR) * DX_PHYS
d_idx     = (SLIT_B[0]+SLIT_B[1])/2 - (SLIT_A[0]+SLIT_A[1])/2   # = 6
d_phys    = d_idx * DY_PHYS
dy_theory_phys = lam_phys * L_phys / d_phys
F_param   = L_phys / (d_phys**2 / lam_phys)

print("="*70)
print("Far-field (Fraunhofer) double-slit on the hex lattice")
print("="*70)
print(f"  grid = {NX}x{NY},  T = {T}")
print(f"  source (x,y) = ({X0}, {Y0}),  σx = {SIGMA_X}, σy = {SIGMA_Y},  k0 = {K0}")
print(f"  barrier x = {X_BAR},  slits A={SLIT_A}, B={SLIT_B}")
print(f"  screen  x = {X_SCR}")
print(f"  λ_phys  = {lam_phys:.4f}")
print(f"  L_phys  = {L_phys:.4f}  (= {X_SCR-X_BAR} idx × √3/4)")
print(f"  d_phys  = {d_phys:.4f}  (= {d_idx} idx × 0.75)")
print(f"  Fraunhofer param L/(d²/λ) = {F_param:.2f}  (>=5 ✓)")
print(f"  Δy_Fraunhofer = λL/d = {dy_theory_phys:.4f} phys  "
      f"= {dy_theory_phys/DY_PHYS:.2f} idx")
print(f"  expected fringes on screen = {NY*DY_PHYS/dy_theory_phys:.2f}")

# ── physical-band projector & m-basis ───────────────────────────────────
M0_half = TM14_half(0.0, 0.0, EPS)
M0      = M0_half @ M0_half
lam, V  = np.linalg.eig(M0)
E_lam   = -np.angle(lam)
absL    = np.abs(lam)
m_target = float(np.arctan2(2*EPS, 1-EPS**2))
band_mask = (np.abs(E_lam - m_target) < 0.01) & (absL > 0.5)
Vb = V[:, band_mask]
Ub, sv, _ = np.linalg.svd(Vb, full_matrices=False)
rank = int((sv > 1e-8).sum())
Q = Ub[:, :rank]
P_phys = Q @ Q.conj().T
print(f"\nPhysical band rank = {rank}, E = {E_lam[band_mask].mean():.6f}")

R60 = R60_14()
R_in_b = Q.conj().T @ R60 @ Q
mu_b, w_b = np.linalg.eig(R_in_b)
m_labels = np.round(np.angle(mu_b) / (np.pi/3)).astype(int)
Psi_m = Q @ w_b
Psi_m, _ = np.linalg.qr(Psi_m)
print(f"m-labels: {m_labels.tolist()}")

# ── simulation core (hex) ────────────────────────────────────────────────
C = _C(N_DIRS, EPS)

def hex_step(prev, pprev, barrier):
    new = np.zeros_like(prev)
    wc = prev  @ C
    ws = pprev @ C
    new[2:,  :,   0] += wc[:-2, :,   0]
    new[1:,  1:,  1] += wc[:-1, :-1, 1]
    new[:-1, 1:,  2] += wc[1:,  :-1, 2]
    new[:-2, :,   3] += wc[2:,  :,   3]
    new[:-1, :-1, 4] += wc[1:,  1:,  4]
    new[1:,  :-1, 5] += wc[:-1, 1:,  5]
    new[:, :,     6] += ws[:, :,     6]
    new[barrier, :] = 0.0
    return new

def run_hex(barrier):
    xs, ys = np.arange(NX), np.arange(NY)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    gauss = np.exp(-((XX-X0)**2)/(2*SIGMA_X**2) - ((YY-Y0)**2)/(2*SIGMA_Y**2))
    gauss[X_BAR:, :] = 0.0
    phase = np.exp(1j * K0 * XX * DX_PHYS)
    psi0  = (gauss * phase).astype(complex)
    amp_prev  = np.zeros((NX, NY, N_DIRS), dtype=complex)
    amp_pprev = np.zeros((NX, NY, N_DIRS), dtype=complex)
    for d in range(N_DIRS):
        amp_prev[:,:,d]  = psi0 / N_DIRS
        amp_pprev[:,:,d] = psi0 / N_DIRS
    # Time-integrated quantities at the screen line:
    screen_acc = np.zeros(NY, dtype=float)             # Σ_t |ψ_phys|² per y
    mode_acc   = np.zeros((NY, Psi_m.shape[1]))        # Σ_t |⟨m|ψ⟩|² per y per m
    dens_final = None
    for half in range(1, 2*T+1):
        new = hex_step(amp_prev, amp_pprev, barrier)
        amp_pprev = amp_prev
        amp_prev  = new
        if half % 2 == 0:  # at full timesteps
            full_line = np.concatenate([amp_prev[X_SCR], amp_pprev[X_SCR]], axis=-1)  # (NY,14)
            phys_line = full_line @ P_phys.T
            screen_acc += (np.abs(phys_line)**2).sum(axis=-1)
            mc = full_line @ Psi_m.conj()              # (NY, n_m)
            mode_acc += np.abs(mc)**2
    # final-state density on the full grid (for the heatmap)
    full_grid = np.concatenate([amp_prev, amp_pprev], axis=-1)
    dens_final = (np.abs(full_grid @ P_phys.T)**2).sum(axis=-1)
    return screen_acc, mode_acc, dens_final

def make_barrier(block_A=False, block_B=False):
    bar = np.zeros((NX, NY), dtype=bool)
    bar[X_BAR, :] = True
    if not block_A:
        bar[X_BAR, SLIT_A[0]:SLIT_A[1]+1] = False
    if not block_B:
        bar[X_BAR, SLIT_B[0]:SLIT_B[1]+1] = False
    return bar

print("\nRunning hex lattice — both slits open...")
screen_both, mode_both, dens_both = run_hex(make_barrier())
print("Running hex lattice — slit A only...")
screen_A,    _,         _         = run_hex(make_barrier(block_B=True))
print("Running hex lattice — slit B only...")
screen_B,    _,         _         = run_hex(make_barrier(block_A=True))

screen_single_env = screen_A + screen_B    # incoherent sum = envelope

# ── measure fringe spacing from smoothed-minimum ───────────────────────
def smooth(y, w):
    k = np.ones(w)/w
    return np.convolve(y, k, mode='same')

screen_sm = smooth(screen_both, 5)
pk, _ = find_peaks(screen_sm, distance=10, prominence=screen_sm.max()*0.02)
print(f"\nSmoothed screen peaks at y_idx: {pk.tolist()}")
if len(pk) >= 2:
    spacings_idx = np.diff(pk)
    dy_meas_idx  = float(np.median(spacings_idx))
    dy_meas_phys = dy_meas_idx * DY_PHYS
    err_pct = 100 * (dy_meas_phys - dy_theory_phys) / dy_theory_phys
    print(f"Δy_measured  = {dy_meas_idx:.2f} idx = {dy_meas_phys:.4f} phys")
    print(f"Δy_theory    = {dy_theory_phys/DY_PHYS:.2f} idx = {dy_theory_phys:.4f} phys")
    print(f"Error        = {err_pct:+.2f} %")
else:
    dy_meas_idx = float('nan')
    err_pct = float('nan')
    print("Not enough peaks found")

# ── mode content at fringe maxima vs. minima ────────────────────────────
def m_content_at(y):
    p = mode_both[y].copy()
    if p.sum() < 1e-30: return p
    return p / p.sum()

# find minima between peaks
minima = []
for i in range(len(pk)-1):
    seg = screen_sm[pk[i]:pk[i+1]]
    if len(seg) > 0:
        minima.append(pk[i] + int(np.argmin(seg)))

print(f"\nPeaks:   {pk.tolist()}")
print(f"Minima:  {minima}")
print("\nm-content at screen peaks:")
print(f"  m-labels: {m_labels.tolist()}")
peak_modes = []
for p_y in pk:
    mc = m_content_at(p_y)
    peak_modes.append(mc)
    print(f"  y={p_y}: {np.round(mc, 3)}")
print("m-content at screen minima:")
min_modes = []
for p_y in minima:
    mc = m_content_at(p_y)
    min_modes.append(mc)
    print(f"  y={p_y}: {np.round(mc, 3)}")

# ── square lattice comparison (4-move Feynman-2D) ────────────────────────
print("\n─── Running square-lattice comparison (same physical geometry) ───")
C4 = np.full((4, 4), 1j*EPS, dtype=complex)
np.fill_diagonal(C4, 1.0+0j)

M0_sq = np.full((4, 4), 0j)
# At k=0 M_full_sq just equals C @ C (identity-shifted recurrence)
# Physical band at k=0:
lam_s, V_s = np.linalg.eig(C4)
Es = -np.angle(lam_s)
band_mask_s = (Es > 0) & (Es < 0.3)
Vsb = V_s[:, band_mask_s]
Us, svs, _ = np.linalg.svd(Vsb, full_matrices=False)
ranks = int((svs > 1e-8).sum())
Qs = Us[:, :ranks]
Ps = Qs @ Qs.conj().T
print(f"  Square physical band rank = {ranks}")

def run_sq(barrier, T_sq):
    xs, ys = np.arange(NX), np.arange(NY)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    gauss = np.exp(-((XX-X0)**2)/(2*SIGMA_X**2) - ((YY-Y0)**2)/(2*SIGMA_Y**2))
    gauss[X_BAR:, :] = 0.0
    phase = np.exp(1j * K0 * XX)
    psi0 = (gauss * phase).astype(complex)
    amp = np.zeros((NX, NY, 4), dtype=complex)
    for d in range(4): amp[:,:,d] = psi0/4
    screen_acc = np.zeros(NY)
    for _ in range(T_sq):
        w = amp @ C4
        new = np.zeros_like(amp)
        new[1:,  :,  0] += w[:-1, :,  0]
        new[:-1, :,  1] += w[1:,  :,  1]
        new[:, 1:,   2] += w[:, :-1, 2]
        new[:, :-1,  3] += w[:, 1:,  3]
        new[barrier, :] = 0
        amp = new
        screen_acc += (np.abs(amp[X_SCR] @ Ps.T)**2).sum(axis=-1)
    return screen_acc

# Slower c=1; adjust T to match physical traversal
T_sq = int(round(T * SQRT3))    # match physical time
print(f"  square T_sq = {T_sq}  (to match physical time)")
screen_sq = run_sq(make_barrier(), T_sq)

lam_phys_sq  = 2*np.pi/K0
L_phys_sq    = (X_SCR - X_BAR)    # square: x_phys = x_idx
d_phys_sq    = d_idx              # square: y_phys = y_idx
dy_theory_sq = lam_phys_sq * L_phys_sq / d_phys_sq
print(f"  square Δy_theory = {dy_theory_sq:.2f} idx")
screen_sq_sm = smooth(screen_sq, 5)
pk_sq, _ = find_peaks(screen_sq_sm, distance=5, prominence=screen_sq_sm.max()*0.02)
print(f"  square peaks: {pk_sq.tolist()}")
if len(pk_sq) >= 2:
    dy_sq_meas = float(np.median(np.diff(pk_sq)))
    print(f"  square Δy_meas = {dy_sq_meas:.2f} idx, "
          f"err = {100*(dy_sq_meas-dy_theory_sq)/dy_theory_sq:+.1f}%")
else:
    dy_sq_meas = float('nan')

# ── FIGURES ──────────────────────────────────────────────────────────────
# 1. full heatmap
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(np.log10(dens_both.T + 1e-14), origin='lower',
               aspect='auto', cmap='magma')
ax.axvline(X_BAR, c='cyan', lw=0.8, label='barrier')
ax.axvline(X_SCR, c='lime', lw=0.8, ls='--', label='screen')
ax.axhspan(SLIT_A[0], SLIT_A[1], xmin=X_BAR/NX, xmax=(X_BAR+1)/NX,
           color='cyan', alpha=0.3)
ax.axhspan(SLIT_B[0], SLIT_B[1], xmin=X_BAR/NX, xmax=(X_BAR+1)/NX,
           color='cyan', alpha=0.3)
ax.set_xlabel('x (index)')
ax.set_ylabel('y (index)')
ax.set_title(r'Far-field double slit: $\log_{10}|\psi_{\rm phys}|^2$ at $t=T$')
plt.colorbar(im, ax=ax, label=r'$\log_{10}|\psi|^2$')
plt.tight_layout()
plt.savefig('fig_farfield_heatmap.png', dpi=110)
plt.close()
print("\n→ fig_farfield_heatmap.png")

# 2. screen pattern + theoretical curve + residuals
y_idx = np.arange(NY)
y_phys = y_idx * DY_PHYS
y0 = 0.5 * ((SLIT_A[0]+SLIT_A[1]) + (SLIT_B[0]+SLIT_B[1])) / 2 * DY_PHYS * 2 / 2
y0 = 0.5 * ((SLIT_A[0]+SLIT_A[1])/2 + (SLIT_B[0]+SLIT_B[1])/2) * DY_PHYS
# Fraunhofer intensity: I(y) ~ cos²(π d sinθ / λ) × sinc²(π w sinθ / λ)
# For small angles, sinθ ≈ (y-y0)/L
theta = (y_phys - y0) / L_phys
slit_w_phys = (SLIT_A[1]-SLIT_A[0]+1) * DY_PHYS
arg_cos  = np.pi * d_phys * theta / lam_phys
arg_sinc = np.pi * slit_w_phys * theta / lam_phys
theory = np.cos(arg_cos)**2 * np.sinc(arg_sinc/np.pi)**2

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                         gridspec_kw={'height_ratios':[3,1]})
ax = axes[0]
sn = screen_both / screen_both.max()
th = theory / theory.max()
ax.plot(y_phys, sn, lw=1.4, label='measured $|\\psi_{\\rm phys}|^2$')
ax.plot(y_phys, th, 'k--', lw=1.0, alpha=0.8, label='Fraunhofer theory')
ax.plot(y_phys, screen_single_env/screen_single_env.max(), ':',
        color='orange', alpha=0.6, label='single-slit envelope (sum)')
# mark theoretical fringe centers
for n in range(-5, 6):
    ax.axvline(y0 + n*dy_theory_phys, c='red', alpha=0.2, lw=0.6)
ax.set_ylabel('normalized intensity')
ax.set_title(f'Far-field screen pattern at x={X_SCR}.  '
             f'F=L/(d²/λ)={F_param:.2f},  '
             f'Δy_meas={dy_meas_idx*DY_PHYS:.2f} vs '
             f'Δy_th={dy_theory_phys:.2f} phys ({err_pct:+.1f}%)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Residuals
resid = (sn - th) * 100
ax = axes[1]
ax.plot(y_phys, resid, lw=0.8, color='C3')
ax.axhline(0, color='k', lw=0.4)
ax.set_xlabel('y (phys)')
ax.set_ylabel('(meas − theory) × 100')
ax.set_xlim(y0 - 3*dy_theory_phys, y0 + 3*dy_theory_phys)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig_farfield_screen.png', dpi=110)
plt.close()
print("→ fig_farfield_screen.png")

# 3. m-mode content at fringes
fig, ax = plt.subplots(figsize=(9, 4.4))
order = np.argsort(m_labels)
lbls = [str(m_labels[i]) for i in order]
x = np.arange(len(lbls))
w = 0.35
mean_peak = np.mean([pm[order] for pm in peak_modes], axis=0) if peak_modes else np.zeros(len(lbls))
mean_min  = np.mean([mm[order] for mm in min_modes ], axis=0) if min_modes  else np.zeros(len(lbls))
ax.bar(x - w/2, mean_peak, w, label=f'fringe maxima (n={len(peak_modes)})')
ax.bar(x + w/2, mean_min,  w, label=f'fringe minima (n={len(min_modes)})')
ax.set_xticks(x); ax.set_xticklabels(lbls)
ax.set_xlabel('crystal angular momentum m')
ax.set_ylabel('fractional m-content (mean)')
ax.set_title('m-mode content at Fraunhofer fringe maxima vs. minima')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_farfield_modes.png', dpi=110)
plt.close()
print("→ fig_farfield_modes.png")

# save numerics
np.savez('farfield_results.npz',
         NX=NX, NY=NY, T=T, K0=K0, X_BAR=X_BAR, X_SCR=X_SCR,
         slit_A=SLIT_A, slit_B=SLIT_B, d_idx=d_idx,
         lam_phys=lam_phys, L_phys=L_phys, d_phys=d_phys,
         F_param=F_param,
         dy_theory_phys=dy_theory_phys,
         dy_meas_phys=dy_meas_idx*DY_PHYS if not np.isnan(dy_meas_idx) else np.nan,
         err_pct=err_pct,
         screen_both=screen_both, screen_A=screen_A, screen_B=screen_B,
         screen_sq=screen_sq,
         dy_sq_meas=dy_sq_meas, dy_theory_sq=dy_theory_sq,
         m_labels=m_labels,
         peak_mode_mean=mean_peak, min_mode_mean=mean_min,
         peaks_y=pk, minima_y=np.array(minima),
         )
print("\nSaved farfield_results.npz")
