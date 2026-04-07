#!/usr/bin/env python3
"""
Experiment 1 — Double slit on the 2+1D hexagonal lattice.

Hex lattice with 7 directions (6 lightlike + 1 straight), 14-dim transfer
matrix per (x,y) site. Wave packet launched from the left, hits a barrier
at x = 30 with two slits, recorded on a screen at x = 80.

We project the full 14-dim state onto the 5-fold-degenerate physical band
(eigenvectors of M_full(k=0) at E ≈ 2ε) before computing densities, which
suppresses the non-physical fast modes.

We additionally diagonalize R_60 within the band → angular-momentum modes
m ∈ {−2,−1,+1,+2,+3} → and track which m sectors are enhanced at the
slit edges.
"""
import numpy as np
import matplotlib.pyplot as plt
from quantum_hex_2d import _C, MOVES_PHYS_XY, N_DIRS, SQRT3, TM14_half
from quantum_spin_structure import R60_14

# ─── parameters ────────────────────────────────────────────────────────────
EPS    = 0.1
SIGMA  = 6.0          # wave packet width (index units)
K0     = 0.3          # initial momentum (physical units along x)
NX, NY = 120, 120
T      = 60           # number of physical time steps
X0, Y0 = 8, 60        # source position (index)
X_BAR  = 30
SLIT_A = (52, 56)     # inclusive y-index range
SLIT_B = (64, 68)
X_SCR  = 80

DX_PHYS = SQRT3 / 4   # physical x per index unit
DY_PHYS = 0.75        # physical y per index unit

print("="*70)
print("Experiment 1 — Hex-lattice double slit")
print("="*70)

# ─── physical-band projector at k=0 ────────────────────────────────────────
M0_half = TM14_half(0.0, 0.0, EPS)
M0      = M0_half @ M0_half
lam, V  = np.linalg.eig(M0)
E_lam   = -np.angle(lam)
absL    = np.abs(lam)
m_target = float(np.arctan2(2*EPS, 1 - EPS**2))

mask    = (np.abs(E_lam - m_target) < 0.01) & (absL > 0.5)
V_band  = V[:, mask]
U_b, sv, _ = np.linalg.svd(V_band, full_matrices=False)
rank    = int((sv > 1e-8).sum())
Q       = U_b[:, :rank]                # 14 × 5
P_phys  = Q @ Q.conj().T               # 14 × 14 projector
print(f"Physical band rank = {rank},  E_band ≈ {E_lam[mask].mean():.6f}")
print(f"Suppression factor on a typical fast mode: see screen normalization")

# ─── joint M_full + R_60 eigenstates within band → m labels ───────────────
R60     = R60_14()
R_in_b  = Q.conj().T @ R60 @ Q          # 5 × 5 unitary
mu_b, w_b = np.linalg.eig(R_in_b)
m_labels = np.round(np.angle(mu_b) / (np.pi/3)).astype(int)
# joint eigenvectors in 14-dim
Psi_m   = Q @ w_b                       # 14 × 5
# orthonormalize Psi_m (eigenvectors of unitary -> already orthonormal up to noise)
Psi_m, _ = np.linalg.qr(Psi_m)
print(f"m-labels of band eigenstates: {sorted(m_labels.tolist())}")

# ─── initial wave packet ──────────────────────────────────────────────────
xs = np.arange(NX); ys = np.arange(NY)
XX, YY = np.meshgrid(xs, ys, indexing='ij')
x_ph = XX * DX_PHYS
y_ph = YY * DY_PHYS
gauss = np.exp(-((XX - X0)**2 + (YY - Y0)**2) / (2*SIGMA**2))
phase = np.exp(1j * K0 * x_ph)
psi0  = (gauss * phase).astype(complex)

amp_prev  = np.zeros((NX, NY, N_DIRS), dtype=complex)
amp_pprev = np.zeros((NX, NY, N_DIRS), dtype=complex)
for d in range(N_DIRS):
    amp_prev[:, :, d]  = psi0 / N_DIRS
    amp_pprev[:, :, d] = psi0 / N_DIRS

# ─── barrier mask ─────────────────────────────────────────────────────────
barrier = np.zeros((NX, NY), dtype=bool)
barrier[X_BAR, :] = True
barrier[X_BAR, SLIT_A[0]:SLIT_A[1]+1] = False
barrier[X_BAR, SLIT_B[0]:SLIT_B[1]+1] = False
print(f"Barrier at x={X_BAR}, slits y∈{SLIT_A} and y∈{SLIT_B}")

C = _C(N_DIRS, EPS)

def step(prev, pprev):
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

# ─── evolve and snapshot ──────────────────────────────────────────────────
SAVE_T = sorted({20, 40, 60})
snaps  = {}
N_HALF = 2 * T
for half in range(1, N_HALF + 1):
    new_amp = step(amp_prev, amp_pprev)
    amp_pprev = amp_prev
    amp_prev  = new_amp
    if half % 2 == 0:
        t = half // 2
        if t in SAVE_T:
            full = np.concatenate([amp_prev, amp_pprev], axis=-1)   # (Nx,Ny,14)
            phys = full @ P_phys.T
            snaps[t] = (np.abs(phys)**2).sum(axis=-1)

# ─── final state, screen pattern ──────────────────────────────────────────
final_full = np.concatenate([amp_prev, amp_pprev], axis=-1)
final_phys = final_full @ P_phys.T
final_dens = (np.abs(final_phys)**2).sum(axis=-1)
screen = final_dens[X_SCR, :]

# ─── theoretical fringe spacing (Fraunhofer) ──────────────────────────────
lam_phys  = 2 * np.pi / K0
L_phys    = (X_SCR - X_BAR) * DX_PHYS
y_a_phys  = ((SLIT_A[0] + SLIT_A[1]) / 2) * DY_PHYS
y_b_phys  = ((SLIT_B[0] + SLIT_B[1]) / 2) * DY_PHYS
d_phys    = abs(y_b_phys - y_a_phys)
dy_theory = lam_phys * L_phys / d_phys
print(f"\nFringe theory:  λ={lam_phys:.4f},  L={L_phys:.4f},  d={d_phys:.4f}")
print(f"  Δy_theory = λL/d = {dy_theory:.4f}  (phys)")
print(f"           = {dy_theory / DY_PHYS:.4f}  (y-index units)")

# ─── measure observed fringe spacing from screen pattern ──────────────────
# limit to a window around slit y range and find peaks
from scipy.signal import find_peaks
# Smooth heavily to find Fraunhofer envelope (suppress lattice fine structure)
def smooth(y, w):
    k = np.ones(w) / w
    return np.convolve(y, k, mode='same')
screen_sm = smooth(screen, 7)
peaks_fine, _  = find_peaks(screen,    distance=2)
peaks_env , _  = find_peaks(screen_sm, distance=8, prominence=screen_sm.max()*0.02)
if len(peaks_env) >= 2:
    dy_meas = float(np.median(np.diff(peaks_env)) * DY_PHYS)
    print(f"  Δy_measured (envelope) = {dy_meas:.4f} (phys)  from "
          f"{len(peaks_env)} envelope peaks at y_idx={peaks_env.tolist()}")
else:
    dy_meas = float('nan')
    print(f"  envelope: only {len(peaks_env)} peak(s) — Fraunhofer comb undersampled")
print(f"  fine-structure peak count = {len(peaks_fine)} "
      f"(spacing {np.median(np.diff(peaks_fine))*DY_PHYS:.3f} phys — lattice scale)")

# ─── mode decomposition: m-content at slit edges vs free wave ─────────────
def m_content(state14):
    """state14: (14,) complex vector → fractional norm in each m sector."""
    proj = Psi_m.conj().T @ state14            # (5,)
    p    = np.abs(proj)**2
    if p.sum() < 1e-30:
        return p
    return p / p.sum()

# 'free wave' reference: at a point well in front of the barrier on axis
free_pt = final_full[X_BAR - 5, NY // 2]
slit_a_edge = final_full[X_BAR + 2, SLIT_A[1] + 1]   # just past upper edge of slit A
slit_b_edge = final_full[X_BAR + 2, SLIT_B[0] - 1]   # just past lower edge of slit B
slit_a_mid  = final_full[X_BAR + 2, (SLIT_A[0]+SLIT_A[1])//2]   # just past slit A
slit_b_mid  = final_full[X_BAR + 2, (SLIT_B[0]+SLIT_B[1])//2]
print("\nm-mode content (fraction):")
print(f"  m-labels      : {m_labels.tolist()}")
print(f"  free upstream : {np.round(m_content(free_pt),3)}")
print(f"  slit A center : {np.round(m_content(slit_a_mid),3)}")
print(f"  slit B center : {np.round(m_content(slit_b_mid),3)}")
print(f"  slit A edge+1 : {np.round(m_content(slit_a_edge),3)}")
print(f"  slit B edge-1 : {np.round(m_content(slit_b_edge),3)}")

# ─── plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
vmax = max(s.max() for s in snaps.values())
for ax, t in zip(axes, SAVE_T):
    im = ax.imshow(np.log10(snaps[t].T + 1e-12), origin='lower',
                   aspect='equal', cmap='magma', vmin=-8, vmax=np.log10(vmax))
    ax.axvline(X_BAR, c='cyan', lw=0.6)
    ax.axvline(X_SCR, c='lime', lw=0.6, ls='--')
    ax.axhspan(SLIT_A[0], SLIT_A[1], xmin=X_BAR/NX, xmax=(X_BAR+0.7)/NX,
               color='cyan', alpha=0.2)
    ax.set_title(f"t = {t}")
    ax.set_xlabel("x (index)"); ax.set_ylabel("y (index)")
fig.suptitle(r"Hex-lattice double slit:  $\log_{10}|\psi_{phys}|^2$")
plt.tight_layout()
plt.savefig('fig_slit_heatmap.png', dpi=110)
plt.close()
print("→ fig_slit_heatmap.png")

fig, ax = plt.subplots(figsize=(8.5, 4.4))
y_phys_arr = np.arange(NY) * DY_PHYS
ax.plot(y_phys_arr, screen / (screen.max() + 1e-30), label='|ψ_phys|² (norm)')
y_center_phys = 0.5 * (y_a_phys + y_b_phys)
for n in range(-6, 7):
    ax.axvline(y_center_phys + n * dy_theory, c='r', alpha=0.3, lw=0.7)
ax.set_xlabel("y (physical units)")
ax.set_ylabel("normalized intensity at screen")
ax.set_title(f"Screen at x={X_SCR} (t={T}). Red = theory Δy={dy_theory:.3f}")
ax.set_xlim(0.6 * NY * DY_PHYS - 0.6*dy_theory*6, 0.6 * NY * DY_PHYS + 0.6*dy_theory*6)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig_slit_screen.png', dpi=110)
plt.close()
print("→ fig_slit_screen.png")

# ─── m-mode bar chart ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.4))
order = np.argsort(m_labels)
labels = [str(m_labels[i]) for i in order]
free   = m_content(free_pt)[order]
slitA  = m_content(slit_a_mid)[order]
slitB  = m_content(slit_b_mid)[order]
edgeA  = m_content(slit_a_edge)[order]
x = np.arange(len(labels))
w = 0.18
ax.bar(x - 1.5*w, free,  w, label='free upstream')
ax.bar(x - 0.5*w, slitA, w, label='slit A center')
ax.bar(x + 0.5*w, slitB, w, label='slit B center')
ax.bar(x + 1.5*w, edgeA, w, label='slit A edge (just past)')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("angular momentum m")
ax.set_ylabel("fractional m-content")
ax.set_title("m-mode content: free wave vs. slit edge")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig_slit_modes.png', dpi=110)
plt.close()
print("→ fig_slit_modes.png")

# ─── persist results for the markdown writer ─────────────────────────────
np.savez('exp1_results.npz',
         dy_theory=dy_theory, dy_meas=dy_meas,
         lam_phys=lam_phys, L_phys=L_phys, d_phys=d_phys,
         m_labels=m_labels, free=free, slitA=slitA, slitB=slitB, edgeA=edgeA,
         screen=screen, NX=NX, NY=NY, X_SCR=X_SCR)
print("\nDone. Numerical results saved to exp1_results.npz")
