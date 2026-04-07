#!/usr/bin/env python3
"""
Experiment 1b — Double slit on a 2+1D *square* lattice (Feynman-checkerboard
analog).

4 lightlike moves: (±1, 0, +1) and (0, ±1, +1) on a square lattice.
Edge length = √2 in spacetime, c = 1 (geometric).
Same iε amplitude rule.

This is the natural 2+1D square analog of the 1+1D Feynman checkerboard.
NOTE: c = 1 here (vs. c = √3 on the hex lattice). To make a fair comparison
we normalize lengths by *wavelength in lattice index units*: at the same k0,
the hex lattice has λ_idx = 2π/(k0·DX_PHYS_hex) = 2π/(0.3·√3/4) ≈ 48.4 idx,
the square lattice has λ_idx = 2π/(k0·1)             ≈ 20.94 idx.
So we *match wavelength in index units* by tuning k0_sq accordingly, and
match L/λ and d/λ to the hex setup.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

EPS = 0.1
N_DIRS_SQ = 5   # 4 moves + 1 "rest" (no-op) = 5; or just 4. Use 4 + rest=1 for symmetry.
# Actually use 4 directions only — Δt = 1 for all, no rest needed for 1st-order recurrence.

# We mirror the *physical* hex experiment by matching λ/L and λ/d ratios.
# Hex (in physical units): λ=20.944, L=21.65, d=9.0 → L/λ=1.034, d/λ=0.430
# On the square lattice (c=1, all moves Δt=1), we choose physical params to
# reproduce those ratios. Use k0_sq = 0.3 (same momentum number); λ_sq = 2π/0.3 ≈ 20.94.
# Then L_sq must be ≈ 21.65 phys = 21.65 index, and d_sq ≈ 9 phys = 9 index.
# Source at x=8, barrier at x=30, screen at x=80, slits y∈[51,55] and [65,69]
# (symmetric around y=60, separation 14 to give d=14 phys → d/λ=0.668).
# To match d/λ=0.430, place slits with y separation 9 → centers at 56 and 65, etc.
# Use slits y∈[54,57] and [62,65] (centers 55.5 and 63.5, sep 8 → d/λ=0.382).

K0     = 0.3
NX, NY = 120, 120
T      = 80           # square lattice slower (c=1 vs √3); need more time
X0, Y0 = 8, 60
X_BAR  = 30
SLIT_A = (54, 57)     # y-index inclusive
SLIT_B = (63, 66)
X_SCR  = 80

print("="*70)
print("Experiment 1b — Square-lattice (Feynman-2D) double slit")
print("="*70)

# 4-direction C matrix
def Cmat(n, eps):
    C = np.full((n, n), 1j*eps, dtype=complex)
    np.fill_diagonal(C, 1.0+0j)
    return C
C4 = Cmat(4, EPS)

# transfer matrix at (kx,ky): 4x4
def TM4(kx, ky, eps):
    C = Cmat(4, eps)
    ph = np.exp(-1j * np.array([kx, -kx, 0, 0])) * np.exp(-1j * np.array([0, 0, ky, -ky]))
    M = np.zeros((4,4), dtype=complex)
    for d in range(4):
        M[d, :] = ph[d] * C[d, :]
    return M

# physical band at k=0
M0 = TM4(0.0, 0.0, EPS)
lam, V = np.linalg.eig(M0)
E_lam  = -np.angle(lam)
absL   = np.abs(lam)
m_target = float(np.arctan2(2*EPS, 1 - EPS**2))
print(f"M_full=M0 at k=0 spectrum:")
for l in lam:
    print(f"  λ={l:.4f}  |λ|={abs(l):.4f}  E={-np.angle(l):.4f}")
mask  = (np.abs(E_lam - m_target) < 0.3) & (absL > 0.5)
V_band = V[:, mask]
U_b, sv, _ = np.linalg.svd(V_band, full_matrices=False)
rank = int((sv > 1e-8).sum())
Q = U_b[:, :rank]
P_phys = Q @ Q.conj().T
print(f"Physical band rank = {rank}")

# Initial wave packet
SIGMA = 6.0
xs, ys = np.arange(NX), np.arange(NY)
XX, YY = np.meshgrid(xs, ys, indexing='ij')
gauss = np.exp(-((XX-X0)**2 + (YY-Y0)**2)/(2*SIGMA**2))
phase = np.exp(1j * K0 * XX)   # square lattice: x_phys = x_idx, c=1
psi0  = (gauss * phase).astype(complex)

amp = np.zeros((NX, NY, 4), dtype=complex)
for d in range(4):
    amp[:,:,d] = psi0 / 4

# barrier
barrier = np.zeros((NX, NY), dtype=bool)
barrier[X_BAR, :] = True
barrier[X_BAR, SLIT_A[0]:SLIT_A[1]+1] = False
barrier[X_BAR, SLIT_B[0]:SLIT_B[1]+1] = False

def step_sq(amp):
    new = np.zeros_like(amp)
    w = amp @ C4
    # d=0: +x
    new[1:,  :,  0] += w[:-1, :,  0]
    # d=1: -x
    new[:-1, :,  1] += w[1:,  :,  1]
    # d=2: +y
    new[:, 1:,   2] += w[:, :-1, 2]
    # d=3: -y
    new[:, :-1,  3] += w[:, 1:,  3]
    new[barrier, :] = 0.0
    return new

for t in range(1, T+1):
    amp = step_sq(amp)

# project + density
phys = amp @ P_phys.T
dens = (np.abs(phys)**2).sum(axis=-1)
screen_sq = dens[X_SCR, :]

# Theoretical fringe spacing — square lattice (c=1, x_phys = x_idx)
lam_phys  = 2 * np.pi / K0
L_phys    = (X_SCR - X_BAR)
y_a_phys  = (SLIT_A[0]+SLIT_A[1])/2
y_b_phys  = (SLIT_B[0]+SLIT_B[1])/2
d_phys    = abs(y_b_phys - y_a_phys)
dy_theory = lam_phys * L_phys / d_phys
print(f"\nSquare-lattice fringe theory:")
print(f"  λ={lam_phys:.4f}, L={L_phys:.4f}, d={d_phys:.4f}")
print(f"  Δy_theory = {dy_theory:.4f} (phys = idx units)")

def smooth(y, w):
    k = np.ones(w)/w
    return np.convolve(y, k, mode='same')
screen_sm = smooth(screen_sq, 7)
peaks_env, _ = find_peaks(screen_sm, distance=8, prominence=screen_sm.max()*0.02)
if len(peaks_env) >= 2:
    dy_meas = float(np.median(np.diff(peaks_env)))
    print(f"  Δy_measured (envelope) = {dy_meas:.4f} idx, from {len(peaks_env)} peaks")
else:
    dy_meas = float('nan')
    print(f"  envelope: only {len(peaks_env)} peak(s)")

# Isotropy: max angular spread of |ψ|² over circles around source
# (just the asymmetry between x and y propagation, simple proxy)
phys0 = amp @ Q
density_total = (np.abs(phys0)**2).sum(axis=-1)
# x-axis vs y-axis comparison (a kind of crude isotropy proxy)
print(f"  density(x=NX//2 +20, y=Y0)  = {density_total[NX//2+20, Y0]:.4e}")
print(f"  density(x=NX//2,    y=Y0+20)= {density_total[NX//2,    Y0+20]:.4e}")

# === Comparison plot vs hex result ===
hex_data = np.load('exp1_results.npz', allow_pickle=True)
hex_screen = hex_data['screen']
hex_dy_th  = float(hex_data['dy_theory'])
hex_dy_me  = float(hex_data['dy_meas'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
# hex
ax = axes[0]
hy = np.arange(int(hex_data['NY'])) * 0.75   # DY_PHYS_hex
ax.plot(hy, hex_screen / (hex_screen.max()+1e-30))
ax.set_title(f"Hex lattice — Δy_th={hex_dy_th:.2f}, Δy_meas={hex_dy_me:.2f}")
ax.set_xlabel("y (phys)")
ax.set_ylabel("normalized intensity")
ax.grid(alpha=0.3)
# square
ax = axes[1]
sy = np.arange(NY)
ax.plot(sy, screen_sq / (screen_sq.max()+1e-30))
ax.set_title(f"Square lattice — Δy_th={dy_theory:.2f}, Δy_meas={dy_meas:.2f}")
ax.set_xlabel("y (idx = phys)")
ax.grid(alpha=0.3)
fig.suptitle("Double-slit screen pattern: hex vs square")
plt.tight_layout()
plt.savefig('fig_slit_comparison.png', dpi=110)
plt.close()
print("→ fig_slit_comparison.png")

# Node count comparison: how many lattice nodes on the screen represent
# one wavelength on each lattice?
nodes_per_lambda_hex = lam_phys / 0.75   # hex y resolution
nodes_per_lambda_sq  = lam_phys / 1.0    # sq lattice resolution
print("\nNode count per wavelength on screen y-axis:")
print(f"  hex   : {nodes_per_lambda_hex:.2f} y-nodes / λ")
print(f"  square: {nodes_per_lambda_sq:.2f} y-nodes / λ")
print(f"  ratio hex/square = {nodes_per_lambda_hex/nodes_per_lambda_sq:.2f}")
print("Equivalent grid (NX*NY) for same physical region @ same resolution:")
hex_phys_extent = 120 * 0.75
sq_nodes_for_same = int(round(hex_phys_extent / 1.0))
print(f"  hex 120x120 covers y ∈ [0, {hex_phys_extent:.1f}] phys")
print(f"  square needs {sq_nodes_for_same}x{sq_nodes_for_same} for same y extent")

np.savez('exp1b_results.npz',
         dy_theory_sq=dy_theory, dy_meas_sq=dy_meas,
         nodes_per_lambda_hex=nodes_per_lambda_hex,
         nodes_per_lambda_sq=nodes_per_lambda_sq,
         screen_sq=screen_sq)
print("\nDone. Saved exp1b_results.npz")
