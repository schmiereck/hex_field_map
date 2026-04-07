#!/usr/bin/env python3
"""
3+1D Quantum Path Integral on the Tetrahedral-Octahedral (FCC / cuboctahedron) lattice.

13 directions:
  d=0..11: 12 cuboctahedron diagonals  (Δt = 0.5,  spatial step √3/2)
  d=12   : straight (0,0,0)            (Δt = 1.0,  uses amp_pprev)

Amplitude rule:  same direction → 1,  direction change → iε.
Edge length (spacetime): 1 for all.   Speed of light c = (√3/2)/0.5 = √3.
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, product

EPS    = 0.1
SQRT3  = np.sqrt(3)
A      = SQRT3 / (2 * np.sqrt(2))   # cuboctahedron coordinate
N_DIRS = 13
C_LIGHT = SQRT3
np.set_printoptions(precision=6, suppress=True, linewidth=160)

# 12 cuboctahedron vertices (signed perms with two ±A and one 0)
MOVES = []
for zero_pos in range(3):
    for s1, s2 in product([+1, -1], repeat=2):
        v = [0.0, 0.0, 0.0]
        idxs = [i for i in range(3) if i != zero_pos]
        v[idxs[0]] = s1 * A
        v[idxs[1]] = s2 * A
        MOVES.append(tuple(v))
MOVES.append((0.0, 0.0, 0.0))   # d=12 straight
MOVES = np.array(MOVES)

# verify edge length 1 (spacetime: spatial² + Δt²)
spatial2 = (MOVES[:12]**2).sum(axis=1)
assert np.allclose(spatial2, 0.75), spatial2
assert np.allclose(np.sqrt(spatial2 + 0.25), 1.0)
assert abs(C_LIGHT - (np.sqrt(0.75) / 0.5)) < 1e-12

def C_mat(eps=EPS):
    C = np.full((N_DIRS, N_DIRS), 1j * eps, dtype=complex)
    np.fill_diagonal(C, 1.0 + 0j)
    return C

def TM_half(kx, ky, kz, eps=EPS):
    """26x26 half-step transfer matrix."""
    C = C_mat(eps)
    phases = np.exp(-1j * (kx * MOVES[:12, 0] + ky * MOVES[:12, 1] + kz * MOVES[:12, 2]))
    A_blk = np.zeros((N_DIRS, N_DIRS), dtype=complex)
    for d in range(12):
        A_blk[d, :] = phases[d] * C[d, :]
    B_blk = np.zeros((N_DIRS, N_DIRS), dtype=complex)
    B_blk[12, :] = C[12, :]
    M = np.zeros((2*N_DIRS, 2*N_DIRS), dtype=complex)
    M[:N_DIRS, :N_DIRS] = A_blk
    M[:N_DIRS, N_DIRS:] = B_blk
    M[N_DIRS:, :N_DIRS] = np.eye(N_DIRS)
    return M

def TM_full(kx, ky, kz, eps=EPS):
    M = TM_half(kx, ky, kz, eps)
    return M @ M

# ─── Task 1: Spectrum at k=0 ─────────────────────────────────────────────────
print("="*72)
print("Task 1: Spectrum of M_full at k=0, ε=0.1")
print("="*72)

M0 = TM_full(0.0, 0.0, 0.0)
lam, V = np.linalg.eig(M0)
E   = -np.angle(lam)
absL = np.abs(lam)

order = np.argsort(E)
print(f"\n{'idx':>3} {'E':>12} {'|λ|':>12} {'Re(λ)':>12} {'Im(λ)':>12}")
for j, i in enumerate(order):
    print(f"{j:>3} {E[i]:>12.6f} {absL[i]:>12.6f} {lam[i].real:>12.6f} {lam[i].imag:>12.6f}")

m_target = float(np.arctan2(2*EPS, 1 - EPS**2))
band_mask = np.abs(E - m_target) < 0.01
print(f"\nm_target = arctan(2ε/(1-ε²)) = {m_target:.6f}")
print(f"Eigenvalues within 0.01 of m_target : {band_mask.sum()}")
print(f"Predicted band degeneracy           : 11")

# Verify straight component zero in band eigenvectors
band_idx = np.where(band_mask)[0]
V_band = V[:, band_idx]
# Sort cleanly via SVD (rank-revealing)
U, sv, _ = np.linalg.svd(V_band, full_matrices=False)
print(f"\nSVD singular values of band eigenvector matrix ({len(band_idx)} vectors):")
print(" ", sv)
rank = int((sv > 1e-8).sum())
print(f"effective dim of band subspace = {rank}")

# Pick the orthonormalised basis (first `rank` columns of U) and check
# residuals + straight component
print(f"\n{'k':>3} {'residual ‖Mψ−λψ‖':>22} {'|ψ_straight|':>22}")
band_evals = []
for k in range(rank):
    psi = U[:, k]
    Mpsi = M0 @ psi
    lam_k = psi.conj() @ Mpsi
    res = np.linalg.norm(Mpsi - lam_k * psi)
    # ψ has 26 components: dirs 0..12 (current) and 13..25 (prev)
    straight_abs = np.sqrt(abs(psi[12])**2 + abs(psi[12+N_DIRS])**2)
    band_evals.append(lam_k)
    print(f"{k:>3} {res:>22.3e} {straight_abs:>22.3e}")

# Check the C matrix s-wave eigenvalue
print("\n─── Eigenstructure of the 13×13 amplitude matrix C ───")
C = C_mat()
lamC = np.linalg.eigvals(C)
for l in sorted(lamC, key=lambda z: -z.imag):
    print(f"  λ_C = {l.real:+.6f}{l.imag:+.6f}j   |λ_C|={abs(l):.6f}")
print(f"  → s-wave (all-ones): λ_C = 1 + 12·iε = 1 + {12*EPS}i  (one eigenvector)")
print(f"  → 12 sum-zero modes: λ_C = 1 - iε    = 1 - {EPS}i      (12-fold)")

# ─── Task 2: Symmetry — Oh group ─────────────────────────────────────────────
print("\n" + "="*72)
print("Task 2: Oh symmetry analysis")
print("="*72)

def perm_for_g(g3):
    """Given a 3x3 signed permutation, return the 13x13 permutation
    matrix on directions (last index 12 = straight, fixed)."""
    P = np.zeros((N_DIRS, N_DIRS))
    P[12, 12] = 1.0
    for d in range(12):
        v = MOVES[d]
        v2 = g3 @ v
        # find which direction matches
        diffs = np.linalg.norm(MOVES[:12] - v2, axis=1)
        d2 = int(np.argmin(diffs))
        assert diffs[d2] < 1e-10
        P[d2, d] = 1.0
    return P

# Enumerate Oh as 48 signed permutations of 3 axes
Oh_3x3 = []
for perm in permutations(range(3)):
    for signs in product([+1, -1], repeat=3):
        g = np.zeros((3, 3))
        for i, p in enumerate(perm):
            g[i, p] = signs[i]
        Oh_3x3.append(g)
assert len(Oh_3x3) == 48

Oh_13 = [perm_for_g(g) for g in Oh_3x3]
# lift to 26
Oh_26 = [np.block([[P, np.zeros((13,13))],[np.zeros((13,13)), P]]) for P in Oh_13]

# Check commutation [M0, R(g)] = 0 for all 48
max_comm = max(np.linalg.norm(M0 @ R - R @ M0) for R in Oh_26)
print(f"max ‖[M_full, R(g)]‖ over 48 Oh elements = {max_comm:.2e}")

# Conjugacy classes by (det, trace) — sufficient to distinguish Oh classes
# Oh classes: E(1), 8C3(8), 6C2'(6), 6C4(6), 3C2(3), i(1), 6S4(6), 8S6(8), 3σh(3), 6σd(6)
def class_key(g):
    det = round(np.linalg.det(g))
    tr  = round(np.trace(g))
    # Distinguish 3C2 (diag) from 6C2' (non-diag), and 3σh from 6σd
    off = float(np.abs(g - np.diag(np.diag(g))).sum())
    is_diag = off < 1e-9
    return (det, tr, is_diag)

# Build a representative for each class and compute χ_band
classes = {}
for g, R in zip(Oh_3x3, Oh_26):
    k = class_key(g)
    classes.setdefault(k, []).append(R)

# χ_band(g) = tr(Q† R Q) where Q is the orthonormal band basis
Q = U[:, :rank]
print(f"\n{'class':<10} {'size':>5} {'(det,tr)':>10} {'χ_band':>20}")
chi_band = {}
for k, Rs in classes.items():
    chi_g = (Q.conj().T @ Rs[0] @ Q).trace()
    chi_band[k] = chi_g
    print(f"{'':<10} {len(Rs):>5} {str(k):>10} {chi_g.real:+.4f}{chi_g.imag:+.4f}j")

# Standard Oh character table
# Class order: E, 8C3, 6C2', 6C4, 3C2, i, 6S4, 8S6, 3σh, 6σd
# Class keys (det,tr): E=(1,3), 8C3=(1,0), 6C2'=(1,-1), 6C4=(1,1),
#                     3C2=(1,-1) — same key as 6C2'! distinguish by size
#                     i=(-1,-3), 6S4=(-1,-1), 8S6=(-1,0), 3σh=(-1,1), 6σd=(-1,1)
# Need smarter class id: use (det, trace, size).
classes2 = classes  # already keyed by (det, tr, is_diag)

CLASS_ORDER = [
    ('E',     ( 1,  3, True)),
    ('8C3',   ( 1,  0, False)),
    ('3C2',   ( 1, -1, True)),
    ('6C4',   ( 1,  1, False)),
    ('6C2p',  ( 1, -1, False)),
    ('i',     (-1, -3, True)),
    ('6S4',   (-1, -1, False)),
    ('8S6',   (-1,  0, False)),
    ('3sh',   (-1,  1, True)),
    ('6sd',   (-1,  1, False)),
]
CHAR_TABLE = {
    'A1g': [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'A2g': [ 1, 1, 1,-1,-1, 1,-1, 1, 1,-1],
    'Eg':  [ 2,-1, 2, 0, 0, 2, 0,-1, 2, 0],
    'T1g': [ 3, 0,-1, 1,-1, 3, 1, 0,-1,-1],
    'T2g': [ 3, 0,-1,-1, 1, 3,-1, 0,-1, 1],
    'A1u': [ 1, 1, 1, 1, 1,-1,-1,-1,-1,-1],
    'A2u': [ 1, 1, 1,-1,-1,-1, 1,-1,-1, 1],
    'Eu':  [ 2,-1, 2, 0, 0,-2, 0, 1,-2, 0],
    'T1u': [ 3, 0,-1, 1,-1,-3,-1, 0, 1, 1],
    'T2u': [ 3, 0,-1,-1, 1,-3, 1, 0, 1,-1],
}

# Compute χ_band per class in CLASS_ORDER
chi_band_vec = []
class_sizes  = []
for name, key in CLASS_ORDER:
    if key not in classes2:
        # try to fall back: group by (det,tr,size); if missing, use 0 (no element)
        chi_band_vec.append(0)
        class_sizes.append(0)
        continue
    R = classes2[key][0]
    chi = complex((Q.conj().T @ R @ Q).trace())
    chi_band_vec.append(chi)
    class_sizes.append(len(classes2[key]))

print("\nχ_band per Oh class:")
for (name, _), c, s in zip(CLASS_ORDER, chi_band_vec, class_sizes):
    print(f"  {name:>6} (size {s}): {c.real:+.4f}{c.imag:+.4f}j")

# Decompose: m_i = (1/48) Σ_class size · χ_irrep(class) · χ_band(class)*
#           = (1/48) Σ size · χ_irrep · conj(χ_band)
print("\nIrrep multiplicities in physical band:")
total = 0
band_decomp = []
for irrep, chars in CHAR_TABLE.items():
    m = 0.0 + 0j
    for chi_irr, chi_b, s in zip(chars, chi_band_vec, class_sizes):
        m += s * chi_irr * np.conj(chi_b)
    m /= 48.0
    if abs(m) > 1e-3:
        print(f"  {irrep:>4}: {m.real:+.3f} {m.imag:+.3f}j")
        if abs(m.imag) < 1e-3 and abs(m.real - round(m.real)) < 1e-2:
            mi = int(round(m.real))
            band_decomp.append((irrep, mi))
            total += mi * (2 if 'E' in irrep and irrep[0] == 'E' else (3 if irrep[0]=='T' else 1))

print(f"\nSum of (multiplicity × irrep dim) = {total}  (should be {rank})")
print("Physical band irrep content:", " ⊕ ".join(f"{m}·{r}" if m>1 else r for r, m in band_decomp))

# Same for the 13-dim direction representation (sanity)
print("\nSanity: irrep decomp of 13-dim direction rep:")
chi_dir = []
for name, key in CLASS_ORDER:
    R = classes2.get(key, [None])[0]
    if R is None:
        chi_dir.append(0); continue
    chi_dir.append(np.trace(R[:13, :13]))
for irrep, chars in CHAR_TABLE.items():
    m = sum(s*ci*cd for ci, cd, s in zip(chars, chi_dir, class_sizes)) / 48.0
    if abs(m) > 1e-3:
        print(f"  {irrep:>4}: {m:+.3f}")

# ─── Task 3: Dispersion ──────────────────────────────────────────────────────
print("\n" + "="*72)
print("Task 3: Dispersion E(k) along 4 directions")
print("="*72)

DIRS = {
    'kx':         np.array([1.0, 0.0, 0.0]),
    'ky':         np.array([0.0, 1.0, 0.0]),
    'face_diag':  np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
    'body_diag':  np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
}

def track_band(kvecs):
    """Track an isotropic sub-band via eigenvector overlap with the k=0 band."""
    # Reference: orthonormal basis Q of the 11-fold band at k=0
    Q_ref = U[:, :rank].copy()
    Es = []
    psi_prev = None
    for j, kv in enumerate(kvecs):
        M = TM_full(*kv)
        lam_k, V_k = np.linalg.eig(M)
        a_k = np.abs(lam_k)
        valid = a_k > 0.5
        idxs = np.where(valid)[0]
        e_v = -np.angle(lam_k[idxs])
        V_v = V_k[:, idxs]
        V_v /= np.linalg.norm(V_v, axis=0, keepdims=True)
        if psi_prev is None:
            # Pick the eigenvector with maximum projection onto the band subspace
            scores = np.linalg.norm(Q_ref.conj().T @ V_v, axis=0)
            i = int(np.argmax(scores))
        else:
            overlaps = np.abs(psi_prev.conj() @ V_v)
            i = int(np.argmax(overlaps))
        psi_prev = V_v[:, i]
        Es.append(e_v[i])
    return np.array(Es)

k_arr = np.linspace(0.0, 0.5, 25)
disp = {}
for name, dir_vec in DIRS.items():
    kvecs = [k * dir_vec for k in k_arr]
    disp[name] = track_band(kvecs)

# Fit c (fixed √3) and m
print(f"\n{'direction':<12} {'m_fit':>10} {'c_fit':>8} {'RMSE':>12}")
for name, Es in disp.items():
    E_ref = np.sqrt(C_LIGHT**2 * k_arr**2 + m_target**2)
    rmse = float(np.sqrt(np.mean((Es - E_ref)**2)))
    print(f"{name:<12} {Es[0]:>10.6f} {C_LIGHT:>8.4f} {rmse:>12.3e}")

# Isotropy: max diff between directions at |k| ≤ 0.4
mask = k_arr <= 0.4
all_E = np.array([disp[n][mask] for n in DIRS])
iso_err = float((all_E.max(axis=0) - all_E.min(axis=0)).max())
print(f"\nIsotropy error (max spread between directions, |k|≤0.4): {iso_err:.3e}")

# ─── Task 4: Group velocity / causality ──────────────────────────────────────
print("\n" + "="*72)
print("Task 4: Group velocity & causality")
print("="*72)

vg_kx = np.gradient(disp['kx'], k_arr)
print(f"max |v_g| along kx in [0, 0.5] = {np.abs(vg_kx).max():.4f}   (c = {C_LIGHT:.4f})")
print(f"v_g(k→0) ≈ {vg_kx[1]:.4f}  (should be small)")

# ─── Task 5: Band splitting at k > 0 ─────────────────────────────────────────
print("\n" + "="*72)
print("Task 5: Band splitting at k > 0")
print("="*72)

k_split = np.linspace(0.0, 0.5, 30)
sub_bands = []
for k in k_split:
    M = TM_full(k, 0.0, 0.0)
    lam_k = np.linalg.eigvals(M)
    e_k = -np.angle(lam_k)
    a_k = np.abs(lam_k)
    near = e_k[(np.abs(e_k - m_target) < 1.0) & (a_k > 0.5)]
    near.sort()
    sub_bands.append(near)
maxn = max(len(s) for s in sub_bands)

# ─── Plots ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for j, i in enumerate(order):
    ax.scatter(j, E[i], c='C0', s=30)
ax.set_xlabel('eigenvalue index (sorted)')
ax.set_ylabel('E = -arg(λ)')
ax.set_title(f'3+1D spectrum at k=0, ε={EPS}')
ax.axhline(m_target, c='r', ls='--', alpha=0.6, label=f'2ε ≈ {m_target:.4f}')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('spectrum_3d.png', dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
for name, Es in disp.items():
    ax.plot(k_arr, Es, '-o', ms=4, label=name)
ax.plot(k_arr, np.sqrt(C_LIGHT**2 * k_arr**2 + m_target**2),
        'k--', alpha=0.7, label=r'$\sqrt{3k^2+m^2}$')
ax.set_xlabel('|k|')
ax.set_ylabel('E')
ax.set_title('3+1D dispersion along 4 directions')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('dispersion_3d.png', dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
for k, sb in zip(k_split, sub_bands):
    ax.scatter([k]*len(sb), sb, c='C0', s=15)
ax.axhline(m_target, c='r', ls='--', alpha=0.6, label=f'2ε ≈ {m_target:.4f}')
ax.set_xlabel('k_x')
ax.set_ylabel('E')
ax.set_title('Band splitting near 2ε along k_x')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('band_splitting_3d.png', dpi=110)
plt.close()

print("\nFigures written: spectrum_3d.png, dispersion_3d.png, band_splitting_3d.png")
