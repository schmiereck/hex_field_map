#!/usr/bin/env python3
"""
Experiment 2 — 3+1D box states on the tetrahedral-octahedral lattice.

We solve the eigenproblem of M_full restricted to a finite cubic box of
side N (odd, so it has a single center site) with hard-wall boundary
conditions, using a matrix-free LinearOperator and scipy ARPACK to find
the eigenvalues nearest the bulk band E = 2ε.

For each eigenvector we compute its content in the four Oh irreps that
make up the bulk physical band:  T1u, T2u, Eg, T2g.

The 12 lightlike directions of the cuboctahedron have integer offsets
(±1, ±1, 0) and permutations, so the "real-space" lattice is just the
simple cubic Z^3, and a half-step is a sum of 12 nearest-neighbour shifts
plus the straight (rest) move.
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, product
from scipy.sparse import csr_matrix, eye as sp_eye
from scipy.sparse.linalg import LinearOperator, eigs, splu

EPS    = 0.1
N_DIRS = 13
np.set_printoptions(precision=4, suppress=True, linewidth=160)

# ─── direction list (integer offsets) — same order as quantum_3d.MOVES ────
MOVES_INT = []
for zero_pos in range(3):
    for s1, s2 in product([+1, -1], repeat=2):
        v = [0, 0, 0]
        idxs = [i for i in range(3) if i != zero_pos]
        v[idxs[0]] = s1
        v[idxs[1]] = s2
        MOVES_INT.append(tuple(v))
MOVES_INT.append((0, 0, 0))    # d=12 straight
MOVES_INT = np.array(MOVES_INT, dtype=int)
assert len(MOVES_INT) == 13

# ─── C matrix ─────────────────────────────────────────────────────────────
def C_mat(eps=EPS):
    C = np.full((N_DIRS, N_DIRS), 1j*eps, dtype=complex)
    np.fill_diagonal(C, 1.0+0j)
    return C
C = C_mat()

# ─── matvec on a finite N×N×N box, hard walls ────────────────────────────
def build_M_half_sparse(N):
    """Build M_half as a sparse CSR matrix on the (N,N,N,2,13) state space.
    M_half maps (cur, prev) -> (new_cur, cur). new_cur = sum_d shifted (cur @ C)_d
    + (prev @ C)[d=12] for the straight move."""
    sz = N*N*N
    dim = sz * 2 * N_DIRS
    rows, cols, data = [], [], []

    def site_idx(x, y, z, half, d):
        return ((((x*N + y)*N + z)*2 + half)*N_DIRS + d)

    # The matrix M_half acts as: out_state = M_half @ in_state, where
    #   out_state[x,y,z,half=0,d_out] = sum over (x',y',z',d_in) of contributions
    # For each output index we list the (in indices, coefficient) pairs.
    # Rule:
    #   new_cur[x,y,z,d_out] = sum_{d'} C[d_out, d'] * cur[x',y',z',d']  shifted by MOVES_INT[d_out]
    #                       (only for d_out in 0..11)
    #   new_cur[x,y,z,12]    = sum_{d'} C[12, d'] * prev[x,y,z,d']
    #   new_prev[x,y,z,d]    = cur[x,y,z,d]
    # In terms of input (in_x,in_y,in_z) for new_cur[x,y,z,d_out]:
    #   if d_out < 12: in = (x,y,z) - MOVES_INT[d_out]; must be in box
    #   if d_out == 12: in = (x,y,z); pull from prev
    for x in range(N):
        for y in range(N):
            for z in range(N):
                for d_out in range(N_DIRS):
                    out_idx = site_idx(x, y, z, 0, d_out)
                    if d_out < 12:
                        dx, dy, dz = MOVES_INT[d_out]
                        ix, iy, iz = x - dx, y - dy, z - dz
                        if not (0 <= ix < N and 0 <= iy < N and 0 <= iz < N):
                            continue
                        for d_in in range(N_DIRS):
                            c = C[d_out, d_in]
                            if c != 0:
                                in_idx = site_idx(ix, iy, iz, 0, d_in)
                                rows.append(out_idx); cols.append(in_idx); data.append(c)
                    else:  # d_out == 12, pull from prev
                        for d_in in range(N_DIRS):
                            c = C[12, d_in]
                            if c != 0:
                                in_idx = site_idx(x, y, z, 1, d_in)
                                rows.append(out_idx); cols.append(in_idx); data.append(c)
                # new_prev[x,y,z,d] = cur[x,y,z,d]
                for d in range(N_DIRS):
                    out_idx = site_idx(x, y, z, 1, d)
                    in_idx  = site_idx(x, y, z, 0, d)
                    rows.append(out_idx); cols.append(in_idx); data.append(1.0+0j)
    M = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=complex)
    return M

def make_matvec_full(N):
    """Returns a function v → M_full @ v.
    v has shape (N**3 * 26,) representing (x,y,z,d_curr OR d_prev)."""
    size = N*N*N * 2 * N_DIRS
    def half_step(state):
        # state shape (N,N,N,2,13). out shape same.
        cur = state[..., 0, :]
        prv = state[..., 1, :]
        wc  = cur @ C       # (N,N,N,13)
        ws  = prv @ C
        new_cur = np.zeros_like(cur)
        for d in range(12):
            dx, dy, dz = MOVES_INT[d]
            sx_src = slice(max(0,-dx), N - max(0,dx))
            sx_dst = slice(max(0, dx), N - max(0,-dx))
            sy_src = slice(max(0,-dy), N - max(0,dy))
            sy_dst = slice(max(0, dy), N - max(0,-dy))
            sz_src = slice(max(0,-dz), N - max(0,dz))
            sz_dst = slice(max(0, dz), N - max(0,-dz))
            new_cur[sx_dst, sy_dst, sz_dst, d] += wc[sx_src, sy_src, sz_src, d]
        # straight move (d=12) uses prev (full step delay) — d=12 component of ws
        new_cur[..., 12] += ws[..., 12]
        # Hard walls already enforced by slice clipping (no wrap, no out-of-box).
        # New state: new prev = cur, new cur = new_cur
        out = np.zeros_like(state)
        out[..., 0, :] = new_cur
        out[..., 1, :] = cur
        return out
    def matvec(v):
        s = v.reshape(N, N, N, 2, N_DIRS)
        s = half_step(s)
        s = half_step(s)
        return s.ravel()
    return LinearOperator((size, size), matvec=matvec, dtype=complex)

# ─── Oh group: 48 signed permutations of 3 axes ──────────────────────────
def build_Oh():
    Oh = []
    for perm in permutations(range(3)):
        for signs in product([+1, -1], repeat=3):
            g = np.zeros((3,3), dtype=int)
            for i, p in enumerate(perm):
                g[i, p] = signs[i]
            Oh.append(g)
    return Oh
Oh_3 = build_Oh()
assert len(Oh_3) == 48

def perm13_for_g(g):
    """Direction permutation for g ∈ Oh."""
    P = np.zeros((N_DIRS, N_DIRS))
    P[12, 12] = 1.0
    for d in range(12):
        v = MOVES_INT[d]
        v2 = g @ v
        diffs = np.linalg.norm(MOVES_INT[:12] - v2, axis=1)
        d2 = int(np.argmin(diffs))
        assert diffs[d2] < 1e-10
        P[d2, d] = 1.0
    return P
PERMS13 = [perm13_for_g(g) for g in Oh_3]

# Class identification (key = (det, tr, is_diag) — distinguishes all 10 Oh classes)
def class_key(g):
    det = round(np.linalg.det(g))
    tr  = round(np.trace(g))
    off = float(np.abs(g - np.diag(np.diag(g))).sum())
    return (det, tr, off < 1e-9)

CLASS_ORDER = [
    ('E',    ( 1,  3, True)),
    ('8C3',  ( 1,  0, False)),
    ('3C2',  ( 1, -1, True)),
    ('6C4',  ( 1,  1, False)),
    ('6C2p', ( 1, -1, False)),
    ('i',    (-1, -3, True)),
    ('6S4',  (-1, -1, False)),
    ('8S6',  (-1,  0, False)),
    ('3sh',  (-1,  1, True)),
    ('6sd',  (-1,  1, False)),
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
IRREP_DIM = {'A1g':1,'A2g':1,'Eg':2,'T1g':3,'T2g':3,'A1u':1,'A2u':1,'Eu':2,'T1u':3,'T2u':3}

# ─── Apply group element g ∈ Oh to a box state ───────────────────────────
# The group acts on positions (around center) AND on direction index.
def apply_group(state, g, P13):
    """state shape (N,N,N,2,13).  Returns the rotated state."""
    N = state.shape[0]
    c = (N - 1) / 2.0
    # New site (x',y',z') gets value from old site g^{-1} @ ((x',y',z') - c) + c
    # Then direction permutation P13 mixes the 13-component vector.
    # For signed-permutation g we have g^{-1} = g^T.
    gT = g.T
    # Build index map by enumerating new coords
    xs = np.arange(N) - c
    ys = np.arange(N) - c
    zs = np.arange(N) - c
    # For each output (x',y',z') compute (x_old,y_old,z_old) = gT @ (x',y',z')
    # gT is signed perm so we can do this with axes swap + flips
    perm = np.argmax(np.abs(g), axis=0)   # perm[i] = which axis of input ends up at output i? careful
    # Easier: build the rotated state by direct indexing.
    # Find permutation of axes and sign flips for gT acting on a 3-vector.
    # gT[i,j] = g[j,i]; only one nonzero per row.
    out = np.empty_like(state)
    Nrange = np.arange(N)
    # Compute new coordinates of every old (x,y,z): (xn,yn,zn) = g @ ((x,y,z) - c) + c
    coords_old = np.stack(np.meshgrid(Nrange, Nrange, Nrange, indexing='ij'), axis=-1) - c
    coords_new = coords_old @ g.T + c  # (N,N,N,3)
    coords_new = np.round(coords_new).astype(int)
    valid = ((coords_new >= 0) & (coords_new < N)).all(axis=-1)
    cx, cy, cz = coords_new[..., 0], coords_new[..., 1], coords_new[..., 2]
    # Apply direction permutation: new_dirs = P13 @ old_dirs (P13 acts on direction axis)
    rotated_dirs = state @ P13.T   # (N,N,N,2,13)
    # Place into out
    out[...] = 0
    out[cx[valid], cy[valid], cz[valid]] = rotated_dirs[valid]
    return out

def compute_irrep_content(eig_vec, N):
    """Returns dict irrep -> ‖P_irr ψ‖² / ‖ψ‖²."""
    psi = eig_vec.reshape(N, N, N, 2, N_DIRS)
    nrm2 = float((np.abs(psi)**2).sum())
    if nrm2 < 1e-30:
        return {}
    # Compute χ(g) = ⟨ψ|R(g)|ψ⟩ for each g, then m_irr = (d_irr/48) Σ χ_irr(g)* χ(g)
    # where the irrep projection is ‖P_irr ψ‖² = (d_irr/|G|) Σ_g χ_irr(g)* ⟨ψ|R(g)|ψ⟩
    chi = np.zeros(48, dtype=complex)
    for k, (g, P13) in enumerate(zip(Oh_3, PERMS13)):
        psi_rot = apply_group(psi, g, P13)
        chi[k] = (psi.conj() * psi_rot).sum() / nrm2
    # Bin by class
    chi_by_class = {}
    for k, g in enumerate(Oh_3):
        ck = class_key(g)
        chi_by_class.setdefault(ck, []).append(chi[k])
    chi_avg = {ck: np.mean(v) for ck, v in chi_by_class.items()}
    sizes   = {ck: len(v) for ck, v in chi_by_class.items()}
    out = {}
    for irrep, chars in CHAR_TABLE.items():
        d_irr = IRREP_DIM[irrep]
        s = 0+0j
        for (name, key), char in zip(CLASS_ORDER, chars):
            if key not in chi_avg: continue
            s += sizes[key] * char * chi_avg[key]
        # ‖P_irr ψ‖² / ‖ψ‖² = (d_irr / 48) * Σ_g χ_irr(g)* ⟨ψ|R(g)|ψ⟩
        # Since χ_irr is real, conj is identity. Take real part.
        out[irrep] = float((d_irr / 48.0) * s.real)
    return out

# ─── Main: scan box sizes and compute eigenvalues + irrep content ────────
def run_box(N, k_eig=20):
    # Build M_half sparse, then compute M_full = M_half @ M_half (still sparse).
    print(f"\n[N={N}] building sparse M_half...", flush=True)
    Mh = build_M_half_sparse(N)
    Mf = (Mh @ Mh).tocsc()
    m_target = float(np.arctan2(2*EPS, 1-EPS**2))
    lam_target = (1+EPS**2) * np.exp(-1j * m_target)
    dim = Mf.shape[0]
    print(f"  dim={dim}, nnz(M_full)={Mf.nnz}, target λ={lam_target:.4f}")
    # Shift-invert: find eigenvalues nearest sigma=lam_target
    vals, vecs = eigs(Mf, k=k_eig, sigma=lam_target, which='LM',
                      maxiter=400, tol=1e-7)
    E = -np.angle(vals)
    aL = np.abs(vals)
    # Sort by distance to lam_target
    dist = np.abs(vals - lam_target)
    order = np.argsort(dist)
    print(f"  found {len(vals)} eigenvalues, sorted by distance to band:")
    print(f"  {'idx':>3} {'|λ|':>8} {'E':>10} {'dist':>10}")
    results = []
    for j in order[:k_eig]:
        irr = compute_irrep_content(vecs[:, j], N)
        nonzero = {k: round(v,2) for k, v in irr.items() if abs(v) > 0.02}
        dom = max(irr.items(), key=lambda kv: kv[1]) if irr else ('—', 0)
        print(f"  {j:>3} {aL[j]:>8.4f} {E[j]:>10.5f} {dist[j]:>10.4f}   "
              f"dom={dom[0]}({dom[1]:.2f})  irreps={nonzero}")
        results.append((vals[j], aL[j], E[j], irr))
    return results

print("="*70)
print("Experiment 2 — 3D box states (sparse ARPACK)")
print("="*70)

m_target = float(np.arctan2(2*EPS, 1-EPS**2))
print(f"Bulk band reference: E = 2ε ≈ {m_target:.6f}")

# Run a few odd box sizes
all_results = {}
for N in [7, 9, 11]:
    all_results[N] = run_box(N, k_eig=40)

# ─── Energy-vs-L plot per dominant irrep ─────────────────────────────────
import json
levels_by_irr = {ir: {N: [] for N in all_results} for ir in ('T1u','T2u','Eg','T2g','A1g','A2u')}
for N, res in all_results.items():
    for lam, aL, E, irr in res:
        cands = {ir: irr.get(ir, 0) for ir in levels_by_irr}
        best_ir, best_v = max(cands.items(), key=lambda kv: kv[1])
        if best_v > 0.5:
            levels_by_irr[best_ir][N].append(E)

# Print the cleanest summary: for each box size, the LOWEST E in each band irrep
print("\n--- Band-irrep ground-state energies (lowest E per irrep) ---")
print(f"{'N':>4}  {'T1u':>10}  {'T2u':>10}  {'Eg':>10}  {'T2g':>10}  {'A1g':>10}  {'A2u':>10}")
for N in sorted(all_results):
    row = [N]
    for ir in ('T1u','T2u','Eg','T2g','A1g','A2u'):
        Es = levels_by_irr[ir][N]
        row.append(min(Es) if Es else float('nan'))
    print(f"{row[0]:>4}  " + "  ".join(f"{x:>10.5f}" for x in row[1:]))

print("\nEnergy levels by dominant irrep (E):")
for ir, dct in levels_by_irr.items():
    for N, Es in dct.items():
        Es_sorted = sorted(set(round(e, 5) for e in Es))
        print(f"  {ir} N={N}: {Es_sorted}")

# Save & plot
fig, ax = plt.subplots(figsize=(8, 5))
colors = {'T1u':'C0','T2u':'C1','Eg':'C2','T2g':'C3','A1g':'C4','A2u':'C5'}
for ir, dct in levels_by_irr.items():
    Ns, Es_min = [], []
    for N, Es in dct.items():
        if Es:
            Ns.append(N); Es_min.append(min(Es))
    if Ns:
        ax.plot(Ns, Es_min, '-o', color=colors[ir], label=f"{ir} (lowest)")
ax.axhline(m_target, c='k', ls='--', alpha=0.6, label='bulk 2ε')
ax.set_xlabel('box size N (lattice units)')
ax.set_ylabel('E (lowest in irrep, near band)')
ax.set_title('3D box states: irrep splitting vs box size')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig_box_spectrum.png', dpi=110)
plt.close()
print("→ fig_box_spectrum.png")

# Save numerics
np.savez('exp2_results.npz',
         m_target=m_target,
         levels=json.dumps({ir: {str(N): sorted(set(round(e,5) for e in Es))
                                  for N, Es in dct.items()}
                            for ir, dct in levels_by_irr.items()}))
print("\nDone. Saved exp2_results.npz")
