"""
Sharpen the analysis of the degenerate cluster at E ~ 2*eps in M_full(k=0).
Strategy:
  1. Eigendecompose M_full at k=0 with high precision.
  2. Examine the exact spacing of eigenvalues near E = 2*eps.
  3. Simultaneously diagonalise M_full and R_60 on the cluster
     (they commute, so a joint eigenbasis exists).
  4. For each cluster member, read off
       - the R_60 quantum number m  (μ = exp(i m π/3))
       - exact eigenvalue λ
       - exact E = -arg(λ) and |λ|
  5. Decide: 5-fold or 6-fold? If 6-fold, is the extra one m=0 (A1)?
"""
import numpy as np
from numpy.linalg import eig
from quantum_spin_structure import build_M_full, R60_14

EPS = 0.1
np.set_printoptions(precision=10, suppress=False, linewidth=160)

M0  = build_M_full(0.0, 0.0, EPS)
R60 = R60_14()

# Sanity: [M0, R60] = 0 ?
comm = np.linalg.norm(M0 @ R60 - R60 @ M0)
print(f"||[M0, R60]|| = {comm:.2e}   (should be ~0)")

# 1. Diagonalise M0
lam, V = eig(M0)
E      = -np.angle(lam)
absl   = np.abs(lam)

order = np.argsort(E)
print(f"\nAll 14 eigenvalues sorted by E (ε={EPS}, target 2ε={2*EPS}):")
print(f"{'idx':>3} {'E':>14} {'|λ|':>14} {'Re(λ)':>14} {'Im(λ)':>14}")
for j, i in enumerate(order):
    mark = "  <-- cluster" if 0.15 < E[i] < 0.25 else ""
    print(f"{j:>3} {E[i]:>14.10f} {absl[i]:>14.10f} "
          f"{lam[i].real:>14.10f} {lam[i].imag:>14.10f}{mark}")

# 2. Cluster: take all eigvals with E in (0.15, 0.25)
cluster_mask = (E > 0.15) & (E < 0.25)
cl_idx = np.where(cluster_mask)[0]
print(f"\n#eigenvalues in (0.15, 0.25): {len(cl_idx)}")

print("\nExact differences within the cluster:")
cl_lam = lam[cl_idx]
cl_E   = E[cl_idx]
cl_a   = absl[cl_idx]
for i in range(len(cl_idx)):
    for j in range(i+1, len(cl_idx)):
        dE = abs(cl_E[i] - cl_E[j])
        dl = abs(cl_lam[i] - cl_lam[j])
        print(f"  ({i},{j}): ΔE = {dE:.3e}   |Δλ| = {dl:.3e}")

# 3. For each cluster eigenvector, compute its R_60 expectation
#    (when degenerate, eig() picks an arbitrary basis, so we have to
#     re-diagonalise R_60 within the cluster).
V_cl = V[:, cl_idx]                          # 14 x N_cl
# Orthonormalise (eigenvectors of a non-Hermitian matrix aren't ⊥, so use SVD)
U, sv, _ = np.linalg.svd(V_cl, full_matrices=False)
print(f"\nSVD singular values of cluster eigenvector matrix:")
print(" ", sv)
print("→ effective dim of cluster subspace =", int((sv > 1e-8).sum()))

# Project R_60 into the cluster subspace using the orthonormal basis U
R_proj = U.conj().T @ R60 @ U
mu_c, w_c = eig(R_proj)
print("\nR_60 eigenvalues on the cluster subspace:")
m_vals = np.angle(mu_c) / (np.pi / 3)
for k in range(len(mu_c)):
    print(f"  μ = {mu_c[k].real:+.6f} {mu_c[k].imag:+.6f}j   "
          f"|μ|={abs(mu_c[k]):.6f}   m={m_vals[k]:+.4f}")

# 4. For each R_60 eigenvector w_c[:,k], reconstruct the joint
#    eigenstate ψ_k = U @ w_c[:,k] and compute its M_full eigenvalue
print("\nJoint M_full + R_60 eigenstates in the cluster:")
print(f"{'m':>8} {'E':>14} {'|λ|':>14} {'<ψ|R60|ψ>':>22}")
joint_data = []
for k in range(len(mu_c)):
    psi = U @ w_c[:, k]
    psi /= np.linalg.norm(psi)
    Mpsi = M0 @ psi
    # Rayleigh quotient gives the eigenvalue (since psi is an eigenvector
    # of M0 within the degenerate cluster up to numerical noise).
    lam_k = (psi.conj() @ Mpsi)
    E_k   = -np.angle(lam_k)
    a_k   = abs(lam_k)
    R_k   = (psi.conj() @ R60 @ psi)
    # Residual: how close is psi to a true M0-eigenstate?
    res = np.linalg.norm(Mpsi - lam_k * psi)
    joint_data.append((m_vals[k], E_k, a_k, lam_k, res))
    print(f"{m_vals[k]:>+8.4f} {E_k:>14.10f} {a_k:>14.10f}  "
          f"{R_k.real:+.6f}{R_k.imag:+.6f}j   res={res:.2e}")

# 5. Identify the m=0 (s-wave / A1) candidate and gap to others
joint_sorted = sorted(joint_data, key=lambda x: abs(x[0]))
print("\nSorted by |m|:")
for m, E_k, a_k, lam_k, res in joint_sorted:
    print(f"  m={m:+.4f}  E={E_k:.10f}  |λ|={a_k:.10f}")

# Find the m≈0 mode (if any) and report its gap
m0_candidates = [d for d in joint_data if abs(d[0]) < 0.5]
m_other       = [d for d in joint_data if abs(d[0]) >= 0.5]
print("\n────────────────────────────────────────────────────────────")
if m0_candidates:
    m0 = m0_candidates[0]
    # Average eigenvalue of the "other" modes
    lam_other = np.array([d[3] for d in m_other])
    lam_mean  = lam_other.mean()
    gap_lam   = abs(m0[3] - lam_mean)
    gap_E     = abs(m0[1] - (-np.angle(lam_mean)))
    gap_abs   = abs(m0[2] - abs(lam_mean))
    print(f"m≈0 candidate found:  m={m0[0]:+.4f}")
    print(f"  λ = {m0[3]}")
    print(f"  E = {m0[1]:.10f}    |λ| = {m0[2]:.10f}")
    print(f"Mean of the other {len(m_other)} cluster modes:")
    print(f"  λ = {lam_mean}")
    print(f"  E = {-np.angle(lam_mean):.10f}    |λ| = {abs(lam_mean):.10f}")
    print(f"Gap |Δλ|  = {gap_lam:.3e}")
    print(f"Gap |ΔE|  = {gap_E:.3e}")
    print(f"Gap Δ|λ|  = {gap_abs:.3e}")
else:
    print("No m≈0 mode in the cluster.")
print("────────────────────────────────────────────────────────────")
