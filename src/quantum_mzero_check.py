"""
Why is m=0 (s-wave / A1) excluded from the physical band at E ≈ 2ε?
And: does the iε amplitude rule have chirality?
"""
import numpy as np
from numpy.linalg import eig
from quantum_spin_structure import build_M_full, R60_14

EPS = 0.1
np.set_printoptions(precision=6, suppress=True, linewidth=160)

M0  = build_M_full(0.0, 0.0, EPS)
R60 = R60_14()

# ─────────────────────────────────────────────────────────────────────
# 1. CHIRALITY TEST
# Build a 7x7 reflection sigma_v (y -> -y): swaps d=1<->5, 2<->4, fixes 0,3,6.
# If iε is non-chiral (CW = CCW), then M_full must commute with diag(σ,σ).
# ─────────────────────────────────────────────────────────────────────
S7 = np.zeros((7,7))
for a, b in {0:0, 1:5, 2:4, 3:3, 4:2, 5:1, 6:6}.items():
    S7[b,a] = 1.0
S14 = np.block([[S7, np.zeros((7,7))], [np.zeros((7,7)), S7]])
print("Chirality test:")
print(f"  ||[M_full, σ]|| = {np.linalg.norm(M0@S14 - S14@M0):.2e}")
print("  → reflection-symmetric ⇒ iε treats CW and CCW identically (NO chirality)")

# Cross-check: m and -m should have the same M_full eigenvalue
lam, V = eig(M0)
E = -np.angle(lam); absl = np.abs(lam)
print("\n  Spectrum (E, |λ|):")
for i in np.argsort(E):
    print(f"    E={E[i]:+.6f}  |λ|={absl[i]:.6f}")

# ─────────────────────────────────────────────────────────────────────
# 2. WHY IS m=0 EXCLUDED FROM THE 2ε BAND?
# Project M_full into each angular-momentum sector (eigenspace of R60).
# R_60 has eigenvalues exp(i·m·π/3) for m = 0,±1,±2,3.
# In each m sector, M_full is a small block — we just read off its
# eigenvalues and compare to E = 2ε.
# ─────────────────────────────────────────────────────────────────────
print("\nDecomposition by angular momentum (R_60 eigenvalues):")
muR, wR = eig(R60)
mR     = np.round(np.angle(muR) / (np.pi/3)).astype(int) % 6
print("  R_60 eigenvalue m-spectrum (14-dim):", sorted(mR.tolist()))

print("\nM_full projected into each m-sector:")
for m in [0, 1, 2, 3, 4, 5]:
    cols = np.where(mR == m)[0]
    if len(cols) == 0: continue
    P = wR[:, cols]
    Q, _ = np.linalg.qr(P)              # orthonormal basis
    Mm = Q.conj().T @ M0 @ Q
    lam_m, _ = eig(Mm)
    print(f"\n  m = {m}  ({'m=' + str(m if m<=3 else m-6)}):  dim = {len(cols)}")
    for l in lam_m:
        E_l = -np.angle(l); a_l = abs(l)
        tag = "  <-- 2ε band" if abs(E_l - 2*EPS) < 0.01 and abs(a_l - 1.01) < 0.01 else ""
        print(f"    λ = {l.real:+.6f}{l.imag:+.6f}j   |λ|={a_l:.6f}   E={E_l:+.6f}{tag}")

# ─────────────────────────────────────────────────────────────────────
# 3. THE STRUCTURAL REASON
# The C-matrix C[d,d'] = 1 if d=d' else iε has eigenvectors:
#   - the all-ones vector (s-wave): eigenvalue 1 + iε·(n−1)
#   - any sum-zero vector: eigenvalue 1 − iε
# So s-wave gets a much larger imaginary part than any other Fourier mode.
# ─────────────────────────────────────────────────────────────────────
print("\n─── Eigenstructure of the 7×7 amplitude matrix C ───")
C = np.full((7,7), 1j*EPS); np.fill_diagonal(C, 1.0+0j)
lamC, _ = eig(C)
for l in sorted(lamC, key=lambda z: -z.imag):
    print(f"  λ_C = {l.real:+.6f}{l.imag:+.6f}j   |λ_C|={abs(l):.6f}")
print("  → s-wave has λ_C = 1 + 6iε ≈ 1 + 0.6i  (one eigenvector)")
print("  → all 6 orthogonal modes have λ_C = 1 - iε ≈ 1 - 0.1i  (6-fold)")
print("  ratio of imaginary parts: 6 — that's why m=0 lives in a different band")
