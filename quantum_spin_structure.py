"""
Analysis of the internal structure of the 14x14 transfer matrix at k=0.
Goal: identify the spin/angular-momentum content of the 5-fold degenerate
physical eigenvalue at E ~ 2*eps in the 2+1D hexagonal model.

Tasks 1-5 from the user's prompt. Produces figures and a results MD.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, matrix_rank

from quantum_hex_2d import TM14_half, _C, MOVES_PHYS_XY, N_DIRS

EPS = 0.1
TOL = 1e-8


# ─────────────────────────────────────────────────────────────────────────
# Build M_full(k=0) and the rotation representation
# ─────────────────────────────────────────────────────────────────────────

def build_M_full(kx, ky, eps=EPS):
    Mh = TM14_half(kx, ky, eps)
    return Mh @ Mh


# Rotation by 60 deg permutes diagonal dirs cyclically: d -> (d+1) mod 6,
# straight (d=6) is fixed.  Build 7x7 permutation P_60 then 14x14 R_60.
def P60_7():
    P = np.zeros((7, 7))
    for d in range(6):
        P[(d + 1) % 6, d] = 1.0
    P[6, 6] = 1.0
    return P


def R60_14():
    P = P60_7()
    R = np.zeros((14, 14))
    R[:7, :7] = P
    R[7:, 7:] = P
    return R


def reflection_7():
    """Reflection y -> -y: maps direction angle phi -> -phi.
    Angles: d=0:0, 1:60, 2:120, 3:180, 4:240, 5:300 → 0,300,240,180,120,60.
    So d=0 fixed, 1<->5, 2<->4, 3 fixed, 6 fixed."""
    S = np.zeros((7, 7))
    pairs = {0: 0, 1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 6}
    for a, b in pairs.items():
        S[b, a] = 1.0
    return S


def reflection_14():
    S = reflection_7()
    R = np.zeros((14, 14))
    R[:7, :7] = S
    R[7:, 7:] = S
    return R


# ─────────────────────────────────────────────────────────────────────────
# TASK 1: Symmetry verification + irrep decomposition under C_6v
# ─────────────────────────────────────────────────────────────────────────

def task1(M0):
    print("\n=== TASK 1: Symmetry & irrep decomposition ===")
    R60 = R60_14()
    S   = reflection_14()

    # Group elements: C_6v has 12 elements: {E, C6, C3, C2, C3^-1, C6^-1,
    #   sigma_v (3), sigma_d (3)}
    elems = {}
    G = np.eye(14)
    for k in range(6):
        elems[f"C6^{k}"] = np.linalg.matrix_power(R60, k)
    for k in range(6):
        elems[f"sigma{k}"] = np.linalg.matrix_power(R60, k) @ S

    # Verify each commutes with M0
    invariance = {}
    for name, R in elems.items():
        diff = np.linalg.norm(R @ M0 @ R.T - M0)
        invariance[name] = diff
    print("Max ||R M R^T - M||:", max(invariance.values()))
    is_invariant = max(invariance.values()) < 1e-10
    print("M_full is C_6v-invariant:", is_invariant)

    # Characters of the 14-dim rep
    chars14 = {name: np.trace(R).real for name, R in elems.items()}
    print("\nCharacters of the 14-dim rep:")
    for n, c in chars14.items():
        print(f"  {n:10s}  chi = {c:+.3f}")

    # C_6v character table.  Classes: E, 2C6, 2C3, C2, 3sigma_v, 3sigma_d
    # Irreps: A1 A2 B1 B2 E1 E2  (dims 1 1 1 1 2 2)
    classes = {
        'E':       ['C6^0'],
        '2C6':     ['C6^1', 'C6^5'],
        '2C3':     ['C6^2', 'C6^4'],
        'C2':      ['C6^3'],
        '3sv':     ['sigma0', 'sigma2', 'sigma4'],
        '3sd':     ['sigma1', 'sigma3', 'sigma5'],
    }
    class_size = {'E':1,'2C6':2,'2C3':2,'C2':1,'3sv':3,'3sd':3}

    chi_class = {cn: np.mean([chars14[g] for g in glist])
                 for cn, glist in classes.items()}
    print("\nClass-averaged characters of 14-dim rep:")
    for k, v in chi_class.items():
        print(f"  {k:5s}  chi = {v:+.3f}")

    # Standard C_6v character table (rows = irreps)
    irreps = {
        'A1': {'E':1,'2C6': 1,'2C3': 1,'C2': 1,'3sv': 1,'3sd': 1},
        'A2': {'E':1,'2C6': 1,'2C3': 1,'C2': 1,'3sv':-1,'3sd':-1},
        'B1': {'E':1,'2C6':-1,'2C3': 1,'C2':-1,'3sv': 1,'3sd':-1},
        'B2': {'E':1,'2C6':-1,'2C3': 1,'C2':-1,'3sv':-1,'3sd': 1},
        'E1': {'E':2,'2C6': 1,'2C3':-1,'C2':-2,'3sv': 0,'3sd': 0},
        'E2': {'E':2,'2C6':-1,'2C3':-1,'C2': 2,'3sv': 0,'3sd': 0},
    }
    h = 12  # group order
    print("\nIrrep multiplicities in 14-dim rep:")
    mults = {}
    for irr, chi_irr in irreps.items():
        n = sum(class_size[c] * chi_irr[c] * chi_class[c]
                for c in classes) / h
        mults[irr] = n
        print(f"  {irr}: {n:+.3f}")

    return elems, mults, is_invariant


# ─────────────────────────────────────────────────────────────────────────
# TASK 2 & 3: 5-fold eigenspace; angular momenta from R60 eigenvalues
# ─────────────────────────────────────────────────────────────────────────

def task2_3(M0, elems):
    print("\n=== TASK 2+3: 5-fold subspace and angular momentum ===")
    lam, vec = eig(M0)
    E = -np.angle(lam)
    absl = np.abs(lam)
    print("Eigenvalues (E, |lam|):")
    order = np.argsort(E)
    for i in order:
        print(f"  E={E[i]:+.5f}  |lam|={absl[i]:.5f}")

    # Identify the 5-fold band near E = 2*eps = 0.2
    target = 2 * EPS
    mask = np.abs(E - target) < 0.05
    idx5 = np.where(mask)[0]
    print(f"\n#eigvals within 0.05 of 2*eps={target}: {len(idx5)}")
    V5 = vec[:, idx5]                       # 14 x 5
    # Orthonormalize
    Q, _ = np.linalg.qr(V5)

    # Project R60 onto this subspace: R5 = Q^H R60 Q
    R60 = elems['C6^1']
    R5  = Q.conj().T @ R60 @ Q
    # diagonalize R5: eigenvalues should be 6th roots of unity
    mu, w = eig(R5)
    m_vals = np.angle(mu) / (np.pi / 3)   # m such that mu = exp(i m pi/3)
    print("\nEigenvalues of R_60 on the 5-fold subspace:")
    for k in range(5):
        print(f"  mu = {mu[k].real:+.4f} {mu[k].imag:+.4f}j   |mu|={abs(mu[k]):.4f}   m = {m_vals[k]:+.3f}")

    # Round m values
    m_round = np.round(m_vals).astype(int) % 6
    print("Angular-momentum content (mod 6):", sorted(m_round.tolist()))

    # Eigenvectors of M0 in original basis, organized by m
    # rotate Q by w to get angular-momentum eigenstates
    V_ang = Q @ w     # 14 x 5
    print("\nAngular-momentum eigenstates (top 7 components = current step):")
    for k in range(5):
        v = V_ang[:7, k]
        # normalise phase
        if np.max(np.abs(v)) > 0:
            v = v / np.exp(1j * np.angle(v[np.argmax(np.abs(v))]))
        print(f"\n m = {m_vals[k]:+.2f}:  straight comp = {V_ang[6,k]:+.4f}")
        for d in range(7):
            label = ['0°','60°','120°','180°','240°','300°','straight'][d]
            print(f"   d={d} {label:8s}: |a|={abs(V_ang[d,k]):.4f}  arg={np.angle(V_ang[d,k]):+.3f}")

    return E, absl, idx5, V_ang, m_vals


# ─────────────────────────────────────────────────────────────────────────
# TASK 4: k-dependence of the 5 sub-bands
# ─────────────────────────────────────────────────────────────────────────

def task4():
    print("\n=== TASK 4: Splitting of the 5-fold degeneracy for k>0 ===")
    ks = np.linspace(0, 0.5, 60)
    E_target = 2 * EPS
    bands = np.full((len(ks), 5), np.nan)
    for i, k in enumerate(ks):
        M = build_M_full(k, 0.0, EPS)
        lam, _ = eig(M)
        E = -np.angle(lam)
        # pick 5 closest to E_target
        idx = np.argsort(np.abs(E - E_target))[:5]
        bands[i] = np.sort(E[idx])
    return ks, bands


# ─────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────

def fig_eigenvectors(V_ang, m_vals, fname):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4),
                             subplot_kw={'projection': 'polar'})
    angles = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
    for k in range(5):
        v = V_ang[:7, k]
        ax = axes[k]
        for d in range(6):
            mag = abs(v[d])
            ph  = np.angle(v[d])
            ax.plot([angles[d], angles[d]], [0, mag], lw=2,
                    color=plt.cm.hsv((ph + np.pi) / (2*np.pi)))
            ax.plot(angles[d], mag, 'o',
                    color=plt.cm.hsv((ph + np.pi) / (2*np.pi)))
        ax.set_title(f"m = {m_vals[k]:+.2f}\n|straight|={abs(v[6]):.3f}",
                     fontsize=9)
        ax.set_ylim(0, max(abs(v[:6]).max()*1.1, 0.1))
        ax.set_xticks(angles)
        ax.set_xticklabels(['0°','60°','120°','180°','240°','300°'],
                           fontsize=7)
        ax.set_yticklabels([])
    fig.suptitle("5-fold subspace at k=0 — angular-momentum eigenstates",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def fig_splitting(ks, bands, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    for j in range(5):
        ax.plot(ks, bands[:, j], lw=1.6)
    ax.axhline(2 * EPS, color='k', ls=':', lw=0.8, label='E = 2ε')
    ax.set_xlabel("k_x")
    ax.set_ylabel("E (band branches)")
    ax.set_title("Splitting of the 5-fold band along k_y=0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def fig_spectrum(E, absl, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(E, absl, c='C0')
    ax.axvline(2*EPS, color='r', ls='--', lw=0.8, label=f'2ε={2*EPS}')
    ax.axvline(EPS,   color='g', ls='--', lw=0.8, label=f'ε={EPS}')
    ax.set_xlabel("E = -arg(λ)")
    ax.set_ylabel("|λ|")
    ax.set_title("Eigenvalues of M_full at k=0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    M0 = build_M_full(0.0, 0.0, EPS)
    print("rank(M_full):", matrix_rank(M0))

    elems, mults, inv = task1(M0)
    E, absl, idx5, V_ang, m_vals = task2_3(M0, elems)
    ks, bands = task4()

    fig_spectrum(E, absl, "spin_spectrum_k0.png")
    fig_eigenvectors(V_ang, m_vals, "spin_eigenvectors_k0.png")
    fig_splitting(ks, bands, "spin_band_splitting.png")

    # Write summary
    m_sorted = sorted(np.round(m_vals).astype(int).tolist())
    with open("RESULTS_Spin_Structure.md", "w", encoding="utf-8") as f:
        f.write("# Spin Structure of the 5-fold Band (2+1D Hexagonal)\n\n")
        f.write(f"ε = {EPS},  M_full = M_half²,  k = 0\n\n")
        f.write(f"rank(M_full) = {matrix_rank(M0)} / 14\n\n")
        f.write("## Task 1 — C_6v invariance & irrep decomposition\n\n")
        f.write(f"M_full is invariant under C_6v: **{inv}**\n\n")
        f.write("Multiplicities of C_6v irreps in the 14-dim representation:\n\n")
        f.write("| Irrep | dim | multiplicity |\n|---|---|---|\n")
        dims = {'A1':1,'A2':1,'B1':1,'B2':1,'E1':2,'E2':2}
        for irr, n in mults.items():
            f.write(f"| {irr} | {dims[irr]} | {n:+.3f} |\n")
        f.write("\n## Task 2+3 — Angular-momentum content of the 5-fold band\n\n")
        f.write(f"Angular momenta m (R_60 eigenvalues = exp(i m π/3)):\n\n")
        f.write(f"`m = {sorted([f'{x:+.2f}' for x in m_vals])}`\n\n")
        f.write(f"Rounded: **{m_sorted}**\n\n")
        f.write("See `spin_eigenvectors_k0.png` for polar plots of the\n")
        f.write("five angular-momentum eigenstates.\n\n")
        f.write("## Task 4 — Splitting for k>0\n\n")
        f.write("See `spin_band_splitting.png`.\n\n")
        f.write("## Conclusion\n\n")
        # Interpret
        unique_m = sorted(set(m_sorted))
        f.write(f"The 5-fold subspace carries angular momenta {m_sorted} (mod 6).\n\n")
        if set(m_sorted) == {-2,-1,0,1,2} or set([x%6 for x in m_sorted]) == {0,1,2,4,5}:
            verdict = "(A) Spin-2 representation: m ∈ {-2,-1,0,+1,+2}."
        else:
            verdict = "(C/D) Lattice-specific reducible representation — see m-list above."
        f.write(f"**Verdict:** {verdict}\n")
    print("\nWrote RESULTS_Spin_Structure.md and 3 figures.")
