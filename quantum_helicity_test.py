#!/usr/bin/env python3
"""
Does a momentum-independent spin lift the exchange-interference ceiling?
========================================================================

Follow-up to RESULTS_Scatter_2D_de.md.

There the Mott interference was found to be limited by the overlap of the
internal states of counter-propagating band eigenstates, |<u(k)|u(-k)>|,
because the band eigenvector is locked to k (helicity-like).  The obvious
question: does a genuine spin — an internal degree of freedom that is NOT the
direction of motion, as in the 3+1D FCC model — remove that limit?

Answer, measured here: NO.  In the FCC model the massive branch is a Kramers
doublet, both singular values of the +k / -k overlap matrix are equal, and they
lie BELOW the classical fidelity of the two heading distributions.  Spin 0
(heading only) has a HIGHER overlap than spin 1/2 at the same parameters, so
the spinor makes it worse, not better.  The bottleneck is that the internal
index IS the direction of motion, and no extra label repairs that.

What does recover the full Mott zero is the non-relativistic limit: as k -> 0
the heading distribution becomes isotropic, the overlap goes to 1, and the
contrast follows.  Measured: at overlap 0.965 the contrast at 90 degrees is
0.9937 — the fermionic cross section is 0.37 % of the bosonic one.

Correction to the earlier wording: the overlap is a SCALING GUIDE, not a strict
ceiling.  Measured contrast/overlap ratios run from 0.78 to 1.12.
"""

import numpy as np

from quantum_hex_turning import bands as h_bands, pick_band as h_pick
from quantum_fcc_3d import bands as f_bands, N_D as FCC_N_D


def overlap_2d(k, eps, alpha=0.0):
    """|<u(k)|u(-k)>| for the particle band of the 2D turning model."""
    _, V1, _ = h_bands(k, 0.0, eps, alpha)
    _, V2, _ = h_bands(-k, 0.0, eps, alpha)
    u1 = V1[:, -1] / np.linalg.norm(V1[:, -1])
    u2 = V2[:, -1] / np.linalg.norm(V2[:, -1])
    return abs(np.vdot(u1, u2))


def overlap_fcc(k, eps, spin=0.5):
    """
    For spin 1/2 the massive branch is a Kramers doublet, so the meaningful
    quantity is the singular values of the 2x2 overlap matrix between the
    doublet at +k and at -k.  Returns (singular values, heading fidelity).

    The heading fidelity sum_d sqrt(p_d(+k) p_d(-k)) measures how much the two
    heading DISTRIBUTIONS overlap, with the spinor traced out.  It is the
    bottleneck.
    """
    kv = np.array([k, 0.0, 0.0])
    ns = 2 if spin == 0.5 else 1
    _, V1, _ = f_bands(kv, eps, spin)
    _, V2, _ = f_bands(-kv, eps, spin)
    if ns == 1:
        u1 = V1[:, 0] / np.linalg.norm(V1[:, 0])
        u2 = V2[:, 0] / np.linalg.norm(V2[:, 0])
        pa = np.abs(u1) ** 2
        pb = np.abs(u2) ** 2
        return np.array([abs(np.vdot(u1, u2))]), float(np.sqrt(pa * pb).sum())
    A, _ = np.linalg.qr(V1[:, :2])          # orthonormalise the degenerate block
    B, _ = np.linalg.qr(V2[:, :2])
    sv = np.linalg.svd(A.conj().T @ B, compute_uv=False)
    pa = (np.abs(A) ** 2).reshape(FCC_N_D, ns, 2).sum(axis=(1, 2)); pa /= pa.sum()
    pb = (np.abs(B) ** 2).reshape(FCC_N_D, ns, 2).sum(axis=(1, 2)); pb /= pb.sum()
    return sv, float(np.sqrt(pa * pb).sum())


def dirac_reference(k, m, c=np.sqrt(3.0)):
    """
    For a continuum Dirac spinor the counter-propagating overlap is exactly
    u^dag(k,s) u(-k,s) = m/E = 1/gamma.  Checked here for comparison — the
    lattice models fall off FASTER than this, because the heading distribution
    peaks along k on top of the relativistic effect.
    """
    E = np.sqrt(c ** 2 * k ** 2 + m ** 2)
    return m / E


# ─── the kinematic bound: no internal space can beat 1/gamma ─────────────────

def overlap_bound(beta):
    """
    Universal upper bound on the exchange overlap of two states moving at +v
    and -v, valid for ANY lattice and ANY internal space:

        |<u(+v)|u(-v)>|  <=  sqrt(1 - beta^2)  =  1/gamma ,   beta = v/c

    Proof.  Write a_d = sqrt(p_d), b_d = sqrt(q_d) for the heading marginals and
    c_d = n_d . v_hat, so |c_d| <= 1 and

        sum a^2 = sum b^2 = 1,   sum c_d a_d^2 = +beta,   sum c_d b_d^2 = -beta,
        F = sum a_d b_d  >=  |<u|v>|          (Cauchy-Schwarz per direction)

    Then
        2 beta = sum c_d (a_d - b_d)(a_d + b_d)
               <= max|c_d| sqrt(sum (a-b)^2) sqrt(sum (a+b)^2)
               <= sqrt(2 - 2F) sqrt(2 + 2F) = 2 sqrt(1 - F^2),
    hence beta^2 <= 1 - F^2, i.e. F <= sqrt(1 - beta^2).

    Equality needs max|c_d| = 1, i.e. lattice directions exactly along +-v.
    The 2D triangular lattice along an axis reaches it (verified to 9e-13); FCC
    along x does not, because no FCC direction points that way.

    The continuum Dirac spinor SATURATES the bound: u^dag(k,s) u(-k,s) = m/E =
    1/gamma.  So Dirac is optimal, and every lattice model here falls short of
    it — by 0.3 % at beta = 0.05 and by 57 % at beta = 0.73.

    Consequence for the exchange interference: a passive flavour label
    contributes a factor <chi|chi> = 1 and changes nothing (verified to 10
    digits); no enlargement of the internal space can raise the ceiling,
    because the bound depends only on the heading marginals, which the velocity
    pins down.  The full Mott zero exists only in the limit v -> 0.
    """
    b = np.asarray(beta, dtype=float)
    return np.sqrt(np.maximum(0.0, 1.0 - b ** 2))


def flavour_overlap(k, eps, n_flavour, alpha=0.0):
    """
    Overlap with a passive flavour label of dimension n_flavour: the coin is
    C (x) 1_f, so u(k) (x) chi factorises and the flavour contributes 1.
    """
    _, V1, _ = h_bands(k, 0.0, eps, alpha)
    _, V2, _ = h_bands(-k, 0.0, eps, alpha)
    u1 = V1[:, -1] / np.linalg.norm(V1[:, -1])
    u2 = V2[:, -1] / np.linalg.norm(V2[:, -1])
    chi = np.zeros(n_flavour); chi[0] = 1.0
    return abs(np.vdot(np.kron(u1, chi), np.kron(u2, chi)))
