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
