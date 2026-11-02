#!/usr/bin/env python3
"""
Two-particle Hilbert space: exclusion, exchange statistics, collisions
======================================================================

Step 3 of ROADMAP_QCD_3D_de.md.

Everything so far was a LINEAR one-particle theory: two "particles" were two
wave packets in the same wavefunction, so they could only interfere (see
RESULTS_Turning_2D_de.md, section 8).  Exclusion needs a genuine two-particle
state.

Part A — exchange statistics, exactly and cheaply
-------------------------------------------------
For a non-interacting pair the two-particle state is built from two evolved
one-particle orbitals,

    Psi(x1,x2) = [ a(x1) b(x2) + e^{i theta} a(x2) b(x1) ] / Norm

with theta = 0 for bosons and theta = pi for fermions, and the one-body
evolution acts orbital by orbital.  No large object has to be stored.

With x = (r, s) and the internal index summed, the pair distribution is

    P(r1,r2) = 1/2 [ rho_a(r1) rho_b(r2) + rho_a(r2) rho_b(r1)
                     + 2 Re( e^{-i theta} K(r1) K*(r2) ) ]
    K(r) = sum_s a(r,s) b*(r,s)

At coincidence, P(r,r) = rho_a rho_b + cos(theta) |K|^2.  Cauchy-Schwarz gives
|K|^2 <= rho_a rho_b, so for fermions (theta = pi) the coincidence density is
non-negative and vanishes exactly where the two internal states are parallel:
the Pauli hole.  For bosons it is enhanced: bunching.

IMPORTANT: the exchange term needs the internal states to OVERLAP.  Two
counter-propagating packets of the same energy have nearly orthogonal internal
states (measured: 0.009 on FCC, 3.7e-16 in the 2D model), so they show no
exchange effect at all — a fact of the model, not of the statistics.

Part B — a real collision
-------------------------
Exchange statistics alone is not an interaction.  A genuine collision needs a
term acting when the two particles meet.  That requires the FULL two-particle
wavefunction, which is affordable in 1+1D; see `contact_evolve`.
"""

import numpy as np


# ─── orbital-level observables (no interaction) ──────────────────────────────

def densities(a, b):
    """rho_a, rho_b and the overlap density K(r) = sum_s a(r,s) b*(r,s)."""
    rho_a = (np.abs(a) ** 2).sum(axis=-1)
    rho_b = (np.abs(b) ** 2).sum(axis=-1)
    K = (a * np.conj(b)).sum(axis=-1)
    return rho_a, rho_b, K


def overlap(a, b):
    return complex((np.conj(a) * b).sum())


def coincidence_density(a, b, theta):
    """P(r,r) = rho_a rho_b + cos(theta) |K|^2, unnormalised."""
    rho_a, rho_b, K = densities(a, b)
    return rho_a * rho_b + np.cos(theta) * np.abs(K) ** 2


def pair_norm(a, b, theta):
    """sum over r1,r2 of the unnormalised P: 1 + cos(theta)|<a|b>|^2."""
    S = overlap(a, b)
    return 1.0 + np.cos(theta) * abs(S) ** 2


def project_axis(a, b, axis=0):
    """Marginals along one Cartesian axis: rho~(x) and K~(x)."""
    rho_a, rho_b, K = densities(a, b)
    ax = tuple(i for i in range(rho_a.ndim) if i != axis)
    return rho_a.sum(axis=ax), rho_b.sum(axis=ax), K.sum(axis=ax)


def pair_correlation_axis(a, b, theta, axis=0, normalise=True):
    """
    P(x1, x2) after integrating out the transverse coordinates.  The exchange
    term factorises under that integration, so this is exact.
    """
    ra, rb, K = project_axis(a, b, axis)
    P = 0.5 * (np.outer(ra, rb) + np.outer(rb, ra)
               + 2 * np.real(np.cos(theta) * np.outer(K, np.conj(K))
                             + np.sin(theta) * np.outer(1j * K, np.conj(K))))
    if normalise:
        n = pair_norm(a, b, theta)
        if abs(n) > 1e-14:
            P = P / n
    return P


def pauli_residual(a, theta=np.pi):
    """
    Two fermions in the SAME orbital: the two-particle state must vanish
    identically.  Returns the largest |P(r1,r2)| of the unnormalised pair
    distribution, which is 0 to machine precision for theta = pi.
    """
    return float(np.abs(pair_correlation_axis(a, a, theta, normalise=False)).max())


# ─── Part B: full two-particle state with a contact interaction ──────────────

def symmetrise(psi2, theta):
    """
    Impose the exchange condition on a two-particle amplitude
    psi2[x1, s1, x2, s2]:   Psi(2,1) = e^{-i theta} Psi(1,2).
    """
    sw = np.transpose(psi2, (2, 3, 0, 1))
    return 0.5 * (psi2 + np.exp(-1j * theta) * sw)


def contact_evolve(psi2, step_one, U_contact, n_steps, theta,
                   store_every=1, observe=None):
    """
    Evolve a full two-particle amplitude psi2[x1, s1, x2, s2].

    Each step: the one-body evolution on particle 1, then on particle 2, then
    a contact phase exp(i*U) wherever the two particles sit on the same site.
    The exchange symmetry commutes with all three, so the sector is preserved.

    `step_one(psi, axis_site, axis_int)` must apply one lattice step to the
    given pair of axes.
    """
    out = []
    for t in range(n_steps + 1):
        if observe is not None and (t % store_every == 0 or t == n_steps):
            out.append(observe(psi2, t))
        if t == n_steps:
            break
        psi2 = step_one(psi2, 0, 1)
        psi2 = step_one(psi2, 2, 3)
        if U_contact:
            n = psi2.shape[0]
            idx = np.arange(n)
            psi2[idx, :, idx, :] *= np.exp(1j * U_contact)
    return psi2, out


# ─── a tractable 1+1D arena for genuine collisions ───────────────────────────
#
# The full two-particle amplitude has dimension (N_sites * n_int)^2, which is
# affordable in 1+1D: with N = 201 sites and 2 headings that is 1.6e5 complex
# numbers.  The model is the unitary Feynman checkerboard, i.e. the 1+1D
# version of the turning-phase coin:
#
#     headings d = 0 (move +1), d = 1 (move -1)
#     C = expm(i * eps * sigma_x)        exactly unitary
#
# so the same amplitude rule as everywhere else: straight = 1, turn = i*eps.

from scipy.linalg import expm as _expm

SX = np.array([[0, 1], [1, 0]], dtype=complex)


def coin_1d(eps):
    return _expm(1j * eps * SX)


def step_1d(psi, ax_x, ax_d):
    """One step for the particle whose (site, heading) axes are given."""
    psi = np.moveaxis(psi, (ax_x, ax_d), (0, 1))
    w = np.tensordot(coin_1d.cache, psi, axes=([1], [1]))     # (2, N, ...)
    w = np.moveaxis(w, 0, 1)                                   # (N, 2, ...)
    out = np.zeros_like(w)
    out[1:, 0] = w[:-1, 0]          # d = 0 moves +1
    out[:-1, 1] = w[1:, 1]          # d = 1 moves -1
    return np.moveaxis(out, (0, 1), (ax_x, ax_d))


def set_eps_1d(eps):
    coin_1d.cache = coin_1d(eps)


def packet_1d(N, x0, k0, sigma, eps, branch=0):
    """One-particle Gaussian in the chosen band of the 1+1D coin."""
    x = np.arange(N) - N // 2
    C = coin_1d(eps)
    ph = np.array([np.exp(-1j * k0), np.exp(1j * k0)])
    T = ph[:, None] * C
    lam, V = np.linalg.eig(T)
    E = -np.angle(lam)
    o = np.argsort(E)
    u = V[:, o[branch]]
    u = u / np.linalg.norm(u)
    env = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x)
    psi = env[:, None] * u[None, :]
    return psi / np.sqrt((np.abs(psi) ** 2).sum())


def two_particle_state(a, b, theta):
    """Psi[x1,d1,x2,d2] = (a(1)b(2) + e^{i theta} b(1)a(2)) / norm."""
    P = np.einsum('id,je->idje', a, b) \
        + np.exp(1j * theta) * np.einsum('id,je->idje', b, a)
    return P / np.sqrt((np.abs(P) ** 2).sum())


def exchange_residual(P, theta):
    """How far Psi is from its exchange condition (should be ~0)."""
    sw = np.transpose(P, (2, 3, 0, 1))
    return float(np.abs(P - np.exp(1j * theta) * sw).max())


def joint_position(P):
    """P(x1, x2), internal indices summed."""
    return (np.abs(P) ** 2).sum(axis=(1, 3))


def same_side(P, x_c):
    """Probability that both particles end up on the same side of x_c."""
    J = joint_position(P)
    n = J.shape[0]
    L = slice(0, x_c)
    R = slice(x_c, n)
    return float(J[L, L].sum() + J[R, R].sum()), float(J.sum())


# ─── relative-coordinate reduction: are there two-body bound states? ─────────
#
# A scattering run can only find a bound state if the initial state overlaps
# it.  The definitive test is to diagonalise the two-particle step operator at
# fixed total momentum Q in the relative coordinate r = x1 - x2:
#
#     Psi(x1,d1,x2,d2) = e^{i Q x2} phi(r, d1, d2)
#
# One step sends r -> r + s(d1) - s(d2) and multiplies by e^{i Q s(d2)}, so the
# operator is (4 * N_r) square — small enough to diagonalise exactly.
# A bound state is an eigenvector that decays in |r|.

def relative_step_matrix(N_r, Q, eps, U=0.0, mode="site"):
    """The full one-step operator in the relative coordinate, as a matrix."""
    C = coin_1d(eps)
    s = np.array([+1, -1])
    dim = N_r * 4
    M = np.zeros((dim, dim), dtype=complex)

    def idx(r, d1, d2):
        return (r % N_r) * 4 + d1 * 2 + d2

    for r in range(N_r):
        for d1 in range(2):
            for d2 in range(2):
                rp = r + s[d1] - s[d2]
                amp = np.exp(1j * Q * s[d2])
                for e1 in range(2):
                    for e2 in range(2):
                        M[idx(rp, d1, d2), idx(r, e1, e2)] += \
                            amp * C[d1, e1] * C[d2, e2]
    if U:
        D = np.ones(dim, dtype=complex)
        r0 = 0
        for d1 in range(2):
            for d2 in range(2):
                if mode == "site" or d1 == d2:
                    D[idx(r0, d1, d2)] = np.exp(1j * U)
        M = D[:, None] * M
    return M


def exchange_op(N_r, Q):
    """
    Exchange in the reduced description:  (X phi)(r,d1,d2) = e^{-i Q r} phi(-r,d2,d1).

    NOTE: this carries an explicit e^{-i Q r}, so it is single valued on the ring
    of N_r relative positions only for QUANTISED total momentum Q = 2 pi m / N_r.
    With an arbitrary Q the operator wraps around inconsistently and does not
    commute with the step (verified: [M,X] = 1.5 at Q = 0.5, 9e-15 at Q = 2 pi m/N).
    """
    dim = N_r * 4
    X = np.zeros((dim, dim), dtype=complex)
    idx = lambda r, d1, d2: (r % N_r) * 4 + d1 * 2 + d2
    for r in range(N_r):
        for d1 in range(2):
            for d2 in range(2):
                X[idx(r, d1, d2), idx(-r, d2, d1)] = np.exp(-1j * Q * r)
    return X


def bound_states_by_statistics(N_r, m_Q, eps, U, mode="site"):
    """
    Most localised eigenstate in the boson (X=+1) and fermion (X=-1) sectors,
    at total momentum Q = 2*pi*m_Q/N_r.  Returns (<|r|>_boson, <|r|>_fermion).
    """
    Q = 2 * np.pi * m_Q / N_r
    X = exchange_op(N_r, Q)
    M = relative_step_matrix(N_r, Q, eps, U, mode)
    r = np.abs(((np.arange(N_r) + N_r // 2) % N_r) - N_r // 2)
    out = {}
    for sgn, lab in ((+1, "boson"), (-1, "fermion")):
        Pr = 0.5 * (np.eye(N_r * 4) + sgn * X)
        w, V = np.linalg.eigh(Pr)
        B = V[:, w > 0.5]
        lam, Vs = np.linalg.eig(B.conj().T @ M @ B)
        Vf = B @ Vs
        wgt = (np.abs(Vf) ** 2).reshape(N_r, 4, -1).sum(axis=1)
        wgt = wgt / wgt.sum(axis=0, keepdims=True)
        out[lab] = float((r[:, None] * wgt).sum(axis=0).min())
    return out["boson"], out["fermion"]


# ─── colour: only the singlet survives gauge averaging ───────────────────────

def colour_average_pair(n_samp=200000, N=3, seed=0):
    """
    A quark transported by U and an antiquark by U*.  The Haar average

        <U_ac U*_bd> = delta_ab delta_cd / N

    projects any colour wavefunction chi_cd onto its trace part: chi -> delta
    * tr(chi)/N.  The SINGLET chi = 1/sqrt(N) is preserved exactly (norm 1) and
    every traceless (octet) component is annihilated.  That is why only colour
    singlets can propagate as asymptotic states.
    """
    from quantum_hex_su3 import haar_su3
    rng = np.random.default_rng(seed)
    acc = np.zeros((N, N, N, N), dtype=complex)
    for _ in range(n_samp):
        U = haar_su3(rng, N)
        acc += np.einsum('ac,bd->abcd', U, np.conj(U))
    acc /= n_samp
    exact = np.einsum('ab,cd->abcd', np.eye(N), np.eye(N)) / N
    return acc, exact

    """
    Diagonalise the relative-coordinate step operator and rank the eigenvectors
    by localisation.  Returns (energies, localisation length <|r|>, vectors).
    """
    M = relative_step_matrix(N_r, Q, eps, U, mode)
    lam, V = np.linalg.eig(M)
    E = -np.angle(lam)
    r = (np.arange(N_r) - N_r // 2)
    w = np.abs(V) ** 2
    w = w.reshape(N_r, 4, -1).sum(axis=1)          # (N_r, n_vec)
    w = w / w.sum(axis=0, keepdims=True)
    rmean = (np.abs(np.roll(r, N_r // 2))[:, None] * w).sum(axis=0)
    return E, rmean, V, np.abs(lam)
