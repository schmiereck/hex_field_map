#!/usr/bin/env python3
"""
SU(3) lattice gauge theory with the Wilson action on the triangular lattice
============================================================================

Step 1 of ROADMAP_QCD_3D_de.md: give the gauge field its own dynamics.

`quantum_hex_su3.py` showed that a STATIC random background obeys a perimeter
law and therefore does not confine.  Confinement requires the links to be
correlated by the Wilson action

    weight  ~  exp( beta * sum_plaquettes (1/N) Re tr U_P )

Lattice
-------
Abstract triangular lattice, integer sites (i, j), three positive link
directions e0 = (1,0), e1 = (0,1), e2 = (-1,1) with e0 + e2 = e1.
Two elementary triangles per site:

    T_A(x):  x -(+0)-> x+e0 -(+2)-> x+e1 -(-1)-> x     W_A = U1(x)^+ U2(x+e0) U0(x)
    T_B(x):  x -(+2)-> x+e2 -(+0)-> x+e1 -(-1)-> x     W_B = U1(x)^+ U0(x+e2) U2(x)

Every triangle contains exactly ONE link of each direction, so links of the
same direction never share a plaquette.  That is a perfect 3-colouring: all
links of one direction can be updated simultaneously, and the whole Metropolis
sweep is vectorised over the lattice.

What to expect
--------------
In two dimensions pure lattice gauge theory factorises: the plaquettes are
independent variables and the Wilson loop obeys an EXACT area law

    <W(C)>  =  w1 ^ (number of triangles enclosed),   w1 = <(1/N) Re tr U_plaq>

This is used here as a validation of the Monte Carlo, not as a discovery:
confinement in 2D is kinematic.  The point of step 1 is a correct, tested
dynamical gauge field that can be carried over to 3+1D, where the area law is
no longer trivial.
"""

import numpy as np
from scipy.linalg import expm

N_C = 3
E = np.array([[1, 0], [0, 1], [-1, 1]])          # e0, e1, e2


# ─── helpers ─────────────────────────────────────────────────────────────────

def sh(A, d):
    """A evaluated at x + d, for a field indexed [i, j, ...]."""
    if d[0] == 0 and d[1] == 0:
        return A
    return np.roll(A, shift=(-int(d[0]), -int(d[1])), axis=(0, 1))


def dag(A):
    return A.conj().swapaxes(-1, -2)


def rtr(A):
    return np.trace(A, axis1=-2, axis2=-1).real


# ─── SU(3) generation ────────────────────────────────────────────────────────

def haar_su3(rng, n=N_C):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q * (np.diag(r) / np.abs(np.diag(r)))[None, :]
    return q / np.linalg.det(q) ** (1.0 / n)


def update_pool(eps, n_pool, rng, n=N_C):
    """
    A pool of SU(n) matrices near the identity, closed under inversion so the
    Metropolis proposal is symmetric.  Returns an array of shape (2*n_pool,n,n).
    """
    out = []
    for _ in range(n_pool):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        H = (a + a.conj().T) / 2.0
        H -= np.trace(H) / n * np.eye(n)
        H /= np.sqrt((np.abs(H) ** 2).sum()) + 1e-30
        X = expm(1j * eps * H)
        out.append(X)
        out.append(X.conj().T)
    return np.array(out)


def init_links(L, rng, hot=True, n=N_C):
    U = np.empty((L, L, 3, n, n), dtype=complex)
    for i in range(L):
        for j in range(L):
            for mu in range(3):
                U[i, j, mu] = haar_su3(rng, n) if hot else np.eye(n)
    return U


# ─── plaquettes, staples, action ─────────────────────────────────────────────

def plaquettes(U):
    """The two elementary triangles at every site: (W_A, W_B)."""
    U0, U1, U2 = U[:, :, 0], U[:, :, 1], U[:, :, 2]
    WA = dag(U1) @ sh(U2, E[0]) @ U0
    WB = dag(U1) @ sh(U0, E[2]) @ U2
    return WA, WB


def mean_plaquette(U):
    WA, WB = plaquettes(U)
    return float((rtr(WA).mean() + rtr(WB).mean()) / (2 * N_C))


def action(U, beta):
    WA, WB = plaquettes(U)
    return beta / N_C * float(rtr(WA).sum() + rtr(WB).sum())


def staple(U, mu):
    """Sigma such that the two plaquettes containing U[x,mu] give
    Re tr( U[x,mu] Sigma[x] ) as their combined contribution."""
    U0, U1, U2 = U[:, :, 0], U[:, :, 1], U[:, :, 2]
    if mu == 0:
        return (dag(U1) @ sh(U2, E[0])
                + sh(U2, -E[2]) @ dag(sh(U1, -E[2])))
    if mu == 1:
        return (dag(U0) @ dag(sh(U2, E[0]))
                + dag(U2) @ dag(sh(U0, E[2])))
    return (dag(U1) @ sh(U0, E[2])
            + sh(U0, -E[0]) @ dag(sh(U1, -E[0])))


# ─── Metropolis ──────────────────────────────────────────────────────────────

def sweep(U, beta, pool, rng, n_hit=4):
    """One full sweep: all three link directions, n_hit proposals each."""
    L = U.shape[0]
    acc = 0
    tot = 0
    for mu in range(3):
        S = staple(U, mu)
        for _ in range(n_hit):
            M = U[:, :, mu] @ S
            old = rtr(M)
            X = pool[rng.integers(len(pool), size=(L, L))]
            new = rtr(X @ M)
            dS = beta / N_C * (new - old)
            take = (dS >= 0) | (rng.random((L, L)) < np.exp(np.minimum(dS, 0.0)))
            U[:, :, mu] = np.where(take[:, :, None, None],
                                   X @ U[:, :, mu], U[:, :, mu])
            acc += int(take.sum()); tot += L * L
    return acc / tot


# ─── Wilson loops ────────────────────────────────────────────────────────────

def wilson_rhombus(U, a, b):
    """
    Translation-averaged (1/N) Re tr W for the closed loop
    a steps along e0, b along e1, a back, b back.
    Encloses a*b unit rhombi = 2*a*b elementary triangles.
    """
    L = U.shape[0]
    W = np.broadcast_to(np.eye(N_C, dtype=complex), (L, L, N_C, N_C)).copy()
    d = np.zeros(2, dtype=int)
    for _ in range(a):
        W = sh(U[:, :, 0], d) @ W
        d = d + E[0]
    for _ in range(b):
        W = sh(U[:, :, 1], d) @ W
        d = d + E[1]
    for _ in range(a):
        d = d - E[0]
        W = dag(sh(U[:, :, 0], d)) @ W
    for _ in range(b):
        d = d - E[1]
        W = dag(sh(U[:, :, 1], d)) @ W
    return rtr(W) / N_C            # (L, L) array, one value per start site


# ─── exact 2D reference: the single-plaquette integral ───────────────────────

def single_plaquette(beta, n_therm=40000, n_meas=800000, eps=None, seed=0):
    """
    In 2D the plaquettes are independent, so the exact reference for
    w1 = <(1/N) Re tr U_P> is a ONE-matrix problem:

        P(W) ~ exp( beta * (1/N) Re tr W ) dW_Haar

    Sampled here by Metropolis on a single SU(3) matrix.  This is the number
    the full lattice simulation must reproduce.
    """
    rng = np.random.default_rng(seed)
    if eps is None:
        eps = float(np.clip(3.0 / np.sqrt(max(beta, 0.2)), 0.2, 4.0))
    pool = update_pool(eps, 120, rng)
    W = haar_su3(rng)
    f = lambda M: float(np.trace(M).real) / N_C
    acc = 0
    vals = []
    for t in range(n_therm + n_meas):
        X = pool[rng.integers(len(pool))]
        Wn = X @ W
        dS = beta * (f(Wn) - f(W))
        if dS >= 0 or rng.random() < np.exp(dS):
            W = Wn
            acc += 1
        if t >= n_therm:
            vals.append(f(W))
    return float(np.mean(vals)), binned_error(vals), acc / (n_therm + n_meas)


# ─── driver ──────────────────────────────────────────────────────────────────

def run(beta, L=24, n_therm=600, n_meas=800, n_sep=4, n_hit=4, eps=None,
        loops=((1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4)),
        seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    if eps is None:                      # start from a scale that suits beta
        eps = float(np.clip(2.4 / np.sqrt(max(beta, 0.2)), 0.15, 3.0))
    pool = update_pool(eps, 60, rng)
    U = init_links(L, rng, hot=True)

    therm = []
    for t in range(n_therm):
        a = sweep(U, beta, pool, rng, n_hit)
        if t < n_therm // 2 and t % 20 == 19:      # retune towards 50% acceptance
            eps = float(np.clip(eps * (1.6 if a > 0.55 else 0.7 if a < 0.45 else 1.0),
                                0.02, 4.0))
            pool = update_pool(eps, 60, rng)
        if t % 10 == 0:
            therm.append(mean_plaquette(U))

    plaq, wl, accs = [], {lp: [] for lp in loops}, []
    for t in range(n_meas):
        for _ in range(n_sep):
            accs.append(sweep(U, beta, pool, rng, n_hit))
        plaq.append(mean_plaquette(U))
        for lp in loops:
            wl[lp].append(float(wilson_rhombus(U, *lp).mean()))
    return dict(beta=beta, L=L, eps=eps, therm=np.array(therm),
                plaq=np.array(plaq),
                wl={k: np.array(v) for k, v in wl.items()},
                acc=float(np.mean(accs)), U=U)


def binned_error(x, n_bin=20):
    x = np.asarray(x)
    m = len(x) // n_bin * n_bin
    if m == 0:
        return float(np.std(x) / max(np.sqrt(len(x)), 1))
    b = x[:m].reshape(n_bin, -1).mean(axis=1)
    return float(np.std(b, ddof=1) / np.sqrt(n_bin))


def haar_reweight(beta, n=400000, seed=1):
    """
    Independent small-beta reference: importance sampling of the single
    plaquette from the Haar measure.  Unbiased, but inefficient for large beta
    (the weight concentrates where Haar rarely samples), so it is only used as
    a cross-check below beta ~ 3.
    """
    rng = np.random.default_rng(seed)
    f = np.array([np.trace(haar_su3(rng)).real / N_C for _ in range(n)])
    w = np.exp(beta * f)
    return float((f * w).sum() / w.sum())
