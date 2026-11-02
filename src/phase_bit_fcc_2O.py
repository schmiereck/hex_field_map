#!/usr/bin/env python3
"""
Paper 3, Experiment S3 — Spin 1/2 in 3D from a FINITE phase-bit register
=======================================================================

quantum_fcc_3d.py transports the spinor with the minimal SU(2) rotation that
takes heading d to the 60-deg neighbour d'.  Those rotations generate an
infinite (dense) subgroup of SU(2): the holonomy angles are not quantised.

Here the transport is replaced by an element of the binary octahedral group
2O (48 elements, the single-qubit Clifford group up to phases):

    Q(d -> d') = the lift of the rotation in O that maps d onto d' and lies
                 closest to the minimal rotation.

For every 60-deg pair there are exactly two rotations in O that map d onto d'
(a 90-deg and a 120-deg rotation); the 90-deg one is always closer (fidelity
0.8165 vs 0.5774 with the minimal SU(2) element), so the choice is unique:
**every 60-deg heading turn rotates the spinor by a quarter turn about a cube
axis**, Q = (1 + i n.sigma)/sqrt(2).  sqrt(2)*Q has entries in Z[i].

Tested
------
1.  Construction: Q(d'->d) = Q(d->d')^-1, inversion Q(-d->-d') = Q(d->d'),
    the Q generate all of 2O.
2.  Finite register: the label (heading d, accumulated g in 2O) reaches only
    48 of 12*48 = 576 combinations — 4 per heading, {+-1, +-i n_d.sigma},
    a Z4 "phase bit" per heading, exactly as in 2D (S1: 12 of 24).
3.  Holonomy of closed walks is quantised to {+1, -1, +-i n.sigma}
    (rotation about the own heading by 0, 2pi, +-pi), versus a continuum of
    angles for the minimal SU(2) transport.  Belt trick: where does -1 appear?
4.  Integer automaton N[x, d, z4, T] (path counts) read out with the irreps of
    2O reproduces the float walk exactly; the double-valued irreps (E1/2,
    E5/2, G3/2) are fermions, the single-valued ones bosons — one table.
5.  Spectrum of the unitary coin expm(i eps G), G = sum |d'><d| (x) Q:
    Kramers doubling at every k, rest masses, causality, isotropy; compared
    with the continuous SU(2) walker.
"""

import itertools
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import (FCC_UNIT, FCC_STEPS, ADJ60, N_D, MOVES_PHYS, DT_HALF,
                            C_LIGHT, su2_transport, qmul, quat_transport)
from paths import data_path

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# ─── the octahedral group and its double cover ───────────────────────────────

def rotations_O():
    out = []
    for perm in itertools.permutations(range(3)):
        for s in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = s[i]
            if np.isclose(np.linalg.det(M), 1):
                out.append(M)
    return out


def axis_angle(R):
    ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if ang < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(ang - np.pi) < 1e-9:
        w, v = np.linalg.eigh((R + np.eye(3)) / 2)
        return v[:, -1], ang
    n = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return n / (2 * math.sin(ang)), ang


def su2_min(u, v):
    """Minimal SU(2) transport u -> v, U(R) = exp(-i th n.s/2), i.e.
    U sigma.w U^dag = sigma.(R w).  Identical to quantum_fcc_3d.su2_transport since
    its sign fix; the old (anti-homomorphic) convention is kept as 'legacy' below."""
    return su2_transport(u, v)


def su2_from_axis(n, ang):
    return math.cos(ang / 2) * I2 + 1j * math.sin(ang / 2) * (n[0] * SX + n[1] * SY + n[2] * SZ)


def rot_from_su2(Q):
    """SO(3) image of an SU(2) matrix: R_ij = tr(s_i Q s_j Q^dag)/2."""
    s = (SX, SY, SZ)
    return np.array([[0.5 * np.real(np.trace(s[i] @ Q @ s[j] @ Q.conj().T))
                      for j in range(3)] for i in range(3)])


def build_transport():
    """Q[d][d'] for 60-deg pairs, plus diagnostics of the choice."""
    O = rotations_O()
    Q = {}
    fid = []
    for d in range(N_D):
        for dp in range(N_D):
            if not ADJ60[d, dp]:
                continue
            Qm = su2_min(FCC_UNIT[d], FCC_UNIT[dp])
            cands = []
            for R in O:
                if np.allclose(R @ FCC_UNIT[d], FCC_UNIT[dp]):
                    n, a = axis_angle(R)
                    for sgn in (1, -1):
                        Qc = sgn * su2_from_axis(n, -a)      # U(R) = exp(-i a n.s/2)
                        f = np.real(np.trace(Qm.conj().T @ Qc)) / 2
                        cands.append((f, np.degrees(a), Qc))
            cands.sort(key=lambda c: -c[0])
            assert cands[0][0] > cands[1][0] + 1e-6, "ambiguous choice"
            Q[(d, dp)] = cands[0][2]
            fid.append((cands[0][0], cands[0][1], cands[1][0], cands[1][1]))
    return Q, fid


def closure(gens):
    """All products of the generators (finite group), as a list of matrices."""
    elems = [I2.copy()]
    frontier = [I2.copy()]
    while frontier:
        new = []
        for a in frontier:
            for g in gens:
                b = g @ a
                if not any(np.allclose(b, e) for e in elems):
                    elems.append(b)
                    new.append(b)
        frontier = new
    return elems


def index_of(M, elems):
    for i, e in enumerate(elems):
        if np.allclose(M, e, atol=1e-9):
            return i
    raise KeyError


# ─── irreps of 2O (used as readouts) ─────────────────────────────────────────

def chi_A2(R):
    """A2 of O: +1 on E, C3, C2(axis); -1 on C4 and C2' (face-diagonal axes)."""
    n, a = axis_angle(R)
    deg = round(np.degrees(a))
    if deg in (0, 120):
        return 1.0
    if deg == 90:
        return -1.0
    # 180 deg: about a cube axis (+1) or a face diagonal (-1)
    return 1.0 if np.isclose(np.max(np.abs(n)), 1.0) else -1.0


def irrep_E_O(R):
    """2-dim irrep E of O via its action on {x^2 - y^2, 2z^2 - x^2 - y^2}."""
    # E is the quotient O -> S3 acting on the three axes; use permutation part
    P = np.abs(R)                        # permutation matrix of the axes
    w = np.exp(2j * np.pi / 3)
    # basis f1 = x2 + w y2 + w^2 z2, f2 = conj
    v1 = np.array([1, w, w * w])
    v2 = np.conj(v1)
    B = np.vstack([v1, v2]).T / np.sqrt(3)            # 3x2
    M = B.conj().T @ P @ B
    return M


def readout_matrix(g, irrep):
    """Matrix of the 2O element g (2x2 SU(2)) in the requested irrep."""
    R = rot_from_su2(g)
    if irrep == 'A1':
        return np.eye(1, dtype=complex)
    if irrep == 'A2':
        return np.array([[chi_A2(R)]], dtype=complex)
    if irrep == 'E1/2':
        return g
    if irrep == 'E5/2':
        return chi_A2(R) * g
    if irrep == 'G3/2':
        return np.kron(g, irrep_E_O(R))
    if irrep == 'T1':
        return R.astype(complex)
    raise ValueError(irrep)


DOUBLE_VALUED = {'E1/2': True, 'E5/2': True, 'G3/2': True,
                 'A1': False, 'A2': False, 'T1': False}


# ─── coin and bands ──────────────────────────────────────────────────────────

def generator(Q, irrep='E1/2', minimal=False):
    """G = sum_{d'~d} |d'><d| (x) rho(Q(d->d'))."""
    dim = readout_matrix(I2, irrep).shape[0] if not minimal else 2
    G = np.zeros((N_D * dim, N_D * dim), dtype=complex)
    for (d, dp), q in Q.items():
        if minimal:
            if minimal == 'legacy':           # old quantum_fcc_3d convention
                blk = su2_transport(FCC_UNIT[d], FCC_UNIT[dp]).conj().T
            else:
                blk = su2_min(FCC_UNIT[d], FCC_UNIT[dp])
        else:
            blk = readout_matrix(q, irrep)
        G[dp * dim:(dp + 1) * dim, d * dim:(d + 1) * dim] = blk
    return G, dim


def bands_k(k, C, dim):
    ph = np.exp(-1j * (MOVES_PHYS @ np.asarray(k, float)))
    T = np.repeat(ph, dim)[:, None] * C
    lam, V = np.linalg.eig(T)
    E = -np.angle(lam) / DT_HALF
    o = np.argsort(E)
    return E[o], V[:, o]


def group_velocity(u, dim):
    w = (np.abs(u) ** 2).reshape(N_D, dim).sum(axis=1)
    return (w / w.sum()) @ MOVES_PHYS / DT_HALF


# ─── holonomy of closed walks ────────────────────────────────────────────────

def holonomies(Q, L_max, with_straight, d0=0, minimal=False):
    """Classify closed walks (back to the same directed edge) by holonomy.
    Returns {L: {class: count}} with class = rounded rotation angle about the
    initial heading in units of 2*pi, taken from the SU(2) element (so -1 = 1.0)."""
    if minimal:
        QT = {k: su2_min(FCC_UNIT[k[0]], FCC_UNIT[k[1]]) for k in Q}
    else:
        QT = Q
    nbrs = [[dp for dp in range(N_D) if ADJ60[d, dp]] + ([d] if with_straight else [])
            for d in range(N_D)]
    u0 = FCC_UNIT[d0]
    res = {L: {} for L in range(1, L_max + 1)}

    def rec(pos, d, g, depth):
        if depth > 0 and pos == (0, 0, 0) and d == d0:
            # g = cos(phi/2) + i sin(phi/2) u0.sigma  (rotation about u0)
            c = np.real(np.trace(g)) / 2
            s = np.real(np.trace(g @ (u0[0] * SX + u0[1] * SY + u0[2] * SZ))) / (2j)
            s = np.real(np.trace((u0[0] * SX + u0[1] * SY + u0[2] * SZ) @ g)) / 2
            phi = 2 * math.atan2(-np.imag(np.trace((u0[0] * SX + u0[1] * SY
                                                    + u0[2] * SZ) @ g)) / 2, c)
            key = round(phi / (2 * np.pi) % 2.0, 6)
            res[depth][key] = res[depth].get(key, 0) + 1
        if depth == L_max:
            return
        if math.sqrt(sum(p * p for p in pos)) / math.sqrt(2.0) > (L_max - depth) + 1e-9:
            return
        for nd in nbrs[d]:
            q = I2 if nd == d else QT[(d, nd)]
            rec(tuple(p + s for p, s in zip(pos, FCC_STEPS[nd])), nd, q @ g, depth + 1)

    rec((0, 0, 0), d0, I2, 0)
    return res


# ─── integer automaton on (x, d, z4, T) ──────────────────────────────────────

def int_automaton(Q, elems, t_max):
    """Path counts N[x, y, z, d, label, T] with label indexing the 4 reachable
    group elements per heading.  Straight = weight 1, 60-deg turn = weight (i eps) Q."""
    # reachable elements per heading (coset of the start heading d0 = 0)
    reach = {d: [] for d in range(N_D)}
    for i, g in enumerate(elems):
        R = rot_from_su2(g)
        for d in range(N_D):
            if np.allclose(R @ FCC_UNIT[0], FCC_UNIT[d]):
                reach[d].append(i)
    lab = {(d, gi): k for d in range(N_D) for k, gi in enumerate(reach[d])}
    # transition table: (d, k) --turn to d'--> (d', k')
    trans = {}
    for (d, dp), q in Q.items():
        for k, gi in enumerate(reach[d]):
            gj = index_of(q @ elems[gi], elems)
            trans[(d, k, dp)] = lab[(dp, gj)]
    n = 2 * t_max + 3
    c = n // 2
    N = np.zeros((n, n, n, N_D, 4, t_max + 1), dtype=np.int64)
    N[c, c, c, 0, lab[(0, index_of(I2, elems))], 0] = 1
    out_hist = []
    for t in range(1, t_max + 1):
        new = np.zeros_like(N)
        for d in range(N_D):
            # straight
            new[..., d, :, :] += np.roll(N[..., d, :, :], tuple(FCC_STEPS[d]), axis=(0, 1, 2))
            for dp in range(N_D):
                if not ADJ60[d, dp]:
                    continue
                src = N[..., d, :, :]
                moved = np.zeros_like(src)
                moved[..., 1:] = src[..., :-1]                 # T -> T + 1
                for k in range(4):
                    new[..., dp, trans[(d, k, dp)], :] += np.roll(
                        moved[..., k, :], tuple(FCC_STEPS[dp]), axis=(0, 1, 2))
        N = new
    return N, reach, c


def readout_int(N, reach, elems, eps, irrep, psi0):
    """A[x, d, :] = sum_{k,T} N * (i eps)^T * rho(g_k) psi0."""
    T = N.shape[-1]
    wT = (1j * eps) ** np.arange(T)
    dim = len(psi0)
    out = np.zeros(N.shape[:3] + (N_D, dim), dtype=complex)
    for d in range(N_D):
        for k, gi in enumerate(reach[d]):
            amp = N[..., d, k, :].astype(float) @ wT
            out[..., d, :] += amp[..., None] * (readout_matrix(elems[gi], irrep) @ psi0)[None, None, None, :]
    return out


def float_graded(Q, irrep, eps, t_max, psi0):
    """Float walk with W[d',d] = delta + i eps ADJ rho(Q): same rule, no integers."""
    dim = len(psi0)
    n = 2 * t_max + 3
    c = n // 2
    psi = np.zeros((n, n, n, N_D, dim), dtype=complex)
    psi[c, c, c, 0] = psi0
    blocks = {k: readout_matrix(q, irrep) for k, q in Q.items()}
    for t in range(t_max):
        new = np.zeros_like(psi)
        for d in range(N_D):
            new[..., d, :] += np.roll(psi[..., d, :], tuple(FCC_STEPS[d]), axis=(0, 1, 2))
            for dp in range(N_D):
                if ADJ60[d, dp]:
                    contrib = 1j * eps * (psi[..., d, :] @ blocks[(d, dp)].T)
                    new[..., dp, :] += np.roll(contrib, tuple(FCC_STEPS[dp]), axis=(0, 1, 2))
        psi = new
    return psi


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    out = {}
    Q, fid = build_transport()
    f = np.array(fid)
    print(f"1. transport: {len(Q)} oriented 60-deg pairs; chosen rotation angle "
          f"{set(np.round(f[:,1]).astype(int))} deg, fidelity with minimal SU(2) "
          f"{f[:,0].min():.4f} (runner-up {set(np.round(f[:,3]).astype(int))} deg, "
          f"{f[:,2].max():.4f})")
    herm = max(np.abs(Q[(dp, d)] - Q[(d, dp)].conj().T).max() for (d, dp) in Q)
    anti = {d: int(np.argmin(np.linalg.norm(FCC_UNIT + FCC_UNIT[d], axis=1))) for d in range(N_D)}
    inv = max(np.abs(Q[(anti[d], anti[dp])] - Q[(d, dp)]).max() for (d, dp) in Q)
    zi = max(np.abs(np.sqrt(2) * q - np.round(np.sqrt(2) * q.real) - 1j * np.round(np.sqrt(2) * q.imag)).max()
             for q in Q.values())
    elems = closure(list(Q.values()))
    print(f"   Q(d'->d) = Q(d->d')^dag: {herm:.1e};  inversion Q(-d->-d') = Q(d->d'): {inv:.1e};  "
          f"sqrt(2)*Q in Z[i]: {zi:.1e};  generated group order: {len(elems)}")
    out['group_order'] = len(elems)

    # 2./4. integer automaton
    T_MAX = 8
    N, reach, c = int_automaton(Q, elems, T_MAX)
    occ = (N.sum(axis=(0, 1, 2, 5)) > 0)
    print(f"2. reachable (heading, register) labels: {occ.sum()} of {N_D*len(elems)} "
          f"({occ.sum()//N_D} per heading)  [{time.time()-t0:.0f}s]")
    g_d0 = [elems[i] for i in reach[0]]
    print("   register at the start heading: " + ", ".join(
        f"{np.round(np.real(np.trace(g))/2,3)}+i({np.round(np.imag(np.trace((FCC_UNIT[0][0]*SX+FCC_UNIT[0][1]*SY+FCC_UNIT[0][2]*SZ)@g))/2,3)})n.s"
        for g in g_d0))
    out['reach_count'] = int(occ.sum())
    devs = {}
    for irrep in ('E1/2', 'E5/2', 'G3/2', 'A1', 'A2', 'T1'):
        dim = readout_matrix(I2, irrep).shape[0]
        psi0 = np.zeros(dim, dtype=complex); psi0[0] = 1
        for eps in (0.2, 0.6):
            A = readout_int(N, reach, elems, eps, irrep, psi0)
            B = float_graded(Q, irrep, eps, T_MAX, psi0)
            devs[(irrep, eps)] = np.abs(A - B).max() / np.abs(B).max()
    print(f"4. integer table read out with irreps vs float walk (t={T_MAX}): max rel. deviation "
          f"{max(devs.values()):.1e}  [{time.time()-t0:.0f}s]")
    out['int_dev'] = max(devs.values())
    # closed walks at the origin by register value
    ret = N[c, c, c, 0].sum(axis=-1)          # (4,) over register labels
    out['return_by_label'] = ret

    # 3. holonomy classes
    for ws in (False, True):
        L = 8 if not ws else 7
        h2 = holonomies(Q, L, ws)
        hm = holonomies(Q, L, ws, minimal=True)
        tag = 'with straight' if ws else 'only 60-deg turns'
        print(f"3. holonomy of closed walks ({tag}, L<={L}):")
        for l in range(1, L + 1):
            if h2[l] or hm[l]:
                print(f"   L={l}: 2O {dict(sorted(h2[l].items()))}   "
                      f"| minimal SU(2): {len(hm[l])} distinct angles, "
                      f"-1: {hm[l].get(1.0, 0)}, +1: {hm[l].get(0.0, 0)}")
        out[f'hol2O_{int(ws)}'] = np.array([[l, k, v] for l in h2 for k, v in h2[l].items()])
        out[f'holmin_{int(ws)}'] = np.array([[l, k, v] for l in hm for k, v in hm[l].items()])
    print(f"   [{time.time()-t0:.0f}s]")

    # 5. spectrum of the unitary coin
    eps = 0.1
    rng = np.random.default_rng(3)
    kr = rng.normal(size=(40, 3)); kr *= (rng.uniform(0.05, 2.5, 40) / np.linalg.norm(kr, axis=1))[:, None]
    for irrep, minimal in (('E1/2', False), ('E5/2', False), ('G3/2', False),
                           ('A1', False), ('T1', False), ('E1/2', True),
                           ('E1/2', 'legacy')):
        G, dim = generator(Q, irrep, minimal)
        C = expm(1j * eps * G)
        unit = np.abs(C.conj().T @ C - np.eye(len(C))).max()
        lam = np.sort(np.linalg.eigvalsh(G))
        levels, mult = np.unique(np.round(-eps * lam / DT_HALF, 8), return_counts=True)
        kram = 0.0
        vmax = 0.0
        for k in kr:
            E, V = bands_k(k, C, dim)
            if DOUBLE_VALUED[irrep]:
                kram = max(kram, np.abs(E[1::2] - E[0::2]).max())
            vmax = max(vmax, max(np.linalg.norm(group_velocity(V[:, j], dim)) for j in range(len(E))))
        name = irrep + {False: ' (2O)', True: ' (minimal SU(2))',
                        'legacy': ' (old quantum_fcc_3d convention)'}[minimal]
        print(f"5. {name}: unitarity {unit:.1e}; rest levels E/eps (mult): "
              + ", ".join(f"{lv/eps:+.4f}({m})" for lv, m in zip(levels, mult))
              + (f"; Kramers splitting over 40 random k: {kram:.1e}" if DOUBLE_VALUED[irrep] else "")
              + f"; max|v_g| = {vmax:.4f} (c = {C_LIGHT:.4f})")
        out[f'rest_{irrep}_{minimal}'] = np.column_stack([levels / eps, mult])
    print(f"   [{time.time()-t0:.0f}s]")

    # 5b. top band dispersion and isotropy: 2O vs minimal SU(2), spin 1/2
    dirs = {'[100]': [1, 0, 0], '[110]': [1, 1, 0], '[111]': [1, 1, 1]}
    ks = np.linspace(0, 1.5, 31)
    for minimal in (False, True):
        G, dim = generator(Q, 'E1/2', minimal)
        C = expm(1j * eps * G)
        for name, v in dirs.items():
            v = np.array(v, float) / np.linalg.norm(v)
            E = np.array([bands_k(k * v, C, dim)[0] for k in ks])
            out[f'disp_{int(minimal)}_{name}'] = E
    out['disp_k'] = ks
    for minimal in (False, True):
        top = {n: out[f'disp_{int(minimal)}_{n}'][:, -1] for n in dirs}
        aniso = [np.ptp([top[n][i] for n in dirs]) / (top['[100]'][i] - top['[100]'][0] + 1e-30)
                 for i in (6, 12, 20)]
        m0 = top['[100]'][0]
        curv = (top['[100]'][1] - m0) / ks[1] ** 2 * 2
        print(f"5b. {'minimal SU(2)' if minimal else '2O'} top band: m0 = {m0:.5f}, "
              f"1/m* = {curv:.4f} (c^2/m0 = {C_LIGHT**2/m0:.4f}); anisotropy of E-m0 across "
              f"[100]/[110]/[111] at k = {ks[6]:.2f}, {ks[12]:.2f}, {ks[20]:.2f}: "
              + ", ".join(f"{a*100:.1f}%" for a in aniso))
    out['eps'] = eps

    np.savez(data_path('phase_bit_fcc_2O.npz'), **out)
    print(f"saved data/phase_bit_fcc_2O.npz  [{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()
