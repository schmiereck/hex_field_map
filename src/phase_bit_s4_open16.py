#!/usr/bin/env python3
"""
Paper 3: gravitational hyperfine structure (Lense-Thirring coupling of the atom to the spin of the central body)
==============================================================================================================

A central body with angular momentum J (along z) has the gravitomagnetic field (round 13)
    b = B_J (z x r) / (r^2 + rc^2)^(3/2),      B_J = 2 c'^2 (kappa/4pi) J
(softened like U).  The walker feels it through the local drag hop (orbit, H1 = b.p) and the internal
rotation rule exp(-i dt (1/2) curl b . J_int) (spin, gyrogravitomagnetic ratio 1).  In the plane of the
2D atom, far outside rc,
    b.p = B_J L_z / rho^3,     (1/2) curl b . S = -(B_J / 2) S_z / rho^3,
i.e. orbit : spin = 2 : -1 as in GR (2G/c^2 r^3 J.L, Omega_LT.S with Omega_LT = -G J/(c^2 r^3) in the
equatorial plane).  First-order shift of the 2p states (large component m_l = 1):
    Delta E ~ B_J <1/rho^3> (m_l - s_z / 2):   2p3/2 : 2p1/2 = 3 : 5
(no spin coupling: 1 : 1; electromagnetic g = 2: 1 : 3).

A.  Reference: radial 2D Dirac in the metric (open15, with the lattice Wilson term) plus the diagonal
    terms m b_phi/rho + s_z (curl b)_z / 2 on each component (small component: m_l = 2, s_z = -1/2 for
    2p3/2; m_l = 0, s_z = +1/2 for 2p1/2): first order and exact in B_J, orbit and spin parts separately.
B.  Lattice: the open15 run (H2, slab 340 x 340 x 2, same start state and pre-filter) with the drag hop
    (brick wall of bond rotations, all registers) and optionally the spin rule after every tick.
    python phase_bit_s4_open16.py B <up|dn> <+1|-1> <rule|norule>
C.  Collect: shifts against the J = 0 runs of open15 and the reference.

Run parts: python phase_bit_s4_open16.py [A] [B spin sign rule] [C]
"""

import functools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import data_path
from phase_bit_s4_open import CP
import phase_bit_s4_open15 as o15

print = functools.partial(print, flush=True)

VARIANT = 'H2'
BJ = float(os.environ.get('BJ', 0.15))       # B_J = 2 c'^2 (kappa/4pi) J
RCBF = float(os.environ.get('RCBF', 0.25))   # softening of b in units of a0 (compact rotating core; U keeps rc = a0/2)
NTICK = int(os.environ.get('NTICK', o15.NTICK))


def b_phi(r, bj, rc):
    """azimuthal component of b = bj (z x r)/(r^2 + rc^2)^(3/2)."""
    return bj * r / (r ** 2 + rc ** 2) ** 1.5


def curl_z(r, bj, rc):
    """(curl b)_z = (1/r) d(r b_phi)/dr."""
    return bj * (2 * rc ** 2 - r ** 2) / (r ** 2 + rc ** 2) ** 2.5


# components of the two 2p states: (m_l, s_z) of the large (P) and small (Q) component
COMP = {1.5: ((1, 0.5), (2, -0.5)), -0.5: ((1, -0.5), (0, 0.5))}


def reference(bj, orbit=True, spin=True, variant=VARIANT, R=None):
    """2p3/2 and 2p1/2 levels with the gravitomagnetic terms (exact in bj) and first-order parts
    (orbit, spin) from the bj = 0 eigenvectors."""
    m, GM, a0, rc = o15.params()
    rcb = RCBF * a0
    from quantum_fcc_3d import LAT_A
    R = o15.NBOX / 2 * LAT_A if R is None else R
    rb = o15.lattice_rb(o15.MU)
    Nfun, Ffun, nexp, _ = o15.VARIANTS[variant]
    U = lambda x: GM / np.sqrt(x ** 2 + rc ** 2)
    Nf, Ff = (lambda x: Nfun(U(x))), (lambda x: Ffun(U(x)))
    Et = m - 0.42 * m * o15.AF ** 2 / 2
    out = {}
    for kap in (1.5, -0.5):
        (mP, sP), (mQ, sQ) = COMP[kap]
        fo = lambda mm: (lambda x: mm * b_phi(x, bj, rcb) / x)
        fs = lambda ss: (lambda x: ss * curl_z(x, bj, rcb) / 2)
        eP = lambda x: (fo(mP)(x) if orbit else 0) + (fs(sP)(x) if spin else 0)
        eQ = lambda x: (fo(mQ)(x) if orbit else 0) + (fs(sQ)(x) if spin else 0)
        w0, v0, gP, gQ = o15.dirac_radial(kap, m, Nf, Ff, R, 8000, Et, k=3, rb=rb, nexp=nexp or 0)
        i0 = np.argmin(np.abs(w0 - Et))
        P, Q = v0[:len(gP), i0], v0[len(gP):, i0]
        first_orb = np.sum(P ** 2 * fo(mP)(gP)) + np.sum(Q ** 2 * fo(mQ)(gQ))
        first_spin = np.sum(P ** 2 * fs(sP)(gP)) + np.sum(Q ** 2 * fs(sQ)(gQ))
        w1, v1, _, _ = o15.dirac_radial(kap, m, Nf, Ff, R, 8000, w0[i0], k=3, rb=rb, nexp=nexp or 0,
                                        extraP=eP, extraQ=eQ)
        i1 = np.argmin(np.abs(w1 - w0[i0]))
        out[kap] = dict(E0=w0[i0], E=w1[i1], orb=first_orb, spin=first_spin, wQ=np.sum(Q ** 2))
    return out


def part_A(out):
    m, GM, a0, rc = o15.params()
    print(f"A. reference: Lense-Thirring shifts of 2p ({VARIANT}, mu {o15.MU}, alpha_f {o15.AF}, a0 {a0:.2f}, rc {rc:.2f}), "
          f"B_J = {BJ}, b softened with rc_b = {RCBF} a0")
    rows = []
    for orbit, spin, lab in ((True, True, 'orbit + spin'), (True, False, 'orbit only')):
        rp, rm = reference(BJ, orbit, spin), reference(-BJ, orbit, spin)
        for kap, nm in ((1.5, '2p3/2'), (-0.5, '2p1/2')):
            a, b_ = rp[kap], rm[kap]
            odd, even = (a['E'] - b_['E']) / 2, (a['E'] + b_['E']) / 2 - a['E0']
            f1 = a['orb'] + (a['spin'] if spin else 0)
            rows.append([int(spin), kap, a['E0'] - m, odd, even, a['orb'], a['spin'], f1, a['wQ']])
            print(f"   {lab:12s} {nm}: E0 - m {a['E0']-m:+.6e}; odd part (E(+J) - E(-J))/2 {odd:+.5e}, even part {even:+.2e}; "
                  f"first order: orbit {a['orb']:+.5e}, spin {a['spin']:+.5e}; small-component weight {a['wQ']:.4f}")
    r = np.array(rows)
    print(f"   ratio 2p3/2 : 2p1/2 (odd parts): orbit + spin {r[0,3]/r[1,3]:.4f} (weak field, rho >> rc: 0.6), "
          f"orbit only {r[2,3]/r[3,3]:.4f} (1)")
    print(f"   spin / orbit (first order, 2p3/2): {r[0,6]/r[0,5]:+.4f} (rho >> rc: -1/4); 2p1/2: {r[1,6]/r[1,5]:+.4f} (+1/4)")
    out['A_rows'] = r
    out['A_par'] = np.array([o15.MU, o15.AF, o15.RCF, o15.NBOX, BJ, RCBF])


# ═════════════════════════════════════════════════════════════════════════════
def drag_ops(shape, X, Y, bj, rc, dt):
    """drag hop on the slab as index lists: per FCC direction s (one of each +-s) and brick-wall parity
    the bond starts ia, ends ib = ia + s (even sublattice only) and cos/sin of the bond angle
    dt (b.s)/(8 l^2) at the bond midpoint.  Parity from the in-plane part of s only, so that the
    two-layer z period does not break the alternation along s (n must be a multiple of 4)."""
    from quantum_fcc_3d import FCC_STEPS, LAT_A
    n, _, nz = shape
    I = np.indices(shape)
    even = I.sum(0) % 2 == 0
    ops = []
    for s in [s for s in FCC_STEPS if tuple(s) > tuple(-s)]:
        sp = s * LAT_A
        mx, my = X + sp[0] / 2, Y + sp[1] / 2
        f = bj / (mx ** 2 + my ** 2 + rc ** 2) ** 1.5
        th = np.repeat((dt * (-f * my * sp[0] + f * mx * sp[1]) / (8 * LAT_A ** 2))[:, :, None], nz, 2)
        q = s[0] ** 2 + s[1] ** 2
        mpar = ((I[0] * s[0] + I[1] * s[1]) // q) % 2
        for par in (0, 1):
            ia = np.flatnonzero((mpar == par) & even)
            x2 = [(I[k].ravel()[ia] + s[k]) % shape[k] for k in range(3)]
            ib = (x2[0] * n + x2[1]) * nz + x2[2]
            t = th.ravel()[ia]
            ops.append((ia, ib, np.cos(t).astype(np.float32)[:, None], np.sin(t).astype(np.float32)[:, None]))
    return ops


def drag_apply(V, ops):
    """in place on a register V (S, 24): A' = c A - s B, B' = c B + s A for every bond of each layer."""
    for ia, ib, c, sn in ops:
        A, B = V[ia], V[ib]
        V[ia] = c * A - sn * B
        V[ib] = c * B + sn * A
    return V


def check_drag(ops, S):
    rng = np.random.default_rng(1)
    v = (rng.normal(size=(S, 24)) + 1j * rng.normal(size=(S, 24))).astype(np.complex64)
    n0 = np.vdot(v, v).real
    w = drag_apply(v, ops)
    return abs(np.vdot(w, w).real / n0 - 1)


DRAGK = int(os.environ.get('DRAGK', 4))   # drag hop every DRAGK ticks with DRAGK x the angle (exact to first order in b)


def part_B(spin, sign, rule):
    from phase_bit_s4_open13 import J_int
    from quantum_fcc_3d import DT_HALF
    sgn = int(sign)
    use_rule = rule == 'rule'
    tag = f'{spin}_{"p" if sgn > 0 else "m"}_{rule}'
    g = o15.setup_B(VARIANT, spin)
    wk, st, Et, m, shape, X, Y, rc, t0 = g['wk'], g['st'], g['Et'], g['m'], g['shape'], g['X'], g['Y'], g['rc'], g['t0']
    bj = sgn * BJ
    rc = RCBF * o15.params()[2]                  # softening of b (compact core)
    ops = drag_ops(shape, X, Y, bj, rc, DRAGK * DT_HALF)
    print(f"   drag hop: B_J = {bj:+.3f}, rc_b = {rc:.2f}, max |b| = {np.max(b_phi(np.hypot(X, Y), BJ, rc)):.2e} (c' = {CP:.3f}), "
          f"every {DRAGK} ticks; norm drift of one drag step on a random state {check_drag(ops, wk.S):.1e}; spin rule: {use_rule}")
    if use_rule:
        jw, jv = np.linalg.eigh(J_int()[2])
        cz = curl_z(np.hypot(X, Y), bj, rc).ravel()
        cz = np.repeat(cz.reshape(shape[0], shape[1], 1), shape[2], 2).ravel()
        phase = np.exp(-1j * DT_HALF * 0.5 * cz[:, None] * jw[None, :]).astype(np.complex64)
        # states are row vectors: a -> a U^T with U = V diag(phase) V^H, i.e. ((a conj(V)) phase) V^T
        jvc, jvt = jv.conj().astype(np.complex64), jv.T.astype(np.complex64)

    def step(st, it):
        st = wk.tick(st)
        out = []
        for a in st:
            if (it + 1) % DRAGK == 0:
                a = drag_apply(a, ops)
            if use_rule:
                a = ((a @ jvc) * phase) @ jvt
            out.append(a)
        return out

    s0 = [a.copy() for a in st]
    sig = np.empty(NTICK + 1, complex)
    for it in range(NTICK + 1):
        sig[it] = sum(np.vdot(a, b) for a, b in zip(s0, st))
        if it < NTICK:
            st = step(st, it)
        if it % 1000 == 0:
            nrm = sum(np.vdot(a, a).real for a in st)
            print(f"      tick {it}  norm {nrm:.6f}  [{time.time()-t0:.0f}s]")
            np.savez_compressed(data_path(f'phase_bit_s4_open16_B_{tag}_partial.npz'), Et=Et - m, sig=sig[:it + 1],
                                par=np.array([o15.MU, o15.AF, o15.RCF, shape[0], it, bj, int(use_rule), RCBF]))
    E, res, wgt = o15.fdm(sig, DT_HALF, o15.fdm_grid(Et, NTICK * DT_HALF))
    best = np.argmax(wgt)
    print(f"   -> level {E[best]-m:+.7e} (J = 0 reference {Et-m:+.7e}), residual {res[best]:.1e}, weight {wgt[best]:.4f}  "
          f"[{time.time()-t0:.0f}s]")
    np.savez_compressed(data_path(f'phase_bit_s4_open16_B_{tag}.npz'), Et=Et - m, sig=sig,
                        par=np.array([o15.MU, o15.AF, o15.RCF, shape[0], NTICK, bj, int(use_rule), RCBF]))
    p = data_path(f'phase_bit_s4_open16_B_{tag}_partial.npz')
    if os.path.exists(p):
        os.remove(p)


# ═════════════════════════════════════════════════════════════════════════════
def level(path):
    from quantum_fcc_3d import DT_HALF
    d = np.load(path)
    m = 2 * o15.MU
    nt = len(d['sig']) - 1
    Et = float(d['Et']) + m
    E, res, wgt = o15.fdm(d['sig'], DT_HALF, o15.fdm_grid(Et, nt * DT_HALF))
    i = np.argmax(wgt)
    alt = [o15.fdm(d['sig'], DT_HALF, o15.fdm_grid(Et, nt * DT_HALF, K, s_)) for K, s_ in ((6, 0.75), (14, 0.5))]
    spread = max(abs(a[0][np.argmax(a[2])] - E[i]) for a in alt)
    return E[i] - m, res[i], spread


def part_C(out):
    print(f"C. Lense-Thirring shifts: lattice vs reference ({VARIANT}, B_J = {BJ})")
    rows = []
    E0 = {s: level(data_path(f'phase_bit_s4_open15_B_{VARIANT}_{s}.npz'))[0] for s in ('up', 'dn')}
    for rule in ('rule', 'norule'):
        rp, rm = reference(BJ, True, rule == 'rule'), reference(-BJ, True, rule == 'rule')
        for spin, kap in (('up', 1.5), ('dn', -0.5)):
            Ls = {}
            for sg in ('p', 'm'):
                pth = data_path(f'phase_bit_s4_open16_B_{spin}_{sg}_{rule}.npz')
                if os.path.exists(pth):
                    Ls[sg] = level(pth)
            if 'p' not in Ls:
                continue
            ref_odd = (rp[kap]['E'] - rm[kap]['E']) / 2
            ref_full = rp[kap]['E'] - rp[kap]['E0']
            if 'm' in Ls:
                lat = (Ls['p'][0] - Ls['m'][0]) / 2
                ev = (Ls['p'][0] + Ls['m'][0]) / 2 - E0[spin]
                kind, ref = 'odd (E(+J) - E(-J))/2', ref_odd
            else:
                lat, ev = Ls['p'][0] - E0[spin], np.nan
                kind, ref = 'E(+J) - E(0)', ref_full
            sp_ = max(v[2] for v in Ls.values())
            rows.append([int(rule == 'rule'), kap, lat, ref, ev, ref_full - ref_odd, sp_, int('m' in Ls)])
            print(f"   {rule:6s} {spin}: {kind}: lattice {lat:+.5e}, reference {ref:+.5e} (ratio {lat/ref:.4f}); "
                  f"even part lattice {ev:+.2e}, reference {ref_full-ref_odd:+.2e}; filter spread {sp_:.1e}")
    r = np.array(rows)
    out['C_rows'] = r
    for rl in (1, 0):
        s = r[r[:, 0] == rl] if len(r) else r
        if len(s) == 2:
            print(f"   {'with' if rl else 'without'} spin rule: ratio 2p3/2 : 2p1/2 lattice {s[0,2]/s[1,2]:.4f}, "
                  f"reference {s[0,3]/s[1,3]:.4f}")


def main():
    args = sys.argv[1:] or ['A']
    if args[0].upper() == 'B':
        part_B(args[1], args[2], args[3])
        return
    path = data_path('phase_bit_s4_open16.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in args:
        {'A': part_A, 'C': part_C}[p.upper()](out)
    np.savez_compressed(path, **out)
    print(f"saved {path}")


if __name__ == '__main__':
    main()
