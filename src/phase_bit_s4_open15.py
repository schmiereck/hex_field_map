#!/usr/bin/env python3
"""
Paper 3: gravitational spin-orbit coupling (fine structure of the gravitational atom)
==============================================================================

A Dirac particle in the static metric ds^2 = -N^2 dt^2 + A^2 dx^2 has the Hamiltonian
(flat measure, symmetric ordering)
    H = beta m N + (c'/2) {F, alpha.p},     F = N / A.
In the weak field (N = 1 - U, A = 1 + gamma U) its Foldy-Wouthuysen form contains the
spin-orbit term  (2 gamma + 1) c'^2/(4 m) (1/r)(dU/dr) sigma.L  (de Sitter 3/2 = gamma + 1/2
per unit spin; Hehl-Ni 1/4 for an accelerated frame, gamma = 0).  The fine structure
E(2p3/2) - E(2p1/2) is therefore 1 : 3 : 3 for wait only (W) : H2 : linear lapse; the
perihelion-type l splitting of the previous round does not enter (same |m_l|).

A.  Radial Dirac reference in 2D (k_z = 0; kappa = +-j labels the two spin blocks), staggered
    grid with the component that is regular at r = 0 on the half grid.  Weak-field check of the
    ratios -1 : 1 : 3 : 3 (A = N, W, H2, linear), and the reference for the lattice runs, with
    and without the lattice's spin-independent Wilson-like term (Delta(k) = m + (1 + r_b) c'^2 k^2/2m,
    linear coupling c1 = c' to 0.2 %).
B.  Lattice: 2D gravitational atom in the full 3D walker on a thin periodic slab (n x n x 2,
    Bloch k_z = 0), U = GM / sqrt(rho^2 + rc^2) about the z axis.  Start: the reference 2p
    envelope x e^{i phi} x rest spinor with J_z = +-1/2 (2p3/2 / 2p1/2), projected onto the
    exact C4 sector j mod 4 of the lattice rotation.  Levels by filter diagonalisation from the
    autocorrelation <psi0|psi(t)> alone (Hann-windowed Fourier vectors at K energies near the
    target, Rayleigh-Ritz with one tick; residual sqrt(1 - |lambda|^2)).
    One run per (variant, spin): python phase_bit_s4_open15.py B <W|H2|lin> <up|dn>
C.  Collect the runs: fine structure lattice vs Dirac reference, ratios W : H2 : linear.

Run parts: python phase_bit_s4_open15.py [A] [B variant spin] [C]
"""

import functools
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import data_path
from phase_bit_s4_open import CP

print = functools.partial(print, flush=True)

MU, AF, RCF = 0.10, 0.30, 0.5            # mass parameter, coupling alpha_f = m GM / c', softening rc = RCF a0
NBOX = int(os.environ.get('NBOX', 340))  # slab n x n x 2
NTICK = int(os.environ.get('NTICK', 8000))       # ticks per run

VARIANTS = {'flat': (lambda U: 1 - U, lambda U: 1 + 0 * U, None, -1),   # A = N (gamma = -1), no lattice scheme
            'W': (lambda U: np.exp(-U), lambda U: np.exp(-U), 0, 0),       # N, F = N/A, n (A = N^-n), gamma
            'H2': (lambda U: np.exp(-U), lambda U: np.exp(-2 * U), 1, 1),
            'lin': (lambda U: 1 - U, lambda U: (1 - U) ** 2, 1, 1)}


def params(mu=MU, af=AF, rcf=RCF):
    m = 2 * mu
    GM = af * CP / m
    a0 = CP / (m * af)
    return m, GM, a0, rcf * a0


# ═════════════════════════════════════════════════════════════════════════════
def dirac_radial(kap, m, Nf, Ff, R, n, Etarget, k=4, rb=0.0, nexp=0):
    """2D radial Dirac in the metric: E P = mN P + c'(sqF d sqF + F kap/r) Q ;
    E Q = -mN Q + c'(-sqF d sqF + F kap/r) P.  P = sqrt(r) x large component (orbital m_l),
    Q = small component; kap = j for the spin-up block, -j for spin down.  The component that is
    ~ r^1/2 at the origin sits on the half grid.  rb: Wilson-like term beta rb c'^2/(2m) N^1/2 Pi^2 N^1/2
    on the large component (Pi^2 = N^(n/2)(-div N^n grad)N^(n/2), orbital m_l of P)."""
    h = R / n
    s = (np.arange(n) + 0.5) * h
    r = (np.arange(n) + 1.0) * h
    if kap > 0:
        gP, gQ = s, r
        sqFP, sqFQ = np.sqrt(Ff(gP)), np.sqrt(Ff(gQ))
        main = sqFP * sqFQ / h + Ff(gP) * kap / gP / 2
        low = -sqFP[1:] * sqFQ[:-1] / h + Ff(gP)[1:] * kap / gP[1:] / 2
        D = sp.diags([main, low], [0, -1], shape=(n, n))
        ml = kap - 0.5
    else:
        gP, gQ = r[:-1], s                 # P = 0 at the outer wall, as for kap > 0 (no edge state)
        sqFP, sqFQ = np.sqrt(Ff(gP)), np.sqrt(Ff(gQ))
        main = -sqFP * sqFQ[:-1] / h + Ff(gP) * kap / gP / 2
        up = sqFP * sqFQ[1:] / h + Ff(gP) * kap / gP / 2
        D = sp.diags([main, up], [0, 1], shape=(n - 1, n))
        ml = -kap + 0.5
    HPP = sp.diags(m * Nf(gP))
    if rb != 0.0:
        w = Nf(gP) ** nexp
        wp = np.gradient(w, h)
        wh = Nf(gP + h / 2) ** nexp
        whm = Nf(np.maximum(gP - h / 2, 1e-12)) ** nexp
        L = sp.diags([wh + whm, -wh[:-1], -whm[1:]], [0, 1, -1]) / h ** 2 \
            + sp.diags(w * (ml ** 2 - 0.25) / gP ** 2 + wp / (2 * gP))
        sc = sp.diags(Nf(gP) ** ((nexp + 1) / 2))
        HPP = HPP + rb * CP ** 2 / (2 * m) * (sc @ L @ sc)
    H = sp.bmat([[HPP, CP * D], [CP * D.T, sp.diags(-m * Nf(gQ))]]).tocsc()
    w_, v = eigsh(H, k=k, sigma=Etarget, which='LM')
    o = np.argsort(w_)
    return w_[o], v[:, o], gP, gQ


def ref_2p(name, mu=MU, af=AF, rcf=RCF, R=None, n=8000, rb=0.0):
    """2p3/2 (kap = 3/2) and 2p1/2 (kap = -1/2) levels and large-component envelopes."""
    m, GM, a0, rc = params(mu, af, rcf)
    Nfun, Ffun, nexp, _ = VARIANTS[name]
    U = lambda x: GM / np.sqrt(x ** 2 + rc ** 2)
    Nf, Ff = (lambda x: Nfun(U(x))), (lambda x: Ffun(U(x)))
    Ry = m * af ** 2 / 2
    Et = m - 0.42 * Ry
    R = 30 * a0 if R is None else R
    res = {}
    for kap in (1.5, -0.5):
        w, v, gP, gQ = dirac_radial(kap, m, Nf, Ff, R, n, Et, k=3, rb=rb, nexp=nexp or 0)
        i = np.argmin(np.abs(w - Et))
        vec = v[:, i]
        P = vec[:len(gP)]
        P = P * np.sign(P[np.argmax(np.abs(P))])
        res[kap] = (w[i], gP, P, w)
    return res


def lattice_rb(mu):
    """Wilson-like term: (Delta_lattice(k) - Dirac)/(c'^2 k^2 / 2m) of the upper doublet at small k."""
    from phase_bit_mass_s4 import bands_C
    from phase_bit_s4_open9 import Xl, LAM0
    m = 2 * mu
    C0 = expm(1j * Xl(LAM0)(mu))
    kk = 0.05
    E = bands_C(np.array([kk, 0, 0]), C0)[0]
    Dk = np.hypot(m, CP * kk)
    Ep = np.sort(E[np.argsort(np.abs(E - Dk))[:2]]).mean()
    return (Ep - Dk) / (CP ** 2 * kk ** 2 / (2 * m))


def part_A(out):
    print("A. radial Dirac reference: fine structure E(2p3/2) - E(2p1/2) for A = N, W, H2, linear lapse")
    rows = []
    for af in (0.05, 0.1, 0.2, 0.3):
        m, GM, a0, rc = params(af=af)
        d = []
        for name in ('flat', 'W', 'H2', 'lin'):
            r = ref_2p(name, af=af)
            d.append(r[1.5][0] - r[-0.5][0])
        d = np.array(d)
        rows.append([af, *d])
        print(f"   alpha_f = {af}: U(0) = {GM/rc:.3f}; delta / (m af^4) = " + " ".join(f"{x/(m*af**4):+.4f}" for x in d)
              + "; ratio to W: " + " ".join(f"{x/d[1]:+.3f}" for x in d) + "   (FW weak field: -1, 1, 3, 3)")
    out['A_ratios'] = np.array(rows)
    rb = lattice_rb(MU)
    m, GM, a0, rc = params()
    print(f"   lattice run: mu = {MU} (m = {2*MU}), alpha_f = {AF}, GM = {GM:.4f}, a0 = {a0:.2f}, rc = {rc:.2f}, "
          f"U(0) = {GM/rc:.3f}; Wilson-like term r_b = {rb:+.4f}")
    ref = []
    for name in ('W', 'H2', 'lin'):
        for rbv in (0.0, rb):
            for R in (NBOX / 2 * 0.6124, 30 * a0):
                r = ref_2p(name, R=R, rb=rbv)
                ref.append([VARIANTS[name][3], rbv, R, r[1.5][0] - m, r[-0.5][0] - m])
                print(f"   {name:3s} r_b = {rbv:+.4f}, R = {R:6.1f}: 2p3/2 {r[1.5][0]-m:+.6e}, 2p1/2 {r[-0.5][0]-m:+.6e}, "
                      f"fine structure {r[1.5][0]-r[-0.5][0]:+.4e}")
    out['A_ref'] = np.array(ref)
    out['A_par'] = np.array([MU, AF, RCF, NBOX, rb])


# ═════════════════════════════════════════════════════════════════════════════
def rotate_c4(v, n, Rint):
    """lattice C4 about the z axis through site (n//2, n//2): v (nreg?, n, n, 2, 24)."""
    c = n // 2
    a = np.roll(v, (-c, -c), axis=(-4, -3))
    a = a[..., :, (-np.arange(n)) % n, :, :]            # C[i, a] = A0[i, -a]
    a = np.swapaxes(a, -4, -3)                           # B[a, b] = C[b, a]  ->  (x, y) -> (-y, x)
    a = np.roll(a, (c, c), axis=(-4, -3))
    return a @ Rint.T


def part_B(variant, spin):
    from phase_bit_s4_open12 import Walker3D
    from phase_bit_s4_open13 import R90_int, J_int
    from phase_bit_mass_s4 import bands_C
    from phase_bit_s4_open9 import Xl, LAM0
    from quantum_fcc_3d import DT_HALF, LAT_A
    t0 = time.time()
    n = NBOX
    shape = (n, n, 2)
    m, GM, a0, rc = params()
    rb = lattice_rb(MU)
    Nfun, Ffun, nexp, _ = VARIANTS[variant]
    xs = (np.arange(n) - n // 2) * LAT_A
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    rho = np.hypot(X, Y)
    U = GM / np.sqrt(rho ** 2 + rc ** 2)
    Nf = np.repeat(Nfun(U)[..., None], 2, axis=2)
    kind = 'W' if variant == 'W' else 'WH'
    nreg = 2 if kind == 'W' else 4
    wk = Walker3D(shape, kind, Nf, mu=MU)
    print(f"B. {variant}, spin {spin}: slab {shape}, half box {n/2*LAT_A:.1f} = {n/2*LAT_A/a0:.1f} a0, N_min {Nf.min():.4f}, "
          f"{nreg} registers, {NTICK} ticks")
    # rest doublet and J_z
    C0 = expm(1j * Xl(LAM0)(MU))
    E0, V0 = bands_C(np.zeros(3), C0)
    Qd, _ = np.linalg.qr(V0[:, np.argsort(np.abs(E0 - m))[:2]])
    Jz = J_int()[2]
    w, v = np.linalg.eigh(Qd.conj().T @ Jz @ Qd)
    u = Qd @ v[:, np.argmax(w) if spin == 'up' else np.argmin(w)]
    jz_int = 0.5 if spin == 'up' else -0.5
    kap = 1.5 if spin == 'up' else -0.5
    # reference envelope (with the Wilson-like term) and target energy
    ref = ref_2p(variant, R=n / 2 * LAT_A, rb=rb)
    Et, gP, P, _ = ref[kap]
    env = np.interp(rho, gP, P / np.sqrt(gP), right=0.0) * np.exp(1j * np.arctan2(Y, X))
    I = np.indices(shape).sum(0) - 2 * (n // 2)
    psi = env[:, :, None, None] * (I % 2 == 0)[..., None] * u[None, None, None, :]
    # project onto the exact C4 sector j = m_l + jz (mod 4)
    Rint = R90_int(2)
    j = 1 + jz_int
    acc = psi.copy()
    r_ = psi
    for k in range(1, 4):
        r_ = rotate_c4(r_, n, Rint)
        acc = acc + np.exp(1j * np.pi * j * k / 2) * r_
    psi = (acc / 4).reshape(-1, 24)
    psi = (psi / np.linalg.norm(psi)).astype(np.complex64)
    st = [psi] + [np.zeros_like(psi) for _ in range(nreg - 1)]
    # symmetry check: C4 commutes with one tick
    a1 = wk.tick([rotate_c4(x.reshape(*shape, 24), n, Rint).reshape(-1, 24).astype(np.complex64) for x in st])
    a2 = [rotate_c4(x.reshape(*shape, 24), n, Rint).reshape(-1, 24) for x in wk.tick(st)]
    comm = np.sqrt(sum(np.linalg.norm(p - q) ** 2 for p, q in zip(a1, a2)))
    print(f"   C4 symmetry: || R U psi - U R psi || = {comm:.2e}; target (reference) E - m = {Et-m:+.6e}")
    # pre-filter against the pi branches of the gates
    ph = np.exp(1j * Et * DT_HALF)
    for _ in range(4):
        st2 = wk.tick(st)
        st = [(a + ph * b) / 2 for a, b in zip(st, st2)]
    nrm = np.sqrt(sum(np.vdot(a, a).real for a in st))
    st = [a / nrm for a in st]
    s0 = [a.copy() for a in st]
    sig = np.empty(NTICK + 1, complex)
    for it in range(NTICK + 1):
        sig[it] = sum(np.vdot(a, b) for a, b in zip(s0, st))
        if it < NTICK:
            st = wk.tick(st)
        if it % 1000 == 0:
            print(f"      tick {it}  [{time.time()-t0:.0f}s]")
            np.savez_compressed(data_path(f'phase_bit_s4_open15_B_{variant}_{spin}_partial.npz'), Et=Et - m,
                                sig=sig[:it + 1], comm=comm, par=np.array([MU, AF, RCF, n, it, rb]))
    E, res, wgt = fdm(sig, DT_HALF, fdm_grid(Et, NTICK * DT_HALF))
    best = np.argmax(wgt)
    for i in range(len(E)):
        print(f"   Ritz: E - m = {E[i]-m:+.7e}, residual {res[i]:.1e}, weight in the start state {wgt[i]:.4f}")
    print(f"   -> level {E[best]-m:+.7e} (reference {Et-m:+.7e}), residual {res[best]:.1e}  [{time.time()-t0:.0f}s]")
    np.savez_compressed(data_path(f'phase_bit_s4_open15_B_{variant}_{spin}.npz'), Et=Et - m, sig=sig, comm=comm,
                        par=np.array([MU, AF, RCF, n, NTICK, rb]))


def fdm_grid(Et, T, K=10, spacing=0.75):
    return Et + spacing * 2 * np.pi / T * (np.arange(K) - (K - 1) / 2)


def fdm(sig, dt, Eq):
    """filter diagonalisation from the autocorrelation c(t) = <psi0|psi(t)> (t = 0..L ticks) of a unitary
    evolution: Rayleigh-Ritz in span{ Phi_j = sum_n a_j(n) psi(n) }, a_j(n) = hann(n) e^{i E_j n dt}.
    <Phi_j|U^p|Phi_k> = sum_d c(d + p) R_jk(d), R_jk(d) = sum_n a_j(n)* a_k(n + d) (FFT), c(-t) = c(t)*.
    Returns Ritz energies, residuals sqrt(1 - |lambda|^2) and weights in psi0."""
    L = len(sig) - 1
    w = np.hanning(L + 2)[1:-1]
    A = w[None, :] * np.exp(1j * np.outer(Eq, np.arange(L) * dt))
    cf = lambda d: np.where(d >= 0, sig[np.abs(d)], np.conj(sig[np.abs(d)]))
    lags = np.arange(-(L - 1), L)
    c0, c1 = cf(lags), cf(lags + 1)
    K = len(Eq)
    S, M = np.zeros((K, K), complex), np.zeros((K, K), complex)
    nfft = 1 << int(np.ceil(np.log2(2 * L)))
    F = np.fft.fft(A, nfft, axis=1)
    for j in range(K):
        Rl = np.fft.ifft(np.conj(F[j])[None, :] * F, axis=1)[:, lags % nfft]
        S[j], M[j] = Rl @ c0, Rl @ c1
    sv, V = np.linalg.eigh((S + S.conj().T) / 2)
    keep = sv > 1e-10 * sv.max()
    X = V[:, keep] / np.sqrt(sv[keep])
    lam, cv = np.linalg.eig(X.conj().T @ M @ X)
    E = -np.angle(lam) / dt
    res = np.sqrt(np.clip(1 - np.abs(lam) ** 2, 0, None))
    wgt = np.abs(((A @ sig[:L]) @ X) @ cv) ** 2
    o = np.argsort(E)
    return E[o], res[o], wgt[o]


def part_C(out):
    print("C. fine structure: lattice vs Dirac reference")
    m, GM, a0, rc = params()
    from quantum_fcc_3d import LAT_A, DT_HALF
    rb = lattice_rb(MU)
    rows = []
    for variant in ('W', 'H2', 'lin'):
        if not all(os.path.exists(data_path(f'phase_bit_s4_open15_B_{variant}_{s}.npz')) for s in ('up', 'dn')):
            continue
        f = {s: np.load(data_path(f'phase_bit_s4_open15_B_{variant}_{s}.npz')) for s in ('up', 'dn')}
        El, rs = {}, {}
        for s_, d_ in f.items():
            nt = int(d_['par'][4])
            E, res, wgt = fdm(d_['sig'], DT_HALF, fdm_grid(float(d_['Et']) + m, nt * DT_HALF))
            i = np.argmax(wgt)
            El[s_], rs[s_] = E[i] - m, res[i]
            # stability: other filter grids
            alt = [fdm(d_['sig'], DT_HALF, fdm_grid(float(d_['Et']) + m, nt * DT_HALF, K, sp_)) for K, sp_ in ((6, 0.75), (14, 0.5))]
            spread = max(abs(a[0][np.argmax(a[2])] - m - El[s_]) for a in alt)
            print(f"   {variant} {s_}: level {El[s_]:+.7e}, residual {rs[s_]:.1e}, weight {wgt[i]:.4f}, filter-grid spread {spread:.1e}")
        r0 = ref_2p(variant, R=NBOX / 2 * LAT_A, rb=0.0)
        r1 = ref_2p(variant, R=NBOX / 2 * LAT_A, rb=rb)
        d_l = El['up'] - El['dn']
        d0 = r0[1.5][0] - r0[-0.5][0]
        d1 = r1[1.5][0] - r1[-0.5][0]
        rows.append([VARIANTS[variant][3], El['up'], El['dn'], r1[1.5][0] - m, r1[-0.5][0] - m, d_l, d1, d0,
                     rs['up'], rs['dn']])
        print(f"   {variant:3s}: lattice 2p3/2 {El['up']:+.6e}, 2p1/2 {El['dn']:+.6e} (residuals {rs['up']:.0e}, {rs['dn']:.0e}); "
              f"reference {r1[1.5][0]-m:+.6e}, {r1[-0.5][0]-m:+.6e}")
        print(f"        fine structure: lattice {d_l:+.4e}, Dirac + Wilson term {d1:+.4e} (ratio {d_l/d1:.3f}), "
              f"pure Dirac {d0:+.4e} (ratio {d_l/d0:.3f})")
    r = np.array(rows)
    out['C_rows'] = r
    if len(r) == 3:
        print(f"   ratios to W: lattice H2 {r[1,5]/r[0,5]:.3f}, lin {r[2,5]/r[0,5]:.3f}; "
              f"reference {r[1,6]/r[0,6]:.3f}, {r[2,6]/r[0,6]:.3f}  (weak-field FW 3, 3)")


def main():
    args = sys.argv[1:] or ['A']
    if args[0].upper() == 'B':
        part_B(args[1], args[2])
        return
    path = data_path('phase_bit_s4_open15.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in args:
        {'A': part_A, 'C': part_C}[p.upper()](out)
    np.savez_compressed(path, **out)
    print(f"saved {path}")


if __name__ == '__main__':
    main()
