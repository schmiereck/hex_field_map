#!/usr/bin/env python3
"""
Paper 3, Experiment S4 — A mass for the 3D spin vortex that keeps time reversal
===============================================================================

In 3D the spin-1/2 walker (minimal SU(2) transport, or the finite 2O register)
is massless: its lowest level is a 4-fold Dirac point with slope c/3.  Here a
mass term is searched for systematically:

1.  Effective Dirac Hamiltonian at the Dirac point.  With P the 4-dim
    eigenspace, H_eff(k) = E0 + sum_i k_i Gamma_i, Gamma_i = P^dag (dr_i/dt) P
    (degenerate Hellmann–Feynman).  A local Hermitian term V added to the coin
    generator, C = expm(i(eps G + mu V)), opens a gap iff the traceless part of
    P^dag V P ANTICOMMUTES with all three Gamma_i.  It keeps Kramers iff it
    commutes with Theta = (1 (x) i sigma_y) K.  Its parity under inversion
    Pi : d -> -d (spinor unchanged) separates an ordinary Dirac mass
    (P-even, "beta") from a parity-odd one ("i beta gamma5").
2.  Scan of local terms: couplings between headings at 60, 90, 120, 180 deg
    with spin factors {transport, identity, i u_d.sigma, i n.sigma}, symmetric
    and antisymmetric, plus on-site terms — for both transport models.
3.  Verification on the full band structure: gap vs mu, Kramers at every k,
    Dirac dispersion, isotropy, and E <-> -E for the 2O register.
4.  Real-space 3+1D dynamics with the winning term: massless shell vs massive
    packet at rest, boosted packets with v < c, Zitterbewegung.
"""

import itertools
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantum_fcc_3d as F
from quantum_fcc_3d import (FCC_UNIT, FCC_STEPS, MOVES_PHYS, DT_HALF, C_LIGHT, LAT_A,
                            N_D, su2_transport, step3)
from phase_bit_fcc_2O import build_transport, generator as gen_2O
from paths import data_path

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
DOT = FCC_UNIT @ FCC_UNIT.T
ANTI = np.array([int(np.argmin(np.linalg.norm(FCC_UNIT + FCC_UNIT[d], axis=1)))
                 for d in range(N_D)])
PI = np.kron(np.eye(N_D)[ANTI], I2)                     # inversion d -> -d
THETA_U = np.kron(np.eye(N_D), 1j * SY)                 # Theta = THETA_U K
DIRV = [np.kron(np.diag(MOVES_PHYS[:, i]), I2) / DT_HALF for i in range(3)]


def sdot(v):
    return v[0] * SX + v[1] * SY + v[2] * SZ


# ─── models ──────────────────────────────────────────────────────────────────

def model_generators():
    Q2O, _ = build_transport()
    G_min = F.generator(0.5)
    G_2O, _ = gen_2O(Q2O, 'E1/2')
    return {'minimal SU(2)': G_min, '2O register': G_2O}, Q2O


def dirac_point(G, which='lowest'):
    """4-dim eigenspace of the lowest (largest-eigenvalue-of-G) level."""
    w, V = np.linalg.eigh(G)
    lam = w.max() if which == 'lowest' else w.min()
    P = V[:, np.isclose(w, lam, atol=1e-9)]
    Gam = [P.conj().T @ D @ P for D in DIRV]
    return P, Gam, lam


# ─── candidate terms ─────────────────────────────────────────────────────────

def coupling(cosang, factor):
    """K = sum over oriented heading pairs at the given angle of |e><d| (x) f(d,e)."""
    K = np.zeros((2 * N_D, 2 * N_D), dtype=complex)
    for d in range(N_D):
        for e in range(N_D):
            if not np.isclose(DOT[d, e], cosang):
                continue
            if factor == 'transport':
                if np.isclose(cosang, -1.0):
                    return None
                f = su2_transport(FCC_UNIT[d], FCC_UNIT[e])
            elif factor == 'identity':
                f = I2
            elif factor == 'i u_d.s':
                f = 1j * sdot(FCC_UNIT[d])
            elif factor == 'i n.s':
                n = np.cross(FCC_UNIT[d], FCC_UNIT[e])
                if np.linalg.norm(n) < 1e-9:
                    return None
                f = 1j * sdot(n / np.linalg.norm(n))
            K[2 * e:2 * e + 2, 2 * d:2 * d + 2] = f
    return K


def candidates():
    out = {}
    for ang, c in (('60°', 0.5), ('90°', 0.0), ('120°', -0.5), ('180°', -1.0)):
        for fac in ('transport', 'identity', 'i u_d.s', 'i n.s'):
            K = coupling(c, fac)
            if K is None:
                continue
            out[f'{ang} {fac}, K+K†'] = 0.5 * (K + K.conj().T)
            out[f'{ang} {fac}, i(K−K†)'] = 0.5j * (K - K.conj().T)
    out['on-site u_d.s'] = sum(np.kron(np.diag(np.eye(N_D)[d]), sdot(FCC_UNIT[d]))
                               for d in range(N_D))
    return out


def classify(V, P, Gam):
    M = P.conj().T @ V @ P
    nrm = np.abs(M).max()
    t_inv = np.abs(THETA_U @ V.conj() @ THETA_U.conj().T - V).max()
    pv = PI @ V @ PI
    parity = '+' if np.allclose(pv, V) else ('−' if np.allclose(pv, -V) else 'mixed')
    if nrm < 1e-10:
        return dict(norm=0.0, T=t_inv, parity=parity, anti=np.nan, comm=np.nan,
                    eig=np.zeros(4), kind='no action')
    M0 = M - np.trace(M) / 4 * np.eye(4)
    s = np.abs(M0).max()
    if s < 1e-10:
        return dict(norm=nrm, T=t_inv, parity=parity, anti=0.0, comm=0.0,
                    eig=np.linalg.eigvalsh(M), kind='shift only')
    anti = max(np.abs(M0 @ g + g @ M0).max() for g in Gam) / s
    comm = max(np.abs(M0 @ g - g @ M0).max() for g in Gam) / s
    if anti < 1e-8:
        kind = 'MASS' if t_inv < 1e-10 else 'mass, breaks T'
    elif comm < 1e-8:
        kind = 'splits cone' if t_inv < 1e-10 else 'splits cone, breaks T'
    else:
        kind = 'mixed'
    return dict(norm=nrm, T=t_inv, parity=parity, anti=anti, comm=comm,
                eig=np.linalg.eigvalsh(M), kind=kind)


# ─── bands ───────────────────────────────────────────────────────────────────

def bands_C(k, C):
    ph = np.exp(-1j * (MOVES_PHYS @ np.asarray(k, float)))
    lam, V = np.linalg.eig(np.repeat(ph, 2)[:, None] * C)
    E = -np.angle(lam) / DT_HALF
    o = np.argsort(E)
    return E[o], V[:, o]


def group_velocity(u):
    w = (np.abs(u) ** 2).reshape(N_D, 2).sum(axis=1)
    return (w / w.sum()) @ MOVES_PHYS / DT_HALF


# ─── real-space dynamics ─────────────────────────────────────────────────────

def lattice(N):
    c = N // 2
    ix = np.arange(N) - c
    IX, IY, IZ = np.meshgrid(ix, ix, ix, indexing='ij')
    sub = ((IX + IY + IZ) % 2 == 0)
    return c, IX * LAT_A, IY * LAT_A, IZ * LAT_A, sub


def band_packet3(N, sigma, k0, C, band, x0=(0.0, 0.0, 0.0), thresh=1e-6, doublet=True):
    """Packet in one Kramers doublet (bands band, band+1 of the sorted 24).
    Smooth gauge: at every k the reference spinor u_ref(k0) is projected onto
    the doublet and normalised, so the whole packet carries one spin state."""
    c, X, Y, Z, sub = lattice(N)
    g = (np.exp(-((X - x0[0]) ** 2 + (Y - x0[1]) ** 2 + (Z - x0[2]) ** 2) / (2 * sigma ** 2))
         * np.exp(1j * (k0[0] * X + k0[1] * Y + k0[2] * Z)) * sub)
    gh = np.fft.fftn(g)
    kk = 2 * np.pi * np.fft.fftfreq(N) / LAT_A
    KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing='ij')
    mask = np.abs(gh) > thresh * np.abs(gh).max()
    ks = np.column_stack([KX[mask], KY[mask], KZ[mask]])
    ph = np.exp(-1j * (ks @ MOVES_PHYS.T))                       # (n, 12)
    T = np.repeat(ph, 2, axis=1)[:, :, None] * C[None]
    lam, U = np.linalg.eig(T)
    E = -np.angle(lam) / DT_HALF
    order = np.argsort(E, axis=1)
    n = np.arange(len(ks))
    uref = bands_C(k0, C)[1][:, band]
    uref /= np.linalg.norm(uref)
    if doublet:
        nb = 2 if doublet is True else int(doublet)
        A = np.stack([U[n, :, order[:, band + j]] for j in range(nb)], axis=2)
        Qd, _ = np.linalg.qr(A)                                    # (n, 24, 2)
        u = np.einsum('nij,nj->ni', Qd, np.einsum('nij,i->nj', Qd.conj(), uref))
    else:
        u = U[n, :, order[:, band]]
        ov = u @ uref.conj()
        u = u * (np.conj(ov) / (np.abs(ov) + 1e-30))[:, None]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    psih = np.zeros((N, N, N, 24), dtype=complex)
    psih[mask] = gh[mask][:, None] * u
    psi = np.fft.ifftn(psih, axes=(0, 1, 2)).astype(np.complex64)
    psi /= np.sqrt((np.abs(psi) ** 2).sum())
    return psi.reshape(N, N, N, N_D, 2)


def plain_packet3(N, sigma, k0, u, x0=(0.0, 0.0, 0.0)):
    c, X, Y, Z, sub = lattice(N)
    g = (np.exp(-((X - x0[0]) ** 2 + (Y - x0[1]) ** 2 + (Z - x0[2]) ** 2) / (2 * sigma ** 2))
         * np.exp(1j * (k0[0] * X + k0[1] * Y + k0[2] * Z)) * sub)
    psi = (g[..., None] * (u / np.linalg.norm(u))[None, None, None, :]).astype(np.complex64)
    psi /= np.sqrt((np.abs(psi) ** 2).sum())
    return psi.reshape(N, N, N, N_D, 2)


def observe(psi, N):
    c, X, Y, Z, _ = lattice(N)
    p = (np.abs(psi) ** 2).sum(axis=(-2, -1))
    s = p.sum()
    r = np.array([(X * p).sum(), (Y * p).sum(), (Z * p).sum()]) / s
    rms = np.sqrt((((X - r[0]) ** 2 + (Y - r[1]) ** 2 + (Z - r[2]) ** 2) * p).sum() / s)
    flat = psi.reshape(-1, 24)
    par = np.real(np.vdot(flat, flat @ PI.T.astype(np.complex64))) / s   # <Pi>
    return r, rms, par, s


def run3(psi, C, N, n_steps, every=1):
    C = C.astype(np.complex64)
    rec = [observe(psi, N)]
    ts = [0.0]
    for t in range(1, n_steps + 1):
        psi = step3(psi, C, 2)
        if t % every == 0:
            rec.append(observe(psi, N))
            ts.append(t * DT_HALF)
    return (np.array(ts), np.array([r[0] for r in rec]), np.array([r[1] for r in rec]),
            np.array([r[2] for r in rec]), np.array([r[3] for r in rec]))


def fit_damped(t, y, omega0):
    from scipy.optimize import curve_fit
    f = lambda t, a, b, A, g, w, ph: a + b * t + A * np.exp(-g * t) * np.cos(w * t + ph)
    best = None
    for ph0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        try:
            p, _ = curve_fit(f, t, y, p0=[y.mean(), 0, 0.5 * np.ptp(y), 0.02, omega0, ph0],
                             maxfev=20000)
            res = np.sum((f(t, *p) - y) ** 2)
            if best is None or res < best[0]:
                best = (res, p)
        except RuntimeError:
            pass
    p = best[1]
    return abs(p[4]), p[3], abs(p[2])


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    out = {}
    models, Q2O = model_generators()
    cands = candidates()

    # 1./2. scan
    scan_rows = []
    for mname, G in models.items():
        P, Gam, lam = dirac_point(G)
        acg = max(np.abs(Gam[i] @ Gam[j] + Gam[j] @ Gam[i]
                         - 2 * (i == j) * (C_LIGHT / 3) ** 2 * np.eye(4)).max()
                  for i in range(3) for j in range(3))
        pc = np.abs(PI @ G - G @ PI).max()
        print(f"1. {mname}: Dirac point at G-eigenvalue {lam:.6f} (E = {-lam:.4f} eps/dt), "
              f"dim {P.shape[1]}; {{Gamma_i, Gamma_j}} = 2 (c/3)^2 delta_ij to {acg:.1e}; "
              f"[Pi, G] = {pc:.1e}")
        print(f"   {'term':30s} {'kind':22s} {'P':>5s} {'|PVP|':>6s}  eig(PVP)")
        for cname, V in cands.items():
            r = classify(V, P, Gam)
            scan_rows.append((mname, cname, r['kind'], r['parity'], r['norm']))
            if r['kind'] in ('no action',):
                continue
            print(f"   {cname:30s} {r['kind']:22s} {r['parity']:>5s} {r['norm']:6.3f}  "
                  f"{np.round(r['eig'], 4).tolist()}")
    out['scan'] = np.array(scan_rows, dtype=object)
    pi_check = np.abs(cands['180° identity, K+K†'] - PI).max()
    print(f"   note: '180° identity, K+K†' IS the inversion operator Pi (difference {pi_check:.1e})")
    print(f"   [{time.time()-t0:.0f}s]")

    # 3. verification on the full bands
    EPS = 0.3
    rng = np.random.default_rng(2)
    kr = rng.normal(size=(30, 3)); kr *= (rng.uniform(0.05, 1.5, 30) / np.linalg.norm(kr, axis=1))[:, None]
    terms = {'Pi (180° identity)': PI,
             '90° identity': cands['90° identity, K+K†'],
             '180° i u_d.s': cands['180° i u_d.s, K+K†']}
    ver = {}
    for mname, G in models.items():
        for tname, V in terms.items():
            P, Gam, lam = dirac_point(G)
            r = classify(V, P, Gam)
            if r['kind'] != 'MASS':
                print(f"3. {mname} + {tname}: {r['kind']} -> skipped")
                continue
            gaps = []
            for mu in (0.01, 0.02, 0.05, 0.1):
                C = expm(1j * (EPS * G + mu * V))
                E0 = bands_C(np.zeros(3), C)[0]
                gaps.append((E0[2] - E0[1]) / mu)
            MU = 0.1
            C = expm(1j * (EPS * G + MU * V))
            kram, sym = 0.0, 0.0
            for k in kr:
                E = bands_C(k, C)[0]
                kram = max(kram, np.abs(E[1::2] - E[0::2]).max())
                sym = max(sym, np.abs(np.sort(E) - np.sort(-E)).max())
            E0 = bands_C(np.zeros(3), C)[0]
            m = 0.5 * (E0[2] - E0[1]); Ec = 0.5 * (E0[2] + E0[1])
            dev = []
            for d in ([1, 0, 0], [1, 1, 1], [1, 2, 3]):
                v = np.array(d, float) / np.linalg.norm(d)
                for k in (0.05, 0.1, 0.2):
                    Ek = bands_C(k * v, C)[0][:4]
                    lower = Ek[1] - Ec
                    dev.append((d, k, (abs(lower) - np.hypot(m, C_LIGHT / 3 * k))
                                / np.hypot(m, C_LIGHT / 3 * k)))
            print(f"3. {mname} + mu*{tname} (eps={EPS}): gap/mu = "
                  f"{np.round(gaps, 6).tolist()}; Kramers split {kram:.1e}; "
                  f"E<->-E asymmetry {sym:.1e}; parity {r['parity']}")
            print("   lower massive branch vs sqrt(m^2+(c/3 k)^2), rel. dev: " + ", ".join(
                f"{d}@{k}: {x:+.3f}" for d, k, x in dev))
            ver[(mname, tname)] = (gaps, kram, sym)
    print(f"   [{time.time()-t0:.0f}s]")

    # band structure for the figure (minimal SU(2), Pi term)
    G = models['minimal SU(2)']
    kline = np.linspace(0, 0.8, 41)
    for mu in (0.0, 0.1):
        C = expm(1j * (EPS * G + mu * PI))
        out[f'bands_mu{mu}'] = np.array([bands_C(k * np.array([1.0, 0, 0]), C)[0] for k in kline])
    out['kline'] = kline
    gap_mu = np.array([0.0, 0.02, 0.05, 0.1, 0.2])
    out['gap_mu'] = gap_mu
    out['gap'] = np.array([np.diff(bands_C(np.zeros(3), expm(1j * (EPS * G + mu * PI)))[0][1:3])[0]
                           for mu in gap_mu])
    G2 = models['2O register']
    out['gap_2O'] = np.array([np.diff(bands_C(np.zeros(3), expm(1j * (EPS * G2 + mu * PI)))[0][1:3])[0]
                              for mu in gap_mu])

    # 3b. structure of the massive quartet: common background + Dirac split
    G = models['minimal SU(2)']
    print("3b. quartet = Ec(k) +- Delta(k): background and split vs Dirac (minimal SU(2), eps=0.3)")
    bg_rows = []
    for mu in (0.0, 0.05, 0.15):
        C = expm(1j * (EPS * G + mu * PI))
        E0q = bands_C(np.zeros(3), C)[0][:4]
        mm = 0.5 * (E0q[2] - E0q[1]); ec0 = 0.5 * (E0q[2] + E0q[1])
        for k in (0.05, 0.1, 0.2, 0.4):
            Eq = bands_C(np.array([k, 0, 0]), C)[0][:4]
            ec = 0.5 * (Eq[2] + Eq[1]) - ec0
            dl = 0.5 * (Eq[2] - Eq[1])
            dref = np.hypot(mm, C_LIGHT / 3 * k)
            bg_rows.append((mu, k, ec / k ** 2, dl / dref - 1))
        print(f"   mu={mu}: m={mm:.3f}; background Ec(k)-Ec(0) = -alpha k^2 with alpha at k=.05/.1/.2/.4: "
              + ", ".join(f"{-r[2]:.3f}" for r in bg_rows[-4:])
              + "; Delta/sqrt(m^2+(c/3 k)^2)-1: " + ", ".join(f"{r[3]:+.3f}" for r in bg_rows[-4:]))
    out['bg_rows'] = np.array(bg_rows)
    for e in (0.1, 0.2, 0.3, 0.5):
        C = expm(1j * e * G)
        ec = [0.5 * sum(bands_C(np.array([k, 0, 0]), C)[0][1:3]) for k in (0.0, 0.02)]
        print(f"   eps={e}: background alpha = {-(ec[1]-ec[0])/0.02**2:.4f}  (alpha*eps = {-(ec[1]-ec[0])/0.02**2*e:.4f})")

    # 4. real-space dynamics, minimal SU(2) + mu*Pi
    N, MU, SIG = 81, 0.15, 4.0
    C0 = expm(1j * EPS * G)
    Cm = expm(1j * (EPS * G + MU * PI))
    E0, U0 = bands_C(np.zeros(3), Cm)
    m = 0.5 * (E0[2] - E0[1])
    print(f"4. dynamics: eps={EPS}, mu={MU}, half-gap m = {m:.4f}, grid {N}^3, sigma {SIG}")
    # (a) massless vs massive at rest (upper doublet of the Dirac pair: bands 2,3)
    t, r0, rms0, par0, nrm0 = run3(band_packet3(N, SIG, np.zeros(3), C0, 2), C0, N, 50, every=2)
    t, r1, rms1, par1, nrm1 = run3(band_packet3(N, SIG, np.zeros(3), Cm, 2), Cm, N, 50, every=2)
    late = t > 12
    s0 = np.polyfit(t[late], rms0[late], 1)[0]
    s1 = np.polyfit(t[late], rms1[late], 1)[0]
    print(f"   (a) at rest: rms growth mu=0 {s0:.4f} (cone c/3 = {C_LIGHT/3:.4f}), "
          f"mu={MU} {s1:.4f}; CoM drift {np.abs(r1[-1]).max():.1e}; "
          f"<Pi> {par1[0]:+.4f} -> {par1[-1]:+.4f}; norms {nrm0[-1]:.5f} / {nrm1[-1]:.5f}"
          f"  [{time.time()-t0:.0f}s]")
    out.update(rest_t=t, rest_rms0=rms0, rest_rms1=rms1, rest_par1=par1, rest_par0=par0)
    # (b) boosted massive packets (upper doublet moves along +k)
    vs, vb, nb = [], [], []
    kb = (0.1, 0.2, 0.4)
    for kk in kb:
        k0 = np.array([kk, 0, 0])
        psi = band_packet3(N, SIG, k0, Cm, 2, x0=(-8.0, 0, 0))
        t, r, rms, par, nrm = run3(psi, Cm, N, 40, every=2)
        v = np.polyfit(t, r[:, 0], 1)[0]
        vg = group_velocity(bands_C(k0, Cm)[1][:, 2])[0]
        vs.append(v); vb.append(vg); nb.append(nrm[-1])
        print(f"   (b) k={kk}: v_CoM = {v:+.4f}, band v_g = {vg:+.4f}, c = {C_LIGHT:.4f}, "
              f"norm {nrm[-1]:.5f}")
    out['boost_k'], out['v_sim'], out['v_band'] = np.array(kb), np.array(vs), np.array(vb)
    # (c) Zitterbewegung: rest spinor of the upper massive doublet, boosted
    k0 = np.array([0.3, 0, 0])
    Ek, Uk = bands_C(k0, Cm)
    Uk = Uk / np.linalg.norm(Uk, axis=0)
    u = (Uk[:, 1] + Uk[:, 2]) / np.sqrt(2)      # lower + upper massive state at k0
    psi = plain_packet3(N, 5.0, k0, u, x0=(0.0, 0, 0))
    # predicted lines of <Pi>(t): weights |c_a c_b <u_a|Pi|u_b>| at frequency E_a - E_b
    cvec = np.linalg.solve(Uk, u)
    PiK = Uk.conj().T @ PI @ Uk
    lines = {}
    for a in range(24):
        for b in range(24):
            om = round(abs(Ek[a] - Ek[b]), 4)
            if om < 1e-3:
                continue
            lines[om] = lines.get(om, 0) + abs(cvec[a] * cvec[b] * PiK[a, b])
    top = sorted(lines.items(), key=lambda x: -x[1])[:3]
    t, r, rms, par, nrm = run3(psi, Cm, N, 70, every=1)
    om_p, g_p, A_p = fit_damped(t, par, top[0][0])
    om_y, g_y, A_y = fit_damped(t, r[:, 1], top[0][0])
    E_pair = 2 * np.hypot(m, C_LIGHT / 3 * 0.3)
    print(f"   (c) ZB at k0=0.3: predicted <Pi> lines (omega: weight) "
          + ", ".join(f"{o:.4f}: {w_:.3f}" for o, w_ in top)
          + f"; Dirac estimate 2 sqrt(m^2+(c/3 k)^2) = {E_pair:.4f}")
    print(f"       fit <Pi>: omega {om_p:.4f}, amp {A_p:.3f}, decay {g_p:.4f};  "
          f"fit y: omega {om_y:.4f}, amp {A_y:.3f};  norm {nrm[-1]:.5f}  [{time.time()-t0:.0f}s]")
    out.update(zb_t=t, zb_par=par, zb_x=r[:, 0], zb_y=r[:, 1], zb_omega_pred=top[0][0],
               zb_omega_par=om_p, zb_omega_y=om_y)
    out['EPS'], out['MU'], out['m'] = EPS, MU, m

    np.savez(data_path('phase_bit_mass_s4.npz'), **out)
    print(f"saved data/phase_bit_mass_s4.npz  [{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()
