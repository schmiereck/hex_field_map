#!/usr/bin/env python3
"""
Report and figure for the two-particle Hilbert space (step 3).
Run:  python3 quantum_two_particle_figs.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_fcc_3d import (bands, coin, step3, DT_HALF, LAT_A)
from quantum_two_particle import *

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
EPS3 = 0.1


# ─── Part A: exchange statistics in the 3+1D FCC model ───────────────────────

def fcc_orbital(N, x0, sigma, eps, which=0):
    c = N // 2
    ix = np.arange(N) - c
    IX, IY, IZ = np.meshgrid(ix, ix, ix, indexing='ij')
    sub = ((IX + IY + IZ) % 2 == 0)
    X, Y, Z = IX * LAT_A, IY * LAT_A, IZ * LAT_A
    _, V, _ = bands(np.zeros(3), eps, 0.5)
    Qm, _ = np.linalg.qr(V[:, :2])
    u = Qm[:, which]
    env = np.exp(-((X - x0) ** 2 + Y ** 2 + Z ** 2) / (2 * sigma ** 2)) * sub
    psi = (env[..., None] * u[None, None, None, :]).astype(np.complex64)
    return psi / np.sqrt((np.abs(psi) ** 2).sum())


def part_A(N=61, n_steps=24):
    print("[A] exchange statistics in the 3+1D FCC model (spin 1/2 orbitals)")
    a = fcc_orbital(N, -5.0, 3.5, EPS3)
    b = fcc_orbital(N, +5.0, 3.5, EPS3)
    print("    Pauli principle, two fermions in the SAME orbital:")
    print("      max |P(x1,x2)| = %.2e   (bosons: %.2e)"
          % (pauli_residual(a), pauli_residual(a, 0.0)))
    C = coin(EPS3, 0.5).astype(np.complex64)
    flat = lambda p: p.reshape(N, N, N, 24)
    cube = lambda p: p.reshape(N, N, N, 12, 2)
    hist = []
    snap = None
    print("      t    <a|b>    coincidence:  distinguishable    fermion       boson")
    for t in range(n_steps + 1):
        fa, fb = flat(a), flat(b)
        S = abs(overlap(fa, fb))
        d = float(coincidence_density(fa, fb, np.pi / 2).sum())
        f = float(coincidence_density(fa, fb, np.pi).sum())
        bo = float(coincidence_density(fa, fb, 0.0).sum())
        hist.append((t * DT_HALF, S, d, f, bo))
        if t in (0, 6):
            print("     %4.1f  %.5f   %13.5e  %12.5e %12.5e"
                  % (t * DT_HALF, S, d, f, bo))
        if t == 6:
            snap = {th: pair_correlation_axis(fa, fb, th, axis=0)
                    for th in (0.0, np.pi / 2, np.pi)}
        if t == n_steps:
            break
        a = cube(step3(cube(a), C, 2))
        b = cube(step3(cube(b), C, 2))
    return np.array(hist), snap


# ─── Part B: genuine collisions in 1+1D ──────────────────────────────────────

def part_B(N=201, eps=0.35, n_steps=80):
    print("\n[B] a real collision: contact interaction, full two-particle state")
    set_eps_1d(eps)
    a = packet_1d(N, -30, +0.8, 6.0, eps, branch=1)
    b = packet_1d(N, +30, -0.8, 6.0, eps, branch=1)
    idx = np.arange(N)

    def run(th, U, mode="site"):
        P = two_particle_state(a, b, th).copy()
        for t in range(n_steps):
            P = step_1d(P, 0, 1); P = step_1d(P, 2, 3)
            if U:
                if mode == "site":
                    P[idx, :, idx, :] *= np.exp(1j * U)
                else:
                    for d in range(2):
                        P[idx, d, idx, d] *= np.exp(1j * U)
        return P

    P0 = run(0.0, 0.0)
    Us = np.linspace(0, 2 * np.pi, 25)
    shift, dJ, ovl = [], [], []
    for U in Us:
        PU = run(0.0, U)
        ov = complex((np.conj(P0) * PU).sum())
        shift.append(np.angle(ov)); ovl.append(abs(ov))
        dJ.append(np.abs(joint_position(PU) - joint_position(P0)).max())
    print("    1D contact scattering only shifts the phase:")
    print("      U=2.0 -> |<Psi_0|Psi_U>| changes, but max|dP(x1,x2)| = %.1e"
          % dJ[np.argmin(np.abs(Us - 2.0))])
    print("      (in 1D two-body contact scattering is integrable: the momentum")
    print("       distribution cannot change, only a phase shift appears)")

    print("    Pauli inertness of a FULL-coincidence contact:")
    inert = {}
    for name, th in (("boson", 0.0), ("fermion", np.pi)):
        base = run(th, 0.0, "full")
        vals = [np.abs(run(th, U, "full") - base).max() for U in (1.0, 2.5, 4.0)]
        inert[name] = vals
        print("      %-8s ||Psi(U)-Psi(0)|| at U=1.0/2.5/4.0 : %s"
              % (name, "  ".join("%.2e" % v for v in vals)))
    return Us, np.array(shift), np.array(dJ), np.array(ovl), inert


def part_C(N_r=120, eps=0.35, m_Q=17):
    print("\n[C] two-body bound states, by statistics "
          "(exact, relative-coordinate diagonalisation)")
    Us = np.linspace(0, 2 * np.pi, 17)
    out = {}
    for mode in ("site", "full"):
        rows = [bound_states_by_statistics(N_r, m_Q, eps, U, mode) for U in Us]
        out[mode] = np.array(rows)
        print("    mode=%-5s  boson  <|r|>min %.2f .. %.2f     fermion  %.2f .. %.2f"
              % (mode, out[mode][:, 0].min(), out[mode][:, 0].max(),
                 out[mode][:, 1].min(), out[mode][:, 1].max()))
    print("    => a site contact binds BOTH (the heading index gives fermions room);")
    print("       a full-coincidence contact binds only bosons — for fermions the")
    print("       localisation never moves from its free value.")
    return Us, out


def part_D():
    print("\n[D] colour: only the singlet survives gauge averaging")
    acc, exact = colour_average_pair(60000)
    print("    <U (x) U*> vs delta*delta/N : max error %.2e" % np.abs(acc - exact).max())
    rng = np.random.default_rng(2)
    sing = np.eye(3) / np.sqrt(3)
    oc = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    oc -= np.trace(oc) / 3 * np.eye(3)
    oc /= np.linalg.norm(oc)
    vals = {}
    for nm, chi in (("singlet", sing), ("octet", oc)):
        vals[nm] = float(np.linalg.norm(np.einsum('abcd,cd->ab', acc, chi)))
        print("    %-8s surviving norm after gauge averaging: %.6f" % (nm, vals[nm]))
    return vals


def figure(hist, snap, Us, shift, dJ, ovl, inert, Ub, bound, colour,
           fname="two_particle.png"):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))

    labels = {0.0: "boson / distinguishable", np.pi: "fermion / distinguishable"}
    Pd = snap[np.pi / 2]
    n = Pd.shape[0]
    ext = [-n // 2, n // 2, -n // 2, n // 2]
    a = ax[0, 0]
    a.imshow(Pd.T, origin='lower', cmap='magma', extent=ext, aspect='equal')
    a.plot([-n // 2, n // 2], [-n // 2, n // 2], 'c--', lw=.8)
    a.set_xlim(-22, 22); a.set_ylim(-22, 22)
    a.set_xlabel("$x_1$"); a.set_ylabel("$x_2$")
    a.set_title("$P(x_1,x_2)$, distinguishable\n(reference)", fontsize=9)
    for j, th in enumerate((np.pi, 0.0)):
        a = ax[0, j + 1]
        R = np.where(Pd > 1e-3 * Pd.max(), snap[th] / np.maximum(Pd, 1e-30), np.nan)
        im = a.imshow(R.T, origin='lower', cmap='RdBu_r', extent=ext,
                      vmin=0, vmax=2, aspect='equal')
        a.plot([-n // 2, n // 2], [-n // 2, n // 2], 'k--', lw=.8)
        a.set_xlim(-22, 22); a.set_ylim(-22, 22)
        a.set_xlabel("$x_1$")
        a.set_title("ratio " + labels[th]
                    + ("\n(Pauli hole on the diagonal)" if th == np.pi
                       else "\n(bunching on the diagonal)"), fontsize=9)
        plt.colorbar(im, ax=a, fraction=.046)

    a = ax[1, 0]
    a.plot(hist[:, 0], hist[:, 3] / hist[:, 2], 'o-', ms=4, label="fermion / distinguishable")
    a.plot(hist[:, 0], hist[:, 4] / hist[:, 2], 's-', ms=4, label="boson / distinguishable")
    a.axhline(1, color='k', lw=.7)
    a.set_xlabel("t"); a.set_ylabel("coincidence ratio")
    a.set_title("antibunching vs bunching\n(3+1D FCC, spin ½ orbitals)", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[1, 1]
    a.plot(Us, ovl, 'o-', ms=4, color='C0')
    a.set_xlabel("contact strength $U$")
    a.set_ylabel("$|\\langle\\Psi_0|\\Psi_U\\rangle|$", color='C0')
    a.set_ylim(0, 1.05)
    a2 = a.twinx()
    a2.semilogy(Us, np.maximum(dJ, 1e-12), 's-', ms=3, color='C3')
    a2.set_ylabel("max $|\\Delta P(x_1,x_2)|$", color='C3')
    a2.set_ylim(1e-8, 1e-1)
    a.set_title("1+1D contact scattering is integrable:\n"
                "a phase shift, no change of the distribution", fontsize=9)
    a.grid(alpha=.3)

    a = ax[1, 2]
    a.plot(Ub, bound["site"][:, 0], '-o', ms=5, color='C0', label="boson, site contact")
    a.plot(Ub, bound["site"][:, 1], '-s', ms=5, color='C3', label="fermion, site contact")
    a.plot(Ub, bound["full"][:, 0], '--^', ms=5, color='C9',
           label="boson, full-coincidence")
    a.plot(Ub, bound["full"][:, 1], '--v', ms=6, color='k', lw=2.2,
           label="fermion, full-coincidence")
    a.set_xlabel("contact strength $U$")
    a.set_ylabel("$\\langle|r|\\rangle$ of the most localised state")
    a.set_title("two-body bound states.  A full-coincidence contact\n"
                "cannot bind fermions at all (flat black dashed)", fontsize=9)
    a.legend(fontsize=6, ncol=2); a.grid(alpha=.3)

    fig.suptitle("Two-particle Hilbert space: exclusion is not interference — "
                 "the Pauli hole, and an interaction that fermions cannot feel",
                 y=1.01)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("\nwrote", fname)


if __name__ == "__main__":
    print("=" * 76)
    print("Step 3: the two-particle Hilbert space")
    print("=" * 76)
    hist, snap = part_A()
    Us, shift, dJ, ovl, inert = part_B()
    Ub, bound = part_C()
    colour = part_D()
    figure(hist, snap, Us, shift, dJ, ovl, inert, Ub, bound, colour)
