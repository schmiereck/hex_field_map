#!/usr/bin/env python3
"""
Figures and numerical report for the turning-phase model.
Run:  python3 quantum_hex_turning_figs.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from quantum_hex_turning import *

EPS_REF = 0.1
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


# ─── Fig 1: geometry and coin operator ───────────────────────────────────────

def fig_geometry(fname="turning_geometry.png"):
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))

    a = ax[0]
    for d in range(N_D):
        dx, dy = MOVES_PHYS[d]
        a.arrow(0, 0, dx, dy, head_width=0.09, length_includes_head=True,
                color=DIR_COLORS[d], lw=2)
        a.text(1.22 * dx, 1.22 * dy, f"d={d}\n{60*d}°", ha='center',
               va='center', color=DIR_COLORS[d], fontsize=8)
    a.add_patch(plt.Circle((0, 0), SQRT3 / 2, fill=False, ls=':', color='gray'))
    a.set_aspect('equal'); a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5)
    a.set_title("6 headings, step $\\sqrt{3}/2$, $\\Delta t=1/2$\n$c=\\sqrt{3}$")
    a.grid(alpha=.3)

    a = ax[1]
    im = a.imshow(NSTEP, cmap='RdBu_r', vmin=-3, vmax=3)
    for i in range(6):
        for j in range(6):
            a.text(j, i, f"{NSTEP[i,j]:+d}", ha='center', va='center', fontsize=8)
    a.set_xlabel("$d_{old}$"); a.set_ylabel("$d_{new}$")
    a.set_title("turn $n$ (units of 60°)\n$+3$ = 180° reversal (ambiguous)")
    plt.colorbar(im, ax=a, fraction=.046)

    for j, al in enumerate([0.0, 0.5]):
        a = ax[2 + j]
        C = coin(EPS_REF, al, "unitary")
        im = a.imshow(np.angle(C), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        for i in range(6):
            for k in range(6):
                a.text(k, i, f"{abs(C[i,k]):.2f}", ha='center', va='center',
                       fontsize=7, color='w')
        a.set_title(f"coin $\\arg C$, $\\alpha={al}$\n(numbers: $|C|$), $\\varepsilon={EPS_REF}$")
        a.set_xlabel("$d_{old}$"); a.set_ylabel("$d_{new}$")
        plt.colorbar(im, ax=a, fraction=.046)

    fig.suptitle("Turning-phase model: geometry and coin operator "
                 "$C=\\exp(i\\varepsilon G_\\alpha)$", y=1.02)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("wrote", fname)


# ─── Fig 2: closed loops, winding numbers ────────────────────────────────────

def fig_loops(L_max=9, fname="turning_loops.png"):
    counts, bad = enumerate_loops(L_max)
    print(f"  closed walks violating  sum(theta) = 360°*integer : {bad}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    a = ax[0]
    Ls = [L for L in sorted(counts) if counts[L]]
    ws = sorted({w for L in Ls for w in counts[L]})
    M = np.array([[counts[L].get(w, 0) for w in ws] for L in Ls], float)
    im = a.imshow(np.log10(M + 1), aspect='auto', cmap='viridis')
    a.set_xticks(range(len(ws))); a.set_xticklabels(ws)
    a.set_yticks(range(len(Ls))); a.set_yticklabels(Ls)
    for i in range(len(Ls)):
        for j in range(len(ws)):
            if M[i, j]:
                a.text(j, i, int(M[i, j]), ha='center', va='center',
                       fontsize=7, color='w')
    a.set_xlabel("winding number $w = \\Sigma\\theta / 360°$")
    a.set_ylabel("loop length $L$")
    a.set_title("closed walks returning to the same directed edge\n"
                "(colour: $\\log_{10}$ count)")
    plt.colorbar(im, ax=a, fraction=.046)

    a = ax[1]
    for L in Ls:
        tot = sum(counts[L].values())
        a.bar([w + 0.09 * (L - 6) for w in counts[L]],
              [c / tot for c in counts[L].values()], width=0.08, label=f"L={L}")
    a.set_xlabel("winding $w$"); a.set_ylabel("relative frequency")
    a.set_title("winding distribution\n$w=0$ ('figure eights') first appears at $L=6$")
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[2]
    al = np.linspace(0, 2, 600)
    A, tot = loop_structure_factor(counts, al)
    a.plot(al, np.abs(A) / abs(A[0]), lw=2)
    a.axvline(0.5, color='r', ls='--', label="$\\alpha=1/2$ (spinor)")
    a.axvline(1.5, color='r', ls='--')
    a.set_xlabel("$\\alpha$"); a.set_ylabel("$|A(\\alpha)| / |A(0)|$")
    a.set_title("loop structure factor\n$A=\\sum_{loops}e^{i2\\pi\\alpha w}$, period 1")
    a.legend(); a.grid(alpha=.3)

    fig.suptitle("The angle sum is quantised: $\\Sigma\\theta = 360°\\cdot w$ "
                 "— so only $\\alpha\\notin\\mathbb{Z}$ produces interference", y=1.02)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("wrote", fname)
    return counts


# ─── Fig 3: rest spectrum and mass vs alpha ──────────────────────────────────

def fig_spectrum(eps=EPS_REF, fname="turning_spectrum.png"):
    al = np.linspace(0, 2, 401)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    a = ax[0]
    Enum = np.array([rest_spectrum(eps, x) for x in al])
    Eana = np.array([rest_spectrum_analytic(eps, x) for x in al])
    for j in range(N_D):
        a.plot(al, Enum[:, j], lw=2.2, color=DIR_COLORS[j])
    a.plot(al, Eana, 'k--', lw=.8)
    a.axvline(0.5, color='r', ls=':'); a.axvline(1.5, color='r', ls=':')
    a.set_xlabel("$\\alpha$"); a.set_ylabel("rest energy $E_m(k=0)$")
    a.set_title(f"$E_m=-4\\varepsilon\\cos(\\pi(\\alpha-m)/3)$, $\\varepsilon={eps}$\n"
                "solid: numeric, dashed: analytic")
    a.grid(alpha=.3)
    print("  max |E_num - E_ana| = %.2e" % np.abs(Enum - Eana).max())

    a = ax[1]
    tol = 1e-9

    def n_distinct(x):
        E = rest_spectrum(eps, x)
        return 1 + int((np.diff(E) > tol).sum())

    nd = np.array([n_distinct(x) for x in al])
    a.plot(al, nd, lw=2, drawstyle='steps-mid')
    a.axvline(0.5, color='r', ls='--', label="$\\alpha=1/2$: 3 doublets")
    a.axvline(0.0, color='b', ls=':', label="$\\alpha=0$: 2 doublets + 2 singlets")
    a.axvline(1.0, color='b', ls=':'); a.axvline(1.5, color='r', ls='--')
    a.set_ylim(2.5, 6.5); a.set_yticks([3, 4, 5, 6])
    a.set_xlabel("$\\alpha$")
    a.set_ylabel("number of distinct rest levels")
    a.set_title("Kramers doubling: only at $\\alpha=1/2$ do all six\n"
                "levels pair up (3 doublets)")
    a.legend(fontsize=7); a.grid(alpha=.3)
    print("  distinct rest levels:  alpha=0 -> %d,  alpha=1/4 -> %d,  alpha=1/2 -> %d"
          % (n_distinct(0.0), n_distinct(0.25), n_distinct(0.5)))

    a = ax[2]
    al2 = np.linspace(0, 1, 201)
    slope = np.array([abs(band_slope_top(eps, x, 1e-3)) for x in al2])
    mres = np.array([rest_energy_top(eps, x) for x in al2])
    a.plot(al2, slope, lw=2, label="$|dE/dk|_{k\\to0}$ of top band")
    a.plot(al2, mres, lw=2, ls='--', label="rest energy $m(\\alpha)$")
    a.axhline(C_LIGHT / 2, color='g', ls=':', label="$c/2=\\sqrt{3}/2$")
    a.axvline(0.5, color='r', ls=':')
    a.set_xlabel("$\\alpha$"); a.set_title(
        "mass knob: massive for all $\\alpha$,\nmassless cone exactly at $\\alpha=1/2$")
    a.legend(fontsize=8); a.grid(alpha=.3)

    fig.suptitle("The turning phase $\\alpha$ is a flux in heading space — "
                 "and it controls the mass", y=1.02)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("wrote", fname)


# ─── Fig 4: dispersion and isotropy ──────────────────────────────────────────

def fig_dispersion(eps=EPS_REF, fname="turning_dispersion.png"):
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.4))
    ks = np.linspace(-1.6, 1.6, 321)
    kz = np.linspace(-0.25, 0.25, 401)

    for j, al in enumerate([0.0, 0.25, 0.5]):
        E = np.array([bands(k, 0.0, eps, al)[0] for k in ks])
        a = ax[0, j]
        for b in range(N_D):
            a.plot(ks, E[:, b], lw=1.3, color='0.6')
        a.plot(ks, E[:, -1], lw=2.2, color='C3', label="top ('particle') band")
        a.plot(ks, np.sqrt(C_LIGHT**2 * ks**2 + rest_energy_top(eps, al)**2),
               'k--', lw=1, label="$\\sqrt{c^2k^2+m^2}$")
        a.plot(ks, C_LIGHT * np.abs(ks), color='g', ls=':', lw=1, label="$c|k|$")
        a.set_xlabel("$k_x$"); a.set_ylabel("$E$")
        a.set_title(f"$\\alpha={al}$")
        a.legend(fontsize=7); a.grid(alpha=.3)

        Ez = np.array([bands(k, 0.0, eps, al)[0] for k in kz])
        a = ax[1, j]
        a.plot(kz, Ez[:, -1], lw=2.2, color='C3')
        a.plot(kz, Ez[:, -2], lw=2.2, color='C0')
        m0 = rest_energy_top(eps, al)
        a.plot(kz, np.sqrt(C_LIGHT**2 * kz**2 + m0**2), 'k--', lw=1,
               label="$\\sqrt{c^2k^2+m^2}$")
        a.plot(kz, m0 + (C_LIGHT / 2) * np.abs(kz), color='g', ls=':', lw=1.2,
               label="$m+\\frac{c}{2}|k|$")
        a.set_xlabel("$k_x$"); a.set_ylabel("$E$")
        a.set_title("zoom on the top pair — "
                    + ("massless cone, slope $c/2$" if al == 0.5 else "gapped, quadratic"),
                    fontsize=9)
        a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[0, 3]
    th = np.linspace(0, 2 * np.pi, 181)
    for al, kmag in [(0.0, 0.3), (0.0, 0.8), (0.5, 0.3), (0.5, 0.8)]:
        v = np.array([np.linalg.norm(
            pick_band(kmag * np.cos(t), kmag * np.sin(t), eps, al)[2]) for t in th])
        a.plot(np.degrees(th), v, lw=1.6,
               label=f"$\\alpha$={al}, |k|={kmag}  ({100*np.ptp(v)/v.mean():.1f}%)")
        print(f"  isotropy alpha={al} |k|={kmag}: |vg| spread = {np.ptp(v):.3e}"
              f" = {100*np.ptp(v)/v.mean():.2f}% of mean {v.mean():.4f}")
    a.axhline(C_LIGHT, color='k', ls='--', lw=1, label="$c=\\sqrt{3}$")
    a.set_xlabel("direction of $k$ [deg]"); a.set_ylabel("$|v_g|$")
    a.set_title("isotropy of $|v_g|$ (6-fold lattice ripple)", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[1, 3]
    kk = np.logspace(-5, np.log10(2.0), 200)
    for al in [0.0, 0.25, 0.5]:
        vv = [np.linalg.norm(group_velocity(bands(k, 0.0, eps, al)[1][:, -1]))
              for k in kk]
        a.plot(kk, vv, lw=1.8, label=f"$\\alpha$={al}")
    a.set_xscale('log')
    a.axhline(C_LIGHT, color='k', ls='--', lw=1, label="$c=\\sqrt{3}$")
    a.axhline(C_LIGHT / 2, color='g', ls=':', lw=1, label="$c/2$")
    a.set_xlabel("$|k|$"); a.set_ylabel("$|v_g|$")
    a.set_title("top-band $|v_g|$ vs $|k|$ (log): $\\to 0$ (massive)\n"
                "but $\\to c/2$ at $\\alpha=1/2$ (massless)", fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    fig.suptitle(f"Band structure, $\\varepsilon={eps}$ "
                 "(6 bands; unitary coin, $|\\lambda|=1$ exactly)", y=1.01)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("wrote", fname)


# ─── Fig 5: moving particle in arbitrary directions ──────────────────────────

def fig_motion(eps=0.5, alpha=0.5, n_steps=120, fname="turning_motion.png"):
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
    angles = [0, 15, 30, 45, 60, 90, 135, 180, 225, 270, 315]

    a = ax[0]
    rows = []
    for ang in angles:
        r = run_packet(n_steps, eps=eps, alpha=alpha, k_mag=0.8,
                       angle_deg=ang, sigma_phys=6.0, store_every=10)
        tr = com_track(r['hist'], r['cx'], r['cy'])
        t = r['times']
        vx = np.polyfit(t, tr[:, 0], 1)[0]
        vy = np.polyfit(t, tr[:, 1], 1)[0]
        a.plot(tr[:, 0], tr[:, 1], '-o', ms=2.5, lw=1.2)
        a.annotate(f"{ang}°", (tr[-1, 0], tr[-1, 1]), fontsize=7)
        rows.append((ang, r['vg'], np.array([vx, vy]), tr[-1, 2]))
    a.set_aspect('equal'); a.grid(alpha=.3)
    a.set_xlabel("x"); a.set_ylabel("y")
    a.set_title(f"centre-of-mass tracks, $|k|=0.8$\n$\\varepsilon={eps}$, "
                f"$\\alpha={alpha}$, {n_steps} steps")

    a = ax[1]
    ang_err = []
    for ang, vp, vm, nrm in rows:
        da = np.degrees(np.arctan2(vm[1], vm[0]) - np.arctan2(vp[1], vp[0]))
        da = (da + 180) % 360 - 180
        ang_err.append(da)
        print("  %4d°  vg_pred=(%+.4f,%+.4f) |v|=%.4f   measured=(%+.4f,%+.4f)"
              " |v|=%.4f  dir.err %+.3f°  norm %.8f"
              % (ang, vp[0], vp[1], np.linalg.norm(vp), vm[0], vm[1],
                 np.linalg.norm(vm), da, nrm))
    a.bar(range(len(angles)), ang_err)
    a.set_xticks(range(len(angles)))
    a.set_xticklabels([f"{x}°" for x in angles], rotation=45, fontsize=7)
    a.set_ylabel("direction error [deg]")
    a.set_title("measured drift direction vs $\\nabla_k E$\n"
                f"max |error| = {max(abs(np.array(ang_err))):.3f}°")
    a.grid(alpha=.3)

    a = ax[2]
    for dphi, st in [(0.0, '-o'), (0.5, '--s'), (1.5, ':^')]:
        r = run_packet(n_steps, eps=eps, alpha=alpha, k_mag=0.8, angle_deg=30,
                       sigma_phys=6.0, extra_phase=dphi, store_every=20)
        tr = com_track(r['hist'], r['cx'], r['cy'])
        a.plot(r['times'], np.hypot(tr[:, 0], tr[:, 1]), st, ms=4,
               label=f"$\\delta$={dphi}/step,  E={r['E']:.4f}")
    a.set_xlabel("t"); a.set_ylabel("$|\\langle r\\rangle|$")
    a.set_title("null test: a direction-independent extra\n"
                "phase $\\delta$ per step shifts $E$, moves nothing")
    a.legend(fontsize=8); a.grid(alpha=.3)

    fig.suptitle("A moving particle in an arbitrary direction: the extra phase "
                 "must depend on the step vector ($k\\cdot\\Delta r$)", y=1.02)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("wrote", fname)


# ─── Fig 6: two packets — interference without exclusion ─────────────────────

def _coarse(p):
    """Merge adjacent x-indices: only every 2nd site of the triangular
    lattice is reachable, which would otherwise show up as a checkerboard."""
    n = (p.shape[0] // 2) * 2
    return p[0:n:2, :] + p[1:n:2, :]


def _packet_at(nx, ny, cx, cy, x0, y0, k_mag, ang_deg, eps, alpha, sigma=7.0):
    """Boosted Gaussian centred at physical (x0, y0), heading ang_deg."""
    a = np.radians(ang_deg)
    kx, ky = k_mag * np.cos(a), k_mag * np.sin(a)
    _, u, vg = pick_band(kx, ky, eps, alpha)
    u = u / np.linalg.norm(u)
    IX, IY = np.meshgrid(np.arange(nx) - cx, np.arange(ny) - cy, indexing='ij')
    X, Y = IX * DX_PHYS, IY * DY_PHYS
    sub = ((IX + IY) % 2 == 0)
    env = (np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
           * np.exp(1j * (kx * X + ky * Y)) * sub)
    psi = env[:, :, None] * u[None, None, :]
    return psi / np.sqrt((np.abs(psi)**2).sum()), u, vg


def spinor_overlap_vs_angle(eps, alpha, k_mag=0.8, angs=None):
    """|<u(0)|u(theta)>| for the particle band.  Exactly 0 at 180 deg."""
    if angs is None:
        angs = np.arange(0, 181, 5.0)
    _, u0, _ = pick_band(k_mag, 0.0, eps, alpha)
    u0 = u0 / np.linalg.norm(u0)
    out = []
    for t in angs:
        a = np.radians(t)
        _, u, _ = pick_band(k_mag * np.cos(a), k_mag * np.sin(a), eps, alpha)
        u = u / np.linalg.norm(u)
        out.append(abs(u0.conj() @ u))
    return angs, np.array(out)


def fig_two_packets(eps=0.5, alpha=0.5, n_steps=90, half_open=30.0,
                    fname="turning_two_packets.png"):
    """
    Two packets crossing at an opening angle of 2*half_open.
    The evolution is linear: they pass straight through each other.
    The only trace of the encounter is an interference pattern, and it exists
    only while the internal spinors are non-orthogonal.
    """
    nx, ny, cx, cy = make_grid(n_steps, 30)
    d0, k_mag = 30.0, 0.8

    def pair(half):
        A, uA, vA = _packet_at(nx, ny, cx, cy,
                               -d0 * np.cos(np.radians(+half)),
                               -d0 * np.sin(np.radians(+half)),
                               k_mag, +half, eps, alpha)
        B, uB, vB = _packet_at(nx, ny, cx, cy,
                               -d0 * np.cos(np.radians(-half)),
                               -d0 * np.sin(np.radians(-half)),
                               k_mag, -half, eps, alpha)
        return A, B, uA, uB, vA

    A, B, uA, uB, vA = pair(half_open)
    ov = abs(uA.conj() @ uB)
    t_meet = d0 / np.linalg.norm(vA)
    print("  crossing +-%.0f deg: |<uA|uB>| = %.4f, |v|=%.3f, meeting at t=%.1f"
          % (half_open, ov, np.linalg.norm(vA), t_meet))

    C = coin(eps, alpha, "unitary")
    both, a_o, b_o = (A + B) / np.sqrt(2), A.copy(), B.copy()
    snaps = sorted({0, int(0.6 * t_meet / DT_HALF), int(t_meet / DT_HALF), n_steps})
    store = {}
    for t in range(n_steps + 1):
        if t in snaps:
            pi_ = (np.abs(both)**2).sum(-1)
            pc_ = 0.5 * ((np.abs(a_o)**2).sum(-1) + (np.abs(b_o)**2).sum(-1))
            store[t] = (pi_, pc_)
        if t < n_steps:
            both = step(both, C); a_o = step(a_o, C); b_o = step(b_o, C)

    # head-on control run, evaluated at its own meeting time
    A2, B2, uA2, uB2, vA2 = pair(90.0)          # +90 and -90 => opening 180 deg
    ov2 = abs(uA2.conj() @ uB2)
    both2, a2, b2 = (A2 + B2) / np.sqrt(2), A2.copy(), B2.copy()
    n2 = int((d0 / np.linalg.norm(vA2)) / DT_HALF)
    for _ in range(n2):
        both2 = step(both2, C); a2 = step(a2, C); b2 = step(b2, C)
    pi2 = (np.abs(both2)**2).sum(-1)
    pc2 = 0.5 * ((np.abs(a2)**2).sum(-1) + (np.abs(b2)**2).sum(-1))
    print("  head-on (180 deg): |<uA|uB>| = %.2e, max interference term = %.2e"
          % (ov2, np.abs(pi2 - pc2).max() / pc2.max()))

    fig, ax = plt.subplots(2, 5, figsize=(19, 6.6))
    xs = ((np.arange((nx // 2) * 2)[0::2] - cx) * DX_PHYS)
    ys = (np.arange(ny) - cy) * DY_PHYS
    ext = [xs[0], xs[-1], ys[0], ys[-1]]

    m_ref = None
    for j, t in enumerate(snaps[:4]):
        pi_, pc_ = store[t]
        a = ax[0, j]
        pc = _coarse(pi_)
        a.imshow(pc.T, origin='lower', aspect='equal', cmap='inferno',
                 extent=ext, vmin=0, vmax=pc.max())
        a.set_xlim(-34, 34); a.set_ylim(-24, 24); a.set_xlabel("x")
        a.set_title(f"$|\\psi_A+\\psi_B|^2$,  $t={t*DT_HALF:.1f}$", fontsize=9)
        if j == 0:
            a.set_ylabel("y")

        a = ax[1, j]
        dif = _coarse(pi_ - pc_)
        m = np.abs(dif).max() + 1e-30
        if t == int(t_meet / DT_HALF):
            m_ref, pk_ref = m, _coarse(pc_).max()
        a.imshow(dif.T, origin='lower', aspect='equal', cmap='RdBu_r',
                 extent=ext, vmin=-m, vmax=m)
        a.set_xlim(-34, 34); a.set_ylim(-24, 24); a.set_xlabel("x")
        a.set_title("interference term, %.0f%% of peak"
                    % (100 * m / (_coarse(pc_).max() + 1e-30)), fontsize=9)
        if j == 0:
            a.set_ylabel("y")

    a = ax[0, 4]
    angs, ovs = spinor_overlap_vs_angle(eps, alpha, k_mag)
    a.plot(angs, ovs, lw=2)
    a.axvline(2 * half_open, color='g', ls='--', label=f"crossing {2*half_open:.0f}°")
    a.axvline(180, color='r', ls='--', label="head-on: exactly 0")
    a.set_ylim(-0.03, 1.05)
    a.set_xlabel("opening angle between the two $k$ [deg]")
    a.set_ylabel("$|\\langle u_A|u_B\\rangle|$")
    a.set_title("internal-spinor overlap\ncontrols whether they interfere at all",
                fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[1, 4]
    dif2 = _coarse(pi2 - pc2)
    frac2 = np.abs(dif2).max() / (_coarse(pc2).max() + 1e-30)
    m2 = m_ref if m_ref else max(np.abs(dif2).max(), 1e-18)
    a.imshow(dif2.T, origin='lower', aspect='equal', cmap='RdBu_r',
             extent=ext, vmin=-m2, vmax=m2)
    a.set_xlim(-34, 34); a.set_ylim(-24, 24)
    a.set_xlabel("x"); a.set_ylabel("y")
    a.set_title("head-on control, full overlap, SAME colour scale:\n"
                "interference term only %.1f%% of peak" % (100 * frac2),
                fontsize=9)

    fig.suptitle("Two packets without exclusion: they pass straight through each "
                 "other.  Interference appears only while they overlap AND their "
                 "internal spinors are non-orthogonal.", y=1.02)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
    print("wrote", fname)


if __name__ == "__main__":
    print("=" * 74)
    print("Turning-phase model on the triangular (hex) lattice")
    print("=" * 74)

    print("\n[1] geometry / coin")
    for mode in ["unitary", "graded", "flat"]:
        C = coin(0.3, 0.5, mode)
        print(f"  {mode:8s}  unitarity error = "
              f"{np.abs(C @ C.conj().T - np.eye(6)).max():.2e}")
    fig_geometry()

    print("\n[2] closed loops / winding numbers")
    fig_loops()

    print("\n[3] rest spectrum, mass vs alpha")
    print("  cone slope at alpha=1/2 : %.6f   (c/2 = %.6f)"
          % (cone_slope(EPS_REF, 0.5), C_LIGHT / 2))
    fig_spectrum()

    print("\n[4] dispersion / isotropy")
    fig_dispersion()

    print("\n[5] moving particle")
    fig_motion()

    print("\n[6] two packets")
    fig_two_packets()
    print("\ndone.")
