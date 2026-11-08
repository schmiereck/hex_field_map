#!/usr/bin/env python3
"""Plot the resolved hidden-mode diagnostic from stored results."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import data_path, fig_path


def main():
    with open(data_path("phase_bit_s4_open24.json"), encoding="utf-8") as f:
        report = json.load(f)
    runs = {r["hidden_period"]: r for r in report["runs"]}
    if set(runs) != {16, 32, 64}:
        raise ValueError("complete W=16,32,64 main run required")
    data = np.load(data_path("phase_bit_s4_open24.npz"))
    colors = {16: "#b77432", 32: "#357c97", 64: "#44734c"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    corr = data["w64_kinetic_corr"]
    t = np.arange(len(corr))
    for j in range(19):
        ax.plot(t, np.abs(corr[:, j]), color=".65", alpha=.45, lw=.8)
    slowest = max(runs[64]["kinetic"]["modes"], key=lambda m: m["first_e_folding_crossing_ticks"] or 10000)["mode"]
    ax.plot(t, abs(corr[:, slowest]), color=colors[64], lw=1.8, label="Slowest of the 19 tested directions")
    ax.axhline(np.exp(-1), color=".35", ls=":", lw=1)
    ax.set(xlim=(0, 120), ylim=(1e-3, 1.2), yscale="log", xlabel="Lag (ticks)",
           ylabel="Normalized correlation magnitude", title="A  Complete linear kinetic basis at k=q=0")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for w, run in sorted(runs.items()):
        j = next(j for j, row in enumerate(run["density_modes"]) if row["mx"] == 0 and row["mw"] == 1)
        omega = data[f"w{w}_omega"]
        power = data[f"w{w}_density_power"][:, j]
        ax.plot(omega, power / power.max(), color=colors[w], label=f"W={w}, q=2π/{w}")
    ax.set(xlim=(0, .65), ylim=(0, 1.08), xlabel="Frequency (radians / tick)",
           ylabel="Power / maximum of each spectrum", title="B  Hidden-direction density signals")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for w, run in sorted(runs.items()):
        rows = [r for r in run["density_modes"] if r["mx"] == 0 and r["mw"] > 0 and r["q"] < .9]
        ax.errorbar([r["q"] for r in rows], [r["omega_peak"] for r in rows],
                    yerr=run["frequency_bin"] / 2, marker="o", ls="none", color=colors[w],
                    label=f"W={w}; m=1,2,4")
    q = np.linspace(0, .86, 150)
    ax.plot(q, q / np.sqrt(2), color=".25", ls="--", lw=1, label="4D sound reference q/√2")
    ax.plot(q, q, color=".6", ls=":", lw=1, label="Free streaming: q")
    ax.set(xlim=(0, .87), ylim=(0, .9), xlabel="Hidden wave number q", ylabel="Peak frequency at visible k=0",
           title="C  Equal q in different compact boxes")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    run = runs[64]
    rows = [r for r in run["density_modes"] if r["mw"] == 1]
    x = np.array([r["kx"] for r in rows])
    q = 2 * np.pi / 64
    for v in (-1, 1):
        ax.plot(x, abs(q + v * x), color=".8", lw=1, ls=":")
    ax.axhline(q, color=".8", ls=":", lw=1)
    ax.plot(x, np.sqrt((x*x + q*q) / 2), color=".25", ls="--", label="4D sound reference √[(k²+q²)/2]")
    ax.errorbar(x, [r["omega_peak"] for r in rows], yerr=run["frequency_bin"] / 2,
                color=colors[64], marker="o", ls="none", label="Measured density maxima, W=64, m=1")
    ax.set(xlabel="Signed visible wave number k", ylabel="Frequency (radians / tick)",
           ylim=(0, .27), title="D  Fixed hidden wave number, varying visible k")
    ax.legend(fontsize=8)

    fig.suptitle("Boolean D4 gas: all kinetic directions and resolved hidden modes", fontsize=14, weight="bold", y=.99)
    fig.text(.5, .018, "12 prepared states per box; 768 ticks. Kinetic probes learned on 6 runs and tested on the other 6.\n"
             "Peak bars show half a frequency bin, not confidence intervals. Maxima are not tracked eigenbranches. "
             "The sound reference assumes local equilibrium.", ha="center", va="bottom", fontsize=8, color=".35")
    fig.tight_layout(rect=(0, .075, 1, .96))
    destination = fig_path("phase_bit_s4_open24.png")
    fig.savefig(destination, dpi=145)
    print(destination)


if __name__ == "__main__":
    main()
