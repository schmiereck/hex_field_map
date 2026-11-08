#!/usr/bin/env python3
"""Rebuild the Bit 4 research figure from stored numerical results."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import data_path, fig_path


def main():
    with open(data_path("phase_bit_s4_open23.json"), encoding="utf-8") as f:
        report = json.load(f)
    data = np.load(data_path("phase_bit_s4_open23.npz"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#267791", "#c6772b", "#386b49", "#9a4f7a"]
    ax = axs[0, 0]
    for variant, probe, color, label in [
        ("original", 0, colors[0], "Original, slow kinetic probe"),
        ("original", 2, colors[1], "Original, fast kinetic probe"),
        ("repaired", 0, colors[2], "Repair, same slow probe"),
        ("repaired", 2, colors[3], "Repair, same fast probe")]:
        corr = data[f"{variant}_0_corr"][:, 0, probe]
        sem = data[f"{variant}_0_sem"][:, 0, probe]
        t = np.arange(len(corr))
        ax.plot(t, np.abs(corr), color=color, label=label, lw=1.5)
        ax.fill_between(t, np.maximum(1e-4, np.abs(corr) - sem), np.abs(corr) + sem, color=color, alpha=.12)
    ax.axhline(np.exp(-1), color="0.4", ls=":", lw=1)
    ax.set(xlim=(0, 64), ylim=(1e-3, 1.2), yscale="log", xlabel="Lag (ticks)",
           ylabel="Normalized correlation magnitude", title="A  Uniform kinetic fluctuations lose correlation")
    ax.legend(fontsize=8, loc="upper right")

    ax = axs[0, 1]
    k, omega = data["original_0_k"], data["original_0_omega"]
    power = data["original_0_power"][:, 1:, 5]
    power = power / power.max(axis=0, keepdims=True)
    mesh = ax.pcolormesh(k[1:], omega, power, cmap="viridis", shading="nearest", vmin=0, vmax=1)
    run = report["runs"][0]
    ax.scatter([p["k"] for p in run["density_peaks"]], [p["omega_peak"] for p in run["density_peaks"]],
               color="white", marker="x", s=24, linewidths=1)
    ax.set(ylim=(0, .42), xlabel="Visible wave number k", ylabel="Frequency (radians / tick)",
           title="B  Density spectrum, original rule, x direction")
    fig.colorbar(mesh, ax=ax, label="Power / maximum at each k", fraction=.045)

    ax = axs[1, 0]
    ax.bar([0, 1], [13, 5], color=[colors[0], colors[2]], width=.55)
    ax.axhline(5, color="0.35", ls=":", label="Number + four momentum components")
    for x, value in enumerate([13, 5]):
        ax.text(x, value + .3, str(value), ha="center", weight="bold")
    ax.set(xticks=[0, 1], xticklabels=["Original collision", "+ eight four-bit gates"], ylim=(0, 15),
           ylabel="Independent additive linear charges", title="C  Exact conservation-law diagnostic")
    ax.legend(fontsize=8, loc="upper right")

    ax = axs[1, 1]
    for i, run in enumerate(report["runs"]):
        ks = [p["k"] for p in run["density_peaks"]]
        peaks = [p["omega_peak"] for p in run["density_peaks"]]
        label = f"Original, {'xyz'[run['axis']]}" if run["variant"] == "original" else "Repair, x"
        ax.errorbar(ks, peaks, yerr=run["frequency_bin"] / 2, color=colors[i],
                    marker="o", ms=4, lw=1, ls="--" if i == 3 else "-", label=label)
    ax.set(xlabel="Visible wave number k", ylabel="Dominant density-peak frequency",
           title="D  Low-k signals approach zero frequency", ylim=(0, .33))
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Boolean D4 gas: conserved charges and collective signals", fontsize=15, weight="bold", y=.99)
    fig.text(.5, .015, "12 independent initial states per run; 512 ticks. Bands: jackknife errors. "
             "Peak bars: half a frequency bin, not confidence intervals.\n"
             "Density maxima are not tracked eigenbranches. No massive-dispersion fit is justified by this screen.",
             ha="center", va="bottom", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, .075, 1, .96))
    output = fig_path("phase_bit_s4_open23.png")
    fig.savefig(output, dpi=170)
    print(output)


if __name__ == "__main__":
    main()
