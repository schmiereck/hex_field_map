#!/usr/bin/env python3
"""Bit test 5: all kinetic directions and resolved compact-space modes.

The microscopic kernel is an exactly equivalent uint32 packing of the
24 Boolean slots of RepairedBitGas. Eight padding bits remain zero. FFTs,
nullspaces, fitted projections and spectra are EXTERNAL ANALYSIS only.

Run: python src/phase_bit_s4_open24.py
Fast kernel validation: python src/phase_bit_s4_open24.py --validate-only
"""

import argparse
import json
import time

import numpy as np
from scipy.linalg import eig, null_space

from phase_bit_s4_open23 import RepairedBitGas, temporal_statistics, ensemble_statistics, peak_summary
from paths import data_path


class PackedD4:
    """Same Boolean rule, one 24-bit word per physical D4 node."""
    def __init__(self, shape):
        self.reference = RepairedBitGas(shape)
        self.shape = tuple(shape)
        self.roots = self.reference.roots
        self.coords = np.array(np.nonzero(self.reference.valid)).T
        self.nodes = len(self.coords)
        lookup = np.full(shape, -1, dtype=np.int32)
        lookup[self.reference.valid] = np.arange(self.nodes, dtype=np.int32)
        self.sources = {}
        for sign in (-1, 1):
            shifted = (self.coords[None, ...] + sign * self.roots[:, None, :]) % np.array(shape)
            self.sources[sign] = lookup[tuple(shifted[..., a] for a in range(4))]
            assert np.all(self.sources[sign] >= 0)
        self.groups = self.reference.groups
        self.extra = self.reference.extra_gates

    def pack(self, state):
        bits = state[self.reference.valid].astype(np.uint32)
        return np.sum(bits << np.arange(24, dtype=np.uint32), axis=1, dtype=np.uint32)

    def unpack(self, words):
        out = self.reference.empty()
        out[self.reference.valid] = (words[:, None] >> np.arange(24, dtype=np.uint32)) & np.uint32(1)
        return out

    def collide(self, words, inverse=False):
        out = words.copy()
        if inverse:
            out = self._scatter(out, inverse=True)
        for group in reversed(self.groups) if inverse else self.groups:
            shifts = [2 * a for a in group]
            a, b, c = [(out >> s) & np.uint32(3) for s in shifts]
            eligible = (((a ^ (a >> 1)) | (b ^ (b >> 1)) | (c ^ (c >> 1))) & np.uint32(1)) == 0
            values = (b, c, a) if inverse else (c, a, b)
            rotated = (values[0] << shifts[0]) | (values[1] << shifts[1]) | (values[2] << shifts[2])
            mask = np.uint32(sum(3 << s for s in shifts))
            out = np.where(eligible, (out & ~mask) | rotated, out)
        return out if inverse else self._scatter(out)

    def _scatter(self, words, inverse=False):
        out = words.copy()
        for gate in reversed(self.extra) if inverse else self.extra:
            a, b, c, d = [out >> j for j in gate]
            eligible = (((a ^ b) | (c ^ d)) & np.uint32(1)) == 0
            eligible &= ((a ^ c) & np.uint32(1)) != 0
            mask = np.uint32(sum(1 << j for j in gate))
            out ^= np.where(eligible, mask, np.uint32(0))
        return out

    def stream(self, words, inverse=False):
        out = np.zeros_like(words)
        for d, source in enumerate(self.sources[1 if inverse else -1]):
            out |= words[source] & np.uint32(1 << d)
        return out

    def tick(self, words, inverse=False):
        return self.collide(self.stream(words, inverse=True), inverse=True) if inverse else self.stream(self.collide(words))

    def counts_xw(self, words):
        """Exact heading counts, summed only over y,z. No Fourier values."""
        nx, ny, nz, nw = self.shape
        grid = words.reshape(nx, ny * nz, nw // 2)
        parity = (np.arange(ny)[:, None] + np.arange(nz)[None, :]).ravel() % 2
        shifts = np.arange(24, dtype=np.uint32)
        even = np.sum((grid[:, parity == 0, :, None] >> shifts) & np.uint32(1), axis=1, dtype=np.int16)
        odd = np.sum((grid[:, parity == 1, :, None] >> shifts) & np.uint32(1), axis=1, dtype=np.int16)
        counts = np.empty((nx, nw, 24), dtype=np.int16)
        x_even = (np.arange(nx) % 2 == 0)[:, None, None]
        counts[:, ::2] = np.where(x_even, even, odd)
        counts[:, 1::2] = np.where(x_even, odd, even)
        return counts


def validate_kernel():
    checked = 0
    rng = np.random.default_rng(2405)
    for shape in ((4, 4, 4, 4), (8, 4, 4, 8)):
        packed = PackedD4(shape)
        for _ in range(4):
            start = rng.integers(0, 2, size=shape + (24,), dtype=np.uint8).astype(bool)
            start[~packed.reference.valid] = False
            state, words = start.copy(), packed.pack(start)
            for _ in range(16):
                state = packed.reference.tick(state)
                words = packed.tick(words)
                assert words.dtype == np.uint32 and not np.any(words >> 24)
                assert np.array_equal(packed.unpack(words), state)
                assert np.array_equal(packed.counts_xw(words), np.count_nonzero(state, axis=(1, 2)))
                checked += 1
            for _ in range(16):
                words = packed.tick(words, inverse=True)
            assert np.array_equal(packed.unpack(words), start)
    return {"forward_ticks_against_boolean_reference": checked, "bit_exact_round_trips": 8,
            "counts_xw_exact": True, "variable_bits_per_node": 24, "constant_zero_padding_bits": 8}


def kinetic_basis(roots):
    charges = np.vstack((np.ones(24), roots.T))
    basis = null_space(charges)
    assert basis.shape == (24, 19)
    error = max(np.max(np.abs(charges @ basis)), np.max(np.abs(basis.T @ basis - np.eye(19))))
    assert error < 1e-12
    return basis, float(error)


def screen(corr, power, omega):
    result = peak_summary(power, omega)
    period = int(np.ceil(2 * np.pi / result["omega_peak"])) if result["omega_peak"] else None
    amplitude = float(abs(corr[period])) if period is not None and period < len(corr) else None
    result.update(period_ticks=period, correlation_after_one_period=amplitude,
                  passes_screen=bool(result["omega_peak"] > 4 * omega[1]
                                     and result["peak_over_width"] >= 3 - 1e-12
                                     and amplitude is not None and amplitude > np.exp(-1)))
    return result


def analyze_kinetic(raw, roots, steps):
    """All 19 directions; learn a probe basis on one half, test on the other."""
    basis, error = kinetic_basis(roots)
    y = raw @ basis
    split = len(y) // 2
    past = y[:split, :-1].reshape(-1, 19)
    future = y[:split, 1:].reshape(-1, 19)
    c0 = past.T @ past / len(past)
    c10 = future.T @ past / len(past)
    projection = c10 @ np.linalg.inv(c0)
    values, left = eig(projection, left=True, right=False)
    ids = sorted(range(19), key=lambda j: (round(abs(np.angle(values[j])), 10), -abs(values[j])))
    values, left = values[ids], left[:, ids]
    left /= np.linalg.norm(left, axis=0)
    assert np.linalg.matrix_rank(left) == 19
    acs, pss = [], []
    for run in y[split:]:
        z = (run @ left.conj())[:, None, :]
        ac, ps = temporal_statistics(z, min(384, steps // 2))
        acs.append(ac)
        pss.append(ps)
    corr, sem, power = ensemble_statistics(acs, pss)
    corr, sem, power = corr[:, 0], sem[:, 0], power[:, 0]
    omega = np.arange((steps + 1) // 2 + 1) * 2 * np.pi / (steps + 1)
    modes = []
    for j, value in enumerate(values):
        first = np.where(np.abs(corr[1:, j]) < np.exp(-1))[0]
        modes.append({"mode": j, "training_eigenvalue": [float(value.real), float(value.imag)],
                      "training_phase": float(np.angle(value)), "training_modulus": float(abs(value)),
                      "first_e_folding_crossing_ticks": int(first[0] + 1) if len(first) else None,
                      "C1": [float(corr[1, j].real), float(corr[1, j].imag)],
                      "C1_jackknife_error": float(sem[1, j]),
                      **screen(corr[:, j], power[:, j], omega)})
    return {"dimension": 19, "basis_residual": error, "training_runs": split,
            "held_out_runs": len(y) - split, "probe_matrix_rank": 19,
            "probe_condition_number": float(np.linalg.cond(left)),
            "warning": "finite-sample one-step projection is analysis, not a microscopic rule",
            "modes": modes}, {"kinetic_corr": corr, "kinetic_sem": sem, "kinetic_power": power,
                               "kinetic_probes": basis @ left, "omega": omega}


def analyze_density(density, current, cases, steps):
    acs, pss, cross = [], [], []
    nt = steps + 1
    window = np.hanning(nt)
    for rho, flow in zip(density, current):
        ac, ps = temporal_statistics(rho[:, :, None], min(384, steps // 2))
        acs.append(ac)
        pss.append(ps)
        fr = np.fft.fft((rho - rho.mean(0)) * window[:, None], axis=0)
        fj = np.fft.fft((flow - flow.mean(0)) * window[:, None], axis=0)
        cross.append((fr, fj))
    corr, sem, power = ensemble_statistics(acs, pss)
    corr, sem, power = corr[:, :, 0], sem[:, :, 0], power[:, :, 0]
    omega = np.arange(nt // 2 + 1) * 2 * np.pi / nt
    rows = []
    for j, case in enumerate(cases):
        row = dict(case)
        if case["mx"] == 0 and case["mw"] == 0:
            assert np.max(np.abs(corr[:, j] - 1)) < 1e-10
            row.update(conserved_zero_mode=True, passes_screen=False)
        else:
            row.update(screen(corr[:, j], power[:, j], omega))
            index = int(np.argmax(power[:, j]))
            r = np.array([fr[index, j] for fr, _ in cross])
            f = np.array([fj[index, j] for _, fj in cross])
            rf = np.mean(r * f.conj())
            denom = float(np.mean(abs(r) ** 2) * np.mean(abs(f) ** 2))
            row["density_current_coherence"] = float(abs(rf) ** 2 / denom) if denom > 0 else None
            row["density_current_cross_phase"] = float(np.angle(rf))
            row["passes_propagating_screen"] = bool(row["passes_screen"] and row["density_current_coherence"] >= .8)
        rows.append(row)
    return rows, {"density_corr": corr, "density_sem": sem, "density_power": power}


def run_period(period, length, steps, samples):
    kernel = PackedD4((length, 4, 4, period))
    mxs, mws = list(range(-3, 4)), [0, 1, -1, 2, -2, 4, -4]
    cases = [{"mx": mx, "mw": mw, "kx": mx * 2 * np.pi / length,
              "q": mw * 2 * np.pi / period} for mx in mxs for mw in mws]
    ix = np.array([c["mx"] % length for c in cases])
    iw = np.array([c["mw"] % period for c in cases])
    density, current, raw = [], [], []
    baseline = 4  # eight physical y,z nodes per (x,w), at half filling.
    started = time.time()
    zero = next(j for j, c in enumerate(cases) if c["mx"] == c["mw"] == 0)
    signs = [(j, next(h for h, other in enumerate(cases) if other["mx"] == 0 and other["mw"] == -c["mw"]))
             for j, c in enumerate(cases) if c["mx"] == 0 and c["mw"] > 0]
    wavevectors = np.array([[c["kx"], 0, 0, c["q"]] for c in cases])
    norm = np.linalg.norm(wavevectors, axis=1)
    directions = wavevectors / np.where(norm > 0, norm, 1)[:, None]
    weights = directions @ kernel.roots.T
    conservation_error = 0
    for seed in range(samples):
        rng = np.random.default_rng(240500 + period * 1000 + seed)
        bits = rng.integers(0, 2, size=(kernel.nodes, 24), dtype=np.uint8).astype(np.uint32)
        words = np.sum(bits << np.arange(24, dtype=np.uint32), axis=1, dtype=np.uint32)
        rho = np.empty((steps + 1, len(cases)), complex)
        flow = np.empty_like(rho)
        origin = np.empty((steps + 1, 24), float)
        initial_charge = None
        for tick in range(steps + 1):
            counts = kernel.counts_xw(words)
            totals = counts.sum(axis=(0, 1), dtype=np.int64)
            charge = np.concatenate(([totals.sum()], totals @ kernel.roots))
            if initial_charge is None:
                initial_charge = charge
            conservation_error = max(conservation_error, int(np.max(np.abs(charge - initial_charge))))
            assert conservation_error == 0 and not np.any(words >> 24)
            fourier = np.fft.fft2(counts.astype(np.int16) - baseline, axes=(0, 1))[ix, iw]
            rho[tick] = fourier.sum(axis=1)
            flow[tick] = np.sum(fourier * weights, axis=1)
            origin[tick] = fourier[zero].real
            for pos, neg in signs:
                assert np.max(abs(fourier[pos] - fourier[neg].conj())) < 1e-8
            if tick < steps:
                words = kernel.tick(words)
        density.append(rho)
        current.append(flow)
        raw.append(origin)
        print(f"W={period}: sample {seed+1}/{samples}, {time.time()-started:.1f}s", flush=True)
    kinetic, kd = analyze_kinetic(np.asarray(raw), kernel.roots, steps)
    acoustic, dd = analyze_density(np.asarray(density), np.asarray(current), cases, steps)
    # Straight free-streaming phase branches are an exact control, not
    # a massive-dispersion mechanism. No stochastic simulation is needed.
    free = [[float(c["kx"] * v[0] + c["q"] * v[3]) for v in kernel.roots] for c in cases]
    report = {"hidden_period": period, "shape": list(kernel.shape), "physical_nodes": kernel.nodes,
              "steps": steps, "samples": samples, "frequency_bin": 2 * np.pi / (steps + 1),
              "physical_conservation_error": conservation_error, "conjugate_hidden_modes_checked": True,
              "kinetic": kinetic, "density_modes": acoustic, "free_stream_unfolded_phases": free}
    return report, {**kd, **dd, "wavevectors": np.array([[c["kx"], c["q"]] for c in cases])}


def dispersion_fits(runs):
    fits = []
    for run in runs:
        for mw in (1, -1, 2, -2, 4, -4):
            rows = [r for r in run["density_modes"] if r["mw"] == mw and abs(r["mx"]) <= 2]
            accepted = all(r.get("passes_propagating_screen", False) for r in rows)
            result = {"hidden_period": run["hidden_period"], "mw": mw,
                      "five_peaks_pass_screen": accepted, "fit": None}
            if accepted:
                x = np.array([r["kx"] for r in rows])
                w = np.array([r["omega_peak"] for r in rows])
                c, b, a = np.polyfit(x, w * w, 2)
                predicted_squared = a + b * x + c * x * x
                if a > 0 and np.all(predicted_squared > 0):
                    result["fit"] = {"omega0": float(np.sqrt(a)), "vg0": float(b / (2 * np.sqrt(a))),
                                     "k_squared_coefficient": float(c),
                                     "max_frequency_residual_bins": float(np.max(abs(np.sqrt(predicted_squared) - w)) / run["frequency_bin"]),
                                     "warning": "finite-bin diagnostic, not a mass derivation"}
            fits.append(result)
    return fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--tag", default="phase_bit_s4_open24")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validation = validate_kernel()
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return
    if args.length < 32 or args.length % 2 or args.steps < 128 or args.samples < 6 or args.samples % 2:
        parser.error("even length >=32, steps >=128, even samples >=6 required")
    if any(p < 16 or p % 2 for p in args.periods):
        parser.error("even hidden periods >=16 required")
    output = {"schema": 1, "kernel_validation": validation, "runs": [], "filling": "1/2",
              "time": "separate synchronous tick", "analysis_only_complex_numbers": True,
              "screen": "peak >4 bins, peak/width >=3, |C(one period)| >1/e; density-current coherence >=0.8"}
    arrays = {}
    for period in args.periods:
        report, data = run_period(period, args.length, args.steps, args.samples)
        output["runs"].append(report)
        output["dispersion_fits"] = dispersion_fits(output["runs"])
        for key, value in data.items():
            if value.dtype.kind == "c":
                value = value.astype(np.complex64)
            elif value.dtype.kind == "f":
                value = value.astype(np.float32)
            arrays[f"w{period}_{key}"] = value
        with open(data_path(args.tag + ".json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            f.write("\n")
        np.savez_compressed(data_path(args.tag + ".npz"), **arrays)
    print("saved " + data_path(args.tag + ".json"), flush=True)


if __name__ == "__main__":
    main()
