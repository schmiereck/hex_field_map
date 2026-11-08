#!/usr/bin/env python3
"""Bit test 4: conserved charges, collective modes, and a conservative repair.

Run: python src/phase_bit_s4_open23.py
The two microscopic kernels use Boolean gates only. Seeded random bits are
external equilibrium test inputs. All complex values, eigenvalues, spectra,
and statistical errors belong to analysis, never to the update rule.

An exact one-step linear projection of the ORIGINAL collision is calculated
by enumerating 4096 routing masks. Iterating that matrix is a closure
approximation, NOT the actual Boolean evolution. Actual time correlations
are measured separately on microscopic trajectories.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
import json
import time

import numpy as np
from scipy.linalg import eig

from phase_bit_s4_open22 import BitGas
from paths import data_path


def insert_basis(basis, vector):
    """Exact rational row elimination, for small conservation-law matrices."""
    row = [Fraction(int(v)) for v in vector]
    for pivot, existing in basis:
        a = row[pivot]
        if a:
            row = [v - a * w for v, w in zip(row, existing)]
    pivot = next((j for j, v in enumerate(row) if v), None)
    if pivot is None:
        return False
    a = row[pivot]
    basis.append((pivot, [v / a for v in row]))
    basis.sort(key=lambda item: item[0])
    return True


def exact_rank(matrix):
    basis = []
    for row in matrix:
        insert_basis(basis, row)
    return len(basis)


def reaction(gate):
    v = np.zeros(24, dtype=int)
    v[list(gate[:2])] = -1
    v[list(gate[2:])] = 1
    return v


def select_repair(gas):
    """Add independent nonzero-total-momentum two-body scatterings.

    Each gate swaps 1100 <-> 0011 in four slots with v0+v1=v2+v3.
    Selection is an exact finite geometric computation done once, not a
    dynamically recomputed real coupling. Eight independent gates suffice.
    """
    basis = []
    for group in gas.groups:
        for a, b in zip(group, group[1:] + group[:1]):
            insert_basis(basis, reaction((2 * a, 2 * a + 1, 2 * b, 2 * b + 1)))
    assert len(basis) == 11
    classes = defaultdict(list)
    for a, b in combinations(range(24), 2):
        classes[tuple(int(v) for v in gas.roots[a] + gas.roots[b])].append((a, b))
    selected = []
    for momentum, pairs in sorted(classes.items()):
        if not any(momentum):
            continue
        for first, second in combinations(pairs, 2):
            gate = first + second
            if insert_basis(basis, reaction(gate)):
                selected.append(gate)
            if len(basis) == 19:
                assert len(selected) == 8
                return selected
    raise AssertionError("D4 pair collisions did not reach physical rank 19")


def scatter_gate(bits):
    """Self-inverse, number/momentum-conserving four-bit scattering."""
    if bits.dtype != np.bool_ or bits.shape[-1] != 4:
        raise ValueError("four Boolean slots required")
    eligible = (bits[..., 0] == bits[..., 1]) & (bits[..., 2] == bits[..., 3]) & (bits[..., 0] != bits[..., 2])
    return bits ^ eligible[..., None]


class RepairedBitGas(BitGas):
    def __init__(self, shape=(4, 4, 4, 4)):
        super().__init__(shape)
        self.extra_gates = select_repair(self)

    def collide(self, state, inverse=False):
        out = state.copy() if inverse else super().collide(state)
        for gate in reversed(self.extra_gates) if inverse else self.extra_gates:
            out[..., gate] = scatter_gate(out[..., gate])
        return super().collide(out, inverse=True) if inverse else out


def conservation_tests():
    old, new = BitGas(), RepairedBitGas()
    local = np.array(list(product((False, True), repeat=4)), dtype=bool)
    transformed = scatter_gate(local)
    assert len({s.tobytes() for s in transformed}) == 16
    assert np.array_equal(scatter_gate(transformed), local)
    assert np.array_equal(local.sum(1), transformed.sum(1))
    for gate in new.extra_gates:
        assert np.array_equal(local.astype(int) @ old.roots[list(gate)],
                              transformed.astype(int) @ old.roots[list(gate)])

    # These small-input witnesses bound the rank from below. Known exact
    # conserved quantities bound it from above, proving the full 24-bit
    # collision's complete additive-linear invariant dimension without
    # pretending that only the individual subgates were tested.
    occupations = [p for n in range(4) for p in combinations(range(24), n)]
    states = np.zeros((len(occupations), 24), dtype=bool)
    for j, p in enumerate(occupations):
        states[j, list(p)] = True
    physical = np.vstack((np.ones(24, dtype=int), old.roots.T))
    q_axes = np.zeros((12, 24), dtype=int)
    for a in range(12):
        q_axes[a, 2 * a], q_axes[a, 2 * a + 1] = 1, -1
    old_charges = np.vstack((np.ones(24, dtype=int), q_axes))
    result = {"extra_gates": [list(g) for g in new.extra_gates], "local_gate_states_exhausted": 16,
              "low_occupation_states_checked": len(states), "physical_charge_rank": exact_rank(physical)}
    for name, gas, charges, expected in [("original", old, old_charges, 13), ("repaired", new, physical, 5)]:
        out = gas.collide(states)
        diff = out.astype(int) - states.astype(int)
        assert np.all(diff @ charges.T == 0)
        assert np.array_equal(gas.collide(out, inverse=True), states)
        witness_rank = exact_rank(diff)
        assert witness_rank == 24 - expected
        result[name] = {"actual_full_tick_collision_change_rank": witness_rank,
                        "additive_linear_charge_dimension_exact": expected,
                        "extra_above_number_and_4momentum": expected - 5}

    rng = np.random.default_rng(2304)
    for gas in (old, new):
        for _ in range(8):
            start = rng.integers(0, 2, size=gas.shape + (24,), dtype=np.uint8).astype(bool)
            start[~gas.valid] = False
            s, inv = start.copy(), gas.invariants(start)
            for _ in range(16):
                s = gas.tick(s)
                gas.check(s)
                assert gas.invariants(s) == inv
            for _ in range(16):
                s = gas.tick(s, inverse=True)
            assert np.array_equal(s, start)
    result["dense_round_trips_per_variant"] = 8
    result["physical_invariant_error"] = 0
    return result


def exact_one_step(gas):
    """At half filling, p_a=n_++n_--1, q_a=n_+-n_-.

    q is unchanged by the original collision. Eligibility of each pair
    (00/11 versus 01/10) is unchanged too. Given its 12-bit mask, p is
    just permuted. Signs of eligible p values are independent and fair
    in the stationary uniform ensemble. Enumerate masks, not real weights.
    """
    counts = np.zeros((12, 12), dtype=np.int64)
    for mask in range(1 << 12):
        active = np.array([bool(mask >> a & 1) for a in range(12)])
        route = np.arange(12)
        for group in gas.groups:
            ids = list(group)
            if np.all(active[ids]):
                route[ids] = route[ids][[2, 0, 1]]
        for destination, source in enumerate(route):
            if active[source]:
                counts[destination, source] += 1
    assert np.all(counts.sum(0) == 2048) and np.all(counts.sum(1) == 2048)
    matrix = counts / 2048  # ANALYSIS ONLY, not a microscopic transition.
    values, left = eig(matrix, left=True, right=False)
    selected = sorted(np.where(np.angle(values) > 1e-8)[0], key=lambda j: np.angle(values[j]))
    assert len(selected) == 5
    probes = left[:, selected]
    probes /= np.linalg.norm(probes, axis=0)
    modes = []
    for j in selected:
        value = values[j]
        omega, gamma = float(np.angle(value)), float(-np.log(abs(value)))
        modes.append({"lambda_real": float(value.real), "lambda_imag": float(value.imag),
                      "omega": omega, "gamma": gamma, "e_folding_ticks": 1 / gamma,
                      "cycles_per_e_folding": omega / (2 * np.pi * gamma)})
    return {"routing_masks_exhausted": 4096, "denominator": 2048,
            "pair_response_numerators": counts.tolist(), "kinetic_reference_modes": modes,
            "warning": "exact at one step; iterated matrix is a memoryless closure approximation"}, matrix, probes


def temporal_statistics(z, maxlag):
    """Per-run unbiased time-origin autocorrelations and Hann periodograms."""
    nt = len(z)
    ft = np.fft.fft(z, n=2 * nt, axis=0)
    corr = np.fft.ifft(ft * ft.conj(), axis=0)[:maxlag + 1]
    corr /= (nt - np.arange(maxlag + 1))[:, None, None]
    window = np.hanning(nt)
    centered = z - z.mean(axis=0, keepdims=True)
    ft = np.fft.fft(centered * window[:, None, None], axis=0)
    power = (ft * ft.conj()).real / np.sum(window * window)
    ids = np.arange(nt // 2 + 1)
    power = (power[ids] + power[(-ids) % nt]) / 2
    return corr, power


def ensemble_statistics(correlations, powers):
    corr = np.asarray(correlations)
    total = corr.sum(axis=0)
    normalized = total / total[0].real[None, ...]
    # Delete-one-run jackknife: independent initial states are the sampling
    # units, not the strongly correlated time origins inside one trajectory.
    n = len(corr)
    jack = (total[None, ...] - corr) / (total[0].real[None, None, ...] - corr[:, 0].real[:, None, ...])
    center = jack.mean(axis=0)
    sem = np.sqrt((n - 1) / n * np.sum(np.abs(jack - center) ** 2, axis=0))
    return normalized, sem, np.mean(powers, axis=0)


def peak_summary(power, frequencies):
    index = int(np.argmax(power))
    height = power[index]
    lo, hi = index, index
    while lo > 0 and power[lo - 1] >= height / 2:
        lo -= 1
    while hi + 1 < len(power) and power[hi + 1] >= height / 2:
        hi += 1
    width = max(frequencies[1], frequencies[hi] - frequencies[lo])
    return {"omega_peak": float(frequencies[index]), "half_height_width": float(width),
            "peak_over_width": float(frequencies[index] / width)}


def simulate(variant, axis, length, steps, samples, matrix, kinetic_probes, maxmode=6):
    shape = [4] * 4
    shape[axis] = length
    gas = (BitGas if variant == "original" else RepairedBitGas)(tuple(shape))
    summed = tuple(a for a in range(4) if a != axis)
    plane_nodes = int(gas.valid.sum()) // length
    probes = np.column_stack((kinetic_probes, np.ones(12) / np.sqrt(12)))
    correlations, powers = [], []
    integer_q_error = 0
    maxlag = min(128, steps // 2)
    started = time.time()
    for seed in range(samples):
        rng = np.random.default_rng(230400 + axis * 1000 + seed)
        state = rng.integers(0, 2, size=gas.shape + (24,), dtype=np.uint8).astype(bool)
        state[~gas.valid] = False
        counts = np.empty((steps + 1, length, 24), dtype=np.int16)
        for tick in range(steps + 1):
            counts[tick] = np.count_nonzero(state, axis=summed)
            if tick < steps:
                state = gas.tick(state)
        gas.check(state)
        direction_totals = counts.sum(axis=1, dtype=np.int64)
        physical = np.column_stack((direction_totals.sum(1), direction_totals @ gas.roots))
        assert np.all(physical == physical[0])
        q = direction_totals[:, ::2] - direction_totals[:, 1::2]
        q_error = int(np.max(np.abs(q - q[0])))
        integer_q_error = max(integer_q_error, q_error)
        if variant == "original":
            assert q_error == 0
        p = counts[..., ::2].astype(np.int32) + counts[..., 1::2] - plane_nodes
        fourier = np.fft.fft(p, axis=1)[:, :maxmode + 1]
        z = fourier @ probes.conj()
        ac, ps = temporal_statistics(z, maxlag)
        correlations.append(ac)
        powers.append(ps)
        print(f"{variant} axis={axis}, sample={seed+1}/{samples}, elapsed={time.time()-started:.1f}s", flush=True)
    corr, sem, power = ensemble_statistics(correlations, powers)
    # Density at k=0 is conserved; roundoff is irrelevant to the other modes.
    assert np.max(np.abs(corr[:, 0, 5] - 1)) < 1e-10
    k = np.arange(maxmode + 1) * (2 * np.pi / length)
    frequencies = np.arange((steps + 1) // 2 + 1) * (2 * np.pi / (steps + 1))
    density_peaks = [dict(mode=m, k=float(k[m]), **peak_summary(power[:, m, 5], frequencies))
                     for m in range(1, maxmode + 1)]
    kinetic = []
    for j in range(5):
        first = np.where(np.abs(corr[1:, 0, j]) < np.exp(-1))[0]
        lag = int(first[0] + 1) if len(first) else None
        summary = peak_summary(power[:, 0, j], frequencies)
        period = int(np.ceil(2 * np.pi / summary["omega_peak"])) if summary["omega_peak"] else None
        resolved = summary["omega_peak"] > 4 * frequencies[1]
        narrow = summary["peak_over_width"] >= 3
        lasts = period is not None and period <= maxlag and abs(corr[period, 0, j]) > np.exp(-1)
        kinetic.append({"probe": j, "first_e_folding_crossing_ticks": lag,
                        "C1_real": float(corr[1, 0, j].real), "C1_imag": float(corr[1, 0, j].imag),
                        "C1_jackknife_error": float(sem[1, 0, j]),
                        "abs_C16": float(abs(corr[16, 0, j])), "C16_jackknife_error": float(sem[16, 0, j]),
                        **summary, "passes_resolved_narrow_one_period_screen": bool(resolved and narrow and lasts)})
    result = {"variant": variant, "axis": axis, "shape": shape, "physical_nodes": int(gas.valid.sum()),
              "samples": samples, "steps": steps, "maxlag": maxlag, "frequency_bin": float(frequencies[1]),
              "physical_conservation_error": 0, "axis_charge_max_change": integer_q_error,
              "kinetic_k0_probes": kinetic, "density_peaks": density_peaks,
              "scope": "finite sample/probe screen, not an exhaustive spectral exclusion"}
    if variant == "original":
        expected = np.array([kinetic_probes[:, j].conj() @ matrix @ kinetic_probes[:, j] for j in range(5)])
        result["C1_vs_exact_max_abs_error"] = float(np.max(np.abs(corr[1, 0, :5] - expected)))
    arrays = {"corr": corr, "sem": sem, "power": power, "k": k, "omega": frequencies}
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--tag", default="phase_bit_s4_open23")
    args = parser.parse_args()
    if args.length < 32 or args.length % 2 or args.steps < 64 or args.samples < 3:
        parser.error("even length >=32, steps >=64, samples >=3 required")
    conservation = conservation_tests()
    local, matrix, probes = exact_one_step(BitGas())
    output = {"schema": 1, "conservation": conservation, "one_step_projection_original": local,
              "microscopic_parameters": {"filling": "1/2", "randomness": "initial test inputs only",
                                         "time": "separate discrete tick", "hidden_period": 4},
              "runs": []}
    arrays = {}
    # Original along all visible axes; repair along x first. It is not
    # claimed isotropic, and a negative screen is not a no-go theorem.
    for variant, axis in [("original", 0), ("original", 1), ("original", 2), ("repaired", 0)]:
        result, data = simulate(variant, axis, args.length, args.steps, args.samples, matrix, probes)
        output["runs"].append(result)
        arrays.update({f"{variant}_{axis}_{key}": value for key, value in data.items()})
        # Save a checkpoint after each completed independent orientation.
        with open(data_path(args.tag + ".json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            f.write("\n")
        np.savez_compressed(data_path(args.tag + ".npz"), **arrays)
    print("saved " + data_path(args.tag + ".json"), flush=True)


if __name__ == "__main__":
    main()
