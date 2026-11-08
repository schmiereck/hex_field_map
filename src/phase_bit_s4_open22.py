#!/usr/bin/env python3
"""Bit test 3: exact D4 hidden-layer dynamics and D4/D5/E8 geometry.

Run: python src/phase_bit_s4_open22.py

The microscopic update uses Boolean gates and fixed permutations ONLY. No
amplitudes, floating-point weights, random transitions, growing counters, or
continuous gate angles occur in BitGas.tick. Seeded randomness supplies test
inputs only. Fractions and integer moments belong to the external analysis.

This is a classical reversible lattice-gas diagnostic, NOT a derivation of
quantum interference, spin, gravity, a Lorentzian signature, or massive
dispersion. Four spatial coordinates and an external discrete tick are used;
w is NOT time in this experiment. The deliberately ordered collision gates
do not inherit the rotational symmetry of the D4 root shell.
"""

import argparse
from collections import Counter
from fractions import Fraction
from itertools import combinations, combinations_with_replacement, product
import json
from pathlib import Path

import numpy as np

from paths import data_path


def d_roots(dim):
    """Integer roots (+-1,+-1,0,...,0), length squared 2."""
    roots = []
    for i, j in combinations(range(dim), 2):
        for a, b in product((-1, 1), repeat=2):
            v = [0] * dim
            v[i], v[j] = a, b
            roots.append(v)
    return np.array(roots, dtype=np.int64)


def e8_roots_twice():
    """2 times the canonical E8 roots; integer coordinates, length squared 8."""
    spinor = [v for v in product((-1, 1), repeat=8)
              if sum(x < 0 for x in v) % 2 == 0]
    return np.vstack((2 * d_roots(8), np.array(spinor, dtype=np.int64)))


def odd_double_factorial(n):
    out = 1
    for j in range(1, n + 1, 2):
        out *= j
    return out


def isotropic_moment(roots, order):
    """Check EVERY independent tensor component against the spherical moment.

    Exact rational arithmetic; valid for this single-radius root shell.
    Rotational invariance of the shell moments does not imply isotropy of
    an arbitrarily chosen collision rule or Lorentz invariance.
    """
    count, dim = roots.shape
    radii = np.sum(roots * roots, axis=1)
    assert np.all(radii == radii[0])
    m = order // 2
    denom = 1
    for j in range(m):
        denom *= dim + 2 * j
    deviations = []
    for indices in combinations_with_replacement(range(dim), order):
        powers = Counter(indices)
        measured = int(np.prod(roots[:, indices], axis=1).sum())
        expected = Fraction(0)
        if all(p % 2 == 0 for p in powers.values()):
            numer = count * int(radii[0]) ** m
            for p in powers.values():
                numer *= odd_double_factorial(p - 1)
            expected = Fraction(numer, denom)
        if measured != expected:
            deviations.append((indices, measured, str(expected)))
    return {"isotropic": not deviations,
            "components_checked": len(list(combinations_with_replacement(range(dim), order))),
            "first_failure": list(deviations[0]) if deviations else None}


def geometry_tests():
    rows = []
    for name, roots, scale in [("D3/FCC", d_roots(3), 1), ("D4", d_roots(4), 1),
                               ("D5", d_roots(5), 1), ("D8", d_roots(8), 1),
                               ("E8", e8_roots_twice(), 2)]:
        m4 = int(np.sum(roots[:, 0] ** 4))
        m22 = int(np.sum(roots[:, 0] ** 2 * roots[:, 1] ** 2))
        row = {"lattice": name, "dimension": roots.shape[1], "neighbors": len(roots),
               "coordinate_scale": scale, "M1111_over_M1122": str(Fraction(m4, m22)),
               "moments": {str(p): isotropic_moment(roots, p) for p in (2, 4, 6)}}
        assert row["moments"]["2"]["isotropic"]
        rows.append(row)
    assert [r["moments"]["4"]["isotropic"] for r in rows] == [False, True, False, False, True]
    assert [r["moments"]["6"]["isotropic"] for r in rows] == [False, False, False, False, True]
    return rows


def pair_gate(bits, inverse=False):
    """Six Boolean slots = three opposite pairs.

    If every opposite pair is either empty (00) or full (11), rotate
    the three pairs. Otherwise leave all bits alone. Rotation preserves
    the eligibility condition, so its inverse is the opposite rotation.
    No counting or arithmetic is required by the gate itself.
    """
    b = np.asarray(bits)
    if b.dtype != np.bool_ or b.shape[-1] != 6:
        raise ValueError("pair_gate requires six Boolean slots")
    eligible = (b[..., 0] == b[..., 1]) & (b[..., 2] == b[..., 3]) & (b[..., 4] == b[..., 5])
    indices = [2, 3, 4, 5, 0, 1] if inverse else [4, 5, 0, 1, 2, 3]
    return np.where(eligible[..., None], b[..., indices], b)


class BitGas:
    """D4 = even coordinate sum, periodic even-sized box, 24 bits per node.

    Tick: local collision circuit, then one-edge directional streaming.
    Each collision is a permutation of 6 local bits; all 24-bit local
    states are covered by composition. Padding sites of odd parity are
    stored but MUST remain zero. They are not physical nodes.
    """

    def __init__(self, shape=(4, 4, 4, 4)):
        if len(shape) != 4 or any(n < 4 or n % 2 for n in shape):
            raise ValueError("four even periods >= 4 required")
        self.shape = tuple(shape)
        axes = sorted(tuple(int(x) for x in v) for v in d_roots(4)
                      if v[np.flatnonzero(v)[0]] > 0)
        self.roots = np.array([w for v in axes for w in (v, tuple(-x for x in v))], dtype=np.int64)
        self.valid = np.indices(shape).sum(axis=0) % 2 == 0
        # Two overlapping partitions of the 12 unoriented axes. These
        # arbitrary ordered choices explicitly break full shell symmetry.
        self.groups = [tuple((j + offset) % 12 for j in range(start, start + 3))
                       for offset in (0, 1) for start in range(0, 12, 3)]

    def empty(self):
        return np.zeros(self.shape + (24,), dtype=bool)

    def collide(self, state, inverse=False):
        out = state.copy()
        for group in reversed(self.groups) if inverse else self.groups:
            slots = [slot for axis in group for slot in (2 * axis, 2 * axis + 1)]
            out[..., slots] = pair_gate(out[..., slots], inverse=inverse)
        return out

    def stream(self, state, inverse=False):
        out = np.empty_like(state)
        sign = -1 if inverse else 1
        for d, v in enumerate(self.roots):
            out[..., d] = np.roll(state[..., d], tuple(sign * int(x) for x in v), axis=(0, 1, 2, 3))
        return out

    def tick(self, state, inverse=False):
        if inverse:
            return self.collide(self.stream(state, inverse=True), inverse=True)
        return self.stream(self.collide(state))

    def project(self, state):
        """3D view retains each heading count, hides only the w coordinate.

        Counts are exact analysis observables, not local microscopic registers.
        Keeping headings makes the closure test stronger than density alone.
        """
        return np.count_nonzero(state, axis=3)

    def invariants(self, state):
        counts = np.count_nonzero(state, axis=(0, 1, 2, 3)).astype(np.int64)
        return int(counts.sum()), tuple(int(v) for v in counts @ self.roots)

    def check(self, state):
        assert state.dtype == np.bool_
        assert state.shape == self.shape + (24,)
        assert not np.any(state[~self.valid])


def gate_tests():
    inputs = np.array(list(product((False, True), repeat=6)), dtype=bool)
    outputs = pair_gate(inputs)
    assert len({row.tobytes() for row in outputs}) == 64
    assert np.array_equal(pair_gate(outputs, inverse=True), inputs)
    assert np.array_equal(inputs.sum(axis=1), outputs.sum(axis=1))
    # Difference within each opposite pair is unchanged: preserves momentum
    # for ANY three opposite root pairs, not just one selected triple.
    signed_in = inputs[:, ::2].astype(int) - inputs[:, 1::2].astype(int)
    signed_out = outputs[:, ::2].astype(int) - outputs[:, 1::2].astype(int)
    assert np.array_equal(signed_in, signed_out)
    return {"local_states_exhausted": 64, "bijective": True,
            "particle_number_exact": True, "opposite_pair_momentum_exact": True}


def reversibility_tests(gas):
    rng = np.random.default_rng(2203)  # TEST INPUTS only, never part of tick.
    checked = 0
    for _ in range(32):
        initial = rng.integers(0, 2, size=gas.shape + (24,), dtype=np.uint8).astype(bool)
        initial[~gas.valid] = False
        invariant = gas.invariants(initial)
        state = initial.copy()
        for _ in range(16):
            state = gas.tick(state)
            gas.check(state)
            assert gas.invariants(state) == invariant
            checked += 1
        for _ in range(16):
            state = gas.tick(state, inverse=True)
        assert np.array_equal(initial, state)
    return {"initial_states": 32, "forward_ticks_checked": checked,
            "bit_exact_round_trips": 32, "particle_number_error": 0, "momentum_error": 0,
            "proof": "composition of exhaustive local bijections and directional translations"}


def single_particle_tests(gas):
    for d, v in enumerate(gas.roots):
        initial = gas.empty()
        initial[0, 0, 0, 0, d] = True
        expected = gas.empty()
        target = tuple(int(x) % n for x, n in zip(v, gas.shape))
        expected[target + (d,)] = True
        assert np.array_equal(gas.tick(initial), expected)
    return {"directions_checked": 24, "collision_inactive": True,
            "one_edge_ballistic_transport": True, "emergent_mass_demonstrated": False}


def projection_tests(gas):
    # Same visible heading counts, different hidden co-location. At xyz=0,
    # w=0 and w=2 are both D4 sites. All four placements are enumerated.
    states = []
    for wp, wm in product((0, 2), repeat=2):
        state = gas.empty()
        state[0, 0, 0, wp, 0] = True
        state[0, 0, 0, wm, 1] = True
        states.append(state)
    macro = gas.project(states[0])
    assert all(np.array_equal(gas.project(s), macro) for s in states)
    outcomes = Counter(gas.project(gas.tick(s)).tobytes() for s in states)
    assert sorted(outcomes.values()) == [2, 2]
    # Free streaming alone DOES close under this projection. Failure is
    # caused by nonlinear collisions and hidden correlations, not by
    # simply discarding the visible heading.
    free = [gas.project(gas.stream(s)) for s in states]
    assert all(np.array_equal(free[0], s) for s in free)

    # Mixed-axis collisions connect headings with and without a w component.
    # Here observe channel sets rather than asserting a second macro law.
    visible_dirs = gas.roots[:, 3] == 0
    hidden_dirs = ~visible_dirs
    transfer = None
    for axis in range(12):
        state = gas.empty()
        state[0, 0, 0, 0, 2 * axis:2 * axis + 2] = True
        after = gas.collide(state)
        before_n = int(state[..., hidden_dirs].sum())
        after_n = int(after[..., hidden_dirs].sum())
        if before_n != after_n:
            transfer = {"input_axis": axis, "hidden_heading_count_before": before_n,
                        "hidden_heading_count_after": after_n}
            break
    assert transfer is not None
    assert int(visible_dirs.sum()) == 12 and int(hidden_dirs.sum()) == 12

    # Search for an exact three-time witness: same current visible state,
    # different previous visible state, different next visible state.
    # A witness disproves first-order Markov closure for this explicitly
    # enumerated, equally weighted collection of history samples.
    triples = []
    for state in states:
        trajectory = [gas.project(state).tobytes()]
        for _ in range(64):
            state = gas.tick(state)
            trajectory.append(gas.project(state).tobytes())
        triples.extend(zip(trajectory[:-2], trajectory[1:-1], trajectory[2:]))
    by_current = {}
    for prev, current, nxt in triples:
        by_current.setdefault(current, {}).setdefault(prev, Counter())[nxt] += 1
    memory_witness = None
    for groups in by_current.values():
        groups = list(groups.values())
        for a, b in combinations(groups, 2):
            choices = set(a) | set(b)
            for outcome in choices:
                pa = Fraction(a[outcome], sum(a.values()))
                pb = Fraction(b[outcome], sum(b.values()))
                if pa != pb:
                    memory_witness = {"conditional_counts_a": [a[outcome], sum(a.values())],
                                      "conditional_counts_b": [b[outcome], sum(b.values())],
                                      "conditional_fractions": [str(pa), str(pb)]}
                    break
            if memory_witness:
                break
        if memory_witness:
            break
    return {"same_initial_macrostate": True, "hidden_initial_states_enumerated": 4,
            "distinct_next_macrostates": len(outcomes), "exact_outcome_counts": sorted(outcomes.values()),
            "uniform_initial_ensemble_fractions": ["1/2", "1/2"],
            "closed_deterministic_macro_rule": False, "free_stream_projection_closes": True,
            "heading_sector_exchange": transfer, "history_samples": len(triples),
            "first_order_memory_witness": memory_witness}


def feedback_gate(state, inverse=False):
    """Separate minimal diagnostic: visible bit a and hidden bits b,c.

    Fredkin (a controls swap b,c), then cyclically move (a,b,c)->(c,a,b).
    The visible bit controls hidden routing, which feeds the next visible
    bit. This is not an extra gate of BitGas and not a quantum operation.
    """
    s = np.asarray(state)
    if s.dtype != np.bool_ or s.shape[-1] != 3:
        raise ValueError("feedback_gate requires three Boolean slots")
    if inverse:
        s = s[..., [1, 2, 0]]
    a, b, c = (s[..., j] for j in range(3))
    fredkin = np.stack((a, np.where(a, c, b), np.where(a, b, c)), axis=-1)
    return fredkin if inverse else fredkin[..., [2, 0, 1]]


def feedback_memory_tests():
    """Exact stationary ensemble of all 8 states, no fitted probabilities.

    An ensemble uniform over a finite permutation is stationary. The
    conditional fractions here genuinely disprove first-order Markov
    behavior for the observed bit in this ensemble.
    """
    s0 = np.array(list(product((False, True), repeat=3)), dtype=bool)
    s1, s2 = feedback_gate(s0), feedback_gate(feedback_gate(s0))
    assert len({s.tobytes() for s in s1}) == 8
    assert np.array_equal(feedback_gate(s1, inverse=True), s0)
    assert np.array_equal(s0.sum(axis=1), s1.sum(axis=1))
    y0, y1, y2 = s0[:, 0], s1[:, 0], s2[:, 0]
    rows = []
    for label, selected in [("Y1=1", y1), ("Y0=0,Y1=1", ~y0 & y1), ("Y0=1,Y1=1", y0 & y1)]:
        numer, denom = int(np.count_nonzero(y2 & selected)), int(np.count_nonzero(selected))
        rows.append({"condition": label, "next_one_count": numer, "total": denom,
                     "fraction": str(Fraction(numer, denom))})
    assert [r["fraction"] for r in rows] == ["1/2", "0", "1"]
    return {"separate_minimal_circuit": True, "states_exhausted": 8,
            "bijective_and_number_conserving": True, "stationary_uniform_ensemble": True,
            "conditional_counts": rows, "first_order_markov": False,
            "microscopic_random_draws": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(data_path("phase_bit_s4_open22.json")))
    args = parser.parse_args()
    gas = BitGas()
    result = {"schema": 1, "geometry": geometry_tests(), "gate": gate_tests(),
              "dynamics": reversibility_tests(gas), "projection": projection_tests(gas),
              "single_particle": single_particle_tests(gas),
              "minimal_feedback_memory": feedback_memory_tests(),
              "physical_nodes": int(gas.valid.sum()), "bits_per_node": 24,
              "spatial_dimensions": 4, "time": "separate discrete tick",
              "scope": "classical Boolean projection diagnostic; no QM or gravity claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
