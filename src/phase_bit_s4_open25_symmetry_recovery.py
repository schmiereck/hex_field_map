#!/usr/bin/env python3
"""Independent recovery of Bit-6 low-occupation symmetry diagnostic.

Reimplements the *published* BitGas/RepairedBitGas local collision from
phase_bit_s4_open22.py, phase_bit_s4_open23.py and the PackedD4 circuit in
phase_bit_s4_open24.py (GitHub main at e2b1e02).  Does not reproduce
Bit-6 packet trajectories or its original JSON/NPZ output.

Microscopic operations use fixed Boolean/permutation gates. Floating-point
numbers are not used in the collision. Optional NumPy is not required.
"""
import argparse
from collections import defaultdict
from itertools import combinations, product
import json
from pathlib import Path


def roots_and_groups():
    roots = []
    for i, j in combinations(range(4), 2):
        for a, b in product((-1, 1), repeat=2):
            v = [0] * 4
            v[i], v[j] = a, b
            if v[next(k for k, x in enumerate(v) if x)] > 0:
                roots.append(tuple(v))
    axes = sorted(roots)
    directions = tuple(v for a in axes for v in (a, tuple(-x for x in a)))
    groups = [tuple((j + offset) % 12 for j in range(start, start + 3))
              for offset in (0, 1) for start in range(0, 12, 3)]
    assert len(directions) == 24 and len(groups) == 8
    return directions, groups


def gf_rank(vectors):
    """Rank of integer vectors over rationals via exact Fraction elimination."""
    from fractions import Fraction
    rows = {}
    for values in vectors:
        v = [Fraction(x) for x in values]
        for pivot in sorted(rows):
            if v[pivot]:
                factor = v[pivot]
                v = [x - factor * y for x, y in zip(v, rows[pivot])]
        pivot = next((i for i, x in enumerate(v) if x), None)
        if pivot is not None:
            scale = v[pivot]
            rows[pivot] = [x / scale for x in v]
    return len(rows)


def select_repair(roots, groups):
    """Same ordered eight-gate rank-increasing repair selection as Bit Test 4."""
    basis = []
    for group in groups:
        for a, b in zip(group, group[1:] + group[:1]):
            v = [0] * 24
            for i, val in ((2*a,-1),(2*a+1,-1),(2*b,1),(2*b+1,1)):
                v[i] += val
            basis.append(v)
    assert gf_rank(basis) == 11
    classes = defaultdict(list)
    for a, b in combinations(range(24), 2):
        classes[tuple(roots[a][j] + roots[b][j] for j in range(4))].append((a, b))
    selected = []
    rank = 11
    for momentum, pairs in sorted(classes.items()):
        if not any(momentum):
            continue
        for first, second in combinations(pairs, 2):
            gate = first + second
            v = [0] * 24
            for idx in first: v[idx] -= 1
            for idx in second: v[idx] += 1
            next_rank = gf_rank(basis + [v])
            if next_rank > rank:
                basis.append(v)
                selected.append(gate)
                rank = next_rank
            if rank == 19:
                assert len(selected) == 8
                return selected
    raise AssertionError('failed to construct repair gates')


class Collision:
    def __init__(self):
        self.roots, self.groups = roots_and_groups()
        self.extra = select_repair(self.roots, self.groups)

    def apply(self, state, inverse=False):
        """Exact Boolean packed collision; matches PackedD4 gate ordering."""
        out = state
        if inverse:
            out = self._scatter(out, inverse=True)
        for group in reversed(self.groups) if inverse else self.groups:
            shifts = [2*a for a in group]
            a,b,c = [(out >> shift) & 3 for shift in shifts]
            if all(v in (0,3) for v in (a,b,c)):
                values = (b,c,a) if inverse else (c,a,b)
                mask = sum(3 << shift for shift in shifts)
                out = (out & ~mask) | sum(v << shift for v,shift in zip(values,shifts))
        return out if inverse else self._scatter(out)

    def _scatter(self, state, inverse=False):
        out = state
        for gate in reversed(self.extra) if inverse else self.extra:
            a,b,c,d = [(out >> j) & 1 for j in gate]
            if a == b and c == d and a != c:
                out ^= sum(1 << j for j in gate)
        return out

    def charges(self, state):
        ids = [i for i in range(24) if state >> i & 1]
        return (len(ids),) + tuple(sum(self.roots[i][j] for i in ids) for j in range(4))


def permute(state, direction_map):
    return sum(1 << direction_map[i] for i in range(24) if state >> i & 1)


def symmetry_maps(roots):
    lookup = {root:i for i,root in enumerate(roots)}
    maps = {}
    ops = {'inversion': lambda v: tuple(-x for x in v)}
    for axis in range(4):
        ops[f'mirror_{axis}'] = (lambda axis: lambda v: tuple(-x if j == axis else x for j,x in enumerate(v)))(axis)
    for a,b in ((0,1),(1,2),(2,3)):
        ops[f'swap_{a}_{b}'] = (lambda a,b: lambda v: tuple(v[b] if j==a else v[a] if j==b else v[j] for j in range(4)))(a,b)
    for name, op in ops.items():
        p = tuple(lookup[op(v)] for v in roots)
        assert sorted(p) == list(range(24))
        maps[name] = p
    return maps


def low_occupation_states():
    for n in range(4):
        for slots in combinations(range(24), n):
            yield sum(1 << idx for idx in slots)


def audit():
    kernel = Collision()
    maps = symmetry_maps(kernel.roots)
    results = {name: {'failures':0, 'first_witness':None} for name in maps}
    checked = 0
    charge_violations = 0
    inversion_failures = 0
    for state in low_occupation_states():
        changed = kernel.apply(state)
        checked += 1
        if kernel.charges(state) != kernel.charges(changed): charge_violations += 1
        if kernel.apply(changed, inverse=True) != state:
            raise AssertionError(f'collision reverse failed at {state}')
        for name,p in maps.items():
            left = kernel.apply(permute(state,p))
            right = permute(changed,p)
            if left != right:
                row = results[name]
                row['failures'] += 1
                if row['first_witness'] is None:
                    row['first_witness'] = {
                        'input': [i for i in range(24) if state >> i & 1],
                        'collide_after_transform': [i for i in range(24) if left >> i & 1],
                        'transform_after_collide': [i for i in range(24) if right >> i & 1]
                    }
    if charge_violations:
        raise AssertionError(f'conservation failed in {charge_violations} states')
    assert checked == 2325
    witness = (1 << 0) | (1 << 22)
    p = maps['inversion']
    l,r = kernel.apply(permute(witness,p)), permute(kernel.apply(witness),p)
    expected = ([7,21],[1,23])
    actual = ([i for i in range(24) if l >> i & 1], [i for i in range(24) if r >> i & 1])
    assert actual == expected, f'Reported Bit 6 witness differs: {actual} vs {expected}'
    assert all(item['failures'] > 0 for item in results.values())
    return {
        'status': 'independent_recovery_check_not_original_Bit6_dataset',
        'upstream_commit': 'e2b1e02995e5ad0a18798c223d721311eaaedd91',
        'states_exhausted': checked,
        'particle_number_and_four_momenta_conserved': charge_violations == 0,
        'collision_reversible_on_tested_states': True,
        'extra_repair_gates': [list(gate) for gate in kernel.extra],
        'symmetries': results,
        'reported_two_bit_inversion_witness_verified': True,
        'limitations': 'No Bit 6 packet campaign or numeric dataset recreated.'
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=None, help='optional JSON output path')
    args = ap.parse_args()
    result = audit()
    data = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data, encoding='utf-8')
    print(data)

if __name__ == '__main__': main()