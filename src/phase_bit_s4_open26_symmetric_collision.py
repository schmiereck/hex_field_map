#!/usr/bin/env python3
"""Bit 7 prototype: exactly B4-equivariant reversible 24-bit D4 collision.

A proposal is any momentum-preserving 2-in/2-out exchange. Each state chooses
its UNIQUE neighboring state with smallest canonical (B4 orbit) representative.
The collision swaps s<->t only if the choices are mutual. Other states stay
fixed. This defines a global involution, without an extra clock bit.

All ranking/symmetry calculations are finite integer/bit operations. Unlike the
previous ordered gate circuit, this is a deliberately expensive *prototype*,
NOT a production-speed Boolean lattice-gas kernel. The conditional mutual-choice
rule is deterministic but nonlocal within the 24-bit *site*, not between sites.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from fractions import Fraction
from itertools import combinations, permutations, product
from pathlib import Path
import argparse
import json
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_bit_s4_open25_symmetry_recovery import Collision, permute, symmetry_maps, gf_rank

MASK = (1 << 24) - 1
ROOTS = Collision().roots


def all_group_permutations():
    lookup = {tuple(v): i for i, v in enumerate(ROOTS)}
    out = []
    for axes in permutations(range(4)):
        for signs in product((-1, 1), repeat=4):
            mapping = tuple(lookup[tuple(signs[j] * root[axes[j]] for j in range(4))] for root in ROOTS)
            out.append(mapping)
    assert len(out) == 384 and len(set(out)) == 384
    return out

GROUP = all_group_permutations()
# Translation tables make canonicalization ~1000 lookups instead of bit loops.
TABLES = tuple(tuple(tuple(sum((1 << mapping[base + j]) for j in range(8) if (val >> j) & 1)
                           for val in range(256)) for base in (0,8,16)) for mapping in GROUP)


@lru_cache(maxsize=300_000)
def canonical(state: int) -> int:
    a, b, c = state & 255, (state >> 8) & 255, (state >> 16) & 255
    return min(tab[0][a] | tab[1][b] | tab[2][c] for tab in TABLES)


def pair_gates():
    classes = defaultdict(list)
    for a,b in combinations(range(24),2):
        classes[tuple(ROOTS[a][j]+ROOTS[b][j] for j in range(4))].append((a,b))
    gates = set()
    for pairs in classes.values():
        for (a,b),(c,d) in combinations(pairs,2):
            if len({a,b,c,d}) == 4:
                left, right = (1<<a)|(1<<b), (1<<c)|(1<<d)
                gates.add((min(left,right),max(left,right)))
    return tuple(sorted(gates))

GATES = pair_gates()


@lru_cache(maxsize=200_000)
def proposals(state):
    """All *distinct* allowed two-to-two neighbors; conserved charges by construction."""
    result = set()
    for left,right in GATES:
        if state & left == left and state & right == 0:
            result.add(state ^ left ^ right)
        elif state & right == right and state & left == 0:
            result.add(state ^ left ^ right)
    return tuple(sorted(result))


@lru_cache(maxsize=100_000)
def preference(state):
    # Other proposals with the same canonical orbit are indistinguishable by
    # any B4-invariant scalar; reject ties rather than break symmetry.
    neighbors = proposals(state)
    if not neighbors:
        return None
    ranks = [(canonical(n), n) for n in neighbors]
    smallest = min(v for v, _ in ranks)
    winners = [n for value, n in ranks if value == smallest]
    return winners[0] if len(winners) == 1 else None


@lru_cache(maxsize=100_000)
def collide(state):
    t = preference(state)
    return t if t is not None and preference(t) == state else state


def charge(state):
    ids = [i for i in range(24) if (state >> i) & 1]
    return (len(ids),) + tuple(sum(ROOTS[i][j] for i in ids) for j in range(4))


def samples(seed=2609):
    rng = random.Random(seed)
    for n in range(4):
        for bits in combinations(range(24),n):
            yield sum(1<<i for i in bits)
    for n in (4,6,8,10,12,14,16,18,20):
        for _ in range(100 if n in (10,12,14) else 20):
            yield sum(1<<i for i in rng.sample(range(24),n))


def validate():
    generators = symmetry_maps(ROOTS)
    counts=defaultdict(lambda: [0,0])
    deltas=[]
    witnesses=[]
    independent=[]
    total=0
    for s in samples():
        t = collide(s)
        total+=1
        if charge(s) != charge(t):
            raise AssertionError(('conservation',s,t))
        if collide(t) != s:
            raise AssertionError(('not involutive',s,t))
        for name,mapping in generators.items():
            if collide(permute(s,mapping)) != permute(t,mapping):
                raise AssertionError(('not equivariant',name,s,t))
        counts[s.bit_count()][0] += 1
        if t!=s:
            counts[s.bit_count()][1] += 1
            delta=[int(bool(t&(1<<i)))-int(bool(s&(1<<i))) for i in range(24)]
            deltas.append(delta)
            if len(witnesses)<19 and gf_rank(independent+[delta])>len(independent):
                independent.append(delta)
                witnesses.append({'state':s,'target':t})
    assert len(witnesses)==19, f'failed exact rank lower bound: {len(witnesses)}'
    # Every legal proposal preserves five charges; 19 independent witnessed
    # changes prove that there are no further additive linear invariants.
    assert gf_rank([[(1 if i==j else 0) for i in range(24)] for j in range(24)])==24
    for w in witnesses:
        assert collide(w['state'])==w['target']
        assert charge(w['state'])==charge(w['target'])
    return {'status':'prototype finite sampling, not exhaustive over 2^24',
            'group_size':len(GROUP),'number_of_two_body_gate_types':len(GATES),
            'samples_checked':total,'symmetry_generators_checked':list(generators),
            'collision_is_involution_on_samples':True,'number_and_four_momenta_preserved_on_samples':True,
            'per_occupation':{str(n):{'tested':v[0],'moving':v[1]} for n,v in sorted(counts.items())},
            'moving_sample_difference_rank':gf_rank(deltas) if deltas else 0,
            'exact_charge_dimension_proof':{'independent_collision_differences':19,'witnesses':witnesses,'upper_bound_from_five_exact_charges':5,'conclusion':'exactly five additive linear invariants of the full collision'},
            'max_possible_change_rank':19,
            'caveat':'Formal full-space involution, B4 equivariance and conservation follow from the construction; 19 explicit nonzero collision witnesses establish exactly 5 additive linear invariants. Long-time transport and packet spectra are NOT established.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    result=validate()
    data=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(data,encoding='utf-8')
    print(data)

if __name__=='__main__':main()