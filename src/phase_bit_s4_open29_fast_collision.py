#!/usr/bin/env python3
"""Bit 10: exact, vectorized orbit evaluation of unchanged Bit-8 local collision.

No approximate pruning, probability, or changed tie-breaking. 384 B4 transforms
are batched with uint32 NumPy table gathers. The collision remains the exact
involution of phase_bit_s4_open27_active_collision.collide.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase_bit_s4_open26_symmetric_collision import TABLES, GATES

# Same tables as Bit 8. uint32 avoids overflow when assembling 24-bit words.
TABS = np.asarray(TABLES, dtype=np.uint32)   # (384, 3, 256)
MUTATIONS = tuple((a | b) for a,b in GATES)


def transformed(state: int) -> np.ndarray:
    s=int(state)
    return TABS[:,0,s&255] | TABS[:,1,(s>>8)&255] | TABS[:,2,(s>>16)&255]


# Only masks from the finite, fixed two-body gate set can ever be proposals.
MASK_CLASS = {m:int(transformed(m).min()) for m in set(MUTATIONS)}


def proposals_fast(state: int) -> tuple[int,...]:
    """Same deterministic set of legal two-body swaps as Bit 8."""
    out=set()
    for left,right in GATES:
        if (state & left)==left and not (state & right):
            out.add(state ^ left ^ right)
        elif (state & right)==right and not (state & left):
            out.add(state ^ left ^ right)
    return tuple(sorted(out))


@lru_cache(maxsize=100000)
def preference_fast(state: int) -> int | None:
    neighbors=proposals_fast(state)
    if not neighbors:return None
    # Orbit of swap mask is equal to the exact Bit-8 canonical(s XOR t).
    best=min(MASK_CLASS[state^t] for t in neighbors)
    choices=[t for t in neighbors if MASK_CLASS[state^t]==best]
    if len(choices)==1:return choices[0]
    # Vectorized exact undirected-edge orbit keys, lexicographic ordering.
    xs=transformed(state)
    ranks=[]
    for t in choices:
        ys=transformed(t)
        low=np.minimum(xs,ys)
        high=np.maximum(xs,ys)
        a=int(low.min())
        b=int(high[low==a].min())
        ranks.append((a,b))
    optimum=min(ranks)
    winners=[t for t,r in zip(choices,ranks) if r==optimum]
    return winners[0] if len(winners)==1 else None


@lru_cache(maxsize=100000)
def collide_fast(state: int) -> int:
    state=int(state)
    t=preference_fast(state)
    return t if t is not None and preference_fast(t)==state else state


def exact_audit(samples, reference, roots, symmetry_maps, permute):
    generators=symmetry_maps(roots)
    movements=0
    for state in samples:
        target=collide_fast(state)
        assert target==reference(state),(state,target,reference(state))
        assert collide_fast(target)==state
        assert state.bit_count()==target.bit_count()
        for axis in range(4):
            assert sum(roots[i][axis] for i in range(24) if state>>i & 1)==sum(roots[i][axis] for i in range(24) if target>>i & 1)
        for g in generators.values():
            assert collide_fast(permute(state,g))==permute(target,g)
        movements+=(target!=state)
    return {'tested':len(samples),'moving':movements,'exact_match':True,
            'involution':True,'charge_conservation':True,'eight_generator_symmetry':True}