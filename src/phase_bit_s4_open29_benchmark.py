#!/usr/bin/env python3
"""Independent Bit 10 collision comparison and packed 4D validation."""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
from phase_bit_s4_open27_active_collision import collide
from phase_bit_s4_open29_fast_collision import collide_fast, exact_audit
from phase_bit_s4_open25_symmetry_recovery import symmetry_maps,permute
from phase_bit_s4_open26_symmetric_collision import ROOTS
from phase_bit_s4_open28_fast4d import FastD4


def measure():
    rng=random.Random(2910)
    states=[sum(1<<i for i in rng.sample(range(24),12)) for _ in range(300)]
    # Explicit unique inputs for cold comparison, sampled without replacement.
    states=list(dict.fromkeys(states))
    t0=time.perf_counter()
    expected=[collide(s) for s in states]
    old_elapsed=time.perf_counter()-t0
    t0=time.perf_counter()
    actual=[collide_fast(s) for s in states]
    fast_elapsed=time.perf_counter()-t0
    assert actual==expected
    audit_states=states[:60]+[0,(1<<24)-1]+[1<<i for i in range(24)]
    audit=exact_audit(audit_states,collide,ROOTS,symmetry_maps,permute)
    packed=[]
    for shape,ticks in [((4,4,4,4),3),((8,4,4,8),2)]:
        e=FastD4(shape)
        # This preserves the original kernel for comparison, but replaces the
        # slow microscopic collision in the packed engine with the exact new one.
        e.cache.clear()
        rng2=random.Random(3400+e.nodes)
        x=np.array([sum(1<<i for i in rng2.sample(range(24),12)) for _ in range(e.nodes)],dtype=np.uint32)
        initial=x.copy()
        conserved=e.invariants(x)
        # Compare EACH forward and reverse step to original slow local collision,
        # using the same vectorized transport for both versions.
        y=x.copy()
        tic=time.perf_counter()
        for _ in range(ticks):
            x=e.stream(np.array([collide_fast(int(v)) for v in x],dtype=np.uint32))
            y=e.stream(np.array([collide(int(v)) for v in y],dtype=np.uint32))
            assert np.array_equal(x,y)
            assert e.invariants(x)==conserved
            assert not np.any(x>>24)
        forward_elapsed=time.perf_counter()-tic
        for _ in range(ticks):
            x=np.array([collide_fast(int(v)) for v in e.stream(x,reverse=True)],dtype=np.uint32)
            y=np.array([collide(int(v)) for v in e.stream(y,reverse=True)],dtype=np.uint32)
            assert np.array_equal(x,y)
        assert np.array_equal(x,initial)
        packed.append({'shape':shape,'nodes':e.nodes,'forward_ticks':ticks,'backward_ticks':ticks,'matches_bit8_at_each_step':True,'roundtrip':True,'conserves_five_charges':True,'combined_forward_comparison_seconds':round(forward_elapsed,6)})
    return {'algorithm':'batched uint32 B4 orbit transform + precomputed swap-mask classes',
            'same_microscopic_rule_as_bit8':True,'unique_half_filling_states':len(states),
            'cold_reference_seconds':round(old_elapsed,6),'cold_vectorized_seconds':round(fast_elapsed,6),
            'cold_speedup':round(old_elapsed/fast_elapsed,2),
            'exact_audit':audit,'4d_validation':packed,
            'caveat':'Hardware-specific timing; warm-cache replay not included; full 65,536-node packet campaign still untested.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args()
    result=measure();s=json.dumps(result,indent=2)+'\n'
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(s)
    print(s)