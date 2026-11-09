#!/usr/bin/env python3
"""Bit 8 independent integration check: local rule + D4 directional streaming.

Finite seeded validation, not a proof of global mixing or a packet experiment.
The microscopic collision is delegated to the Bit 8 integer/bit rule.
"""
import argparse
import json
import random
import time
from pathlib import Path

from phase_bit_s4_open27_active_collision import ROOTS, collide, charge, GROUP
from phase_bit_s4_open25_symmetry_recovery import symmetry_maps, permute, gf_rank


def indices(shape):
    assert len(shape) == 4 and all(n>=4 and n%2==0 for n in shape)
    sites = [(x,y,z,w) for x in range(shape[0]) for y in range(shape[1])
             for z in range(shape[2]) for w in range(shape[3]) if (x+y+z+w)%2 == 0]
    return sites, {xy:i for i,xy in enumerate(sites)}


def stream(state, shape, inverse=False):
    sites, lookup = indices(shape)
    assert len(state)==len(sites)
    result=[0]*len(state)
    direction=(-1 if inverse else 1)
    for i,coordinate in enumerate(sites):
        word=state[i]
        for h,v in enumerate(ROOTS):
            if word & (1<<h):
                dest=tuple((coordinate[j]+direction*v[j])%shape[j] for j in range(4))
                result[lookup[dest]] |= 1<<h
    return result


def tick(state,shape,inverse=False):
    return [collide(s) for s in stream(state,shape,inverse=True)] if inverse else stream([collide(s) for s in state],shape)


def charges(state):
    values=[0]*5
    for s in state:
        local=charge(s)
        for j in range(5):values[j]+=local[j]
    return tuple(values)


def check(seed=2809, half_samples=512, ticks=3, shape=(4,4,4,4)):
    rng=random.Random(seed)
    generators=symmetry_maps(ROOTS)
    start=time.monotonic()
    changed=0
    witnesses=[]
    for i in range(half_samples):
        occupied=rng.sample(range(24),12)
        s=sum(1<<j for j in occupied)
        t=collide(s)
        assert collide(t)==s
        assert charge(t)==charge(s)
        for p in generators.values():
            assert collide(permute(s,p))==permute(t,p)
        changed+=int(s!=t)
        if s!=t and len(witnesses)<6:
            witnesses.append({'start':s,'end':t})
    scan_seconds=time.monotonic()-start
    nodes, _ = indices(shape)
    state=[sum(1<<j for j in rng.sample(range(24),12)) for _ in nodes]
    initial=state.copy()
    invariants=charges(state)
    timestep_seconds=[]
    for j in range(ticks):
        at=time.monotonic();state=tick(state,shape);timestep_seconds.append(round(time.monotonic()-at,3))
        assert charges(state)==invariants
    for j in range(ticks):
        state=tick(state,shape,inverse=True)
        assert charges(state)==invariants
    assert state==initial
    return {'seed':seed,'sample_n':half_samples,'occupation':12,'moving':changed,
        'moving_fraction':changed/half_samples,'symmetry_generators_checked':list(generators),
        'invariant_count':5,'group_size':len(GROUP),
        'lattice_shape':list(shape),'physical_nodes':len(nodes),
        'forward_ticks':ticks,'reverse_ticks':ticks,'exact_full_lattice_roundtrip':True,
        'exact_total_number_and_momenta':True,
        'seconds_half_density_screening':round(scan_seconds,3),
        'forward_tick_seconds':timestep_seconds,
        'note':'Finite sample and small-box check; no phase-packet or long-time mixing result.'}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--half-samples',type=int,default=512)
    p.add_argument('--ticks',type=int,default=3)
    p.add_argument('--out',type=Path,default=None)
    args=p.parse_args()
    result=check(half_samples=args.half_samples,ticks=args.ticks)
    content=json.dumps(result,indent=2)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(content,encoding='utf8')
    print(content)