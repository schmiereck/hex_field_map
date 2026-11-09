#!/usr/bin/env python3
"""Bit 9: cached packed 4D lattice engine for the unchanged Bit-8 collision.

The *microscopic rule* is exactly phase_bit_s4_open27_active_collision.collide.
Only sparse memoization, 24-bit packing and vectorized deterministic streaming
are changed. No approximation, lookup-table truncation, or stochastic updates.

A cold half-filled state may still require expensive symmetry canonicalization.
This is a bounded-size demonstration engine, not yet a high-throughput solver.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase_bit_s4_open27_active_collision import ROOTS, collide, charge

FULL=(1<<24)-1

class FastD4:
    def __init__(self,shape=(4,4,4,4)):
        self.shape=tuple(map(int,shape))
        if len(self.shape)!=4 or any(n<4 or n%2 for n in self.shape):
            raise ValueError('four even periods >= 4 required')
        self.coords=np.argwhere((np.indices(self.shape).sum(axis=0)%2)==0).astype(np.int32)
        self.nodes=len(self.coords)
        lookup=np.full(self.shape,-1,dtype=np.int32)
        lookup[tuple(self.coords.T)]=np.arange(self.nodes,dtype=np.int32)
        self.sources=[]
        for v in ROOTS:
            src=(self.coords-np.asarray(v,dtype=np.int32))%np.asarray(self.shape,dtype=np.int32)
            ids=lookup[tuple(src.T)]
            assert np.all(ids>=0)
            self.sources.append(ids)
        self.sources=np.asarray(self.sources,dtype=np.int32)
        self.masks=np.asarray([1<<i for i in range(24)],dtype=np.uint32)
        self.cache={}
        self.cache_hits=0
        self.cache_misses=0

    def collision(self,words):
        """Exact site-local collision; compute only previously unseen 24-bit words."""
        words=np.asarray(words,dtype=np.uint32)
        assert words.shape==(self.nodes,)
        unique,inverse=np.unique(words,return_inverse=True)
        output=np.empty(len(unique),dtype=np.uint32)
        for i,s in enumerate(unique):
            k=int(s)
            val=self.cache.get(k)
            if val is None:
                val=int(collide(k))
                self.cache[k]=val
                self.cache[val]=k # involutive, exact reverse map
                self.cache_misses+=1
            else:self.cache_hits+=1
            output[i]=val
        return output[inverse]

    def stream(self,words,reverse=False):
        """24 direction-wise gather operations, with exact reversible permutation."""
        words=np.asarray(words,dtype=np.uint32)
        if words.shape!=(self.nodes,):raise ValueError('wrong state shape')
        if reverse:
            # Forward source map: destination i gets state[i-v].
            # Reverse: source i receives state[i+v], equivalent inverse heading map.
            out=np.zeros_like(words)
            for d,src in enumerate(self.sources):
                out[src] |= words & self.masks[d]
            return out
        out=np.zeros_like(words)
        for d,src in enumerate(self.sources):
            out |= words[src]&self.masks[d]
        return out

    def tick(self,words,reverse=False):
        return self.collision(self.stream(words,reverse=True)) if reverse else self.stream(self.collision(words))

    def invariants(self,words):
        occ=np.empty(24,dtype=np.int64)
        for d in range(24):occ[d]=int(np.count_nonzero(words & self.masks[d]))
        roots=np.asarray(ROOTS,dtype=np.int64)
        return (int(occ.sum()),)+tuple(int(x) for x in (occ@roots))


def slow_stream(state,coords,shape,reverse=False):
    lookup={tuple(p):i for i,p in enumerate(coords)}
    out=[0]*len(state)
    direction=-1 if reverse else 1
    for i,p in enumerate(coords):
        for d,v in enumerate(ROOTS):
            if state[i]&(1<<d):
                dest=tuple((int(p[j])+direction*int(v[j]))%shape[j] for j in range(4))
                out[lookup[dest]]|=1<<d
    return out

def reference_tick(state,engine,reverse=False):
    if reverse:
        return [collide(v) for v in slow_stream(state,engine.coords,engine.shape,reverse=True)]
    return slow_stream([collide(v) for v in state],engine.coords,engine.shape)


def run(shape=(4,4,4,4),ticks=3,seed=2909,occupation=12):
    rng=random.Random(seed)
    e=FastD4(shape)
    initial=np.asarray([sum(1<<d for d in rng.sample(range(24),occupation))
                        for _ in range(e.nodes)],dtype=np.uint32)
    x=initial.copy(); y=initial.tolist()
    inv=e.invariants(x)
    timings=[]
    for _ in range(ticks):
        t=time.perf_counter();x=e.tick(x);timings.append(time.perf_counter()-t)
        y=reference_tick(y,e)
        assert x.tolist()==y,'packed and reference differ'
        assert e.invariants(x)==inv,'charges changed'
        assert not np.any(x >> np.uint32(24)), 'padding bits not zero'
    for _ in range(ticks):
        x=e.tick(x,reverse=True)
        y=reference_tick(y,e,reverse=True)
        assert x.tolist()==y,'reverse differs'
        assert e.invariants(x)==inv
    assert np.array_equal(x,initial)
    # Warm cache replay, same site states, measured separately.
    t=time.perf_counter(); a=e.collision(initial); warm=time.perf_counter()-t
    t=time.perf_counter(); b=e.collision(initial); warm2=time.perf_counter()-t
    assert np.array_equal(a,b)
    return {'rule':'Bit8 unchanged', 'seed':seed,'shape':list(shape),
        'physical_nodes':e.nodes,'occupation_per_initial_node':occupation,
        'forward_ticks':ticks,'backward_ticks':ticks,
        'exact_reference_forward_and_reverse':True,'roundtrip':True,
        'number_and_four_momenta_conserved':True,'padding_bits_zero':True,
        'forward_tick_seconds':[round(v,6) for v in timings],
        'warm_collision_seconds':[round(warm,6),round(warm2,6)],
        'distinct_cached_words':len(e.cache), 'collision_cache_hits':e.cache_hits,
        'collision_cache_misses':e.cache_misses,
        'caveat':'Cold collision evaluation is not accelerated; no packet physics claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shape',type=int,nargs=4,default=[4,4,4,4]);p.add_argument('--ticks',type=int,default=3);p.add_argument('--occupation',type=int,default=12);p.add_argument('--out',type=Path)
    args=p.parse_args();res=run(tuple(args.shape),args.ticks,occupation=args.occupation)
    data=json.dumps(res,indent=2)+'\n'
    if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(data)
    print(data)