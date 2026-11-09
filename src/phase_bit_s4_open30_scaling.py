#!/usr/bin/env python3
"""Bit 11: bounded, repeatable scaling survey of unchanged Bit-10 VectorOrbitD4.

Run separate invocations per size for independent process RSS/caches. Never
assume a warm cache represents cold-new-state throughput. 4D periodic D4 grid.
"""
import argparse, json, random, resource, sys, time, tracemalloc
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from phase_bit_s4_open29_engine import VectorOrbitD4
from phase_bit_s4_open29_fast_collision import collide_fast, preference_fast

def run(shape, ticks, density, seed):
    rng=random.Random(seed)
    t=time.perf_counter(); engine=VectorOrbitD4(shape); setup=time.perf_counter()-t
    if density==12:
        words=np.fromiter((sum(1<<j for j in rng.sample(range(24),12)) for _ in range(engine.nodes)),np.uint32,count=engine.nodes)
    else:
        words=np.fromiter((sum(1<<j for j in range(24) if rng.random()<density/24) for _ in range(engine.nodes)),np.uint32,count=engine.nodes)
    initial_unique=len(np.unique(words))
    initial=words.copy(); inv=engine.invariants(words)
    logs=[]
    for step in range(ticks):
        before=len(engine.cache)
        cc=collide_fast.cache_info()
        pc=preference_fast.cache_info()
        t=time.perf_counter(); words=engine.tick(words); secs=time.perf_counter()-t
        assert engine.invariants(words)==inv
        assert not np.any(words>>np.uint32(24))
        logs.append({'tick':step+1,'elapsed_seconds':round(secs,6),
                     'cache_entries_before':before,'cache_entries_after':len(engine.cache),
                     'new_cache_entries':len(engine.cache)-before,
                     'unique_states_after':len(np.unique(words)),
                     'collide_lru_misses_delta':collide_fast.cache_info().misses-cc.misses,
                     'preference_lru_misses_delta':preference_fast.cache_info().misses-pc.misses})
    t=time.perf_counter()
    for _ in range(ticks):words=engine.tick(words,reverse=True)
    reverse_secs=time.perf_counter()-t
    assert np.array_equal(words,initial)
    t=time.perf_counter();engine.tick(initial);warm=time.perf_counter()-t
    cache_bytes_lower_bound=(len(engine.cache)*8+engine.sources.nbytes+engine.coords.nbytes+initial.nbytes)
    return {'shape':list(shape),'physical_nodes':engine.nodes,'occupation':density,'seed':seed,
            'initial_unique_states':initial_unique,'setup_seconds':round(setup,6),
            'forward':logs,'reverse_total_seconds':round(reverse_secs,6),
            'warm_initial_tick_seconds':round(warm,6), 'exact_roundtrip':True,
            'five_charges_conserved':True,'padding_zero':True,
            'final_collision_cache_entries':len(engine.cache),
            'collision_cache_hits':engine.cache_hits,'collision_cache_misses':engine.cache_misses,
            'kernel_array_bytes':int(engine.sources.nbytes+engine.coords.nbytes+initial.nbytes),
            'cache_and_array_lower_bound_bytes':int(cache_bytes_lower_bound),
            'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'note':'maxrss is peak PROCESS RSS, includes runtime/imports; cache lower bound is not real Python dict memory.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shape',nargs=4,type=int,required=True)
    p.add_argument('--ticks',type=int,default=2);p.add_argument('--density',type=int,default=12)
    p.add_argument('--seed',type=int,default=3011);p.add_argument('--out',type=Path)
    a=p.parse_args(); r=run(tuple(a.shape),a.ticks,a.density,a.seed)
    text=json.dumps(r,indent=2)+'\n'
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(text)