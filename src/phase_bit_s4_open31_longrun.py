#!/usr/bin/env python3
"""Reproducible bounded/unbounded Bit-10 collision and 4D reversibility tests."""
import argparse,json,random,resource,time
from pathlib import Path
import numpy as np
from phase_bit_s4_open29_engine import VectorOrbitD4
from phase_bit_s4_open29_fast_collision import collide_fast,preference_fast
from phase_bit_s4_open31_bounded_cache import BoundedOrbitD4

def initial_state(nodes,seed,occupation):
    rng=random.Random(seed)
    return np.fromiter((sum(1<<d for d in rng.sample(range(24),occupation)) for _ in range(nodes)),dtype=np.uint32,count=nodes)

def run(shape=(8,4,4,8),ticks=16,max_entries=1024,seed=3112,occupation=12,compare=True):
    collide_fast.cache_clear();preference_fast.cache_clear()
    engine=BoundedOrbitD4(shape,max_entries)
    ref=VectorOrbitD4(shape) if compare else None
    initial=initial_state(engine.nodes,seed,occupation)
    cur=initial.copy(); other=initial.copy() if compare else None
    inv=engine.invariants(initial)
    timeline=[]
    start=time.perf_counter()
    for tick in range(ticks):
        begin=time.perf_counter();cur=engine.tick(cur);dt=time.perf_counter()-begin
        if ref is not None:
            other=ref.tick(other)
            assert np.array_equal(cur,other),(tick,'reference mismatch')
        assert engine.invariants(cur)==inv,(tick,'charges')
        assert not np.any(cur>>np.uint32(24))
        timeline.append({'tick':tick+1,'seconds':round(dt,6),'entries':len(engine.cache),
                         'evictions':engine.evictions,'engine_misses':engine.pair_misses,
                         'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         'inner_collide_cache_entries':collide_fast.cache_info().currsize,
                         'inner_preference_cache_entries':preference_fast.cache_info().currsize})
    forward_time=time.perf_counter()-start
    reverse_start=time.perf_counter()
    for tick in range(ticks):
        cur=engine.tick(cur,reverse=True)
        if ref is not None:
            other=ref.tick(other,reverse=True)
            assert np.array_equal(cur,other),(tick,'reverse reference mismatch')
    assert np.array_equal(cur,initial)
    if ref is not None:assert np.array_equal(other,initial)
    ci=collide_fast.cache_info();pi=preference_fast.cache_info()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'shape':list(shape),'nodes':engine.nodes,'seed':seed,'occupation':occupation,
            'forward_ticks':ticks,'reverse_ticks':ticks,'reference_comparison':bool(compare),
            'all_bitwise_equal_to_unbounded':bool(compare),'exact_roundtrip':True,
            'number_and_four_momenta_conserved':True,'padding_zero':True,
            'engine_cache':engine.cache_stats(),'maxrss_kib_linux':rss,
            'inner_collide_lru':ci._asdict(),'inner_preference_lru':pi._asdict(),
            'reference_cache_entries':len(ref.cache) if ref is not None else None,
            'forward_elapsed_seconds':round(forward_time,4),
            'reverse_elapsed_seconds':round(time.perf_counter()-reverse_start,4),
            'timeline':timeline}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shape',type=int,nargs=4,default=[8,4,4,8]);p.add_argument('--ticks',type=int,default=16);p.add_argument('--capacity',type=int,default=1024);p.add_argument('--occupation',type=int,default=12);p.add_argument('--seed',type=int,default=3112);p.add_argument('--no-reference',action='store_true');p.add_argument('--out',type=Path)
    a=p.parse_args();res=run(tuple(a.shape),a.ticks,a.capacity,a.seed,a.occupation,not a.no_reference)
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps({k:v for k,v in res.items() if k!='timeline'},indent=2))