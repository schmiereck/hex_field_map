#!/usr/bin/env python3
"""Bit 13: compare cache policies on a controlled, structured packet surrogate.

This is NOT the four-quadrature Bit-6 preparation. It is a deterministic,
spatially varying finite-bit input used only to stress/cache-profile the
unchanged Bit-8 microscopic collision. No cache participates in the physics.
"""
import argparse, json, time, random, resource
from collections import Counter
from pathlib import Path
import numpy as np
from phase_bit_s4_open31_bounded_cache import BoundedOrbitD4
from phase_bit_s4_open29_fast_collision import collide_fast, preference_fast

class HotsetOrbitD4(BoundedOrbitD4):
    """Static bounded frequent-word table from initial input; exact fallback."""
    def __init__(self,shape,capacity,initial):
        super().__init__(shape,0)
        common=Counter(map(int,initial)).most_common(capacity)
        self.hot={k:int(collide_fast(k)) for k,n in common}
        self.hot_hits=0
        self.hot_misses=0
    def collision(self,words):
        unique,inv=np.unique(np.asarray(words,dtype=np.uint32),return_inverse=True)
        vals=np.empty(len(unique),dtype=np.uint32)
        for i,s in enumerate(unique):
            k=int(s)
            v=self.hot.get(k)
            if v is None:
                self.hot_misses+=1
                v=int(collide_fast(k))
            else:self.hot_hits+=1
            vals[i]=v
        return vals[inv]
    def cache_stats(self):
        return {'capacity_entries':len(self.hot),'hits':self.hot_hits,'misses':self.hot_misses,'policy':'static initial frequency'}

def initial_words(coords,seed,kind):
    rng=random.Random(seed)
    if kind=='random12':
        return np.asarray([sum(1<<i for i in rng.sample(range(24),12)) for _ in coords],dtype=np.uint32)
    # Small alphabet of exactly half-filled bit masks. Choose with a spatial
    # Gaussian envelope, x/w carrier and seeded discrete thresholds.
    masks=[]
    for _ in range(6):
        masks.append(sum(1<<i for i in rng.sample(range(24),12)))
    nx=max(int(v) for v in coords[:,0])+1
    nw=max(int(v) for v in coords[:,3])+1
    out=[]
    for x,y,z,w in coords:
        amplitude=np.exp(-.5*((int(x)-nx/2)/max(2,nx/8))**2)
        carrier=np.cos(2*np.pi*(4*int(x)/nx+int(w)/nw))
        score=.5+.35*amplitude*carrier
        k=0 if rng.random()<score else 1
        if rng.random()<.10:k=2+rng.randrange(4)
        out.append(masks[k])
    return np.asarray(out,dtype=np.uint32)

def run(shape=(8,4,4,8),ticks=8,capacity=256,seed=3213,kind='packet_surrogate'):
    e0=BoundedOrbitD4(shape,0)
    initial=initial_words(e0.coords,seed,kind)
    methods=('none','lru','hotset')
    result={'shape':list(shape),'nodes':e0.nodes,'ticks':ticks,'capacity':capacity,'seed':seed,
            'input_kind':kind,'input_unique_words':int(len(np.unique(initial))),
            'input_description':'Gaussian x envelope, x/w cosine carrier, six 12-bit masks, discrete seeded sampling' if kind=='packet_surrogate' else 'independent uniform 12-of-24 mask per node',
            'policies':{},'same_forward_final':False,'same_reverse_initial':False}
    results=[]
    inv=e0.invariants(initial)
    for method in methods:
        collide_fast.cache_clear();preference_fast.cache_clear()
        e=BoundedOrbitD4(shape,capacity if method=='lru' else 0) if method!='hotset' else HotsetOrbitD4(shape,capacity,initial)
        a=initial.copy(); ts=[]; start=time.perf_counter()
        for t in range(ticks):
            t0=time.perf_counter();a=e.tick(a);ts.append(time.perf_counter()-t0)
            assert e.invariants(a)==inv
            assert not np.any(a>>24)
        forward=a.copy(); elapsed=time.perf_counter()-start
        for _ in range(ticks):a=e.tick(a,reverse=True)
        assert np.array_equal(a,initial)
        info=e.cache_stats(); info.update({'forward_seconds':round(elapsed,5),
                 'tick_seconds':[round(float(v),5) for v in ts],
                 'engine_cache_entries':len(e.cache),
                 'inner_collide_cache_entries':collide_fast.cache_info().currsize,
                 'inner_preference_cache_entries':preference_fast.cache_info().currsize,
                 'peak_rss_kib_process_cumulative':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        result['policies'][method]=info
        results.append(forward)
    assert all(np.array_equal(results[0],a) for a in results[1:])
    result['same_forward_final']=True
    result['same_reverse_initial']=True
    result['all_invariants_exact']=True
    result['caveat']='Process RSS cumulative across policies; timings serial with shared Python allocator and cached imported tables; not isolated benchmarks. Bit-6 physical packet preparation not yet restored.'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shape',nargs=4,type=int,default=[8,4,4,8]);p.add_argument('--ticks',type=int,default=8);p.add_argument('--capacity',type=int,default=256);p.add_argument('--kind',choices=['packet_surrogate','random12'],default='packet_surrogate');p.add_argument('--out',type=Path)
    a=p.parse_args();out=run(tuple(a.shape),a.ticks,a.capacity,kind=a.kind)
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='policies'},indent=2))
    for name,info in out['policies'].items():print(name,info['forward_seconds'],info.get('hits'),info.get('misses'))