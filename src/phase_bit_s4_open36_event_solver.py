#!/usr/bin/env python3
"""Bit 17: exact sparse event-driven perturbation around a known D4 background.

The microscopic collision is unchanged Bit 8 via Bit 10's collide_fast.
D = P XOR B. Only sites where D != 0 need the difference between
C(B XOR D) and C(B); unchanged sites cancel exactly. Transport the 24
bit channels individually by a precomputed D4 neighbor permutation.

For a spatially uniform background B(x)=b, C(b) is also uniform and
streaming leaves uniform words unchanged. Thus baseline costs one local
collision per tick. An arbitrary background still needs its full update.
This is an exact alternative representation, not a compression guarantee.
"""
from __future__ import annotations
import argparse, json, time, zlib
from pathlib import Path
import numpy as np
from phase_bit_s4_open31_bounded_cache import BoundedOrbitD4
from phase_bit_s4_open29_fast_collision import collide_fast
from phase_bit_s4_open33_shells import init_pair
from phase_bit_s4_open34_front_rays import residual_bytes

FULL=(1<<24)-1

class SparseD4Events:
    def __init__(self, engine):
        self.e=engine
        # sources[d, dst] = src; invert permutation to get targets[d, src].
        self.targets=np.empty_like(engine.sources)
        ids=np.arange(engine.nodes, dtype=np.int32)
        for direction in range(24):
            self.targets[direction, engine.sources[direction]]=ids
        assert np.all(self.targets[0, engine.sources[0]]==ids)

    def advance(self, delta:dict[int,int], baseline):
        """An exact forward step of the sparse XOR difference D, given B_t.

        baseline is either a homogeneous 24-bit int or an array of length N.
        Does not mutate baseline or the supplied sparse dictionary.
        """
        next_delta={}; source_sites=0; source_bits=0
        for site,change in delta.items():
            if not (0<=site<self.e.nodes and 0<change<=FULL):
                raise ValueError('Invalid sparse state')
            base = int(baseline) if isinstance(baseline,(int,np.integer)) else int(baseline[site])
            after = int(collide_fast(base ^ change)) ^ int(collide_fast(base))
            correction = after ^ change
            if correction: source_sites+=1;source_bits+=correction.bit_count()
            # Streaming each occupied XOR bit to its destination.
            while after:
                low=after & -after
                direction=low.bit_length()-1
                dest=int(self.targets[direction,site])
                new=next_delta.get(dest,0)^low
                if new:next_delta[dest]=new
                else:next_delta.pop(dest,None)
                after-=low
        return next_delta, {'collision_evaluation_sites':len(delta),
            'nonlinear_source_sites':source_sites,'nonlinear_source_bits':source_bits,
            'outgoing_active_sites':len(next_delta),
            'outgoing_direction_bits':sum(x.bit_count() for x in next_delta.values())}

    def reconstruct(self, delta, baseline):
        if isinstance(baseline,(int,np.integer)):
            out=np.full(self.e.nodes, int(baseline),dtype=np.uint32)
        else:out=np.asarray(baseline,dtype=np.uint32).copy()
        for site,word in delta.items():out[site]^=np.uint32(word)
        return out

    def as_sparse(self, diff):
        return {int(i):int(diff[i]) for i in np.flatnonzero(diff)}


def measure_case(shape=(8,4,4,8),ticks=6,seed=1701,kind='uniform_half'):
    e=BoundedOrbitD4(shape,max_pairs=0)
    solver=SparseD4Events(e)
    if kind=='uniform_half':
        b0=0x8cd1ae  # active, momentum-preserving Bit8 collision
        baseline=np.full(e.nodes,b0,dtype=np.uint32)
        pulse=baseline.copy()
        where=np.where(np.all(e.coords==0,axis=1))[0]
        assert len(where)==1
        pulse[where[0]]=np.uint32(FULL)
        homogeneous=True
    elif kind=='vacuum':
        b0=0
        baseline=np.zeros(e.nodes,dtype=np.uint32)
        pulse=baseline.copy()
        where=np.where(np.all(e.coords==0,axis=1))[0]
        pulse[where[0]]=np.uint32(FULL)
        homogeneous=True
    elif kind=='random_half':
        baseline,pulse,_,_=init_pair(e,'background_pulse',seed,(0,0,0,0))
        b0=None
        homogeneous=False
    else:raise ValueError(kind)
    initial_base=baseline.copy(); initial_pulse=pulse.copy()
    reference_base=baseline.copy(); reference_pulse=pulse.copy()
    base_word=int(b0) if homogeneous else None
    delta=solver.as_sparse(pulse^baseline)
    entries=[]
    for tick in range(ticks):
        old_b=base_word if homogeneous else baseline
        current=solver.reconstruct(delta,old_b)
        assert np.array_equal(current,reference_pulse)
        assert np.array_equal(baseline,reference_base)
        t0=time.perf_counter()
        next_delta, diag=solver.advance(delta,old_b)
        event_time=time.perf_counter()-t0
        # Advance baseline separately (exact and O(1) only if homogeneous).
        t0=time.perf_counter()
        if homogeneous:
            base_word=int(collide_fast(base_word))
            baseline.fill(np.uint32(base_word))
        else:
            baseline=e.tick(baseline)
        base_time=time.perf_counter()-t0
        t0=time.perf_counter()
        reference_pulse=e.tick(reference_pulse)
        reference_base=e.tick(reference_base)
        full_pair_time=time.perf_counter()-t0
        delta=next_delta
        reconstruction=solver.reconstruct(delta,base_word if homogeneous else baseline)
        assert np.array_equal(reconstruction,reference_pulse), f'mismatch tick {tick+1}'
        assert np.array_equal(baseline,reference_base)
        # No extra microscopic degrees: charge checks on reconstructed reference.
        assert e.invariants(reconstruction)==e.invariants(initial_pulse)
        assert e.invariants(baseline)==e.invariants(initial_base)
        payload=residual_bytes(reconstruction^baseline)
        entries.append({'tick':tick+1,**diag,'fraction_evaluated_sites':diag['collision_evaluation_sites']/e.nodes,
            'event_seconds':round(event_time,6),'baseline_seconds':round(base_time,6),
            'reference_pair_seconds':round(full_pair_time,6),
            'delta_zlib_bytes':len(zlib.compress(payload,9)),
            'reference_equal':True})
    # Backward on both explicit references to independently verify underlying dynamics.
    for _ in range(ticks):
        reference_pulse=e.tick(reference_pulse,reverse=True)
        reference_base=e.tick(reference_base,reverse=True)
    assert np.array_equal(reference_pulse,initial_pulse)
    assert np.array_equal(reference_base,initial_base)
    assert np.array_equal(solver.reconstruct(delta,base_word if homogeneous else baseline),
                          solver.reconstruct(delta,baseline))
    return {'test':'Bit17 exact sparse D4 perturbation/event evolution',
        'shape':list(shape),'physical_nodes':e.nodes,'ticks':ticks,'kind':kind,'seed':seed,
        'microscopic_rule':'unchanged Bit8 exact collision and D4 stream',
        'baseline':'one homogeneous word per tick' if homogeneous else 'dense evolving background',
        'exact_stepwise_match':True,'forward_reverse_reference_roundtrip':True,
        'charges_conserved':True,'active_sites_are_not_only_nonlinear_source_sites':True,
        'scope':'Sparse perturbation relative to a specified baseline; background cost excluded from event_seconds',
        'entries':entries}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shape',type=int,nargs=4,default=[8,4,4,8]);p.add_argument('--ticks',type=int,default=6)
    p.add_argument('--kind',choices=['vacuum','uniform_half','random_half'],default='uniform_half')
    p.add_argument('--seed',type=int,default=1701);p.add_argument('--out',type=Path)
    a=p.parse_args();r=measure_case(tuple(a.shape),a.ticks,a.seed,a.kind)
    text=json.dumps(r,indent=2)+'\n'
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(a.kind,r['physical_nodes'],[(x['tick'],x['collision_evaluation_sites'],x['nonlinear_source_sites'],x['event_seconds']) for x in r['entries']])
if __name__=='__main__':main()