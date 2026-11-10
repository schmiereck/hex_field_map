#!/usr/bin/env python3
"""Bit 18. Exact sparse event evolution with radial support measurements.

This measures sitewise XOR perturbations relative to one spatially uniform
Bit-8 background. No assumption of an isotropic wave or continuum geometry.
With a periodic lattice, radial diagnostics are only nonwrapped before the
shortest period / 2. Beyond this point they describe torus distance.
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
from phase_bit_s4_open36_event_solver import SparseD4Events, FULL
from phase_bit_s4_open31_bounded_cache import BoundedOrbitD4
from phase_bit_s4_open29_fast_collision import collide_fast


def radial_metrics(e, delta, center=(0,0,0,0)):
    n=len(delta)
    if not n:
        return {'active_sites':0,'active_direction_bits':0,'mean_r':0.,'r10':0.,'r50':0.,'r90':0.,'outer_radius':0.,
                'inner_half_ball_fraction':0.,'radial_shell_occupancy':[]}
    positions=np.fromiter(delta.keys(),dtype=np.int32,count=n)
    vals=list(delta.values())
    periodic=np.minimum((e.coords[positions]-center)%e.shape,(center-e.coords[positions])%e.shape)
    radii=np.sqrt(np.sum(periodic.astype(np.int32)**2,axis=1))
    order=np.argsort(radii)
    r10,r50,r90=np.quantile(radii,[.1,.5,.9])
    maximum=float(radii.max())
    all_dist=np.minimum((e.coords-center)%e.shape,(center-e.coords)%e.shape)
    all_r=np.sqrt(np.sum(all_dist.astype(np.int32)**2,axis=1))
    inner_threshold=float(r90/2)
    inside=(all_r<=inner_threshold)
    occupied=(radii<=inner_threshold)
    frac=float(np.count_nonzero(occupied)/max(1,np.count_nonzero(inside)))
    edges=np.arange(0,np.ceil(maximum)+1,1.)
    if len(edges)<2:edges=np.array([0.,1.])
    h=np.histogram(radii,bins=edges)[0]
    h_all=np.histogram(all_r,bins=edges)[0]
    return {'active_sites':n,'active_direction_bits':int(sum(x.bit_count() for x in vals)),
            'mean_r':float(radii.mean()),'r10':float(r10),'r50':float(r50),'r90':float(r90),
            'outer_radius':maximum,'inner_half_ball_fraction':frac,
            'radial_shell_occupancy':[[int(a),int(b)] for a,b in zip(h,h_all)]}


def run(shape,ticks=50,kind='uniform_half',full_check_ticks=2,roundtrip=False,time_budget=150.0):
    shape=tuple(shape)
    e=BoundedOrbitD4(shape,max_pairs=0)
    solver=SparseD4Events(e)
    center=np.zeros(4,dtype=np.int32)
    base_word={'uniform_half':0x8cd1ae,'vacuum':0}[kind]
    original_base=base_word
    delta={int(np.flatnonzero(np.all(e.coords==center,axis=1))[0]):FULL}
    original_delta=delta.copy()
    records=[];start=time.perf_counter()
    control=None
    if full_check_ticks:
        control=solver.reconstruct(delta,base_word)
    for t in range(1,ticks+1):
        t0=time.perf_counter()
        delta,diag=solver.advance(delta,base_word)
        base_word=int(collide_fast(base_word))
        dt=time.perf_counter()-t0
        if control is not None:
            control=e.tick(control)
            assert np.array_equal(solver.reconstruct(delta,base_word),control),f'mismatch at tick {t}'
            if t>=full_check_ticks:control=None
        metrics=radial_metrics(e,delta)
        records.append({'tick':t,'seconds':round(dt,5),**diag,**metrics,
            'support_fraction':len(delta)/e.nodes,'nonwrapped_geometry':t<min(shape)/2})
        if t % 5==0 or t==1:
            print(f'tick {t:3d} sites={len(delta):6d}/{e.nodes}, sources={diag["nonlinear_source_sites"]:5d}, r90={metrics["r90"]:6.2f}, dt={dt:.2f}s',flush=True)
        if time.perf_counter()-start>time_budget:
            print('time budget reached',flush=True)
            break
    # Exact reverse without a dense comparison, using S^{-1} then collision on affected sites.
    if roundtrip and len(records)==ticks:
        for t in range(ticks):
            previous_base=int(collide_fast(base_word))
            unstream={}
            for site,word in delta.items():
                x=word
                while x:
                    bit=x&-x; direction=bit.bit_length()-1
                    previous_site=int(e.sources[direction,site])
                    out=unstream.get(previous_site,0)^bit
                    if out:unstream[previous_site]=out
                    else:unstream.pop(previous_site,None)
                    x-=bit
            delta={}
            for site,word in unstream.items():
                uncollided=int(collide_fast(base_word ^ word)) ^ previous_base
                if uncollided:delta[site]=uncollided
            base_word=previous_base
        assert base_word==original_base
        assert delta==original_delta,f"reverse sparse mismatch: sites={len(delta)}"
    return {'test':'Bit18 long duration exact sparse D4 active-region geometry',
        'shape':list(shape),'nodes':e.nodes,'requested_ticks':ticks,'completed_ticks':len(records),
        'kind':kind,'full_dense_comparison_ticks':min(len(records),full_check_ticks),
        'exact_roundtrip':bool(roundtrip and len(records)==ticks),
        'shortest_period':min(shape),'periodic_wrap_warning_at_tick':min(shape)//2,
        'elapsed_seconds':round(time.perf_counter()-start,4),'records':records}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shape',type=int,nargs=4,default=[8,8,8,8]);
    p.add_argument('--ticks',type=int,default=50);p.add_argument('--kind',choices=['uniform_half','vacuum'],default='uniform_half');
    p.add_argument('--full-check-ticks',type=int,default=2);p.add_argument('--roundtrip',action='store_true');
    p.add_argument('--time-budget',type=float,default=150.);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=run(a.shape,a.ticks,a.kind,a.full_check_ticks,a.roundtrip,a.time_budget)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('saved',a.out,flush=True)