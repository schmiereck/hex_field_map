#!/usr/bin/env python3
"""Bit 14: falsifiable shell-compression diagnostic for Bit-8 D4 Boolean gas.

Microdynamics: EXACT established collision+streaming, with engine cache disabled.
This study analyzes density deviations, NOT a complete 24-bit microstate encoding.
Shells use the periodized D4 Euclidean squared radius around a fixed origin.
No Fourier q-filter or random transition is used in the microscopic ticks.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_bit_s4_open31_bounded_cache import BoundedOrbitD4
from phase_bit_s4_open29_fast_collision import collide_fast, preference_fast


def shell_labels(coords, shape, center):
    offsets = (coords.astype(np.int64) - np.asarray(center)) % np.asarray(shape)
    distances = np.minimum(offsets, np.asarray(shape)-offsets)
    return (distances*distances).sum(axis=1).astype(np.int32)


def analyze(y, labels, ks=(1, 2, 4, 8)):
    """Orthogonal radial projection plus optimally selected K radial shells.

    Each radial basis has one *independent, integer-squared radius* bin. This
    is an optimistic upper bound on fits from K narrow shells at fixed center:
    no cross-radial cancellation and amplitudes may have either sign.
    """
    y = np.asarray(y, dtype=np.float64)
    bands, index = np.unique(labels, return_inverse=True)
    counts = np.bincount(index, minlength=len(bands)).astype(float)
    means = np.bincount(index, weights=y, minlength=len(bands)) / counts
    power = counts * (means**2)
    order = np.argsort(-power)
    norm = float(y@y)
    explained = {} 
    for k in ks:
        idx=order[:k]
        fit=np.where(np.isin(index, idx), means[index], 0.)
        err=float(np.sum((y-fit)**2))
        explained[str(k)] = {'energy_retained':float(max(0., 1-err/norm)) if norm else 1.,
                             'relative_l2_error':float(np.sqrt(err/norm)) if norm else 0.}
    residual=float(np.sum((y-means[index])**2))
    return {'energy':norm,'radial_bands':len(bands),'nonzero_radial_bands':int(np.count_nonzero(np.abs(means)>1e-12)),
            'full_radial_energy_retained':float(max(0.,1-residual/norm)) if norm else 1.,
            'full_radial_relative_l2_error':float(np.sqrt(residual/norm)) if norm else 0.,
            'k_shells':explained,
            'dominant_r2':[int(bands[j]) for j in order[:min(8,len(order))] if power[j]>1e-12]}


def angular_fit(y, labels, coords, shape, center, k=4, order=1):
    """Best K disjoint radial shells, with 4D angular polynomial profiles.

    This is a *much more expressive* upper-bound fit, not a low-cost exact
    field encoding. Return effective parameter count to expose overfitting.
    """
    y=np.asarray(y,dtype=float)
    v=(coords.astype(float)-np.asarray(center)) % np.asarray(shape)
    v=np.where(v>np.asarray(shape)/2,v-np.asarray(shape),v)
    radius=np.sqrt(np.sum(v*v,axis=1))
    v=v/np.maximum(1.,radius)[:,None]
    features=[np.ones(len(v))]
    if order>=1: features.extend(v[:,j] for j in range(4))
    if order>=2: features.extend(v[:,i]*v[:,j] for i in range(4) for j in range(i,4))
    a=np.column_stack(features)
    groups=[]
    for r in np.unique(labels):
        inds=np.where(labels==r)[0]
        z=a[inds]
        coef=np.linalg.lstsq(z,y[inds],rcond=None)[0]
        predicted=z@coef
        energy=float(predicted@predicted)
        groups.append((energy,int(np.linalg.matrix_rank(z)),int(r),int(len(inds))))
    groups.sort(reverse=True)
    selected=groups[:k]
    denom=float(y@y)
    return {'energy_retained':float(sum(x[0] for x in selected)/denom) if denom else 1.,
            'effective_linear_parameters':sum(x[1] for x in selected),
            'selected_r2':[x[2] for x in selected],
            'model':'radial shells with degree <= '+str(order)+' 4D angular polynomials'}


def shell_observables(engine, a, b, labels, origin):
    x = np.array([int(i).bit_count() for i in a], dtype=np.float64)
    y = np.array([int(i).bit_count() for i in b], dtype=np.float64)
    d=x-y
    report=analyze(d,labels)
    report['angular_degree1_k4']=angular_fit(d, labels, engine.coords, engine.shape, origin,4,1)
    report['angular_degree2_k4']=angular_fit(d, labels, engine.coords, engine.shape, origin,4,2)
    report['total_signed_density_deviation']=int(d.sum())
    report['absolute_density_deviation']=float(np.abs(d).sum())
    report['active_delta_density_sites']=int(np.count_nonzero(d))
    # A 3D observational slice is not a full projection. It uses only w=origin_w.
    slice_mask=(engine.coords[:,3]==origin[3])
    report['visible_w0_slice']=analyze(d[slice_mask],shell_labels(engine.coords[slice_mask,:3],engine.shape[:3],origin[:3]))
    return report


def init_pair(engine, kind, seed, origin):
    rng=np.random.default_rng(seed)
    labels=shell_labels(engine.coords,engine.shape,origin)
    if kind=='vacuum_pulse':
        baseline=np.zeros(engine.nodes,dtype=np.uint32)
        pulse=baseline.copy()
        center=np.where(np.all(engine.coords==np.asarray(origin),axis=1))[0]
        assert len(center)==1
        pulse[center[0]]=np.uint32((1<<24)-1)
        # Random control: same 24 particles, one direction bit at each of 24
        # different random nodes. No assumption of equal momentum.
        random=baseline.copy()
        targets=rng.choice(engine.nodes,size=24,replace=False)
        for direction,node in enumerate(targets): random[node]|=np.uint32(1<<direction)
        return baseline,pulse,random,labels
    if kind=='background_pulse':
        # The shared half-filled background ensures the difference is the
        # response to a finite local change under *nonlinear* collisions.
        background=np.zeros(engine.nodes,dtype=np.uint32)
        for j in range(engine.nodes):
            bits=rng.choice(24,12,replace=False)
            background[j]=np.bitwise_or.reduce(np.uint32(1)<<bits)
        pulse=background.copy()
        center=np.where(np.all(engine.coords==np.asarray(origin),axis=1))[0]
        assert len(center)==1
        c=int(center[0]); missing=[d for d in range(24) if not (int(pulse[c])>>d)&1]
        pulse[c]=np.uint32((1<<24)-1)
        random=background.copy()
        nodes=rng.choice(np.array([i for i in range(engine.nodes) if i!=c]),len(missing),replace=False)
        for direction,node in zip(missing,nodes):
            # Set an available zero bit on the chosen node to produce exactly
            # the same NUMBER of additions (but not identical momentum).
            empty=[d for d in range(24) if not (int(random[node])>>d)&1]
            chosen=empty[int(rng.integers(len(empty)))]
            random[node]|=np.uint32(1<<chosen)
        assert sum(int(v).bit_count() for v in pulse)-sum(int(v).bit_count() for v in background)==len(missing)
        assert sum(int(v).bit_count() for v in random)-sum(int(v).bit_count() for v in background)==len(missing)
        return background,pulse,random,labels
    raise ValueError(kind)


def run(shape=(8,4,4,8), ticks=4, seed=1409, kind='vacuum_pulse'):
    collide_fast.cache_clear();preference_fast.cache_clear()
    e=BoundedOrbitD4(shape,max_pairs=0)
    center=tuple(0 for _ in shape)
    assert any(np.all(e.coords==center,axis=1))
    background,pulse,random,labels=init_pair(e,kind,seed,center)
    originals=[v.copy() for v in (background,pulse,random)]
    initial_charges=[e.invariants(v) for v in originals]
    current=[v.copy() for v in originals]
    history=[]
    for tick in range(ticks+1):
        entry={'tick':tick,'pulse':shell_observables(e,current[1],current[0],labels,center),
               'random_control':shell_observables(e,current[2],current[0],labels,center)}
        if tick<ticks:
            changed=[]
            for v in current:
                cv=e.collision(v)
                changed.append(int(np.count_nonzero(cv!=v)))
            entry['collision_changed_sites_before_next_tick']=dict(zip(('background','pulse','random_control'),changed))
        history.append(entry)
        for v,charges in zip(current,initial_charges):
            assert e.invariants(v)==charges
            assert not np.any(v>>np.uint32(24))
        if tick<ticks: current=[e.tick(v) for v in current]
    forward=[v.copy() for v in current]
    for _ in range(ticks):
        current=[e.tick(v,reverse=True) for v in current]
    for a,b in zip(current,originals):assert np.array_equal(a,b), 'roundtrip failed'
    return {'experiment':'Bit14 radial shell decomposition of exact Bit8 gas',
            'kind':kind,'shape':list(shape),'physical_nodes':e.nodes,'seed':seed,
            'ticks':ticks,'center':list(center),'microscopic_collision':'unchanged Bit8 (vectorized Bit10 implementation)',
            'cache':'engine disabled; bounded internal lru_cache 100000',
            'quantity':'local particle-density difference against identical background',
            'shell_basis':'orthogonal radius^2 bins in periodic 4D shortest-distance metric at fixed center',
            'visible_detector':'single w=0 slice, no hidden w Fourier filter',
            'baseline':'vacuum' if kind=='vacuum_pulse' else 'same seeded 12/24 half-filled background',
            'random_control':'same number of added particles, spatially dispersed',
            'initial_particle_numbers':[a[0] for a in initial_charges],
            'all_five_charges_conserved':True,'exact_forward_backward_roundtrip':True,
            'history':history,'limitations':['density only, not full 24-bit exact representation','fixed shell center, periodized distance','finite deterministic samples, no ensemble uncertainties','shell dictionary optimistically selects its best radius bins']}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shape',type=int,nargs=4,default=(8,4,4,8))
    parser.add_argument('--ticks',type=int,default=4)
    parser.add_argument('--kind',choices=['vacuum_pulse','background_pulse'],default='vacuum_pulse')
    parser.add_argument('--seed',type=int,default=1409)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    result=run(tuple(args.shape),args.ticks,args.seed,args.kind)
    text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if args.out: args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text)
    for v in result['history']:
        print(v['tick'], 'pulse k4',round(v['pulse']['k_shells']['4']['energy_retained'],4),
              'full_radial',round(v['pulse']['full_radial_energy_retained'],4),
              'random full',round(v['random_control']['full_radial_energy_retained'],4),
              'changed',v.get('collision_changed_sites_before_next_tick'))