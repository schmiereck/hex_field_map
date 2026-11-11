#!/usr/bin/env python3
"""Bit 20: controlled phase-coded packet experiment, exact Bit-8 collisions.

Reconstructed (not the lost Bit-6 source). Uses integer Numba local rule
validated against Bit-8/Bit-10. Eight independent seeds PER (q,k) case,
4 real quadrature trajectories per seed, complex numbers only as diagnostics.
"""
from __future__ import annotations
import argparse, json, time, math, sys
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4, audit, collide_numba
from phase_bit_s4_open38_phased_packet import (directional_profile,packed_from_threshold,
 count_xw,response_channels,mirror_x_words)
from phase_bit_s4_open29_fast_collision import collide_fast


def synthesize_case(e,seed,km,qm,ticks,amplitude,free=False):
    rng=np.random.default_rng(seed)
    u16=rng.integers(0,65536,size=(e.nodes,24),dtype=np.uint16)
    flat_index=e.coords[:,0].astype(np.int32)*e.shape[3]+e.coords[:,3]
    differences=[]
    for phase in (0.,np.pi/2):
        plusminus=[]
        for sign in (1,-1):
            p=directional_profile(e,km,phase,amplitude,q_mode=qm)
            state=packed_from_threshold(u16,.5+sign*(p-.5))
            initial_charges=e.invariants(state)
            output=np.empty((ticks+1,3,e.shape[0]),dtype=np.complex64)
            for tick in range(ticks+1):
                density=count_xw(e,state,flat_index)
                output[tick]=response_channels(e,density,q_mode=qm)
                if tick<ticks:
                    state=e.stream(state) if free else e.tick(state)
            assert e.invariants(state)==initial_charges
            plusminus.append(output)
        differences.append((plusminus[0]-plusminus[1])*0.5)
    return differences[0] + 1j*differences[1]


def symmetry_audit(e,ticks=8):
    rng=np.random.default_rng(3901)
    s=rng.integers(0,1<<24,size=e.nodes,dtype=np.uint32)
    mirrored=mirror_x_words(e,s)
    originals=s.copy();mirrored_original=mirrored.copy()
    inv=e.invariants(s)
    for t in range(ticks):
        assert np.array_equal(mirror_x_words(e,s),mirrored),('mirror',t)
        s=e.tick(s);mirrored=e.tick(mirrored)
        assert e.invariants(s)==inv
    assert np.array_equal(mirror_x_words(e,s),mirrored)
    for _ in range(ticks):
        s=e.tick(s,reverse=True); mirrored=e.tick(mirrored,reverse=True)
    assert np.array_equal(s,originals)
    assert np.array_equal(mirrored,mirrored_original)
    return {'mirror_x_equivariant':True,'roundtrip':True,'ticks_each_way':ticks,'physical_nodes':e.nodes}


def run(shape=(48,4,4,16),ticks=24,seeds=8,qs=(1,2,3),modes=(0,3,-3),amplitude=.105):
    started=time.perf_counter()
    a=audit(n=350)
    e=NumbaOrbitD4(shape)
    mirror=symmetry_audit(e,6)
    arrays={}
    meta=[]
    for q in qs:
        for km in modes:
            samples=[];reference=[]
            for i in range(seeds):
                # Independent thresholds ACROSS q,k modes as well as seed index.
                seed=390100+q*10000+(km+30)*100+i
                signal=synthesize_case(e,seed,km,q,ticks,amplitude)
                control=synthesize_case(e,seed,km,q,ticks,amplitude,free=True)
                samples.append(signal);reference.append(control)
                print(f'q={q} k={km:+d} seed={i+1}/{seeds} elapsed={time.perf_counter()-started:.1f}s',flush=True)
            key=f'q{q}_k{km:+d}'
            arrays[key+'_bit8']=np.asarray(samples,dtype=np.complex64)
            arrays[key+'_free']=np.asarray(reference,dtype=np.complex64)
            meta.append({'q_mode':q,'k_mode':km,'seed_first':390100+q*10000+(km+30)*100,
                         'raw_array_key_bit8':key+'_bit8','raw_array_key_free':key+'_free'})
    info={'experiment':'Bit20 independent-seed q-dispersion packet study',
          'source':'Bit19 independent reconstruction of lost Bit6 experiment',
          'shape':list(shape),'physical_nodes':e.nodes,'ticks':ticks,'independent_seeds_per_case':seeds,
          'modes':list(modes),'hidden_q_modes':list(qs),'amplitude':amplitude,
          'quadratures':'cos and sin each plus/minus sign, four REAL finite-bit trajectories per case and seed',
          'phase_angles_only_in_initialization_and_external_analysis':True,
          'microscopic_dynamics':'exact Bit8 local involution + D4 streaming; Numba implementation audited against Bit10',
          'numba_collision_audit':a,'symmetry_audit':mirror,'all_five_charges_conserved_first_last':True,
          'cases':meta,'full_ensemble_elapsed_seconds':round(time.perf_counter()-started,3),
          'limits':['small y,z periods 4 may cause early periodic aliasing',
            'phase-coded reconstructed preparation not Bit6 eigenmode',
            'short times / finite seed uncertainty; no inferred mass or stable particle']}
    return info,arrays

def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',type=int,nargs=4,default=[48,4,4,16]);p.add_argument('--ticks',type=int,default=24);p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--qs',type=int,nargs='+',default=[1,2,3]);p.add_argument('--modes',type=int,nargs='+',default=[0,3,-3]);p.add_argument('--amplitude',type=float,default=.105)
    p.add_argument('--out',type=Path,default=Path('data/bit20_raw.npz'));p.add_argument('--metadata',type=Path,default=Path('data/bit20_metadata.json'))
    args=p.parse_args();meta,data=run(tuple(args.shape),args.ticks,args.seeds,tuple(args.qs),tuple(args.modes),args.amplitude)
    args.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.out,**data)
    args.metadata.write_text(json.dumps(meta,indent=2)+'\n')
    print('saved',args.out,args.metadata,flush=True)

if __name__=='__main__': main()
