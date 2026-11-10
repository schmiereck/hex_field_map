#!/usr/bin/env python3
"""Bit 19: phase-coded packet response of exact symmetric Bit-8 D4 Boolean gas.

This is a documented *reconstruction*, not the missing original Bit-6 source.
Old Bit-6 spectral eigenvectors/threshold details were not recovered. Phase
preparations use 16-bit uniform thresholds, 24 direction-dependent acoustic
reference weights and real Boolean trajectories. Complex fields are EXTERNAL
quadrature analyses of four real trajectories, never microscopic amplitudes.
"""
from __future__ import annotations
import argparse, json, math, time, sys
from pathlib import Path
import numpy as np
from phase_bit_s4_open31_bounded_cache import BoundedOrbitD4
from phase_bit_s4_open27_active_collision import ROOTS
from phase_bit_s4_open29_fast_collision import collide_fast, preference_fast
from phase_bit_s4_open25_symmetry_recovery import symmetry_maps

PC=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)


def packed_from_threshold(u16, probability):
    """Integer 16-bit Bernoulli threshold, fixed randomness within a seed."""
    thresholds=np.floor(np.clip(probability,0.,1.)*65536).astype(np.int32)
    flags=u16.astype(np.int32)<thresholds
    out=np.zeros(len(flags),dtype=np.uint32)
    for d in range(24):out|=flags[:,d].astype(np.uint32)<<np.uint32(d)
    return out


def directional_profile(e, k_mode, phase, amplitude=0.105, q_mode=1):
    nx,ny,nz,nw=e.shape
    x0=nx//2
    dx=(e.coords[:,0].astype(np.int32)-x0+nx//2)%nx-nx//2
    k=2*np.pi*k_mode/nx
    q=2*np.pi*q_mode/nw
    theta=k*dx+q*e.coords[:,3]-phase
    gaussian=np.exp(-.5*(dx/(nx/6.4))**2)
    wavevector=np.array([k,0,0,q])
    speed=np.linalg.norm(wavevector)
    roots=np.asarray(ROOTS, dtype=np.float64)
    # Reference linear acoustic current/density, not a measured Bit8 eigenmode.
    # For D4: mean(v_i v_j)=delta_ij/2, so beta=sqrt(2)/|wavevector|.
    factors=1+(math.sqrt(2)/speed)*(roots@wavevector)
    assert np.min(factors)>=-1.01 and np.max(factors)<=3.01
    p=.5 + amplitude*gaussian[:,None]*np.cos(theta)[:,None]*factors[None,:]
    assert 0.0<np.min(p) and np.max(p)<1.0
    return p


def count_xw(e, words, flat_index):
    densities=(PC[words&255].astype(np.int16)+PC[(words>>8)&255].astype(np.int16)
             +PC[(words>>16)&255].astype(np.int16))
    return np.bincount(flat_index, weights=densities,minlength=e.shape[0]*e.shape[3]).reshape(e.shape[0],e.shape[3])


def response_channels(e, density, q_mode=1):
    nx,_,_,nw=e.shape
    wphase=np.exp(-2j*np.pi*q_mode*np.arange(nw)/nw)
    return np.stack((density[:,0],density@wphase/nw,density.mean(axis=1)),axis=0)


def synthesize(seed, mode, e, epochs, kind='bit8', amplitude=0.105):
    """Four real trajectories, paired thresholds, return 3×time×x complex response."""
    rng=np.random.default_rng(seed)
    u16=rng.integers(0,65536,size=(e.nodes,24),dtype=np.uint16)
    xs=e.coords[:,0].astype(np.int32)
    ws=e.coords[:,3].astype(np.int32)
    flat_index=xs*e.shape[3]+ws
    channels=[]
    starts=[]
    conserved=[]
    for phase in (0.,np.pi/2):
        differences=[]
        for sign in (1,-1):
            p=directional_profile(e,mode,phase,amplitude)
            state=packed_from_threshold(u16,.5+sign*(p-.5))
            starts.append(state.copy())
            charges=e.invariants(state)
            conserved.append(charges)
            trajectory=np.empty((epochs+1,3,e.shape[0]),dtype=np.complex128)
            for tick in range(epochs+1):
                if tick in (0,epochs):
                    assert e.invariants(state)==charges, (seed,mode,phase,sign,tick)
                trajectory[tick]=response_channels(e,count_xw(e,state,flat_index))
                if tick<epochs:
                    state=e.stream(state) if kind=='free' else e.tick(state)
            differences.append(trajectory)
        channels.append((differences[0]-differences[1])/2.)
    signal=channels[0]+1j*channels[1]
    return signal, {'threshold_bits':16,'seeds':seed,'starts':len(starts),'total_number_initial':sum(c[0] for c in conserved),
                    'all_five_charges_verified_at_start_end':True}


def pos_width(intensity, x0):
    # Limited-width time window before visible x period dominates.
    nx=len(intensity)
    xx=(np.arange(nx)-x0+nx//2)%nx-nx//2
    sm=float(np.sum(intensity))
    if sm<=0:return (float('nan'),float('nan'))
    mu=float(np.sum(intensity*xx)/sm)
    var=float(np.sum(intensity*(xx-mu)**2)/sm)
    return mu,float(np.sqrt(max(0.,var)))


def analyze(signals, mode, shape, max_tick):
    signals=np.asarray(signals)
    averaged=signals.mean(axis=0)
    ns, nt, nchan, nx = len(signals),signals.shape[1],signals.shape[2],shape[0]
    assert averaged.shape==(nt,3,nx)
    x0=nx//2
    x=np.arange(nx)-x0
    k=2*np.pi*mode/nx
    envelope=np.exp(-.5*(x/(nx/6.4))**2)
    demod=np.exp(-1j*k*x)*envelope
    trajectories=[]
    for channel in range(nchan):
        arr=averaged[:,channel]
        a=np.sum(arr*demod[None,:],axis=1)/envelope.sum()
        norm=max(1e-12,float(abs(a[0])))
        amplitudes=np.abs(a)/norm
        phases=np.unwrap(np.angle(a*np.conj(a[0])))
        energy=np.abs(arr)**2
        centers=np.array([pos_width(row,x0) for row in energy])
        window=np.arange(max(2,nt//6),min(nt,max_tick+1))
        velocity=float(np.polyfit(window,centers[window,0],1)[0])
        freq=float(np.polyfit(window,phases[window],1)[0])
        trajectories.append(dict(channel=['visible_w0','hidden_fourier_q','unweighted_w_mean'][channel],
          initial_amplitude=float(abs(a[0])), final_amplitude_ratio=float(amplitudes[-1]),
          drift_velocity=velocity,phase_frequency_slope=freq,
          center=[float(y) for y in centers[:,0]],width=[float(y) for y in centers[:,1]],
          demod_amplitude_ratio=[float(v) for v in amplitudes],
          demod_phase=[float(v) for v in phases],
          average_signal_power=[float(v) for v in np.mean(energy,axis=1)]))
    # First: identify genuine signal above seed-to-seed fluctuations.
    qsignals=signals[:,:,1,:]
    qpowers=np.mean(np.abs(qsignals)**2,axis=2)
    coh=np.mean(np.abs(averaged[:,1,:])**2,axis=1)
    spread=np.mean(np.abs(qsignals-averaged[None,:,1,:])**2,axis=(0,2))
    return {'seed_count':ns,'mode_k':mode,'channels':trajectories,
       'hidden_q_coherent_power_over_seed_variance_by_tick':[float(v/max(1e-12,z)) for v,z in zip(coh,spread)],
       'explicit_complex_quadratures_only_in_postprocessing':True}


def mirror_x_words(e, words):
    mappings=symmetry_maps(tuple(tuple(int(v) for v in root) for root in ROOTS))
    p=mappings['mirror_0']
    coords=e.coords.copy();coords[:,0]=(-coords[:,0])%e.shape[0]
    full=np.full(e.shape,-1,dtype=np.int32)
    full[tuple(e.coords.T)]=np.arange(e.nodes)
    target=full[tuple(coords.T)]
    assert np.all(target>=0)
    out=np.zeros_like(words)
    for d, new_d in enumerate(p):
        out[target]|=(((words>>np.uint32(d))&1)<<np.uint32(new_d))
    return out


def symmetry_control(e,seed=1901,ticks=5):
    rng=np.random.default_rng(seed)
    state=rng.integers(0,1<<24,size=e.nodes,dtype=np.uint32)
    transformed=mirror_x_words(e,state)
    exact=[]
    for t in range(ticks+1):
        exact.append(bool(np.array_equal(mirror_x_words(e,state),transformed)))
        if t<ticks:
            state=e.tick(state)
            transformed=e.tick(transformed)
    assert all(exact)
    for _ in range(ticks):
        state=e.tick(state,reverse=True)
        transformed=e.tick(transformed,reverse=True)
    assert np.array_equal(state,mirror_x_words(e,transformed))
    return {'spatial_mirror_x_equivariance':True,'forward_ticks':ticks,
            'bit_exact_reverse':True,'tested_random_sites':e.nodes,'scope':'mirror-transformed same microstate, NOT an independent velocity estimate'}


def run(shape=(32,4,4,16),ticks=24,seeds=2,amplitude=0.105):
    start=time.perf_counter()
    e=BoundedOrbitD4(shape,max_pairs=0)
    prep={};output={}
    for mode in (0,2,-2):
        scans=[]
        for i in range(seeds):
            a,meta=synthesize(190100+i,mode,e,ticks,'bit8',amplitude)
            scans.append(a)
            print(f'mode={mode:+d} seed={i+1}/{seeds} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        # independent seed sets (not x-mirrored) and measurements for +/- k.
        output[str(mode)]=analyze(scans,mode,shape,ticks)
    # Fast free-streaming control uses same thresholding but no collisions.
    free={}
    for mode in (0,2,-2):
        scans=[synthesize(190100+i,mode,e,ticks,'free',amplitude)[0] for i in range(seeds)]
        free[str(mode)]=analyze(scans,mode,shape,ticks)
    mirror=symmetry_control(e)
    return {'experiment':'Bit19 independent reconstructed phase-coded D4 packet (not original Bit6 implementation)',
      'kernel':'exact Bit8 collision + 4D streaming; bounded inner caches; engine cache disabled',
      'shape':list(shape),'nodes':e.nodes,'ticks':ticks,'seeds':seeds,
      'seeds_are_independent_across_ensemble_but_common_thresholds_within_quadratures':True,
      'q_mode':1,'x_modes':[0,2,-2],'amplitude':amplitude,
      'spatial_aperture_sigma':shape[0]/6.4,
      'microstate':'finite 24 Boolean heading bits per site','reference_omega':'sqrt((k²+q²)/2), only used in initial directional weights',
      'microscopic_rule_changed':False,'initialization_reconstruction_limit':'Published Bit6 description only; exact original seed schedule, initial spectral eigenvectors and diagnostics unavailable',
      'measurements':output,'free_streaming_control':free,'symmetry_control':mirror,
      'conservation_checked_every_real_trajectory_first_last':True,
      'elapsed_seconds':round(time.perf_counter()-start,3)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shape',nargs=4,type=int,default=[32,4,4,16]);ap.add_argument('--ticks',type=int,default=24);ap.add_argument('--seeds',type=int,default=2)
    ap.add_argument('--amplitude',type=float,default=.105);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    result=run(tuple(a.shape),a.ticks,a.seeds,a.amplitude)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(f"SAVED {a.out} elapsed={result['elapsed_seconds']:.1f}s",flush=True)

if __name__=='__main__':main()