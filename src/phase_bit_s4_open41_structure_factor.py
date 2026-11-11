#!/usr/bin/env python3
"""Bit-Test 22: equilibrium dynamical structure factor on an exact 24-bit D4 gas.

Computes stationary density/current Fourier-time series from independent
unbiased Bernoulli(1/2) occupation states; exact Bit-8 collision and
reversible D4 direction streaming (Bit-20 Numba implementation), with
free-streaming controls using exactly the same initial microstates.

Here S(kx,q,omega) is truly an equilibrium fluctuation power spectrum,
not the earlier forced-response amplitude. Complex values and FFT are
EXTERNAL measurements only; all microscopic updates remain Boolean.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4
from phase_bit_s4_open27_active_collision import ROOTS

ROOTS_NP=np.asarray(ROOTS,dtype=np.int16)
BYTE_DENS=np.array([i.bit_count() for i in range(256)],dtype=np.int16)
BYTE_X=np.array([[sum(ROOTS_NP[base+d,0] for d in range(8) if (s>>d)&1) for s in range(256)] for base in (0,8,16)],dtype=np.int16)
BYTE_W=np.array([[sum(ROOTS_NP[base+d,3] for d in range(8) if (s>>d)&1) for s in range(256)] for base in (0,8,16)],dtype=np.int16)


def site_observables(words):
    a=(words&255).astype(np.int16)
    b=((words>>8)&255).astype(np.int16)
    c=((words>>16)&255).astype(np.int16)
    density=BYTE_DENS[a]+BYTE_DENS[b]+BYTE_DENS[c]
    jx=BYTE_X[0,a]+BYTE_X[1,b]+BYTE_X[2,c]
    jw=BYTE_W[0,a]+BYTE_W[1,b]+BYTE_W[2,c]
    return density,jx,jw


def spatial_probes(engine, state, modes, bins):
    """Fourier coefficients of density, x-current, w-current for each (km,qm)."""
    observed=site_observables(state)
    out=np.empty((len(modes),3),dtype=np.complex128)
    for component,values in enumerate(observed):
        sums=np.bincount(bins,weights=values,minlength=engine.shape[0]*engine.shape[3])
        grid=sums.reshape(engine.shape[0],engine.shape[3]); fft=np.fft.fft2(grid)
        for j,(km,qm) in enumerate(modes):
            out[j,component]=fft[km%engine.shape[0],qm%engine.shape[3]]/np.sqrt(engine.nodes)
    return out


def run(shape=(48,4,4,32),ticks=80,seeds=8,k_modes=(0,2,3,4,-2,-3,-4),q_physical_indices=(1,2,3),seed0=412200,free_control=True,reverse_audit=True):
    started=time.perf_counter()
    e=NumbaOrbitD4(shape)
    if shape[3]%16:raise ValueError('w period must be multiple of 16 to preserve physical q')
    qs=[i*(shape[3]//16) for i in q_physical_indices]
    modes=[(k,q) for q in qs for k in k_modes]
    bins=e.coords[:,0].astype(np.int64)*shape[3]+e.coords[:,3]
    result=np.zeros((seeds,ticks+1,len(modes),3),dtype=np.complex64)
    controls=np.zeros_like(result) if free_control else None
    audit_log=[]
    for seed in range(seeds):
        rng=np.random.default_rng(seed0+seed)
        initial=rng.integers(0,1<<24,size=e.nodes,dtype=np.uint32)
        state=initial.copy();free_state=initial.copy()
        conserved=e.invariants(state)
        first_tick=time.perf_counter()
        for tick in range(ticks+1):
            result[seed,tick]=spatial_probes(e,state,modes,bins)
            if free_control:controls[seed,tick]=spatial_probes(e,free_state,modes,bins)
            if tick<ticks:
                state=e.tick(state)
                if free_control:free_state=e.stream(free_state)
            if tick in (ticks//2,ticks):assert e.invariants(state)==conserved
        reversible=None
        if reverse_audit and seed==0:
            final=state.copy()
            for _ in range(ticks):state=e.tick(state,reverse=True)
            reversible=bool(np.array_equal(state,initial)); assert reversible
            assert np.array_equal(final,e.tick(initial) if ticks==1 else final)
        audit_log.append({'seed':seed0+seed,'seconds':round(time.perf_counter()-first_tick,3),
                          'number_four_momenta_conserved':True,'exact_roundtrip_test':reversible})
        print(f'Bit22 equilibrium seed={seed+1}/{seeds} shape={shape} ticks={ticks} seconds={audit_log[-1]["seconds"]}',flush=True)
    meta={'experiment':'Bit22 stationary equilibrium density/current D4 structure factor',
          'shape':list(shape),'physical_nodes':e.nodes,'ticks':ticks,'seeds':seeds,'seed0':seed0,
          'initial_states':'24 iid fair bits per physical D4 site (uniform invariant measure of bijective full evolution)',
          'physical_q_modes':[int(i) for i in q_physical_indices],
          'k_modes':[int(k) for k in k_modes],
          'mode_pairs':[{'k_mode':int(k),'q_mode':int(q),'k_physical':2*np.pi*k/shape[0], 'q_physical':2*np.pi*q/shape[3]} for k,q in modes],
          'probes':['density','current_x','current_w'],
          'fourier_normalization':'2D spatial FFT / sqrt(physical D4 nodes)',
          'dynamics':'unchanged deterministic Bit8 + streaming via audited Bit20 Numba',
          'free_streaming_same_start':free_control,'reverse_audit_on_first_seed':reverse_audit,
          'audit':audit_log,'elapsed_seconds':time.perf_counter()-started}
    return meta, {'equilibrium_bit8':result,**({'equilibrium_free':controls} if controls is not None else {})}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shape',type=int,nargs=4,default=[48,4,4,32]);ap.add_argument('--ticks',type=int,default=80);ap.add_argument('--seeds',type=int,default=8)
    ap.add_argument('--seed0',type=int,default=412200);ap.add_argument('--k-modes',type=int,nargs='+',default=[0,2,3,4,-2,-3,-4]);ap.add_argument('--q-indices',type=int,nargs='+',default=[1,2,3]);ap.add_argument('--out',type=Path,default=Path('data/bit22_equilibrium.npz'));ap.add_argument('--meta',type=Path,default=Path('data/bit22_equilibrium_metadata.json'));ap.add_argument('--no-reverse',action='store_true')
    a=ap.parse_args();meta,data=run(tuple(a.shape),a.ticks,a.seeds,tuple(a.k_modes),tuple(a.q_indices),a.seed0,reverse_audit=not a.no_reverse)
    a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out,**data)
    a.meta.parent.mkdir(parents=True,exist_ok=True);a.meta.write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
