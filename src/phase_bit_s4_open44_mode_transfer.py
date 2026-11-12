#!/usr/bin/env python3
"""Bit25: phase-resolved signed four-state nonlinear scattering susceptibility.

Do NOT interpret the four-state difference as a particle scattering S-matrix.
Exact unchanged Bit8 collision and streaming; wave/complex arithmetic is observation only.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4
from phase_bit_s4_open43_mode_prep import matrix_from_json, select, ROOTS
from phase_bit_s4_open43_pair_collision import prep, ROOTI, BITS

K_VALUES=np.pi*np.array([-2,-1,0,1,2])/8
Q_VALUES=np.pi*np.array([-4,-2,0,2,4])/8


def initialize_observer(engine):
    coords=engine.coords
    # All permitted spatial Fourier modes for length 48, hidden length 32.
    modes=[(float(k),float(q)) for q in Q_VALUES for k in K_VALUES]
    phases=np.stack([np.exp(-1j*(k*coords[:,0]+q*coords[:,3])) for k,q in modes])
    return modes,np.ascontiguousarray(phases)


def signed_response(states):
    # independently signed difference, unlike the XOR bit diagnostic
    masks=BITS[None,:]
    f0,fA,fB,fAB=states
    r=((fAB[:,None]&masks)!=0).astype(np.int8)
    r-=((fA[:,None]&masks)!=0).astype(np.int8)
    r-=((fB[:,None]&masks)!=0).astype(np.int8)
    r+=((f0[:,None]&masks)!=0).astype(np.int8)
    obs=np.column_stack((r.sum(axis=1),r@ROOTI[:,0],r@ROOTI[:,3])).astype(np.float64)
    return obs, int(np.count_nonzero(r)),int(np.count_nonzero(np.any(r,axis=1)))


def experiment(A, *, shape=(48,4,4,32), amplitude=.22, separation=16,
               phase=0., branch='left', seeds=8,ticks=14,seed0=250100,reverse_first=True):
    e=NumbaOrbitD4(tuple(shape));modes,phases=initialize_observer(e)
    k=np.pi/8;q=np.pi/4
    vR,_=select(A,k,q,0,1)
    vB,_=select(A,k,q,0,-1) if branch=='left' else select(A,k,q,3,1)
    spectra=np.empty((seeds,ticks+1,len(modes),3),dtype=np.complex64)
    local=np.empty((seeds,ticks+1,3),dtype=np.int64)
    t0=time.monotonic()
    for s in range(seeds):
        states,details=prep(e,vR,vB,amplitude=amplitude,separation=separation,
                            phase=phase,seed=seed0+s,k=k,q=q)
        initial=[z.copy() for z in states];charges=[e.invariants(z) for z in states]
        for t in range(ticks+1):
            obs,nonzero_bits,nonzero_sites=signed_response(states)
            assert np.max(np.abs(np.sum(obs,axis=0)))==0, 'signed conservation'
            spectra[s,t]=(phases @ obs /e.nodes).astype(np.complex64)
            local[s,t]=[nonzero_bits,nonzero_sites,int(np.abs(obs).sum())]
            if t<ticks:states=[e.tick(z) for z in states]
        assert all(e.invariants(z)==c for z,c in zip(states,charges))
        if reverse_first and s==0:
            back=[z.copy() for z in states]
            for _ in range(ticks):back=[e.tick(z,reverse=True) for z in back]
            assert all(np.array_equal(z,o) for z,o in zip(back,initial))
        if s==0:
            free=[z.copy() for z in initial]
            for t in range(ticks+1):
                ff,nb,ns=signed_response(free)
                assert nb==ns==0 and not np.any(ff), 'free four-state control'
                if t<ticks:free=[e.stream(z) for z in free]
        print(f'{branch} phase={phase:.3f} amp={amplitude:.2f} seed={s+1}/{seeds} '
              f'end sites={local[s,-1,1]} elapsed={time.monotonic()-t0:.1f}s',flush=True)
    name=f'bit25_{branch}_a{round(100*amplitude):02d}_p{int(round(phase/np.pi*2)):02d}'
    meta={'name':name,'experiment':'Bit25 phase resolved mixed signed nonlinear response',
          'shape':list(shape),'nodes':e.nodes,'amplitude':amplitude,'separation':separation,
          'phase':phase,'branch':branch,'seeds':seeds,'ticks':ticks,'seed0':seed0,
          'k_values':K_VALUES.tolist(),'q_values':Q_VALUES.tolist(),
          'modes':modes,'fields':['density','current_x','current_w'],
          'initial_support_disjoint':True,'bit8_rule_unchanged':True,
          'signed_charge_zero_all_times':True,'first_seed_exact_reverse':reverse_first,
          'free_four_state_response_identically_zero':True,
          'notes':['normalized Fourier amplitude is a signed nonlinear mixed response, not a transition probability',
                   'complex data and mode labels are diagnostics, not microphysics',
                   'finite seed ensemble and short finite periodic lattice; no asymptotic scattering']}
    return meta,spectra,local


def summarize(meta,z,local):
    # Ensemble expectation (phase locked), intensity and incoherent variance.
    mean=z.mean(axis=0,dtype=np.complex128)
    power=np.mean(abs(z.astype(np.complex128))**2,axis=0)
    coherent=abs(mean)**2
    seeds=meta['seeds'];t0=4;tt=meta['ticks']
    sm={}
    for im,(k,q) in enumerate(meta['modes']):
        response={}
        for j,field in enumerate(meta['fields']):
            co=float(np.mean(coherent[t0:tt+1,im,j]));p=float(np.mean(power[t0:tt+1,im,j]))
            # unbiased signal-power estimate under i.i.d. seed noise (may be negative):
            signal=(seeds*co-p)/(seeds-1) if seeds>1 else 0
            response[field]={'mean_coherent_power':co,'mean_total_power':p,
                              'unbiased_coherent_power_estimate':float(signal),
                              'coherence_fraction_raw':float(co/p) if p else 0.0,
                              'at_t8_mean_complex':[float(mean[min(8,tt),im,j].real),float(mean[min(8,tt),im,j].imag)]}
        sm[f'k={k/np.pi:+.3f}pi,q={q/np.pi:+.3f}pi']=response
    ranked=[]
    for key,v in sm.items():
        ranked.append({'mode':key,'coh_density':v['density']['unbiased_coherent_power_estimate'],
                       'coh_Jx':v['current_x']['unbiased_coherent_power_estimate'],
                       'coh_Jw':v['current_w']['unbiased_coherent_power_estimate']})
    ranked.sort(key=lambda x:abs(x['coh_density']),reverse=True)
    return {'settings':meta,'first_nonzero_per_seed':[int(np.flatnonzero(row[:,0]>0)[0]) if np.any(row[:,0]>0) else None for row in local],
            'mean_signed_channel_sites':local[:,:,1].mean(axis=0).tolist(),
            'mean_abs_signed_observable':local[:,:,2].mean(axis=0).tolist(),
            'mode_statistics':sm,'top_density_modes':ranked[:12],
            'noise_note':'finite-N sample mean squared has 1/N power noise bias; unbiased estimator (N*coherent-total)/(N-1); negative estimates are allowed',
            'power_note':'all powers are squared normalized signed mixed response Fourier amplitudes, NOT conserved physical energy'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--matrix',default='data/bit24_collision_matrix.json')
    p.add_argument('--out',default='data/bit25_left_a22_p00')
    p.add_argument('--branch',choices=['left','w'],default='left')
    p.add_argument('--phase',type=float,default=0.)
    p.add_argument('--amp',type=float,default=.22)
    p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--ticks',type=int,default=14)
    p.add_argument('--seed0',type=int,default=250100)
    a=p.parse_args();A=matrix_from_json(a.matrix)
    meta,z,local=experiment(A,amplitude=a.amp,phase=a.phase,branch=a.branch,seeds=a.seeds,ticks=a.ticks,seed0=a.seed0)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(out)+'.npz',mode_amplitude=z,local=local)
    Path(str(out)+'.json').write_text(json.dumps(summarize(meta,z,local),indent=2,ensure_ascii=False)+'\n')
    print('TOP',json.dumps(summarize(meta,z,local)['top_density_modes'][:5],indent=2))

if __name__=='__main__':main()
