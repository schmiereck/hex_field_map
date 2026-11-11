#!/usr/bin/env python3
"""Bit25 analysis: coherent complex phase-cycled connected response vs raw RMS.

Predeclared diagnostic window t=5..10 (after causal onset at t=4). Each
nonzero mode tested by random independent complex phase rotations of seed
harmonics; these preserve within-seed magnitude and temporal correlation.
Bonferroni correction across the 9 nonzero spatial channels x 3 observables.
No claim of conserved microscopic energy or a scattering S-matrix.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from phase_bit_s4_open44_outgoing import CHANNELS,OBS

PRE=(5,10)

def analyze(root,name,permutations=2048,seed=2555):
    base=Path(root)/'data'/f'bit25_{name}'
    meta=json.loads(base.with_suffix('.json').read_text())
    v=np.load(base.with_suffix('.npz'))['values'].astype(np.complex128)
    # The (0,0) charge channel is exactly zero by integer identity; remove
    # recorded single-precision Fourier roundoff of order 1e-9.
    v[:,:,:,0,:]=0.
    stats=np.load(base.with_suffix('.npz'))['stats']
    n,phases,nt,nc,no=v.shape
    if phases!=4 or not np.allclose(meta['phases'],[0,np.pi/2,np.pi,3*np.pi/2]):
        raise ValueError('requires exactly four equally spaced phase controls')
    # H1 is phase-dependent connected response; changing phi for B means
    # this response rotates predictably when a coherent phase-locked coupling exists.
    phi=np.asarray(meta['phases'])
    harmonic=np.einsum('sptco,p->stco',v,np.exp(-1j*phi),optimize=True)/4
    harmonic2=np.einsum('sptco,p->stco',v,np.exp(-2j*phi),optimize=True)/4
    harmonic0=v.mean(axis=1)
    mean=harmonic.mean(axis=0)
    rms=np.sqrt(np.mean(abs(v)**2,axis=(0,1)))
    cycles=np.sqrt(np.mean(abs(harmonic)**2,axis=0))
    coh=(abs(mean)**2)/(cycles**2+1e-30)
    a,b=PRE;lo=min(nt-1,a);hi=min(nt-1,b)
    if lo>=hi:raise ValueError('time series too short')
    selected=harmonic[:,lo:hi+1]
    # Preselected time-window joint statistic, no selecting a lucky tick.
    signal=np.sum(abs(selected.mean(axis=0))**2,axis=0)
    uncoh=np.sum(np.mean(abs(selected)**2,axis=0),axis=0)
    estim=(n*signal/(uncoh+1e-30)-1)/(n-1) if n>1 else np.zeros_like(signal)
    rng=np.random.default_rng(seed)
    null=np.zeros((permutations,nc,no),dtype=np.float64)
    for j in range(permutations):
        # Seed-sign / phase scramble: balanced circular complex null, preserving
        # each trajectory's temporal correlation, broad-band spectrum and scale.
        offsets=rng.integers(0,4,size=n)
        z=np.sum(selected*(1j**offsets)[:,None,None,None],axis=0)/n
        null[j]=np.sum(abs(z)**2,axis=0)
    p=(1+(null>=signal).sum(axis=0))/(permutations+1)
    p[0,:]=1.0;estim[0,:]=0.;coh[:,0,:]=0.
    klist=list(CHANNELS)
    result={'schema':1,'case':name,'seeds':n,'phases':meta['phases'],'time_window_inclusive':[lo,hi],
        'channel_names':klist,'observables':list(OBS),
        'seed_null':'independent random fourth-root-of-unity phases, constant across time for each seed',
        'null_permutations':permutations,'multiple_testing':'Bonferroni 27 = 9 nonzero spatial channels x 3 observables; 0,0 conserved separately',
        'sample_size_limitation':'The finite seeded estimates and permutation test describe only this specific phase-cycle and preparation',
        'max_charges_zero':bool(np.max(abs(v[:,:,:,0,:]))<1e-7),
        'max_initial_connected':float(np.max(abs(v[:,:,0]))),
        'max_before_causal_onset':float(np.max(abs(v[:,:,:4]))),
        'interaction_sites_by_tick':stats[:,:,:,0].mean(axis=(0,1)).tolist(),
        'sum_of_phase_harmonic1_squares_by_channel':{k:{OBS[d]:float(signal[c,d]) for d in range(no)} for c,k in enumerate(klist)},
        'coherent_fraction_by_channel':{k:{OBS[d]:float(signal[c,d]/(uncoh[c,d]+1e-30)) for d in range(no)} for c,k in enumerate(klist)},
        'unbiased_coherent_fraction_estimate':{k:{OBS[d]:float(estim[c,d]) for d in range(no)} for c,k in enumerate(klist)},
        'permutation_p_value':{k:{OBS[d]:float(p[c,d]) for d in range(no)} for c,k in enumerate(klist)},
        'bonferroni_significant':{k:{OBS[d]:bool(p[c,d] < .05/27) for d in range(no)} for c,k in enumerate(klist)},
        'rms_four_state_amplitude_by_tick':{k:{OBS[d]:rms[:,c,d].tolist() for d in range(no)} for c,k in enumerate(klist)},
        'complex_phase_cycled_mean_by_tick':{k:{OBS[d]:[[float(z.real),float(z.imag)] for z in mean[:,c,d]] for d in range(no)} for c,k in enumerate(klist)},
        'phase_cycled_rms_by_tick':{k:{OBS[d]:cycles[:,c,d].tolist() for d in range(no)} for c,k in enumerate(klist)},
        'phase_second_harmonic_rms':{k:{OBS[d]:float(np.sqrt(np.mean(abs(harmonic2[:,lo:hi+1,c,d])**2))) for d in range(no)} for c,k in enumerate(klist)},
        'phase_independent_rms':{k:{OBS[d]:float(np.sqrt(np.mean(abs(harmonic0[:,lo:hi+1,c,d])**2))) for d in range(no)} for c,k in enumerate(klist)},
        'caution':'Connection is an ensemble-averaged signed four-state occupation difference, not directly energy, particle scattering, or higher-dimensional causal mediation.'}
    pth=Path(root)/'data'/f'bit25_{name}_analysis.json';pth.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--names',nargs='+',default=['baseline']);p.add_argument('--permutations',type=int,default=2048)
    a=p.parse_args()
    for name in a.names:
        r=analyze(a.root,name,a.permutations)
        print(name,'onsites',round(r['interaction_sites_by_tick'][-1]),'coherence:')
        for k in ('sum','sum_x','sum_w','carrier','hidden'):
            print(' ',k,'frac',r['coherent_fraction_by_channel'][k], 'p',r['permutation_p_value'][k])
