#!/usr/bin/env python3
"""Bit22 paired-seed bootstrap diagnostics of free-branch broadening and polarizations."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from phase_bit_s4_open41_spectral_analysis import spectral,window_fft,free_omegas,frequency_line_fraction


def band_bootstrap(bit8,free,k,q,reps=1200,seed=4102):
    rng=np.random.default_rng(seed)
    nt=bit8.shape[1]
    n=len(bit8)
    omegas=2*np.pi*np.fft.fftfreq(nt)
    dω=2*np.pi/nt
    valid=np.abs(omegas)>.16
    matches=np.zeros_like(valid)
    for w in free_omegas(k,q):
        if abs(w)>.08:
            matches|=(np.abs(np.angle(np.exp(1j*(omegas-w))))<=1.15*dω)
    # Individual seed-frequency powers: cheap paired bootstrap reweighting.
    w=np.hanning(nt)
    def powers(array):
        x=array[:,:,0].astype(np.complex128)
        f=np.fft.fft((x-x.mean(axis=1,keepdims=True))*w[None,:],axis=1)
        return abs(f)**2/sum(w*w)
    a,b=powers(bit8),powers(free)
    inds=rng.integers(0,n,size=(reps,n));ap=a[inds].mean(axis=1);bp=b[inds].mean(axis=1)
    af=ap[:,valid&matches].sum(axis=1)/ap[:,valid].sum(axis=1)
    bf=bp[:,valid&matches].sum(axis=1)/bp[:,valid].sum(axis=1)
    pointa=a[:,valid&matches].mean(axis=0).sum()/a[:,valid].mean(axis=0).sum()
    pointb=b[:,valid&matches].mean(axis=0).sum()/b[:,valid].mean(axis=0).sum()
    return {'bit8_fraction':float(pointa),'free_fraction':float(pointb),'difference_bit8_minus_free':float(pointa-pointb),
      'difference_bootstrap_95ci':list(map(float,np.quantile(af-bf,[.025,.975]))),
      'bootstrap_reps':reps,'paired_same_microstate_per_seed':True,
      'free_band_tolerance':'1.15*2pi/(ticks+1), |omega|>0.16'}

def polarization_bootstrap(array,k,q,omega0,reps=1200,seed=4201):
    f=window_fft(array)
    frequency=2*np.pi*np.fft.fftfreq(array.shape[1]);j=int(np.argmin(abs(frequency-omega0)))
    x=abs(f[:,j,1])**2;w=abs(f[:,j,2])**2
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(array),size=(reps,len(array)))
    ratio=(x[idx].mean(axis=1)-w[idx].mean(axis=1))/(x[idx].mean(axis=1)+w[idx].mean(axis=1)+1e-30)
    return {'omega':float(frequency[j]),'jx_power':float(x.mean()),'jw_power':float(w.mean()),
            'polarization_index_jx_minus_jw_over_sum':float((x.mean()-w.mean())/(x.mean()+w.mean())),
            'bootstrap_95ci':list(map(float,np.quantile(ratio,[.025,.975])))}

def run(dataset,metadata):
    result={}
    for p,mfile in zip(dataset,metadata):
        meta=json.load(open(mfile))
        with np.load(p) as z:a=z['equilibrium_bit8'];b=z['equilibrium_free']
        label='yz8' if meta['shape'][1]==8 else 'w32';cases={}
        for qm in (2,4,6):
            ix=next(i for i,r in enumerate(meta['mode_pairs']) if r['k_mode']==3 and r['q_mode']==qm)
            k=meta['mode_pairs'][ix]['k_physical'];q=meta['mode_pairs'][ix]['q_physical']
            row={'free_branch_concentration':band_bootstrap(a[:,:,ix,:],b[:,:,ix,:],k,q,seed=4121+qm)}
            if qm==4:
                row['x_sector_peak']=polarization_bootstrap(a[:,:,ix,:],k,q,k)
                row['w_sector_peak']=polarization_bootstrap(a[:,:,ix,:],k,q,-q)
            cases[str(qm)] = row
        result[label]={'nodes':meta['physical_nodes'],'seeds':meta['seeds'],'ticks':meta['ticks'],'k_mode':3,'physical_k':float(2*np.pi*3/meta['shape'][0]),'q':cases}
    return {'evidence':'Bit22 q-sensitive equilibrium spectral branch comparisons',
            'uncertainty':'Paired bootstrap resamples independent initial fields; finite N and 80 ticks; intervals descriptive.',
            'measured_peaks':'+k for x dominated sector, -q for w dominated sector at physical q=pi/4',
            'cases':result}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--equilibrium',nargs='+',required=True);p.add_argument('--metadata',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=run(a.equilibrium,a.metadata);Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
