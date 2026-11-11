#!/usr/bin/env python3
"""Bit22: density/current dynamic structure factors and falsifiable peak diagnostics.

Equilibrium S of independent Bernoulli-half-filled microstates is fundamentally
separate from earlier finite-amplitude four-trajectory driven response spectra.
All spectra are EXTERNAL diagnostics. Finite-window peaks alone are not modes.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from phase_bit_s4_open27_active_collision import ROOTS

PROBES=['density','current_x','current_w','current_parallel']

def window_fft(z):
    """FFT with seedwise temporal mean removed and Hann taper; arrays seed×time×probe."""
    z=np.asarray(z,dtype=np.complex128)
    t=z.shape[1]
    hann=np.hanning(t)
    return np.fft.fft((z-z.mean(axis=1,keepdims=True))*hann[None,:,None],axis=1)/np.sqrt(np.sum(hann*hann))

def longitudinal(a,k,q):
    if k==0 and q==0:raise ValueError('zero wavevector')
    jm=(k*a[...,1]+q*a[...,2])/np.hypot(k,q)
    return np.concatenate((a,jm[...,None]),axis=-1)

def spectral(a,k,q):
    f=window_fft(longitudinal(a,k,q))
    autos=(np.abs(f)**2).mean(axis=0)    # ω × 4
    cross=(f[:,:,0]*np.conj(f[:,:,3])).mean(axis=0)
    coherence=(np.abs(cross)**2/np.maximum(1e-30,autos[:,0]*autos[:,3]))
    oms=2*np.pi*np.fft.fftfreq(a.shape[1])
    return oms,autos,coherence,f

def free_omegas(k,q):
    roots=np.asarray(ROOTS,dtype=float)
    ws=np.sort(np.unique(np.round(k*roots[:,0]+q*roots[:,3],8)))
    return ws

def peaks_of(p,omega,num=4):
    median=float(np.median(p[(np.abs(omega)>.16)&(np.abs(omega)<2.95)]))
    sorted_idx=np.argsort(omega)
    freq,power=omega[sorted_idx],p[sorted_idx]
    ind,_=find_peaks(power, prominence=max(1e-9,median*0.8),distance=2)
    ind=[i for i in ind if abs(freq[i])>.16 and abs(freq[i])<2.95]
    ind=sorted(ind,key=lambda i:power[i],reverse=True)[:num]
    return [(float(freq[i]),float(power[i])) for i in ind]

def frequency_line_fraction(omega,p,k,q):
    dω=2*np.pi/len(omega)
    theory=free_omegas(k,q)
    # all theoretically allowed single-particle free streaming phase branches,
    # except |ω|<0.16, where mean subtraction makes spectral weight ambiguous.
    valid=np.abs(omega)>.16
    fit=np.zeros_like(valid)
    for w in theory:
        if abs(w)>.08:
            fit |= np.abs(np.angle(np.exp(1j*(omega-w))))<=1.15*dω
    return float(p[valid&fit].sum()/max(1e-20,p[valid].sum())),[float(w) for w in theory]

def summarize_case(a,meta,modeidx,source):
    shape=meta['shape']; k=meta['mode_pairs'][modeidx]['k_physical'];q=meta['mode_pairs'][modeidx]['q_physical']
    oms,auto,coh,f=spectral(a[:,:,modeidx,:],k,q)
    rho=auto[:,0]; jx=auto[:,1]; jw=auto[:,2]; jl=auto[:,3]
    train=f[:len(f)//2]; test=f[len(f)//2:]
    trainpow=(abs(train[:,:,0])**2).mean(axis=0);testpow=(abs(test[:,:,0])**2).mean(axis=0)
    top=peaks_of(rho,oms,5)
    dω=2*np.pi/len(oms)
    line_frac,theory=frequency_line_fraction(oms,rho,k,q)
    peaks=[]
    for w,p in top:
        idx=int(np.argmin(abs(oms-w)))
        train_peaks=peaks_of(trainpow,oms,8);test_peaks=peaks_of(testpow,oms,8)
        matched_train=any(abs(ww-w)<=1.1*dω for ww,_ in train_peaks)
        matched_test=any(abs(ww-w)<=1.1*dω for ww,_ in test_peaks)
        near_free=any(abs(np.angle(np.exp(1j*(w-freq))))<=1.15*dω for freq in theory)
        peaks.append({'omega':w,'density_power':p,'jx_power':float(jx[idx]),'jw_power':float(jw[idx]),
                      'longitudinal_current_power':float(jl[idx]),'density_longitudinal_current_coherence':float(coh[idx]),
                      'peak_in_train':matched_train,'peak_in_heldout':matched_test,
                      'within_one_bin_of_free_streaming_phase':near_free})
    upper=rho[(oms>=.16)&(oms<=2.95)]; lower=rho[(oms<=-.16)&(oms>=-2.95)]
    return {'source':source,'q':float(q),'k':float(k),'q_mode':meta['mode_pairs'][modeidx]['q_mode'],
            'k_mode':meta['mode_pairs'][modeidx]['k_mode'],
            'seed_count':len(a),'nt':a.shape[1], 'frequency_bin_width':float(dω),
            'density_integrated_positive_frequency':float(upper.sum()),
            'density_integrated_negative_frequency':float(lower.sum()),
            'density_free_streaming_lines_power_fraction':line_frac,
            'free_streaming_phase_branches':theory,'peaks':peaks}

def driven_response(bit20_dir):
    """Coherent packet FFT from previously archived Bit20 raw response (NOT equilibrium S)."""
    z=np.load(Path(bit20_dir)/'data/bit20_raw.npz')
    meta=json.load(open(Path(bit20_dir)/'data/bit20_metadata.json'))
    result={}
    for q in (1,2,3):
        for k in (0,3,-3):
            key=f'q{q}_k{k:+d}_bit8'
            a=z[key]   # seed×25×3 channels×48 x-positions
            nx=a.shape[-1];xx=np.arange(nx);kreal=2*np.pi*k/nx
            # Fourier k at each time, from spatially localized coherent packet.
            ak=(a[:,:,1,:]*np.exp(-1j*kreal*xx)[None,None,:]).sum(axis=-1)
            mean=ak.mean(axis=0)
            han=np.hanning(len(mean));ft=np.fft.fft(mean*han);freq=2*np.pi*np.fft.fftfreq(len(mean))
            power=np.abs(ft)**2/max(1e-16,(han*han).sum())
            peak=peaks_of(power,freq,4)
            result[f'q{q}_k{k:+d}']={'q':float(q*np.pi/8),'k':float(kreal),'nt':len(mean),
               'time_frequency_resolution':float(2*np.pi/len(mean)),
               'coherent_density_response_peak_omega':[round(w,5) for w,p in peak],
               'coherent_density_response_peak_power':[round(p,5) for w,p in peak],
               'important':'Driven response spectrum |FFT(E_seed delta rho)|^2. Not equilibrium S; no current channel stored in Bit20.'}
    return result

def analyze(paths,meta_paths,bit20_dir,output):
    collections={}
    for src,(filename,metafile) in enumerate(zip(paths,meta_paths)):
        meta=json.load(open(metafile))
        with np.load(filename) as z:
            for dynamic in ('equilibrium_bit8','equilibrium_free'):
                arr=z[dynamic]
                name=('w32' if meta['shape'][1]==4 else 'yz8')+'_'+('bit8' if dynamic=='equilibrium_bit8' else 'free')
                allrows=[summarize_case(arr,meta,j,name) for j in range(len(meta['mode_pairs']))]
                collections[name]={'metadata_file':str(Path(metafile).name),'cases':allrows}
    response=driven_response(bit20_dir)
    out={'experiment':'Bit22 equilibrium S_rho,S_jx,S_jw and longitudinal coherence',
         'spectrum_definition':'seed average |FFT_time(Hann(t)*(O(k,q,t)-mean_t O(k,q,t)))|^2 / sum Hann^2',
         'microscopic_equilibrium':'uniform iid Bernoulli(1/2) 24-bit field, invariant under any bijective collision+streaming',
         'time_fft_sign':'numpy forward FFT exp(-i omega_index t); reported frequencies may have sign opposite to e^{-i omega t} physical convention',
         'complexity_control':'separate fixed-seed free streaming; finite 80-tick Hann resolution and 8/4-seed sampling',
         'split_check':'Top frequency bins among first and last half of independent seeds; 1-bin matching. Exploratory, not conclusive mode detection.',
         'limits':['Spectral peaks are not proof of eigenmodes.','Time mean removal suppresses low frequencies.',
                   'Response pulse preparation and equilibrium S probe different ensembles and may disagree nonlinearly.',
                   'No finite-volume or frequency-continuum extrapolation.'],
         'equilibrium':collections,'bit20_driven_coherent_density_response':response}
    Path(output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    return out

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--equilibrium',nargs='+',required=True);a.add_argument('--metadata',nargs='+',required=True);a.add_argument('--bit20-dir',required=True);a.add_argument('--out',required=True)
    q=a.parse_args();analyze(q.equilibrium,q.metadata,q.bit20_dir,q.out)
