#!/usr/bin/env python3
"""Publication-style spectral figures for Bit22. Purely diagnostic."""
from pathlib import Path
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase_bit_s4_open41_spectral_analysis import spectral, free_omegas


def make(prefix='data/bit22_equilibrium_w32',out='figures'):
    meta=json.load(open(prefix+'_metadata.json'))
    data=np.load(prefix+'.npz')
    bit8=data['equilibrium_bit8'];free=data['equilibrium_free']
    Path(out).mkdir(exist_ok=True,parents=True)
    i=next(j for j,row in enumerate(meta['mode_pairs']) if row['q_mode']==4 and row['k_mode']==3)
    k=meta['mode_pairs'][i]['k_physical'];q=meta['mode_pairs'][i]['q_physical']
    freq,ps,coh,f=spectral(bit8[:,:,i,:],k,q)
    _,frees,_,_=spectral(free[:,:,i,:],k,q)
    order=np.argsort(freq); omega=freq[order]
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
    ax=axes[0]
    ax.plot(omega,ps[order,0],lw=1.8,label='Bit-8: Dichte')
    ax.plot(omega,frees[order,0],lw=1.35,linestyle='--',label='Freies Streaming: Dichte')
    for w in free_omegas(k,q):
        if abs(w)>0.04:ax.axvline(w,color='0.82',linestyle=':',linewidth=.7,zorder=-1)
    ax.set_title(r'Bit-Test 22 · $k_x=\pi/8$, $q=\pi/4$ · 12.288 D4-Knoten')
    ax.set_ylabel(r'$S_{\rho\rho}(k,q,\omega)$')
    ax.legend(loc='upper right');ax.grid(alpha=.15)
    ax=axes[1]
    ax.plot(omega,ps[order,1],lw=1.9,label='Bit-8: x-Strom')
    ax.plot(omega,ps[order,2],lw=1.9,label='Bit-8: w-Strom')
    for w in free_omegas(k,q):
        if abs(w)>0.04:ax.axvline(w,color='0.82',linestyle=':',linewidth=.7,zorder=-1)
    ax.set_ylabel(r'$S_{JJ}(k,q,\omega)$');ax.set_xlabel(r'Zeitfrequenz $\omega$ [rad / Tick]')
    ax.legend(loc='upper right');ax.grid(alpha=.15)
    axes[0].set_xlim(-1.45,1.45);fig.savefig(Path(out)/'bit22_density_current_q2.svg');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,constrained_layout=True)
    for ax,qm in zip(axes,[2,4,6]):
        j=next(j for j,row in enumerate(meta['mode_pairs']) if row['q_mode']==qm and row['k_mode']==3)
        kk=meta['mode_pairs'][j]['k_physical'];qq=meta['mode_pairs'][j]['q_physical']
        x,s,_,_=spectral(bit8[:,:,j,:],kk,qq)
        _,fs,_,_=spectral(free[:,:,j,:],kk,qq)
        oi=np.argsort(x); ax.plot(x[oi],s[oi,0],lw=1.8,label='Bit 8')
        ax.plot(x[oi],fs[oi,0],linestyle='--',lw=1.2,label='Frei')
        ax.set_ylabel('Dichteleistung');ax.grid(alpha=.16)
        ax.set_title({2:r'$q=\pi/8$',4:r'$q=\pi/4$',6:r'$q=3\pi/8$'}[qm],loc='left')
        ax.legend(loc='upper right')
    axes[-1].set_xlabel(r'Zeitfrequenz $\omega$ [rad / Tick]');axes[-1].set_xlim(-1.7,1.7)
    fig.suptitle(r'Bit-Test 22: $S_{\rho\rho}$ bei $k_x=\pi/8$')
    fig.savefig(Path(out)/'bit22_q_spectral_comparison.svg');plt.close(fig)

if __name__=='__main__':make()
