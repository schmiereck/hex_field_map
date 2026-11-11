#!/usr/bin/env python3
"""Reproducible Bit25 diagnostic figures, no claim of asymptotic scattering."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make(root='.'):
    root=Path(root);p=root/'figures';p.mkdir(parents=True,exist_ok=True)
    cases={'weak':'x + x, ε=0.11','baseline':'x + x, ε=0.22','strong':'x + x, ε=0.30','xw':'x + w, ε=0.22'}
    fig,ax=plt.subplots(figsize=(8.0,4.4))
    for case,label in cases.items():
        v=np.load(root/'data'/f'bit25_{case}.npz')
        s=v['stats'][:,:,:,0].mean((0,1))
        ax.plot(np.arange(len(s)),s,marker='o',markersize=3,label=label)
    ax.set(xlabel='Tick',ylabel='Knoten mit verbundenem XOR-Anteil',title='Nichtlinearer Einflussbereich zweier D4-Pakete')
    ax.set_xlim(0,12);ax.grid(alpha=.25);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(p/'bit25_nonlinear_growth.svg');fig.savefig(p/'bit25_nonlinear_growth.png',dpi=160);plt.close(fig)

    v=np.load(root/'data'/'bit25_baseline.npz')['values'].astype(np.complex128)
    phases=np.arange(4)*np.pi/2
    H=np.mean(v*np.exp(-1j*phases)[None,:,None,None,None],axis=1)
    fig,ax=plt.subplots(figsize=(8.0,4.4));tt=np.arange(H.shape[1])
    for c,label in enumerate(['Dichte ρ','Strom Jx','Strom Jw']):
        h=H[:,:,1,c]
        mean=abs(h.mean(axis=0))
        se=np.sqrt(np.sum(abs(h-h.mean(axis=0))**2,axis=0)/(len(h)*(len(h)-1)))
        ax.plot(tt,mean,marker='o',markersize=3,label=f'{label}: |Mittel|')
        ax.plot(tt,se,linestyle=':',linewidth=1,label=f'{label}: Standardfehler')
    ax.axvline(4,linestyle='--',linewidth=.8)
    ax.set(xlabel='Tick',ylabel='Normierte komplexe Fourier-Amplitude',
           title='Phasengekoppelte Antwort im Mischkanal (2kx, 2q)')
    ax.set_xlim(0,12);ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(p/'bit25_sum_coherence.svg');fig.savefig(p/'bit25_sum_coherence.png',dpi=160);plt.close(fig)

    d=json.loads((root/'data'/'bit25_exact_local_vertex.json').read_text())
    names=['right_left','right_w'];obs=['density','current_x','current_w']
    fig,ax=plt.subplots(figsize=(8.0,4.4));xs=np.arange(3);width=.34
    for ix,n in enumerate(names):
        vals=d['pairs'][n]['after_one_stream_at_2k_2q']
        ys=[abs(complex(*vals[o])) for o in obs]
        ax.bar(xs+(ix-.5)*width,ys,width,label='x rechts + '+('x links' if ix==0 else 'w'))
    ax.set_xticks(xs,['Dichte ρ','Jx','Jw']);ax.set_ylabel('Betrag des lokalen komplexen Mischkoeffizienten')
    ax.set_title('Exakter lokaler quadratischer Vertex (alle 2²⁴ Zustände)')
    ax.legend();ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(p/'bit25_exact_local_vertex.svg');fig.savefig(p/'bit25_exact_local_vertex.png',dpi=160);plt.close(fig)
    return [str(x) for x in sorted(p.glob('bit25_*.svg'))]

if __name__=='__main__':
 print('\n'.join(make(Path(__file__).resolve().parent.parent)))
