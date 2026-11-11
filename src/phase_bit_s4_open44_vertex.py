#!/usr/bin/env python3
"""Supplemental Bit25 local quadratic susceptibility at independent half-filling.

For exact local Boolean collision C, X_i~Ber(1/2), s_j=2X_j-1,
t_i=2C(X)_i-1, the Hessian (j != k) of the mean output is
H_{ijk} = 2 E[t_i s_j s_k], with H_{ijj}=0.

For the interaction-induced part C(X)-X, the complex contraction with
incoming eigenmode vectors a,b is
V_i(a,b)=2 E[(t_i-s_i) ((s.a)(s.b) - a.b)].
Uniform 24-bit words are sampled independently; real/complex numbers
occur in this external perturbative analysis, never in the bit rule.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4
from phase_bit_s4_open44_parallel import ParallelOrbitD4
from phase_bit_s4_open43_mode_prep import ROOTS,select,matrix_from_json
from phase_bit_s4_open44_outgoing import K,Q,CHANNELS

BITS=np.uint32(1)<<np.arange(24,dtype=np.uint32)


def estimate(root='.',blocks=12,samples_per_block=40000,seed=255001):
    A=matrix_from_json(Path(root)/'data/bit24_collision_matrix.json')
    right,_=select(A,K,Q,0,1)
    left,_=select(A,K,Q,0,-1)
    w,_=select(A,K,Q,3,1)
    pairs={'x_right_x_left':(right,left),'x_right_w':(right,w)}
    e=ParallelOrbitD4((4,4,4,4))
    rng=np.random.default_rng(seed)
    results={name:[] for name in pairs}
    beg=time.perf_counter()
    for block in range(blocks):
        s=rng.integers(0,1<<24,size=samples_per_block,dtype=np.uint32)
        t=e.collision(s[:e.nodes]) # first use to compile
        from phase_bit_s4_open44_parallel import _collision_parallel,LEFT,RIGHT,MUT,RANK,TABLES
        out=_collision_parallel(s,LEFT,RIGHT,MUT,RANK,TABLES)
        sign=((((s[:,None]&BITS[None,:])!=0).astype(np.int8))*2-1).astype(np.int8)
        out_sign=((((out[:,None]&BITS[None,:])!=0).astype(np.int8))*2-1).astype(np.int8)
        diff=(out_sign-sign).astype(np.float32)
        for name,(va,vb) in pairs.items():
            z=(sign@va)*(sign@vb)-np.dot(va,vb)
            # 2 * expectation gives exact mixed probability derivative.
            source=2*(diff.T@z)/samples_per_block
            results[name].append(source)
        print('vertex batch',block+1,'/',blocks,'elapsed',round(time.perf_counter()-beg,2),flush=True)
    report={'schema':1,'experiment':'exact Bit8 collision local half-filling quadratic susceptibility',
            'reference':'H_ijk=2 E[(2C_i-1) s_j s_k] for j!=k, diagonal zero',
            'sample_blocks':blocks,'samples_per_block':samples_per_block,'total_samples':blocks*samples_per_block,
            'same_uniform_ensemble_for_both_mode_pairs':True,'local_charge_vertex_zero_by_exact_conservation':True,
            'warning':'This is local one-tick iid half-filling susceptibility; it does NOT model the correlated background at finite-time packet intersection.',
            'pairs':{}}
    phases=np.exp(-1j*(2*K*ROOTS[:,0]+2*Q*ROOTS[:,3]))
    for name,vectors in results.items():
        v=np.asarray(vectors)
        mean=v.mean(axis=0);err=v.std(axis=0,ddof=1)/np.sqrt(blocks)
        tr=np.stack([np.ones(24),ROOTS[:,0],ROOTS[:,3]],axis=0)
        charges=tr@mean
        # microscopic conservation means every sample's source has charge sum=0
        charge_error=max(abs(charges))
        # Streaming of a newly created kinetic collision vertex can induce
        # nonzero outgoing Fourier density and currents at 2k,2q.
        out=phases*mean
        projections={n:[float(z.real),float(z.imag)] for n,z in [('rho',out.sum()),('jx',out@ROOTS[:,0]),('jw',out@ROOTS[:,3])]} 
        vmean=np.linalg.norm(mean)
        noise=np.sqrt(np.sum(np.mean(abs(v-mean)**2,axis=0)/blocks))
        report['pairs'][name]={'source_vector_complex':[[float(z.real),float(z.imag)] for z in mean],
           'source_vector_complex_standard_error':[float(abs(z)) for z in err],
           'source_vector_l2':float(vmean),'source_vector_l2_standard_error_scale':float(noise),
           'five_charge_violation_mc':float(max(abs(np.concatenate(([mean.sum()],mean@ROOTS))))),
           'outgoing_after_one_streaming_at_2k_2q':projections}
    pth=Path(root)/'data'/'bit25_local_vertex.json';pth.write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--blocks',type=int,default=12);p.add_argument('--samples',type=int,default=40000);a=p.parse_args()
 r=estimate(a.root,a.blocks,a.samples)
 for n,p in r['pairs'].items():print(n,'vertex norm',p['source_vector_l2'],'noise scale',p['source_vector_l2_standard_error_scale'], 'outgoing',p['outgoing_after_one_streaming_at_2k_2q'])
