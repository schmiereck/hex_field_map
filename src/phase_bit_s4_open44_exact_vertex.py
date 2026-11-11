#!/usr/bin/env python3
"""Exhaustive 2^24-input exact local Bit8 *quadratic* collision vertex.

Enumerates all 16,777,216 local words (no random states): for input sign
s_j=2X_j-1, output sign t_i=2C(X)_i-1, the mixed Bernoulli probability
Hessian is H_ijk=2 E[(t_i-s_i)s_j s_k] for j != k; H_ijj=0.
For complex input direction vectors a,b,
 V_i(a,b)=2 E[(t_i-s_i)((s.a)(s.b)-sum(a_j*b_j))].

The Boolean input enumeration and collision evaluation are exact. Vector
contraction uses external float64 approximate eigenvectors from Bit24.
This measures a single local collision near iid half-filling, *not* the
nonlocal finite-time packet response on a correlated background.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from phase_bit_s4_open44_parallel import _collision_parallel,LEFT,RIGHT,MUT,RANK,TABLES
from phase_bit_s4_open43_mode_prep import ROOTS,select,matrix_from_json
from phase_bit_s4_open44_outgoing import K,Q

BITS=np.uint32(1)<<np.arange(24,dtype=np.uint32)
N=1<<24

def exact_vertex(root='.',chunk=32768):
    A=matrix_from_json(Path(root)/'data/bit24_collision_matrix.json')
    vR,_=select(A,K,Q,0,1)
    vL,_=select(A,K,Q,0,-1)
    vW,_=select(A,K,Q,3,1)
    pairs={'right_left':(vR,vL),'right_w':(vR,vW)}
    acc={name:np.zeros(24,dtype=np.complex128) for name in pairs}
    # block means provide a diagnostic of convergence but do not replace exact enumeration.
    observed=[0]*24
    started=time.perf_counter()
    for start in range(0,N,chunk):
        end=min(start+chunk,N)
        words=np.arange(start,end,dtype=np.uint32)
        out=_collision_parallel(words,LEFT,RIGHT,MUT,RANK,TABLES)
        s=((((words[:,None]&BITS[None,:])!=0).astype(np.int8))*2-1)
        t=((((out[:,None]&BITS[None,:])!=0).astype(np.int8))*2-1)
        diff=(t-s).astype(np.float64)
        for name,(a,b) in pairs.items():
            z=(s@a)*(s@b)-np.dot(a,b)
            acc[name]+=2*(diff.T@z)
        if (start//chunk+1)%32==0 or end==N:
            print(f'EXACT VERTEX {end:,}/{N:,} ({end/N*100:.1f}%) elapsed {time.perf_counter()-started:.1f}s',flush=True)
    phase=np.exp(-1j*(2*K*ROOTS[:,0]+2*Q*ROOTS[:,3]))
    out={'schema':1,'exhaustive_states':N,'state_domain':'all 24-bit local words, no Monte Carlo',
       'exact_local_collision':'Bit8 via identical integer gates, independent of all packet simulations',
       'vertex_formula':'V_i=2 mean[(t_i-s_i)((s dot a)(s dot b)-sum_j a_j b_j)]',
       'quadratic_context':'iid half-filled Bernoulli inputs; only local instantaneous second-order perturbation',
       'no_claim':'No evidence by itself of outgoing coherent far-field mode, stable particle, actual energy flow or higher-dimensional mediation',
       'seconds':time.perf_counter()-started,'pairs':{}}
    for name,v in acc.items():
        v=v/N
        z=phase*v
        charge=np.concatenate(([v.sum()],v@ROOTS))
        out['pairs'][name]={'source_vector':[[float(t.real),float(t.imag)] for t in v],
          'source_norm':float(np.linalg.norm(v)),
          'local_charge_error':float(np.max(abs(charge))),
          'after_one_stream_at_2k_2q':{key:[float(val.real),float(val.imag)] for key,val in
             {'density':z.sum(),'current_x':z@ROOTS[:,0],'current_w':z@ROOTS[:,3]}.items()}}
        print('EXACT',name,'norm',out['pairs'][name]['source_norm'],
              'outgoing',out['pairs'][name]['after_one_stream_at_2k_2q'],flush=True)
    p=Path(root)/'data'/'bit25_exact_local_vertex.json';p.write_text(json.dumps(out,indent=2)+'\n')
    return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--chunk',type=int,default=32768);a=p.parse_args();exact_vertex(a.root,a.chunk)
