#!/usr/bin/env python3
"""Bit23: statistical collision tangent about independent half-filled Bernoulli bits.

K_ij = d E[Y_i] / d p_j at p=(1/2,...,1/2).
At uniform product equilibrium this is EXACTLY E[(2Y_i-1)(2X_j-1)].
The collision Y=C(X) is the unchanged 24-bit deterministic Bit8 involution.
The Jacobian is the derivative of an ENSEMBLE mean, not of an individual bit.
K_B4 uses B4 group-average, and K_exact_constraints adds the proven
number/momentum charge and involution constraints. This DOES NOT assert
that multi-tick one-point evolution closes without correlations.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import _collision_array,LEFT,RIGHT,MUT,RANK,TABLES,collide_numba
from phase_bit_s4_open26_symmetric_collision import GROUP,ROOTS

R=np.asarray(ROOTS,dtype=np.float64)
A=np.vstack((np.ones(24),R.T))
P=A.T @ np.linalg.inv(A@A.T) @ A
Q=np.eye(24)-P
PERMS=np.asarray(GROUP,dtype=np.int16)
BITS=np.arange(24,dtype=np.uint32)

def symmetrize(K):
    # K'_{perm(i),perm(j)} = K_{i,j}; apply simultaneous inverse indexing.
    target=np.zeros_like(K)
    for p in PERMS: target[np.ix_(p,p)] += K
    return target/len(PERMS)

def estimate_batches(seeds=8,n_per_seed=100000):
    series=[];counts=[];started=time.perf_counter()
    for a in range(seeds):
        rng=np.random.default_rng(420000+a)
        acc=np.zeros((24,24),dtype=np.float64);changed=0
        left=n_per_seed
        while left:
            n=min(25000,left)
            s=rng.integers(0,1<<24,size=n,dtype=np.uint32)
            t=_collision_array(s,LEFT,RIGHT,MUT,RANK,TABLES)
            changed+=int(np.count_nonzero(s!=t))
            x=((s[:,None]>>BITS)&1).astype(np.float64)*2-1
            y=((t[:,None]>>BITS)&1).astype(np.float64)*2-1
            acc+=y.T@x
            left-=n
        raw=acc/n_per_seed
        series.append(symmetrize((raw+raw.T)*.5))
        counts.append(changed/n_per_seed)
        print(f'K-seed {a+1}/{seeds} activity {counts[-1]:.4f} time {time.perf_counter()-started:.1f}s',flush=True)
    mean=np.mean(series,axis=0)
    return np.array(series),mean,counts

def collision_invariant_check(K):
    return {'left_charge_error_inf':float(np.max(abs(A@K-A))),
       'right_charge_error_inf':float(np.max(abs(K@A.T-A.T))),
       'symmetry_transpose_error_inf':float(np.max(abs(K-K.T))),
       'B4_error_inf':float(max(np.max(abs(K[np.ix_(p,p)]-K)) for p in PERMS)),
       'largest_singular_value':float(np.linalg.svd(K,compute_uv=False)[0])}

def derived(K):
    return P+Q@K@Q

def eig_table(K,kx,q,small=1e-9):
    phase=np.exp(-1j*(kx*R[:,0]+q*R[:,3]))
    D=np.diag(phase)
    M=D@K
    vals,vecs=np.linalg.eig(M)
    idx=np.lexsort((np.angle(vals),-np.abs(vals)))
    vals,vecs=vals[idx],vecs[:,idx]
    modes=[]
    for n,(val,v) in enumerate(zip(vals,vecs.T)):
        vn=v/(np.linalg.norm(v)+1e-30)
        density=abs(np.sum(vn))**2/(24*np.vdot(vn,vn).real)
        jx=abs(R[:,0]@vn)**2/(12*np.vdot(vn,vn).real)
        jw=abs(R[:,3]@vn)**2/(12*np.vdot(vn,vn).real)
        modes.append({'id':n,'eigenvalue':[float(val.real),float(val.imag)],'modulus':float(abs(val)),
          'temporal_phase':float(np.angle(val)),'density_weight':float(density),
          'jx_weight':float(jx),'jw_weight':float(jw),'vector':[[float(z.real),float(z.imag)] for z in vn]})
    return modes

def run(seeds=8,n_per_seed=100000):
    if seeds<2 or n_per_seed<1000:raise ValueError('ensemble too small')
    raw,K,changes=estimate_batches(seeds,n_per_seed)
    cal=derived(K)
    eigc=np.linalg.eigvalsh(cal)
    qs=[np.pi/8,np.pi/4,3*np.pi/8]
    k=np.pi/8
    spectra={str(round(q/np.pi,5)):eig_table(cal,k,q) for q in qs}
    info={'test':'Bit23 collision ensemble tangent K_ij = E[(2Y_i-1)(2X_j-1)] at p_i=1/2',
      'samples':seeds*n_per_seed,'seed_count':seeds,'samples_per_seed':n_per_seed,
      'random_initial_bits':'independent uniform 24-bit local states, not fixed occupation 12',
      'microscopic_rule':'exact Bit8 Numba checked against Bit10',
      'tangent_mean_field_assumption':'multitick one-point perturbations ignore dynamically generated correlations',
      'symmetry_projection':'B4 average and exact left/right five conserved-charge projector',
      'collision_active_fractions':changes,
      'raw_mean_validation':collision_invariant_check(K),
      'constrained_validation':collision_invariant_check(cal),
      'eigenvalues_collision':[float(x) for x in eigc],
      'eigenmodes_by_hidden_q':spectra,'mode_k':k,
      'ensemble_seed_standard_error_max':float(np.max(np.std(raw,axis=0,ddof=1)/np.sqrt(seeds))),
      'limitations':['one-site product-state tangent, not exact many-tick Liouvillian spectrum',
        'uniform unbiased local Bernoulli distribution rather than fixed-density canonical ensemble',
        'eigenvalues are complex diagnostics only; 24-bit dynamics stays integer and reversible']}
    return info,{'K_batches':raw,'K_empirical_B4':K,'K_charge_exact':cal,'charge_projector':P}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=8);p.add_argument('--samples',type=int,default=100000);p.add_argument('--out',type=Path,default=Path('data/bit23_collision_operator.json'));p.add_argument('--npz',type=Path,default=Path('data/bit23_collision_operator.npz'))
    args=p.parse_args();d,arr=run(args.seeds,args.samples)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.npz.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');np.savez_compressed(args.npz,**arr)
    print(json.dumps({k:v for k,v in d.items() if k not in ['eigenmodes_by_hidden_q','eigenvalues_collision']},indent=2))
if __name__=='__main__':main()
