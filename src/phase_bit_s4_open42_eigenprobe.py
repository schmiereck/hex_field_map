#!/usr/bin/env python3
"""Bit23: exact Bit8 gas tests of individually seeded eigenvectors of the product-state linear tangent.
The linear eigenvector is used ONLY to prepare finite-bit Bernoulli initial states.
The subsequent dynamics is the unchanged reversible Bit8 collision+streaming.
"""
from __future__ import annotations
import argparse, json,sys,time
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4, audit
from phase_bit_s4_open38_phased_packet import packed_from_threshold
from phase_bit_s4_open27_active_collision import ROOTS
from phase_bit_s4_open42_linear_collision import eig_table,R

BITS=np.arange(24,dtype=np.uint32)

def branch_selection(modes,q,branch):
    target=-np.pi/8 if branch in ['x','mixed'] else -q
    key='jx_weight' if branch=='x' else ('jw_weight' if branch=='w' else 'density_weight')
    scoring=lambda m: (abs(np.angle(np.exp(1j*(m['temporal_phase']-target))))<.07,m[key])
    filtered=[m for m in modes if abs(np.angle(np.exp(1j*(m['temporal_phase']-target))))<.07]
    if not filtered:raise RuntimeError('no eligible branch '+str((q,branch)))
    chosen=max(filtered,key=lambda m:m[key])
    return chosen

def make_initial(e,random,mode_vector,theta,amplitude):
    weights=np.real(np.exp(1j*theta[:,None])*mode_vector[None,:])
    p=np.clip(.5+amplitude*weights,.01,.99)
    return packed_from_threshold(random,p)

def sample_single(e,v,kx,q,ticks,seed,amplitude=.07,free=False,check_reverse=False):
    random=np.random.default_rng(seed).integers(0,65536,size=(e.nodes,24),dtype=np.uint16)
    theta=kx*e.coords[:,0]+q*e.coords[:,3]
    w=np.exp(-1j*theta)
    positive=make_initial(e,random,v,theta,+amplitude)
    negative=make_initial(e,random,v,theta,-amplitude)
    initial_positive=positive.copy()
    initial_negative=negative.copy()
    charge_pos=e.invariants(positive);charge_neg=e.invariants(negative)
    trace=np.empty((ticks+1,24),dtype=np.complex128)
    for t in range(ticks+1):
        x=((positive[:,None]>>BITS)&1).astype(np.int8)
        y=((negative[:,None]>>BITS)&1).astype(np.int8)
        trace[t]=w@(x-y).astype(np.float64)/e.nodes
        if t<ticks:
            positive=e.stream(positive) if free else e.tick(positive)
            negative=e.stream(negative) if free else e.tick(negative)
    assert e.invariants(positive)==charge_pos and e.invariants(negative)==charge_neg
    if not free and check_reverse:
        for _ in range(ticks):
            positive=e.tick(positive,reverse=True);negative=e.tick(negative,reverse=True)
        assert np.array_equal(positive,initial_positive)
        assert np.array_equal(negative,initial_negative)
    return trace

def run(shape=(48,4,4,32),ticks=10,seeds=8,amplitude=.075,selected=('q2_x','q2_w','q1_mixed','q3_x')):
    start=time.perf_counter()
    e=NumbaOrbitD4(shape)
    kx=np.pi/8
    data=np.load(Path(__file__).resolve().parents[1]/'data/bit23_collision_operator.npz')
    K=data['K_charge_exact'];out={};summary=[]
    for code in selected:
        qn,branch=code.split('_')
        q=int(qn[1:])*np.pi/8
        modes=eig_table(K,kx,q)
        choice=branch_selection(modes,q,branch)
        v=np.array([complex(*p) for p in choice['vector']])
        v=v/max(abs(v))
        phase=np.exp(-1j*(kx*R[:,0]+q*R[:,3]))
        M=phase[:,None]*K
        lam=complex(*choice['eigenvalue'])
        residual=float(np.linalg.norm(M@v-lam*v)/np.linalg.norm(v))
        assert residual<1e-10
        traces=[];free_traces=[]
        for s in range(seeds):
            z=sample_single(e,v,kx,q,ticks,seed=423000+s+int(qn[1:])*100+(0 if branch=='x' else 1000 if branch=='w' else 2000),amplitude=amplitude,check_reverse=(s==0))
            traces.append(z)
            print(code,s+1,'/',seeds,'seconds',round(time.perf_counter()-start,2),flush=True)
        ar=np.array(traces);mean=np.mean(ar,axis=0)
        expected=amplitude*np.array([lam**t*v for t in range(ticks+1)])
        vhat=v/np.linalg.norm(v)
        overlap=mean@vhat.conj()
        predicted=expected@vhat.conj()
        orthogonal=np.linalg.norm(mean-overlap[:,None]*vhat[None,:],axis=1)/np.maximum(np.linalg.norm(mean,axis=1),1e-14)
        rel_amplitude=np.abs(overlap)/np.abs(overlap[0])
        ph=np.unwrap(np.angle(overlap*np.conj(overlap[0])))
        per_seed_overlap=ar@vhat.conj()
        init_amps=np.mean(np.abs(per_seed_overlap[:,0]))
        ratio=np.mean(per_seed_overlap[:,1]/per_seed_overlap[:,0])
        ratio_se=float(np.std(per_seed_overlap[:,1]/per_seed_overlap[:,0],ddof=1)/np.sqrt(seeds))
        entry={'code':code,'q':float(q),'k':float(kx),'branch':branch,
               'mode_id':choice['id'],'eigenvalue':choice['eigenvalue'],'eigenvalue_modulus':choice['modulus'],
               'mode_collision_jx_weight':choice['jx_weight'],'mode_collision_jw_weight':choice['jw_weight'],
               'mode_density_weight':choice['density_weight'],
               'vector_eigen_residual':residual,'seed_count':seeds,'node_count':e.nodes,
               'time_ticks':ticks,'threshold_amplitude':amplitude,
               'one_step_measured_complex_ratio':[float(ratio.real),float(ratio.imag)],
               'one_step_measured_minus_eigenvalue':float(abs(ratio-lam)),
               'one_step_standard_error_complex_magnitude':ratio_se,
               't1_prediction_relative_complex_vector_error':float(np.linalg.norm(mean[1]-expected[1])/np.linalg.norm(expected[1])),
               'projected_amplitude_ratio_t':rel_amplitude.tolist(),
               'predicted_amplitude_ratio_t':[float(abs(lam)**t) for t in range(ticks+1)],
               'projected_phase_t':ph.tolist(),
               'predicted_phase_t':[float(np.angle(lam)*t) for t in range(ticks+1)],
               'orthogonal_energy_fraction_t':(orthogonal**2).tolist(),
               'sample_size_warning':'ensemble projection has nonzero finite-seed fluctuations'}
        summary.append(entry)
        out[code+'_traces']=ar.astype(np.complex64)
        out[code+'_vector']=v.astype(np.complex64)
    obj={'test':'Bit23 exact finite-bit eigenvector preparation vs linear mean-field prediction',
      'shape':list(shape),'ticks':ticks,'seeds_per_case':seeds,'cases':summary,
      'all_five_charges_conserved':True,'first_seed_per_case_forward_backward_exact':True,
      'limit':'statistical linear tangent exact only at first tick in infinitesimal-amplitude limit; multi-tick correlations can cause failure',
      'time_seconds':time.perf_counter()-start}
    return obj,out

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=8);p.add_argument('--ticks',type=int,default=10);p.add_argument('--shape',type=int,nargs=4,default=[48,4,4,32]);p.add_argument('--selected',nargs='+',default=['q2_x','q2_w','q1_mixed','q3_x']);p.add_argument('--out',type=Path,default=Path('data/bit23_eigenprobe.json'));p.add_argument('--npz',type=Path,default=Path('data/bit23_eigenprobe.npz'))
    args=p.parse_args();data,arrays=run(tuple(args.shape),args.ticks,args.seeds,selected=tuple(args.selected));args.out.parent.mkdir(exist_ok=True,parents=True);args.npz.parent.mkdir(exist_ok=True,parents=True);args.out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');np.savez_compressed(args.npz,**arrays)
    for case in data['cases']: print(case['code'],'t1err',round(case['t1_prediction_relative_complex_vector_error'],3),'ratioerr',round(case['one_step_measured_minus_eigenvalue'],3),'orth_tfinal',round(case['orthogonal_energy_fraction_t'][-1],3))
if __name__=='__main__':main()
