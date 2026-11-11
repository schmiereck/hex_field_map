#!/usr/bin/env python3
"""Bit 21 -- geometry falsification of the q-dependent center sign reversal.

The microscopic rule is the unchanged Bit-8 24-bit collision.  Its Numba
implementation and the Bit-20 real, finite 16-bit threshold preparations are
reused without altering physics.  Physical k and q are held fixed as periods
change; complex quantities are diagnostics of four real quadrature runs.

Examples:
 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python src/phase_bit_s4_open40_falsify_q.py \
 --shape 48 4 4 32 --q 2 --modes 3 -3 --ticks 12 --seeds 8 --out data/bit21_w32_q2.json \
 --raw data/bit21_w32_q2.npz
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4,collide_numba
from phase_bit_s4_open39_dispersion import synthesize_case,symmetry_audit
from phase_bit_s4_open39_analyze import metrics,bootstrap_ci
from phase_bit_s4_open38_phased_packet import pos_width


def analyze_one(arr,control,km,shape,ticks,q_physical):
    x0=shape[0]//2;tx=np.arange(2,min(8,ticks)+1)
    bm=metrics(arr,km,shape);fm=metrics(control,km,shape)
    seed_centers=np.array([[pos_width(abs(row)**2,x0)[0] for row in sample[:,1]] for sample in arr]);
    free_seed_centers=np.array([[pos_width(abs(row)**2,x0)[0] for row in sample[:,1]] for sample in control]);
    seed_slopes=np.array([np.polyfit(tx,cc[tx],1)[0] for cc in seed_centers]);
    free_seed_slopes=np.array([np.polyfit(tx,cc[tx],1)[0] for cc in free_seed_centers]);
    vis=arr.mean(axis=0)[:,0,:]
    vis_center=np.array([pos_width(abs(row)**2,x0)[0] for row in vis]);
    tr={'k_mode':km,'physical_k':float(2*np.pi*km/shape[0]),'physical_q':float(q_physical),'n_seeds':int(len(arr)),
       'early_window':[int(tx[0]),int(tx[-1])],
       'ensemble_center_slope':float(np.polyfit(tx,bm['centers'][tx],1)[0]),
       'seed_center_slope_mean':float(seed_slopes.mean()),
       'seed_center_slope_95ci':bootstrap_ci(seed_slopes,seed=21111+km),
       'seed_center_slopes':[float(v) for v in seed_slopes],
       'free_ensemble_center_slope':float(np.polyfit(tx,fm['centers'][tx],1)[0]),
       'free_seed_center_slope_mean':float(free_seed_slopes.mean()),
       'free_seed_center_slopes':[float(v) for v in free_seed_slopes],
       'visible_w0_ensemble_center_slope':float(np.polyfit(tx,vis_center[tx],1)[0]),
       'hidden_amplitude_initial':float(abs(bm['z'][0,1])),
       'hidden_amplitude_ratio_final':float(bm['amplitude'][-1]),
       'free_hidden_amplitude_ratio_final':float(fm['amplitude'][-1]),
       'width_initial':float(bm['widths'][0]),'width_final':float(bm['widths'][-1]),
       'width_by_tick':[float(x) for x in bm['widths']],
       'center_by_tick':[float(x) for x in bm['centers']],
       'free_center_by_tick':[float(x) for x in fm['centers']],
       'note':'center slope of |coherent q-response(x)|^2; NOT spectral group velocity'}
    return tr


def run(shape=(48,4,4,32),q=2,modes=(3,-3),ticks=12,seeds=8,amplitude=.105,
        symmetry=False):
    start=time.perf_counter()
    e=NumbaOrbitD4(shape)
    assert shape[0]==48 and shape[0]%2==0 and shape[3]%16==0
    qmode=q*(shape[3]//16)
    physical_q=2*np.pi*qmode/shape[3]
    assert abs(physical_q-q*np.pi/8)<1e-12
    if symmetry: sym=symmetry_audit(e,ticks=3)
    else:sym=None
    arrays={};rows=[]
    for km in modes:
        raw=[];fre=[]
        for si in range(seeds):
            seed=390100+q*10000+(km+30)*100+si
            a=synthesize_case(e,seed,km,qmode,ticks,amplitude,free=False)
            b=synthesize_case(e,seed,km,qmode,ticks,amplitude,free=True)
            raw.append(a);fre.append(b)
            print(f'shape={shape} q={q} qmode={qmode} k={km:+} seed={si+1}/{seeds} time={time.perf_counter()-start:.1f}s',flush=True)
        raw=np.asarray(raw,dtype=np.complex64);fre=np.asarray(fre,dtype=np.complex64)
        key=f'q{q}_k{km:+d}';arrays[key+'_bit8']=raw;arrays[key+'_free']=fre
        rows.append(analyze_one(raw,fre,km,shape,ticks,physical_q))
    result={'experiment':'Bit21 spatial geometry falsification of observed Bit20 q reversal',
            'shape':list(shape),'physical_nodes':e.nodes,'ticks':ticks,'independent_seeds_per_case':seeds,
            'physical_q':float(physical_q),'physical_k_abs':float(2*np.pi*3/shape[0]),
            'physical_q_base_index':q,'actual_hidden_q_index':qmode,
            'amplitude':amplitude,'symmetric_collision':'exact Bit8 involution via validated Numba kernel',
            'seeds_aligned_with_bit20_by_physical_q_and_mode':True,
            'reconstructed_initialization':'Bit19/20 four real finite-threshold phase-cycled trajectories',
            'symmetry_audit':sym,'rows':rows,'wall_seconds':time.perf_counter()-start,
            'physical_claim_limit':'Finite-time q-response center, not eigenmode or particle group velocity',
            'relative_comparison_warning':'y/z enlargement also changes microscopic finite-volume ensemble of transverse modes'}
    return result,arrays

def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',nargs=4,type=int,default=[48,4,4,32]);p.add_argument('--q',type=int,default=2)
    p.add_argument('--modes',nargs='+',type=int,default=[3,-3]);p.add_argument('--ticks',type=int,default=12);p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--amplitude',type=float,default=.105);p.add_argument('--symmetry',action='store_true')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--raw',type=Path)
    a=p.parse_args();res,raw=run(tuple(a.shape),a.q,tuple(a.modes),a.ticks,a.seeds,a.amplitude,a.symmetry)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n')
    if a.raw:a.raw.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.raw,**raw)
    print('RESULT',a.out,'time',res['wall_seconds'],flush=True)
if __name__=='__main__':main()
