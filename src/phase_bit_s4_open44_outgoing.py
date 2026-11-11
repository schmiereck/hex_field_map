#!/usr/bin/env python3
"""Bit25: phase-resolved exact connected two-packet response in outgoing (kx,q) channels.

Microscopic model: untouched Bit8 collision + reversible D4 streaming.
R(t)=n(AB)-n(A)-n(B)+n(0), a SIGNED four-state connected response.
This is not a physical S matrix nor a conserved energy transfer.
Equal pseudo-random uint16 backgrounds paired within all phases/amplitudes.
The Fourier phase and eigenvectors are ONLY preparation/observation aides.
"""
import argparse,json,time,sys,os
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4
from phase_bit_s4_open43_mode_prep import ROOTS,BITS,select,matrix_from_json
from phase_bit_s4_open43_pair_collision import prep

K=np.pi/8; Q=np.pi/4
ROOTSI=np.asarray(ROOTS,dtype=np.int8)
PHASES=np.array([0,np.pi/2,np.pi,3*np.pi/2],float)
# Values periodic for x=48,w=32, exactly representable Fourier modes.
CHANNELS={'dc':(0.,0.),'sum':(2*K,2*Q),'sum_x':(2*K,0.),'sum_w':(0.,2*Q),
          'carrier':(K,Q),'opposite':(-K,Q),'q_flip':(K,-Q),'lower':(K,0.),
          'hidden':(0.,Q),'conjugate':(-2*K,-2*Q)}
OBS=('rho','jx','jw')


def measure_sample(engine,states,weights,free=False):
    """Exactly signed 24-component response, Fourier transformed externally."""
    bits=np.asarray(BITS,np.uint32)
    b=[]
    for a in states:
        b.append(((a[:,None]&bits)!=0).astype(np.int8))
    r=(b[3]-b[1]-b[2]+b[0]).astype(np.int8)
    # All five connected additive global charges must vanish identically.
    totals=r.sum(axis=0,dtype=np.int64)
    charges=np.concatenate(([totals.sum()], totals@ROOTSI.astype(np.int64)))
    if np.any(charges):raise AssertionError(('mixed charges',charges.tolist()))
    f=weights @ r.astype(np.float32)/engine.nodes # [channels,24]
    rho=f.sum(axis=1)
    jx=f@ROOTS[:,0]
    jw=f@ROOTS[:,3]
    xor=states[0]^states[1]^states[2]^states[3]
    # use independently defined signed and XOR diagnostics
    projected=np.stack((rho,jx,jw),axis=-1)
    # Zero mode follows the exact integer charge identity, not float32 rounding.
    projected[0,:]=0.
    return projected, int(np.count_nonzero(xor)),int(np.bitwise_count(xor).sum()),int(np.abs(r).sum())


def run_case(A,*,shape=(48,4,4,32),amplitude=.22,separation=16,branch='left',
             phases=PHASES,seeds=8,ticks=12,seed0=250001,
             reverse_first=False,free_first=True):
    tstart=time.perf_counter();
    if os.environ.get("BIT25_PARALLEL", "0")=="1":
        from phase_bit_s4_open44_parallel import ParallelOrbitD4
        e=ParallelOrbitD4(shape)
    else:
        e=NumbaOrbitD4(shape)
    vr,info_r=select(A,K,Q,0,1)
    vb,info_b=select(A,K,Q,0,-1) if branch=='left' else select(A,K,Q,3,1)
    theta=np.asarray([[kx,q] for kx,q in CHANNELS.values()])
    coords=e.coords
    weights=np.exp(-1j*(theta[:,0,None]*coords[:,0][None,:]+theta[:,1,None]*coords[:,3][None,:])).astype(np.complex64)
    values=np.zeros((seeds,len(phases),ticks+1,len(CHANNELS),len(OBS)),dtype=np.complex64)
    stats=np.zeros((seeds,len(phases),ticks+1,3),dtype=np.int32)
    charges_checked=0; roundtrip=False;free_check=False
    for i in range(seeds):
        for j,phi in enumerate(phases):
            states,pre=prep(e,vr,vb,amplitude=amplitude,separation=separation,phase=float(phi),seed=seed0+i)
            if i==j==0:
                if pre['initial_overlap_bits']!=0:raise AssertionError('overlap')
                starting=[s.copy() for s in states]
                if free_first:
                    linear=[s.copy() for s in states]
                    for t in range(ticks+1):
                        if np.any(linear[0]^linear[1]^linear[2]^linear[3]):raise AssertionError('free nonlinear')
                        if t<ticks:linear=[e.stream(s) for s in linear]
                    free_check=True
            for t in range(ticks+1):
                c,sites,bits,absr=measure_sample(e,states,weights)
                values[i,j,t]=c;stats[i,j,t]=[sites,bits,absr]
                charges_checked+=1
                if t<ticks:states=[e.tick(s) for s in states]
            if reverse_first and i==j==0:
                for _ in range(ticks):states=[e.tick(s,reverse=True) for s in states]
                if not all(np.array_equal(a,b) for a,b in zip(states,starting)):raise AssertionError('reverse')
                roundtrip=True
        print('amp',amplitude,'branch',branch,'seed',i+1,'/',seeds,'elapsed',round(time.perf_counter()-tstart,1),flush=True)
    if np.any(stats[:,:,0,0]) or np.max(np.abs(values[:,:,:,0,:]))>1e-7:raise AssertionError('zero-mode not zero')
    return {'values':values,'stats':stats,'metadata':{'shape':list(shape),'physical_nodes':e.nodes,'seeds':seeds,'ticks':ticks,
            'amplitude':amplitude,'separation':separation,'branch':branch,'phases':list(map(float,phases)),
            'channels':{name:[float(a),float(b)] for name,(a,b) in CHANNELS.items()},'observables':list(OBS),
            'seed0':seed0,'right_mode':info_r,'left_mode':info_b,
            'free_xor_null_verified':free_check,'roundtrip_checked':roundtrip,'charge_check_observations':charges_checked,
            'seconds':time.perf_counter()-tstart}}


def save_case(root,name,data):
    root=Path(root);(root/'data').mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'data'/f'bit25_{name}.npz',values=data['values'],stats=data['stats'])
    (root/'data'/f'bit25_{name}.json').write_text(json.dumps(data['metadata'],indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--matrix',default='data/bit24_collision_matrix.json');p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--name',default='baseline');p.add_argument('--amplitude',type=float,default=.22);p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--ticks',type=int,default=12);p.add_argument('--phase',type=float,nargs='*');p.add_argument('--branch',choices=['left','w'],default='left')
    p.add_argument('--shape',type=int,nargs=4,default=(48,4,4,32));p.add_argument('--seed0',type=int,default=250001)
    p.add_argument('--reverse-first',action='store_true');a=p.parse_args()
    A=matrix_from_json(a.matrix);ph=PHASES if not a.phase else np.array(a.phase,float)
    save_case(a.root,a.name,run_case(A,shape=tuple(a.shape),amplitude=a.amplitude,seeds=a.seeds,
                    ticks=a.ticks,phases=ph,seed0=a.seed0,branch=a.branch,reverse_first=a.reverse_first))
