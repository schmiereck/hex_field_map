#!/usr/bin/env python3
"""Bit24: exact four-state finite-difference for collisions of 24-channel D4 eigenpackets.

Microscopic dynamics: unchanged Bit8 reversible collision + D4 streaming.
All four states share the SAME uniform 16-bit threshold background; XOR masks
encode the initially disjoint targeted eigenpackets. The four-state XOR
interaction vanishes exactly at t=0 and under collision-free streaming.
This is a *nonlinearity diagnostic* (not a two-particle scattering S-matrix).
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4
from phase_bit_s4_open43_mode_prep import ROOTS, BITS, select, matrix_from_json

MASK24=(1<<24)-1
ROOTI=np.asarray(ROOTS,dtype=np.int8)
BITS=np.asarray(BITS,dtype=np.uint32)


def signed_offset(x,c,length):
    return (np.asarray(x,dtype=np.int32)-c+length//2)%length-length//2


def packed(bits):
    return np.sum(bits.astype(np.uint32)*BITS[None,:],axis=1,dtype=np.uint32)


def prep(engine, vec_right, vec_left, *, amplitude=.22, separation=16,
         phase=0., sigma=3.5, cut_radius=5, seed=24001,
         k=np.pi/8, q=np.pi/4):
    """Common threshold, disjoint spatial supports, exact XOR composition.

    Initial p_i=1/2 + epsilon * g(x-c) * Re(v_i exp(i(kx+qw+phi))).
    The perturbation is quantized to a 16-bit integer threshold, independently
    for each microscopic bit. 16-bit thresholds occur only at initialization.
    """
    shape=engine.shape; Lx=int(shape[0])
    assert (separation%2)==0 and separation>=2*cut_radius+2
    cR=Lx//2-separation//2; cL=Lx//2+separation//2
    x=engine.coords[:,0];w=engine.coords[:,3]
    dr=signed_offset(x,cR,Lx);dl=signed_offset(x,cL,Lx)
    gR=np.where(abs(dr)<=cut_radius, np.exp(-.5*(dr/sigma)**2), 0.)
    gL=np.where(abs(dl)<=cut_radius, np.exp(-.5*(dl/sigma)**2), 0.)
    assert not np.any((gR>0)&(gL>0)), 'must have disjoint supports'
    theta=k*x+q*w
    pr=amplitude*gR[:,None]*np.real(np.exp(1j*theta[:,None])*vec_right[None,:])
    pl=amplitude*gL[:,None]*np.real(np.exp(1j*(theta+phase)[:,None])*vec_left[None,:])
    assert np.max(np.abs(pr))<.5 and np.max(np.abs(pl))<.5
    rng=np.random.default_rng(seed)
    u=rng.integers(0,65536,size=(engine.nodes,24),dtype=np.uint16)
    base=(u<32768)
    thR=np.clip(np.rint(32768+65536*pr),0,65536).astype(np.int32)
    thL=np.clip(np.rint(32768+65536*pl),0,65536).astype(np.int32)
    a=base^(u<thR)
    b=base^(u<thL)
    # The initial flip masks are *disjoint*, so this is simultaneous
    # application of both targeted perturbations to the same background.
    x0=packed(base)
    xA=x0^packed(a)
    xB=x0^packed(b)
    xAB=x0^packed(a)^packed(b)
    assert not np.any(packed(a)&packed(b))
    assert np.array_equal(xAB.astype(np.uint32),packed((u<np.clip(np.rint(32768+65536*(pr+pl)),0,65536).astype(np.int32))))
    return [x0,xA,xB,xAB], {'center_right':cR,'center_left':cL,
        'initial_bits_A':int(np.bitwise_count(packed(a)).sum()),
        'initial_bits_B':int(np.bitwise_count(packed(b)).sum()),
        'initial_overlap_bits':0,
        'min_initial_x_support_gap':int(separation-2*cut_radius)}


def projections(engine,states, ks=(0, np.pi/8, np.pi/4), qs=(0, np.pi/4, np.pi/2)):
    """Microscopic signed four-state difference, Fourier project to ρ, Jx, Jw."""
    x=engine.coords[:,0];w=engine.coords[:,3]
    # signed channel-wise mixed difference: f(A+B)-f(A)-f(B)+f(0)
    f0,fA,fB,fAB=states
    r=(((fAB[:,None]&BITS[None,:])!=0).astype(np.int8)
        -((fA[:,None]&BITS[None,:])!=0).astype(np.int8)
        -((fB[:,None]&BITS[None,:])!=0).astype(np.int8)
        +((f0[:,None]&BITS[None,:])!=0).astype(np.int8))
    observables=np.column_stack([r.sum(axis=1,dtype=np.int16), r@ROOTI[:,0],r@ROOTI[:,3]])
    channels={}
    for q in qs:
        p=np.exp(-1j*q*w)
        # This 3D-slice analysis is not equivalent to q=0; q-phase is external.
        for j,name in enumerate(('density','current_x','current_w')):
            z=observables[:,j]
            prof=np.bincount(x,weights=z*p.real,minlength=engine.shape[0])+1j*np.bincount(x,weights=z*p.imag,minlength=engine.shape[0])
            powers={f'{k/np.pi:.3f}pi':float(abs(np.dot(prof,np.exp(-1j*k*np.arange(engine.shape[0]))))**2)/engine.nodes**2 for k in ks}
            channels[f'q={q/np.pi:.3f}pi_{name}']={'x_profile_power':float(np.sum(abs(prof)**2))/engine.nodes**2,
                'fourier_power':powers}
    return r,observables,channels


def measure_case(A,shape=(48,4,4,32), *, amplitude=.22, separation=16,
                 phase=0., sigma=3.5,cut_radius=5,seeds=4,ticks=16,
                 seed0=240100,free_control=True,do_reverse=False,branch_B="left"):
    e=NumbaOrbitD4(shape)
    k=np.pi/8;q=np.pi/4
    vR,mR=select(A,k,q,0,1)
    vL,mL=select(A,k,q,0,-1)
    vW,mW=select(A,k,q,3,1)
    if branch_B not in ("left","w","right"): raise ValueError(branch_B)
    vec_B={"left":vL,"right":vR,"w":vW}[branch_B]
    t0=time.perf_counter()
    histories=[]
    for seed in range(seeds):
        states,prep_details=prep(e,vR,vec_B,amplitude=amplitude,separation=separation,
                                 phase=phase,sigma=sigma,cut_radius=cut_radius,
                                 seed=seed0+seed,k=k,q=q)
        original=[s.copy() for s in states]
        initial_charges=[e.invariants(s) for s in states]
        hist=[]
        for t in range(ticks+1):
            ix=states[0]^states[1]^states[2]^states[3]
            cnt=int(np.bitwise_count(ix).sum())
            active=int(np.count_nonzero(ix))
            # Important: full additive signed density/current difference is *not*
            # equal to XOR interaction. Record both independently.
            r,ob,ch=projections(e,states)
            signed_nonzero=int(np.count_nonzero(r))
            charge_sums=(int(np.sum(ob[:,0])),)+tuple(int(z) for z in np.sum(r,axis=0,dtype=np.int64)@ROOTI.astype(np.int64))
            assert charge_sums==(0,0,0,0,0),('nonzero mixed conserved charge',charge_sums)
            hist.append({'tick':t,'xor_bits':cnt,'xor_sites':active,
                         'signed_channel_nonzero':signed_nonzero,
                         'signed_density_l1':int(abs(ob[:,0]).sum()),
                         'signed_current_x_l1':int(abs(ob[:,1]).sum()),
                         'signed_current_w_l1':int(abs(ob[:,2]).sum()),
                         'channels':ch})
            if t<ticks:
                states=[e.tick(s) for s in states]
        if do_reverse and seed==0:
            check=[s.copy() for s in states]
            for _ in range(ticks):
                check=[e.tick(s,reverse=True) for s in check]
            assert all(np.array_equal(s,o) for s,o in zip(check,original))
        assert all(e.invariants(s)==c for s,c in zip(states,initial_charges))
        if free_control and seed==0:
            free=[s.copy() for s in original]
            for t in range(ticks+1):
                assert not np.any(free[0]^free[1]^free[2]^free[3]),('free streaming nonlinear',t)
                if t<ticks:free=[e.stream(s) for s in free]
        histories.append({'seed':seed0+seed,'initial':prep_details,'time_series':hist})
        print('case amp',amplitude,'sep',separation,'phase',round(phase,3),'seed',seed+1,'/',seeds,
              'first_active',next((h['tick'] for h in hist if h['xor_bits']),None),
              'end_xor_bits',hist[-1]['xor_bits'], 'elapsed',round(time.perf_counter()-t0,2),flush=True)
    return {'schema':1,'experiment':'Bit24 exact two mode packets four-state XOR mixed response',
      'shape':list(shape),'physical_nodes':e.nodes,'seeds':seeds,'ticks':ticks,
      'amplitude':amplitude,'separation':separation,'relative_hidden_phase':phase,
      'sigma':sigma,'truncation_radius':cut_radius,'k':float(k),'q':float(q),
      'branch_right':mR,'branch_left':mL,'branch_w_control':mW,'branch_B':branch_B,
      'seed0':seed0,'initial_state':'shared uint16 threshold; two disjoint compact Gaussian local eigenvector preparations, XOR composition',
      'microdynamics':'exact unchanged Bit8 collision+stream, deterministic reversible',
      'checks':{'independent_four_state_charge':True,'signed_interaction_charge_zero':True,
        'xor_free_streaming_zero':bool(free_control),'exact_four_state_roundtrip_first_seed':bool(do_reverse)},
      'limitations':['XOR mixed difference is a bitwise nonlinear-interaction diagnostic, not a scattering amplitude',
        'Finite seeds, random initial half-filled microstates','Packet modes from statistical 24x24 collision tangent',
        'A and B supports disjoint; no change to microphysics'],
      'seed_histories':histories,'elapsed_seconds':time.perf_counter()-t0}


def summarize(run):
    hs=run['seed_histories'];t=run['ticks'];n=run['physical_nodes']
    keys=['xor_bits','xor_sites','signed_channel_nonzero','signed_density_l1','signed_current_x_l1','signed_current_w_l1']
    values={k:np.asarray([[h['time_series'][tt][k] for tt in range(t+1)] for h in hs],dtype=float) for k in keys}
    a=values['xor_sites'];mean=a.mean(0)
    first=[next((z['tick'] for z in h['time_series'] if z['xor_bits']>0),None) for h in hs]
    return {'settings':{k:run[k] for k in ('shape','seeds','ticks','amplitude','separation','relative_hidden_phase','sigma','truncation_radius')} | {'branch_B':run.get('branch_B','left')},
            'first_xor_interaction_tick_by_seed':first,
            'all_seeds_initial_exact_superposition':all(h['time_series'][0]['xor_bits']==0 for h in hs),
            'full_bit_charge_and_micro_reversal_checks':run['checks'],
            'mean_active_sites_by_tick':mean.tolist(),
            'mean_xor_bits_by_tick':values['xor_bits'].mean(0).tolist(),
            'mean_signed_density_l1_by_tick':values['signed_density_l1'].mean(0).tolist(),
            'mean_signed_current_x_l1_by_tick':values['signed_current_x_l1'].mean(0).tolist(),
            'mean_signed_current_w_l1_by_tick':values['signed_current_w_l1'].mean(0).tolist(),
            'fraction_active_at_end':float(mean[-1]/n),
            'xor_sites_at_end_seed_std':float(a[:,-1].std(ddof=1)) if len(hs)>1 else None}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--matrix',default='data/bit24_collision_matrix.json');p.add_argument('--amplitude',type=float,default=.22)
 p.add_argument('--separation',type=int,default=16);p.add_argument('--phase',type=float,default=0.)
 p.add_argument('--seeds',type=int,default=4);p.add_argument('--ticks',type=int,default=16)
 p.add_argument('--seed0',type=int,default=240100);p.add_argument('--shape',nargs=4,type=int,default=(48,4,4,32))
 p.add_argument('--out',type=Path,default=Path('data/bit24_pilot.json'));a=p.parse_args()
 A=matrix_from_json(a.matrix)
 r=measure_case(A,tuple(a.shape),amplitude=a.amplitude,separation=a.separation,phase=a.phase,seeds=a.seeds,ticks=a.ticks,seed0=a.seed0,do_reverse=True)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(summarize(r),indent=2))
