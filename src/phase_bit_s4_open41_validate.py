#!/usr/bin/env python3
"""Bit22 independent free-branch analytic check + equilibrium data audits."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4
from phase_bit_s4_open27_active_collision import ROOTS
from phase_bit_s4_open41_structure_factor import site_observables


def analytic_free_seed(meta,observed):
    e=NumbaOrbitD4(tuple(meta['shape']))
    roots=np.asarray(ROOTS,dtype=float)
    rng=np.random.default_rng(meta['seed0'])
    initial=rng.integers(0,1<<24,size=e.nodes,dtype=np.uint32)
    occupancies=((initial[:,None]>>np.arange(24,dtype=np.uint32)[None,:])&1).astype(float)
    ticks=meta['ticks'];t=np.arange(ticks+1)
    error=0.; errors_by_probe=[0.,0.,0.]
    for j,mode in enumerate(meta['mode_pairs']):
        k,q=mode['k_physical'],mode['q_physical']
        spatial=np.exp(-1j*(k*e.coords[:,0]+q*e.coords[:,3]))
        directional=spatial @ occupancies/np.sqrt(e.nodes)
        velocity_phases=np.exp(-1j*t[:,None]*(k*roots[None,:,0]+q*roots[None,:,3]))
        predicted=np.column_stack((velocity_phases@directional,
                                  velocity_phases@(directional*roots[:,0]),
                                  velocity_phases@(directional*roots[:,3])))
        d=np.abs(predicted-observed[0,:,j,:])
        error=max(error,float(d.max()))
        errors_by_probe=np.maximum(errors_by_probe,np.max(d,axis=0))
    assert error<1e-4,('analytic free stream failed',error)
    return {'first_seed_all_modes_all_ticks_exact_up_to_float32_error':True,
            'max_abs_error':error,'max_error_density_jx_jw':list(map(float,errors_by_probe)),
            'analytic_branches':'omega_d=k_x*v_dx+q*v_dw for 24 D4 streaming directions'}


def validate(paths,metas):
    reports={}
    for path,metafile in zip(paths,metas):
        meta=json.load(open(metafile))
        with np.load(path) as raw:
            a=raw['equilibrium_bit8'];b=raw['equilibrium_free']
        assert a.shape==b.shape==(meta['seeds'],meta['ticks']+1,len(meta['mode_pairs']),3)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        analytic=analytic_free_seed(meta,b)
        st=np.mean(np.abs(a[:,:meta['ticks']//2])**2,axis=(0,1))
        en=np.mean(np.abs(a[:,meta['ticks']//2+1:])**2,axis=(0,1))
        rel=np.abs(st-en)/np.maximum(1e-9,.5*(st+en))
        n=int(meta['physical_nodes'])
        # Mean number remains fixed per trajectory, asserted during simulation.
        assert n==np.prod(meta['shape'])//2
        reports[Path(path).stem]={'shape':meta['shape'],'seeds':meta['seeds'],'ticks':meta['ticks'],
             'per_seed_five_charge_conservation':all(x['number_four_momenta_conserved'] for x in meta['audit']),
             'full_backward_roundtrip_first_seed':meta['audit'][0]['exact_roundtrip_test'],
             'analytic_free_streaming':analytic,
             'density_power_first_vs_last_half_relative_change_median':float(np.median(rel[:,0])),
             'density_power_first_vs_last_half_relative_change_max':float(np.max(rel[:,0])),
             'note':'Stationarity of uniform measure follows bijectivity; finite-seed empirical half-time powers fluctuate.'}
    return {'validation':'Bit22 spectral equilibrium experiment','cases':reports,
        'warning':'Exact free identities test streaming/observables; collision correctness additionally covered by original Bit20/21 and current conservation/roundtrip, not exhaustive 2^24 enumeration.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--equilibrium',nargs='+',required=True);p.add_argument('--metadata',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=validate(a.equilibrium,a.metadata);Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
