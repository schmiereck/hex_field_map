#!/usr/bin/env python3
"""Bit15 control: lossless directed-front ray predictor + exact XOR residual.

A ray is a single direction bit with an initial node. Its free evolution follows
BitGas directional streaming. The ray list is encoded once per initial pulse;
subsequent states are represented by a sparse per-tick correction. Thus total
storage across T frames must pay the initial rays ONCE, not once per frame.
This is NOT a solver of the nonlinear interactions: exact residual needs truth.
"""
import argparse
import json
import struct
import zlib
from pathlib import Path
import numpy as np
from phase_bit_s4_open34_shell_bits import raw_bytes
from phase_bit_s4_open33_shells import init_pair
from phase_bit_s4_open31_bounded_cache import BoundedOrbitD4


def ray_bytes(initial_delta):
    n=len(initial_delta);width=2 if n<=65535 else 4
    pairs=[(i,b) for i,x in enumerate(initial_delta) for b in range(24) if int(x)>>b & 1]
    h=bytearray(b'RY15'+struct.pack('<BIH',width,n,len(pairs)))
    for i,b in pairs:h+=int(i).to_bytes(width,'little')+bytes([b])
    return bytes(h)

def decode_rays(blob):
    assert blob[:4]==b'RY15'
    width,n,k=struct.unpack_from('<BIH',blob,4)
    offset=11;out=np.zeros(n,dtype=np.uint32)
    for _ in range(k):
        i=int.from_bytes(blob[offset:offset+width],'little');offset+=width
        b=blob[offset];offset+=1
        if i>=n or b>=24:raise ValueError('bad ray')
        out[i]^=np.uint32(1<<b)
    assert offset==len(blob)
    return out

def residual_bytes(residual):
    n=len(residual);width=2 if n<=65535 else 4
    ix=np.flatnonzero(residual)
    if len(ix)>65535:raise ValueError('too many residuals')
    blob=bytearray(b'RE15'+struct.pack('<BIH',width,n,len(ix)))
    for i in ix:blob+=int(i).to_bytes(width,'little')+int(residual[i]).to_bytes(3,'little')
    return bytes(blob)

def decode_residual(blob):
    assert blob[:4]==b'RE15'
    width,n,k=struct.unpack_from('<BIH',blob,4)
    offset=11;out=np.zeros(n,dtype=np.uint32)
    for _ in range(k):
        i=int.from_bytes(blob[offset:offset+width],'little');offset+=width
        v=int.from_bytes(blob[offset:offset+3],'little');offset+=3
        if i>=n:raise ValueError('bad index')
        out[i]=v
    assert offset==len(blob)
    return out

def run(shape=(8,4,4,8),ticks=3,kind='vacuum_pulse',seed=1509):
    e=BoundedOrbitD4(shape,max_pairs=0)
    base,pulse,control,_=init_pair(e,kind,seed,(0,0,0,0))
    initial=[base.copy(),pulse.copy(),control.copy()]
    states=[x.copy() for x in initial]
    rows={}
    for j,name in [(1,'pulse'),(2,'random_control')]:
        delta=states[j]^states[0]
        rays=ray_bytes(delta)
        assert np.array_equal(decode_rays(rays),delta)
        rows[name]={'ray_initial_bytes':len(rays),'ray_initial_zlib_bytes':len(zlib.compress(rays,9)),
                    'ray_count':sum(int(v).bit_count() for v in delta),'times':[]}
    predicted={j:states[j]^states[0] for j in (1,2)}
    for t in range(ticks+1):
        for j,name in [(1,'pulse'),(2,'random_control')]:
            delta=states[j]^states[0]
            resid=delta^predicted[j]
            blob=residual_bytes(resid)
            assert np.array_equal(predicted[j]^decode_residual(blob),delta)
            raw=raw_bytes(delta)
            rows[name]['times'].append({'tick':t,'raw_zlib_bytes':len(zlib.compress(raw,9)),
                   'residual_sites':int(np.count_nonzero(resid)), 'residual_changed_bits':sum(int(v).bit_count() for v in resid),
                   'residual_bytes':len(blob),'residual_zlib_bytes':len(zlib.compress(blob,9))})
        if t<ticks:
            states=[e.tick(s) for s in states]
            for j in predicted:predicted[j]=e.stream(predicted[j])
    for _ in range(ticks):states=[e.tick(s,reverse=True) for s in states]
    assert all(np.array_equal(a,b) for a,b in zip(initial,states))
    for name in rows:
        rr=rows[name]
        rr['total_raw_zlib_bytes']=sum(x['raw_zlib_bytes'] for x in rr['times'])
        rr['total_rays_plus_residual_zlib_bytes']=rr['ray_initial_zlib_bytes']+sum(x['residual_zlib_bytes'] for x in rr['times'])
    return {'experiment':'Bit15 lossless ray-front baseline plus collision residual','shape':list(shape),'kind':kind,'seed':seed,'ticks':ticks,
            'encoder_is_exact':True,'reverse_microdynamics_exact':True,'results':rows,
            'scope':'XOR perturbation relative to simultaneously evolved baseline; does not encode baseline',
            'limitations':['each correction requires truth evolution','ray predictor is not an autonomous nonlinear algorithm',
              'zlib data rates do not include complete shape and software specification overhead']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shape',nargs=4,type=int,default=[8,4,4,8]);p.add_argument('--ticks',type=int,default=3);p.add_argument('--seed',type=int,default=1509);p.add_argument('--kind',choices=['vacuum_pulse','background_pulse'],default='vacuum_pulse');p.add_argument('--out',type=Path)
    a=p.parse_args();r=run(tuple(a.shape),a.ticks,a.kind,a.seed)
    if a.out:a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
    for n,v in r['results'].items():print(n,v['total_raw_zlib_bytes'],v['total_rays_plus_residual_zlib_bytes'],[(x['tick'],x['residual_sites']) for x in v['times']])