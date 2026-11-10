#!/usr/bin/env python3
"""Bit 15: LOSSLESS 24-bit shell encoding test for the unchanged Bit-8 D4 gas.

The evolved truth is a pair of packed 4D bit fields.  The encoded object is
XOR(pulse, background); the background is NOT counted in the differential rate.
A selected squared-radius shell (optionally angular sector) carries one 24-bit
mask, replicated over its sites, with an explicitly encoded residual of
(node-index, 24-bit XOR word) pairs. Decoding is exact, even after collisions.
This is a diagnostic codec, not a faster update algorithm.
"""
import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_bit_s4_open33_shells import init_pair, shell_labels
from phase_bit_s4_open31_bounded_cache import BoundedOrbitD4

MAGIC=b'SH15'

def sector_ids(coords, shape, center):
    v=(coords.astype(np.int32)-np.asarray(center,dtype=np.int32))%np.asarray(shape,dtype=np.int32)
    v=np.where(v>np.asarray(shape)//2,v-np.asarray(shape),v)
    axis=np.argmax(np.abs(v),axis=1)
    sign=(v[np.arange(len(v)),axis]<0).astype(np.int32)
    return (2*axis+sign).astype(np.int32)

def group_labels(coords,shape,center,kind):
    r=shell_labels(coords,shape,center)
    if kind=='radial':return r
    if kind=='angular8':return 8*r+sector_ids(coords,shape,center)
    raise ValueError(kind)

def mask_majority(words):
    """Optimal 24-bit constant mask under Hamming loss, deterministic zero tie."""
    bits=((words[:,None]>>np.arange(24,dtype=np.uint32))&1).sum(axis=0)
    return int(np.sum((bits*2>len(words)).astype(np.uint32)<<np.arange(24,dtype=np.uint32)))

def prepare_groups(delta,labels):
    groups=[]
    for label in np.unique(labels):
        indices=np.flatnonzero(labels==label)
        mask=mask_majority(delta[indices])
        if not mask:continue
        gain=sum(int(x).bit_count() - int(int(x)^mask).bit_count() for x in delta[indices])
        if gain>0:groups.append((gain,int(label),mask))
    return sorted(groups,key=lambda g:(-g[0],g[1]))

def encode(delta,labels,k=4):
    """Self-contained on-disk format; group labels regenerated from shared fixed geometry.

    4 byte magic, 1 byte node-index width, 4 byte count, 2 byte number of
    templates, 2 byte number of residual sites. Templates store label uint32 +
    mask uint24; residuals store node index uint16/uint32 + mask uint24.
    """
    delta=np.asarray(delta,dtype=np.uint32)
    n=len(delta);width=2 if n<=65535 else 4
    candidates=prepare_groups(delta,labels)[:k]
    pred=np.zeros(n,dtype=np.uint32)
    for _,label,mask in candidates:pred[labels==label]=np.uint32(mask)
    residual=np.bitwise_xor(delta,pred)
    inds=np.flatnonzero(residual).tolist()
    if len(inds)>65535:raise ValueError('Too many residual sites for this benchmark format')
    out=bytearray(MAGIC+struct.pack('<BIHH',width,n,len(candidates),len(inds)))
    for _,label,mask in candidates:out+=struct.pack('<I',label)+int(mask).to_bytes(3,'little')
    for i in inds:
        out+=int(i).to_bytes(width,'little')+int(residual[i]).to_bytes(3,'little')
    return bytes(out),{'k':k,'templates':len(candidates),'residual_sites':len(inds),'hamming_error_bits':int(sum(int(x).bit_count() for x in residual)), 'chosen_labels':[v[1] for v in candidates]}

def decode(blob,labels):
    if blob[:4]!=MAGIC:raise ValueError('bad magic')
    width,n,tcount,rcount=struct.unpack_from('<BIHH',blob,4)
    if width not in (2,4) or n!=len(labels):raise ValueError('shape mismatch')
    offset=13
    out=np.zeros(n,dtype=np.uint32)
    for _ in range(tcount):
        label=struct.unpack_from('<I',blob,offset)[0];offset+=4
        mask=int.from_bytes(blob[offset:offset+3],'little');offset+=3
        out[labels==label]=np.uint32(mask)
    for _ in range(rcount):
        i=int.from_bytes(blob[offset:offset+width],'little');offset+=width
        mask=int.from_bytes(blob[offset:offset+3],'little');offset+=3
        if i>=n:raise ValueError('invalid residual index')
        out[i]^=np.uint32(mask)
    if offset!=len(blob):raise ValueError('trailing data')
    return out

def raw_bytes(delta):
    return b''.join(int(v).to_bytes(3,'little') for v in delta)

def measure(delta, labels, ks):
    raw=raw_bytes(delta)
    result={'raw_bytes':len(raw),'raw_zlib_bytes':len(zlib.compress(raw,9)),
            'nonzero_delta_nodes':int(np.count_nonzero(delta)),
            'changed_bits':int(sum(int(v).bit_count() for v in delta)), 'modes':{}}
    for mode in ('radial','angular8'):
        table=[]
        for k in ks:
            blob,meta=encode(delta,labels[mode],k)
            assert np.array_equal(decode(blob,labels[mode]),delta)
            table.append({**meta,'encoded_bytes':len(blob),'encoded_zlib_bytes':len(zlib.compress(blob,9)),
                          'ratio_vs_raw':round(len(blob)/len(raw),5),
                          'ratio_vs_raw_zlib':round(len(zlib.compress(blob,9))/len(zlib.compress(raw,9)),5)})
        result['modes'][mode]=table
    return result

def run(shape=(8,4,4,8),ticks=3,seed=1509,kind='vacuum_pulse',ks=(0,1,4,8,16)):
    e=BoundedOrbitD4(shape,max_pairs=0)
    center=(0,0,0,0)
    base,pulse,control,_=init_pair(e,kind,seed,center)
    originals=[x.copy() for x in (base,pulse,control)]
    inv=[e.invariants(x) for x in originals]
    states=[x.copy() for x in originals]
    labels={mode:group_labels(e.coords,e.shape,center,mode) for mode in ('radial','angular8')}
    report=[]
    for t in range(ticks+1):
        entry={'tick':t}
        for name,s in [('pulse',states[1]),('random_control',states[2])]:
            delta=np.bitwise_xor(s,states[0]);entry[name]=measure(delta,labels,ks)
        report.append(entry)
        if t<ticks:states=[e.tick(s) for s in states]
        for s,charges in zip(states,inv):assert e.invariants(s)==charges
    for _ in range(ticks):states=[e.tick(s,reverse=True) for s in states]
    assert all(np.array_equal(x,y) for x,y in zip(states,originals))
    return {'experiment':'Bit15 exact discrete direction masks on radial/sector 4D shell supports',
            'shape':list(shape),'physical_nodes':e.nodes,'seed':seed,'kind':kind,'ticks':ticks,
            'microdynamics':'unchanged Bit8 reversible local collision + 4D streaming',
            'quantity':'FULL 24-bit XOR difference of pulse and shared evolving background',
            'baseline_not_included_in_encoding':True,'encoder':'selected shell 24-bit mask + explicit sparse XOR residual; exact decoder',
            'geometric_support':'periodic D4 Euclidean squared radius, fixed center; angular 8 dominant-axis sectors',
            'lossless_roundtrip':True,'exact_reverse_evolution':True,'five_charges_conserved':True,
            'ks':list(ks),'history':report,
            'limitations':['not a representation of the shared background','encoder selection is expensive','selected shells do not evolve autonomously','only finite samples, no long-run scalability claim']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shape',nargs=4,type=int,default=[8,4,4,8]);p.add_argument('--ticks',type=int,default=3);p.add_argument('--seed',type=int,default=1509);p.add_argument('--kind',choices=['vacuum_pulse','background_pulse'],default='vacuum_pulse');p.add_argument('--out',type=Path)
    a=p.parse_args(); result=run(tuple(a.shape),a.ticks,a.seed,a.kind)
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    for entry in result['history']:
        print(entry['tick'],[(name, entry[name]['raw_zlib_bytes'], [(mode,min((v['encoded_zlib_bytes'],v['k']) for v in entry[name]['modes'][mode])) for mode in ['radial','angular8']]) for name in ['pulse','random_control']],flush=True)