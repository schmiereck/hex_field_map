#!/usr/bin/env python3
"""Bit 16: exact nonlinear correction sources, D4-neighbor clusters, storage audit.

The reference lattice evolution is unchanged. Given baseline B and perturbed P,
D=P XOR B. For collision C, source Q=C(P) XOR C(B) XOR D. Then
D'=stream(D XOR Q). Analyze Q clusters and ray-residual R=D XOR stream^t(D0).
This is an exact algebraic identity, NOT a closed fast front evolution.
"""
from __future__ import annotations
import argparse, json, struct, time, zlib
from pathlib import Path
import numpy as np
from phase_bit_s4_open33_shells import init_pair
from phase_bit_s4_open31_bounded_cache import BoundedOrbitD4
from phase_bit_s4_open34_front_rays import ray_bytes, residual_bytes
from phase_bit_s4_open34_shell_bits import raw_bytes


def clusters(words, engine):
    """Components of nonzero 24-bit words in periodic D4 root-neighbor graph."""
    active = np.flatnonzero(words)
    active_mask = np.zeros(engine.nodes, dtype=np.bool_)
    active_mask[active]=True
    seen = np.zeros(engine.nodes, dtype=np.bool_)
    groups=[]
    # engine.sources[:, i] are the 24 incoming neighbors at site i;
    # ROOTS includes opposite directions, so this is undirected adjacency.
    for start in active:
        if seen[start]:continue
        seen[start]=True
        stack=[int(start)]; component=[]
        while stack:
            i=stack.pop();component.append(i)
            for neighbor in engine.sources[:,i]:
                j=int(neighbor)
                if active_mask[j] and not seen[j]:
                    seen[j]=True;stack.append(j)
        groups.append(sorted(component))
    groups.sort(key=lambda g:(-len(g),g[0]))
    assert sum(map(len, groups))==len(active)
    return groups


def cluster_bytes(words, groups):
    """Lossless, individually delimited 4D components, node index and 24-bit data."""
    n=len(words); width=2 if n<=65535 else 4
    b=bytearray(b'CF16'+struct.pack('<BIH',width,n,len(groups)))
    for g in groups:
        b.extend(struct.pack('<I',len(g)))
        for i in g:
            b.extend(int(i).to_bytes(width,'little'))
            b.extend(int(words[i]).to_bytes(3,'little'))
    return bytes(b)


def cluster_decode(blob):
    assert blob[:4]==b'CF16'
    width,n,k=struct.unpack_from('<BIH',blob,4)
    ptr=11; out=np.zeros(n,dtype=np.uint32)
    for _ in range(k):
        size=struct.unpack_from('<I',blob,ptr)[0];ptr+=4
        for __ in range(size):
            i=int.from_bytes(blob[ptr:ptr+width],'little');ptr+=width
            val=int.from_bytes(blob[ptr:ptr+3],'little');ptr+=3
            if i>=n or val==0 or out[i]:raise ValueError('invalid component')
            out[i]=val
    if ptr!=len(blob):raise ValueError('trailing component data')
    return out


def stats(words, engine):
    groups=clusters(words,engine)
    blob=cluster_bytes(words,groups)
    assert np.array_equal(cluster_decode(blob),words)
    sparse=residual_bytes(words)
    assert len(sparse)>0
    sizes=[len(g) for g in groups]
    return {'sites':int(np.count_nonzero(words)),
        'changed_direction_bits':sum(int(x).bit_count() for x in words),
        'components':len(groups),'largest_component':max(sizes,default=0),
        'singleton_components':sum(k==1 for k in sizes),
        'top_component_sizes':sizes[:12],
        'cluster_bytes':len(blob),'cluster_zlib_bytes':len(zlib.compress(blob,9)),
        'sparse_bytes':len(sparse),'sparse_zlib_bytes':len(zlib.compress(sparse,9)),
        'dense_zlib_bytes':len(zlib.compress(raw_bytes(words),9))}


def run(shape=(8,4,4,8),ticks=8,seed=1601,kind='background_pulse'):
    engine=BoundedOrbitD4(shape,max_pairs=0)
    baseline,pulse,control,_=init_pair(engine,kind,seed,(0,0,0,0))
    starts=[baseline.copy(),pulse.copy(),control.copy()]
    states=[x.copy() for x in starts]
    predictions=[None,states[1]^states[0],states[2]^states[0]]
    results={}; t0=time.perf_counter()
    for j,name in [(1,'pulse'),(2,'random_control')]:
        init=predictions[j]
        results[name]={'initial_ray_bytes':len(ray_bytes(init)),
            'initial_ray_zlib_bytes':len(zlib.compress(ray_bytes(init),9)),
            'times':[]}
    for t in range(ticks+1):
        for j,name in [(1,'pulse'),(2,'random_control')]:
            delta=states[j]^states[0]
            residual=delta^predictions[j]
            assert np.array_equal(predictions[j]^residual,delta)
            entry={'tick':t,'delta':stats(delta,engine),'ray_residual':stats(residual,engine)}
            results[name]['times'].append(entry)
        if t==ticks:break
        coll=[engine.collision(x) for x in states]
        for j,name in [(1,'pulse'),(2,'random_control')]:
            delta=states[j]^states[0]
            # New nonlinear source: the only difference from free stream.
            source=coll[j]^coll[0]^delta
            derived=engine.stream(delta^source)
            newbaseline=engine.stream(coll[0]);newstate=engine.stream(coll[j])
            assert np.array_equal(derived,newstate^newbaseline), 'source identity false'
            results[name]['times'][-1]['nonlinear_source_next_tick']=stats(source,engine)
        states=[engine.stream(x) for x in coll]
        for j in [1,2]: predictions[j]=engine.stream(predictions[j])
    assert all(engine.invariants(x)==engine.invariants(y) for x,y in zip(states,starts))
    for _ in range(ticks):states=[engine.tick(x,reverse=True) for x in states]
    assert all(np.array_equal(x,y) for x,y in zip(states,starts)), 'microscopic reversibility failed'
    for row in results.values():
        row['total_direct_dense_zlib_bytes']=sum(v['delta']['dense_zlib_bytes'] for v in row['times'])
        row['total_ray_plus_sparse_corrections_zlib_bytes']=row['initial_ray_zlib_bytes']+sum(v['ray_residual']['sparse_zlib_bytes'] for v in row['times'])
        row['total_ray_plus_cluster_corrections_zlib_bytes']=row['initial_ray_zlib_bytes']+sum(v['ray_residual']['cluster_zlib_bytes'] for v in row['times'])
    return {'experiment':'Bit16 exact D4 connected correction fronts','kind':kind,'seed':seed,
      'shape':list(shape),'physical_nodes':engine.nodes,'ticks':ticks,
      'elapsed_seconds':round(time.perf_counter()-t0,3),'exact_microdynamics_roundtrip':True,
      'exact_non_linear_source_identity':True,'exact_component_encode_decode':True,
      'reference_background_excluded_from_compression':True,
      'limitations':['clusters are formed offline after exact collision calculation',
        'cluster format uses sparse indices; grouping alone adds metadata and may cost bytes',
        'cluster count alone is not information-theoretic compressibility',
        'the predicted directed rays are not a spherical basis or autonomous simulator'],
      'results':results}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shape',type=int,nargs=4,default=[8,4,4,8])
    parser.add_argument('--ticks',type=int,default=8)
    parser.add_argument('--kind',choices=['background_pulse','vacuum_pulse'],default='background_pulse')
    parser.add_argument('--seed',type=int,default=1601)
    parser.add_argument('--out',type=Path)
    a=parser.parse_args(); result=run(tuple(a.shape),a.ticks,a.seed,a.kind)
    s=json.dumps(result,indent=2)+'\n'
    if a.out:a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(s)
    for name,row in result['results'].items():
        print(a.kind,a.shape,name,'direct/rays+sparse/rays+clusters',row['total_direct_dense_zlib_bytes'],row['total_ray_plus_sparse_corrections_zlib_bytes'],row['total_ray_plus_cluster_corrections_zlib_bytes'])
        print('cluster evolution',[(t['tick'],t['ray_residual']['sites'],t['ray_residual']['components'],t['ray_residual']['largest_component']) for t in row['times']])