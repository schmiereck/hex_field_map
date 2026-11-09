#!/usr/bin/env python3
"""Bit 8: symmetric orbit-edge mutual matching collision, experimental.

For every allowed two-body swap s--t prioritize the B4 orbit of the four
changed bits, then the B4 orbit of the undirected edge (s,t).
Only unique minimum edges are eligible; swap only on mutual selection.
This is an exactly reversible B4-equivariant involution, preserving occupancy
and all four integer momenta. Unlike Bit 7, it can select moves at half filling.
WARNING: expensive canonical orbit calculation; not yet a fast lattice kernel.
"""
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
import argparse, json, random, sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase_bit_s4_open26_symmetric_collision import (TABLES, proposals, charge, GROUP, ROOTS, MASK, canonical)
from phase_bit_s4_open25_symmetry_recovery import symmetry_maps, permute, gf_rank


def transform(s,table):
    return table[0][s & 255] | table[1][(s>>8)&255] | table[2][(s>>16)&255]

@lru_cache(maxsize=200000)
def edge_orbit_key(s,t):
    return min((min(a,b), max(a,b)) for table in TABLES for a,b in [(transform(s,table),transform(t,table))])

@lru_cache(maxsize=200000)
def preference(s):
    ns=proposals(s)
    if not ns:return None
    mask_rank=[canonical(s^t) for t in ns]
    best=min(mask_rank)
    finalists=[t for t,k in zip(ns,mask_rank) if k==best]
    if len(finalists)==1:
        return finalists[0]
    edge_rank=[edge_orbit_key(s,t) for t in finalists]
    second=min(edge_rank)
    winners=[t for t,k in zip(finalists,edge_rank) if k==second]
    return winners[0] if len(winners)==1 else None


@lru_cache(maxsize=200000)
def collide(s):
    t=preference(s)
    return t if t is not None and preference(t)==s else s


def audit(seed=2709, per_density=12):
    rng=random.Random(seed)
    generators=symmetry_maps(ROOTS)
    counts=defaultdict(lambda:[0,0]); deltas=[]; witness=[]
    for n in (2,3,4,6,8,10,12,14,16,18,20,21,22):
        for _ in range(per_density):
            s=sum(1<<i for i in rng.sample(range(24),n))
            t=collide(s)
            assert charge(s)==charge(t)
            assert collide(t)==s
            for name,g in generators.items():
                assert collide(permute(s,g))==permute(t,g), (name,s,t)
            counts[n][0]+=1
            if s!=t:
                counts[n][1]+=1
                d=[int(bool(t&(1<<i)))-int(bool(s&(1<<i))) for i in range(24)]
                deltas.append(d)
                if len(witness)<19 and gf_rank([w['delta'] for w in witness]+[d])>len(witness):
                    witness.append({'state':s,'target':t,'delta':d})
    return {'rule':'unique minimum (swap-mask orbit, undirected-edge orbit), mutual choice',
        'seed':seed,'samples_per_occupation':per_density,
        'number_of_symmetries':len(GROUP),
        'sampled_symmetry_generators':list(generators),
        'per_occupation':{str(k):{'tested':v[0],'moving':v[1]} for k,v in sorted(counts.items())},
        'moving_sample_difference_rank':gf_rank(deltas) if deltas else 0,
        'independent_witness_count':len(witness),
        'witnesses':witness,
        'note':'Exact structural proof: local two-body swaps preserve charges, an invariant undirected edge priority with tie rejection is equivariant, and mutual choice is an involution. Rank 19 requires 19 verified independent moving witnesses.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--samples',type=int,default=12);p.add_argument('--out',type=Path);args=p.parse_args()
    x=audit(per_density=args.samples);s=json.dumps(x,indent=2)+'\n'
    if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(s)
    print(json.dumps({k:v for k,v in x.items() if k!='witnesses'},indent=2))