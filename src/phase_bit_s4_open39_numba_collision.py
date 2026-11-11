#!/usr/bin/env python3
"""Bit 20 optional exact Numba accelerator for the unchanged Bit-8 local rule.

It uses the *same* 282 two-body gates, 384 B4 permutations, mutation-class
ranking and unique reciprocal choice. No new physical rule or approximation.
Verify independently against collide_fast before using for simulations.
"""
from __future__ import annotations
import numpy as np
from numba import njit
from phase_bit_s4_open29_fast_collision import TABS, MASK_CLASS, collide_fast
from phase_bit_s4_open26_symmetric_collision import GATES
from phase_bit_s4_open28_fast4d import FastD4

LEFT=np.array([a for a,b in GATES],dtype=np.int64)
RIGHT=np.array([b for a,b in GATES],dtype=np.int64)
MUT=np.bitwise_or(LEFT,RIGHT)
RANK=np.array([MASK_CLASS[int(m)] for m in MUT],dtype=np.int64)
TABLES=np.ascontiguousarray(TABS,dtype=np.int64)

@njit(cache=True)
def _transform(s, g, tabs):
    return tabs[g,0,s&255] | tabs[g,1,(s>>8)&255] | tabs[g,2,(s>>16)&255]

@njit(cache=True)
def _edge_key_less(s,t,best_a,best_b,tabs):
    a=1<<30
    b=1<<30
    for j in range(384):
        ss=_transform(s,j,tabs)
        tt=_transform(t,j,tabs)
        low=ss if ss<tt else tt
        high=tt if ss<tt else ss
        if low<a or (low==a and high<b):
            a=low;b=high
    return a,b

@njit(cache=True)
def _preference(s,left,right,masks,ranks,tabs):
    best=1<<30
    n=0
    # Number of finalists <= number of gates; duplicates dedup explicitly
    candidates=np.empty(282,dtype=np.int64)
    for k in range(len(left)):
        l=left[k];r=right[k]
        if ((s & l)==l and (s & r)==0) or ((s & r)==r and (s & l)==0):
            rank=ranks[k]
            if rank<=best:
                t=s^masks[k]
                if rank<best:
                    n=0;best=rank
                found=False
                for i in range(n):
                    if candidates[i]==t:
                        found=True;break
                if not found:
                    candidates[n]=t;n+=1
    if n==0:return -1
    if n==1:return candidates[0]
    best_a=1<<30;best_b=1<<30
    winner=-1;multiple=False
    for j in range(n):
        t=candidates[j]
        a,b=_edge_key_less(s,t,best_a,best_b,tabs)
        if a<best_a or (a==best_a and b<best_b):
            best_a=a;best_b=b;winner=t;multiple=False
        elif a==best_a and b==best_b:
            multiple=True
    return -1 if multiple else winner

@njit(cache=True)
def _collide(s,left,right,masks,ranks,tabs):
    t=_preference(s,left,right,masks,ranks,tabs)
    if t<0:return s
    partner=_preference(t,left,right,masks,ranks,tabs)
    return t if partner==s else s

@njit(cache=True)
def _collision_array(words,left,right,masks,ranks,tabs):
    out=np.empty(len(words),dtype=np.uint32)
    for i in range(len(words)):
        out[i]=_collide(int(words[i]),left,right,masks,ranks,tabs)
    return out


def collide_numba(s:int)->int:
    return int(_collide(int(s),LEFT,RIGHT,MUT,RANK,TABLES))

class NumbaOrbitD4(FastD4):
    def collision(self,words):
        words=np.asarray(words,dtype=np.uint32)
        if words.shape!=(self.nodes,):raise ValueError('wrong shape')
        return _collision_array(words,LEFT,RIGHT,MUT,RANK,TABLES)


def audit(n=500,seed=2011):
    from phase_bit_s4_open27_active_collision import charge
    rng=np.random.default_rng(seed)
    test=list(range(30))+[int(v) for v in rng.integers(0,1<<24,size=n,dtype=np.int64)]
    test += [int(sum(1<<j for j in rng.choice(24,k,replace=False))) for k in (4,8,10,12,14,18,22) for _ in range(n//7)]
    for s in test:
        a=collide_numba(s);b=collide_fast(s)
        assert a==b,(s,a,b)
        assert collide_numba(a)==s
        assert charge(a)==charge(s)
    return {'cases':len(test),'exact_bit8_bit10_match':True,'involution':True,'five_charges_conserved':True}

if __name__=='__main__':
    print(audit())
