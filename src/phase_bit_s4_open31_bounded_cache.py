#!/usr/bin/env python3
"""Bit 12: bounded exact collision memoization for the unchanged Bit-8 rule.

Each encountered state owns at most one LRU entry s -> C(s). The reverse
state may need a separate entry or an exact recomputation. Dropping entries
changes only runtime, NEVER the collision. Capacity zero disables engine
memoization entirely. Inner Bit-10 lru_cache(maxsize=100000) caches
are independently bounded; they must not be confused with the engine cache.
"""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
from phase_bit_s4_open29_engine import VectorOrbitD4
from phase_bit_s4_open29_fast_collision import collide_fast

class BoundedOrbitD4(VectorOrbitD4):
    def __init__(self,shape=(4,4,4,4),max_pairs=4096):
        super().__init__(shape)
        if max_pairs < 0: raise ValueError('max_pairs must be >= 0')
        self.max_pairs=int(max_pairs)
        self.cache=OrderedDict()  # canonical representative -> partner
        self.pair_hits=0
        self.pair_misses=0
        self.evictions=0
        self.computations=0

    def collision(self,words):
        words=np.asarray(words,dtype=np.uint32)
        if words.shape!=(self.nodes,):raise ValueError('wrong packed site array')
        unique,inverse=np.unique(words,return_inverse=True)
        output=np.empty(len(unique),dtype=np.uint32)
        for i,item in enumerate(unique):
            s=int(item)
            # A pair needs two directional keys to look up in O(1); store the
            # mapping in a bounded dict instead of a second unbounded index.
            # The mapping is always symmetric because the collision is involutive.
            val=self.cache.get(s)
            if val is not None:
                self.cache.move_to_end(s)
                self.pair_hits+=1
                self.cache_hits+=1
            else:
                val=int(collide_fast(s))
                self.computations+=1
                self.pair_misses+=1
                self.cache_misses+=1
                if self.max_pairs:
                    # One key per encountered input; its inverse can be
                    # recomputed on a miss.
                    self.cache[s]=val
                    if len(self.cache)>self.max_pairs:
                        self.cache.popitem(last=False)
                        self.evictions+=1
            output[i]=val
        assert len(self.cache)<=self.max_pairs
        return output[inverse]

    def cache_stats(self):
        return {'capacity_entries':self.max_pairs,'current_entries':len(self.cache),
                'hits':self.pair_hits,'misses':self.pair_misses,
                'evictions':self.evictions,'exact_evaluations':self.computations}