#!/usr/bin/env python3
"""Drop-in packed D4 engine with identical Bit-8 collision and faster cold lookup."""
from __future__ import annotations
import numpy as np
from phase_bit_s4_open28_fast4d import FastD4
from phase_bit_s4_open29_fast_collision import collide_fast


class VectorOrbitD4(FastD4):
    """No dynamics change relative to FastD4; only cold collision evaluation."""
    def collision(self, words):
        words=np.asarray(words,dtype=np.uint32)
        if words.shape!=(self.nodes,):
            raise ValueError('wrong packed site array')
        unique, inverse=np.unique(words,return_inverse=True)
        output=np.empty(len(unique),dtype=np.uint32)
        for i,s in enumerate(unique):
            k=int(s)
            val=self.cache.get(k)
            if val is None:
                val=collide_fast(k)
                self.cache[k]=val
                self.cache[val]=k
                self.cache_misses+=1
            else:
                self.cache_hits+=1
            output[i]=val
        return output[inverse]