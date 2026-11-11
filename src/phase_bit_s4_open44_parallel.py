#!/usr/bin/env python3
"""Optional Bit25 CPU-parallel exact Bit8 collision evaluation; no rule change."""
import numpy as np
from numba import njit,prange
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4,_collide,LEFT,RIGHT,MUT,RANK,TABLES

@njit(parallel=True,cache=True)
def _collision_parallel(words,left,right,mut,rank,tabs):
    out=np.empty(len(words),dtype=np.uint32)
    for i in prange(len(words)):
        out[i]=_collide(int(words[i]),left,right,mut,rank,tabs)
    return out

class ParallelOrbitD4(NumbaOrbitD4):
    def collision(self,words):
        a=np.asarray(words,dtype=np.uint32)
        if a.shape!=(self.nodes,):raise ValueError(a.shape)
        return _collision_parallel(a,LEFT,RIGHT,MUT,RANK,TABLES)

if __name__=='__main__':
    rng=np.random.default_rng(250025)
    for shape in [(4,4,4,4),(16,4,4,16)]:
        a=NumbaOrbitD4(shape);b=ParallelOrbitD4(shape)
        x=rng.integers(0,1<<24,size=a.nodes,dtype=np.uint32)
        for t in range(4):
            assert np.array_equal(a.collision(x),b.collision(x))
            assert np.array_equal(a.tick(x),b.tick(x))
            x=a.tick(x)
        y=x.copy()
        for t in range(4): y=b.tick(y,reverse=True)
        # only check internal exact reversibility of parallel engine
        z=y.copy()
        for t in range(4): z=b.tick(z)
        assert np.array_equal(z,x)
    print('Parallel collision exact vs serial for two geometries and 8 ticks')
