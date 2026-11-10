#!/usr/bin/env python3
"""Quick independent structural smoke checks for Bit19, no 24-tick campaign."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from phase_bit_s4_open38_phased_packet import (synthesize, symmetry_control,
     directional_profile, packed_from_threshold, count_xw)
from phase_bit_s4_open31_bounded_cache import BoundedOrbitD4


def validate():
    e=BoundedOrbitD4((8,4,4,8),max_pairs=0)
    rng=np.random.default_rng(190438)
    threshold=rng.integers(0,65536,size=(e.nodes,24),dtype=np.uint16)
    p=directional_profile(e,2,0.)
    word=packed_from_threshold(threshold,p)
    assert np.array_equal((word>>np.uint32(24)),np.zeros(e.nodes,dtype=np.uint32))
    index=e.coords[:,0]*e.shape[3]+e.coords[:,3]
    counts=count_xw(e,word,index)
    assert int(counts.sum())==sum(int(v).bit_count() for v in word)
    for seed in (100,200):
        check=symmetry_control(e,seed,ticks=3)
        assert check['spatial_mirror_x_equivariance'] and check['bit_exact_reverse']
    a,meta=synthesize(190100,2,e,2,'bit8')
    b,_=synthesize(190100,2,e,2,'free')
    assert np.allclose(a[0],b[0])
    assert meta['all_five_charges_verified_at_start_end']
    return {'shapes':[8,4,4,8], 'sites':e.nodes,'tests':['directional_initialization',
       'explicit_16bit_threshold','exact_24bit_population_count','x_mirror_two_independent_seeds',
       'forward_inverse_roundtrip','same_collision_free_initial_quadrature',
       'five_charges_initial_final'], 'all_passed':True}

if __name__=='__main__':
    print(json.dumps(validate(),indent=2))