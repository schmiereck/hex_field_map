#!/usr/bin/env python3
"""Independent executable correctness/ensemble provenance checks for Bit 20."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4, audit
from phase_bit_s4_open39_dispersion import symmetry_audit, synthesize_case
from phase_bit_s4_open31_bounded_cache import BoundedOrbitD4

def run():
    tested=audit(n=700,seed=3939)
    small=(8,4,4,8)
    numba=NumbaOrbitD4(small)
    reference=BoundedOrbitD4(small,max_pairs=0)
    rng=np.random.default_rng(3939)
    v=rng.integers(0,1<<24,numba.nodes,dtype=np.uint32)
    x=v.copy();y=v.copy()
    invariants=numba.invariants(x)
    for _ in range(4):
        x=numba.tick(x)
        y=reference.tick(y)
        assert np.array_equal(x,y), 'Numba vs Bit10 packed reference mismatch'
        assert numba.invariants(x)==invariants
    for _ in range(4):
        x=numba.tick(x,reverse=True)
        y=reference.tick(y,reverse=True)
        assert np.array_equal(x,y)
    assert np.array_equal(x,v)
    mirror=symmetry_audit(NumbaOrbitD4((48,4,4,16)),6)
    # A phase-coded q=2 state, not only random bits, must respect x reflection.
    e=NumbaOrbitD4((48,4,4,16))
    from phase_bit_s4_open38_phased_packet import directional_profile,packed_from_threshold,mirror_x_words
    rr=np.random.default_rng(3902)
    thresholds=rr.integers(0,65536,size=(e.nodes,24),dtype=np.uint16)
    s=packed_from_threshold(thresholds,directional_profile(e,3,0.,.105,q_mode=2))
    reflected=mirror_x_words(e,s)
    for _ in range(6):
        assert np.array_equal(mirror_x_words(e,s),reflected)
        s=e.tick(s);reflected=e.tick(reflected)
    assert np.array_equal(mirror_x_words(e,s),reflected)
    for _ in range(6):
        s=e.tick(s,reverse=True)
        reflected=e.tick(reflected,reverse=True)
    assert np.array_equal(mirror_x_words(e,s),reflected)
    # File-level ensemble uniqueness and mode indexing.
    root=Path(__file__).resolve().parent.parent
    meta=json.loads((root/'data/bit20_metadata.json').read_text())
    npz=np.load(root/'data/bit20_raw.npz')
    seeds=set()
    for entry in meta['cases']:
        q=entry['q_mode'];k=entry['k_mode']
        for j in range(8):
            seed=entry['seed_first']+j
            assert seed not in seeds
            seeds.add(seed)
        for key in (entry['raw_array_key_bit8'],entry['raw_array_key_free']):
            assert npz[key].shape==(8,25,3,48)
            assert np.isfinite(npz[key]).all()
    assert len(seeds)==72
    return {'status':'passed','local_collision_audit':tested,
      'numba_bit10_reference_exact_four_forward_reverse_ticks_512_nodes':True,
      'q2_phase_preparation_mirror_x_six_ticks_exact':True,
      'full_grid_symmetry_audit':mirror,
      'all_raw_ensembles':9,'independent_seeds':72,'real_microtrajectories':288,
      'forward_ticks_per_trajectory':24,
      'numba_and_free_raw_array_keys':len(npz.files),
      'raw_npz_sha256':hashlib.sha256((root/'data/bit20_raw.npz').read_bytes()).hexdigest()}

if __name__=='__main__':
    r=run();root=Path(__file__).resolve().parent.parent
    (root/'data/bit20_validation.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
