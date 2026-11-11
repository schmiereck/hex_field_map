#!/usr/bin/env python3
"""Bit 21 audits: finite-bit kernel equivalence and exact mirrored q-packet evolution."""
import argparse,json
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4,audit
from phase_bit_s4_open38_phased_packet import directional_profile,packed_from_threshold,mirror_x_words


def check_mirror_q(shape=(48,4,4,32),q=2,ticks=4):
    e=NumbaOrbitD4(shape)
    qm=q*(shape[3]//16)
    rng=np.random.default_rng(214040)
    u=rng.integers(0,65536,(e.nodes,24),dtype=np.uint16)
    profile=directional_profile(e,3,0.35,.105,qm)
    a=packed_from_threshold(u,profile)
    b=mirror_x_words(e,a)
    a0=a.copy();b0=b.copy();charge=e.invariants(a)
    for _ in range(ticks):
        assert np.array_equal(mirror_x_words(e,a),b)
        a=e.tick(a);b=e.tick(b)
        assert e.invariants(a)==charge
    assert np.array_equal(mirror_x_words(e,a),b)
    for _ in range(ticks):
        a=e.tick(a,reverse=True);b=e.tick(b,reverse=True)
    assert np.array_equal(a,a0) and np.array_equal(b,b0)
    return {'shape':list(shape),'physical_nodes':e.nodes,'physical_q':float(q*np.pi/8),
            'phase_coded_mirror_x_exact_each_tick':True,'roundtrip_exact':True,
            'five_charges_conserved':True,'ticks_each_direction':ticks}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    r={'collision_equivalence':audit(n=140,seed=2140),
       'mirrored_packet_checks':[check_mirror_q(s) for s in ((48,4,4,32),(48,8,8,16),(48,8,8,32))]}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
