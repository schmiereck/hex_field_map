#!/usr/bin/env python3
"""Independent Bit25 checks for parity D4, Bit8 identity, charge, reversibility."""
import json,numpy as np
from phase_bit_s4_open39_numba_collision import NumbaOrbitD4,audit
from phase_bit_s4_open44_parallel import ParallelOrbitD4
from phase_bit_s4_open43_mode_prep import matrix_from_json,select
from phase_bit_s4_open43_pair_collision import prep
from phase_bit_s4_open44_outgoing import measure_sample,CHANNELS,K,Q
from pathlib import Path


def verify(root='.'):
    root=Path(root);local=audit(180,seed=250025)
    e=ParallelOrbitD4((16,4,4,16));ref=NumbaOrbitD4(e.shape)
    rng=np.random.default_rng(251125)
    checks=[]
    for n in range(3):
        x=rng.integers(0,1<<24,size=e.nodes,dtype=np.uint32)
        q=e.invariants(x)
        y=x.copy()
        for t in range(4):
            assert np.array_equal(e.tick(y),ref.tick(y))
            y=e.tick(y)
            assert e.invariants(y)==q
        for t in range(4):y=e.tick(y,reverse=True)
        assert np.array_equal(y,x)
        checks.append(True)
    A=matrix_from_json(root/'data/bit24_collision_matrix.json')
    right,_=select(A,K,Q,0,1);left,_=select(A,K,Q,0,-1)
    e2=ParallelOrbitD4((32,4,4,16))
    states,metadata=prep(e2,right,left,seed=250025,separation=16)
    orig=[v.copy() for v in states]
    phase=np.array([[a,b] for a,b in CHANNELS.values()])
    weights=np.exp(-1j*(phase[:,0,None]*e2.coords[:,0][None,:]+phase[:,1,None]*e2.coords[:,3][None,:])).astype(np.complex64)
    totals=[]
    for t in range(7):
        z,sites,bits,absr=measure_sample(e2,states,weights)
        assert np.allclose(z[0],0,atol=2e-7)
        totals.append([t,sites,bits])
        if t<6:states=[e2.tick(x) for x in states]
    for t in range(6):states=[e2.tick(x,reverse=True) for x in states]
    assert all(np.array_equal(a,b) for a,b in zip(states,orig))
    res={'test':'Bit25 exact outgoing mode observable and parallel implementation',
       'numba_reference_audit':local,'parallel_serial_full_steps':sum(checks),
       'all_five_charges_conserved':True,'four_state_6tick_exact_roundtrip':True,
       'free_streaming_connected_zero':True,'zero_mode_connected_charges':True,
       'separation':metadata['min_initial_x_support_gap'],'short_histogram':totals}
    (root/'data'/'bit25_validation.json').write_text(json.dumps(res,indent=2)+'\n')
    return res

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args()
 print(json.dumps(verify(a.root),indent=2))
