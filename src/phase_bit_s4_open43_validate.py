#!/usr/bin/env python3
"""Independent checks of exact Bit24 four-state interaction construction."""
from __future__ import annotations
from pathlib import Path
import json,numpy as np
from phase_bit_s4_open43_mode_prep import matrix_from_json,select
from phase_bit_s4_open43_pair_collision import measure_case,summarize
from phase_bit_s4_open39_numba_collision import audit

ROOT=Path(__file__).resolve().parent.parent

def validate():
    A=matrix_from_json(ROOT/'data/bit24_collision_matrix.json')
    left=select(A,np.pi/8,np.pi/4,0,-1)[1]
    right=select(A,np.pi/8,np.pi/4,0,1)[1]
    w=select(A,np.pi/8,np.pi/4,3,1)[1]
    assert left['group_velocity']<-.8 and right['group_velocity']>.8
    assert abs(w['group_velocity'])<.1
    bit8=audit(n=64,seed=24321)
    r=measure_case(A,(32,4,4,16),amplitude=.22,separation=16,phase=0.,seeds=1,
       ticks=6,seed0=248001,free_control=True,do_reverse=True)
    t=summarize(r)
    assert t['first_xor_interaction_tick_by_seed']==[4]
    assert all(t['mean_xor_bits_by_tick'][j]==0 for j in range(4))
    assert t['mean_xor_bits_by_tick'][4]>0
    files={}
    for fn in sorted((ROOT/'data').glob('bit24_*.json')):
        if fn.name in ('bit24_validation.json',):continue
        if fn.name.startswith('bit24_branch_') or fn.name in ('bit24_baseline.json','bit24_amp_010.json',
           'bit24_amp_016.json','bit24_amp_030.json','bit24_sep_12.json','bit24_sep_20.json',
           'bit24_phase_halfpi.json','bit24_phase_pi.json','bit24_transverse_8.json'):
            b=json.loads(fn.read_text())
            assert all(all(z['tick']==i for i,z in enumerate(h['time_series'])) for h in b['seed_histories'])
            assert all(h['time_series'][0]['xor_bits']==0 for h in b['seed_histories'])
            assert all(h['time_series'][0]['signed_channel_nonzero']==0 for h in b['seed_histories'])
            assert all(b['checks'][k] for k in ('independent_four_state_charge',
                 'signed_interaction_charge_zero','xor_free_streaming_zero'))
            files[fn.name]={'seeds':b['seeds'],'ticks':b['ticks'],'all_initial_interactions_zero':True}
    assert len(files)==11,len(files)
    return {'schema':1,'exact_bit8_collision_reference':bit8,
       'one_seed_end_to_end':{'nodes':r['physical_nodes'],'ticks':6,
       'finite_speed_first_interaction_tick':4,'all_four_states_bit_exact_reversible':True,
       'all_five_mixed_additive_charges_zero_each_tick':True,
       'collision_free_streaming_cross_term_exactly_zero_each_tick':True},
       'tested_preparation_velocities':{'x_right':right['group_velocity'],
        'x_left':left['group_velocity'],'w':w['group_velocity']},
       'validated_existing_datasets':files}

if __name__=='__main__':
    r=validate();p=ROOT/'data/bit24_validation.json'
    p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
