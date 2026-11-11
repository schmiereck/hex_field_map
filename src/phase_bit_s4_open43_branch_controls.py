#!/usr/bin/env python3
"""Independent x-right + w-polarized, and co-moving x-right+x-right controls."""
from pathlib import Path
import json, numpy as np
from phase_bit_s4_open43_pair_collision import measure_case,summarize
from phase_bit_s4_open43_mode_prep import matrix_from_json
root=Path(__file__).resolve().parent.parent
A=matrix_from_json(root/'data/bit24_collision_matrix.json')
for mode in ('w','right'):
 p=root/'data'/f'bit24_branch_{mode}.json'
 if p.exists(): run=json.loads(p.read_text())
 else:
  run=measure_case(A,(48,4,4,32),amplitude=.22,separation=16,phase=0,
    seeds=4,ticks=14,seed0=241001,branch_B=mode)
  p.write_text(json.dumps(run,indent=2)+'\n')
 print('BRANCH',mode,'onset',summarize(run)['first_xor_interaction_tick_by_seed'],
       'active t4,t8,t14',[round(summarize(run)['mean_active_sites_by_tick'][t],1) for t in [4,8,14]],flush=True)
