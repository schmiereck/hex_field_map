#!/usr/bin/env python3
"""Controlled amplitude, separation, phase and transverse-size scan for Bit24.

Seed alignment is intentional: equal shapes use identical initial uint16
random thresholds across cases, yielding paired comparisons.
"""
from pathlib import Path
import argparse,json,time,numpy as np
from phase_bit_s4_open43_pair_collision import measure_case,summarize
from phase_bit_s4_open43_mode_prep import matrix_from_json

CASES=[
 ('baseline',.22,16,0.,8,14,(48,4,4,32)),
 ('amp_010',.10,16,0.,4,14,(48,4,4,32)),
 ('amp_016',.16,16,0.,4,14,(48,4,4,32)),
 ('amp_030',.30,16,0.,4,14,(48,4,4,32)),
 ('sep_12',.22,12,0.,4,14,(48,4,4,32)),
 ('sep_20',.22,20,0.,4,14,(48,4,4,32)),
 ('phase_halfpi',.22,16,np.pi/2,4,14,(48,4,4,32)),
 ('phase_pi',.22,16,np.pi,4,14,(48,4,4,32)),
 ('transverse_8',.22,16,0.,2,12,(48,8,8,32)),
]

def run(root:Path,selected=None):
  A=matrix_from_json(root/'data/bit24_collision_matrix.json')
  combined={'experiment':'Bit24 two eigenpacket exact four-state nonlinear difference',
    'reference':'Bit23 precomputed 24x24 statistical collision tangent, exact Bit8 gas for ticks',
    'cases':{},'seed_policy':'matched uint16 backgrounds seed=241001+index',
    'warning':'nonzero nonlinear XOR response does NOT imply stable bound states or asymptotic scattering'}
  outfile=root/'data/bit24_summary.json';start=time.perf_counter()
  for name,amp,sep,phase,seeds,ticks,shape in CASES:
    if selected and name not in selected:continue
    p=root/'data'/f'bit24_{name}.json'
    if p.exists():
      r=json.loads(p.read_text());print('REUSE',name,flush=True)
    else:
      r=measure_case(A,shape,amplitude=amp,separation=sep,phase=phase,
        seeds=seeds,ticks=ticks,seed0=241001,free_control=True,do_reverse=(name=='baseline'))
      p.write_text(json.dumps(r,indent=2)+'\n')
    combined['cases'][name]=summarize(r)
    outfile.write_text(json.dumps(combined,indent=2)+'\n')
    print('DONE',name,'duration',round(time.perf_counter()-start,1),'end mean xor sites',round(combined['cases'][name]['mean_active_sites_by_tick'][-1],1),flush=True)
  return combined

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent);p.add_argument('--cases',nargs='*');a=p.parse_args()
 run(a.root,set(a.cases) if a.cases else None)
