#!/usr/bin/env python3
"""Deterministically collect independently checkpointed Bit20 q-runs."""
import argparse, json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('data'));a=p.parse_args()
    all_data={};metadata=[];elapsed=0
    for q in (1,2,3):
        arr=np.load(a.root/f'bit20_q{q}.npz')
        meta=json.loads((a.root/f'bit20_q{q}_metadata.json').read_text())
        assert meta['hidden_q_modes']==[q],meta
        assert meta['independent_seeds_per_case']==8
        assert meta['shape']==[48,4,4,16] and meta['ticks']==24
        elapsed+=meta['full_ensemble_elapsed_seconds']
        for case in meta['cases']:
            for key in (case['raw_array_key_bit8'],case['raw_array_key_free']):
                assert key not in all_data
                arr_value=arr[key]
                assert arr_value.shape==(8,25,3,48),(key,arr_value.shape)
                all_data[key]=arr_value
            metadata.append(case)
    assert len(all_data)==18 and len(metadata)==9
    meta['hidden_q_modes']=[1,2,3]
    meta['cases']=metadata
    meta['full_ensemble_elapsed_seconds']=round(elapsed,3)
    (a.root/'bit20_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    np.savez_compressed(a.root/'bit20_raw.npz',**all_data)
    print('merged',len(metadata),'cases',len(all_data),'complex arrays',round(elapsed,1),'s',flush=True)
if __name__=='__main__': main()
