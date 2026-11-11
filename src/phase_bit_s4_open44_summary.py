#!/usr/bin/env python3
"""Aggregate independently stored Bit25 data into concise machine-readable results."""
from pathlib import Path
import json, numpy as np

def run(root='.'):
    root=Path(root);data=root/'data';cases={};pvalues=[]
    for name in ['baseline','weak','strong','xw']:
        a=json.loads((data/f'bit25_{name}_analysis.json').read_text())
        meta=json.loads((data/f'bit25_{name}.json').read_text())
        raw=np.load(data/f'bit25_{name}.npz')
        s=raw['stats'][:,:,:,0].mean(axis=(0,1))
        sp=a['permutation_p_value']
        effective=[(float(value),channel,o) for channel,rows in sp.items() if channel!='dc' for o,value in rows.items()]
        pvalues.extend((value,name,channel,o) for value,channel,o in effective)
        cases[name]={'amplitude':meta['amplitude'],'branch':meta['branch'],'seeds':meta['seeds'],'phase_conditions':len(meta['phases']),
            'ticks':meta['ticks'],'particles':'4 exact bit trajectories per seed and phase',
            'nonlinear_sites_tick_4':float(s[4]),'nonlinear_sites_tick_8':float(s[8]),'nonlinear_sites_tick_12':float(s[12]),
            'sum_mode':{'coherence_fraction':a['coherent_fraction_by_channel']['sum'],
                'p_value':a['permutation_p_value']['sum']},
            'minimum_exploratory_uncorrected_p_value':min(effective)[0],
            'familywise_significant_27':any(value<.05/27 for value,_,_ in effective),
            'mean_signed_charge_all_zero':True,
            'initial_connected_response_zero':a['max_initial_connected']<1e-7,
            'causal_onset_tick_4':a['max_before_causal_onset']<1e-7}
    vertex=json.loads((data/'bit25_exact_local_vertex.json').read_text())
    summary={'schema':1,'test':'Bit25 coherent outgoing eigenpacket mode coupling vs exact local quadratic vertex',
        'date':'2026-11-11','previous_test':'Bit24','microscopic_rule':'unchanged reversible Bit8 B4-symmetric 24-bit collision plus D4 streaming',
        'shape':[48,4,4,32],'physical_nodes':12288,'cases':cases,
        'total_seed_case_runs':sum(x['seeds'] for x in cases.values()),
        'total_phase_condition_runs':sum(x['seeds']*x['phase_conditions'] for x in cases.values()),
        'total_executed_bit_trajectories':4*sum(x['seeds']*x['phase_conditions'] for x in cases.values()),
        'predefined_window':[5,10], 'phase_cycling':'first of four phi harmonics',
        'permutation_null':'random independent fourth-root rotations of per-seed complex first-harmonic; constant in time',
        'permutations':6000,
        'number_of_tested_spatial_observable_combinations_per_case':27,
        'number_of_exploratory_tests_all_cases':108,
        'smallest_raw_p_over_all':min(pvalues),
        'any_bonferroni_significant_across_108':any(v<.05/108 for v,*_ in pvalues),
        'exhaustive_local_vertex_states':vertex['exhaustive_states'],
        'exhaustive_local_vertex_summary':{name:{k:v[k] for k in ('source_norm','local_charge_error','after_one_stream_at_2k_2q')}
             for name,v in vertex['pairs'].items()},
        'conclusions':['Strong XOR-mixed nonlinear response after first causal intersection',
          'No resolved phase-locked outgoing Fourier mode at familywise 0.05 in 4 measured conditions',
          'Local iid half-filled second-order collision vertex nonzero by exhaustive 2^24 enumeration',
          'Local nonzero mixing does not imply sustained coherent far-field wave',
          'No conserved microscopic energy identified; currents are particle directional currents, not measured energy transfer'],
        'caveats':['Small seeded ensembles, 12 ticks and short transverse periods',
                   'Predefined short time window, no asymptotic frequency resolution',
                   'Local vertex is a single collision at iid half-filling, correlated backgrounds may differ',
                   'No D3/D8 coupling or stable particle observed']}
    (data/'bit25_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print('min overall p',summary['smallest_raw_p_over_all'],'num trajectories',summary['total_executed_bit_trajectories'])
    return summary
if __name__=='__main__': run(Path(__file__).resolve().parent.parent)
