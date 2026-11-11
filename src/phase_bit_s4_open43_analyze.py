#!/usr/bin/env python3
"""Reproducible compact Bit24 synthesis from fully archived per-seed trajectories."""
from pathlib import Path
import json,numpy as np
from phase_bit_s4_open43_pair_collision import summarize

ROOT=Path(__file__).resolve().parent.parent
CASES=['baseline','amp_010','amp_016','amp_030','sep_12','sep_20','phase_halfpi','phase_pi',
       'transverse_8','branch_w','branch_right']

def run():
    originals={n:json.loads((ROOT/'data'/f'bit24_{n}.json').read_text()) for n in CASES}
    stats={n:summarize(r) for n,r in originals.items()}
    raw0=originals['baseline']
    ref=np.array([[h['time_series'][t]['xor_sites'] for t in range(raw0['ticks']+1)] for h in raw0['seed_histories']])
    reference_first4=ref[:4]
    phase={}
    for n in ('phase_halfpi','phase_pi'):
        x=np.array([[h['time_series'][t]['xor_sites'] for t in range(originals[n]['ticks']+1)] for h in originals[n]['seed_histories']])
        phase[n]={'matched_seed_mean_xor_site_difference_vs_phase0_t4':float(np.mean(x[:,4]-reference_first4[:,4])),
                  'matched_seed_mean_xor_site_difference_vs_phase0_t8':float(np.mean(x[:,8]-reference_first4[:,8])),
                  'matched_seed_mean_xor_site_difference_vs_phase0_t14':float(np.mean(x[:,14]-reference_first4[:,14])),
                  'phase0_matched_first_four_seed_t4_mean':float(reference_first4[:,4].mean()),
                  'phase0_matched_first_four_seed_t14_mean':float(reference_first4[:,14].mean())}
    # Averages of per-seed *power* are not coherent ensemble Fourier amplitudes.
    channel_powers={}
    for tick in (4,8,12,14):
        channel_powers[str(tick)]={}
        for q in ('0.000pi','0.250pi','0.500pi'):
            vals={}
            for field in ('density','current_x','current_w'):
                power=[h['time_series'][tick]['channels'][f'q={q}_{field}']['x_profile_power'] for h in raw0['seed_histories']]
                vals[field+'_mean_incoherent_power']=float(np.mean(power))
                vals[field+'_seed_std']=float(np.std(power,ddof=1))
            channel_powers[str(tick)][q]=vals
    result={'schema':1,'topic':'Microscopic nonlinear interaction of two targeted D4 eigenpacket perturbations',
      'measurement':'Four-state XOR mixed finite difference, per-seed; NOT physical cross section, not S-matrix',
      'initial_material':'independent half-filled 24-bit microstate via shared quantized uniform random threshold',
      'bit_dynamics':'unchanged exact reversible Bit8 collision and streaming',
      'quantized_threshold_bits_only_at_initial_preparation':16,
      'microscopic_state_bits_per_site':24,
      'all_cases':stats,
      'canonical_case':{'initial_flip_bits_by_seed':[
        [h['initial']['initial_bits_A'],h['initial']['initial_bits_B']] for h in raw0['seed_histories']],
        'first_nonlinear_response_tick_all_8_seeds':stats['baseline']['first_xor_interaction_tick_by_seed'],
        'eight_seed_response_xor_sites_at_tick4':float(np.mean(ref[:,4])),
        'eight_seed_response_xor_sites_at_tick14':float(np.mean(ref[:,14])),
        'eight_seed_sites_at_tick14_range':[int(ref[:,-1].min()),int(ref[:,-1].max())]},
      'causal_onset':[{'initial_packet_center_separation':n,'cut_radius':5,
         'minimum_gap':n-10,'measured_first_tick_all_seeds':stats[f'sep_{n}']['first_xor_interaction_tick_by_seed'] if n!=16 else stats['baseline']['first_xor_interaction_tick_by_seed']}
         for n in (12,16,20)],
      'paired_hidden_phase_comparison':phase,
      'outgoing_density_current_fourier_powers':channel_powers,
      'interpretation_limits':['Four-state XOR nonadditivity measures microscopic interaction; does not identify asymptotic scattering states',
        'Intensity of signed-density spectral residual is seed-averaged power and includes incoherent microstate noise',
        'Co-moving and w-polarized controls also create nonlinear difference, so a head-on collision is not uniquely responsible',
        'At late ticks the 4D torus and short y/z axes may influence spatial spreading',
        'No demonstrated stable binding or unique outgoing mode, no D4 to D3/D8 physical mechanism']}
    (ROOT/'data'/'bit24_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    out=run()
    print('11 controlled cases, total seeds',sum(v['settings']['seeds'] for v in out['all_cases'].values()))
    for name,x in out['all_cases'].items():
        print(name,'onset',x['first_xor_interaction_tick_by_seed'],'t8',round(x['mean_active_sites_by_tick'][8],1),'end',round(x['mean_active_sites_by_tick'][-1],1))
