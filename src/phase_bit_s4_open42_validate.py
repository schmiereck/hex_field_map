#!/usr/bin/env python3
"""Bit23 independent integrity/physics control: exact collision, tangent,
predictive holdout vs previously archived Bit21, separate eigenmode tests."""
from pathlib import Path
import json,hashlib,numpy as np
from phase_bit_s4_open39_numba_collision import audit,collide_numba
from phase_bit_s4_open25_symmetry_recovery import symmetry_maps,permute
from phase_bit_s4_open27_active_collision import ROOTS,charge
from phase_bit_s4_open42_linear_collision import collision_invariant_check,A,P,Q

BASE=Path(__file__).resolve().parents[1]

def validate():
    audited=audit(n=900,seed=4223)
    rng=np.random.default_rng(4233);sym=symmetry_maps(ROOTS)
    for _ in range(100):
        s=int(rng.integers(0,1<<24));t=collide_numba(s)
        assert charge(s)==charge(t)
        for key,perm in sym.items():assert collide_numba(permute(s,perm))==permute(t,perm),(key,s)
    op=json.loads((BASE/'data/bit23_collision_operator.json').read_text()); n=np.load(BASE/'data/bit23_collision_operator.npz')
    C=n['K_charge_exact'];K=n['K_empirical_B4']
    assert abs(op['raw_mean_validation']['left_charge_error_inf'])<.005
    assert collision_invariant_check(C)['left_charge_error_inf']<1e-11
    assert max(abs(np.linalg.eigvalsh(C)))<=1+1e-11
    assert np.max(abs(C-C.T))<1e-11
    assert np.max(abs(P@Q))<1e-12
    pred=json.loads((BASE/'data/bit23_packet_predictions.json').read_text());rr=[]
    for q,block in pred['predictions'].items():
        for x in block['rows']:
            err=abs(x['exact_Bit21_velocity_2_8']-x['tangent_velocity_2_8']);rr.append(err)
    assert len(rr)==6 and max(rr)<.035
    for q,sg in [(1,1),(2,-1),(3,1)]:
        r=next(x for x in pred['predictions'][str(q)]['rows'] if x['k']>0)
        assert np.sign(r['tangent_velocity_2_8'])==sg
    probes={}
    for name in ['q2_x','q2_w','q1_mixed','q3_x']:
        obj=json.loads((BASE/f'data/bit23_eigenprobe_{name}.json').read_text())
        assert obj['first_seed_per_case_forward_backward_exact']
        entry=obj['cases'][0]
        assert entry['seed_count']==8 and entry['time_ticks']==10
        assert entry['t1_prediction_relative_complex_vector_error']<.04
        assert entry['one_step_measured_minus_eigenvalue']<.01
        assert entry['orthogonal_energy_fraction_t'][-1]<.02
        probes[name]={'t1_vector_relative_error':entry['t1_prediction_relative_complex_vector_error'],
           'one_step_complex_ratio_error':entry['one_step_measured_minus_eigenvalue'],
           't10_orthogonal_energy_fraction':entry['orthogonal_energy_fraction_t'][-1],
           't10_amplitude_measured':entry['projected_amplitude_ratio_t'][-1],
           't10_amplitude_linear_predicted':entry['predicted_amplitude_ratio_t'][-1]}
    inter=json.loads((BASE/'data/bit23_mode_interference.json').read_text())
    assert inter['reconstruction_center_error_max']<1e-8
    assert inter['full_velocity_2_8']<-.3 and inter['no_cross_terms_velocity_2_8']>0
    assert inter['family_center_velocity_2_8']['x']>.8
    return {'test':'Bit23 independently checked diagnostics','collision_reference':audited,
       'B4_symmetry_100_local_states':True,'five_invariants_and_reversibility':'passed',
       'tangent_matrix_charge_error_inf':collision_invariant_check(C)['left_charge_error_inf'],
       'negative_q2_and_positive_neighbor_signs_reproduced':True,
       'out_of_fit_six_packet_slope_errors':rr,'max_packet_slope_error':max(rr),
       'eigenmode_tests':probes,'interference_cross_term_sign_flip':True,
       'interference_reconstruction_center_error_max':inter['reconstruction_center_error_max'],
       'no_claim_of_exact_multitick_tangent_closure':True,'passed':True}

if __name__=='__main__':
 d=validate(); (BASE/'data/bit23_validation.json').write_text(json.dumps(d,indent=2)+'\n')
 print(json.dumps(d,indent=2))
