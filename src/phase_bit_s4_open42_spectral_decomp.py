#!/usr/bin/env python3
"""Bit23: explain the q-specific centroid via superposition of kinetic eigenmodes.
Spectral amplitudes of the *initial acoustic-reference directional profile* are
computed in the eigenbasis of D(k,q) K; no fitting to exact trajectories.
"""
from pathlib import Path
import json,numpy as np
from phase_bit_s4_open42_linear_collision import R,derived
from phase_bit_s4_open42_predict_packets import simulate

def out():
    data=np.load(Path(__file__).resolve().parents[1]/'data/bit23_collision_operator.npz')
    K=data['K_charge_exact'];kb=data['K_batches']
    dispersion=[]
    for qid in [1,2,3]:
        q=qid*np.pi/8;k=np.pi/8
        wave=np.array([k,0,0,q]); beta=np.sqrt(2)/np.linalg.norm(wave)
        acoustic=1+beta*(R@wave)
        D=np.exp(-1j*(k*R[:,0]+q*R[:,3]))
        M=D[:,None]*K
        eig,vec=np.linalg.eig(M)
        alpha=np.linalg.solve(vec,acoustic)
        contributions=[]
        for idx in range(24):
            c=alpha[idx]*vec[:,idx]
            contributions.append({'eigen_id':idx,'phase':float(np.angle(eig[idx])),
                'damping':float(abs(eig[idx])), 'density_amplitude':float(abs(np.sum(c))),
                'jx_amplitude':float(abs(R[:,0]@c)), 'jw_amplitude':float(abs(R[:,3]@c)),
                'density_complex':[float(np.sum(c).real),float(np.sum(c).imag)],
                'significant_density_fraction':float(abs(np.sum(c))/abs(acoustic.sum()))})
        contributions.sort(key=lambda a:-a['density_amplitude'])
        dispersion.append({'q':float(q),'q_index':qid,'acoustic_profile_density_initial':float(acoustic.sum()),
            'main_eigencomponents_by_density':contributions[:12]})
    boot={}
    for qi in (1,2,3):
        q=qi*np.pi/8;series={}
        for sign in (-1,1):
            k=sign*np.pi/8
            preds=[simulate(derived(s),k,q)['velocity_2_8'] for s in kb]
            series[str(sign)]={'eight_batch_predictions':preds,'mean':float(np.mean(preds)),
                'sd_across_batches':float(np.std(preds,ddof=1)),
                'sem_ensemble_estimate':float(np.std(preds,ddof=1)/np.sqrt(len(preds)))}
        boot[str(qi)]=series
    return {'decomposition':dispersion,'independent_operator_batch_predictive_checks':boot,
            'note':'B4 sym+charge projection removes known exact constraints, each batch otherwise independent; spread reflects collision-operator sampling error, not Bit21 finite-seed fluctuations'}
if __name__=='__main__':
 d=out();p=Path(__file__).resolve().parents[1]/'data/bit23_spectral_decomposition.json';p.write_text(json.dumps(d,indent=2)+'\n')
 for case in d['decomposition']:
  print('\nq',case['q_index'])
  for m in case['main_eigencomponents_by_density'][:8]:print('phase',round(m['phase'],3),'damp',round(m['damping'],3),'dens_amp',round(m['density_amplitude'],3),'jx_amp',round(m['jx_amplitude'],3),'jw_amp',round(m['jw_amplitude'],3))
 for q,s in d['independent_operator_batch_predictive_checks'].items():
  print('q',q,[(sign,round(v['mean'],4),round(v['sem_ensemble_estimate'],4)) for sign,v in s.items()])
