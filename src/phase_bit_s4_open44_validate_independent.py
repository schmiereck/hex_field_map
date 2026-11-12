"""Validate archived Bit25 Fourier response and exact four-state controls."""
import json
from pathlib import Path
import numpy as np
D=Path('data')
paths=['bit25_left_a22_p00','bit25_left_a22_p00_part2','bit25_left_a22_p01',
       'bit25_left_a22_p02','bit25_left_a10_p00','bit25_w_a22_p00','bit25_w_a22_p00_part2']
validation={'source':'saved exact Bit8 four-state runs','cases':len(paths),'tests':{}}
for stem in paths:
    p=json.loads((D/(stem+'.json')).read_text())
    z=np.load(D/(stem+'.npz'))['mode_amplitude'];local=np.load(D/(stem+'.npz'))['local']
    n=p['settings']['seeds'];t=p['settings']['ticks']; modes=np.asarray(p['settings']['modes'])
    assert z.shape==(n,t+1,25,3) and local.shape==(n,t+1,3)
    assert p['settings']['signed_charge_zero_all_times']
    assert p['settings']['free_four_state_response_identically_zero']
    assert p['settings']['first_seed_exact_reverse']
    assert np.all(local[:,0,:]==0)
    assert np.all(z[:,0,:,:]==0)
    zero=np.flatnonzero((abs(modes[:,0])<1e-9)&(abs(modes[:,1])<1e-9));assert len(zero)==1
    assert np.max(abs(z[:,:,zero[0],:]))<1e-12
    for j,(k,q) in enumerate(modes):
        match=np.flatnonzero(np.isclose(modes[:,0],-k)&np.isclose(modes[:,1],-q))
        if len(match)==1:assert np.max(abs(z[:,:,j,:]-z[:,:,match[0],:].conj()))<1e-10
    assert all(0<=x<=t for x in p['first_nonzero_per_seed'])
    validation['tests'][stem]={'states':int(4*n),'time_steps':t,'zero_mode_global_charges':True,
      'complex_conjugation':True,'exact_free_null':True,'reverse_first':True}
Path('data/bit25_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
