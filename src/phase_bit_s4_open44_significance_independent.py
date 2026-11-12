"""Enumerate exact seed-sign null for phase-locked mixed response; diagnostic, not a particle amplitude test."""
import itertools, json
from pathlib import Path
import numpy as np
D=Path('data')
files={'baseline':['bit25_left_a22_p00','bit25_left_a22_p00_part2'],
'phase90':['bit25_left_a22_p01'],'phase180':['bit25_left_a22_p02'],
'lowamp':['bit25_left_a10_p00'],'xw':['bit25_w_a22_p00','bit25_w_a22_p00_part2']}
res={}
for name,pats in files.items():
    z=np.concatenate([np.load(D/(p+'.npz'))['mode_amplitude'] for p in pats]).astype('complex128')
    js=json.loads((D/(pats[0]+'.json')).read_text()); modes=np.array(js['settings']['modes'])
    selectors={'q0':abs(modes[:,1])<1e-6,'qinput':abs(modes[:,1]-np.pi/4)<1e-6,'qdouble':abs(modes[:,1]-np.pi/2)<1e-6,
               'all':modes[:,1]>=-1e-6}
    print(name,len(z))
    res[name]={}
    n=len(z)
    signs=np.array(list(itertools.product((-1,1),repeat=n)),dtype=float)
    # one-sided coh test: total squared mean across time, k and two currents, no peak cherry-picking
    for window,ts in {'early':slice(4,8),'late':slice(8,13)}.items():
        for band,sel in selectors.items():
            sig=z[:,ts][:, :,sel,:]
            # all frequencies x currents ; aggregate powers over time and spatial modes
            total=float(np.mean(abs(sig)**2))
            observed=float(np.mean(abs(sig.mean(axis=0))**2))
            null=np.einsum('sn,ntmc->stmc',signs,sig,optimize=True)/n
            statistic=np.mean(abs(null)**2,axis=(1,2,3))
            pvalue=float(np.mean(statistic>=observed-1e-18))
            unbiased=(n*observed-total)/(n-1)
            row={'total_power':total,'coherent_power':observed,'coh_fraction':observed/total,'unbiased_coherent':unbiased,
                 'exact_signflip_p':pvalue,'n_signflip':len(signs)}
            res[name][f'{window}_{band}']=row
            print(window,band,'raw_frac',round(observed/total,3),'unbias',round(unbiased,9),'p',round(pvalue,3))
# matched baseline phase 90 and 180 direct complex cross correlations across seed/time/modes
z0=np.load(D/'bit25_left_a22_p00.npz')['mode_amplitude']
for key,file in [('phase90','bit25_left_a22_p01'),('phase180','bit25_left_a22_p02')]:
    zp=np.load(D/(file+'.npz'))['mode_amplitude']
    mode=np.array(json.loads((D/(file+'.json')).read_text())['settings']['modes'])
    res[key]['paired_phase_correlation']={}
    for band,v in [('q0',0.),('qdouble',np.pi/2),('qinput',np.pi/4)]:
        idx=np.where(abs(mode[:,1]-v)<1e-6)[0]
        a=z0[:,4:8][:,:,idx,:].ravel();b=zp[:,4:8][:,:,idx,:].ravel()
        co=np.vdot(a,b)/np.sqrt(np.vdot(a,a)*np.vdot(b,b))
        res[key]['paired_phase_correlation'][band]=[float(co.real),float(co.imag)]
        print('paired',key,band,'corr',co)
(D/'bit25_significance_independent.json').write_text(json.dumps(res,indent=2)+'\n')
