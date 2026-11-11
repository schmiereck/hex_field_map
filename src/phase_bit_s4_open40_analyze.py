#!/usr/bin/env python3
"""Bit 21: finite-geometry falsification and diagnostic of k-dependence.

Reads complete per-seed JSON/NPZ from phase_bit_s4_open40_falsify_q.py.
All quoted velocities are finite-time centers of a complex *diagnostic* q-filtered
signal, NOT a certified group velocity or single-mode dispersion eigenvalue.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from phase_bit_s4_open39_analyze import metrics

NAMES=['w32_q1','w32_q2','w32_q3','yz8_q1','yz8_q2','yz8_q3','joint_q2','w32_q2_k_scan']

def average_ci(a,b,seed=1,reps=10000):
    rng=np.random.default_rng(seed)
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    ai=rng.integers(len(a),size=(reps,len(a)));bi=rng.integers(len(b),size=(reps,len(b)))
    bs=((a[ai].mean(axis=1)+b[bi].mean(axis=1))/2,
        (a[ai].mean(axis=1)-b[bi].mean(axis=1))/2)
    return {'even_mean':float((a.mean()+b.mean())/2),'even_95ci':[float(v) for v in np.quantile(bs[0],[.025,.975])],
            'odd_mean':float((a.mean()-b.mean())/2),'odd_95ci':[float(v) for v in np.quantile(bs[1],[.025,.975])]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dir',type=Path,default=Path('data'));p.add_argument('--bit20',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--figure',type=Path,required=True);p.add_argument('--kfigure',type=Path,required=True)
    args=p.parse_args();base=json.load(open(args.bit20));data={}
    for name in NAMES:
        r=json.load(open(args.dir/f'bit21_{name}.json'))
        d=dict(r)
        assert r['independent_seeds_per_case']==(4 if name=='joint_q2' else 8)
        assert r['ticks']==12
        assert all(x['physical_q']==r['physical_q'] for x in r['rows'])
        if name!='w32_q2_k_scan':
            plus=next(x for x in r['rows'] if x['k_mode']==3)
            minus=next(x for x in r['rows'] if x['k_mode']==-3)
            d['odd_even_center_velocity']=average_ci(plus['seed_center_slopes'],minus['seed_center_slopes'],seed=2100+len(name))
            d['positive_seed_all_reversed'] = bool(all(x<0 for x in plus['seed_center_slopes'])) if r['physical_q']==np.pi/4 else None
            d['negative_seed_all_reversed'] = bool(all(x>0 for x in minus['seed_center_slopes'])) if r['physical_q']==np.pi/4 else None
            assert np.isclose(plus['physical_k'],-minus['physical_k'])
            assert (plus['early_window']==minus['early_window']==[2,8])
        data[name]=d
    raw20_byq={}
    for row in base['rows']:
        if row['k_mode'] in (-3,3):
            raw20_byq[(row['q_mode'],row['k_mode'])]=row['early_center_slope_2_to_8']
    # Spectral control: early phase slope vs k is NOT enough to infer transport.
    ph={}
    for km in (2,3,4):
        suffix='w32_q2_k_scan' if km!=3 else 'w32_q2'
        with np.load(args.dir/f'bit21_{suffix}.npz') as arrays:
            ph[str(km)]={}
            for typ in ('bit8','free'):
                arr=arrays[f'q2_k{km:+d}_{typ}'];met=metrics(arr,km,[48,4,4,32])
                idx=np.arange(2,9)
                omega=-float(np.polyfit(idx,met['phase'][idx],1)[0])
                ph[str(km)][typ]={'early_demod_phase_frequency':omega,'coherent_amplitude_ratio_at_8':float(met['amplitude'][8]),
                                  'coherent_amplitude_ratio_at_12':float(met['amplitude'][12])}
    kdiff=(4-2)*2*np.pi/48
    inferred={typ:(ph['4'][typ]['early_demod_phase_frequency']-ph['2'][typ]['early_demod_phase_frequency'])/kdiff
              for typ in ('bit8','free')}
    result={'name':'Bit21 geometry controls of the q-dependent packet response',
            'microscopic_rule':'Bit8 unchanged; JIT exact benchmarked reference',
            'bit20_original_shape':base['shape'],
            'bit20_original_qk_center_slopes':{f'q{q}_k{k:+d}':v for (q,k),v in raw20_byq.items()},
            'cases':data,'q2_k_scan_phase_frequency':ph,'q2_finite_difference_frequency_slopes':inferred,
            'frequency_warning':'The free-streaming control has negative d(early fitted omega)/dk but POSITIVE packet displacement. This simple frequency finite difference must NOT be presented as group velocity of an eigenmode.',
            'physical_limitations':['Finite t=2..8 window, width broadening by t=12',
             'Transient packet center motion and demodulated phase are not sufficient to identify a single dispersion eigenmode',
             'Two separate geometry enlargements have eight seeds; combined enlargement has four seeds',
             'Amplitude at k_mode 4 has strong decay by tick 12',
             'Periodic x length remains 48; no thermodynamic-limit or long-time claim',
             'No result concerning a D3/D4/D8 interaction mechanism']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')

    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    qs=np.array([np.pi/8,np.pi/4,3*np.pi/8])
    for geoname,label in (('w32','Hidden period W=32'),('yz8','Transverse periods y=z=8')):
        values=[next(x for x in data[f'{geoname}_q{j}']['rows'] if x['k_mode']==3)['ensemble_center_slope'] for j in (1,2,3)]
        ax.plot(qs,values,marker='o',label=label)
    ax.axhline(0,lw=1,ls='--')
    ax.set_xticks(qs,['π/8','π/4','3π/8']);ax.set_xlabel('Physical hidden wavenumber q');ax.set_ylabel('Early +kx packet center slope, sites/tick')
    ax.set_title('Bit21: sign reversal persists in independent finite-size controls');ax.grid(alpha=.25);ax.legend();fig.tight_layout();args.figure.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.figure);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.2,4.8))
    xkm=[2,3,4]
    for typ,label in [('bit8','Interacting Bit8'),('free','Free streaming')]:
        vals=[]
        for j in xkm:
            name='w32_q2' if j==3 else 'w32_q2_k_scan'
            vals.append(next(x for x in data[name]['rows'] if x['k_mode']==j)['ensemble_center_slope'] if typ=='bit8' else next(x for x in data[name]['rows'] if x['k_mode']==j)['free_ensemble_center_slope'])
        ax.plot(xkm,vals,marker='o',label=label)
    ax.axhline(0,lw=1,ls='--');ax.set_xticks(xkm,['π/12','π/8','π/6']);ax.set_xlabel('Physical visible kx');ax.set_ylabel('Early packet center slope, sites/tick')
    ax.set_title('Bit21: q=π/4 sign reversal is not universal across kx');ax.legend();ax.grid(alpha=.25);fig.tight_layout();args.kfigure.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.kfigure);plt.close(fig)
    print('Saved summary',args.out)
    for name in NAMES:
        rr=data[name]
        print(name,'shape',rr['shape'],'velocities',[(x['k_mode'],round(x['ensemble_center_slope'],3)) for x in rr['rows']]);
    print('q2 finite_difference_frequency_slopes',inferred)

if __name__=='__main__':main()
