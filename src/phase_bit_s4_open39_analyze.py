#!/usr/bin/env python3
"""Bit20: phase, amplitude, widths and honest seed-level bootstrap uncertainty.

This external analysis never influences the microscopic Boolean dynamics.
"""
import argparse,json
from pathlib import Path
import numpy as np
from phase_bit_s4_open38_phased_packet import pos_width

CHANNELS=('visible_w0','hidden_fourier_q','unweighted_w_mean')

def metrics(arr,km,shape):
    # arr: Nseed × time × 3channels × Nx
    nx=shape[0];x0=nx//2
    x=np.arange(nx)-x0
    k=2*np.pi*km/nx
    env=np.exp(-.5*(x/(nx/6.4))**2)
    demod=np.exp(-1j*k*x)*env
    mean=arr.mean(axis=0)
    z=(mean*demod[None,None,:]).sum(axis=-1)/env.sum()
    seed_z=(arr*demod[None,None,None,:]).sum(axis=-1)/env.sum()
    amplitude=np.abs(z[:,1])/max(abs(z[0,1]),1e-9)
    centers=[];widths=[]
    for row in mean[:,1,:]:
        loc,width=pos_width(abs(row)**2,x0)
        centers.append(loc);widths.append(width)
    phase=np.unwrap(np.angle(z[:,1]*np.conjugate(z[0,1])))
    return {'amplitude':amplitude,'z':z,'seed_z':seed_z,'centers':np.array(centers),
      'widths':np.array(widths),'phase':phase}

def bootstrap_ci(x,seed=1001,reps=500):
    rng=np.random.default_rng(seed)
    vals=np.asarray(x,dtype=float)
    n=len(vals)
    picks=rng.integers(0,n,size=(reps,n))
    boot=vals[picks].mean(axis=1)
    return [float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]

def analyze(meta,raw):
    results=[]
    for case in meta['cases']:
        km,qm=case['k_mode'],case['q_mode']
        bx=raw[case['raw_array_key_bit8']];fx=raw[case['raw_array_key_free']]
        bm=metrics(bx,km,meta['shape']);fm=metrics(fx,km,meta['shape'])
        t0=2;t1=min(8,meta['ticks']); ti=np.arange(t0,t1+1)
        # Seed-level individual demod temporal slopes; pairing across time avoids overconfident time-bin SEs.
        seed_phases=np.unwrap(np.angle(bm['seed_z'][:,:,1]*np.conjugate(bm['seed_z'][:,0:1,1])),axis=1)
        seed_slopes=np.array([np.polyfit(ti,p[ti],1)[0] for p in seed_phases]);
        seed_centers=np.array([[pos_width(abs(row)**2,meta['shape'][0]//2)[0] for row in seed[:,1,:]] for seed in bx]);
        seed_center_slopes=np.array([np.polyfit(ti,cent[ti],1)[0] for cent in seed_centers]);
        visible_mean=bx.mean(axis=0)[:,0,:]; free_visible_mean=fx.mean(axis=0)[:,0,:];
        visible_centers=np.array([pos_width(abs(row)**2,meta['shape'][0]//2)[0] for row in visible_mean]);
        free_visible_centers=np.array([pos_width(abs(row)**2,meta['shape'][0]//2)[0] for row in free_visible_mean]);
        visible_seed_centers=np.array([[pos_width(abs(row)**2,meta['shape'][0]//2)[0] for row in seed[:,0,:]] for seed in bx]);
        visible_seed_slopes=np.array([np.polyfit(ti,cent[ti],1)[0] for cent in visible_seed_centers]);
        seed_amplitudes=np.abs(bm['seed_z'][:,:,1]);
        seed_end_ratios=seed_amplitudes[:,-1]/np.maximum(seed_amplitudes[:,0],1e-9);
        cohs=np.abs(bm['z'][:,1])**2
        fluctuations=np.mean(np.abs(bx[:,:,1,:]-bx.mean(axis=0)[None,:,1,:])**2,axis=(0,2))
        coherent_power=np.mean(np.abs(bx.mean(axis=0)[:,1,:])**2,axis=1)
        report={'q_mode':qm,'q':float(2*np.pi*qm/meta['shape'][3]),'k_mode':km,'k':float(2*np.pi*km/meta['shape'][0]),
           'seed_count':len(bx),'initial_hidden_amplitude':float(abs(bm['z'][0,1])),
           'initial_visible_amplitude':float(abs(bm['z'][0,0])),
           'initial_w_average_amplitude':float(abs(bm['z'][0,2])),
           'amplitude_ratio_by_tick':[float(v) for v in bm['amplitude']],
           'free_amplitude_ratio_by_tick':[float(v) for v in fm['amplitude']],
           'width_by_tick':[float(v) for v in bm['widths']],
           'center_by_tick':[float(v) for v in bm['centers']],
           'free_center_by_tick':[float(v) for v in fm['centers']],
           'early_center_slope_2_to_8':float(np.polyfit(ti,bm['centers'][ti],1)[0]),
           'early_free_center_slope_2_to_8':float(np.polyfit(ti,fm['centers'][ti],1)[0]),
           'visible_w0_early_center_slope_2_to_8':float(np.polyfit(ti,visible_centers[ti],1)[0]),
           'visible_w0_free_early_center_slope_2_to_8':float(np.polyfit(ti,free_visible_centers[ti],1)[0]),
           'visible_w0_individual_seed_center_slope_mean':float(visible_seed_slopes.mean()),
           'visible_w0_seed_center_slope_95ci':bootstrap_ci(visible_seed_slopes,seed=6000+qm*10+km),
           'early_phase_slope_2_to_8':float(np.polyfit(ti,bm['phase'][ti],1)[0]),
           'early_free_phase_slope_2_to_8':float(np.polyfit(ti,fm['phase'][ti],1)[0]),
           'mid_phase_slope_9_to_16':float(np.polyfit(np.arange(9,17),bm['phase'][9:17],1)[0]) if meta['ticks']>=16 else None,
           'late_phase_slope_17_to_24':float(np.polyfit(np.arange(17,25),bm['phase'][17:25],1)[0]) if meta['ticks']>=24 else None,
           'mid_free_phase_slope_9_to_16':float(np.polyfit(np.arange(9,17),fm['phase'][9:17],1)[0]) if meta['ticks']>=16 else None,
           'seed_phase_slope_mean':float(np.mean(seed_slopes)),
           'seed_phase_slope_95ci':bootstrap_ci(seed_slopes,seed=2000+qm*10+km),
           'seed_phase_slopes':[float(v) for v in seed_slopes],
           'seed_center_slopes':[float(v) for v in seed_center_slopes],
           'seed_center_slope_mean':float(np.mean(seed_center_slopes)),
           'seed_center_slope_95ci':bootstrap_ci(seed_center_slopes,seed=3000+qm*10+km),
           'individual_seed_last_amplitude_ratio_mean':float(seed_end_ratios.mean()),
           'individual_seed_last_amplitude_ratio_95ci':bootstrap_ci(seed_end_ratios,seed=4000+qm*10+km),
           'coherent_power_over_seed_variance_tick_8':float(coherent_power[t1]/max(1e-14,fluctuations[t1])),
           'coherent_power_over_seed_variance_last':float(coherent_power[-1]/max(1e-14,fluctuations[-1])),
           'sample_ratio_tick8':float(bm['amplitude'][t1]),
           'sample_ratio_tick_last':float(bm['amplitude'][-1]),
           'free_sample_ratio_tick_last':float(fm['amplitude'][-1]),
           'width_tick0':float(bm['widths'][0]),'width_tick_last':float(bm['widths'][-1])}
        results.append(report)
    byq=[]
    for qm in meta['hidden_q_modes']:
        series={row['k_mode']:row for row in results if row['q_mode']==qm}
        plus,minus=series[3],series[-3]
        rng=np.random.default_rng(5000+qm)
        vp=np.asarray(plus['seed_center_slopes']);vm=np.asarray(minus['seed_center_slopes'])
        bs=np.array([(vp[rng.integers(0,len(vp),size=len(vp))].mean()+vm[rng.integers(0,len(vm),size=len(vm))].mean())/2 for _ in range(1000)])
        byq.append({'q_mode':qm,
          'independent_seed_even_drift_95ci':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],
          'independent_seed_even_drift_estimate':float((vp.mean()+vm.mean())/2),
          'mean_odd_slope_2_to_8':float((plus['early_center_slope_2_to_8']-minus['early_center_slope_2_to_8'])/2),
          'even_drift_slope_2_to_8':float((plus['early_center_slope_2_to_8']+minus['early_center_slope_2_to_8'])/2),
          'visible_w0_even_drift_slope_2_to_8':float((plus['visible_w0_early_center_slope_2_to_8']+minus['visible_w0_early_center_slope_2_to_8'])/2),
          'visible_w0_odd_slope_2_to_8':float((plus['visible_w0_early_center_slope_2_to_8']-minus['visible_w0_early_center_slope_2_to_8'])/2),
          'phase_frequency_k0':series[0]['early_phase_slope_2_to_8'],
          'phase_frequency_plus':plus['early_phase_slope_2_to_8'],
          'phase_frequency_minus':minus['early_phase_slope_2_to_8'],
          'caveat':'Early slopes are descriptive finite-window packet observables, not certified dispersion eigenfrequencies'})
    return {'status':'completed','raw_data':'data/bit20_raw.npz','independent_seeds_per_qk':meta['independent_seeds_per_case'],
      'shape':meta['shape'],'ticks':meta['ticks'],'rows':results,'q_comparison':byq,
      'limits':meta['limits'],'no_stable_particle_claim':True,'no_mass_claim':True}

def make_figure(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.8))
    for r in result['rows']:
        if r['k_mode']==3:
            ax.plot(r['amplitude_ratio_by_tick'],label=f"q index {r['q_mode']}, +k")
            ax.plot(r['free_amplitude_ratio_by_tick'],'--',alpha=.45)
    ax.set(xlabel='Tick',ylabel='Coherent hidden-q carrier amplitude / initial',title='Bit20: exact Bit-8 collision (solid) vs free streaming (dashed)')
    ax.legend();ax.grid(alpha=.2)
    path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path);plt.close(fig)

def main():
    a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,default=Path('data/bit20_raw.npz'));a.add_argument('--metadata',type=Path,default=Path('data/bit20_metadata.json'));a.add_argument('--out',type=Path,default=Path('data/bit20_results.json'));a.add_argument('--figure',type=Path,default=Path('figures/bit20_q_amplitudes.svg'))
    args=a.parse_args();meta=json.loads(args.metadata.read_text());arr=np.load(args.raw)
    result=analyze(meta,arr);args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    make_figure(result,args.figure)
    for r in result['rows']:
        print('q',r['q_mode'],'k',r['k_mode'],'A24',round(r['sample_ratio_tick_last'],3),'width',round(r['width_tick0'],2),'->',round(r['width_tick_last'],2),'v2-8',round(r['early_center_slope_2_to_8'],3),flush=True)
if __name__=='__main__':main()
