from pathlib import Path
import numpy as np,json
from phase_bit_s4_open44_mode_transfer import summarize
D=Path('data')
runs={'baseline':['bit25_left_a22_p00','bit25_left_a22_p00_part2'],'phase90':['bit25_left_a22_p01'],'phase180':['bit25_left_a22_p02'],'lowamp':['bit25_left_a10_p00'],'xw':['bit25_w_a22_p00','bit25_w_a22_p00_part2']}
out={}
for name,paths in runs.items():
  js=[json.loads((D/(x+'.json')).read_text()) for x in paths]
  a=[np.load(D/(x+'.npz')) for x in paths]
  z=np.concatenate([x['mode_amplitude'] for x in a],axis=0)
  loc=np.concatenate([x['local'] for x in a],axis=0)
  meta=js[0]['settings'].copy();meta['seeds']=len(z)
  obj=summarize(meta,z,loc)
  out[name]=obj
  print('\n',name,'SEEDS',len(z),'first',obj['first_nonzero_per_seed'],'t12 sites',obj['mean_signed_channel_sites'][-1])
  # grouped by q over k for signal and total
  for q in [0,.25,.5]:
    # use nonnegative q only
    modes=[x for key,x in obj['mode_statistics'].items() if f'q={q:+.3f}pi' in key]
    m0=sum(x['density']['unbiased_coherent_power_estimate'] for x in modes)
    raw=sum(x['density']['mean_total_power'] for x in modes)
    x0=sum(x['current_x']['unbiased_coherent_power_estimate'] for x in modes)
    w0=sum(x['current_w']['unbiased_coherent_power_estimate'] for x in modes)
    print('q',q,'density adjusted',round(m0,8),'raw',round(raw,8),'Jx',round(x0,8),'Jw',round(w0,8))
  print('Top',obj['top_density_modes'][:4])
(D/'bit25_synthesis.json').write_text(json.dumps(out,indent=2)+'\n')
