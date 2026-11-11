#!/usr/bin/env python3
"""Paper-ready q-v plot from Bit20 results (no hidden q filtering in physics)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent.parent
r=json.loads((root/'data/bit20_results.json').read_text())
xs=[];vp=[];vm=[];v0=[];visp=[];vism=[];freep=[];freem=[];cip=[];cim=[]
for q in (1,2,3):
    rows={a['k_mode']:a for a in r['rows'] if a['q_mode']==q}
    p=rows[3];m=rows[-3];z=rows[0]
    xs.append(q*np.pi/8)
    vp.append(p['early_center_slope_2_to_8']);vm.append(m['early_center_slope_2_to_8']);v0.append(z['early_center_slope_2_to_8'])
    visp.append(p['visible_w0_early_center_slope_2_to_8']);vism.append(m['visible_w0_early_center_slope_2_to_8'])
    freep.append(p['early_free_center_slope_2_to_8']);freem.append(m['early_free_center_slope_2_to_8'])
    cip.append(p['seed_center_slope_95ci']);cim.append(m['seed_center_slope_95ci'])
fig,ax=plt.subplots(figsize=(8.5,5.3))
ax.axhline(0,ls=':',lw=1,color='gray')
ax.plot(xs,vp,'-o',label='k = +π/8, hidden q channel')
ax.plot(xs,vm,'-o',label='k = −π/8, hidden q channel')
ax.plot(xs,visp,'--s',label='k = +π/8, visible w=0 slice',alpha=.7)
ax.plot(xs,vism,'--s',label='k = −π/8, visible w=0 slice',alpha=.7)
ax.plot(xs,freep,':^',label='k = +π/8, no collision',alpha=.8)
ax.plot(xs,freem,':^',label='k = −π/8, no collision',alpha=.8)
ax.set_xticks(xs,['π/8','π/4','3π/8'])
ax.set(xlabel='Hidden wavenumber q',ylabel='Early intensity-centroid slope (sites/tick), ticks 2–8',title='Bit20: Direction reversal of finite-time packet response')
ax.grid(alpha=.2);ax.legend(fontsize=9,loc='lower right')
fig.tight_layout(); fig.savefig(root/'figures/bit20_q_center_velocities.svg');plt.close(fig)
