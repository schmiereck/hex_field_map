#!/usr/bin/env python3
"""Scientific figures for Bit24; no modification of original project figures."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)

def run():
  d=json.loads((ROOT/'data/bit24_summary.json').read_text())['all_cases']
  fig,ax=plt.subplots(figsize=(8.2,4.5))
  for n,title in [('sep_12','Abstand 12'),('baseline','Abstand 16'),('sep_20','Abstand 20')]:
      r=d[n];y=r['mean_active_sites_by_tick'];ax.plot(np.arange(len(y)),np.array(y)/12288*100,marker='o',markersize=3,label=title)
  ax.set_xlabel('Tick');ax.set_ylabel('Knoten mit nichtlinearer XOR-Antwort [%]')
  ax.set_title('Bit 24 | Lokaler Wechselwirkungsbeginn nach Anfangsabstand')
  ax.legend();ax.grid(alpha=.25);fig.tight_layout()
  for suffix in ('svg','png'):fig.savefig(OUT/f'bit24_causal_fronts.{suffix}',dpi=160)
  plt.close(fig)

  fig,ax=plt.subplots(figsize=(8.2,4.5))
  for n,title in [('amp_010','Amplitude 0,10'),('amp_016','Amplitude 0,16'),('baseline','Amplitude 0,22'),('amp_030','Amplitude 0,30')]:
      r=d[n];y=r['mean_active_sites_by_tick'];ax.plot(range(len(y)),np.array(y)/12288*100,label=title)
  ax.set_xlabel('Tick');ax.set_ylabel('Betroffene Knoten [%]')
  ax.set_title('Bit 24 | Frühe Amplitudenabhängigkeit, spätes Zusammenwachsen')
  ax.legend();ax.grid(alpha=.25);fig.tight_layout()
  for suffix in ('svg','png'):fig.savefig(OUT/f'bit24_amplitude_growth.{suffix}',dpi=160)
  plt.close(fig)

  fig,ax=plt.subplots(figsize=(8.2,4.5))
  for n,title in [('baseline','x rechts + x links'),('branch_w','x rechts + w'),('branch_right','x rechts + x rechts')]:
      r=d[n];y=r['mean_active_sites_by_tick'];ax.plot(range(len(y)),np.array(y)/12288*100,label=title)
  ax.set_xlabel('Tick');ax.set_ylabel('Betroffene Knoten [%]')
  ax.set_title('Bit 24 | Nichtlinearität bei unterschiedlichen Modenpaaren')
  ax.legend();ax.grid(alpha=.25);fig.tight_layout()
  for suffix in ('svg','png'):fig.savefig(OUT/f'bit24_branch_controls.{suffix}',dpi=160)
  plt.close(fig)
  print('plots saved',len(list(OUT.iterdir())))

if __name__=='__main__':run()
