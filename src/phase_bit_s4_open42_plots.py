#!/usr/bin/env python3
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; FIG=ROOT/'figures';FIG.mkdir(parents=True,exist_ok=True)
P=json.loads((ROOT/'data/bit23_packet_predictions.json').read_text())
qs=np.array([1,2,3]);qnames=['π/8','π/4','3π/8']
fig,ax=plt.subplots(figsize=(8.6,4.8))
for sign,label,marker in [(1,'kₓ=+π/8','o'),(-1,'kₓ=−π/8','s')]:
 exact=[];pred=[]
 for q in qs:
  r=next(x for x in P['predictions'][str(q)]['rows'] if np.sign(x['k'])==sign)
  exact.append(r['exact_Bit21_velocity_2_8']);pred.append(r['tangent_velocity_2_8'])
 ax.plot(qs-.035,exact,marker=marker,linestyle='-',label=label+' · exakter Bit21-Lauf')
 ax.plot(qs+.035,pred,marker=marker,linestyle='--',label=label+' · Bit23-Vorhersage')
ax.axhline(0,linewidth=.8,color='gray');ax.set_xticks(qs,qnames)
ax.set_ylabel('Frühe Schwerpunktverschiebung / Tick');ax.set_xlabel('Verborgene Wellenzahl q')
ax.set_title('Ohne Anpassung: linearer Operator trifft die q-abhängige Bewegungsumkehr')
ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(FIG/'bit23_predicted_vs_exact.svg');fig.savefig(FIG/'bit23_predicted_vs_exact.png',dpi=150);plt.close(fig)
I=json.loads((ROOT/'data/bit23_mode_interference.json').read_text())
fig,ax=plt.subplots(figsize=(8.6,4.8))
T=np.arange(len(I['full_center_by_tick']))
for y,label in [(I['full_center_by_tick'],'Kohärente Summe · vorzeichenumgekehrt'),
 (I['no_cross_term_center_by_tick'],'Ohne Interferenz-Kreuzterme'),
 (I['family_centers']['x'],'Nur x-dominierte Moden'),
 (I['family_centers']['w'],'Nur w-dominierte Moden')]:
 ax.plot(T,y,marker='.',label=label)
ax.axhline(0,linewidth=.8,color='gray');ax.axvspan(2,8,alpha=.08)
ax.set_xlabel('Tick');ax.set_ylabel('Schwerpunkt x − x₀');ax.set_title('q=π/4: Negative Paketverschiebung entsteht durch Interferenz')
ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'bit23_interference_decomposition.svg');fig.savefig(FIG/'bit23_interference_decomposition.png',dpi=150);plt.close(fig)
print('wrote figures')
