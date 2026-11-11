#!/usr/bin/env python3
"""Bit23 out-of-fit check: predict Bit21 packet centroid from local 24x24 tangent.
NO fit to Bit20/21 data is performed. True Bit21 measurements are read only for comparison.
A slowly varying phase-coded acoustic-reference packet evolves under the
statistically linearized one-point theory K, then exact kinematics streams it.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from phase_bit_s4_open42_linear_collision import R

def simulate(K,k,q,nx=48,nw=32,ticks=12):
    xx=np.arange(nx);ww=np.arange(nw)
    dx=((xx-nx//2+nx//2)%nx)-nx//2
    gauss=np.exp(-.5*(dx/(nx/6.4))**2)
    wave=np.array([k,0,0,q]);beta=np.sqrt(2)/np.linalg.norm(wave)
    factors=1+beta*(R@wave)
    init=gauss[:,None,None]*np.exp(1j*(k*dx[:,None]+q*ww[None,:]))[:,:,None]*factors[None,None,:]
    f=init.copy();centers=[];widths=[];amps=[]
    for t in range(ticks+1):
        visible=f.sum(axis=2)
        channel=visible@np.exp(-1j*q*ww)/nw
        inten=abs(channel)**2
        norm=inten.sum()
        center=float((inten*dx).sum()/norm)
        width=float(np.sqrt(max(0,(inten*(dx-center)**2).sum()/norm)))
        centers.append(center);widths.append(width);amps.append(float(np.linalg.norm(channel)))
        if t<ticks:
            g=np.einsum('ij,xwj->xwi',K,f,optimize=True)
            f=np.empty_like(g)
            for d,v in enumerate(R.astype(int)):
                f[:,:,d]=np.roll(g[:,:,d],shift=(int(v[0]),int(v[3])),axis=(0,1))
    vel=float(np.polyfit(np.arange(2,9),np.asarray(centers)[2:9],1)[0])
    return {'velocity_2_8':vel,'centers':centers,'widths':widths,'l2_amplitudes_normalized':(np.array(amps)/amps[0]).tolist()}

def run():
    K=np.load(Path(__file__).resolve().parents[1]/'data/bit23_collision_operator.npz')['K_charge_exact']
    out={}
    for m in (1,2,3):
        q=m*np.pi/8
        ref=json.loads((Path(__file__).resolve().parents[1]/'data'/('bit21_w32_q'+str(m)+'.json')).read_text())
        rows=[]
        for sign in (1,-1):
            k=sign*np.pi/8
            model=simulate(K,k,q)
            free=simulate(np.eye(24),k,q)
            exact=next(row for row in ref['rows'] if np.sign(row['physical_k'])==sign)
            rows.append({'k':float(k),'exact_Bit21_velocity_2_8':exact['ensemble_center_slope'],
                'exact_Bit21_free_velocity_2_8':exact['free_ensemble_center_slope'],
                'tangent_velocity_2_8':model['velocity_2_8'],
                'tangent_free_velocity_2_8':free['velocity_2_8'],
                'tangent_centers':model['centers'],'exact_centers':exact['center_by_tick'][:len(model['centers'])],
                'tangent_widths':model['widths']})
        out[str(m)]={'q':float(q),'rows':rows}
    return {'test':'Bit23 out-of-sample 24x24 collision-tangent packet centroid predictor','shape':[48,4,4,32],
      'source_exact':'Bit21 eight-seed W32 runs, no fitted parameters from their velocities',
      'operator':'Bit23 800000 half-filled local collision samples, exact B4+charges',
      'predictions':out,'caveat':'Finite-amplitude exact bit dynamics may develop connected correlations beyond one-point tangent.'}

if __name__=='__main__':
    out=run();p=Path(__file__).resolve().parents[1]/'data/bit23_packet_predictions.json';p.write_text(json.dumps(out,indent=2)+'\n');
    for q,block in out['predictions'].items():
        for r in block['rows']:
            print('q',q,'k',np.sign(r['k']),'bit21',round(r['exact_Bit21_velocity_2_8'],3),'linear',round(r['tangent_velocity_2_8'],3),'free',round(r['tangent_free_velocity_2_8'],3),'freebit21',round(r['exact_Bit21_free_velocity_2_8'],3))
