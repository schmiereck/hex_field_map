#!/usr/bin/env python3
"""Reproducible Bit24 24-channel tangent eigenvectors from a frozen matrix.

The 24x24 real matrix is a *statistical* preparation aid, never the actual
microscopic collision rule. Every Bit24 tick uses the unchanged exact Bit8 rule.
The frozen matrix comes from the independent supplemental Bit23 400k-state
sample and has known five-charge and B4 symmetry projection.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from phase_bit_s4_open27_active_collision import ROOTS as DISCRETE_ROOTS
ROOTS=np.array(DISCRETE_ROOTS,dtype=np.float64)
BITS=np.uint32(1)<<np.arange(24,dtype=np.uint32)

def matrix_from_json(path):
    obj=json.loads(Path(path).read_text())
    a=np.array(obj['projected_matrix'],dtype=np.float64)
    assert a.shape==(24,24) and np.all(np.isfinite(a))
    return a

def modes(A,k,q):
    phase=ROOTS[:,0]*k+ROOTS[:,3]*q
    M=np.exp(-1j*phase)[:,None]*A
    return np.linalg.eig(M)

def select(A,k,q,component,sign=1):
    eig,vecs=modes(A,k,q)
    norm=(24*np.sum(abs(vecs)**2,axis=0))
    w=np.abs(ROOTS[:,component]@vecs)**2/norm
    d=np.abs(vecs.sum(axis=0))**2/norm
    candidates=np.where((sign*(-np.angle(eig))>.1)&(d>.07))[0]
    if len(candidates)==0:raise RuntimeError('no mode with requested polarity')
    j=int(candidates[np.argmax(w[candidates])])
    v=vecs[:,j]
    v=v*np.exp(-1j*np.angle(v.sum()))/max(np.abs(v))
    epsk=.015;freq=[];matches=[]
    for newk in (k-epsk,k+epsk):
        ev,evc=modes(A,newk,q)
        ov=abs(evc.conj().T@vecs[:,j])/np.linalg.norm(evc,axis=0)/np.linalg.norm(vecs[:,j]);jj=int(np.argmax(ov))
        matches.append(float(ov[jj]));freq.append(float(-np.angle(ev[jj])))
    vg=float((np.unwrap(freq)[1]-np.unwrap(freq)[0])/(2*epsk))
    return v,{'frequency':float(-np.angle(eig[j])),'modulus':float(abs(eig[j])),
      'density_overlap':float(d[j]),
      'xcurrent_overlap':float(abs(ROOTS[:,0]@vecs[:,j])**2/norm[j]),
      'wcurrent_overlap':float(abs(ROOTS[:,3]@vecs[:,j])**2/norm[j]),
      'group_velocity':vg,'nearby_eigenvector_overlap':matches}
