#!/usr/bin/env python3
"""Bit23 Fourier eigenmode decomposition of q=pi/4 phase packet.
The x-FFT of the full linear packet is exactly evolved per k using M(k,q).
Decompose by the polarization of each 24x24 eigenvector into x-like,
w-like and remaining branches. The sum recovers the full response; an
'incoherent' intensity sum removes all cross-terms, as interference control.
"""
from pathlib import Path
import numpy as np,json
from phase_bit_s4_open42_linear_collision import R
from phase_bit_s4_open42_predict_packets import simulate

def center(intensity,nx):
    xx=np.arange(nx); dx=((xx-nx//2+nx//2)%nx)-nx//2
    n=np.sum(intensity)
    return float(np.sum(dx*intensity)/n) if n>1e-12 else 0.0

def analyze(q=np.pi/4,k0=np.pi/8,nx=48,ticks=12):
    K=np.load(Path(__file__).resolve().parents[1]/'data/bit23_collision_operator.npz')['K_charge_exact']
    x=np.arange(nx);dx=((x-nx//2+nx//2)%nx)-nx//2
    wave=np.array([k0,0,0,q]);a=1+np.sqrt(2)/np.linalg.norm(wave)*(R@wave)
    init=np.exp(-.5*(dx/(nx/6.4))**2)*np.exp(1j*k0*dx)
    g0=init[:,None]*a[None,:]
    trans=np.fft.fft(g0,axis=0)
    family_names=['x','w','rest'];gout=np.zeros((3,ticks+1,nx,24),dtype=complex)
    max_condition=0.;max_reconstruction_error=0.
    for n,k in enumerate(2*np.pi*np.fft.fftfreq(nx)):
        M=np.exp(-1j*(k*R[:,0]+q*R[:,3]))[:,None]*K
        eigen,V=np.linalg.eig(M)
        max_condition=max(max_condition,float(np.linalg.cond(V)))
        c=np.linalg.solve(V,trans[n]);rec=V@c
        max_reconstruction_error=max(max_reconstruction_error,float(np.max(abs(rec-trans[n]))))
        # Norm of eigen-directional x and w currents (relative polarization).
        Jx=abs(R[:,0]@V)**2;Jw=abs(R[:,3]@V)**2
        # Modes with no current, including some density background, are 'rest'.
        groups=np.full(24,2,dtype=int)
        groups[(Jx>0.02)&(Jx>3*Jw)]=0
        groups[(Jw>0.02)&(Jw>3*Jx)]=1
        for t in range(ticks+1):
            v=V@(c*eigen**t)
            for j in range(3):
                vi=V@(c*eigen**t*(groups==j))
                gout[j,t,n]=vi
    responses=np.fft.ifft(gout,axis=2).sum(axis=-1)
    full=np.sum(responses,axis=0)
    target=simulate(K,k0,q,nx=nx,nw=32,ticks=ticks)
    errors=np.abs(np.array([center(abs(full[t])**2,nx) for t in range(ticks+1)])-target['centers'])
    xc=np.array([center(abs(full[t])**2,nx) for t in range(ticks+1)])
    incoherent_intensities=np.sum(abs(responses)**2,axis=0)
    ic=np.array([center(incoherent_intensities[t],nx) for t in range(ticks+1)])
    separate=np.array([[center(abs(responses[j,t])**2,nx) for t in range(ticks+1)] for j in range(3)])
    sl=lambda a:float(np.polyfit(np.arange(2,9),a[2:9],1)[0])
    return {'q':float(q),'k':float(k0),'full_velocity_2_8':sl(xc),
        'no_cross_terms_velocity_2_8':sl(ic),
        'family_center_velocity_2_8':{name:sl(separate[j]) for j,name in enumerate(family_names)},
        'family_average_power_fractions':{name:float(np.sum(abs(responses[j,2:9])**2)/np.sum(abs(responses[:,2:9])**2)) for j,name in enumerate(family_names)},
        'reconstruction_center_error_max':float(np.max(errors)),
        'maximum_eigenvector_condition_number':max_condition,
        'maximum_fourier_inversion_error':max_reconstruction_error,
        'full_center_by_tick':xc.tolist(),'no_cross_term_center_by_tick':ic.tolist(),
        'family_centers':{name:separate[j].tolist() for j,name in enumerate(family_names)},
        'criterion':'x: abs(Jx)^2>0.02 and >3 abs(Jw)^2; w similarly; rest otherwise',
        'caveats':['polarization partitions can jump where branches cross; approximate diagnostic, not exact invariant subspaces',
            'incoherent control artificially removes cross terms but keeps identical amplitudes']}
if __name__=='__main__':
 d=analyze();p=Path(__file__).resolve().parents[1]/'data/bit23_mode_interference.json';p.write_text(json.dumps(d,indent=2)+'\n')
 for k in ['full_velocity_2_8','no_cross_terms_velocity_2_8','family_center_velocity_2_8','family_average_power_fractions','reconstruction_center_error_max','maximum_eigenvector_condition_number']:
  print(k,d[k])
