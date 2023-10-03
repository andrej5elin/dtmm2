#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:54:18 2023

@author: andrej
"""

from dtmm2.tmm import alphaffi, alphaf, layer_mat, f, system_mat, reflection_mat, reflect, EHz, phase_mat, poynting
from dtmm2.wave import betaphi, k0, mask2beta, mask2beta
#from dtmm.window import gaussian_beam

from dtmm2.linalg import multi_dot, dotmv, dotmf, dotmdm, dotmm
from dtmm2.rotation import rotation_matrix, rotate_vector

#from dtmm.field import field2intensity

import fft_transform
import numpy as np

import matplotlib.pyplot as plt

import dtmm2.tmm

def ne(wavelength):
    return 1.8447 + 24151/(wavelength**2)

def no(wavelength):
    return 1.6347 + 24151/(wavelength**2)

def theory(wavelength, intensity):
    new = ne(wavelength)
    ne2w = ne(wavelength/2)
    lamb = wavelength * 1e-9
    
    deltak = 4 * np.pi * (ne2w- new)/lamb
    c = 3e8
    eps0 = 8.85e-12
    
    return intensity**2 *32* np.pi**2 * DNLO**2 / lamb**2 / c/eps0/new**2/ne2w *np.sin(deltak * THICKNESS/2*1e-6)**2/(deltak**2)
    

#def ne(wavelength):
#    return 1.6447 + 24151/(wavelength**2)

#def no(wavelength):
#    return 1.4347 + 24151/(wavelength**2)

THICKNESS = 8.27
PITCH = 0.2756 
PIXELSIZE = 8.27*2

NLAYERS = THICKNESS * 1000 / PIXELSIZE
NLAYERS = int(NLAYERS) #if NLAYERS == int(NLAYERS) else 1/0

TWIST = THICKNESS/PITCH * np.pi  *2 



WAVELENGTH = 900


WAVELENGTHS = np.linspace(300*2,550.*2,100)

WAVELENGTHS = np.linspace(500*2,540.*2,100)
WAVELENGTHS = np.linspace(532*2,534.*2,100)
#WAVELENGTHS = np.linspace(340*2,360.*2,100)
#WAVELENGTHS = np.linspace(300*2,550.*2,200)
WAVELENGTHS = np.linspace(500*2,550.*2,33)
WAVELENGTHS = np.linspace(350,550.*2,100)
N0 = 1.5


#cmf = load_tcmf([WAVELENGTH])

K0 = k0(WAVELENGTH, PIXELSIZE)

import numpy as np


twist = np.linspace(0,TWIST, NLAYERS+1)[:NLAYERS]


intensity0 = []
intensity1 = []



beta = 0
phi = 0

z0 = 376.73*2

for WAVELENGTH in WAVELENGTHS:
    power = (10**6) * z0
    
    
    avec = np.asarray((1,0,1j,0)) * np.sqrt(power) /np.sqrt(2)
    
    #N0 = ne(WAVELENGTH)
    
    print(WAVELENGTH)
    
    K0 = k0(WAVELENGTH, PIXELSIZE)
    
    
    epsv = np.ones((NLAYERS,3))
    epsv[...,0] = no(WAVELENGTH)**2
    epsv[...,1] = no(WAVELENGTH)**2
    epsv[...,2] = ne(WAVELENGTH)**2
    epsa =  np.zeros((NLAYERS,3))
    epsa[...,1] = np.pi/2
    epsa[...,2] = twist
    
    
    epsv0 = np.ones((NLAYERS,3))
    epsv0[...,0] = no(WAVELENGTH)**2
    epsv0[...,1] = no(WAVELENGTH)**2
    epsv0[...,2] = no(WAVELENGTH)**2
    
    epsa0 =  np.zeros((NLAYERS,3))
    epsa0[...,1] = np.pi/2
    epsa0[...,2] = twist
    
    
    
    kds = np.array((K0,)* NLAYERS)
        
    print('building layer matrices')
    
    #for i in range(NLAYERS):
    #    matrices.append(layer_mat(kds[i], epsv[i],epsa[i], beta,phi))
    #    matrices2.append(layer_mat(kds2[i], epsv2[i],epsa[i], beta,phi))
    
    alpha, f1, f1i = alphaffi(beta,phi,epsv,epsa)
    phase1 = phase_mat(alpha,-kds)
    
    matrices1 = dotmdm(f1,phase1,f1i)
    

    alpha0, f0, f0i = alphaffi(beta,phi,epsv0,epsa0)
    phase0 = phase_mat(alpha,-kds)
    phase0r = phase_mat(alpha,kds)

    phase0h = phase_mat(alpha,-kds/2)
    phase0hr = phase_mat(alpha,kds/2)  
    
    cmat = dotmdm(f0,phase0hr,f0i)
    
    
    
    matrices1e = dotmm(matrices1, cmat)
    matrices1e = dotmm(cmat, matrices1e)
    
    
    
    #smat = system_mat(matrices1, f0,f0)
    #rmat = reflection_mat(smat)
    
    rr = dotmm(matrices1e,f0i)
    rr = dotmm(f0,rr)
    
    
    smate = system_mat(matrices1e, f0,f0)
    rmate = reflection_mat(smate)
    
    ampl = np.ones(shape = cmat.shape[:-1], dtype = cmat.dtype)

    # ampl[...,0] = 1/rmate[...,0,0].real
    # ampl[...,1] = rmate[...,1,1].real
    # ampl[...,2] = 1/rmate[...,2,2].real
    # ampl[...,3] = rmate[...,3,3].real        
        
    #ampl[...,0] = rr[...,0,0].real
    #ampl[...,1] = rr[...,1,1].real
    #ampl[...,2] = rr[...,2,2].real
    #ampl[...,3] = rr[...,3,3].real    
    
    tmat = dotmdm(f0,ampl,f0i)
    
    scatt = np.zeros_like(tmat)
    
    #rmate = rr
    
    scatt[...,0,0] = rmate[...,0,0].imag#/rmate[...,0,0].real
    scatt[...,1,1] = rmate[...,1,1].imag
    scatt[...,2,2] = rmate[...,2,2].imag#/rmate[...,2,2].real
    scatt[...,3,3] = rmate[...,3,3].imag   
    
    for i in (1,2,3):
        scatt[...,i,0] = rmate[...,i,0]#/rmate[...,0,0].real

    for i in (0,1,3):
        scatt[...,i,2] = rmate[...,i,2]#/rmate[...,2,2].real

    for i in (0,2,3):
        scatt[...,i,1] = rmate[...,i,1] 

    for i in (0,1,2):
        scatt[...,i,3] = rmate[...,i,3]   




        
    smat = dotmm(scatt,f0i)
    smat = dotmm(f0,smat)
    
    matrices0= dotmdm(f0,phase0,f0i)
    
    
    matrices0_half = dotmdm(f0,phase0h,f0i)
    
    # matrices0 = []
    
    # for i,m in enumerate(matrices0_half):
    #     matrices0.append(m)
    #     matrices0.append(tmat[i])
    #     matrices0.append(m)
        
        
        
    #matrices = layer_mat(-kds, epsv,epsa, beta,phi)
    #matrices2 = layer_mat(-kds2, epsv2,epsa, beta,phi)

    #matrices_half = layer_mat(kds/2, epsv,epsa, beta,phi)
    #matrices2_half  = layer_mat(-kds2/2, epsv2,epsa, beta,phi)
        
    print('building system matrix')
    
    stack_matrix1 = multi_dot(matrices1,reverse = True)
    stack_matrix0 = multi_dot(matrices0,reverse = True)
    
    #f0 = f(beta, phi, epsv0, epsa0)
    #f02 = f(beta, phi, epsv0, epsa0)

    fin = dtmm2.tmm.normalize_f(f(beta, phi, epsv0[0], epsa0[0]))

    

    
    

    smat1 = system_mat(stack_matrix1, fin,fin)
    smat0 = system_mat(stack_matrix0, fin,fin)
    
    print('reflection matrices')
    
    rmat1 = reflection_mat(smat1)
    rmat0 = reflection_mat(smat0)
    
    print('reflecting')
    
    field_in = dtmm2.tmm.avec2fvec(avec,fin)
    
    field_out_true = reflect(field_in, rmat1, fmatin = fin, fmatout = fin)
    

    field_in = dtmm2.tmm.avec2fvec(avec,fin)
    
    print(dtmm2.tmm.poynting(field_in)/z0)

    field_out = reflect(field_in, rmat0, fmatin = fin, fmatout = fin)
    
    
    
    print(dtmm2.tmm.poynting(field_out)/z0)


    field = field_out.copy()
    pert = field.copy()
    pert[...] = 0
    
    
    
    
    gain = []
    print('Propagating')
    
    field2 = None
    
    
    # 1/0
    
    
    
    # for i in range(NLAYERS):
    #     ref = dotmv(matrices1[i], field)
    #     field = dotmv(matrices0[i], field)
    #     pert = dotmv(matrices0[i], pert)
        
    #     pert += (ref - field)
    
    i0 = np.abs(poynting(field))
    

    
    for i in range(NLAYERS):
        
        #field_ref = dotmv(matrices1[i], field)
        
        
        field = dotmv(matrices0_half[i], field)
        
        pert = dotmv(matrices0_half[i], pert)
        

        
        #pert = dotmv(tmat[i], pert)
        
        #pert += dotmv(smat[i], field)
    
        ref = dotmv(matrices1e[i],field)
        
        #field = dotmv(tmat[i], field)
        
        pert += (ref - field)
        
    
    
        field = dotmv(matrices0_half[i], field)
        
        pert = dotmv(matrices0_half[i], pert)
        
        fact = i0/(poynting(field+pert))
        
        pert = pert*fact**0.5
        field = field*fact**0.5
        
        print(fact)
        
        
        


    # print('Final step')
      
    
    field_in_pert = np.zeros_like(field_out)
    
    field_out = reflect(field_in_pert, rmat0, fmatin = fin, fmatout = fin, gvec = pert)
    

    # print('done')
    
    intensity0.append(dtmm2.tmm.poynting(field_out))
    intensity1.append(dtmm2.tmm.poynting(field_out_true))
    
    




