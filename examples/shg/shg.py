#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:54:18 2023

@author: andrej
"""

from dtmm2.tmm import alphaffi, alphaf, layer_mat, f, system_mat, reflection_mat, reflect, EHz, phase_mat
from dtmm2.wave import betaphi, k0, mask2beta, mask2beta
#from dtmm.window import gaussian_beam

from dtmm2.linalg import multi_dot, dotmv, dotmf, dotmdm
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
PIXELSIZE = 8.27

NLAYERS = THICKNESS * 1000 / PIXELSIZE
NLAYERS = int(NLAYERS) #if NLAYERS == int(NLAYERS) else 1/0

TWIST = THICKNESS/PITCH * np.pi  *2 



WAVELENGTH = 900
WAVELENGTH2 = WAVELENGTH/2.

WAVELENGTHS = np.linspace(300*2,550.*2,100)

WAVELENGTHS = np.linspace(500*2,540.*2,100)
WAVELENGTHS = np.linspace(532*2,534.*2,100)
#WAVELENGTHS = np.linspace(340*2,360.*2,100)
#WAVELENGTHS = np.linspace(300*2,550.*2,200)
WAVELENGTHS = np.linspace(500*2,550.*2,33)
WAVELENGTHS = np.linspace(350,550.*2,100)
N0 = 1.5



DNLO = 5.6e-12


#cmf = load_tcmf([WAVELENGTH])

K0 = k0(WAVELENGTH, PIXELSIZE)
K02 = k0(WAVELENGTH2, PIXELSIZE)

import numpy as np


twist = np.linspace(0,TWIST, NLAYERS+1)[:NLAYERS]


intensity = []
intensity2 = []
intensity2r = []



beta = 0
phi = 0

z0 = 376.73*2

for WAVELENGTH in WAVELENGTHS:
    power = (10**6) * z0
    
    
    avec = np.asarray((1,0,1j,0)) * np.sqrt(power) /np.sqrt(2)
    
    #N0 = ne(WAVELENGTH)
    
    print(WAVELENGTH)
    
    WAVELENGTH2 = WAVELENGTH/2
    K0 = k0(WAVELENGTH, PIXELSIZE)
    K02 = k0(WAVELENGTH2, PIXELSIZE)
    
    epsv0 = np.ones((3,))
    epsa0 = np.zeros((3,))
    epsv0[...] = no(WAVELENGTH)**2
    epsv02 = np.ones((3,))
    epsv02[...] = no(WAVELENGTH2)**2
    
    epsv = np.ones((NLAYERS,3))
    epsv[...,0] = no(WAVELENGTH)**2
    epsv[...,1] = no(WAVELENGTH)**2
    epsv[...,2] = ne(WAVELENGTH)**2
    epsa =  np.zeros((NLAYERS,3))
    epsa[...,1] = np.pi/2
    epsa[...,2] = twist
    
    
    epsv2 = np.ones((NLAYERS,3))
    epsv2[...,0] = no(WAVELENGTH2)**2
    epsv2[...,1] = no(WAVELENGTH2)**2
    epsv2[...,2] = ne(WAVELENGTH2)**2
    
    
    kds = np.array((K0,)* NLAYERS)
    kds2 = np.array((K02,)* NLAYERS)
    
    matrices = []
    matrices2 = []
    
    print('building layer matrices')
    
    #for i in range(NLAYERS):
    #    matrices.append(layer_mat(kds[i], epsv[i],epsa[i], beta,phi))
    #    matrices2.append(layer_mat(kds2[i], epsv2[i],epsa[i], beta,phi))
    
    alpha, f1, f1i = alphaffi(beta,phi,epsv,epsa)
    phase = phase_mat(alpha,-kds)
    phase_half = phase_mat(alpha,kds/2)
    
    matrices = dotmdm(f1,phase,f1i)
    matrices_half = dotmdm(f1,phase_half,f1i)

    alpha2, f2, f2i = alphaffi(beta,phi,epsv2,epsa)
    phase2 = phase_mat(alpha,-kds)
    phase2_half = phase_mat(alpha,kds/2)
    
    matrices2 = dotmdm(f2,phase2,f2i)
    
    matrices2_half = dotmdm(f2,phase2_half,f2i)

        
    #matrices = layer_mat(-kds, epsv,epsa, beta,phi)
    #matrices2 = layer_mat(-kds2, epsv2,epsa, beta,phi)

    #matrices_half = layer_mat(kds/2, epsv,epsa, beta,phi)
    #matrices2_half  = layer_mat(-kds2/2, epsv2,epsa, beta,phi)
        
    print('building system matrix')
    
    stack_matrix = multi_dot(matrices,reverse = False)
    stack_matrix2 = multi_dot(matrices2,reverse = False)
    
    #f0 = f(beta, phi, epsv0, epsa0)
    #f02 = f(beta, phi, epsv0, epsa0)

    f0 = dtmm2.tmm.normalize_f(f(beta, phi, epsv0, epsa0))
    f02 = dtmm2.tmm.normalize_f(f(beta, phi, epsv02, epsa0))

    
    smat = system_mat(stack_matrix, f0,f0)
    smat2 = system_mat(stack_matrix2, f02,f02)
    
    print('reflection matrices')
    
    rmat = reflection_mat(smat)
    rmat2 = reflection_mat(smat2)
    
    print('reflecting')
    

    field_in = dtmm2.tmm.avec2fvec(avec,f0)
    
    print(dtmm2.tmm.poynting(field_in)/z0)

    field_out = reflect(field_in, rmat, fmatin = f0, fmatout = f0)
    print(dtmm2.tmm.poynting(field_out)/z0)


    field = field_out.copy()
    
    
    
    gain = []
    print('Propagating')
    
    field2 = None
    
    for i in reversed(range(NLAYERS)):
        
        
        Ex = field[0]
        Ey = field[2]
        
        EH = EHz(field,beta,phi, epsv[i], epsa[i])
   
        
        Ez = EH[0]
        
        E = np.empty(Ez.shape + (3,), dtype = Ez.dtype)
        E[...,0] = Ex
        E[...,1] = Ey
        E[...,2] = Ez
    
        r = rotation_matrix(epsa[i])
        
        E = dotmv(r.T, E)
    
    
        
        lmat_ref = layer_mat(kds[i], epsv[i],epsa[i])
        
        epsvnlo = epsv[i] + 0j
        
        epsvnlo[...,2] += 2*DNLO * E[...,2]
        

        
        lmat = layer_mat(kds[i], epsvnlo  ,epsa[i])
        
        
        field_ref = dotmv(lmat_ref, field)
        field_nlo = dotmv(lmat, field)
        
        dif = field_nlo - field_ref 
        
        dif = dotmv(matrices_half[i],dif)
        dif = dotmv(matrices2_half[i],dif)
        
        matrix = matrices[i]
        matrix2 = matrices2[i]
        
        field = dotmv(matrix, field)
        if field2 is not None:
            field2 = dotmv(matrix2,field2)
            field2 += dif
        else:
            field2 = dif
            

    print('Final step')
    field2_in = np.zeros_like(field2)
      
    
    field2_out = reflect(field2_in, rmat2, fmatin = f02, fmatout = f02, gvec = field2)
    

    print('done')
    
    intensity.append(dtmm2.tmm.poynting(field_out))
    intensity2.append(dtmm2.tmm.poynting(field2_out))
    intensity2r.append(dtmm2.tmm.poynting(field2_in))
    
    




