#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:44:06 2023

@author: andrej
"""
from dtmm.wave import eigenmask
from dtmm2.field import field2modes
from dtmm2.conf import CDTYPE
from dtmm2.linalg import inv, dotmf, dotmm
from dtmm2.tmm import field_mat
from dtmm2.matrix import field_refraction_mat, field_cover_mat, apply_field_matrix, apply_mask
from dtmm2.fft import fft2, ifft2
import numpy as np    

def scalar_field(wave, jones = (1,0), intensity = 1., epsilon = None, angles = None, direction = +1):
    fmat0 = field_mat() #vacuum fmat
    fmat0i = inv(fmat0)
    fmat = field_mat(epsilon, angles)
    
    mat = dotmm(fmat, fmat0i)
         
    jones = np.asarray(jones)
    extra = wave.shape[0:-2]
    height, width = wave.shape[-2:]
    
    out = np.empty(shape = extra + (4,) + (height, width), dtype = CDTYPE)
    
    if direction in (+1, "forward"):
    
        out[...,0,:,:] = wave * jones[...,0] * intensity**0.5
        out[...,1,:,:] = wave * jones[...,0] * intensity**0.5
        out[...,2,:,:] = wave * jones[...,1] * intensity**0.5
        out[...,3,:,:] = -wave * jones[...,1] * intensity**0.5
    elif direction in (-1, "backward"):
        out[...,0,:,:] = wave * jones[...,0] * intensity**0.5
        out[...,1,:,:] = -wave * jones[...,0] * intensity**0.5
        out[...,2,:,:] = -wave * jones[...,1] * intensity**0.5
        out[...,3,:,:] = wave * jones[...,1] * intensity**0.5        
        
    dotmf(mat, out, out)
    
    return out

def vector_field(field, k = 1, epsilon = None, angles = None, aspect = 1, mask = None, out = None):
    """Takas a scalar field and converts it to  a proper vectorial field.
    
    This operation is essentially an idealized beam compression operation which 
    takes a paraxial infinite width beam and uses idealized lens to compress it to 
    a finite size beam. 
    """
    height, width = field.shape[-2:]
    refmat = field_refraction_mat(shape = (height, width), k = k, epsilon = epsilon, angles = angles, mask = mask)
    refmat = apply_mask(refmat, mask)
    
    ffield = ifft2(field, out = out)
    ffield = dotmf(refmat,ffield, out = ffield)
    field = fft2(ffield, out = ffield)
    return field


    
