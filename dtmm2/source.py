#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:44:06 2023

@author: andrej
"""
from dtmm2.wave import eigenmask
from dtmm2.field import field2modes
from dtmm2.conf import CDTYPE
from dtmm2.linalg import inv, dotmf, dotmm
from dtmm2.tmm import field_mat
from dtmm2.matrix import field_refraction_mat, field_cover_mat, apply_field_matrix, apply_mask
from dtmm2.fft import fft2, ifft2
from dtmm2.data import uniaxial_order
import numpy as np   

def jones_field(wave, jones = (1,0), intensity = 1.):
    """Builds a jones vector field"""
    jones = np.asarray(jones)
    extra = wave.shape[0:-2]
    height, width = wave.shape[-2:]
    
    out = np.empty(shape = extra + (2,) + (height, width), dtype = CDTYPE)
      
    out[...,0,:,:] = wave * jones[...,0] * intensity**0.5
    out[...,1,:,:] = wave * jones[...,1] * intensity**0.5
    return out 

def scalar_field2(jfield, direction = 1, epsilon = None, angles = None):
    fmat0 = field_mat() #vacuum fmat
    fmat0i = inv(fmat0)
    fmat = field_mat(epsilon, angles)    

    jfield = np.asarray(jfield)
    extra = jfield.shape[0:-3]
    n, height, width = jfield.shape[-3:]
        
    out = np.empty(shape = extra + (4,height, width), dtype = CDTYPE)
    
    if direction in (+1, "forward"):
        out[...,0,:,:] = jfield[...,0,:,:]
        out[...,1,:,:] = 0
        out[...,2,:,:] = jfield[...,1,:,:]
        out[...,3,:,:] = 0
    elif direction in (-1, "backward"):
        out[...,0,:,:] = 0
        out[...,1,:,:] = jfield[...,0,:,:]
        out[...,2,:,:] = 0
        out[...,3,:,:] = jfield[...,1,:,:]  
    mat = dotmm(fmat, fmat0i)
    
    dotmf(mat, out, out)
    return out


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
    
    ffield = fft2(field, out = out)
    ffield = dotmf(refmat,ffield, out = ffield)
    field = ifft2(ffield, out = ffield)
    return field

def convert_field(field, conversion = "j+>m", k = 1, epsilon = None, angles = None, aspect = 1, mask = None, out = None):
    conversion = str(conversion)
    infmt,outfmt = conversion.split(">")
    
    if infmt == "j+":
        field = jp2j(field)
    elif infmt == "j-":
        field = jm2j(field)
    if outfmt == "m":
        height, width = field.shape[-2:]
        refmat = field_refraction_mat(shape = (height, width), k = k, epsilon = epsilon, angles = angles, mask = mask)
        refmat = apply_mask(refmat, mask)
        
        
def _empty_j4_from_j2(field):
    field = np.asarray(field)
    extra = field.shape[0:-3]
    n, height, width = field.shape[-3:]
    out = np.empty(shape = extra + (4,height, width), dtype = CDTYPE)  
    return out, field  
            
def jp2j(field):
    out, field = _empty_j4_from_j2(field)  
    out[...,0,:,:] = field[...,0,:,:]
    out[...,1,:,:] = 0
    out[...,2,:,:] = field[...,1,:,:]
    out[...,3,:,:] = 0
    return out
    
def jm2j(field):
    out, field = _empty_j4_from_j2(field)  
    out[...,0,:,:] = 0
    out[...,1,:,:] = field[...,0,:,:]
    out[...,2,:,:] = 0
    out[...,3,:,:] = field[...,1,:,:]
    return out

def j2jp(field):
    return field[...,0::2,:,:]
            
def j2jm(field):
    return field[...,1::2,:,:]

