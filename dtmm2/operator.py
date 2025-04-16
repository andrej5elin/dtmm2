#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:00:33 2023

@author: andrej
"""
from dtmm2.fft import fft2, ifft2
from dtmm2.linalg import dotmf, inv, dotmm, dotmdm
from dtmm2.wave import betaphi, k0
from dtmm2.tmm import field_mat, _default_epsv_epsa, field_eig, phase_mat
from dtmm2.data import refind2eps
from dtmm2.conf import FDTYPE
import numpy as np

def focal_pixelsize(shape, k,  aspect = 1., focal = 50):
    """Computes pixel size and aspect ratio in the focal plane of the lens from
    the normalized wavenumber, shape  and object side pixel aspect ratio."""
    p0 = 2*np.pi / k/shape[0] * focal *1000*1000 
    p1 = 2*np.pi / k/shape[1] * focal * aspect *1000*1000
    return p0, p1/p0
  
def lens_transform(field, lens_refmat = None, field_refmat= None, origin = "center", out = None):
    if origin not in ("center", "corner"):
        raise ValueError("Origin must be either 'center' or 'corner'")
    if field_refmat is not None:
        field_refmat = inv(field_refmat)
        
    norm = 1/(field.shape[-2] * field.shape[-1]) ** 0.5
    if field_refmat is not None:
        field = fft2(field)
        field = dotmf(field_refmat, field, out = field)
        field = ifft2(field, out = out)
        
    if origin == "center":
        field = np.fft.ifftshift(field, axes = (-2,-1))
        
    if lens_refmat is not None:  
        field = dotmf(lens_refmat, field)
    field = fft2(field, out = out)    
    if origin == "center":
        field = np.fft.fftshift(field, axes = (-2,-1))

    return np.multiply(field, norm ,out = out)
