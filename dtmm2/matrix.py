#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:38:23 2023

@author: andrej
"""

from dtmm2.fft import fft2, ifft2
from dtmm2.linalg import dotmf, inv, dotmm, dotmdm, dotmv
from dtmm2.wave import betaphi, k0, eigenmask
from dtmm2.tmm import field_mat, _default_epsv_epsa, field_eig, phase_mat, field_eigi, _default_beta_phi, transfer_mat, system_mat, reflection_mat, hzez_mat
from dtmm2.data import refind2eps
from dtmm2.conf import FDTYPE
import numpy as np

from dtmm2.wave import mask2beta, mask2phi, eigenmask, k0

def _inspect_mask_out(mask, out):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if out is None:
            out = [None] * len(mask)
        else:
            out = [_inspect_mask_out(m,o)[1] for m,o in zip(mask, out)]
            if len(out) != len(mask):
                raise ValueError("Invalid out length")
    elif mask.ndim != 2:
        raise ValueError("Invalid dimension of the mode mask")
    else:
        if out is not None:
            if not isinstance(mask, np.ndarray):
                raise ValueError("Invalid output")
    return mask, out 
            

def _is_tuple(obj):
    return isinstance(obj, tuple)

def _default_beam_mask(mask, k):
    if mask is None:
        mask = eigenmask((128,128), k, aspect = 1.)
    return mask

def fmat2mmat(fmat, mask, out = None):
    """Converts field matrix to mode/masked matrix
    
    >>> mask = eigenmask((256,256),[1,2])
    >>> mmat_ref = mode_cover_mat(mask, [1,2])
    >>> fmat = field_cover_mat((256,256),[1,2], mask = mask)
    >>> mmat = fmat2mmat(fmat, mask) 
    >>> np.allclose(mmat[0], mmat_ref[0]) and np.allclose(mmat[1], mmat_ref[1])
    True
    """
    mask, out = _inspect_mask_out(mask, out)
    if mask.ndim == 3:
        return tuple((fmat2mmat(f,m,o) for f,m,o in zip(fmat, mask, out)))
    if out is None:
        return fmat[...,mask,:,:]
    else:
        out[...] = fmat[...,mask,:,:]
        return out

def mmat2fmat(bmat, mask, out = None):
    """Converts mode/masked matrix to field matrix
    
    >>> mask = eigenmask((256,256),[1,2])
    >>> mmat = mode_cover_mat(mask, [1,2])
    >>> fmat_ref = field_cover_mat((256,256),[1,2], mask = mask)
    >>> fmat = mmat2fmat(mmat, mask) 
    >>> np.allclose(fmat, fmat_ref)
    True
    """
    if mask.ndim == 3:
        shape = bmat[0].shape[:-3] + mask.shape + bmat[0].shape[-2:]
        dtype = bmat[0].dtype
    else:
        shape = bmat.shape[:-3] + mask.shape + bmat.shape[-2:]
        dtype = bmat.dtype
    if out is not None:
        out[...] = 0
    else:
        out = np.zeros(shape = shape, dtype = dtype)
    if mask.ndim == 3:
        for i,m in enumerate(bmat):
            out[...,i,mask[i],:,:] = m
    else:
        out[...,mask,:,:] = bmat
    return out
        
def cover_mat(k, d_cover = 0, n_cover = 1.5, n = 1, beta = None, phi = None, out = None):
    k = np.asarray(k, FDTYPE)
    beta, phi = _default_beta_phi(beta, phi)
    alpha, fmat = field_eig((n**2,n**2,n**2), beta = beta, phi = phi)
    alpha0, fmat0 = field_eig((n_cover**2,n_cover**2,n_cover**2), beta = beta, phi = phi)
    fmat0i = inv(fmat0, out = fmat) #we dont need fmat, so we can reuse it
    
    alphac = alpha0 - alpha * n / n_cover
    #alphac = alphac - alphac[...,0,0,:][...,None,None,:]  
    
    kd = k[...,None,None] * d_cover
    pmat = phase_mat(alphac, kd)
    
    out = dotmdm(fmat0,pmat,fmat0i,out = out) 
    return out

def field_cover_mat(shape, k, d_cover = 0., n_cover = 1.5, n = 1,  aspect = 1, mask = None, out = None):    
    k = np.asarray(k, FDTYPE)
    beta, phi = betaphi(shape,k, aspect = aspect)
    out = cover_mat(k, d_cover, n_cover, n, beta, phi, out = out)
    return apply_mask(out, mask, copy = False)

def mode_cover_mat(mask, k, d_cover = 0., n_cover = 1.5, n = 1,  aspect = 1, out = None):
    mask, out = _inspect_mask_out(mask, out)
    beta = mask2beta(mask,k, aspect)
    phi = mask2phi(mask,k,aspect)
    if _is_tuple(beta):
        return tuple((cover_mat(k, d_cover, n_cover, n, b, p, out = o) for b,p,o in zip(beta, phi, out)))
    else:
        return cover_mat(k, d_cover, n_cover, n, beta, phi, out = out) 
        
def refraction_mat(epsilon = None, angles = None, beta = None, phi = None, out = None):
    epsilon, angles = _default_epsv_epsa(epsilon, angles)   
    beta, phi = _default_beta_phi(beta, phi)
    f = field_mat(epsilon, angles, beta = beta, phi = phi)
    beta0 = np.zeros_like(beta)
    f0 = field_mat(epsilon, beta = beta0, phi = phi)
    f0i = inv(f0,f0)
    out = f0 if out is None else out
    out = dotmm(f, f0i, out = out)  
    return out

def field_refraction_mat(shape, k, epsilon = None, angles = None, aspect = 1., mask = None, out = None):
    beta, phi = betaphi(shape,k, aspect = aspect)
    out = refraction_mat(epsilon, angles, beta, phi, out = out)
    return apply_mask(out, mask, copy = False)
    
def mode_refraction_mat(mask, k, epsilon = None, angles = None, aspect = 1., out = None):
    mask, out = _inspect_mask_out(mask, out)
    epsilon, angles = _default_epsv_epsa(epsilon, angles)    
    beta = mask2beta(mask,k, aspect)
    phi = mask2phi(mask,k,aspect)
    if _is_tuple(beta):
        return tuple((refraction_mat(epsilon, angles, b, p, out = o) for b,p,o in zip(beta, phi, out)))
    else:
        return refraction_mat(epsilon, angles, beta, phi, out = out)

def diffraction_mat(k,d,epsilon = None, angles = None, beta = None, phi = None, out = None):
    k = np.asarray(k)
    if k.ndim == 1:
        kd = k[:,None,None]*d
    else:
        kd = k * d
    out = transfer_mat(kd, epsilon, angles, beta, phi, out = out)
    return out

def field_diffraction_mat(shape, k, d = 1, epsilon = None, angles = None, aspect = 1., mask = None, out = None ):
    beta, phi = betaphi(shape,k, aspect = aspect)
    out = diffraction_mat(k,d,epsilon, angles, beta, phi, out = out)
    return apply_mask(out, mask, copy = False)

def mode_diffraction_mat(mask, k, d = 1, epsilon = None, angles = None, aspect = 1., out = None):
    mask, out = _inspect_mask_out(mask, out)
    epsilon, angles = _default_epsv_epsa(epsilon, angles)    
    beta = mask2beta(mask,k, aspect)
    phi = mask2phi(mask,k,aspect)
    if _is_tuple(beta):
        return tuple((diffraction_mat(ki,d, epsilon, angles, b,p, out = o) for ki,b,p,o in zip(k,beta,phi,out)))
    else: 
        return diffraction_mat(k,d, epsilon, angles, beta,phi,out = out)
    
def mode_hzez_mat(mask, k, epsilon = None, angles = None, aspect = 1., out = None):
    mask, out = _inspect_mask_out(mask, out)
    epsilon, angles = _default_epsv_epsa(epsilon, angles)        
    beta = mask2beta(mask,k, aspect)
    phi = mask2phi(mask,k,aspect)
    if _is_tuple(beta):
        return tuple((hzez_mat(epsilon,angles,b,p, out = o) for b,p,o in zip(k,beta,phi,out)))
    else:
        return hzez_mat(epsilon,angles, beta, phi)
    
def transmittance_mat(epsilon1 = None, angles1 = None, epsilon2 = None, angles2 = None, beta = None, phi = None, mode = +1, out = None):
    epsilon1, angles1 = _default_epsv_epsa(epsilon1, angles1) 
    epsilon2, angles2 = _default_epsv_epsa(epsilon2, angles2) 
    beta, phi = _default_beta_phi(beta, phi)  
    fmat1 = field_mat(epsilon1,angles1, beta, phi)
    fmat2 = field_mat(epsilon2,angles2, beta, phi)
    smat = system_mat(fmatin = fmat1, fmatout = fmat2)
    rmat = reflection_mat(smat)
    if mode == +1:
        fmatout = fmat2
        fmatini = inv(fmat1, out = fmat1)
        rmat[...,1::2] = 0
        rmat[...,1::2,:] = 0
        return dotmm(fmatout,dotmm(rmat,fmatini), out = out)
    else:
        fmatin = fmat1
        fmatouti = inv(fmat2, out = fmat2)
        rmat[...,0::2] = 0
        rmat[...,0::2,:] = 0   
        return dotmm(fmatin,dotmm(rmat,fmatouti), out = out)
        
def field_transmittance_mat(shape, k, epsilon1 = None, angles1 = None, epsilon2 = None, angles2 = None, mode = +1, aspect = 1., mask = None, out = None):
    beta, phi = betaphi(shape,k, aspect = aspect)
    mat = transmittance_mat(epsilon1, angles1, epsilon2, angles2, beta,phi, mode, out = out)
    return apply_mask(mat, mask ,copy = False)

def mode_transmittance_mat(mask, k, epsilon1 = None, angles1 = None, epsilon2 = None, angles2 = None, mode = +1, aspect = 1., out = None):
    mask, out = _inspect_mask_out(mask, out)
    epsilon1, angles1 = _default_epsv_epsa(epsilon1, angles1)   
    epsilon2, angles2 = _default_epsv_epsa(epsilon1, angles1)    
    beta = mask2beta(mask,k, aspect)
    phi = mask2phi(mask,k,aspect) 
    if _is_tuple(beta):
        return tuple((transmittance_mat(epsilon1, angles1, epsilon2, angles2, b,p, mode = mode, out = o) for b,p,o in zip(beta,phi,out)))
    else:
        return transmittance_mat(epsilon1, angles1, epsilon2, angles2, beta, phi, mode = mode,  out = out)

def apply_mask(mat, mask = None, copy = True):
    """Applies mode mask to a given field matrix"""
    if _is_tuple(mat):
        if mask is None:
            mask = [None] * len(mat)
        return tuple((apply_mask(mt,msk,copy) for mt,msk in zip(mat,mask)))
    
    mat = np.asarray(mat)
    if copy == True:
        mat = mat.copy()
    if mask is not None:
        mask = np.asarray(mask)
        mat[...,np.logical_not(mask),:,:] = 0
    return mat
    
def apply_field_matrix(field, mat, out = None):
    ffield = fft2(field, out = out)
    ffield = dotmf(mat,ffield, out = ffield)
    field = ifft2(ffield, out = ffield)
    return field   

def apply_mode_matrix(mode, mat, out = None):
    if isinstance(mode, tuple):
        if out is None:
            out = [None] * len(mode)
        return tuple((apply_mode_matrix(m,mt,o) for m, mt, o in zip(mode,mat,out)))
    return dotmv(mat, mode, out = out)
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
