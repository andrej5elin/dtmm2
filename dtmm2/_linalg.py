"""
Numba optimized linear algebra functions for 4x4, 3x3 and 2x2 complex or real matrices.

These are non-vectorized low level implementsations. The vectorized functions are
in the linalg module.
"""

from __future__ import absolute_import, print_function, division
from dtmm2.conf import NCDTYPE, NFDTYPE, NUMBA_TARGET,NUMBA_PARALLEL, NUMBA_CACHE, NUMBA_FASTMATH, CDTYPE, FDTYPE
from numba import njit, prange, guvectorize, boolean
import numpy as np
from dtmm2.conf import deprecation

if not NUMBA_PARALLEL:
    prange = range
  
#delaration for dot function
DOTMV_DECL = [(NFDTYPE[:,:],NFDTYPE[:],NFDTYPE[:]),(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])]
DOTMM_DECL = [(NFDTYPE[:,:],NFDTYPE[:,:],NFDTYPE[:,:]),(NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:])]
DOTMD_DECL = [(NFDTYPE[:,:],NFDTYPE[:],NFDTYPE[:,:]),(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])]
DOTVV_DECL = [(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:]),(NCDTYPE[:],NCDTYPE[:],NCDTYPE[:])]      
DOTMDM_DECL = [(NFDTYPE[:,:],NFDTYPE[:],NFDTYPE[:,:],NFDTYPE[:,:]),(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])]
#declarations for inverse functions
INV_DECL = [(NCDTYPE[:, :], NCDTYPE[:, :]),(NFDTYPE[:, :], NFDTYPE[:, :])]
EIG_DECL = [(NFDTYPE[:,:],boolean[:],NFDTYPE[:],NFDTYPE[:,:]), (NCDTYPE[:,:],boolean[:],NCDTYPE[:], NCDTYPE[:,:])]         

CROSS_DECL = [(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:]),(NCDTYPE[:],NCDTYPE[:],NCDTYPE[:])]


@njit([NFDTYPE(NFDTYPE[:]),NFDTYPE(NCDTYPE[:])], cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)        
def _vecabs2(v):
    """Computes vec.(vec.conj)"""
    out = 0.
    for i in range(len(v)):
        out = out + v[i].real**2 + v[i].imag**2
    return out

@njit([NCDTYPE(NCDTYPE[:]),NFDTYPE(NFDTYPE[:])], cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)        
def _vnorm2(v):
    """Computes vec.vec"""
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]

@njit(CROSS_DECL, cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)        
def _cross(v1,v2,v3):
    """performs vector cross product"""
    v30 = v1[1] * v2[2] - v1[2] * v2[1]
    v31 = v1[2] * v2[0] - v1[0] * v2[2]
    v32 = v1[0] * v2[1] - v1[1] * v2[0]
    v3[0] = v30
    v3[1] = v31
    v3[2] = v32
    return v3

_eigvec0_decl = [
                 (NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:]),
                 (NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:])
                 ]

@njit(_eigvec0_decl, cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)            
def _eigvec0(a00,a11,a22,a01,a02,a12,eval0,vec0, tmp1, tmp2, tmp3):
    """The first eigenvector"""
    
    row0 = vec0
    row0[0] = a00 - eval0
    row0[1] = a01
    row0[2] = a02
   
    row1 = tmp1
    row1[0] = a01 
    row1[1] = a11 - eval0
    row1[2] = a12
    
    row2 = tmp2
    row2[0] = a02 
    row2[1] = a12
    row2[2] = a22 - eval0
    
    r0xr1 = _cross(row0, row1, tmp3 ) #tmp[0] is empty
    r1xr2 = _cross(row1, row2, tmp1 ) #row1 is in tmp1, but it can be deleted
    r0xr2 = _cross(row0, row2, tmp2 )
    
    d0 = _vecabs2(r0xr1)
    d1 = _vecabs2(r0xr2)
    d2 = _vecabs2(r1xr2)
    
    dmax = d0
    imax = 0
    
    if d1 > dmax:
        dmax = d1
        imax = 1
    if d2 > dmax:
        imax = 2
        
    if imax == 0:
        norm = 1/(_vnorm2(r0xr1)**0.5)
        vec0[:] = r0xr1[:] 
    elif imax == 1:
        norm = 1/(_vnorm2(r0xr2)**0.5)
        vec0[:] = r0xr2[:] 
    else:
        norm = 1/(_vnorm2(r1xr2)**0.5)
        vec0[:] = r1xr2[:] 
        
    vec0 *= norm

@njit([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:]),(NCDTYPE[:],NCDTYPE[:],NCDTYPE[:])], cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH) 
def _orthogonal_basis(w,u,v):
    """build an orthogonal basis based on normalized input vector w"""
    if abs(w[0]) > abs(w[1]):
        u[0] = (-w[2])
        u[1] = 0
        u[2] = w[0]
        scale = 1 / (_vecabs2(u)**0.5)
        u *= scale
    else:
        u[0] = 0
        u[1] = w[2]
        u[2] = (-w[1])
        scale = 1 / (_vecabs2(u)**0.5)
        u *= scale   
    _cross(w,u,v)

    
_eigvec1_decl = [
                 (NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE[:],NFDTYPE,NFDTYPE[:],NFDTYPE[:],NFDTYPE[:]),
                 (NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE[:],NCDTYPE,NCDTYPE[:],NCDTYPE[:],NCDTYPE[:])
                 ]   

@njit(_eigvec1_decl, cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)            
def _eigvec1(a00,a11,a22,a01,a02,a12,vec0,eval1, vec1, tmp1, tmp2):
    """The second eigenvector"""
    u = vec1
    v = tmp1
    AU = tmp2
    
    _orthogonal_basis(vec0, u, v)
    
    AU[0] = a00 * u[0] + a01 * u[1] + a02 * u[2]
    AU[1] = a01 * u[0] + a11 * u[1] + a12 * u[2]
    AU[2] = a02 * u[0] + a12 * u[1] + a22 * u[2]
    
    m00 = u[0] * AU[0] + u[1] * AU[1] + u[2] * AU[2] - eval1
    
    AV = tmp2
   
    AV[0] = a00 * v[0] + a01 * v[1] + a02 * v[2]
    AV[1] = a01 * v[0] + a11 * v[1] + a12 * v[2]
    AV[2] = a02 * v[0] + a12 * v[1] + a22 * v[2]    
    
    m01 = u[0] * AV[0] + u[1] * AV[1] + u[2] * AV[2]
    m11 = v[0] * AV[0] + v[1] * AV[1] + v[2] * AV[2] - eval1
    
    absm00 = abs(m00)
    absm01 = abs(m01)
    absm11 = abs(m11)
    
    if absm00 >= absm11:
        m = max(absm00,absm01)
        if m > 0:
            if absm00 >= absm01:
                m01 /= m00 
                m00 = 1/((1+m01**2)**0.5)
                m01 *= m00
            else:
                m00 /= m01
                m01 = 1/((1+m00**2)**0.5) 
                m00 *= m01
            for i in range(3):
                vec1[i] = m01* u[i] - m00 * v[i] #vec1 is u so we are doing vec1 = m01 * u
    else:
        m = max(absm11,absm01)
        if m > 0:
            if absm11 >= absm01:
                m01 /= m11
                m11= 1/((1+m01**2)**0.5)
                m01 *= m11
            else:
                m11 /= m01
                m01 = 1/((1+m11**2)**0.5) 
                m11 *= m01
                
            for i in range(3):
                vec1[i] = m11* u[i] - m01 * v[i]   
 
    
@njit([(NFDTYPE[:],NFDTYPE[:,:],NFDTYPE[:],NFDTYPE[:,:]),(NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def _sort_eigvec(eps,r, epsout, rout):
    """Eigen modes sorting based on eigenvalues. Finds extraordinary axis,
    performs cyclic permutation of axes to move the extraordinary axis to 3"""
    e0 = eps[0]#.real #real part of the refractive index
    e1 = eps[1]#.real
    e2 = eps[2]#.real
    
    m2 = np.abs(e1-e0)
    m1 = np.abs(e2-e0)
    m0 = np.abs(e2-e1)
    
    #assume extraordinary is 2    
    i,j,k = 0,1,2
    
    #if we need to move eigenavlue to axis 2... do a cyclical permutation, to preserve right-handed coordinate system
    if m2 > m1 or m2 > m0:
        if m1 > m0:
            #extraordinary is 0
            i,j,k = 1,2,0
        else:
            #extraordinary is 1
            i,j,k = 2,0,1
            
    #perform sorting

    eps0 = eps[i]
    eps1 = eps[j]
    eps2 = eps[k]
    
    r00 = r[i,0]
    r01 = r[i,1]
    r02 = r[i,2]
    
    r10 = r[j,0]
    r11 = r[j,1]
    r12 = r[j,2]
    
    r20 = r[k,0]
    r21 = r[k,1]
    r22 = r[k,2]
    
    rout[0] = (r00,r01,r02)
    rout[1] = (r10,r11,r12)
    rout[2] = (r20,r21,r22)
          
    epsout[0] = eps0
    epsout[1] = eps1
    epsout[2] = eps2
        

@njit(INV_DECL,  cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)   
def _inv2x2(src, dst):
    """Inverse of a 2x2 matrix
    """
    # Extract individual elements
    a = src[0, 0]
    b = src[0, 1]
    c = src[1, 0]
    d = src[1, 1]

    # Calculate determinant
    det = a * d - b * c
    if det == 0.:
        det = 0.
    else:
        det = 1. / det

    # Calcualte inverse
    dst[0, 0] =  d * det
    dst[0, 1] = -b * det
    dst[1, 0] = -c * det
    dst[1, 1] =  a * det


@njit(INV_DECL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)   
def _inv4x4(src,dst):
    """inverse 4x4 matrix, dst can be src for inplace transform"""    
    #calculate pairs for first 8 elements (cofactors)
    tmp0 = src[2,2] * src[3,3]
    tmp1 = src[3,2] * src[2,3]
    tmp2 = src[1,2] * src[3,3]
    tmp3 = src[3,2] * src[1,3]
    tmp4 = src[1,2] * src[2,3]
    tmp5 = src[2,2] * src[1,3]
    tmp6 = src[0,2] * src[3,3]
    tmp7 = src[3,2] * src[0,3]
    tmp8 = src[0,2] * src[2,3]
    tmp9 = src[2,2] * src[0,3]
    tmp10 = src[0,2] * src[1,3]
    tmp11 = src[1,2] * src[0,3]
    # calculate first 8 elements (cofactors)
    dst00 = tmp0*src[1,1] + tmp3*src[2,1] + tmp4*src[3,1] -tmp1*src[1,1] - tmp2*src[2,1] - tmp5*src[3,1]
    dst01 = tmp1*src[0,1] + tmp6*src[2,1] + tmp9*src[3,1] - tmp0*src[0,1] - tmp7*src[2,1] - tmp8*src[3,1]
    dst02 = tmp2*src[0,1] + tmp7*src[1,1] + tmp10*src[3,1] - tmp3*src[0,1] - tmp6*src[1,1] - tmp11*src[3,1]
    dst03 = tmp5*src[0,1] + tmp8*src[1,1] + tmp11*src[2,1] - tmp4*src[0,1] - tmp9*src[1,1] - tmp10*src[2,1]
    
    dst10 = tmp1*src[1,0] + tmp2*src[2,0] + tmp5*src[3,0] - tmp0*src[1,0] - tmp3*src[2,0] - tmp4*src[3,0]
    dst11 = tmp0*src[0,0] + tmp7*src[2,0] + tmp8*src[3,0] - tmp1*src[0,0] - tmp6*src[2,0] - tmp9*src[3,0]
    dst12 = tmp3*src[0,0] + tmp6*src[1,0] + tmp11*src[3,0] - tmp2*src[0,0] - tmp7*src[1,0] - tmp10*src[3,0]
    dst13 = tmp4*src[0,0] + tmp9*src[1,0] + tmp10*src[2,0] - tmp5*src[0,0] - tmp8*src[1,0] - tmp11*src[2,0]
    # calculate pairs for second 8 elements (cofactors) 
    tmp0 = src[2,0]*src[3,1]
    tmp1 = src[3,0]*src[2,1]
    tmp2 = src[1,0]*src[3,1]
    tmp3 = src[3,0]*src[1,1]
    tmp4 = src[1,0]*src[2,1]
    tmp5 = src[2,0]*src[1,1]
    tmp6 = src[0,0]*src[3,1]
    tmp7 = src[3,0]*src[0,1]
    tmp8 = src[0,0]*src[2,1]
    tmp9 = src[2,0]*src[0,1]
    tmp10 = src[0,0]*src[1,1]
    tmp11 = src[1,0]*src[0,1]

    dst20 = tmp0*src[1,3] + tmp3*src[2,3] + tmp4*src[3,3] - (tmp1*src[1,3] + tmp2*src[2,3] + tmp5*src[3,3])
    dst21 = tmp1*src[0,3] + tmp6*src[2,3] + tmp9*src[3,3] - (tmp0*src[0,3] + tmp7*src[2,3] + tmp8*src[3,3])
    dst22 = tmp2*src[0,3] + tmp7*src[1,3] + tmp10*src[3,3] - (tmp3*src[0,3] + tmp6*src[1,3] + tmp11*src[3,3])
    dst23 = tmp5*src[0,3] + tmp8*src[1,3] + tmp11*src[2,3] - (tmp4*src[0,3] + tmp9*src[1,3] + tmp10*src[2,3])
    dst30 = tmp2*src[2,2] + tmp5*src[3,2] + tmp1*src[1,2] - (tmp4*src[3,2] + tmp0*src[1,2] + tmp3*src[2,2])
    dst31 = tmp8*src[3,2] + tmp0*src[0,2] + tmp7*src[2,2] - (tmp6*src[2,2] + tmp9*src[3,2] + tmp1*src[0,2])
    dst32 = tmp6*src[1,2] + tmp11*src[3,2] + tmp3*src[0,2] - (tmp10*src[3,2] + tmp2*src[0,2] + tmp7*src[1,2])
    dst33 = tmp10*src[2,2] + tmp4*src[0,2] + tmp9*src[1,2] - (tmp8*src[1,2] + tmp11*src[2,2] + tmp5*src[0,2])
    
    
    #/* calculate determinant */
    det=src[0,0]*dst00+src[1,0]*dst01+src[2,0]*dst02+src[3,0]*dst03
    #/* calculate matrix inverse */
    if det == 0:
        det = 0.
    else:
        det = 1./det
    
    dst[0,0] = dst00 * det
    dst[0,1] = dst01 * det
    dst[0,2] = dst02 * det
    dst[0,3] = dst03 * det

    dst[1,0] = dst10 * det
    dst[1,1] = dst11 * det
    dst[1,2] = dst12 * det
    dst[1,3] = dst13 * det
    
    dst[2,0] = dst20 * det
    dst[2,1] = dst21 * det
    dst[2,2] = dst22 * det
    dst[2,3] = dst23 * det
    
    dst[3,0] = dst30 * det
    dst[3,1] = dst31 * det
    dst[3,2] = dst32 * det
    dst[3,3] = dst33 * det
        
@njit(DOTMM_DECL,  cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotmm4(a,b,out):
    
    a0 = a[0,0]*b[0,0]+a[0,1]*b[1,0]+a[0,2]*b[2,0]+a[0,3]*b[3,0]
    a1 = a[0,0]*b[0,1]+a[0,1]*b[1,1]+a[0,2]*b[2,1]+a[0,3]*b[3,1]
    a2 = a[0,0]*b[0,2]+a[0,1]*b[1,2]+a[0,2]*b[2,2]+a[0,3]*b[3,2]
    a3 = a[0,0]*b[0,3]+a[0,1]*b[1,3]+a[0,2]*b[2,3]+a[0,3]*b[3,3] 

    b0 = a[1,0]*b[0,0]+a[1,1]*b[1,0]+a[1,2]*b[2,0]+a[1,3]*b[3,0]
    b1 = a[1,0]*b[0,1]+a[1,1]*b[1,1]+a[1,2]*b[2,1]+a[1,3]*b[3,1]
    b2 = a[1,0]*b[0,2]+a[1,1]*b[1,2]+a[1,2]*b[2,2]+a[1,3]*b[3,2]
    b3 = a[1,0]*b[0,3]+a[1,1]*b[1,3]+a[1,2]*b[2,3]+a[1,3]*b[3,3] 

    c0 = a[2,0]*b[0,0]+a[2,1]*b[1,0]+a[2,2]*b[2,0]+a[2,3]*b[3,0]
    c1 = a[2,0]*b[0,1]+a[2,1]*b[1,1]+a[2,2]*b[2,1]+a[2,3]*b[3,1]
    c2 = a[2,0]*b[0,2]+a[2,1]*b[1,2]+a[2,2]*b[2,2]+a[2,3]*b[3,2]
    c3 = a[2,0]*b[0,3]+a[2,1]*b[1,3]+a[2,2]*b[2,3]+a[2,3]*b[3,3]
    
    d0 = a[3,0]*b[0,0]+a[3,1]*b[1,0]+a[3,2]*b[2,0]+a[3,3]*b[3,0]
    d1 = a[3,0]*b[0,1]+a[3,1]*b[1,1]+a[3,2]*b[2,1]+a[3,3]*b[3,1]
    d2 = a[3,0]*b[0,2]+a[3,1]*b[1,2]+a[3,2]*b[2,2]+a[3,3]*b[3,2]
    d3 = a[3,0]*b[0,3]+a[3,1]*b[1,3]+a[3,2]*b[2,3]+a[3,3]*b[3,3]     
 
    out[0,0] = a0
    out[0,1] = a1
    out[0,2] = a2
    out[0,3] = a3

    out[1,0] = b0
    out[1,1] = b1
    out[1,2] = b2
    out[1,3] = b3  

    out[2,0] = c0
    out[2,1] = c1
    out[2,2] = c2
    out[2,3] = c3 
    
    out[3,0] = d0
    out[3,1] = d1
    out[3,2] = d2
    out[3,3] = d3 


@njit(DOTMM_DECL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotmm3(a,b,out):
    a0 = a[0,0]*b[0,0]+a[0,1]*b[1,0]+a[0,2]*b[2,0]
    a1 = a[0,0]*b[0,1]+a[0,1]*b[1,1]+a[0,2]*b[2,1]
    a2 = a[0,0]*b[0,2]+a[0,1]*b[1,2]+a[0,2]*b[2,2]

    b0 = a[1,0]*b[0,0]+a[1,1]*b[1,0]+a[1,2]*b[2,0]
    b1 = a[1,0]*b[0,1]+a[1,1]*b[1,1]+a[1,2]*b[2,1]
    b2 = a[1,0]*b[0,2]+a[1,1]*b[1,2]+a[1,2]*b[2,2]

    c0 = a[2,0]*b[0,0]+a[2,1]*b[1,0]+a[2,2]*b[2,0]
    c1 = a[2,0]*b[0,1]+a[2,1]*b[1,1]+a[2,2]*b[2,1]
    c2 = a[2,0]*b[0,2]+a[2,1]*b[1,2]+a[2,2]*b[2,2]
    

    out[0,0] = a0
    out[0,1] = a1
    out[0,2] = a2

    out[1,0] = b0
    out[1,1] = b1
    out[1,2] = b2

    out[2,0] = c0
    out[2,1] = c1
    out[2,2] = c2
    
@njit(DOTMM_DECL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotmm2(a,b,out):
    a0 = a[0,0]*b[0,0]+a[0,1]*b[1,0]
    a1 = a[0,0]*b[0,1]+a[0,1]*b[1,1]

    b0 = a[1,0]*b[0,0]+a[1,1]*b[1,0]
    b1 = a[1,0]*b[0,1]+a[1,1]*b[1,1] 
   
    out[0,0] = a0
    out[0,1] = a1

    out[1,0] = b0
    out[1,1] = b1
    
@njit([(NFDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotr2m(r,a,out):
    a0 = a[0,0]*r[0]-a[2,0]*r[1]
    a1 = a[0,1]*r[0]-a[2,1]*r[1]
    a2 = a[0,2]*r[0]-a[2,2]*r[1]
    a3 = a[0,3]*r[0]-a[2,3]*r[1] 

    b0 = a[1,0]*r[0]+a[3,0]*r[1]
    b1 = a[1,1]*r[0]+a[3,1]*r[1]
    b2 = a[1,2]*r[0]+a[3,2]*r[1]
    b3 = a[1,3]*r[0]+a[3,3]*r[1] 

    c0 = a[0,0]*r[1]+a[2,0]*r[0]
    c1 = a[0,1]*r[1]+a[2,1]*r[0]
    c2 = a[0,2]*r[1]+a[2,2]*r[0]
    c3 = a[0,3]*r[1]+a[2,3]*r[0] 
    
    d0 = -a[1,0]*r[1]+a[3,0]*r[0]
    d1 = -a[1,1]*r[1]+a[3,1]*r[0]
    d2 = -a[1,2]*r[1]+a[3,2]*r[0]
    d3 = -a[1,3]*r[1]+a[3,3]*r[0]   
 
    out[0,0] = a0
    out[0,1] = a1
    out[0,2] = a2
    out[0,3] = a3

    out[1,0] = b0
    out[1,1] = b1
    out[1,2] = b2
    out[1,3] = b3  

    out[2,0] = c0
    out[2,1] = c1
    out[2,2] = c2
    out[2,3] = c3 
    
    out[3,0] = d0
    out[3,1] = d1
    out[3,2] = d2
    out[3,3] = d3 
    
    
@njit([(NCDTYPE[:,:],NFDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotmr2(a,r,out):
    a0 = a[0,0]*r[0]-a[0,2]*r[1]
    a1 = a[0,1]*r[0]-a[0,3]*r[1]
    a2 = a[0,0]*r[1]+a[0,2]*r[0]
    a3 = a[0,1]*r[1]+a[0,3]*r[0] 
    
    b0 = a[1,0]*r[0]-a[1,2]*r[1]
    b1 = a[1,1]*r[0]-a[1,3]*r[1]
    b2 = a[1,0]*r[1]+a[1,2]*r[0]
    b3 = a[1,1]*r[1]+a[1,3]*r[0] 

    c0 = a[2,0]*r[0]-a[2,2]*r[1]
    c1 = a[2,1]*r[0]-a[2,3]*r[1]
    c2 = a[2,0]*r[1]+a[2,2]*r[0]
    c3 = a[2,1]*r[1]+a[2,3]*r[0] 

    d0 = a[3,0]*r[0]-a[3,2]*r[1]
    d1 = a[3,1]*r[0]-a[3,3]*r[1]
    d2 = a[3,0]*r[1]+a[3,2]*r[0]
    d3 = a[3,1]*r[1]+a[3,3]*r[0] 
 
    out[0,0] = a0
    out[0,1] = a1
    out[0,2] = a2
    out[0,3] = a3

    out[1,0] = b0
    out[1,1] = b1
    out[1,2] = b2
    out[1,3] = b3  

    out[2,0] = c0
    out[2,1] = c1
    out[2,2] = c2
    out[2,3] = c3 
    
    out[3,0] = d0
    out[3,1] = d1
    out[3,2] = d2
    out[3,3] = d3 
    
@njit([(NFDTYPE[:],NCDTYPE[:],NCDTYPE[:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotr2v(r,a,out):
    out[0]= a[0]*r[0]-a[2]*r[1]
    out[1]= a[1]*r[0]+a[3]*r[1]
    out[2]= a[0]*r[1]+a[2]*r[0]    
    out[3]= -a[1]*r[1]+a[3]*r[0]

@njit(DOTMV_DECL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmv3(a, b, out):
    out0 = b[0] * a[0,0] + b[1] * a[0,1] + b[2] * a[0,2]  
    out1 = b[0] * a[1,0] + b[1] * a[1,1] + b[2] * a[1,2] 
    out2 = b[0] * a[2,0] + b[1] * a[2,1] + b[2] * a[2,2] 
    out[0]= out0
    out[1]= out1
    out[2]= out2

@njit(DOTMV_DECL,cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmv4(a, b, out):
    out0 = b[0] * a[0,0] + b[1] * a[0,1] + b[2] * a[0,2] + b[3] * a[0,3] 
    out1 = b[0] * a[1,0] + b[1] * a[1,1] + b[2] * a[1,2] + b[3] * a[1,3] 
    out2 = b[0] * a[2,0] + b[1] * a[2,1] + b[2] * a[2,2] + b[3] * a[2,3] 
    out3 = b[0] * a[3,0] + b[1] * a[3,1] + b[2] * a[3,2] + b[3] * a[3,3] 
    out[0]= out0
    out[1]= out1
    out[2]= out2
    out[3]= out3  
    
@njit(DOTMV_DECL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmv24(a, b, out):
    out0 = a[0,0] * b[0] + a[0,1] * b[1] + a[0,2] * b[2] +a[0,3] * b[3]
    out1 = a[1,0] * b[0] + a[1,1] * b[1] + a[1,2] * b[2] +a[1,3] * b[3]
    out[0]= out0
    out[1]= out1
      
@njit(DOTMV_DECL,cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmv2(a, b, out):
    out0 = a[0,0] * b[0] + a[0,1] * b[1] 
    out1 = a[1,0] * b[0] + a[1,1] * b[1]
    out[0]= out0
    out[1]= out1
    
@njit(DOTVV_DECL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotvv2(a, b, out):
    out[0] = np.conj(a[0]) * b[0]  + np.conj(a[1]) * b[1]

@njit(DOTVV_DECL,  cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotvv3(a, b, out):
    out0 = np.conj(a[0]) * b[0] 
    out1 = np.conj(a[1]) * b[1]
    out2 = np.conj(a[2]) * b[2] 
    out[0] = out0 + out1 + out2
    
@njit(DOTVV_DECL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotvv4(a, b, out):
    out0 = np.conj(a[0]) * b[0] 
    out1 = np.conj(a[1]) * b[1]
    out2 = np.conj(a[2]) * b[2] 
    out3 = np.conj(a[3]) * b[3]
    out[0] = out0+ out1 + out2 + out3

    
@njit(DOTMD_DECL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotmd4(a, b, out):
    a0 = a[0,0]*b[0]
    a1 = a[0,1]*b[1]
    a2 = a[0,2]*b[2]
    a3 = a[0,3]*b[3] 
    
    out[0,0] = a0
    out[0,1] = a1
    out[0,2] = a2
    out[0,3] = a3

    a0 = a[1,0]*b[0]
    a1 = a[1,1]*b[1]
    a2 = a[1,2]*b[2]
    a3 = a[1,3]*b[3] 
    
    out[1,0] = a0
    out[1,1] = a1
    out[1,2] = a2
    out[1,3] = a3  

    a0 = a[2,0]*b[0]
    a1 = a[2,1]*b[1]
    a2 = a[2,2]*b[2]
    a3 = a[2,3]*b[3] 
    
    out[2,0] = a0
    out[2,1] = a1
    out[2,2] = a2
    out[2,3] = a3    
 
    a0 = a[3,0]*b[0]
    a1 = a[3,1]*b[1]
    a2 = a[3,2]*b[2]
    a3 = a[3,3]*b[3] 
    
    out[3,0] = a0
    out[3,1] = a1
    out[3,2] = a2
    out[3,3] = a3
    
@njit(DOTMD_DECL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotmd2(a, b, out):
    assert a.shape[0] == 2
    a0 = a[0,0]*b[0]
    a1 = a[0,1]*b[1]
    
    out[0,0] = a0
    out[0,1] = a1

    a0 = a[1,0]*b[0]
    a1 = a[1,1]*b[1]
  
    out[1,0] = a0
    out[1,1] = a1

@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmf4(a, b, out):
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            b0 = b[0,i,j]
            b1 = b[1,i,j]
            b2 = b[2,i,j]
            b3 = b[3,i,j]
            
            out0 = a[i,j,0,0] * b0 + a[i,j,0,1] * b1 + a[i,j,0,2] * b2 +a[i,j,0,3] * b3
            out1 = a[i,j,1,0] * b0 + a[i,j,1,1] * b1 + a[i,j,1,2] * b2 +a[i,j,1,3] * b3
            out2 = a[i,j,2,0] * b0 + a[i,j,2,1] * b1 + a[i,j,2,2] * b2 +a[i,j,2,3] * b3
            out3 = a[i,j,3,0] * b0 + a[i,j,3,1] * b1 + a[i,j,3,2] * b2 +a[i,j,3,3] * b3
            
            out[0,i,j]= out0
            out[1,i,j]= out1
            out[2,i,j]= out2
            out[3,i,j]= out3  
            
@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmf2(a, b, out):
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            b0 = b[0,i,j]
            b1 = b[1,i,j]
            
            out0 = a[i,j,0,0] * b0 + a[i,j,0,1] * b1 
            out1 = a[i,j,1,0] * b0 + a[i,j,1,1] * b1 
            
            out[0,i,j]= out0
            out[1,i,j]= out1
            
@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmdmf4(a, d, b, f,out):
    for i in range(f.shape[1]):
        for j in range(f.shape[2]):
            f0 = f[0,i,j]
            f1 = f[1,i,j]
            f2 = f[2,i,j]
            f3 = f[3,i,j]
            
            out0 = b[i,j,0,0] * f0 + b[i,j,0,1] * f1 + b[i,j,0,2] * f2 +b[i,j,0,3] * f3
            out1 = b[i,j,1,0] * f0 + b[i,j,1,1] * f1 + b[i,j,1,2] * f2 +b[i,j,1,3] * f3
            out2 = b[i,j,2,0] * f0 + b[i,j,2,1] * f1 + b[i,j,2,2] * f2 +b[i,j,2,3] * f3
            out3 = b[i,j,3,0] * f0 + b[i,j,3,1] * f1 + b[i,j,3,2] * f2 +b[i,j,3,3] * f3
            
            b0 = out0*d[i,j,0]
            b1 = out1*d[i,j,1]
            b2 = out2*d[i,j,2]
            b3 = out3*d[i,j,3]
            
            out[0,i,j]= a[i,j,0,0] * b0 + a[i,j,0,1] * b1 + a[i,j,0,2] * b2 +a[i,j,0,3] * b3
            out[1,i,j]= a[i,j,1,0] * b0 + a[i,j,1,1] * b1 + a[i,j,1,2] * b2 +a[i,j,1,3] * b3
            out[2,i,j]= a[i,j,2,0] * b0 + a[i,j,2,1] * b1 + a[i,j,2,2] * b2 +a[i,j,2,3] * b3
            out[3,i,j]= a[i,j,3,0] * b0 + a[i,j,3,1] * b1 + a[i,j,3,2] * b2 +a[i,j,3,3] * b3   

@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmdmf2(a, d, b, f,out):
    for i in range(f.shape[1]):
        for j in range(f.shape[2]):
            f0 = f[0,i,j]
            f1 = f[1,i,j]
            
            out0 = b[i,j,0,0] * f0 + b[i,j,0,1] * f1 
            out1 = b[i,j,1,0] * f0 + b[i,j,1,1] * f1 
            
            b0 = out0*d[i,j,0]
            b1 = out1*d[i,j,1]
            
            out[0,i,j]= a[i,j,0,0] * b0 + a[i,j,0,1] * b1 
            out[1,i,j]= a[i,j,1,0] * b0 + a[i,j,1,1] * b1  
    

