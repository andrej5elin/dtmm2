"""
Transfer Matrix Method
======================

4x4 transfer matrix and 2x2 scattering matrix method functions for 1D calculation. 

The implementation is based on standard formulation of 4x4 transfer matrix method.




4x4 method
----------

Layers are stacked in the z direction, field vectors describing the field are 
m = (Ex,Hy,Ey,Hx), Core functionality is defined by field matrix calculation
functions:
    
Field vector creation/conversion functions
++++++++++++++++++++++++++++++++++++++++++

* :func:`.avec` for amplitude vector (eigenmode amplitudes).
* :func:`.fvec` for field vector creation,
* :func:`.avec2fvec` for amplitude to field conversion.
* :func:`.fvec2avec` for field to amplitude conversion.

Field matrix functions
++++++++++++++++++++++

* :func:`.fmat` for general field matrix.
* :func:`.f_iso` for isotropic input and output field matrix.
* :func:`.ffi_iso` besides field matrix, computes also the inverse of the field matrix. 
* :func:`.alphaf` for general field vectors and field coefficents calculation (eigensystem calculation). 
* :func:`.alphaffi` computes also the inverse of the field matrix.
* :func:`.phase_mat` for phase matrix calculation.

Layer/stack computation
+++++++++++++++++++++++

* :func:`.layer_mat` for layer matrix calculation Mi=Fi.Pi.Fi^-1
* :func:`.stack_mat` for stack matrix caluclation M = M1.M2.M3....
* :func:`.system_mat` for system matrix calculation Fin^-1.M.Fout

Transmission/reflection calculation 
+++++++++++++++++++++++++++++++++++

* :func:`.transmit4x4` to work with the computed system  matrix
* :func:`.transfer4x4` or :func:`.transfer` for higher level interface 

Intensity and Ez Hz field
+++++++++++++++++++++++++

* :func:`.poynting` the z component of the Poynting vector.
* :func:`.intensity` the absolute value of the Poytning vector.
* :func:`.EHz` for calculation of the z component of the E and H fields.

2x2 method
----------
    
todo..

"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm2.conf import NCDTYPE,NFDTYPE, NUDTYPE, CDTYPE, FDTYPE, NUMBA_TARGET, \
                        NUMBA_PARALLEL, NUMBA_CACHE, NUMBA_FASTMATH, DTMMConfig, deprecation
from dtmm2.rotation import  _calc_rotations_uniaxial, _calc_rotations, _rotate_diagonal_tensor, _rotate_tensor
from dtmm2.linalg import  dotmdm, dotmm, inv, dotmv
from dtmm2._linalg import _dotr2m, _dotr2v, _dotmr2
from dtmm2.data import refind2eps
from dtmm2.rotation import rotation_vector2, rotation_matrix, rotate_tensor, rotate_diagonal_tensor
from dtmm2.print_tools import print_progress

import dtmm2.rotation as rotation

import numba as nb
from numba import prange
import time

# free space impedance
Z0 = 376.73031366857 

if NUMBA_PARALLEL == False:
    prange = range

#: default eigenfiled calculation method, can be any of AVAILABLE_FIELD_EIG_METHODS keys.
FIELD_EIG_METHOD = "auto"

#: maps eigenfield caluclation method string to integer
AVAILABLE_FIELD_EIG_METHODS = {"auto" : 0,
                               "isotropic" : 1,
                               "uniaxial" : 2,
                               "biaxial" : 3,
                               "general" : 4}

sqrt = np.sqrt

# available mode parameters.
_mode_int = {"r" : +1, "t" : +1, 1 : 1, -1 : -1}
_mode_int_none = {"b" : None, None : None}
_mode_int_none.update(_mode_int)

def _mode_to_int_or_none(mode):
    try:
        return _mode_int_none[mode]
    except KeyError:
        raise ValueError("Invalid propagation mode '{}'.".format(mode))

def _mode_to_int(mode):
    try :
        return _mode_int[mode]
    except KeyError:
        raise ValueError("Invalid propagation mode '{}'.".format(mode))


# low level implementations and worker functions
#-----------------------------------------------

@nb.njit([(NFDTYPE,NCDTYPE[:],NCDTYPE[:,:])])                                                                
def _auxiliary_matrix_eps(beta,eps,Lm):
    """Computes all elements of the auxiliary matrix of shape 4x4 for non-magnetic systems.
    
    Electric permitivity tensor is assumed to be symmetric, while relative permeabiliity is unity.
    """
    #copy in case we are reusing eps memory for output
    eps0 = eps[0]
    eps1 = eps[1]
    eps2 = eps[2]
    eps3 = eps[3]
    eps4 = eps[4]
    eps5 = eps[5]
    
    eps2m = 1./eps2
    eps4eps2m = eps4*eps2m
    eps5eps2m = eps5*eps2m
    
    Lm[0,0] = (-beta*eps4eps2m)
    Lm[0,1] = 1.-beta*beta*eps2m
    Lm[0,2] = (-beta*eps5eps2m)
    Lm[0,3] = 0.
    Lm[1,0] = eps0- eps4*eps4eps2m
    Lm[1,1] = Lm[0,0]
    Lm[1,2] = eps3- eps5*eps4eps2m
    Lm[1,3] = 0.
    Lm[2,0] = 0.
    Lm[2,1] = 0.
    Lm[2,2] = 0.
    Lm[2,3] = -1. 
    Lm[3,0] = (-1.0*Lm[1,2])
    Lm[3,1] = (-1.0*Lm[0,2])
    Lm[3,2] = beta * beta + eps5*eps5eps2m - eps1  
    Lm[3,3] = 0.  
 
    
@nb.njit([(NFDTYPE,) + (NCDTYPE,)*36 + (NCDTYPE[:,:],)])          
def _auxiliary_matrix_m(beta,
                        m11,m12,m13,m14,m15,m16,
                        m21,m22,m23,m24,m25,m26,
                        m31,m32,m33,m34,m35,m36,
                        m41,m42,m43,m44,m45,m46,      
                        m51,m52,m53,m54,m55,m56, 
                        m61,m62,m63,m64,m65,m66, 
                        Lm):
    
    d = m33 * m66 - m36 * m63
    
    a31 = (m61 * m36 - m31 * m66)/ d 
    a32 = ((m62-beta) * m36 - m32 * m66)/d 
    a34 = (m64 * m36 - m34 * m66) / d
    a35 = (m65 * m36 - (m35 + beta)* m66) / d
    a61 = (m63 * m31 - m33 * m61)/ d
    a62 = (m63 * m32 - m33 * (m62 - beta)) / d
    a64 = (m63 * m34 - m33 * m64) / d
    a65 = (m63 * (m35 + beta) - m33 * m65) / d
    
    d21 = m11 + m13 * a31 + m16 * a61
    d23 = m12 + m13 * a32 + m16 * a62
    d24 = -(m14 + m13 * a34 + m16 * a64)
    d22 = m15 + m13 * a35 + m16 * a65
    d41 = m21 + m23* a31 + (m26 - beta) * a61
    d43 = m22 + m23 * a32 + (m26-beta) * a62
    d44 = -(m24 + m23*a34 + (m26 - beta) * a64)
    d42 = m25 + m23*a35 + (m26 - beta) * a65
    d31 = -(m41 + m43*a31 + m46 * a61)
    d33 = -(m42 + m43 * a32 + m46* a62)
    d34 = m44 + m43*a34 + m46 * a64
    d32 = -(m45 + m43 * a35 + m46 * a65)
    d11 = m51 + (m53 + beta) * a31 + m56 * a61
    d13 = m52 + (m53 + beta) * a32 + m56 * a62
    d14 = -(m54 + (m53 + beta) * a34 + m56 * a64)
    d12 = m55 + (m53 + beta) * a35 + m56 * a65
    
    Lm[0,0] = d11
    Lm[0,1] = d12
    Lm[0,2] = d13
    Lm[0,3] = -d14
    Lm[1,0] = d21
    Lm[1,1] = d22
    Lm[1,2] = d23
    Lm[1,3] = -d24
    Lm[2,0] = d31
    Lm[2,1] = d32
    Lm[2,2] = d33
    Lm[2,3] = -d34    
    Lm[3,0] = -d41
    Lm[3,1] = -d42
    Lm[3,2] = -d43
    Lm[3,3] = d44      
    

@nb.njit([(NFDTYPE,NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:])])                                                                
def _auxiliary_matrix_eps_mu(beta,eps, mu, Lm):
    """Computes all elements of the auxiliary matrix of shape 4x4 for magnetic systems.
    
    The permitivity and permeability tensors are both assumed to be symmetric.
    
    See [Berreman] for details.
    """
    m11 = eps[0] #eps11
    m22 = eps[1] #eps22
    m33 = eps[2] #eps33
    m12 = eps[3] #eps12
    m21 = eps[3] #eps is symmetric
    m13 = eps[4] #eps13
    m31 = eps[4] #eps is symmetric
    m23 = eps[5] #eps23
    m32 = eps[5] #eps is symmetric
    
     
    m44 = mu[0] #mu11
    m55 = mu[1] #mu22
    m66 = mu[2] #mu33
    m45 = mu[3] #mu12
    m54 = mu[3] #mu is symmetric
    m46 = mu[4] #mu13
    m64 = mu[4] #mu is symmetric
    m56 = mu[5] #mu23
    m65 = mu[5] #mu is symmetric
    
    m41 = 0
    m51 = 0
    m61 = 0
    m42 = 0
    m52 = 0
    m62 = 0    
    m43 = 0
    m53 = 0
    m63 = 0  

    m14 = 0
    m15 = 0
    m16 = 0
    m24 = 0
    m25 = 0
    m26 = 0    
    m34 = 0
    m35 = 0
    m36 = 0  
    
    _auxiliary_matrix_m(beta,
                            m11,m12,m13,m14,m15,m16,
                            m21,m22,m23,m24,m25,m26,
                            m31,m32,m33,m34,m35,m36,
                            m41,m42,m43,m44,m45,m46,      
                            m51,m52,m53,m54,m55,m56, 
                            m61,m62,m63,m64,m65,m66, 
                            Lm)
    
@nb.njit([(NFDTYPE,NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:])])                                                                
def _auxiliary_matrix_general(beta,eps, mu, rho, rhop, Lm):
    """Computes all elements of the auxiliary matrix of shape 4x4 for optically active systems

    See [Berreman] for details.
    """
    m11 = eps[0,0]
    m22 = eps[1,1]
    m33 = eps[2,2] 
    m12 = eps[0,1]
    m21 = eps[1,0] 
    m13 = eps[0,2] 
    m31 = eps[2,0] 
    m23 = eps[1,2] 
    m32 = eps[2,1] 
    
    m44 = mu[0,0]
    m55 = mu[1,1]
    m66 = mu[2,2]
    m45 = mu[0,1]
    m54 = mu[1,0]
    m46 = mu[0,2]
    m64 = mu[2,0]
    m56 = mu[1,2]
    m65 = mu[2,1]
    
    m41 = rho[0,0]
    m51 = rho[1,0]
    m61 = rho[2,0]
    m42 = rho[0,1]
    m52 = rho[1,1]
    m62 = rho[2,1]    
    m43 = rho[0,2]
    m53 = rho[1,2]
    m63 = rho[2,2]  

    m14 = rhop[0,0]
    m15 = rhop[0,1]
    m16 = rhop[0,2]
    m24 = rhop[1,0]
    m25 = rhop[1,1]
    m26 = rhop[1,2]    
    m34 = rhop[2,0]
    m35 = rhop[2,1]
    m36 = rhop[2,2]  
    
    _auxiliary_matrix_m(beta,
                            m11,m12,m13,m14,m15,m16,
                            m21,m22,m23,m24,m25,m26,
                            m31,m32,m33,m34,m35,m36,
                            m41,m42,m43,m44,m45,m46,      
                            m51,m52,m53,m54,m55,m56, 
                            m61,m62,m63,m64,m65,m66, 
                            Lm)
    
@nb.guvectorize([(NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:])],
                 "(),(k),(n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)                                                               
def _auxiliary_matrix_eps_vec(beta,eps,dummy,Lm):
    """Computes all elements of the auxiliary matrix of shape 4x4."""
    assert len(eps) == 6
    _auxiliary_matrix_eps(beta[0],eps,Lm)
    
@nb.guvectorize([(NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:])],
                 "(),(k),(k),(n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)                                                               
def _auxiliary_matrix_eps_mu_vec(beta,eps,mu,dummy,Lm):
    """Computes all elements of the auxiliary matrix of shape 4x4."""
    assert len(eps) == 6
    _auxiliary_matrix_eps_mu(beta[0],eps,mu,Lm)
    
@nb.guvectorize([(NFDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])],
                 "(),(k,k),(k,k),(k,k),(k,k),(n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)                                                               
def _auxiliary_matrix_general_vec(beta,eps,mu,rho,rhop,dummy,Lm):
    """Computes all elements of the auxiliary matrix of shape 4x4."""
    assert len(eps) == 4
    _auxiliary_matrix_general(beta[0],eps,mu,rho,rhop,Lm)
    
def auxiliary_matrix(beta = None, epsilon = None, mu = None, out = None):
    """Computes auxiliary matrix in refraction plane"""
    beta, phi = _default_beta_phi(beta,None)
    epsilon = _default_epsilon(epsilon)
    
    if mu is None:
        # simplified (non-magnetic) system
        return _auxiliary_matrix_eps_vec(beta,epsilon,_dummy_array,out)
    else:
        mu = _default_epsilon(mu)
        return _auxiliary_matrix_eps_mu_vec(beta,epsilon,mu,_dummy_array,out)
  
# @nb.njit([(NFDTYPE,NCDTYPE,NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
# def _f_iso(beta,eps,F):
#     """computes eigenvalue alpha and eigenvector field matrix of isotropic material"""
#     #n = eps0[0]**0.5
#     aout = sqrt(eps-beta**2)
#     if aout != 0.:
#         gpout = eps/aout
#         gsout = -aout
#         F[0,0] = 0.5 
#         F[0,1] = 0.5
#         F[0,2] = 0.
#         F[0,3] = 0.
#         F[1,0] = 0.5 * gpout 
#         F[1,1] = -0.5 * gpout 
#         F[1,2] = 0.
#         F[1,3] = 0.
#         F[2,0] = 0.
#         F[2,1] = 0.
#         F[2,2] = 0.5 
#         F[2,3] = 0.5
#         F[3,0] = 0.
#         F[3,1] = 0.
#         F[3,2] = 0.5 * gsout 
#         F[3,3] = -0.5 * gsout 
#     else:
        
#         F[...]=0.

#set to 0.5 or 1.        
POYNTING_NORM = 0.5
        
@nb.njit([(NFDTYPE,NCDTYPE,NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _f_iso(beta,eps,F):
    """computes eigenvalue alpha and eigenvector field matrix of isotropic material"""
    #n = eps0[0]**0.5
    aout = sqrt(eps-beta**2)
    if aout != 0.:
        gpout = eps/aout
        gsout = -aout
        pfact = 1/np.sqrt(POYNTING_NORM*np.abs(np.real(gpout)))
        sfact = 1/np.sqrt(POYNTING_NORM*np.abs(np.real(gsout)))
        F[0,0] = pfact
        F[0,1] = pfact
        F[0,2] = 0.
        F[0,3] = 0.
        F[1,0] = pfact * gpout 
        F[1,1] = - pfact * gpout 
        F[1,2] = 0.
        F[1,3] = 0.
        F[2,0] = 0.
        F[2,1] = 0.
        F[2,2] = sfact 
        F[2,3] = sfact
        F[3,0] = 0.
        F[3,1] = 0.
        F[3,2] = sfact * gsout 
        F[3,3] = -sfact * gsout 
    else:
        
        F[...]=0.

@nb.njit([(NFDTYPE,NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaf_iso(beta,eps0,alpha,F):
    """computes eigenvalue alpha and eigenvector field matrix of isotropic material"""
    #n = eps0[0]**0.5
    aout = sqrt(eps0[0]-beta**2)
    if aout != 0.:
        gpout = eps0[0]/aout
        gsout = -aout
        alpha[0] = aout
        alpha[1] = -aout
        alpha[2] = aout
        alpha[3] = -aout 
        F[0,0] = 0.5 
        F[0,1] = 0.5
        F[0,2] = 0.
        F[0,3] = 0.
        F[1,0] = 0.5 * gpout 
        F[1,1] = -0.5 * gpout 
        F[1,2] = 0.
        F[1,3] = 0.
        F[2,0] = 0.
        F[2,1] = 0.
        F[2,2] = 0.5 
        F[2,3] = 0.5
        F[3,0] = 0.
        F[3,1] = 0.
        F[3,2] = 0.5 * gsout 
        F[3,3] = -0.5 * gsout 
    else:
        
        F[...]=0.
        alpha[...] = 0.

@nb.njit([NFDTYPE(NCDTYPE[:])], cache = NUMBA_CACHE)
def _poynting(field):
    """Computes poynting vector from the field vector"""
    tmp1 = (field[0].real * field[1].real + field[0].imag * field[1].imag)
    tmp2 = (field[2].real * field[3].real + field[2].imag * field[3].imag)
    return (tmp1-tmp2)*POYNTING_NORM    

@nb.njit([(NFDTYPE,NCDTYPE[:],NFDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaf_uniaxial(beta,eps0,R,alpha,F): 
    """computes eigenvalue alpha and eigenvector field matrix of uniaxial material"""
    #uniaxial case
    ct = R[2,2]
    st = -R[2,0] 
    st2 = st * st
    ct2 = ct * ct
    
    sf = -R[0,1]
    cf = R[1,1]

    eps11 = eps0[0]

    
    delta = eps0[2] -  eps11
    if beta == 0. : #same as calculation for beta !=0, except faster... no multiplying with zeros
        ev02 =  eps11 
        evs = sqrt(ev02)
        u = eps11 + delta * ct2
        w = eps11 * (ev02 + delta)
        sq = sqrt(u*w)/u
        evpp = sq
        evpm = -sq
        
    else: #can also be used for beta=0... just slower
        ev02 =  eps11 - beta * beta
        evs = sqrt(ev02)
        
        u = eps11 + delta * ct2
        gama = beta * cf
        v = gama * delta * 2 * st * ct
        w = gama * gama * (delta * st2)- eps11 * (ev02 + delta)
        
        sq = sqrt(v*v-4*u*w)/2/u
        v = v/2/u
        
        evpp = -v + sq
        evpm = -v - sq
        

    alpha[0] = evpp
    alpha[1] = evpm
    alpha[2] = evs
    alpha[3] = -evs    


    if beta == 0.:
        # we must hanlde the beta = 0 case because in case st = 0, we get invalid
        # eigenvectors.
        if st == 0.: 
            # this is the only combination that yields degenerate case for any
            # phi angle. Therefore, in order to preserve proper eigenvalue sorting
            # we must enforce the phi = 0 here, so sf=0 and cf=1 to return the
            # solution bound to the eigenframe of the material.
            sf = 0.
            cf = 1.
        
        # compared to beta !=0 case, we do not have dependence on st here.
        # the eigenvectors are only a function of the phi angle.
        
        eps11sf = eps11 * sf
        evssf = evs*sf
        evscf = evs*cf
        eps11cf = eps11*cf

        F[0,2] = -evssf
        F[1,2] = -eps11sf
        F[2,2] = evscf
        F[3,2] = -eps11cf
        
        F[0,3] = -evssf
        F[1,3] = eps11sf 
        F[2,3] = evscf 
        F[3,3] = eps11cf
        
        F[0,0] = cf
        F[1,0] = evpp *cf
        F[2,0] = sf
        F[3,0] = -evpp *sf 
        
        F[0,1] = cf
        F[1,1] = evpm *cf
        F[2,1] = sf
        F[3,1] = -evpm *sf    
        
    else:
        # general case
        sfst = (R[1,2])
        cfst = (R[0,2])                   
                                    
        ctbeta = ct * beta
        ctbetaeps11 = ctbeta / eps11
        eps11sfst = eps11 * sfst
        evssfst = evs*sfst
        evscfst = evs*cfst
        evsctbeta = evs*ctbeta
        ev02cfst = ev02*cfst
        ev02cfsteps11 = ev02cfst/eps11

        F[0,2] = evssfst 
        F[1,2] = eps11sfst
        F[2,2] = ctbeta - evscfst
        F[3,2] = ev02cfst - evsctbeta    

        F[0,3] = -evssfst
        F[1,3] = eps11sfst
        F[2,3] = evscfst + ctbeta
        F[3,3] = ev02cfst + evsctbeta
        
        F[0,0] = evpp*ctbetaeps11 - ev02cfsteps11
        F[1,0] = ctbeta - evpp *cfst 
        F[2,0] = -sfst
        F[3,0] = evpp *sfst
        
        F[0,1] = -evpm*ctbetaeps11 + ev02cfsteps11
        F[1,1] = evpm *cfst - ctbeta
        F[2,1] = sfst
        F[3,1] = -evpm *sfst 
        
    #normalize base vectors
    for j in range(4):
        tmp = 0.
        for i in range(4):
            tmp += F[i,j].real * F[i,j].real + F[i,j].imag * F[i,j].imag
        
        tmp = (tmp) ** 0.5
        
        #tmp = _poynting(F[:,j])
        if tmp != 0:
            F[0,j] = F[0,j]/tmp 
            F[1,j] = F[1,j]/tmp 
            F[2,j] = F[2,j]/tmp 
            F[3,j] = F[3,j]/tmp 
            
@nb.njit([(NCDTYPE[:,:], NCDTYPE[:,:])])
def _normalize_mat(fmat, out):
    for i in range(4):
        if fmat.shape[0] == 6:
            n = np.abs(_poynting(fmat[1:-1,i]))**0.5
        else:
            n = np.abs(_poynting(fmat[:,i]))**0.5
        if n == 0.:
            n = 0.
        else:
            n = 1./n
        out[0,i] = fmat[0,i] * n 
        out[1,i] = fmat[1,i] * n
        out[2,i] = fmat[2,i] * n
        out[3,i] = fmat[3,i] * n
        if fmat.shape[0] == 6:
            out[4,i] = fmat[4,i] * n
            out[5,i] = fmat[5,i] * n    
            
@nb.njit([(NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE)
def _copy_sorted(alpha,fmat, out_alpha, out_fmat):
    """Eigen modes sorting based on the computed poynting vector direction"""
    i = 0
    j = 1

    ok = True
     
    for k in range(4):
        p = _poynting(fmat[:,k])

        if p >= 0.:
            if i >=4:
                ok = False
            if ok:
                out_alpha[i] = alpha[k]
                out_fmat[:,i] = fmat[:,k]
            i = i + 2
        else:
            if j >=4:
                ok = False
            if ok : 
                out_alpha[j] = alpha[k]
                out_fmat[:,j] = fmat[:,k]
            j = j + 2
    if ok == False:
        print("Could not sort eigenvectors! Setting the field matrix eigenvectors to zero, and eigenvalue to np.nan!")
        for i in range(4):
            #indicate that something went wrong, and that sorting was unsucesful
            out_alpha[i] = np.nan
            out_fmat[:,i] = 0

@nb.guvectorize([(NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])],
                 "(n),(n,n)->(n),(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)            
def _copy_sorted_vec(alpha,fmat, out_alpha, out_fmat):
    alpha = alpha.copy()# need to copy for inplace conversion
    fmat = fmat.copy()# need to copy for inplace conversion
    _copy_sorted(alpha,fmat, out_alpha, out_fmat)
                    
@nb.guvectorize([(NUDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:])],
                 "(),(),(),(m),(l),(k),(n)->(n),(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaf_vec(case,beta,phi,rv,epsv,epsa,dummy,alpha,F):
    """eigenvalue solver. Depending on the material parameter
    we choose isotropic, uniaxial  or a biaxial solver.
    
    Becaue the auxiliary matrix is written in the rotated frame (in the plane of incidence with phi = 0)
    We need to rotate the computed vectors using _dotr2m 
    """
    CASE = case[0]
    
    # auto determine CASE
    if CASE == 0:
        if len(epsv) == 6 and (epsv[3] != 0 or epsv[4] != 0 or epsv[5] != 0):
            # in case we use 6 component tensor, and at least one offdiagonal is not zero
            # we must treat the sample as a biaxial
            CASE = 4 # biaxial with off-diag epsilon
        elif (epsv[0] == epsv[1]):
            if epsv[1]==epsv[2]:
                CASE = 1 #isotropic
            else:
                CASE = 2 #uniaxial
        else:
            CASE = 3 #biaxial with diagonal eps
            
    #F is a 4x4 matrix... we can use 3x3 part for Rotation matrix and F[3] for eps  temporary data
            
    #isotropic case
    if CASE == 1:
        _alphaf_iso(beta[0],epsv,alpha,F)
        _dotr2m(rv,F,F)
    #uniaxial
    elif CASE == 2:
        R = F.real
        _calc_rotations_uniaxial(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        _alphaf_uniaxial(beta[0],epsv,R,alpha,F)
        _dotr2m(rv,F,F)
        
    else:#biaxial case or 6-component
        R = F.real 
        eps = np.empty((6,), epsv.dtype)
        _calc_rotations(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        if CASE == 3:
            _rotate_diagonal_tensor(R,epsv,eps)
        else: #CASE 0, we must rotate full 6-component tensor
            assert len(epsv) >= 6
            _rotate_tensor(R,epsv,eps)
        _auxiliary_matrix_eps(beta[0],eps,F) #calculate Lm matrix and put it to F
        alpha0,F0 = np.linalg.eig(F)
        _copy_sorted(alpha0,F0,alpha,F)#copy data and sort it
        _dotr2m(rv,F,F)
        
@nb.guvectorize([(NUDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NFDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])],
                 "(),(),(),(j),(k),(l),(m,n)->(n),(m,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaf6_vec(case,beta,phi,rv,epsv,epsa,dummy,alpha,F):
    """eigenvalue solver. Depending on the material parameter
    we choose isotropic, uniaxial  or a biaxial solver.
    
    Becaue the auxiliary matrix is written in the rotated frame (in the plane of incidence with phi = 0)
    We need to rotate the computed vectors using _dotr2m 
    """
    
    if dummy.shape[0] == 6:
        FULL_MATRIX = True # output a 6x4 matrix
    else:
        FULL_MATRIX = False # output a 4x4 matrix

    CASE = case[0]
    
    # auto determine CASE
    if CASE == 0:
        if len(epsv) == 6 and (epsv[3] != 0 or epsv[4] != 0 or epsv[5] != 0):
            # in case we use 6 component tensor, and at least one offdiagonal is not zero
            # we must treat the sample as a biaxial
            CASE = 4 # biaxial with off-diag epsilon
        elif (epsv[0] == epsv[1]):
            if epsv[1]==epsv[2]:
                CASE = 1 #isotropic
            else:
                CASE = 2 #uniaxial
        else:
            CASE = 3 #biaxial with diagonal eps
            
    #F is a 6x4 matrix... we can use 3x3 part for Rotation matrix and F[:,3] for eps temporary data      
    
    R = F.real[0:3,0:3] #reuse array

    #isotropic case
    if CASE == 1:
        if FULL_MATRIX:
            e0 = 0. #eps_zx/eps_zz
            e1 = -beta[0]/epsv[2] #beta/eps_zz
            e2 = 0. #eps_zy/eps_zz
        _alphaf_iso(beta[0],epsv,alpha,F)
    #uniaxial
    elif CASE == 2:
        _calc_rotations_uniaxial(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        if FULL_MATRIX:
            eps = F[:,3]
            _rotate_diagonal_tensor(R,epsv,eps)
            e0 = -eps[4]/eps[2] #eps_zx/eps_zz
            e1 = -beta[0]/eps[2] #beta/eps_zz
            e2 = -eps[5]/eps[2] #eps_zy/eps_zz
        _alphaf_uniaxial(beta[0],epsv,R,alpha,F)
        
    else:#biaxial case or 6-component
        _calc_rotations(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        if FULL_MATRIX == True:
            eps = F[:,3] # we can use F matrix
        else:
            eps = np.empty((6,), epsv.dtype)
        
        if CASE == 3:
            _rotate_diagonal_tensor(R,epsv,eps)
        else: #CASE 0, we must rotate full 6-component tensor
            assert len(epsv) >= 6
            _rotate_tensor(R,epsv,eps)
        if FULL_MATRIX:
            e0 = -eps[4]/eps[2] #eps_zx/eps_zz
            e1 = -beta[0]/eps[2] #beta/eps_zz
            e2 = -eps[5]/eps[2] #eps_zy/eps_zz
        _auxiliary_matrix_eps(beta[0],eps,F) #calculate Lm matrix and put it to F
        alpha0,F0 = np.linalg.eig(F[0:4])
        _copy_sorted(alpha0,F0,alpha,F[0:4])#copy data and sort it
        
    if FULL_MATRIX:
        
        F00 = beta[0]*F[2,0]
        F01 = beta[0]*F[2,1]
        F02 = beta[0]*F[2,2]
        F03 = beta[0]*F[2,3]
        
        F50 = e0 * F[0,0] + e1 * F[1,0] + e2 * F[2,0]
        F51 = e0 * F[0,1] + e1 * F[1,1] + e2 * F[2,1]
        F52 = e0 * F[0,2] + e1 * F[1,2] + e2 * F[2,2]
        F53 = e0 * F[0,3] + e1 * F[1,3] + e2 * F[2,3]
    
    _dotr2m(rv,F,F)
    
    if FULL_MATRIX:
        #reorder and fill
        F[5,0] = F50
        F[5,1] = F51
        F[5,2] = F52
        F[5,3] = F53
        
        F[4] = F[3]
        F[3] = F[2]
        F[2] = F[1]
        F[1] = F[0]
            
        F[0,0] = F00
        F[0,1] = F01
        F[0,2] = F02
        F[0,3] = F03
        
    #_normalize_mat(F,F)

@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(l),(),(m,n)->(m,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _b_vec(beta,phi,rv,eps,dummy,F):
    if dummy.shape[0] == 6:
        FULL_MATRIX = True # output a 6x4 matrix
    else:
        FULL_MATRIX = False # output a 4x4 matrix

    _f_iso(beta[0],eps[0],F)
    
    if FULL_MATRIX:
        e0 = 0. #eps_zx/eps_zz
        e1 = -beta[0]/eps[0] #beta/eps_zz
        e2 = 0. #eps_zy/eps_zz
    
    _dotmr2(F,rv,F)

    if FULL_MATRIX:
        
        F00 = beta[0]*F[2,0]
        F01 = beta[0]*F[2,1]
        F02 = beta[0]*F[2,2]
        F03 = beta[0]*F[2,3]
        
        F50 = e0 * F[0,0] + e1 * F[1,0] + e2 * F[2,0]
        F51 = e0 * F[0,1] + e1 * F[1,1] + e2 * F[2,1]
        F52 = e0 * F[0,2] + e1 * F[1,2] + e2 * F[2,2]
        F53 = e0 * F[0,3] + e1 * F[1,3] + e2 * F[2,3]
    
    _dotr2m(rv,F,F)
    
    if FULL_MATRIX:
        #reorder and fill
        F[5,0] = F50
        F[5,1] = F51
        F[5,2] = F52
        F[5,3] = F53
        
        F[4] = F[3]
        F[3] = F[2]
        F[2] = F[1]
        F[1] = F[0]
            
        F[0,0] = F00
        F[0,1] = F01
        F[0,2] = F02
        F[0,3] = F03    
    
    
@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(m),(),(m,n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _f_iso_vec(beta,phi,rv,eps,dummy,F):
    _f_iso(beta[0],eps[0],F)
    _dotr2m(rv,F,F)
    
@nb.guvectorize([(NUDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NFDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(),(m),(l),(k),(m,n)->(n),(n,n),(m,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphafhe_vec(case,beta,phi,rv,epsv,epsa,dummy,alpha,F,HE):
    """eigenvalue solver. Depending on the material parameter
    we choose isotropic, uniaxial  or a biaxial solver.
    
    Becaue the auxiliary matrix is written in the rotated frame (in the plane of incidence with phi = 0)
    We need to rotate the computed vectors using _dotr2m 
    """
    
    CASE = case[0]
    
    #F is a 4x4 matrix... we can use 3x3 part for Rotation matrix and F[3] for eps  temporary data
    if CASE == 0:
        if len(epsv) == 6 and (epsv[3] != 0 or epsv[4] != 0 or epsv[5] != 0):
            # in case we use 6 component tensor, and at least one offdiagonal is not zero
            # we must treat the sample as a biaxial
            CASE = 4 # biaxial with off-diag epsilon
        elif (epsv[0] == epsv[1]):
            if epsv[1]==epsv[2]:
                CASE = 1 #isotropic
            else:
                CASE = 2 #uniaxial
        else:
            CASE = 3 #biaxial with diagonal eps
           
    R = F.real

    #isotropic case
    if CASE == 1:
        e0 = 0. #eps_zx/eps_zz
        e1 = -beta[0]/epsv[2] #beta/eps_zz
        e2 = 0. #eps_zy/eps_zz
        _alphaf_iso(beta[0],epsv,alpha,F)
    #uniaxial
    elif CASE == 2:
        eps = np.empty((6,),epsv.dtype)
        _calc_rotations_uniaxial(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        _rotate_diagonal_tensor(R,epsv,eps)
        _alphaf_uniaxial(beta[0],epsv,R,alpha,F)
        e0 = -eps[4]/eps[2] #eps_zx/eps_zz
        e1 = -beta[0]/eps[2] #beta/eps_zz
        e2 = -eps[5]/eps[2] #eps_zy/eps_zz
    else:#biaxial case or 6-component
        eps = np.empty((6,),epsv.dtype)
        #eps = F.ravel()#reuse F memory (eps is length 6 1D array)
        _calc_rotations(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        if CASE == 3:
            _rotate_diagonal_tensor(R,epsv,eps)
        else: #CASE 0, we must rotate full 6-component tensor
            assert len(epsv) >= 6
            _rotate_tensor(R,epsv,eps)
        e0 = -eps[4]/eps[2] #eps_zx/eps_zz
        e1 = -beta[0]/eps[2] #beta/eps_zz
        e2 = -eps[5]/eps[2] #eps_zy/eps_zz
        _auxiliary_matrix_eps(beta[0],eps,F) #calculate Lm matrix and put it to F
        alpha0,F0 = np.linalg.eig(F[0:4])
        _copy_sorted(alpha0,F0,alpha,F)#copy data and sort it
    
    _dotr2m(rv,F,F)
        
    e0 = -eps[4]/eps[2] #eps_zx/eps_zz
    e1 = -beta[0]/eps[2] #eps_zx/eps_zz
    e2 = -eps[5]/eps[2] #eps_zy/eps_zz
    
    HE[0,0] = -beta[0] * rv[1]
    HE[0,1] = 0
    HE[0,2] = beta[0] * rv[0]
    HE[0,3] = 0
    
    HE[1,0] = e0 * rv[0] - e2 * rv[1]
    HE[1,1] = e1* rv[0]
    HE[1,2] = e0 * rv[1] + e2 * rv[0]
    HE[1,3] = -e1* rv[1]
    
        
@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NFDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(m),(l),(k),(m,n)->(m,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _he_vec(beta,phi,rv,epsv,epsa,dummy,out):
    """eigenvalue solver. Depending on the material parameter
    we choose isotropic, uniaxial  or a biaxial solver.
    
    Becaue the auxiliary matrix is written in the rotated frame (in the plane of incidence with phi = 0)
    We need to rotate the computed vectors using _dotr2m 
    """
    R = np.empty(shape = (3,3), dtype = rv.dtype)
    
    if len(epsv) == 6 and (epsv[3] != 0 or epsv[4] != 0 or epsv[5] != 0):
        # in case we use 6 component tensor, and at least one offdiagonal is not zero
        # we must treat the sample as a biaxial
        CASE = 0 # biaxial with off-diag epsilon
    elif (epsv[0] == epsv[1]):
        if epsv[1]==epsv[2]:
            CASE = 1 #isotropic
        else:
            CASE = 2 #uniaxial
    else:
        CASE = 3 #biaxial with diagonal eps
                    
    #isotropic or uniaxial case
    if CASE == 1 or CASE == 2:
        _calc_rotations_uniaxial(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
    else:#biaxial case or 6-component
        _calc_rotations(phi[0],epsa,R)
        
    eps = out.ravel() #reuse F memory (eps is length 6 1D array)

    if CASE == 0: #CASE 0, we must rotate full 6-component tensor
        _rotate_tensor(R,epsv,eps)
    else: 
        _rotate_diagonal_tensor(R,epsv,eps)
        
    e0 = -eps[4]/eps[2] #eps_zx/eps_zz
    e1 = -beta[0]/eps[2] #eps_zx/eps_zz
    e2 = -eps[5]/eps[2] #eps_zy/eps_zz
    
    out[0,0] = -beta[0] * rv[1]
    out[0,1] = 0
    out[0,2] = beta[0] * rv[0]
    out[0,3] = 0
    
    out[1,0] = e0 * rv[0] - e2 * rv[1]
    out[1,1] = e1* rv[0]
    out[1,2] = e0 * rv[1] + e2 * rv[0]
    out[1,3] = -e1* rv[1]
        
        
#dummy arrays for gufuncs    
_dummy_array = np.empty((4,),CDTYPE)
_dummy_array4 = _dummy_array
_dummy_array6 = np.empty((6,),CDTYPE)
_dummy_array64 = np.empty((6,4),CDTYPE)
_dummy_array44 = np.empty((4,4),CDTYPE)
_dummy_array24 = np.empty((2,4),CDTYPE)
#_dummy_array2 = np.empty((9,),CDTYPE)
_dummy_EH = np.empty((2,),CDTYPE)

    
def _alphaf(beta,phi,epsv,epsa,out = None):
    rv = rotation_vector2(phi) 
    return _alphaf_vec(beta,phi,rv,epsv,epsa,_dummy_array, out = out)

def _default_beta_phi(beta, phi):
    """Checks the validity of beta, phi arguments and sets default values if needed"""
    beta = np.asarray(beta, FDTYPE) if beta is not None else np.asarray(0., FDTYPE)
    phi = np.asarray(phi, FDTYPE) if phi is not None else np.asarray(0., FDTYPE)
    return beta, phi

def _default_epsv_epsa(epsv, epsa):
    """Checks the validity of epsv, epsa arguments and sets default values if needed"""
    epsv = np.asarray(epsv, CDTYPE) if epsv is not None else np.asarray((1.,1.,1.), CDTYPE)
    epsa = np.asarray(epsa, FDTYPE) if epsa is not None else np.asarray((0.,0.,0.), FDTYPE)
    assert epsv.shape[-1] >= 3
    assert epsa.shape[-1] >= 3
    return epsv, epsa

def _default_eps_iso(eps):
    eps = np.asarray(eps, CDTYPE) if eps is not None else np.asarray(1., CDTYPE)
    #epsv[-1] = epsv[-1].mean()
    return eps
    

def _default_epsilon(epsilon):
    """Checks the validity of epsilon argument and sets default values if needed"""
    epsilon = np.asarray(epsilon, CDTYPE) if epsilon is not None else np.asarray((1.,1.,1.,0.,0.,0.), CDTYPE)
    assert epsilon.shape[-1] >= 6
    return epsilon

def _as_field_vec(fvec):
    """converts input to valid field vector"""
    fvec = np.asarray(fvec, dtype = CDTYPE)
    assert fvec.shape[-1] == 4
    return fvec

# user functions
#---------------

FORCEPSV = {'factor' : 1} 

def eig(m, normalize = False):
    """same as np.linalg.eig. For 4x4 matrix it also performs pynting vector 
    eigenvector sorting. Optionally, normalizes the vectors to unit intensity"""
    alpha,F = np.linalg.eig(m)
    if F.shape[-1] == 4:
        alpha,F = _copy_sorted_vec(alpha,F,out = (alpha,F))
        if normalize == True:
            F = normalize_f(F,out = F)
        return alpha, F
    else:
        return alpha, F

def field_eig(epsilon = None, angles = None, beta = None, phi = None, normalize = True, out = None):
    """Computes alpha and field arrays (eigen values and eigen vectors arrays).
    
    Broadcasting rules apply.
    
    Parameters
    ----------
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    epsilon : (...,3) array or (..., 6) array, optional
        Dielectric tensor eigenvalues array (defaults to unity), or full dielectric
        tensor array (diagonal and off-diagonal values)
    angles : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)) for 
        rotation of the epsilon tensor to laboratory frame.
    out : (ndarray,ndarray), optional
        Output arrays.
       
    Returns
    -------
    alpha, fieldmat: (ndarray, ndarray)
        Eigen values and eigen vectors arrays. 
    """
    
    beta, phi = _default_beta_phi(beta,phi)
    epsv, epsa = _default_epsv_epsa(epsilon, angles)
    rv = rotation_vector2(phi)
    case = AVAILABLE_FIELD_EIG_METHODS[FIELD_EIG_METHOD]
    if out is None:
        out = _alphaf_vec(case,beta,phi,rv,epsv,epsa,_dummy_array)
    else:
        out = _alphaf_vec(case,beta,phi,rv,epsv,epsa,_dummy_array, out = out)
        
    # if out is None:
    #     out = _alphaf_vec(beta,phi,rv,epsv,epsa,_dummy_array)
    # else:
    #     1/0
    #     out = _alphaf_vec(beta,phi,rv,epsv,epsa,_dummy_array, out = out)
        
    # # if FORCEPSV['factor'] != 1:
    # #     epsveff = (epsv[...,0] + epsv[...,1] + epsv[...,2])/3
    # #     delta = (epsv[...,0] - epsveff) * FORCEPSV['factor']
        
    # #     epsv = epsv.copy()
    # #     epsv[...,0] = epsveff + delta
    # #     epsv[...,1] = epsveff + delta
    # #     epsv[...,2] = epsveff - 2*delta
                   

        
    #     _, f =_alphaf_vec(beta,phi,rv,epsv,epsa,_dummy_array)
    #     out = out[0], f
    if normalize:
        normalize_f(out[1],out[1])
    return out

def field6_eig(epsilon = None, angles = None, beta = None, phi = None,  normalize = True, field = "m", out = None):
    """Computes alpha and field arrays (eigen values and eigen vectors arrays).
    
    Broadcasting rules apply.
    
    Parameters
    ----------
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    epsilon : (...,3) array or (..., 6) array, optional
        Dielectric tensor eigenvalues array (defaults to unity), or full dielectric
        tensor array (diagonal and off-diagonal values)
    angles : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)) for 
        rotation of the epsilon tensor to laboratory frame.
    out : (ndarray,ndarray), optional
        Output arrays.
       
    Returns
    -------
    alpha, fieldmat: (ndarray, ndarray)
        Eigen values and eigen vectors arrays. 
    """
    
    beta, phi = _default_beta_phi(beta,phi)
    epsv, epsa = _default_epsv_epsa(epsilon, angles)
    rv = rotation_vector2(phi)
    case = AVAILABLE_FIELD_EIG_METHODS[FIELD_EIG_METHOD]
    

    if out is None:
        out = _alphaf6_vec(case,beta,phi,rv,epsv,epsa,_dummy_array64)
    else:
        out = _alphaf6_vec(case,beta,phi,rv,epsv,epsa,_dummy_array64, out = out)

    if normalize:
        normalize_f(out[1],out[1])
    return out

def he_eig(epsilon = None, angles = None, beta = None, phi = None,  normalize = False,  out = None):
    """Same as field_eig, but returns also a result of hzez_mat"""
    beta, phi = _default_beta_phi(beta,phi)
    epsv, epsa = _default_epsv_epsa(epsilon, angles)
    rv = rotation_vector2(phi)  
    case = AVAILABLE_FIELD_EIG_METHODS[FIELD_EIG_METHOD]
    if out is None:
        out = _alphafhe_vec(case, beta,phi,rv,epsv,epsa,_dummy_array24)
    else:
        out = _alphafhe_vec(case, beta,phi,rv,epsv,epsa,_dummy_array24, out = out)
        
    if normalize:
        normalize_f(out[1],out[1])
    return out

def field_eigi(epsilon = None, angles = None, beta=None, phi=None, normalize = False, out = None):
    """Computes alpha and field arrays (eigen values and eigen vectors arrays)
    and inverse of the field array. See also :func:`field_eig` 
    
    Broadcasting rules apply.
    
    Parameters
    ----------
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    epsilon : (...,3) array or (..., 6) array, optional
        Dielectric tensor eigenvalues array (defaults to unity), or full dielectric
        tensor array (diagonal and off-diagonal values)
    angles : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    out : (ndarray,ndarray,ndarray), optional
        Output arrays.
       
    Returns
    -------
    alpha, field, ifield  : (ndarray, ndarray, ndarray)
        Eigen values and eigen vectors arrays and its inverse
     
    Examples    
    --------
    
    This is equivalent to
    
    >>> alpha,field = alphaf(0,0, [2,2,2], [0.,0.,0.])
    >>> ifield = inv(field)
    """
    if out is not None:
        a,f,fi = out
        field_eig(epsilon,angles,beta, phi, out = (a,f), normalize = normalize)
        inv(f,fi)
    else:
        a,f = field_eig(epsilon,angles, beta, phi, normalize = normalize)
        fi = inv(f)
    return a,f,fi

# for backward compatibility
def alphaf(beta = None, phi = None, epsv = None, epsa = None, out = None):
    deprecation("Deprecated, use field_eig instead.")
    return field_eig(epsv,epsa,beta,phi, out = out)

def alphaffi(beta = None, phi = None, epsv = None, epsa = None, out = None):
    deprecation("Deprecated, use field_eig instead.")
    return field_eigi(epsv,epsa,beta,phi, out = out)

def field_mat(epsilon = None, angles = None, beta = None, phi = None, normalize = True, **kwargs):
    """Computes field arrays ( eigen vectors arrays).
    
    Broadcasting rules apply.
    
    Parameters
    ----------
    epsilon : (...,3) array or (..., 6) array, optional
        Dielectric tensor eigenvalues array (defaults to unity), or full dielectric
        tensor array (diagonal and off-diagonal values)
    angles : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
       
    Returns
    -------
    F : ndarray
        Eigen vectors arrays. 
    """
    alpha, f = field_eig(epsilon,angles,beta, phi, normalize = normalize)
    return f

def field_iso_mat(eps = None, beta = None, phi = None, out = None):
    beta, phi = _default_beta_phi(beta,phi)
    eps = _default_eps_iso(eps)    
    rv = rotation_vector2(phi) 
    out = _f_iso_vec(beta,phi,rv,eps,_dummy_array24, out = out)
    return out    

def beam_mat(eps = None,beta = None,phi = None, field = "m",out = None):
    """Computes the beam matrix.
    
    Broadcasting rules apply.
    
    Parameters
    ----------
    eps : float, optional
        Isotropic dielectric tensor eigenvalue (defaults to unity)
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)

    Returns
    -------
    B : ndarray
        The beam matrix.
    """
    beta, phi = _default_beta_phi(beta,phi)
    eps = _default_eps_iso(eps)
    rv = rotation_vector2(phi) 
    
    if field == "m":
        out = _b_vec(beta,phi,rv,eps,_dummy_array44, out = out)
    elif field == "f": 
        out = _b_vec(beta,phi,rv,eps,_dummy_array64, out = out)
    return out

def refraction_mat(eps = None,beta = None,phi = None, invert = False, out = None):
    if invert == False:
        F0 = field_iso_mat(1.,0.,phi,out = out)
        F1 =  field_iso_mat(eps,beta,phi)
        F0i = inv(F0, out = F0)
        return dotmm(F1,F0i, out = F0i)
    else:
        F0 = field_iso_mat(1.,0.,phi)
        F1 =  field_iso_mat(eps,beta,phi, out = out)
        F1i = inv(F1, out = F1)
        return dotmm(F0,F1i, out = F1i)        
    
def fmat(beta = None, phi = None, epsv = None, epsa = None):
    deprecation("Please use field_mat instead")
    return field_mat(epsv,epsa,beta,phi)

#alias  for bacward compatibility
f = fmat


def alphaE(beta = None,phi = None,epsv = None,epsa = None, mode = +1, normalize_fmat = False, out = None):
    """Computes E-field eigenvalue and eigenvector matrix for the 2x2 formulation.
    
    Broadcasting rules apply to all arguments, except `mode`.
    
    Parameters
    ----------
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    epsv : (...,3) array, optional
        Dielectric tensor eigenvalues array (defaults to unity).
    epsa : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    normalize_fmat : bool
        Whether to normalize field matrix prior to converting to E matrix
    out : (ndarray,ndarray), optional
        Output arrays where results are written.
        
    Returns
    -------
    alpha, field : (ndarray, ndarray)
        Eigen values and eigen vectors arrays.
    
    """
    mode = _mode_to_int(mode)
    alpha,f = field_eig(epsv,epsa, beta, phi)
    if normalize_fmat == True:
        normalize_f(f,f)
    e = E_mat(f,mode = mode, copy = False)
    if mode == 1:
        alpha = alpha[...,::2]
    else:
        alpha = alpha[...,1::2]
    if out is not None:
        out[0][...] = alpha
        out[1][...] = e
    else:
        out = alpha.copy(), e.copy()
    return out

def alphaEEi(beta = None, phi = None, epsv = None, epsa = None, mode = +1, normalize_fmat = False, out = None):
    """Computes E-field eigenvalue and eigenvector matrix and inverse of the 
    eigenvector array for the 2x2 formulation. See also :func:`alphaE` 
    
    Broadcasting rules apply to all arguments, except `mode`.
    
    Parameters
    ----------
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    epsv : (...,3) array, optional
        Dielectric tensor eigenvalues array (defaults to unity).
    epsa : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    normalize_fmat : bool
        Whether to normalize field matrice prior to converting to E matrix
    out : (ndarray,ndarray,ndarray), optional
        Output arrays where results are written.

    Returns
    -------
    alpha, field, ifield : (ndarray, ndarray, ndarray)
        Eigen values and eigen vectors arrays and its inverse
    """
    mode = _mode_to_int(mode)
    if out is None:
        alpha,E = alphaE(beta,phi,epsv,epsa, mode = mode, normalize_fmat = normalize_fmat)
        return alpha, E, inv(E)
    else:
        alpha, E, Ei = out
        alpha, E = alphaE(beta,phi,epsv,epsa, mode = mode, normalize_fmat = normalize_fmat, out = (alpha, E))
        Ei = inv(E,out = Ei)
        return alpha, E, Ei 


@nb.vectorize([NCDTYPE(NCDTYPE,NFDTYPE)],
    target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
def _phase_mat_vec(alpha,kd):
    return np.exp(NCDTYPE(1j)*kd*alpha)

def _phasem(alpha,kd,out = None):
    kd = np.asarray(kd,FDTYPE)[...,None]
    out = _phase_mat_vec(alpha,kd,out)
    #if out.shape[-1] == 4:
    #    out[...,1::2]=0.
    return out

#to make autoapi happy...    
def phasem(*args,**kwargs):
    return _phasem(*args,**kwargs)
    

DELETEME = {1: 0., -1 : 0.}

def phase_mat(alpha, kd, mode = None,  out = None):
    """Computes a 4x4 or 2x2 diagonal matrix from eigenvalue matrix alpha 
    and wavenumber. 
    
    The output is a diagonal, that is, a vector of length 2 or 4, depending on
    the input alpha array.
    
    Broadcasting rules apply to all arguments, except `mode`.
    
    Parameters
    ----------
    alpha : array
        The eigenvalue alpha array of shape (...,4) or (...,2).
    kd : float
        The kd phase value (layer thickness times wavenumber in vacuum).
    mode : int, optional
        If specified, converts the phase matrix to 2x2, taking either forward 
        propagating mode (+1), or negative propagating mode (-1).
    out : ndarray, optional
        Output array where results are written.
        
    Returns
    -------
    diag : array
        Phase diagonal matrix of shape (...,4) or (...,2).
    """
    mode = _mode_to_int_or_none(mode)
    alpha = np.asarray(alpha)
    
    if mode is not None:
        if alpha.shape[-1] != 4:
            raise ValueError("alpha array must be a 4-vector if mode is set.")
    
    kd = np.asarray(kd, dtype = FDTYPE)
    if out is None:
        if mode is None:
            b = np.broadcast(alpha,kd[...,None])
        else:
            b = np.broadcast(alpha[...,::2],kd[...,None])
        out = np.empty(b.shape, dtype = CDTYPE)
        
   # alpha = alpha.copy()
    
    #alpha[...,1::2] -= 1j*DELETEME[-1]
    #alpha[...,::2] += 1j*DELETEME[+1]
    
    if mode == +1:
        phasem(alpha[...,::2],kd, out = out)
    elif mode == -1:
        phasem(alpha[...,1::2],kd,out = out)
    elif mode is None:
        out = phasem(alpha,kd, out = out) 
    else:
        raise ValueError("Unknown propagation mode.")
    # if DELETEME[-1] != 0:  
    #     out[...,1::2] *= DELETEME[-1]
    # if DELETEME[1] != 0:  
    #     out[...,::2] *= DELETEME[1]       
    return out 

def fvec2hevec(field, epsilon = None, angles = None, beta = None, phi = None, out = None):
    """Converts input field4 (Ex, Hy, Ey, Hx) vector to a field6 vector (Hz, Ex, Hy, Ey, Hx, Ez)"""
    epsilon,angles = _default_epsv_epsa(epsilon, angles)
    beta, phi = _default_beta_phi(beta, phi)
    rv = rotation_vector2(-phi)
    
    return _field6(field, beta, phi, rv, epsilon,angles, _dummy_array6, out)

def hevec(fvec, hzez = None):
    if hzez is None:
        hzez = hzez_mat()
    if fvec.shape[-1] != 6:
        out = np.empty(fvec.shape[:-1] + (6,), dtype = fvec.dtype)
        out[...,1:-1] = fvec
    else:
        out = fvec
    dotmv(hzez, out[...,1:-1], out[...,0::5])
    return out

def hevec2fvec(field, copy = False):
    """Returns a view of the field4 array form the field6 array"""
    field = np.asarray(field)
    if field.shape[-1] != 6:
        raise ValueError("Not a valid field6 array")
    out = field[...,1::-1]
    return out.copy() if copy == True else out

def Evec(fvec, copy = False):
    """Returns a view of the E field part of the total field"""
    fvec = np.asarray(fvec)
    if fvec.shape[-1] == 4:
        out = fvec[...,::2]
    elif fvec.shape[-1] == 6:
        out = fvec[...,1::2]
    else:
        raise ValueError("Invalid field shape")
        
    return out.copy() if copy == True else out
    
def Hvec(fvec, copy = False):
    """Returns a view of the H field part of the total field"""
    fvec = np.asarray(fvec)
    if fvec.shape[-1] == 4:
        out = fvec[...,-1::-2]
    elif fvec.shape[-1] == 6:
        out = fvec[...,-2::-2]
    else:
        raise ValueError("Invalid field shape")
        
    return out.copy() if copy == True else out        
        

def iphase_mat(alpha, kd, cfact = 0.1, mode = +1,  out = None):
    """Computes incoherent 4x4 phase matrix from eigenvalue matrix alpha and wavenumber"""
    mode = _mode_to_int(mode)
    if mode == +1:
        np.add(alpha[...,1::2],alpha[...,1::2].real*2*1j*cfact, out = alpha[...,1::2])
    else:
        np.add(alpha[...,::2],-alpha[...,::2].real*2*1j*cfact, out = alpha[...,::2])
    return phase_mat(alpha, kd, mode = None, out = out)


@nb.guvectorize([(NCDTYPE[:], NFDTYPE[:])],
                    "(n)->()", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
def poynting(fvec, out):
    """Calculates a z-component of the poynting vector from the field vector
    
    Parameters
    ----------
    fvec : (...,4,4) array
        Field matrix array.
    out : ndarray, optional
        Output array where results are written.
        
    Results
    -------
    poynting : array
        The z component of the poynting vector.
    """
    assert fvec.shape[0] == 4
    out[0] = _poynting(fvec)

def fmat2poynting(fmat, out = None):
    """Calculates poynting vectors (z component) from the field matrix.
    
    Parameters
    ----------
    fmat : (...,4,4) array
        Field matrix array.
    out : ndarray, optional
        Output array where results are written.
        
    Returns
    -------
    vec : (...,4) array
        Fmat's columns poynting vector z component.
    """
    axes = list(range(fmat.ndim))
    n = axes.pop(-2)
    axes.append(n)
    fmat = fmat.transpose(*axes)
    return poynting(fmat, out = out)

# @nb.guvectorize([(NCDTYPE[:,:], NCDTYPE[:,:])],
#                     "(n,n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
# def normalize_mat(fmat, out):
#     """Normalizes columns of field matrix so that fmat2poytning of the resulted
#     matrix returns ones
    
#     Parameters
#     ----------
#     fmat : (...,4,4) array
#         Field matrix array.
#     out : ndarray, optional
#         Output array where results are written.
#     """
#     assert fmat.shape[0] == 4 and fmat.shape[1] == 4 
#     for i in range(4):
#         n = np.abs(_poynting(fmat[:,i]))**0.5
#         if n == 0.:
#             n = 0.
#         else:
#             n = 1./n
#         out[0,i] = fmat[0,i] * n 
#         out[1,i] = fmat[1,i] * n
#         out[2,i] = fmat[2,i] * n
#         out[3,i] = fmat[3,i] * n





@nb.guvectorize([(NCDTYPE[:,:], NCDTYPE[:,:])],
                    "(m,n)->(m,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
def normalize_mat(fmat, out):
    """Normalizes columns of field matrix so that fmat2poytning of the resulted
    matrix returns ones
    
    Parameters
    ----------
    fmat : (...,4,4) array
        Field matrix array.
    out : ndarray, optional
        Output array where results are written.
    """
    assert fmat.shape[0] in (6,4) and fmat.shape[1] == 4 
    _normalize_mat(fmat, out)

normalize_f = normalize_mat

def intensity(fvec,out = None):
    """Calculates absolute value of the z-component of the poynting vector
    
    Parameters
    ----------
    fvec : (...,4) array
        Field vector array.
    out : ndarray, optional
        Output array where results are written.    
    """
    fvec = _as_field_vec(fvec)
    p = poynting(fvec)
    return np.abs(p)

def projection_mat(fmat, fmati = None, mode = +1, out = None):
    """Calculates projection matrix from the given field matrix. By multiplying
    the field with this matrix you obtain only the forward (mode = +1) or
    backward (mode = -1) propagating field,
    
    Parameters
    ----------

    fmat : (...,4,4) array
        Field matrix array.
    fmati : (...,4,4)
        The inverse of the field matrix. If not provided it is computed from `fmat`.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    out : ndarray, optional
        Output array where results are written.   
    
    """
    mode = _mode_to_int(mode)
    if fmati is None:
        fmati = inv(fmat)
    diag = np.zeros(fmat.shape[:-1],fmat.dtype)
    if mode == 1:
        diag[...,0::2] = 1.
    else:
        diag[...,1::2] = 1.   
    return dotmdm(fmat,diag,fmati, out)   

# def interface_transmission_mat(fmatin = None, fmatout = None, mode = +1, out = None):
#     if mode == +1:
#         fmatini = inv(fmatin)
#         #stsem matrix for transfering output field to input field
#         smat = system_mat(fmatini = fmatini, fmatout = fmatout, transfer = "backward")
#         tmat = np.zeros_like(smat)
#         tmat[...,0::2,0::2] = inv(smat[...,0::2,0::2])
#         return dotmm(fmatout,dotmm(tmat,fmatini))
#     else:
#         fmatouti = inv(fmatout)
#         smat = system_mat(fmatin = fmatin, fmatouti = fmatouti, transfer = "forward")
#         tmat = np.zeros_like(smat)
#         tmat[...,1::2,1::2]  = inv(smat[...,1::2,1::2])
#         return dotmm(fmatin,dotmm(tmat,fmatouti))

def field_transmission_mat(rmat, fmatin = None, fmatout = None, mode = +1):
    mode = _mode_to_int(mode)
    if mode == +1:
        fmatini = inv(fmatin)
        tmat = np.zeros_like(rmat) 
        tmat[0::2,0::2] = rmat[0::2,0::2]
        return dotmm(fmatout,dotmm(tmat,fmatini))
    else:
        fmatouti = inv(fmatout)
        tmat = np.zeros_like(rmat)
        tmat[...,1::2,1::2]  = rmat[1::2,1::2]
        return dotmm(fmatin,dotmm(tmat,fmatouti))       

def field_reflection_mat(rmat, fmatin = None, fmatout = None, mode = +1):
    mode = _mode_to_int(mode)
    if mode == +1:
        fmatini = inv(fmatin)
        tmat = np.zeros_like(rmat) 
        tmat[1::2,0::2] = rmat[1::2,0::2]
        return dotmm(fmatin,dotmm(tmat,fmatini))
    else:
        fmatouti = inv(fmatout)
        tmat = np.zeros_like(rmat)
        tmat[...,0::2,1::2]  = rmat[0::2,1::2]
        return dotmm(fmatout,dotmm(tmat,fmatouti))  


def project(fvec, fmat, fmati = None, mode = +1, out = None):
    """Projects field vector using the given field matrix. By multiplying
    the field with the projection matrix you obtain only the forward (mode = +1) or
    backward (mode = -1) propagating field
    
    Parameters
    ----------
    fvec : (...,4) array
        INput field vector
    fmat : (...,4,4) array
        Field matrix array.
    fmati : (...,4,4)
        The inverse of the field matrix. If not provided it is computed from `fmat`.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    out : ndarray, optional
        Output array where results are written.   
    """
    pmat = projection_mat(fmat, fmati = fmati, mode = mode)
    return dotmv(pmat,fvec, out = out)

def EHz(fvec, beta = None, phi = None, epsv = None, epsa = None, out = None):
    """Constructs the z component of the electric and magnetic fields.
    
    Broadcasting rules apply.
    
    Parameters
    ----------
    fvec : (...,4,4) array
        Field matrix array.
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    epsv : (...,3) array, optional
        Dielectric tensor eigenvalues array (defaults to unity).
    epsa : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    out : (ndarray,ndarray), optional
        Output arrays where results are written.
        
    Returns
    -------
    EHz : ndarray
        EHz array of shape (...,2)
    """
    fvec = _as_field_vec(fvec)
    beta, phi = _default_beta_phi(beta, phi)
    epsv,epsa = _default_epsv_epsa(epsv, epsa)
    rv = rotation_vector2(-phi)
    
    return _EHz(fvec,beta,phi,rv, epsv,epsa,_dummy_EH,out)


    
@nb.guvectorize([(NCDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:])],
                 "(n),(),(),(m),(l),(k),(o)->(o)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _EHz(fvec, beta,phi,rv,epsv,epsa,dummy,out):
    eps = np.empty(shape = (6,), dtype = epsv.dtype)
    R = np.empty(shape = (3,3), dtype = rv.dtype)
    frot = np.empty_like(fvec)
    _dotr2v(rv,fvec,frot)
    
    _calc_rotations_uniaxial(phi[0],epsa,R)
    _rotate_diagonal_tensor(R,epsv,eps)
    out[0] = - (eps[4]*frot[0] + eps[5]*frot[2] + beta[0] * frot[1]) / eps[2]
    out[1] = beta[0] * frot[2]



@nb.guvectorize([(NCDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:])],
                 "(n),(),(),(m),(l),(k),(o)->(o)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _field6(fvec, beta,phi,rv,epsv,epsa,dummy,out):
    """Converts field4 to field6"""
    if len(epsv) == 6 and (epsv[3] != 0 or epsv[4] != 0 or epsv[5] != 0):
        # in case we use 6 component tensor, and at least one offdiagonal is not zero
        # we must treat the sample as a biaxial
        CASE = 0 # biaxial with off-diag epsilon
    elif (epsv[0] == epsv[1]):
        if epsv[1]==epsv[2]:
            CASE = 1 #isotropic
        else:
            CASE = 2 #uniaxial
    else:
        CASE = 3 #biaxial with diagonal eps
    
    eps = np.empty(shape = (6,), dtype = epsv.dtype)
    R = np.empty(shape = (3,3), dtype = rv.dtype)
    frot = np.empty_like(fvec)
    _dotr2v(rv,fvec,frot)
    if CASE == 1 or CASE == 2:
        _calc_rotations_uniaxial(phi[0],epsa,R)
    else:
        _calc_rotations(phi[0],epsa,R)
    if CASE == 0:
        _rotate_tensor(R,epsv,eps)
    else:  
        _rotate_diagonal_tensor(R,epsv,eps)
    out[0] = beta[0] * frot[2]
    out[1] = fvec[0]
    out[2] = fvec[1]
    out[3] = fvec[2]
    out[4] = fvec[3]   
    out[5] = - (eps[4]*frot[0] + eps[5]*frot[2] + beta[0] * frot[1]) / eps[2]



def T_mat(fmatin, fmatout, fmatini = None, fmatouti = None, mode = +1):
    """Computes amplitude interface transmittance matrix.
    
    Parameters
    ----------
    fmatin : (...,4,4) array
        Input field matrix array.
    fmatout : (...,4,4) array
        Output field matrix array.
    fmatini : (...,4,4) array
        Inverse of the input field matrix array.
    fmatouti : (...,4,4) array, optional
        Inverse of the output field matrix array. If not provided, it is computed
        from `fmatout`.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    """
    mode = _mode_to_int(mode)
    if fmatini is None:
        fmatini = inv(fmatin)
    if fmatouti is None:
        fmatouti = inv(fmatout)
    Sf = dotmm(fmatini,fmatout)
    Sb = dotmm(fmatouti,fmatin)
    out = np.zeros_like(Sf)
    if mode == +1:
        out[...,::2,::2] = inv(Sf[...,::2,::2])
        out[...,1::2,1::2] = Sb[...,1::2,1::2]
        return out
    else:
        out[...,1::2,1::2] = inv(Sf[...,1::2,1::2])
        out[...,::2,::2] = (Sb[...,::2,::2])
        return out
   
def S_mat(fmatin, fmatout, fmatini = None, overwrite_fmatin = False, mode = +1):
    """Computes the S matrix.
    
    Parameters
    ----------
    fmatin : (...,4,4) array
        Input field matrix array.
    fmatout : (...,4,4) array
        Output field matrix array.
    fmatini : (...,4,4) array
        Inverse of the input field matrix array.
    fmatouti : (...,4,4) array, optional
        Inverse of the output field matrix array. If not provided, it is computed
        from `fmatout`.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    """
    mode = _mode_to_int(mode)
    if overwrite_fmatin == True:
        out = fmatin
    else:
        out = None
    if fmatini is None:
        fmatini = inv(fmatin, out = out)
    S = dotmm(fmatini,fmatout, out = out)
    if mode == +1:
        return S[...,::2,::2],S[...,1::2,0::2]
    else:
        return S[...,1::2,1::2],S[...,0::2,1::2]


# def transmission_mat(fmatin, fmatout, fmatini = None, mode = +1,out = None):
#     """Computes the transmission matrix.
    
#     Parameters
#     ----------
#     fmatin : (...,4,4) array
#         Input field matrix array.
#     fmatout : (...,4,4) array
#         Output field matrix array.
#     fmatini : (...,4,4) array
#         Inverse of the input field matrix array.
#     fmatouti : (...,4,4) array, optional
#         Inverse of the output field matrix array. If not provided, it is computed
#         from `fmatout`.
#     mode : int
#         Either +1, for forward propagating mode, or -1 for negative propagating mode.
#     out : ndarray, optional
#         Output array where results are written.
#     """
#     mode = _mode_to_int(mode)
#     A,B = S_mat(fmatin, fmatout, fmatini = fmatini, mode = mode)
#     if mode == +1:
#         A1 = fmatin[...,::2,::2]
#         A2 = fmatout[...,::2,::2]
#     else:
#         A1 = fmatin[...,1::2,1::2]
#         A2 = fmatout[...,1::2,1::2]      
#     Ai = inv(A, out = out)
#     A1i = inv(A1)
#     return dotmm(dotmm(A2,Ai, out = Ai),A1i, out = Ai)
#
#def reflection_mat(fin, fout, fini = None, mode = +1,out = None):
#    A,B = S_mat(fin, fout, fini = fini, mode = mode)
#    if mode == +1:
#        A1p = fin[...,::2,::2]
#        A1m = fin[...,::2,1::2]
#    elif mode == -1:
#        A1p = fin[...,1::2,1::2]
#        A1m = fin[...,1::2,::2]  
#    else:
#        raise ValueError("Unknown propagation mode.")
#    Ai = inv(A, out = out)
#    A1pi = inv(A1p)
#    return dotmm(dotmm(dotmm(A1m,B,out = Ai),Ai, out = Ai),A1pi, out = Ai)

def tr_mat(fmatin, fmatout, fmatini = None, overwrite_fmatin = False, mode = +1, out = None):
    """Computes the 2x2 t and r matrix.
    
    Parameters
    ----------
    fmatin : (...,4,4) array
        Input field matrix array.
    fmatout : (...,4,4) array
        Output field matrix array.
    fmatini : (...,4,4) array
        Inverse of the input field matrix array.
    overwrite_fmatin : bool
        Specifies whether fmatin can be overwritten or not.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    out : (ndarray,ndarray), optional
        Output arrays where results are written.
    """    
    mode = _mode_to_int(mode)
    if overwrite_fmatin == True:
        er = E_mat(fmatin, mode = mode * (-1), copy = True)
    else:
        er = E_mat(fmatin, mode = mode * (-1), copy = False)
    et = E_mat(fmatout, mode = mode, copy = False)
    eti,eri = Etri_mat(fmatin, fmatout, fmatini = fmatini, overwrite_fmatin = overwrite_fmatin, mode = mode, out = out)
    return dotmm(et,eti, out = eti), dotmm(er,eri, out = eri)

def t_mat(fmatin, fmatout, fmatini = None, overwrite_fmatin = False, mode = +1, out = None):
    """Computes the 2x2 t matrix.
    
    Parameters
    ----------
    fmatin : (...,4,4) array
        Input field matrix array.
    fmatout : (...,4,4) array
        Output field matrix array.
    fmatini : (...,4,4) array
        Inverse of the input field matrix array.
    overwrite_fmatin : bool
        Specifies whether fmatin can be overwritten or not.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    out : ndarray, optional
        Output array where results are written.
    """   
    eti = Eti_mat(fmatin, fmatout, fmatini = fmatini, overwrite_fmatin = overwrite_fmatin, mode = mode, out = out)
    et = E_mat(fmatout, mode = mode, copy = False)
    return dotmm(et,eti, out = eti)

def E_mat(fmat, mode = None, copy = True):
    """Computes the E field matrix.
    
    Parameters
    ----------
    fmat : (...,4,4) array
        Field matrix array.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.

    """ 
    mode = _mode_to_int_or_none(mode)
    if mode == +1:
        e = fmat[...,::2,::2]
    elif mode == -1:
        e = fmat[...,::2,1::2]
    else:
        ep = fmat[...,::2,::2]
        en = fmat[...,::2,1::2]
        out = np.zeros_like(fmat)
        out[...,::2,::2] = ep
        out[...,1::2,1::2] = en
        return out 
    return e.copy() if copy else e  

def Eti_mat(fmatin, fmatout, fmatini = None, overwrite_fmatin = False, mode = +1, out = None):
    """Computes the inverse of the E field matrix (no reflections).
    
    Parameters
    ----------
    fmatin : (...,4,4) array
        Input field matrix array.
    fmatout : (...,4,4) array
        Output field matrix array.
    fmatini : (...,4,4) array
        Inverse of the input field matrix array.
    overwrite_fmatin : bool
        Specifies whether fmatin can be overwritten or not.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    out : ndarray, optional
        Output array where results are written.
    """     
    A = E_mat(fmatin, mode = mode, copy = False) 
    #Ai = inv(A, out = out)
    Ai = inv(A)
    St,Sr = S_mat(fmatin, fmatout, fmatini = fmatini, overwrite_fmatin = overwrite_fmatin, mode = mode)
    Sti = inv(St, out = St)
    return dotmm(Sti,Ai, out = out)


def Etri_mat(fmatin, fmatout, fmatini = None, overwrite_fmatin = False, mode = +1, out = None):
    """Computes the inverse of the E field matrix (with reflections).
    
    Parameters
    ----------
    fmatin : (...,4,4) array
        Input field matrix array.
    fmatout : (...,4,4) array
        Output field matrix array.
    fmatini : (...,4,4) array
        Inverse of the input field matrix array.
    overwrite_fmatin : bool
        Specifies whether fmatin can be overwritten or not.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    out : ndarray, optional
        Output array where results are written.
    """       
    out1, out2 = out if out is not None else (None, None)
    A = E_mat(fmatin, mode = mode, copy = False)
    Ai = inv(A, out = out1)  
    St,Sr = S_mat(fmatin, fmatout, fmatini = fmatini, overwrite_fmatin = overwrite_fmatin, mode = mode)
    Sti = inv(St, out = St)
    ei = dotmm(Sti,Ai,out = Ai)
    return ei, dotmm(Sr,ei, out = out2)
    
def E2H_mat(fmat, mode = +1, out = None): 
    """Computes the H field matrix from the field matrix.
    
    Parameters
    ----------
    
    fmat : (...,4,4) array
        Field matrix array.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    out : ndarray, optional
        Output array where results are written.
    """ 
    mode = _mode_to_int(mode)      
    if mode == +1:
        A = fmat[...,::2,::2]
        B = fmat[...,1::2,::2]
    else:
        A = fmat[...,::2,1::2]
        B = fmat[...,1::2,1::2]
    Ai = inv(A, out = out)
    return dotmm(B,Ai, out = Ai)  

def f_iso(beta = 0., phi = 0., n = 1.):
    """Returns field matrix for isotropic layer of a given refractive index
    and beta, phi parameters
    
    Broadcasting rules apply to all arguments, except `n`
    
    Parameters
    ----------
    
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    n : float
        Refractive index of the medium (1. by default).
    """
    epsv = refind2eps([n]*3)
    epsa = np.zeros(shape = (3,),dtype= FDTYPE)
    alpha, f = field_eig(epsv,epsa, beta, phi)    
    return f

def ffi_iso(beta=0.,phi = 0., n=1):
    """Returns field matrix and inverse of the field matrix for isotropic layer 
    of a given refractive index and beta, phi parameters.
    
    Broadcasting rules apply to all arguments, except `n`
    
    Parameters
    ----------
    
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    n : float
        Refractive index of the medium (1. by default).
    """
    epsv = refind2eps([n]*3)
    epsa = np.zeros(shape = (3,),dtype= FDTYPE)
    alpha, f, fi = field_eigi(epsv,epsa, beta,phi)    
    return f,fi

    


class TransferMatrix1D():
    def __init__(self, kd = 1, epsilon = None, angles = None, beta = None, phi = None):
        self._kd = np.asarray(kd)
        self._epsilon, self._angles = _default_epsv_epsa(epsilon, angles)
        self._beta, self._phi = _default_beta_phi(beta, phi)
        self._field_matrix = None
        self._alpha_matrix = None
        self._ifield_matrix = None
        self._phase_matrix = None
        self._transfer_matrix = None
        self._hzez_matrix = None
        self._rotation_matrix = None
        
    @property   
    def kd(self):
        return self._kd        
        
    @property   
    def epsilon(self):
        return self._epsilon

    @property   
    def angles(self):
        return self._angles 
    
    @property   
    def beta(self):
        return self._beta

    @property   
    def phi(self):
        return self._phi 
    
    def clear_eig(self):
        """Clears out temporary field matrices"""
        self._field_matrix = None
        self._ifield_matrix = None
        self._alpha_matrix = None
        self._phase_matrix = None
        
    def clear_transfer_matrix(self):
        """Clears transfer matrix"""
        self._transfer_matrix = None
        self._hzez_matrix = None
        
    def clear_all(self):
        """Clears all data."""
        self.clear_eig()
        self.clear_transfer_matrix()
       
    def compute_field_eig(self, out = None):
        """Computes eigen field matrices"""
        out = field_eig(self.epsilon,self.angles, self.beta, self.phi, out = out)
        self._alpha_matrix, self._field_matrix  = out
        return out
    
    def compute_field_eigi(self, out = None):
        """Computes eigen field matrices and inverse of the field matrix"""
        out = field_eigi(self.epsilon,self.angles, self.beta, self.phi, out = out)
        self._alpha_matrix, self._field_matrix, self._ifield_matrix = out
        return out
    
    def compute_he_eig(self, out = None):
        """Computes eigen field matrices and hzez matrix"""
        out = he_eig(self.epsilon,self.angles, self.beta, self.phi, out = None)
        self._alpha_matrix, self._field_matrix, self._hzez_matrix = out
        return out
    
    def compute_transfer_matrix(self, with_hzez = False, out = None):
        out = transfer_mat(self.kd, self.epsilon,self.angles, self.beta, self.phi, with_hzez = with_hzez, out = out)
        if with_hzez == True:
            self._transfer_matrix, self._hzez_matrix = out
        else:
            self._transfer_matrix = out
        return out
    
    def get_epsilon_tensor(self):
        """Returns epsilon tensor in the laboratory frame."""
        r = rotation_matrix(self.angles)
        if self.epsilon.shape[-1] == 3:
            return rotate_diagonal_tensor(r,self.epsilon)  
        else:
            return rotate_tensor(r,self.epsilon)  
    
    def get_transfer_matrix(self, copy = False):
        if self._transfer_matrix is None:
            if self._field_matrix is None:
                self._transfer_matrix = transfer_mat(self.kd, self.epsilon,self.angles, self.beta, self.phi)
            else:
                fmat = self.get_field_matrix()
                pmat = self.get_phase_matrix()
                ifmat = self.get_ifield_matrix()
                self._transfer_matrix = dotmdm(fmat,pmat,ifmat)
        out = self._transfer_matrix
        return out if copy == False else out.copy()
        
    def get_field_matrix(self, copy = False):      
        if self._field_matrix is None:
            self.compute_field_eig()
        out = self._field_matrix
        return out if copy == False else out.copy()
    
    def get_ifield_matrix(self, copy = False):      
        if self._ifield_matrix is None:
            if self._field_matrix is None:
                self.compute_field_eigi()
        out = self._ifield_matrix
        return out if copy == False else out.copy()            
    
    def get_phase_matrix(self, copy = False):
        if self._phase_matrix is None:
            if self._alpha_matrix is None:
                alpha = self.get_alpha_matrix()
                self._phase_matrix = phase_mat(alpha,self.kd)
        out = self._phase_matrix
        return out if copy == False else out.copy()
    
    def get_alpha_matrix(self, copy = False):
        if self._alpha_matrix is None:
            self.get_field_matrix() #will also set alpha matrix
        out = self._alpha_matrix
        return out if copy == False else out.copy()
        
    def get_hzez_matrix(self, copy = False):
        if self._hzez_matrix is None:
            self._hzez_matrix = hzez_mat(self.epsilon, self.angles, self.beta, self.phi)
        out = self._hzez_matrix
        return out if copy == False else out.copy()

    def transfer_fvec(self, field, out = None):
        mat = self.get_transfer_matrix()
        return dotmv(mat, field, out = out)
    
    def transfer_hevec(self, field, out = None):
        hzez = self.get_hzez_matrix()
        # the inner part is fvec
        f = field[...,1:-1].copy()
        if out is None:
            f = self.transfer_fvec(f)
            return hevec(f, hzez)
        else:
            #take the inner part and stor into inner part
            self.transfer_fvec(f, out = out[...,1:-1])
            # will only update first and last component of out.
            return hevec(out, hzez)
        
    def print_info(self):
        pass
 
from dtmm2.wave import mask2beta, mask2phi, eigenmask, k0

def _default_beam_mask(mask, k, aspect = 1):
    if mask is None:
        mask = eigenmask((128,128), k, aspect = aspect)
    return mask

def modal_beam_mat(k, epsilon = None,  mask = None, field = "m", aspect = 1.):
    k = np.asarray(k)
    mask = _default_beam_mask(mask, k, aspect)
    eps = np.asarray(epsilon, CDTYPE) if epsilon is not None else np.asarray(1., CDTYPE)
    
    #add axis
    eps = eps[...,None]
    betas = mask2beta(mask,k,aspect)
    phis = mask2phi(mask,k,aspect)
    
    if k.ndim == 0:
        return beam_mat(eps,betas,phis, field = field)
    else:
        out = (beam_mat(eps,betas[i],phis[i],field = field) for i in range(len(k)))
        return tuple(out)

def modal_field_eig(k, epsilon = None, angles = None, normalize = True, mask = None):
    k = np.asarray(k)
    epsilon, angles = _default_epsv_epsa(epsilon, angles)
    mask = _default_beam_mask(mask, k)
    
    #add axis
    epsilon = epsilon[...,None,:]
    angles = angles[...,None,:]
    
    betas = mask2beta(mask,k)
    phis = mask2phi(mask,k)
    
    if k.ndim == 0:
        return field_eig(epsilon,angles, betas, phis, normalize = normalize)
    else:
        out = (field_eig(epsilon,angles, betas[i], phis[i], normalize = normalize) for i in range(len(k0)))
        return tuple(out)

def modal_field_mat(k, epsilon = None, angles = None, normalize = True, mask = None):
    k = np.asarray(k)
    epsilon, angles = _default_epsv_epsa(epsilon, angles)
    mask = _default_beam_mask(mask, k)
    
    #add axis
    epsilon = epsilon[...,None,:]
    angles = angles[...,None,:]
    
    betas = mask2beta(mask,k)
    phis = mask2phi(mask,k)
    
    if k.ndim == 0:
        return field_mat(epsilon,angles, betas, phis, normalize = normalize)
    else:
        out = (field_mat(epsilon,angles, betas[i], phis[i], normalize = normalize) for i in range(len(k0)))
        return tuple(out)
    
def beam_eig(k, epsilon = None, angles = None, normalize = True, mask = None):
    deprecation("Deprecated. Use modal_field_mat instead")
    return modal_field_eig(k, epsilon, angles, normalize, mask)

def modal_transfer_mat(k, d, epsilon = None, angles = None, mask = None, with_hzez = False):
    k = np.asarray(k)
    epsilon, angles = _default_epsv_epsa(epsilon, angles)
    mask = _default_beam_mask(mask, k)
    
    #add axis
    epsilon = epsilon[...,None,:]
    angles = angles[...,None,:]
    
    betas = mask2beta(mask,k)
    phis = mask2phi(mask,k)
    
    if k.ndim == 0:
        return transfer_mat(k*d,epsilon,angles, betas, phis, with_hzez = with_hzez)
    else:
        out = (transfer_mat(k[i]*d,epsilon,angles, betas[i], phis[i], with_hzez = with_hzez) for i in range(len(k0)))
        return tuple(out)

def beam_transfer_mat(k, d, epsilon = None, angles = None, mask = None, with_hzez = False):
    deprecation("Deprecated. Use modal_transfer_mat instead")
    return modal_transfer_mat(k, d, epsilon, angles, mask, with_hzez)

def transfer_beam(mat, beam, hzez = None, out = None):
    if isinstance(mat, tuple):
        if hzez is None:
            hzez = [None] * len(mat)
        if out is None:
            out = [None] * len(mat)
        return tuple((transfer_beam(m,b,h,o) for m, b, h, o in zip(mat,beam,hzez,out)) )
    return transfer_wave(mat, beam, hzez, out)
    
def transfer_wave(mat, wave, hzez = None, out = None):
    wave = np.asarray(wave)
    if wave.shape[-1] == 4:
        return dotmv(mat, wave, out)
    elif wave.shape[-1] == 6:
        fvec = wave[...,1:-1]
        if out is None:
            fvec = transfer_wave(fvec)
            return hevec(fvec, hzez)
        else:
            #take the inner part and stor into inner part
            transfer_wave(fvec, out = out[...,1:-1])
            # will only update first and last component of out.
            return hevec(out, hzez)
    
def transfer_mat(kd, epsilon = None, angles = None, beta = None, phi = None, with_hzez = False, out = None):
    """Computes berreman transfer matrix of a single layer M=F.P.Fi,
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    kd : float
        The kd phase value (layer thickness times wavenumber in vacuum). For 
        forward transform it is a positive number, while for backward trasnform
        set it as a negative number.
    epsilon : (...,3) array or (..., 6) array, optional
        Dielectric tensor eigenvalues array (defaults to unity), or full dielectric
        tensor array (diagonal and off-diagonal values)
    angles : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    cfact : float, optional
        Coherence factor, only used in combination with `4x4_2` method.
    method : str
        One of 4x4 (4x4 berreman - trasnmittance + reflectance), 
        2x2 (2x2 jones - transmittance only), 
        4x4_1 (4x4, single reflections - transmittance only),
        2x2_1 (2x2, single reflections - transmittance only) 
        4x4_2 (4x4, partially coherent reflections - transmittance only) 
    fmatin : ndarray, optional
        Used in combination with 2x2_1 method. It specifies the field matrix of 
        the input media in order to compute fresnel reflections. If not provided 
        it reverts to 2x2 with no reflections.
        
    out : ndarray, optional
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the layer.
    """

    epsilon, angles = _default_epsv_epsa(epsilon, angles)
    beta, phi = _default_beta_phi(beta, phi)
    
    if with_hzez == True:
        if out is None:
            alpha,fmat,hzez = he_eig(epsilon, angles, beta, phi)
        else:
            alpha = np.empty(out.shape[:-1], out.dtype)
            _out1,_out2 = out
            _out = alpha, _out1, _out2 
            alpha,fmat,hzez = he_eig(epsilon, angles, beta, phi, out = _out)
        fmati = inv(fmat)
    else:
        alpha,fmat,fmati = field_eigi(epsilon, angles, beta, phi)

    #TODO: optimize, store to alpha if possible (broadcastable)
    pmat = phase_mat(alpha,kd, out = alpha)
    #pmat = phase_mat(alpha,kd)
    
    if with_hzez == True:
        if out is None:
            return dotmdm(fmat,pmat, fmati, out = fmat), hzez 
        else:
            return dotmdm(fmat,pmat, fmati, out = _out1), hzez
    else:
        #we can overwrite fmat during dotmdm, but not fmati! 
        out = fmat if out is None else out
        return dotmdm(fmat,pmat, fmati, out = out)
        
def hzez_mat(epsilon = None, angles = None, beta = None, phi = None, out = None):
    """Constructs a 2x4 matrix for conversion from field vector (Ex, Hy, Ey, Hx) 
    to Hz,Ez vector"""
    
    epsilon, angles = _default_epsv_epsa(epsilon, angles)
    beta, phi = _default_beta_phi(beta, phi)
    # we mast rotate back to reference frame, so -phi (equivalent to transpose rotation matrix)
    rv = rotation_vector2(phi) 
    return _he_vec(beta,phi,rv,epsilon,angles,_dummy_array24, out = out)
        
    
def layer_mat(kd, epsv = None, epsa = None, beta = 0,phi = 0, cfact = 0.1, method = "4x4", fmatin = None, retfmat = False, out = None):
    """Computes characteristic matrix of a single layer M=F.P.Fi,
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    kd : float
        The kd phase value (layer thickness times wavenumber in vacuum).
    epsv : (...,3) array or (..., 6) array, optional
        Dielectric tensor eigenvalues array (defaults to unity), or full dielectric
        tensor array (diagonal and off-diagonal values)
    epsa : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    cfact : float, optional
        Coherence factor, only used in combination with `4x4_2` method.
    method : str
        One of 4x4 (4x4 berreman - trasnmittance + reflectance), 
        2x2 (2x2 jones - transmittance only), 
        4x4_1 (4x4, single reflections - transmittance only),
        2x2_1 (2x2, single reflections - transmittance only) 
        4x4_2 (4x4, partially coherent reflections - transmittance only) 
    fmatin : ndarray, optional
        Used in combination with 2x2_1 method. It specifies the field matrix of 
        the input media in order to compute fresnel reflections. If not provided 
        it reverts to 2x2 with no reflections.
        
    out : ndarray, optional
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the layer.
    """    
    if method in ("2x2","2x2_1"):
        alpha,fmat = field_eig(epsv,epsa, beta, phi)
        f = E_mat(fmat, mode = +1, copy = False)
        if fmatin is not None and method == "2x2_1":
            fi = Eti_mat(fmatin, fmat, mode = +1)
        else:
            fi = inv(f)
        pmat = phase_mat(alpha[...,::2],kd)

    elif method in ("4x4","4x4_1","4x4_2"):
        alpha,fmat,fi = field_eigi(epsv,epsa, beta, phi)
        fmat = normalize_f(fmat)
        fi = inv(fmat)
        f = fmat
            
        if method == "4x4_2":
            np.add(alpha[...,1::2],alpha[...,1::2].real*2*1j*cfact, out = alpha[...,1::2])

        pmat = phase_mat(alpha,-kd)
        if method == "4x4_1":
            pmat[...,1::2] = 0.
    else:
        raise ValueError("Unknown method!")
        
    out = dotmdm(f,pmat,fi,out = out) 
    
    if retfmat == False:
        return out   
    else:
        return fmat, out 

def stack_mat(kd,epsv,epsa, beta = 0, phi = 0, cfact = 0.01, method = "4x4", out = None):
    """Computes a stack characteristic matrix M = M_1.M_2....M_n if method is
    4x4, 4x2(2x4) and a characteristic matrix M = M_n...M_2.M_1 if method is
    2x2.
    
    Note that this function calls :func:`layer_mat`, so numpy broadcasting 
    rules apply to kd[i], epsv[i], epsa[i], beta and phi. 
    
    Parameters
    ----------
    kd : array of floats
        A sequence of phase values (layer thickness times wavenumber in vacuum).
        len(kd) must match len(epsv) and len(epsa).
    epsv : (...,3) array, optional
        Dielectric tensor eigenvalues array (defaults to unity).
    epsa : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    cfact : float
        Coherence factor, only used in combination with `4x4_r` and `4x4_2` methods.
    method : str
        One of 
        4x4 (4x4 berreman), 
        2x2 (2x2 jones), 
        4x4_1 (4x4, single reflections - transmittance only), 
        2x2_1 (2x2, single reflections), 
        4x4_2 (4x4, incoherent reflections - transmittance only) 
    out : ndarray, optional
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the stack.
    """
    t0 = time.time()
    mat = None
    tmat = None
    fmat = None
    n = len(kd)

    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building stack matrix.")
    for pi,i in enumerate(range(n)):
        print_progress(pi,n) 
        if method == "2x2_1":
            fmat, mat = layer_mat(kd[i],epsv[i],epsa[i],beta = beta, phi = phi, cfact = cfact, method = method, fmatin = fmat, out = mat, retfmat = True)
        else:
            mat = layer_mat(kd[i],epsv[i],epsa[i],beta = beta, phi = phi, cfact = cfact, method = method, out = mat)

        if pi == 0:
            if out is None:
                out = mat.copy()
            else:
                out[...] = mat
        else:
            if tmat is not None:
                dotmm(tmat,mat,mat)
            if method.startswith("2x2"):
                dotmm(mat,out,out)
            else:
                dotmm(out,mat,out)
    print_progress(n,n) 
    t = time.time()-t0
    if verbose_level >1:
        print("     Done in {:.2f} seconds!".format(t))  
    return out 

_m1 = np.array([[1.,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1],
         [0,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [0,0,0,0]])

_m0 = np.array([[1.,0,0,0,0,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,0,0,0,0,1]])    
    
def system_mat(cmat = None,fmatin = None, fmatout = None, fmatini = None, fmatouti = None, transfer = "backward", out = None):
    """Computes a system matrix from a characteristic matrix. 
    For backward transfer it does Fin-1.C.Fout
    For forward transfer it does Fout-1.C.Fin
    
    Parameters
    ----------
    cmat : (...,4,4) array
        Characteristic matrix.
    fmatin : (...,4,4) array
        Input field matrix array.
    fmatout : (...,4,4) array
        Output field matrix array.
    fmatini : (...,4,4) array
        Inverse of the input field matrix array.
    out : ndarray, optional
        Output array where results are written.
    """
    if transfer == "forward":
        #matrices roles are reversed
        fmatin, fmatini, fmatout, fmatouti = fmatout, fmatouti, fmatin ,fmatini
    elif transfer != "backward":
        raise ValueError("Unknown transfer direction.")
    
    if fmatini is None:
        if fmatin is None:
            fmatin = f_iso()
        fmatini = inv(fmatin)
    if fmatout is None:
        if fmatouti is not None:
            fmatout = inv(fmatouti)
        else:
            fmatout = fmatin
    if cmat is not None:
        if cmat.shape[-1] == 8:
            dotmm(fmatini,cmat[...,0:4,0:4],out = cmat[...,0:4,0:4])
            dotmm(cmat[...,0:4,0:4],fmatout,out = cmat[...,0:4,0:4])
            dotmm(fmatini,cmat[...,4:8,4:8],out = cmat[...,4:8,4:8])
            dotmm(cmat[...,4:8,4:8],fmatout,out = cmat[...,4:8,4:8])
            if out is None:
                out = np.empty_like(cmat[...,0:4,0:4])
            
            out[...] = dotmm(_m0,dotmm(cmat,_m1))
            return out

        else:
            out = dotmm(fmatini,cmat,out = out)
            return dotmm(out,fmatout,out = out)  
    else:
        return dotmm(fmatini,fmatout,out = out)


def reflection_mat(smat, transfer = 'backward',out = None):
    """Computes a 4x4 reflection matrix.
    
    Parameters
    ----------
    smat : (...,4,4) array
        System matrix.
    transfer: str
        Either 'backward' or 'forward', specifying the direction in which
        the system matrix transfers the field vector.
    out : ndarray, optional
        Output array where results are written.
    """
    m1 = np.zeros_like(smat)
    m2 = np.zeros_like(smat)
    #fill diagonals
    if transfer == 'backward':
        for i in range(smat.shape[-1]//2):
            m1[...,i*2+1,i*2+1] = 1.
            m2[...,i*2,i*2] = -1.
        m1[...,:,0::2] = -smat[...,:,0::2]
        m2[...,:,1::2] = smat[...,:,1::2]

    elif transfer ==  'forward':
        for i in range(smat.shape[-1]//2):
            m1[...,i*2,i*2] = 1.
            m2[...,i*2+1,i*2+1] = -1.   
        m1[...,:,1::2] = -smat[...,:,1::2]
        m2[...,:,0::2] = smat[...,:,0::2]

    else:
        raise ValueError('Uknown transfer direction')
        
    m1 = inv(m1)
    out = dotmm(m1,m2, out = out)
    return out

def Si_mat(smat):
    m = np.zeros_like(smat)
    #fill diagonals
    for i in range(smat.shape[-1]//2):
        m[...,i*2+1,i*2+1] = 1.
    m[...,:,1::2] = -smat[...,:,0::2]
    m = inv(m)
    return m    

def fvec2E(fvec, fmat = None, fmati = None, mode = +1, inplace = False):
    """Converts field vector to E vector. If inplace == True, it also 
    makes fvec forward or backward propagating. 
    
    Parameters
    ----------
    fvec : (...,4,4) array
        Field vector array.
    fmat : (...,4,4) array
        Field matrix array.
    fmati : (...,4,4) array, optional
        The inverse of the field matrix. If not provided it is computed from `fmat`.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    inplace : bool, optional
    """
    mode = _mode_to_int(mode)
    if inplace == True:
        out  = fvec
    else:
        out = None
    if fmat is None:
        fmat = f_iso()
    pmat = projection_mat(fmat, fmati = fmati, mode = mode)
    return dotmv(pmat,fvec, out = out)[...,::2]
    
def E2fvec(evec, fmat = None, mode = +1, out = None):
    """Converts E vector to field vector
    
    Parameters
    ----------
    evec : (...,2) array
        E field vector array.
    fmat : (...,4,4) array
        Field matrix array.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    out : ndarray, optional
        Output array where results are written.
    """
    evec = np.asarray(evec)
    if fmat is None:
        fmat = f_iso()
    e2h = E2H_mat(fmat, mode = mode)
    hvec = dotmv(e2h, evec)
    if out is None:
        out = np.empty(shape = evec.shape[:-1] + (4,), dtype = hvec.dtype) #hvec is complex
    out[...,::2] = evec
    out[...,1::2] = hvec
    return out
    
# def transmit2x2(fvecin, cmat, fmatout = None, tmatin = None, tmatout = None, fvecout = None):
#     """Transmits field vector using 2x2 method.
    
#     This functions takes a field vector that describes the input field and
#     computes the output transmited field using the 2x2 characteristic matrix.
    
#     Parameters
#     ----------
#     fvecin : (...,4) array
#         Input field vector array. This function will update the input array  
#         with the calculated reflected field.
#         Characteristic matrix.
#     fmatout : (...,4,4) array
#         Output field matrix array.
#     tmatin : (...,2,2) array
#         The transmittance matrix from the input medium to the first layer.
#     tmatout : (...,2,2) array, optional
#         The transmittance matrix from the last layer to the output maedium.
#     fvecout : (...,4) array, optional
#         The ouptut field vector array. This function will update the output array 
#         with the calculated transmitted field.
    
#     """
#     b = np.broadcast(fvecin[...,0][...,None,None],cmat)
#     if fmatout is None:
#         fmatout = f_iso(1,0,0) 
#     if fvecout is not None:
#         fvecout[...] = 0 
#     else:   
#         fvecout = np.zeros(b.shape[:-2] + (4,), fvecin.dtype)
        
#     if tmatin is not None:
#         evec = dotmv(tmatin, fvecin[...,::2], out = fvecout[...,::2])
#     else:
#         evec = fvecin[...,::2]
#     eout = dotmv(cmat, evec, out = fvecout[...,::2])
#     if tmatout is not None:
#         eout = dotmv(tmatout, eout, out = fvecout[...,::2])
#     e2h = E2H_mat(fmatout, mode = +1)
#     hout = dotmv(e2h, eout, out = fvecout[...,1::2])
#     return fvecout

def avec2svec(avecin, avecout = None, out = None):
    """Converts an amplitude vector(s) to a scattering vector.
    
    Parameters
    ----------
    avecin : ndarray
        The amplitude vector on the input size
    avecout : ndarray, optional
        The amplitude vector on the output size. Defaults to zeros.
    out : ndarray, optional
        The output array.
    """
    if out is None:
        out = np.empty_like(avecin)
    out[...,0::2] = avecin[...,0::2]
    if avecout is not None:
        out[...,1::2] = avecout[...,1::2]
    else:
        out[...,1::2] = 0.
    return out

def svec2avec(svec, avecin = None, avecout = None):
    """Converts a scattering vector to input,output amplitude vectors
    
    Parameters
    ----------
    svec : ndarray
        A scattering vector
    avecin : ndarray, optional
        Amplitude vector on the input size. The refelcted part of the smat will
        be set to this vector (the transmitted part will remain intact).
    avecout : ndarray, optional
        Amplitude vector on the output size. The transmitted part of the smat will
        be set to this vector (the reflected part will remain intact).
    
    Returns
    -------
    out : ndarray, ndarray
        A tuple of input and ouptut amplitude vectors.
    
    """
    if avecin is None:
        avecin = np.zeros_like(svec)
    if avecout is None:
        avecout = np.zeros_like(svec)
    avecin[...,1::2] = svec[...,1::2]
    avecout[...,0::2] = svec[...,0::2]
    return avecin, avecout

def reflect_svec(svec, rmat, out = None):
    b = np.broadcast(svec[..., None],rmat[...,0:4,0:4])
    
    if svec.shape != b.shape[:-1]:
        raise ValueError("Input field vector should have shape of {}".format(b.shape[:-1]))
    
    svec = np.broadcast_to(svec, b.shape[:-1])
    return dotmv(rmat, svec, out = out)

def reflect_avec(avecin, rmat, avecout = None, gvecin = None, gvecout = None, out_type = "full"):
    """Reflects/Transmits amplitude vector using 4x4 reflection matrix.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
   
    Parameters
    ----------
    avecin : (...,4) array
        Input amplitude vector array. This function will update the input array  
        with the calculated reflected field
    rmat : (...,4,4) array
        Reflection matrix.
    avecout : (...,4) array, optional
        The ouptut amplitude vector array. This function will update the output array 
        with the calculated transmitted field.
    gvecin : (...,4) array, optional
        Specifies the amplitude gain vector at the input side. 
    gvecout : (...,4) array, optional
        Specifies the amplitude gain vector at the output side. 
    
    """
    svec = avec2svec(avecin, avecout)
    svec = reflect_svec(svec, rmat)
    
    if out_type == "full":
        avecin, avecout = svec2avec(svec, avecin, avecout)
    else:
        avecin, avecout = svec2avec(svec)

    
    b = np.broadcast(avecin[..., None],rmat[...,0:4,0:4])
    
    if avecin.shape != b.shape[:-1]:
        raise ValueError("Input field vector should have shape of {}".format(b.shape[:-1]))
     
    avecin = np.broadcast_to(avecin, )

        
    avec = avecin
    bvec = avecout
    
    a = np.zeros(b.shape[:-1], avec.dtype)
    a[...,0::2] = avec[...,0::2]
    avec = a.copy()#so that it broadcasts

    if gvecin is not None:
        a[...,0::2] -= gvecin[...,0::2]
        
    if avecout is not None:
        a[...,1::2] = bvec[...,1::2] 
    else:
        bvec = np.zeros_like(avec)
        
    if gvecout is not None:
        a[...,1::2] -= gvecout[...,1::2]
        

    out = dotmv(rmat,a, out = a)
    
    if gvecin is not None:
        out[...,1::2] += gvecin[...,1::2]

    if gvecout is not None:
        out[...,0::2] += gvecout[...,0::2]
        
    return out
        
    avecin[...,1::2] = out[...,1::2]
    bvec[...,::2] = out[...,::2]
        
    return bvec

def reflect_avec(avecin, rmat, avecout = None, gvecin = None, gvecout = None):
    """Reflects/Transmits amplitude vector using 4x4 reflection matrix.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
   
    Parameters
    ----------
    avecin : (...,4) array
        Input amplitude vector array. This function will update the input array  
        with the calculated reflected field
    rmat : (...,4,4) array
        Reflection matrix.
    avecout : (...,4) array, optional
        The ouptut amplitude vector array. This function will update the output array 
        with the calculated transmitted field.
    gvecin : (...,4) array, optional
        Specifies the amplitude gain vector at the input side. 
    gvecout : (...,4) array, optional
        Specifies the amplitude gain vector at the output side. 
    """
    b = np.broadcast(avecin[..., None],rmat[...,0:4,0:4])
    
    if avecin.shape != b.shape[:-1]:
        raise ValueError("Input field vector should have shape of {}".format(b.shape[:-1]))
      
    
    out = transmit_avec(avecin,rmat,avecout,gvecin,gvecout)
    
    avecin[...,1::2] = out[...,1::2]
    if avecout is None:
        avecout = np.zeros_like(out)
    avecout[...,::2] = out[...,::2]
        
    return avecout
    

def reflect_fvec(fvecin, rmat, fmatin = None, fmatout = None, fmatini = None, fmatouti = None, fvecout = None, gvecin = None, gvecout = None):
    """Reflects/Transmits field vector using 4x4 reflection matrix.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
   
    Parameters
    ----------
    fvecin : (...,4) array
        Input field vector array. This function will update the input array  
        with the calculated reflected field
    rmat : (...,4,4) array
        Reflection matrix.
    fmatin : (...,4,4) array
        Input field matrix array.
    fmatout : (...,4,4) array
        Output field matrix array.
    fmatini : (...,4,4) array
        Inverse of the input field matrix array.
    fmatouti : (...,4,4) array, optional
        Inverse of the output field matrix array. If not provided, it is computed
        from `fmatout`.
    fvecout : (...,4) array, optional
        The ouptut field vector array. This function will update the output array 
        with the calculated transmitted field.
    gvecin : (...,4) array, optional
        Specifies the field gain vector at the input side. 
    gvecout : (...,4) array, optional
        Specifies the field gain vector at the output side. 
    """
    b = np.broadcast(fvecin[..., None],rmat[...,0:4,0:4], fmatin, fmatout)
    
    if fvecin.shape != b.shape[:-1]:
        raise ValueError("Input field vector should have shape of {}".format(b.shape[:-1]))
    if fvecout is not None and fvecout.shape != b.shape[:-1]:
        raise ValueError("Output field vector should have shape of {}".format(b.shape[:-1]))
    if fmatini is None:
        if fmatin is None:
            fmatin = f_iso()
        fmatini = inv(fmatin)
    if fmatin is None:
        fmatin = inv(fmatini)
    if fmatouti is None:
        if fmatout is None:
            fmatout = fmatin
            fmatouti = fmatini
        else:
            fmatouti = inv(fmatout)
    if fmatout is None:
        fmatout = inv(fmatouti)
        
    avecin = dotmv(fmatini,fvecin)
    
    if gvecin is not None:
        gvecin = dotmv(fmatini,gvecin)

    if gvecout is not None:
        gvecout = dotmv(fmatouti,gvecout)
        
    if fvecout is not None:
        avecout = dotmv(fmatouti,fvecout, out = fvecout)
    else:
        avecout = None
        
    avecout = reflect_avec(avecin,rmat,avecout,gvecin,gvecout)
        
    dotmv(fmatin,avecin, out = fvecin)    
    return dotmv(fmatout,avecout, out = fvecout)

def reflect(fvecin, rmat, fmatin = None, fmatout = None, fmatini = None, fmatouti = None, fvecout = None, gvecin = None, gvecout = None):
    """Reflects/Transmits field vector using 4x4 reflection matrix.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
   
    Parameters
    ----------
    fvecin : (...,4) array
        Input field vector array. This function will update the input array  
        with the calculated reflected field
    rmat : (...,4,4) array
        Reflection matrix.
    fmatin : (...,4,4) array
        Input field matrix array.
    fmatout : (...,4,4) array
        Output field matrix array.
    fmatini : (...,4,4) array
        Inverse of the input field matrix array.
    fmatouti : (...,4,4) array, optional
        Inverse of the output field matrix array. If not provided, it is computed
        from `fmatout`.
    fvecout : (...,4) array, optional
        The ouptut field vector array. This function will update the output array 
        with the calculated transmitted field.
    gvecin : (...,4) array, optional
        Specifies the field gain vector at the input side. 
    gvecout : (...,4) array, optional
        Specifies the field gain vector at the output side. 
    """
    b = np.broadcast(fvecin[..., None],rmat[...,0:4,0:4], fmatin, fmatout)
    
    if fvecin.shape != b.shape[:-1]:
        raise ValueError("Input field vector should have shape of {}".format(b.shape[:-1]))
    if fvecout is not None and fvecout.shape != b.shape[:-1]:
        raise ValueError("Output field vector should have shape of {}".format(b.shape[:-1]))
    if fmatini is None:
        if fmatin is None:
            fmatin = f_iso()
        fmatini = inv(fmatin)
    if fmatin is None:
        fmatin = inv(fmatini)
    if fmatouti is None:
        if fmatout is None:
            fmatout = fmatin
            fmatouti = fmatini
        else:
            fmatouti = inv(fmatout)
    if fmatout is None:
        fmatout = inv(fmatouti)
        
    avec = dotmv(fmatini,fvecin)
    
    
    a = np.zeros(b.shape[:-1], avec.dtype)
    a[...,0::2] = avec[...,0::2]
    avec = a.copy()#so that it broadcasts

    if gvecin is not None:
        gvecin = dotmv(fmatini,gvecin)

    if gvecout is not None:
        gvecout = dotmv(fmatouti,gvecout)

    if gvecin is not None:
        a[...,0::2] -= gvecin[...,0::2]
        

    if fvecout is not None:
        bvec = dotmv(fmatouti,fvecout)
        a[...,1::2] = bvec[...,1::2] 
    else:
        bvec = np.zeros_like(avec)
        
    if gvecout is not None:
        a[...,1::2] -= gvecout[...,1::2]
        

    out = dotmv(rmat,a, out = fvecout)
    
    avec[...,1::2] = out[...,1::2]
    bvec[...,::2] = out[...,::2]
    
    if gvecin is not None:
        avec[...,1::2] += gvecin[...,1::2]

    if gvecout is not None:
        bvec[...,0::2] += gvecout[...,0::2]
        
    dotmv(fmatin,avec,out = fvecin)    
    return dotmv(fmatout,bvec,out = out)


def transmit2x2(fvecin, cmat, fmatin = None, fmatout = None, fmatini = None, fmatouti = None, tmatin = None, tmatout = None, fvecout = None):
    if fmatini is None:
        if fmatin is None:
            fmatin = f_iso()
        fmatini = inv(fmatin)
    if fmatin is None:
        fmatin = inv(fmatini)
    if fmatouti is None:
        if fmatout is None:
            fmatout = fmatin
            fmatouti = fmatini
        else:
            fmatouti = inv(fmatout)
    if fmatout is None:
        fmatout = inv(fmatouti)

    Ein = fvec2E(fvecin, fmati = fmatini)
    if tmatin is not None:
        Ein = dotmv(tmatin, Ein, out = Ein)

    Eout = dotmv(cmat,Ein, out = Ein)
    if tmatout is not None:
        Eout = dotmv(tmatout, Eout, out = Eout)    
    
    fvecout = E2fvec(Eout, fmat = fmatout, out = fvecout)
    return fvecout

def transmit4x4(fvecin, cmat = None, fmatin = None, fmatout = None, fmatini = None, fmatouti = None, fvecout = None):
    """Reflects/Transmits field vector using 4x4 method.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
   
    Parameters
    ----------
    fvecin : (...,4) array
        Input field vector array. This function will update the input array  
        with the calculated reflected field
    cmat : (...,4,4) array
        Characteristic matrix.
    fmatin : (...,4,4) array
        Input field matrix array.
    fmatout : (...,4,4) array
        Output field matrix array.
    fmatini : (...,4,4) array
        Inverse of the input field matrix array.
    fmatouti : (...,4,4) array, optional
        Inverse of the output field matrix array. If not provided, it is computed
        from `fmatout`.
    fvecout : (...,4) array, optional
        The ouptut field vector array. This function will update the output array 
        with the calculated transmitted field.
    """
    b = np.broadcast(fvecin[..., None],cmat[...,0:4,0:4], fmatin, fmatout)
    
    if fvecin.shape != b.shape[:-1]:
        raise ValueError("Input field vector should have shape of {}".format(b.shape[:-1]))
    if fvecout is not None and fvecout.shape != b.shape[:-1]:
        raise ValueError("Output field vector should have shape of {}".format(b.shape[:-1]))
    if fmatini is None:
        if fmatin is None:
            fmatin = f_iso()
        fmatini = inv(fmatin)
    if fmatin is None:
        fmatin = inv(fmatini)
    if fmatouti is None:
        if fmatout is None:
            fmatout = fmatin
            fmatouti = fmatini
        else:
            fmatouti = inv(fmatout)
    if fmatout is None:
        fmatout = inv(fmatouti)
        
    smat = system_mat(cmat = cmat,fmatini = fmatini, fmatout = fmatout)
     
    avec = dotmv(fmatini,fvecin)
    
    
    a = np.zeros(b.shape[:-1], avec.dtype)
    a[...,0::2] = avec[...,0::2]
    avec = a.copy()#so that it broadcasts

    if fvecout is not None:
        bvec = dotmv(fmatouti,fvecout)
        a[...,1::2] = bvec[...,1::2] 
    else:
        bvec = np.zeros_like(avec)

    r = reflection_mat(smat)
    out = dotmv(r,a, out = fvecout)
    
    avec[...,1::2] = out[...,1::2]
    bvec[...,::2] = out[...,::2]
        
    dotmv(fmatin,avec,out = fvecin)    
    return dotmv(fmatout,bvec,out = out)


def transmit(fvecin, cmat = None, fmatin = None, fmatout = None, fmatini = None, fmatouti = None, tmatin = None, tmatout = None, fvecout = None):
    """Reflects/Transmits field vector using 4x4 method.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
   
    Parameters
    ----------
    fvecin : (...,4) array
        Input field vector array. This function will update the input array  
        with the calculated reflected field
    cmat : (...,4,4) array
        Characteristic matrix.
    fmatin : (...,4,4) array
        Input field matrix array.
    fmatout : (...,4,4) array
        Output field matrix array.
    fmatini : (...,4,4) array
        Inverse of the input field matrix array.
    fmatouti : (...,4,4) array, optional
        Inverse of the output field matrix array. If not provided, it is computed
        from `fmatout`.
    tmatin : (...,2,2) array
        Input transmission matrix for 2x2 method only.
    tmatout : (...,2,2) array
        Output transmission matrix for 2x2 method only.        
    fvecout : (...,4) array, optional
        The ouptut field vector array. This function will update the output array 
        with the calculated transmitted field.
    """
    if cmat is None:
        raise ValueError("`cmat` is a required argument.")

    if cmat.shape[-1] == 2:
        # a 2x2 problem
        return transmit2x2(fvecin,cmat, fmatin = fmatin, fmatout = fmatout, fmatini = fmatini, fmatouti = fmatouti, tmatin = tmatin, tmatout = tmatout, fvecout = fvecout )
    else:
        # must be 4x4 
        return transmit4x4(fvecin,cmat, fmatin = fmatin, fmatout = fmatout, fmatini = fmatini, fmatouti = fmatouti, fvecout = fvecout )
    


def transfer4x4(fvecin, kd, epsv, epsa,  beta = 0., phi = 0., nin = 1., nout = 1., 
             method = "4x4", reflect_in = False, reflect_out = False, fvecout = None):
    """Tranfers field using 4x4 method.
    
    see also :func:`transfer`
    
    Parameters
    ----------
    fvecin : (...,4) array
        Input field vector array. This function will update the input array  
        with the calculated reflected field
    kd : array of floats
        A sequence of phase values (layer thickness times wavenumber in vacuum).
        len(kd) must match len(epsv) and len(epsa).
    epsv : (...,3) array, optional
        Dielectric tensor eigenvalues array (defaults to unity).
    epsa : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    nin : float
        Input layer refractive index.
    nout : float
        Output layer refractive index.
    method : str
        Any of 4x4, 4x4_1, 4x4_2.
    fvecout : (...,4) array, optional
        The ouptut field vector array. This function will update the output array 
        with the calculated transmitted field.
    """
    if reflect_in == True or reflect_out == True:
        raise ValueError("options reflect_in and reflect_out are not yet supported")
    
    if method not in ("4x4", "4x4_1","4x4_2"):
        raise ValueError("Unknown method '{}'!".format(method))
           
    fveci = fvecin
    fvecf = fvecout
    
    fmatin = f_iso(n = nin, beta  = beta, phi = phi)
    fmatout = f_iso(n = nout, beta  = beta, phi = phi)
        
    if reflect_in == True:
        
        #make fresnel reflection of the input (forward propagating) field
        fin = f_iso(n = nin, beta  = beta, phi = phi)
        alpha,fmatin = field_eig(beta = beta, phi = phi, epsilon = epsv[0], angles = epsa[0])
        
        t = T_mat(fmatin = fin, fmatout = fmatin, mode = +1)
        fveci = dotmv(t,fveci, out = fveci)
        
#        tmati = t_mat(fin,fmatin, mode = +1)
#        evec0 = fvec2E(fveci, fin, mode = +1)
#        evec = dotmv(tmati,evec0)
#        fveci = E2fvec(evec,fmatin, mode = +1)
#        fvecin = E2fvec(evec0,fin, mode = +1, out = fvecin)
    else:
        fmatin = f_iso(n = nin, beta  = beta, phi = phi)
      
    if reflect_out == True:
        #make fresnel reflection of the output (backward propagating) field
        alpha,fmatout = field_eig(beta = beta, phi = phi, epsilon = epsv[-1], angles = epsa[-1])
        fout = f_iso(n = nout, beta  = beta, phi = phi)
        if fvecf is not None:
            t = T_mat(fmatin = fout, fmatout = fmatout, mode = -1)
            fvecf = dotmv(t,fvecf)            
            
#            tmatf = t_mat(fmatout,fout, mode = -1)
#            evec0 = fvec2E(fvecf, fout, mode = -1)
#            evec = dotmv(tmatf,evec) 
#            fvecf = E2fvec(evec,fmatout, mode = -1)
#            fvecout = E2fvec(evec0,fout, mode = -1, out = fvecout)
    else:
        fmatout = f_iso(n = nout, beta  = beta, phi = phi)
    
    cmat = stack_mat(kd, epsv, epsa, method = method)
    smat = system_mat(cmat, fmatin, fmatout)
    rmat = reflection_mat(smat)
    fvecf = reflect(fveci, rmat = rmat, fmatin = fmatin, fmatout = fmatout, fvecout = fvecf)

    if reflect_in == True:
        #make fresnel reflection of the input (backward propagating) field
        t = T_mat(fmatin = fmatin, fmatout = fin, mode = -1)
        fveci = dotmv(t,fveci, out = fvecin)        
        
#        tmati = t_mat(fin,fmatin, mode = -1)
#        evec = fvec2E(fveci, mode = -1)
#        evec = dotmv(tmati,evec)
#        fveci = E2fvec(evec,fout, mode = -1) 
#        np.add(fvecin, fveci, out = fvecin)


    if reflect_out == True:
        #make fresnel reflection of the output (forward propagating) field
        
        t = T_mat(fmatin = fmatout, fmatout = fout, mode = 1)
        fvecf = dotmv(t,fvecf, out = fvecout)    
#        tmati = t_mat(fmatout,fout, mode = +1)
#        evec = fvec2E(fvecf, mode = +1)
#        evec = dotmv(tmati,evec)
#        fvecf = E2fvec(evec,fout, mode = +1) 
#        fvecf = np.add(fvecout, fvecf, out = fvecout)  

    return fvecf

def transfer2x2(evec, kd, epsv, epsa,  beta = 0., phi = 0., nin  = None, nout = None, 
             method = "2x2", out = None):
    """Tranfers E-vec using 2x2 scattering matrices.
    
    Parameters
    ----------
    evec : (...,2) array
        Input E-field vector array. This function will update the input array  
        with the calculated reflected field
    kd : array of floats
        A sequence of phase values (layer thickness times wavenumber in vacuum).
        len(kd) must match len(epsv) and len(epsa).
    epsv : (...,3) array, optional
        Dielectric tensor eigenvalues array (defaults to unity).
    epsa : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    nin : float
        Input layer refractive index.
    nout : float
        Output layer refractive index.
    method : str
        Any of 2x2, 2x2_1.
    out : (...,2) array, optional
        The ouptut field vector array. This function will update the output array 
        with the calculated transmitted field.
"""
    
    if method not in ("2x2", "2x2_1"):
        raise ValueError("Unknown method!")
        
    if nin is not None:
       fin = f_iso(n = nin, beta  = beta, phi = phi)
       alpha,fout = field_eig(beta = beta, phi = phi, epsilon = epsv[0], angles = epsa[0])
       tmat = t_mat(fin,fout)
       evec = dotmv(tmat,evec)
    
    cmat = stack_mat(kd, epsv, epsa, method = method)
    evec = dotmv(cmat,evec, out = out)
    
    if nout is not None:
       fout = f_iso(n = nout, beta  = beta, phi = phi)
       alpha,fin = field_eig(beta = beta, phi = phi, epsilon = epsv[-1], angles = epsa[-1])
       tmat = t_mat(fin,fout)
       dotmv(tmat,evec, out = evec)  
       
    return evec

def transfer(fvecin, kd, epsv, epsa,  beta = 0., phi = 0., nin = 1., nout = 1., 
             method = "2x2", reflect_in = False, reflect_out = False, fvecout = None):
    """Transfer input field vector through a layered material specified by the propagation
    constand k*d, eps tensor (epsv, epsa) and input and output isotropic media.
    
    Parameters
    ----------
    fvecin : (...,4) array
        Input field vector array. This function will update the input array  
        with the calculated reflected field
    kd : array of floats
        A sequence of phase values (layer thickness times wavenumber in vacuum).
        len(kd) must match len(epsv) and len(epsa).
    epsv : (...,3) array, optional
        Dielectric tensor eigenvalues array (defaults to unity).
    epsa : (...,3) array, optional
        Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    beta : float, optional
        The beta parameter of the field (defaults to 0.)
    phi : float, optional
        The phi parameter of the field (defaults to 0.)
    nin : float
        Input layer refractive index.
    nout : float
        Output layer refractive index.
    method : str
        Any of 4x4, 2x2, 2x2_1 or 4x4_1, 4x4_2, 4x4_r
    reflect_in : bool
        Defines how to treat reflections from the input media and the first layer.
        If specified it does an incoherent reflection from the first interface.
    reflect_out : bool
        Defines how to treat reflections from the last layer and the output media.
        If specified it does an incoherent reflection from the last interface.
    fvecout : (...,4) array, optional
        The ouptut field vector array. This function will update the output array 
        with the calculated transmitted field.                
    """
    
    if method.startswith("2x2"):
        fin = f_iso(n = nin, beta  = beta, phi = phi)
        fout = f_iso(n = nout, beta  = beta, phi = phi)
        evec =  fvec2E(fvecin, fmat = fin, mode = +1)
        nin = nin if reflect_in == True else None
        nout = nout if reflect_out == True else None
        evec = transfer2x2(evec, kd, epsv, epsa,  beta = beta, phi = phi, nin  = nin, nout = nout, 
             method = method)
        return E2fvec(evec,fout, mode = +1, out = fvecout) 
    else:
        return transfer4x4(fvecin, kd, epsv, epsa,  beta = beta, phi = phi, nin = nin, nout = nout, 
             method = method,  reflect_in = reflect_in, reflect_out = reflect_out, fvecout = fvecout)        
       
transfer1d = transfer
        
def avec(jones = (1,0), amplitude = 1., mode = +1, out = None):
    """Constructs amplitude vector.
    
    Numpy broadcasting rules apply for jones, and amplitude parameters
    
    Parameters
    ----------
    jones : jonesvec
        A jones vector, describing the polarization state of the field.
    amplitude : complex
        Amplitude of the field.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    out : ndarray, optional
        Output array where results are written.
        
    Returns
    -------
    avec : ndarray
        Amplitude vector of shape (4,).
        
    Examples
    --------
    
    X polarized light with amplitude = 1
    >>> avec()
    array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    
    X polarized light with amplitude 1 and y polarized light with amplitude 2.
    >>> b = avec(jones = ((1,0),(0,1)),amplitude = (1,2))
    >>> b[0]
    array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    >>> b[1]
    array([0.+0.j, 0.+0.j, 2.+0.j, 0.+0.j])
    """
    mode = _mode_to_int(mode)
    jones = np.asarray(jones)
    amplitude = np.asarray(amplitude)  
    c,s = jones[...,0], jones[...,1] 
    b = np.broadcast(c, amplitude)
    shape = b.shape + (4,)
    if out is None:
        out = np.empty(shape,CDTYPE)
    assert out.shape[-1] == 4
    if mode == +1:
        out[...,0] = c
        out[...,2] = s
        out[...,1] = 0.
        out[...,3] = 0.
    else:
        out[...,1] = c
        out[...,3] = s 
        out[...,0] = 0.
        out[...,2] = 0.

    out = np.multiply(out, amplitude[...,None] ,out = out)     
    return out

def fvec(fmat, jones = (1,0),  amplitude = 1., mode = +1, out = None):
    """Build field vector form a given polarization state, amplitude and mode.
    
    This function calls avec and then avec2fvec, see avec for details.
    
    Parameters
    ----------
    fmat : (...,4,4) array
        Field matrix array.
    jones : jonesvec
        A jones vector, describing the polarization state of the field.
    amplitude : complex
        Amplitude of the field.
    mode : int
        Either +1, for forward propagating mode, or -1 for negative propagating mode.
    out : ndarray, optional
        Output array where results are written.
    
    Returns
    -------
    fvec : ndarray
        Field vector of shape (...,4).
    
    Examples
    --------
    
    X polarized light traveling at beta = 0.4 and phi = 0.2 in medium with n = 1.5
    
    >>> fmat = f_iso(beta = 0.4, phi = 0.2, n = 1.5)
    >>> m = fvec(fmat, jones = jonesvec((1,0), phi = 0.2))
    
    This is equivalent to
    
    >>> a = avec(jones = jonesvec((1,0), phi = 0.2))
    >>> ma = avec2fvec(a,fmat)
    >>> np.allclose(ma,m)
    True
    """
    #a.shape != out.shape in general, so we are not using out argument here
    a = avec(jones, amplitude, mode)
    return avec2fvec(a, fmat, out = out)


def fvec2avec(fvec, fmat, normalize_fmat = True, out = None):
    """Converts field vector to amplitude vector
    
    Parameters
    ----------
    fvec : ndarray
        Input field vector
    fmat : ndarray
        Field matrix 
    normalize_fmat : bool, optional
        Setting this to false will not normalize the field matrix. In this case 
        user has to make sure that the normalization of the field matrix has 
        been performed prior to calling this function by calling normalize_f.
    out : ndarray, optional
        Output array
        
    Returns
    -------
    avec : ndarray
        Amplitude vector
    """
    if normalize_fmat == True:
        fmat = normalize_f(fmat)
    fmati = inv(fmat)
    return dotmv(fmati,fvec, out = out)

def avec2fvec(avec, fmat, normalize_fmat = True, out = None):
    """Converts amplitude vector to field vector
    
    Parameters
    ----------
    avec : ndarray
        Input amplitude vector
    fmat : ndarray
        Field matrix 
    normalize_fmat : bool, optional
        Setting this to false will not normalize the field matrix. In this case 
        user has to make sure that the normalization of the field matrix has 
        been performed prior to calling this function by calling normalize_f.
    out : ndarray, optional
        Output array
        
    Returns
    -------
    fvec : ndarray
        Field vector.
    """
    if normalize_fmat == True:
        fmat = normalize_f(fmat)
    return dotmv(fmat, avec, out = out)
    
__all__ = ["alphaf","alphaffi","phase_mat", "fvec", "avec", "fvec2avec",
           "avec2fvec","f_iso","ffi_iso","layer_mat","poynting","intensity",
           "transfer4x4","transfer","transfer2x2","reflect",
           "layer_mat","system_mat","stack_mat","EHz"]

if __name__ == "__main__":
    import doctest
    #doctest.testmod()