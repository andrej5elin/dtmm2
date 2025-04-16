"""
Numba optimized linear algebra functions for 4x4, 3x3 and 2x2 complex or real matrices.
"""

from dtmm2._linalg import DOTMV_DECL, DOTMM_DECL, DOTMD_DECL, DOTVV_DECL ,DOTMDM_DECL, INV_DECL, EIG_DECL, \
    CROSS_DECL, CDTYPE, FDTYPE,\
    NUMBA_PARALLEL, NFDTYPE, NCDTYPE, NUMBA_CACHE, NUMBA_FASTMATH, boolean, guvectorize, NUMBA_TARGET
    
from dtmm2._linalg import _cross, _eigvec0, _eigvec1, _sort_eigvec, _inv2x2,_inv4x4, _dotmm4,\
    _dotmd4, _dotmm3, _dotmm2, _dotmd2, _dotr2m, _dotmv4, _dotmv3, _dotmv24, _dotmv2,\
        _dotmf2, _dotmf4, _dotmdmf2, _dotmdmf4, _vnorm2, _dotvv4, _dotvv3, _dotvv2

import numpy as np

@guvectorize(CROSS_DECL,"(n),(n)->(n)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def cross(a,b,out):
    """A direct replacement for np.cross function"""
    assert a.shape[0] == 3
    _cross(a,b, out)

#--------- Eigenvalue functions

@guvectorize(EIG_DECL, '(m,m),()->(m),(m,m)', target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)   
def _tensor_eig(tensor, is_real, eig,vec):
    """Computes eigenvalues of a tensor using analytical noniterative algorithm
    adapted from A Robust Eigensolver for 3 × 3 Symmetric Matrices by 
    David Eberly @ Geometric Tools.
    
    Input array is overwritten.
    """
    
    #if norm of the offdiagonal elements is smaller than this, treat the
    #tensor as a diagonal tensor. This improves handling of real diagonal epsilon 
    #data treated as complex data, where the complex part introduces unwanted
    # rotations of the eigenframe, in particular when working with floats.
    normtol = 1e-12
        
    m1 = max(abs(tensor[0,0]),abs(tensor[1,1]))
    m2 = max(abs(tensor[2,2]),abs(tensor[0,1]))
    m3 = max(abs(tensor[0,2]),abs(tensor[1,2]))
    
    scale = max(max(m1,m2),m3)
    
    if scale == 0.:
        #zero matrix.. set eigenvalues to zero
        eig[0] = 0.
        eig[1] = 0.
        eig[2] = 0.
        vec[0] = (1,0,0)
        vec[1] = (0,1,0)
        vec[2] = (0,0,1)
    else:
        #precondition the matrix to avoid floating-point overflow
        scalem = 1 / scale
        a00 = tensor[0,0]*scalem
        a11 = tensor[1,1]*scalem
        a22 = tensor[2,2]*scalem
        a01 = tensor[0,1]*scalem
        a02 = tensor[0,2]*scalem
        a12 = tensor[1,2]*scalem
        
        q = (a00 + a11 + a22)/3
        
        b00 = a00 - q
        b11 = a11 - q
        b22 = a22 - q
            
        norm = a01**2 + a02**2 + a12**2
        
        if abs(norm) > normtol:
            p = ((b00**2 + b11**2 +b22**2 + 2 * norm)/6)**0.5
        
            c00 = b11 * b22 - a12 * a12
            c01 = a01 * b22 - a12 * a02
            c02 = a01 * a12 - b11 * a02
            
            half_det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p * 2)
            
            if is_real[0] == True:
                #for real data, half_det is between -1 and 1, to avoid possible rounding error, clamp it, just to make sure
                half_det = min(max(half_det.real, -1),1)
                
            phi = np.arccos(half_det)/3.
            
            #this number is closest to 2*np.pi/3 in float or double
            twoThirdsPi = NFDTYPE(2.09439510239319549)
            
            eig0 = np.cos(phi + twoThirdsPi) * 2
            eig2 = np.cos(phi) * 2
            eig1 = -(eig0 + eig2)
            
            #eigenvalues sorted by increasing 
            eig0 = (eig0 * p + q) 
            eig1 = (eig1 * p + q) 
            eig2 = (eig2 * p + q) 
            
            #now compute the eigenvectors
            if half_det.real >= 0:
                _eigvec0(a00,a11,a22,a01,a02,a12,eig2,vec[2], vec[1],vec[0], eig)    
                _eigvec1(a00,a11,a22,a01,a02,a12,vec[2],eig1,vec[1],vec[0],eig)
                _cross(vec[1],vec[2],vec[0])
            else:
                _eigvec0(a00,a11,a22,a01,a02,a12,eig0,vec[0],vec[1],vec[2],eig) 
                _eigvec1(a00,a11,a22,a01,a02,a12,vec[0],eig1,vec[1],vec[2],eig) 
                _cross(vec[0],vec[1],vec[2])
                
            eig[0] = (eig0 *  scale) 
            eig[1] = (eig1 *  scale) 
            eig[2] = (eig2 *  scale) 
        else:
            #matrix is diagonal
            eig[0] = a00 *  scale
            eig[1] = a11 *  scale
            eig[2] = a22 *  scale
            vec[0] = (1,0,0)
            vec[1] = (0,1,0)
            vec[2] = (0,0,1)
        
        _sort_eigvec(eig,vec,eig,vec)

@guvectorize([(NFDTYPE[:],NFDTYPE[:,:],NFDTYPE[:],NFDTYPE[:,:]),(NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], '(m),(m,m)->(m),(m,m)', target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def sort_eigvec(e,v, eigval,eigvec):
    """Sorts eigenvalues and eigenvectors.
    Eigenvalues are sorted so that e[2] is the most distinct (extraordinary).
    """
    v = v.transpose()
    _sort_eigvec(e,v,eigval,eigvec.transpose())    
    
def eig(matrix,overwrite_x = False):
    """Computes eigenvalues and eigenvectors of 3x3 matrix using numpy.linalg.eig.
    Eigenvalues are sorted so that eig[2] is the most distinct (extraordinary).

    Parameters
    ----------
    matrix: (...,3,3) array 
        A 3x3 matrix.
    overwrite_x : bool, optional
        If set, the function will write eigenvectors as rows in the input array.
        
    Returns
    -------
    w : (..., 3) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are ordered so that the third eigenvalue is most
        distinct and first two are least distinct
    
    v : (..., 3, 3) array
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.
    """
    e,v = np.linalg.eig(matrix)
    if overwrite_x == True:
        out = (np.empty(matrix.shape[:-1],matrix.dtype), matrix)
        return sort_eigvec(e,v, *out)
    else:
        return sort_eigvec(e,v)
        
def tensor_eig(tensor, overwrite_x = False):
    """Computes eigenvalues and eigenvectors of a tensor.
    
    Eigenvalues are sorted so that eig[2] is the most distinct (extraordinary).
    
    If tensor is provided as a length 6 matrix, the elements are 
    a[0,0], a[1,1], a[2,2], a[0,1], a[0,2], a[1,2]. If provided as a (3x3) 
    matrix a, the rest of the elements are silently ignored.
    
    Parameters
    ----------
    tensor : (...,6) or (...,3,3) array 
        A length 6 array or 3x3 matrix
    overwrite_x : bool, optional
        If tensor is (...,3,3) array, the function will write eigenvectors
        as rows in the input array.
        
    Returns
    -------
    w : (..., 3) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are ordered so that the third eigenvalue is most
        distinct and first two are least distinct
    
    v : (..., 3, 3) array
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.
    """
    tensor = np.asarray(tensor)
    matrix = tensor
    out = None
    
    _is_real = (np.iscomplexobj(matrix) == False)
    
    _eig = _tensor_eig
    #_eig = np.linalg.eig
    
    if tensor.shape[-1] == 6:
        matrix = np.empty(tensor.shape[0:-1] + (3,3),tensor.dtype)
        matrix[...,0,0] = tensor[...,0]
        matrix[...,1,1] = tensor[...,1]
        matrix[...,2,2] = tensor[...,2]
        matrix[...,0,1] = tensor[...,3]
        matrix[...,0,2] = tensor[...,4]
        matrix[...,1,2] = tensor[...,5]
        # we can overwrite temporary matrix data
        out = (np.empty(tensor.shape[:-1] + (3,),tensor.dtype), matrix)
    elif matrix.shape[-2:] != (3,3):
        raise ValueError("Not a matrix of size (...,3,3) or (...,6)")
    else:
        if overwrite_x == True:
            out = (np.empty(matrix.shape[:-1],matrix.dtype), matrix)
            

    if out is None:
        eigv, vec = _eig(matrix,_is_real)
    else:
        eigv, vec = _eig(matrix, _is_real, out[0], out[1])
        
    return eigv, np.swapaxes(vec, -1,-2) #as returned by np.linalg.eig, eigenvectors are in columns, not rows

#--------- Inverse functions
    
@guvectorize(INV_DECL, '(m,m)->(m,m)', target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)   
def inv4x4(src,dst):
    """inverse 4x4 matrix, dst can be src for inplace transform"""
    assert src.shape[0]==4
    _inv4x4(src,dst)
    
@guvectorize(INV_DECL, '(m,m)->(m,m)', target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)   
def inv2x2(src, dst):
    """Inverse of a 2x2 matrix
    """
    assert src.shape[0]==2
    _inv2x2(src,dst)

@guvectorize(INV_DECL, '(n,n)->(n,n)', target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def inv_numba(mat, out):
    """numba gufunc version of np.linalg.inv"""
    inv = np.linalg.inv(mat)
    out[...] = inv
        
def inv(mat, out = None, method = "auto"):
    """inv(mat), gufunc
    
    Calculates inverse of a complex matrix. It uses an analytical algorithm for 
    2x2 and 4x4 shapes. It also works for general shape, but it takes a numerical
    approach calling np.linalg.inv function.
    
    Parameters
    ----------
    mat : ndarray
       Input array
    
    Examples
    --------
    
    >>> a = np.random.randn(4,4) + 0j
    >>> ai = inv(a)
    
    >>> from numpy.linalg import inv
    >>> ai2 = inv(a)
    
    >>> np.allclose(ai2,ai)
    True
    """
    mat = np.asarray(mat)
    if method == 'auto' :
        if mat.shape[-2:] == (2,2):
            return inv2x2(mat, out)
        elif mat.shape[-2:] == (4,4):
            return inv4x4(mat, out)
        else:
            return inv_numba(mat,out)
    elif method == "numba":
         return inv_numba(mat,out)
    elif method == 'numpy':
        if out is not None:
            raise ValueError("Cannot use out array to store inverse matrix")
        return np.linalg.inv(mat)
    else:
        raise ValueError(f"Unknown method `{method}`")
        

#--------- dotmm functions
           
@guvectorize(DOTMM_DECL, "(n,n),(n,n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotmm4(a,b,out):
    assert a.shape[0] == 4
    _dotmm4(a,b,out)
    
@guvectorize(DOTMM_DECL, "(n,n),(n,n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotmm3(a,b,out):
    assert a.shape[0] == 3
    _dotmm3(a,b,out)
 
@guvectorize(DOTMM_DECL, "(n,n),(n,n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotmm2(a,b,out):
    assert a.shape[0] == 2
    _dotmm2(a,b,out)   
    
@guvectorize(DOTMM_DECL, "(n,m),(m,k)->(n,k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotmm_numba(a,b,out):
    tmp = np.zeros_like(out)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            #tmp = NCDTYPE(0)
            for k in range(a.shape[1]):
                tmp[i,j] += a[i,k] *b[k,j]
                
    out[...] = tmp
    
def dotmm(a,b, out = None, method = "auto"):
    """Dot product of two matrices"""
    a = np.asarray(a)
    if method == 'auto':
        if a.shape[-2:] == (2,2):
            return dotmm2(a,b,out)
        elif a.shape[-2:] == (3,3):
            return dotmm3(a,b,out)
        elif a.shape[-2:] == (4,4):
            return dotmm4(a,b,out)
        else:
            return dotmm_numba(a,b,out)
    elif method == 'numpy':
        return np.matmul(a,b,out)
    elif method == "numba":
        return dotmm_numba(a,b,out)
    else:
        raise ValueError(f"Unknown method `{method}`")
  

#--------- dotmd functions      
  
@guvectorize(DOTMD_DECL, "(n,n),(n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotmd4(a, b, out):
    assert a.shape[0] == 4
    _dotmd4(a,b,out)

@guvectorize(DOTMD_DECL, "(n,n),(n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotmd2(a, b, out):
    assert a.shape[0] == 2
    _dotmd2(a,b,out)
    
@guvectorize(DOTMD_DECL, "(n,m),(m)->(n,m)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotmd_numba(a,b,out):
    tmp = np.zeros_like(out)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            tmp[i,j] += a[i,j] *b[j]
                
    out[...] = tmp
    
def dotmd(a,d, out = None):
    a = np.asarray(a)
    if a.shape[-2:] == (2,2) :
        return dotmd2(a, d, out)
    elif a.shape[-2:] == (4,4):
        return dotmd4(a, d, out)
    else:
        return dotmd_numba(a, d, out)
    

#--------- dotmdm functions    
          
@guvectorize(DOTMDM_DECL, "(n,n),(n),(n,n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotmdm4(a,d,b,out):
    _dotmd4(a,d,out)
    _dotmm4(out,b,out)
    
@guvectorize(DOTMDM_DECL, "(n,n),(n),(n,n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotmdm2(a,d,b,out):
    _dotmd2(a,d,out)
    _dotmm2(out,b,out)
    
def dotmdm(a, d, b, out = None):
    """dotmdm(a, d, b)
    
Computes a dot product of a 4x4 (or 2x2) matrix with a diagonal matrix (4- or 2-vector) 
and another 4x4 (or 2x2) matrix.

Take care when performing inplace transform.

The function performs dotmd(a,d,out) and then dotmm(out,b,out)
Therefore, care must be made when setting the output array so that you do not 
overwrite data during the conversion.

dotmdm(a,d,b, out = b) will not give correct result. 
dotmdm(a,d,b, out = a) is safe on the other hand.
"""
    a = np.asarray(a)
    d = np.asarray(d)
    b = np.asarray(b)
    if a.shape[-2:] == (4,4) and b.shape == (4,4):
        return dotmdm4(a,d,b,out)
    elif a.shape[-2:] == (2,2) and b.shape == (2,2):
        return dotmdm2(a,d,b,out)
    else:
        out = dotmd(a,d,out)
        return dotmm(out,b,out)    

@guvectorize([(NFDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],"(k),(n,n)->(n,n)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
def dotr2m(r,a,out):
    assert len(r) == 2
    return _dotr2m(r,a,out)


#--------- dotmv functions    

@guvectorize(DOTMV_DECL,"(n,n),(n)->(n)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def dotmv4(a, b, out):
    assert a.shape[0] == 4 
    _dotmv4(a,b,out)
    
@guvectorize(DOTMV_DECL,"(n,n),(n)->(n)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def dotmv3(a, b, out):
    assert a.shape[0] == 3
    _dotmv3(a,b,out)
    
@guvectorize(DOTMV_DECL,"(m,n),(n)->(m)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def dotmv24(a, b, out):
    assert a.shape[0] == 2
    assert a.shape[1] == 4
    _dotmv24(a,b,out)

@guvectorize(DOTMV_DECL,"(n,n),(n)->(n)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def dotmv2(a, b, out):
    assert a.shape[0] == 2
    _dotmv2(a,b,out)
    
@guvectorize(DOTMV_DECL,"(n,m),(m)->(n)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def dotmv_numba(a,b,out):
    tmp = np.zeros_like(out)
    for i in range(a.shape[0]):
        for k in range(a.shape[1]):
            tmp[i] += a[i,k] *b[k]
                
    out[...] = tmp
    
def dotmv(a, b, out = None):
    """dotmv(a, b)
    
Computes a dot product of a matrix of shape (n,m) and vector of shape (m,).  
"""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[-2:] == (2,2):
        return dotmv2(a, b, out)
    elif a.shape[-2:] == (3,3):
        return dotmv3(a, b, out)
    elif a.shape[-2:] == (4,4):
        return dotmv4(a, b, out)
    elif a.shape[-2:] == (2,4):
        return dotmv24(a, b, out)
    else:
        #general matrix/vector product
        return dotmv_numba(a, b, out)

#--------- dotvv functions    

@guvectorize(DOTVV_DECL, "(n),(n)->()", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotvv2(a, b, out):
    assert a.shape[0] == 2
    _dotvv2(a,b,out)
 
@guvectorize(DOTVV_DECL, "(n),(n)->()", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotvv3(a, b, out):
    assert a.shape[0] == 3
    _dotvv3(a,b,out)
    
@guvectorize(DOTVV_DECL, "(n),(n)->()", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotvv4(a, b, out):
    assert a.shape[0] == 4
    _dotvv4(a,b,out)

@guvectorize(DOTVV_DECL, "(n),(n)->()", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def dotvv_numba(a,b,out):
    tmp = 0.
    for i in range(a.shape[0]):
        tmp += a[i] *b[i]
    out[0] = tmp
    
def dotvv(a,b,out = None):
    """Scalar product of two vectors"""
    a = np.asarray(a)
    if a.shape[-1]== 2:
        return dotvv2(a, b, out)
    elif a.shape[-1]== 3:
        return dotvv3(a, b, out)    
    elif a.shape[-1]== 4:
        return dotvv4(a, b,out)   
    else:
        return dotvv_numba(a, b,out)

#--------- dotmf functions    

@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,k,n,n),(n,m,k)->(n,m,k)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmf(a, b, out):
    """dotmf(a, b)
    
Computes a dot product of an array of 4x4 (or 2x2) matrix with 
a field array or an E-array (in case of 2x2 matrices).
"""
    if b.shape[0] == 2:
        _dotmf2(a, b, out)
    else:
        assert b.shape[0] >= 4 #make sure it is not smaller than 4
        _dotmf4(a, b, out)

def dotmf(a,b, out = None):
    """dotmf(a, b)
    
Computes a dot product of an array of 4x4 (or 2x2) matrix with 
a field array or an E-array (in case of 2x2 matrices).
"""
    try:
        return _dotmf(a, b, out)
    except:
        a = np.asarray(a)
        b = np.asarray(b)
        a = broadcast_m(a, b)  
        return _dotmf(a, b, out)
       
@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,k,n,n),(m,k,n),(m,k,n,n),(n,m,k)->(n,m,k)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmdmf(a, d,b,f, out):
    if f.shape[0] == 2:
        _dotmdmf2(a, d,b,f, out)
    else:    
        assert f.shape[0] >= 4 #make sure it is not smaller than 4
        _dotmdmf4(a,d, b, f,out)
        
def dotmdmf(a,d,b,f, out = None):
    """dotmdmf(a, d, b, f)
    
Computes a dot product of an array of 4x4 (or 2x2) matrices, array of diagonal matrices, 
another array of matrices and a field array or an E-array (in case of 2x2 matrices).

Notes
-----
This is equivalent to

>>> dotmf(dotmdm(a,d,b),f)
"""
    try:
        return _dotmdmf(a, d,b,f, out)
    except:
        m = dotmdm(a,d,b)
        return dotmf(m,f, out)

def matmul(a,b,out = None):
    """A direct replacement for np.matmul function"""
    a,b = np.asarray(a), np.asarray(b)
    if a.ndim == 1 and b.ndim == 1:
        return dotvv(a,b,out)
    else:
        return dotmm(a,b,out)

def bdotmm(m1,m2):
    
    new_shape = m1.shape[:-4] + (m1.shape[-4] * m1.shape[-2], m1.shape[-3] * m1.shape[-1])
    _m1 = np.moveaxis(m1,-2,-3).reshape(new_shape)
    _m2 = np.moveaxis(m2,-2,-3).reshape(new_shape)
    
    new_shape = m1.shape[:-4] + (m1.shape[-4], m1.shape[-2], m1.shape[-3], m1.shape[-1])
    out = np.matmul(_m1,_m2).reshape(new_shape)
    return np.moveaxis(out,-2,-3)

def _bdotmm_ref(m1,m2, out = None):
    """same as bdotmm, but slower, for testing"""
    m1,m2 = np.broadcast_arrays(m1,m2)
    if out is None:
        out = np.zeros(m1.shape, CDTYPE)
    else:
        out[...] = 0.
    n = m1.shape[-3] 
    assert n == m1.shape[-4]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                out[...,i,j,:,:] += dotmm(m1[...,i,k,:,:],m2[...,k,j,:,:])
    return out


def bdotmd(m1,m2, out = None):
    """Performs a dot product of two nxn block matrices of blocks 4x4 or 2x2.
    The second matrix is block diagonal matrix of shape nx4x4 or nx2x2,
    The first matrix must be of shape nxnx4x4 or nxnx2x2 that describe two mxm matrices
    (m = 4*n or m = 2*n) of blocks of size 4x4 or 2x2.
    """
    if out is None:
        out = np.empty(shape = m1.shape, dtype = CDTYPE)

    for j in range(m1.shape[-3]):
        dotmm(m1[j],m2,out[j])
    return out

def _bdotmd_ref(m1,d):
    """same as bdotmd, but slower, for testing"""
    out = np.zeros_like(m1)
    for i in range(m1.shape[0]):
        for k in range(m1.shape[0]):
            out[i,k] +=  np.dot(m1[i,k],d[k])
    return out


def bdotdm(m1,m2, out = None):
    """Performs a dot product of two nxn block matrices of blocks 4x4 or 2x2.
    The first matrix is block diagonal matrix of shape nx4x4 or nx2x2,
    The second matrix must be of shape nxnx4x4 or nxnx2x2 that describe two mxm matrices
    (m = 4*n or m = 2*n) of blocks of size 4x4 or 2x2.
    """
    assert m2 is not out
    if out is None:
        out = np.empty(shape = m2.shape, dtype = CDTYPE)

    for j in range(m2.shape[-3]):
        dotmm(m1,m2[:,j],out[:,j])
    return out

def _bdotdm_ref(d,m):
    """Same as bdotdm, but slower, for testing"""
    out = np.zeros_like(m)
    for j in range(m.shape[0]):
        for k in range(m.shape[0]):
            out[k,j] +=  np.dot(d[k],m[k,j])
    return out

@guvectorize([(NFDTYPE[:],NFDTYPE[:]),(NCDTYPE[:],NCDTYPE[:])],"(n)->()",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def vnorm2(a,out):
    out[0] = _vnorm2(a)
    

def multi_dot_old(arrays,  axis = 0, reverse = False):
    """Computes dot product of multiple 2x2 or 4x4 matrices. If reverse is 
    specified, it is performed in reversed order. Axis defines the axis over 
    which matrices are multiplied."""
    out = None
    if axis != 0:
        arrays = np.asarray(arrays)
        indices = range(arrays.shape[axis])
        #arrays = np.rollaxis(arrays, axis)
        arrays = np.moveaxis(arrays, axis, 0)
    else:
        indices = range(len(arrays))
    if reverse == True:
        indices = reversed(indices)
    for i in indices:
        if out is None:
            out = np.asarray(arrays[i]).copy()
        else:
            b = np.broadcast(out, arrays[i])
            if b.shape == out.shape:
                out = dotmm(out, arrays[i], out = out)
            else:
                out = dotmm(out, arrays[i])
    return out

def multi_dot(arrays,  axis = 0, reverse = None, transfer = "backward"):
    """Computes dot product of multiple 2x2 or 4x4 matrices. If reverse is 
    specified, it is performed in reversed order. Axis defines the axis over 
    which matrices are multiplied."""
    if reverse is not None:
        deprecation("reverse is deprecated, use transfer argument instead")
    if transfer not in ("backward", "forward"):
        raise ValueError("Invalid transfer direction")
    
    out = None
    if axis != 0:
        arrays = np.asarray(arrays)
        arrays = np.moveaxis(arrays, axis, 0)
    if reverse == True:
        arrays = reversed(arrays)
        
    for a in arrays:
        if out is None:
            out = a.copy()
        else:
            b = np.broadcast(out, a)
            if b.shape == out.shape:
                _out = out 
            else:
                _out = None
            if transfer == "backward":
                out = dotmm(out, a, out = _out)
            else:
                out = dotmm(a,out, out = _out)
                    
    return out

def dotchi2v(a, v):
    t2 = np.asarray(a, FDTYPE)
    v = np.asarray(v)

    dtype = CDTYPE if np.iscomplexobj(v) else FDTYPE

    shape = np.broadcast_shapes(t2.shape[:-1],v.shape) + (3,)
    
    out = np.zeros(shape = shape, dtype = dtype)
    out[...,:,0] += t2[...,:,0]*v[...,None,0]
    out[...,:,1] += t2[...,:,5]*v[...,None,0]
    out[...,:,2] += t2[...,:,4]*v[...,None,0]
    
    out[...,:,0] += t2[...,:,5]*v[...,None,1]
    out[...,:,1] += t2[...,:,1]*v[...,None,1]
    out[...,:,2] += t2[...,:,3]*v[...,None,1]

    out[...,:,0] += t2[...,:,4]*v[...,None,2]
    out[...,:,1] += t2[...,:,3]*v[...,None,2]
    out[...,:,2] += t2[...,:,2]*v[...,None,2]
    return out  

def broadcast_m(m, field):
    """Broadcasts matrix m to match field spatial indices (the last two axes) so
    that it can be used in dot functions."""
    shape = m.shape[:-4]+ field.shape[-2:] + m.shape[-2:]
    return np.broadcast_to(m, shape)

def broadcast_d(d, field):
    """Broadcasts diagonal matrix d to match field spatial indices (the last two axes) so
    that it can be used in dot functions."""
    shape = d.shape[:-3]+ field.shape[-2:] + d.shape[-1:]
    return np.broadcast_to(d, shape)
    
__all__ = ["inv","cross", "dotmm","dotmf","dotmv","dotmdm","dotmd","multi_dot","eig","tensor_eig","matmul","dotvv","vnorm2"]

