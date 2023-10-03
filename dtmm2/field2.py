import numpy as np
from dtmm2.tmm import EHz, poynting

Z0 = 377
Z02 = Z0*2

def as_total_field(field, beta = None, phi = None, epsv = None, epsa = None, out = None):
    """Converts (Ex,Hy,Ey,Hx) field array to (Hz,Ex,Hy,Ey,Hx,Ez) total field array"""
    if out is None:
        out = np.empty(field.shape[:-1] + (6,), dtype = field.dtype)
    out[..., 1:-1] = field
    #Hz is at index 0, Ez is at index -1 (5) so reverse view
    tmp = out[...,-1::-5]
    EHz(field, beta, phi, epsv, epsa, out = tmp)
    return out

class LayerData1D():
    def __init__(self, d, epsv, epsa, chi2 = None, source = None):
        self._thickness = d
        self._epsv = epsv
        self._epsa = epsa
        self._chi2 = chi2
    
class LayerDataList1D():
    def __init__(self, d, epsv, epsa):
        self._thickness = d
        self._epsv = epsv
        self._epsa = epsa
        
    def __getitem__(self, value):
        return LayerData1D(self._d[value], self._epsv[value], self._epsa[value])
    
    def __len__(self):
        return len(self._d)

class Field1D():
    def __init__(self,field):
        self._field = np.asarray(field)
        if self._field.ndim == 0 or self._field.shape[-1] != 4:
            raise ValueError("Invalid field shape")
                                 
    @property
    def field(self):
        return self._field
    
    @property
    def Ex(self):
        return self._field[...,0]
    
    @property
    def Ey(self):
        return self._field[...,2]
    
    @property
    def E(self):
        return self._field[...,0::2]
    
    @property
    def Hx(self):
        return self._field[...,3]    
    
    @property
    def Hy(self):
        return self._field[...,1]      
    
    @property
    def H(self):
        return self._field[...,-1::-2]  
    
    def get_flux(self):
        return poynting(self._field)
    
class TotalField1D(Field1D):
    def __init__(self, total_field):
        self._total_field = np.asarray(total_field)
        if self._total_field.ndim == 0 or self._total_field.shape[-1] != 6:
            raise ValueError("Invalid field shape")
        self._field = self._total_field[...,1:-1]
                                 
    @property
    def total_field(self):
        return self._total_field
    
    def E(self):
        return self._total_field[...,1::2]
    
    @property
    def H(self):
        return self._total_field[...,-2::-2]  
    
class FieldList1D(Field1D):
    def __init__(self,field):
        self._field = np.asarray(field)
        if self._field.ndim in (0,1) or self._field.shape[-1] != 4:
            raise ValueError("Invalid field shape") 
            
    def __getitem__(self, value):
        return Field1D(self._field[value])
    
    def __len__(self):
        return len(self._field)
    
class TotalFieldList1D(Field1D):
    def __init__(self, total_field):
        self._total_field = np.asarray(total_field)
        if self._total_field.ndim in (0,1) or self._total_field.shape[-1] != 6:
            raise ValueError("Invalid field shape")
        self._field = self._total_field[...,1:-1]

    def __getitem__(self, value):
        return TotalField1D(self._total_field[value])
    
    def __len__(self):
        return len(self._total_field)