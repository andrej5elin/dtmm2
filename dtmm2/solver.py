#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:58:10 2023

@author: andrej
"""

class MatrixSolver:
    def __init__(self, height = np.inf, width = np.inf, resolution = 1):
        """
        Paramters
        ---------
        shape : (int,int)
            Cross-section shape of the field data.
        wavelengths : float or array
            A list of wavelengths (in nanometers) or a single wavelength for 
            which to create the solver.
        pixelsize : float
            Pixel size in (nm).
        resolution : float
            Approximate sub-layer thickness (in units of pixelsize) used in the 
            calculation. With `resolution` = 1, layers thicker than `pixelsize` will
            be split into severeal thinner layers. Exact number of layers used
            in the calculation is obtained from :func:`get_optimal_steps`.
        mask : ndarray, optional
            A fft mode mask array. If not set, :func:`.wave.eigenmask` is 
            used with `betamax` to create such mask.
        method : str
            Either '4x4' (default), '4x4_1' or '2x2'.
        betamax : float, optional
            If `mask` is not set, this value is used to create the mask. If not 
            set, default value is taken from the config.
        """
        x,y = shape
        if not (isinstance(x, int) and isinstance(y, int)):
            raise ValueError("Invalid field shape.")
        self._shape = x,y
        self._wavelengths = np.asarray(wavelengths)
        if self.wavelengths.ndim not in (0,1):
            raise ValueError("`wavelengths` must be a scalar or an 1D array.")
        self._pixelsize = float(pixelsize)
        self._wavenumbers = k0(self.wavelengths, pixelsize)
        self._resolution = int(resolution)
        
        method = str(method) 
        if method in AVAILABLE_MATRIX_SOLVER_METHODS :
            self._method = method
        else:
            raise ValueError("Unsupported method {}".format(method))
        
        if mask is None:
            betamax = get_default_config_option("betamax", betamax)
            self._mask = eigenmask(shape, self.wavenumbers, betamax)
        else:
            self.mask = mask
            