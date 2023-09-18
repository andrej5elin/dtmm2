
class BaseMatrixSolver(self):
    """Base class for all matrix-based solvers."""
    
    #: module that implements TMM
    tmm = tmm3d

    # field array data
    _field_out = None
    _field_in = None
    _modes_in = None
    _modes_out = None
    
    _mask = None
    
    #: resize parameter for layer_met calculation
    _resize = 1
    
    
    def __init__(self, shape, height = np.inf, width = np.inf)
    
    def __init__(self, shape, wavelengths = [500], pixelsize = 100, resolution = 1,  mask = None, method = "4x4", betamax = None):
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
            
    @property       
    def stack_memory_size(self):
        """Specifies memory requirement for stack matrix."""
        size = 2 if self.method.startswith("2x2") else 4
        dt = np.dtype(CDTYPE)
        nmodes = np.asarray(self.nmodes)
        B = (nmodes**2 * size**2 * dt.itemsize).sum()
        kB = B/1024
        MB = kB/1024
        GB = MB/1024
        return {"B": B, "kB" : kB, "MB" : MB, "GB" : GB}
          
    def print_solver_info(self):
        """prints solver info"""
        print(" $ dim : {}".format(self.dim))
        print(" $ shape : {}".format(self.shape))
        print(" # pixel size : {}".format(self.pixelsize))
        print(" # resolution : {}".format(self.resolution))
        print(" $ method : {}".format(self.method))
        print(" $ wavelengths : {}".format(self.wavelengths))
        print(" $ n modes : {}".format(self.nmodes))
        
    def print_data_info(self):
        pass
    
    def print_memory_info(self):
        """prints memory requirement"""
        size_dict = self.stack_memory_size
        size_out = size_dict["B"]
        key_out = "B"
        
        for key in ("kB", "MB", "GB"):
            if size_dict[key] > 1:
                size_out = size_dict[key]
                key_out = key
                
        print(" $ stack size ({}) : {}".format(key_out, size_out))
                
        
    def print_info(self):
        """prints all info"""
        print("-----------------------")
        print("Solver: ")
        self.print_solver_info()
        print("-----------------------")
        print("Memory: ")
        self.print_memory_info()
        print("-----------------------")
        print("Data:")
        self.print_data_info()

    @property            
    def field_dim(self):
        """Required minimum field dim."""
        return self.dim if self.wavelengths.ndim == 0 else self.dim + 1
    
    @property 
    def field_shape(self):
        """Required field shape."""
        return self.wavelengths.shape + (4,) + self.shape
    
    @property
    def mask(self):
        return self._mask
    
    @mask.setter      
    def mask(self, mask):
        """Sets fft mask.
        
        Parameters
        ----------
        mask : ndarray
            A boolean mask, describing fft modes that are used in the solver.
        """
        mask = np.asarray(mask)
        mask_shape = self.wavelengths.shape + self.shape
        
        if mask.shape != mask_shape:
            raise ValueError("Mask shape must be of shape {}".format(mask_shape))
        if not mask.dtype == bool:
            raise ValueError("Not a bool mask.")
            
        self._mask = mask
        self.clear_matrices()
        self.clear_data()
    
    def clear_matrices(self):
        self._stack_matrix = None
        self._refl_matrix = None
        self._field_matrix_in = None
        self._field_matrix_out = None
        self._trans_matrix = None
        self.layer_matrices = []
        
    def clear_data(self):
        self._modes_in = None
        self._modes_out = None
        self._field_in = None
        self._field_out = None

    def _validate_field(self, field):
        field = np.asarray(field)
            
        if field.shape[-self.field_dim:] != self.field_shape:
            raise ValueError("Field shape not comaptible with solver's requirements. Must be (at least) of shape {}".format(self.field_shape))
        return field
    
    def _validate_modes(self, modes):
        _, modes = self.tmm._validate_modes(self.mask, modes)
        return modes
    
    def _field2modes(self,field):
        _, modes = field2modes(field, self.wavenumbers, mask = self.mask)  
        return modes
    
    def _modes2field(self, modes, out = None):
        field = modes2field(self.mask, modes, out = out)  
        return field     
            
    @property
    def field_in(self):
        """Input field array"""
        if self._field_in is not None:
            return self._field_in
        elif self._modes_in is not None:
            self._field_in = self._modes2field(self._modes_in)
            return self._field_in
        
    @field_in.setter
    def field_in(self, field):
        self._field_in =  self._validate_field(field)   
        # convert field to modes, we must set private attribute not to owerwrite _field_in
        self._modes_in = self._field2modes(self._field_in)    
        
    @property
    def field_out(self):
        """Output field array"""
        if self._field_out is not None:
            return self._field_out
        elif self._modes_out is not None:
            self._field_out = self._modes2field(self._modes_out)
            return self._field_out
    
    @field_out.setter
    def field_out(self, field):
        self._field_out = self._validate_field(field)   
        # convert field to modes, we must set private attribute not to owerwrite _field_out
        self._modes_out = self._field2modes(self._field_out)    
        
    @property
    def modes_in(self):
        return self._modes_in

    @modes_in.setter
    def modes_in(self, modes):
        self._modes_in = self._validate_modes(modes)
        self._field_in = None
        
    @property
    def modes_out(self):
        return self._modes_out

    @modes_out.setter
    def modes_out(self, modes):
        self._modes_out = self._validate_modes(modes)
        self._field_out = None

    def _get_field_data(self, field, copy):
        if copy:
            return field.copy(), self.wavelengths.copy(), self.pixelsize
        else:
            return field, self.wavelengths, self.pixelsize
        
    def get_field_data_in(self, copy = True):
        """Returns input field data tuple.
        
        If copy is set to True (default), a copy of the field_out array is made. 
        """
        return self._get_field_data(self.field_in, copy)

    def get_field_data_out(self, copy = True):
        """Returns output field data tuple.
        
        If copy is set to True (default), a copy of the field_out array is made. 
        """
        return self._get_field_data(self.field_out, copy)
    
    def _get_f_iso(self, n):
        return self.tmm._f_iso(self.mask, self.wavenumbers, n = n, shape = self.material_shape)
    
    def _get_layer_mat(self,d,epsv,epsa, nsteps = 1):
        if self.dispersive == True:
            return self.tmm._dispersive_layer_mat(self.wavenumbers,d,epsv,epsa,mask = self.mask, method = self.method,nsteps = nsteps,resize = self._resize, wavelength = self.wavelengths)
        else:
            return self.tmm._layer_mat(self.wavenumbers,d,epsv,epsa,mask = self.mask, method = self.method,nsteps = nsteps, resize = self._resize)

    def _get_stack_matrix(self,d,epsv,epsa,nsteps = 1):
        return self.tmm._stack_mat(self.wavenumbers,d,epsv,epsa, method = self.method, mask = self.mask, nsteps = nsteps, resize = self._resize)
        
    def calculate_field_matrix(self,nin = 1., nout = 1.):
        """Calculates field matrices. 
        
        This must be called after you set material data.
        """
        if self.material_shape is None:
            raise ValueError("You must first set material data")
        self._nin = float(nin)
        self._nout = float(nout)
        #now build up matrices field matrices       
        self._field_matrix_in = self._get_f_iso(self.nin)
        self._field_matrix_out = self._get_f_iso(self.nout)
        
    def calculate_reflectance_matrix(self):
        """Calculates reflectance matrix.
        
        Available in "4x4","4x4_1" methods only. This must be called after you 
        have calculated the stack and field matrices.
        """
        if self.method not in ("4x4","4x4_1"):
            raise ValueError("reflectance matrix is available in 4x4 and 4x4_1 methods only")
        if self.stack_matrix is None:
            raise ValueError("You must first calculate stack matrix")
        if self.field_matrix_in is None:
            raise ValueError("You must first calculate field matrix")
        mat = self.tmm._system_mat(self.stack_matrix, self.field_matrix_in, self.field_matrix_out)
        mat = self.tmm._reflection_mat(mat)
        self._refl_matrix = mat
      
    def calculate_transmittance_matrix(self):
        """Calculates transmittance matrix.
        
        Available in "2x2" method only. This must be called after you have
        calculated the stack matrix.
        """
        if self.method != "2x2":
            raise ValueError("transmittance matrix is available in 2x2 method only")
        if self.stack_matrix is None:
            raise ValueError("You must first calculate stack matrix")
        
        self._trans_matrix  = self.tmm._transmission_mat(self.stack_matrix) 
       
    def transfer_field(self, field_in = None, field_out = None):
        """Transfers field.
        
        This must be called after you have calculated the transmittance/reflectance 
        matrix.
        
        Parameters
        ----------
        field_in : ndarray, optional
            If set, the field_in attribute will be set to this value prior to
            transfering the field.
        field_out : ndarray, optional
            If set, the field_out attribute will be set to this value prior to
            transfering the field.            
        """
        if self.refl_matrix is None and self.trans_matrix is None:
            raise ValueError("You must first create reflectance/transmittance matrix")
        if field_in is not None:
            self.field_in = field_in
        if field_out is not None:
            self.field_out = field_out       
        if self.modes_in is None:
            raise ValueError("You must first set `field_in` data or set the `field_in` argument.")
        self.transfer_modes()
        return self.field_out

    def transfer_modes(self, modes_in = None, modes_out = None):
        """Transfers modes.
        
        This must be called after you have calculated the transmittance/reflectance 
        matrix.
        
        Parameters
        ----------
        modes_in : mode, optional
            If set, the mode_in attribute will be set to this value prior to
            transfering the field.
        modes_out : mode, optional
            If set, the mode_out attribute will be set to this value prior to
            transfering the field.            
        """

        if self.refl_matrix is None and self.trans_matrix is None:
            raise ValueError("You must first create reflectance/transmittance matrix")
        if modes_in is not None:
            self.modes_in = modes_in
        if modes_out is not None:
            self.modes_out = modes_out       
            
        if self.modes_in is None:
            raise ValueError("You must first set `field_in` data or set the `field_in` argument.")

        grouped_modes_in = self.grouped_modes_in
        grouped_modes_out = self.grouped_modes_out
    
        # transfer field
        if self.method.startswith("4x4"):
            grouped_modes_out = self.tmm._reflect(grouped_modes_in, rmat = self.refl_matrix, fmatin = self.field_matrix_in, fmatout = self.field_matrix_out, fvecout = grouped_modes_out)
        else:
            grouped_modes_out = self.tmm._transmit(grouped_modes_in, tmat = self.trans_matrix, fmatin = self.field_matrix_in, fmatout = self.field_matrix_out, fvecout = grouped_modes_out)
        
        # now ungroup and convert modes to field array. This also sets the modes_out 
        self.grouped_modes_out = grouped_modes_out
        
        # in case we have field_out array, we update that as well
        if self._field_out is not None:
            self._modes2field(self.modes_out, out = self._field_out) 
    
        if self.method.startswith("4x4"):
            #we need to update input field because of relfections. This also sets the modes_in
            self.grouped_modes_in = grouped_modes_in
            # in case we have field_in array, we update that as well
            if self._field_in is not None:
                self._modes2field(self.modes_in, out = self._field_in) 
        return self.modes_out

    def _group_modes(self, modes):
        if modes is not None:
            m = self.tmm.mode_masks(self.mask, shape = self.material_shape)
            return self.tmm.group_modes(modes, m)
    
    def _ungroup_modes(self, modes):
        if modes is not None:
            m = self.tmm.mode_masks(self.mask, shape = self.material_shape)
            return self.tmm.ungroup_modes(modes, m)

    @property
    def grouped_modes_in(self):
        """Grouped input modes"""
        return self._group_modes(self.modes_in) 

    @grouped_modes_in.setter
    def grouped_modes_in(self, modes):
        self._modes_in = self._ungroup_modes(modes)

    @property
    def grouped_modes_out(self):
        """Grouped output modes"""
        return self._group_modes(self.modes_out) 

    @grouped_modes_out.setter
    def grouped_modes_out(self, modes):
        self._modes_out = self._ungroup_modes(modes)
    
    @property
    def nmodes(self):
        """Number of coupled modes per wavelength"""
        if self.wavelengths.ndim == 0:
            if self.mask is not None:
                return self.mask.sum()
        else:
            if self.mask is not None:
                return tuple((m.sum() for m in self.mask))  



class MatrixSolver1D:
    def compute_transfer_matrices(self):
        pass
    
    def compute_reflectance_matrix(self):
        pass
    
    
        