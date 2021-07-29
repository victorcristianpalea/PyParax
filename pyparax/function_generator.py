import pyparax.external_imports as external_imports
import pyparax.internal_imports as internal_imports

class standard_initial_conditions:
    """
    This is a collection of basic functions that might be needed in order to fast check/implement optical beam profiles.
    
    Note 1: It is not necessary to use this class, however it might save time to implement here frequently used functions for an increase in productivity. 
    
    Note 2: All the functions are implemented so that their variables are related to the physical scale.
    """
    
    def spatial_domain_generator(dim = 1):
        """
        Generate the transversal spatial domain using the parameters in numeric_parameters.py.
        
        Inputs:
            dim = int - The number of dimensions of the transversal space. 
        Outputs:
            if dim = 1
                spatial_domain = 1-dimensional array; float - The 1-dimensional spatial domain.
            if dim = 2
                spatial_domain_x, spatial_domain_y = tuple of two 2-dimensional arrays; float - The arrays with the coordinates for the 2-dimensional spatial domain. 
        """
        
        #1-dimensional case
        if dim == 1:
            
            #Compute the domain 
            spatial_domain = external_imports.np.arange(0, internal_imports.p.N_x)* internal_imports.p.dx - internal_imports.p.x0 - internal_imports.p.N_x/2.* internal_imports.p.dx
            return spatial_domain
        
        #2-dimensional case
        elif dim == 2:
            
            #Compute the domain
            spatial_domain_x, spatial_domain_y = external_imports.np.meshgrid(external_imports.np.arange(0, internal_imports.p.N_x)* internal_imports.p.dx - internal_imports.p.x0 - internal_imports.p.N_x/2.* internal_imports.p.dx, 
                                         external_imports.np.arange(0, internal_imports.p.N_y)*internal_imports.p.dy - internal_imports.p.y0 - internal_imports.p.N_y/2.* internal_imports.p.dy, )
            return spatial_domain_x, spatial_domain_y

    def generate_gauss_1d(mean, sigma):
        """
        Generate a 1-dimensional Gaussian.
        
        Inputs:
            mean = float - The mean of the Gaussian.
            sigma = float - The standard deviation of the Gaussian.
        Outputs:
            f = 1-dimensional array; float - The Gaussian function.
        """
        
        #Compute the domain
        spatial_domain = standard_initial_conditions.spatial_domain_generator()
        
        #Compute the Gaussian function
        f = 1/sigma/external_imports.np.sqrt(2*external_imports.np.pi)*external_imports.np.exp(-(spatial_domain-mean)**2/2/sigma**2)      
        return f
        
    def generate_gauss_2d(mean_x, mean_y, sigma_x, sigma_y):
        """
        Generate a 2-dimensional Gaussian.
        
        Inputs:
            mean_x = float - The mean on the X axis of the Gaussian.
            mean_y = float - The mean on the Y axis of the Gaussian.
            sigma_x = float - The standard deviation the X axis of the Gaussian.
            sigma_y = float - The standard deviation the Y axis of the Gaussian.
        Outputs:
            f = 2-dimensional array; float - The Gaussian function.
        """
        
        #Compute the domain
        spatial_domain_x, spatial_domain_y = standard_initial_conditions.spatial_domain_generator(dim = 2)
        
        #Compute the Gaussian function
        f = 1./sigma_x/sigma_y/2/external_imports.np.pi*external_imports.np.exp(-(spatial_domain_x - mean_x)**2/2/sigma_x**2-(spatial_domain_y - mean_y)**2/2/sigma_y**2)
        return f
        
    def generate_airy_1d(scale, offset, decay):
        """
        Generate a 1-dimensional Airy function.
        
        Inputs:
            scale = float - For spatial domain rescaling
            offset = float - For shifting the spatial domain
            decay = float positive - For truncating the function 
        Outputs:
            f = 1-dimensional array; float - The Airy function.
        """
        
        #Compute the domain
        spatial_domain = standard_initial_conditions.spatial_domain_generator()
        
        #Compute the Airy function
        if decay == 0:
            f = external_imports.ss.airy((spatial_domain-offset) * scale)[0]
        else:
            f = external_imports.ss.airy((spatial_domain-offset) * scale)[0] * external_imports.np.exp((spatial_domain-offset)/decay)
        return f
    
    def generate_airy_2d(scale, offset, decay):
        """
        Generate a 2-dimensional Airy function by multiplication of two Airy functions i.e. Ai(x,y) = Ai(x)*Ai(y).
        
        Inputs:
            scale = float - For spatial domain rescaling
            offset = float - For shifting the spatial domain
            decay = float positive - For truncating the function 
        Outputs:
            f = 2-dimensional array; float - The Airy function.
        """
        
        #Compute the 1-dimensional Airy function
        f_airy = standard_initial_conditions.generate_airy_1d(scale, offset, decay)
        
        #Initialize the 2-dimensional variable for storing the Airy function
        f = external_imports.np.ones((internal_imports.p.N_x, internal_imports.p.N_y))
        
        #Multiply along one axis
        f = f_airy * f
        
        #Transpose the result
        f = external_imports.np.transpose(f)
        
        #Multiply along the other axis 
        f = f_airy * f
        
        #Transpose the result to return to the original arangement.
        f = external_imports.np.transpose(f)
        return f
    
class optical_elements:
    """
    This class contains some functions that model basic optical elements.
    """
    def lens_theory(focal, x_dom = None, y_dom = None, wavelength = None, n0 = None, offset_x = 0, offset_y = 0, dim = 1, fft_flag = False):
        """
        Computes the phase mask (complex function) of a lens in the paraxial approximation.
        
        Inputs:
            focal = float - The focal length of the lens.
            x_dom = array; float - The transversal spatial domain on the X axis. 
            y_dom = array; float - The transversal spatial domain on the Y axis.
            wavelength = float - The wavelength of the beam in vacuum.
            n0 = float - The refractive index of the medium.
            offset_x = float - The offset along the X axis of the lens.
            offset_y = float - The offset along the Y axis of the lens.
            dim = int - The number of dimensions for the transverse space.
            fft_flag = boolean - if True, the phase mask is computed in Fourier space.
        Outputs:
            if dim = 1
                phase = 1-dimensional array; complex - The phase mask (complex function) of the lens
            if dim = 2
                phase = 2-dimensional array; complex - The phase mask (complex function) of the lens
        """
        
        #Check if wavelength and n0 pare given by user. Otherwise take the values from numeric_parameters.py
        if wavelength == None:
            wavelength = internal_imports.p.wavelength
        if n0 == None:
            n0 = internal_imports.p.n0
        
        #Compute the wave number
        k = 2*external_imports.np.pi/wavelength*n0
        
        #The 1-dimensional transversal case
        if dim == 1:
            
            #Check if focal length = 0 
            if focal != 0:
                
                #If it is the case, compute the lens phase mask in Fourier space
                if fft_flag == True:
                    
                    #Compute the frequencies 
                    freqs = external_imports.np.fft.fftshift(external_imports.np.fft.fftfreq(internal_imports.p.N_x, internal_imports.p.dx))
                    
                    #Compute the phase
                    phase = external_imports.np.exp(-2j*external_imports.np.pi*freqs*offset_x)*(2*external_imports.np.pi*focal/k/1j)**0.5*external_imports.np.exp(1j*2*external_imports.np.pi**2*focal/k*freqs**2) 
                    return phase
                
                #If it is the case, compute the lens phase mask in real space
                else:
                    
                    #If needed compute the transversal spatial domain
                    if x_dom is None:
                        x_dom = standard_initial_conditions.spatial_domain_generator(dim = 1)
                    
                    #Compute the phase
                    phase = k/2/focal * (x_dom-offset_x)**2
                    phase = external_imports.np.exp(-1j*phase) 
                    return phase
            else:
                #If focal lenth = 0, the phase mask does nothing
                return external_imports.np.ones(internal_imports.p.N_x)
            
        #The 2-dimensional transversal case
        elif dim == 2:
            
            #Check if focal length = 0 
            if focal != 0:
                
                #If it is the case, compute the lens phase mask in Fourier space
                if fft_flag == True:
                    
                    #Compute the frequencies 
                    freqs_x = external_imports.np.fft.fftshift(external_imports.np.fft.fftfreq(internal_imports.p.N_x, internal_imports.p.dx))
                    freqs_y = external_imports.np.fft.fftshift(external_imports.np.fft.fftfreq(internal_imports.p.N_y, internal_imports.p.dy))
                    x_freqs, y_freqs = external_imports.np.meshgrid(freqs_x, freqs_y)
                    
                    #Compute the phase
                    phase = (external_imports.np.exp(-2j*external_imports.np.pi*x_freqs*offset_x)*external_imports.np.exp(-2j*external_imports.np.pi*y_freqs*offset_y)*2*external_imports.np.pi*focal/1j/k)*external_imports.np.exp(1j*2*external_imports.np.pi**2*focal/k*(x_freqs**2+y_freqs**2))
                    return phase
                
                #If it is the case, compute the lens phase mask in real space
                else:
                    
                    #If needed compute the transversal spatial domain
                    if x_dom is None or y_dom is None:
                        x_dom, y_dom = standard_initial_conditions.spatial_domain_generator(dim = 2)
                    else:
                        x_dom, y_dom = external_imports.np.meshgrid(x_dom, y_dom)
                    
                    #Compute the phase
                    phase = k/2/focal * ((x_dom-offset_x)**2+(y_dom-offset_y)**2)
                    phase = external_imports.np.exp(-1j*phase)
                    return phase
            
            #If the focal length = 0, return a phase mask that does not change the profile
            else:
                phase = external_imports.np.ones(internal_imports.p.N_x) 
                return phase
            
    def linear_phase(tilt_x = 0, tilt_y = 0, wavelength = None, n0 = None, dim = 1, fft_flag = False):
        """
        Computes a phase mask that tilts the propagation direction with a given angle.
        
        Inputs:
            tilt_x = float - The tilt parameter for the X axis.
            tilt_y = float - The tilt parameter for the Y axis.
            wavelength = float - The wavelength of the beam in vacuum.
            n0 = float - The refractive index of the medium.
            dim = int - The number of dimensions for the transverse space.
            fft_flag = boolean - if True, the phase mask is computed in Fourier space.
        Outputs:
            if dim = 1
                phase = 1-dimensional array; complex - The phase mask (complex function) of the lens
            if dim = 2
                phase = 2-dimensional array; complex - The phase mask (complex function) of the lens
        """
        
        #Check if wavelength and n0 pare given by user. Otherwise take the values from numeric_parameters.py
        if wavelength == None:
            wavelength = internal_imports.p.wavelength
        if n0 == None:
            n0 = internal_imports.p.n0
            
        #Compute the wave number
        k = 2*external_imports.np.pi/wavelength*n0
        
        #The 1-dimensional transversal case
        if dim == 1:
            
            #Compute the transversal spatial domain
            x = standard_initial_conditions.spatial_domain_generator(dim = 1)
            
            #Compute the phase in real space
            phase = x*k*tilt_x
            
            #If needed, ompute the phase in Fourier space
            if fft_flag == True:
                phase = external_imports.np.fft.fftshift(external_imports.np.fft.fft(external_imports.np.fft.fftshift(external_imports.np.exp(1j*phase))))
                return phase
            else:
                return external_imports.np.exp(1j*phase)
            
        #The 2-dimensional transversal case
        elif dim == 2:
            
            #Compute the transversal spatial domain
            x, y = standard_initial_conditions.spatial_domain_generator(dim = 2)
            
            #Compute the phase in real space
            phase = x*k*tilt_x + y*k*tilt_y
            
            #If needed, compute the phase in Fourier space
            if fft_flag == True:
                phase = external_imports.np.fft.fftshift(external_imports.np.fft.fft2(external_imports.np.fft.fftshift(external_imports.np.exp(1j*phase))))
                return phase
            else:
                return external_imports.np.exp(1j*phase)
            
    def amplitude_mask_circular(width, smooth = 1, x_size = None, y_size = None, offset_x = 0, offset_y = 0, dim = 1):
        """
        Computes a circular amplitude mask.
        
        Inputs:
            width = float - The diameter of the apperture.
            smooth = int - Number of pixels and interations on which the edges of the apperture are smoothened.
            x_size = int - Number of pixels on the X axis.
            y_size = int - Number of pixels on the y axis.
            offset_x = float - The offset along the X axis of the lens.
            offset_y = float - The offset along the Y axis of the lens.
            dim = int - The number of dimensions for the transverse space.
        Outputs:
            if dim = 1
                mask = 1-dimensional array; real - The amplitude mask (real function).
            if dim = 2
                mask = 2-dimensional array; real - The amplitude mask (real function).
        """
        
        #The 1-dimensional transversal case
        if dim == 1:
            
            #Check if size parameter is given and initialize it with the default value if not
            if x_size is None:
                x_size = internal_imports.p.N_x
                
            #Define the amplitude mask variable
            mask = external_imports.np.zeros(x_size)
            
            #Convert the width and offset_x variables from physical units to pixels
            width_in_pixels = width/internal_imports.p.dx
            offset_x_in_pixels = offset_x / internal_imports.p.dx

            #Compute the amplitude mask
            mask[int((x_size-width_in_pixels)/2-offset_x_in_pixels):int((x_size-width_in_pixels)/2+width_in_pixels-offset_x_in_pixels)] = 1
            
            #If smoothening of the edges is chosen, apply it
            if external_imports.np.round(smooth) > 1:
                
                #Define the convolution box based on the value of the smooth parameter
                box = external_imports.np.ones(external_imports.np.round(smooth))/external_imports.np.round(smooth)
                
                #Apply the convolution box for a number of times given by the smooth parameter
                for i in range(external_imports.np.round(smooth)):
                    mask = external_imports.np.convolve(mask, box, mode='same')
                    
                #Shift the resulting smoothened image to compensate the shift given by the convolution
                mask = external_imports.np.roll(mask, -int(external_imports.np.round(smooth/2.)))
        
        #The 2-dimensional transversal case
        if dim == 2:
            
            #Check if size parameters are given and initialize them with the default values if not
            if x_size is None or y_size is None:
                x_size = internal_imports.p.N_x
                y_size = internal_imports.p.N_y
            
            #Define the amplitude mask variable
            mask = external_imports.np.zeros((x_size, y_size))
            
            #Convert the width, offset_x, and offset_y variables from physical units to pixels
            width_in_pixels_x = int(width/internal_imports.p.dx)
            width_in_pixels_y = int(width/internal_imports.p.dy)
            offset_x_in_pixels = int(offset_x / internal_imports.p.dx)
            offset_y_in_pixels = int(offset_y / internal_imports.p.dy)
            
            #Compute the grid indices
            x_dom, y_dom = external_imports.np.meshgrid(external_imports.np.arange(x_size)-int(x_size/2), external_imports.np.arange(y_size)-int(y_size/2))
            
            #Compute the amplitude mask
            mask = (((x_dom-offset_x_in_pixels)/width_in_pixels_x)**2 + ((y_dom-offset_y_in_pixels)/width_in_pixels_y)**2 < 1.) + 0.
            
            #If smoothening of the edges is chosen, apply it
            if external_imports.np.round(smooth) > 1:
                
                #Define the convolution box based on the value of the smooth parameter
                box = external_imports.np.ones([external_imports.np.round(smooth), external_imports.np.round(smooth)])/external_imports.np.round(smooth)**2
                
                #Apply the convolution box for a number of times given by the smooth parameter
                for i in range(external_imports.np.round(smooth)):
                    mask = external_imports.sps.convolve2d(mask, box, mode='same')
                
                #Shift the resulting smoothened image to compensate the shift given by the convolution
                mask = external_imports.np.roll(mask, -int(external_imports.np.round(smooth/2.)), axis = 0)
                mask = external_imports.np.roll(mask, -int(external_imports.np.round(smooth/2.)), axis = 1)
                
        return mask
                