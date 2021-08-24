import pyparax.external_imports as external_imports
import pyparax.internal_imports as internal_imports

class numeric_fourier_solver_1d:
    """
    Numerical solver for the paraxial propagation equation. The system is 1+1-dimensional.
    """
    def linear(f0,
              wavelength = None,
              n0 = 1,
              dx = None,
              dz = None,
              steps = None, 
              forward = True,
              print_progress = False,
              output_full = True,
              fft_input = False,
              fft_output = False):
        """
        Computes the solution for the partial differential equation (PDE) for the 1-dimensional wave equation in the paraxial approximation using the Fourier Transform.
        Inputs:
            f0 = 1-dimensional array; complex - The initial condition of the PDE.
            wavelength = float; real - The wavelength of the electromagnetic wave in vacuum.
            n0 = float; real - The refractive index of the medium. 
            dx = float; real - The step for the spatial transverse axis.
            dz = float; real - The step for the spatial propagation axis.
            steps = int; natural - The number of iterations the solver has to compute.
            forward = boolean - If False, the propagation direction is reversed.
            print_progress = boolean - If True, the iteration at which the solver computing is printed.
            output_full = boolean - If True, the function returns the solution for each iteration. If False, only the last iteration is returned.
            fft_input = boolean - If True, f0 is considered to be the FFT of the initial condition, so that no extra FFT is applied on it.
            fft_output = boolean - If True, the output is returned in Fourier space.
        Outputs:
            CONDITION: 
                if output_full = True
                    f = 2-dimensional array; complex - The solution of the PDE for every iteration. The shape is steps x size(f0)
                if output_full = False
                    f = 1-dimensional array; complex - The solution of the PDE for the last iteration. 
        """
        #Check if required parameters are inserted. If not, they are imported from the numeric_parameters.py file
        if wavelength == None:
            wavelength = internal_imports.p.wavelength
        if dx == None:
            dx = internal_imports.p.dx
        if dz == None:
            dz = internal_imports.p.dz
        if steps == None:
            steps = internal_imports.p.N_z
        if external_imports.np.size(f0) == internal_imports.p.N_x:
            size = internal_imports.p.N_x
        else:
            size = external_imports.np.size(f0)
        
        #Compute the wavenumber
        k = 2*external_imports.np.pi/wavelength*n0
        
        #Compute the Fast Fourier Transform (FFT) of the initial condition
        if fft_input == True:
            fft_f0 = external_imports.np.copy(f0)
        else:
            fft_f0 = external_imports.np.fft.fftshift(external_imports.np.fft.fft(external_imports.np.fft.fftshift(f0)))
        #Compute the frequencies associated with the transvers spatial grid
        freqs_x = external_imports.np.fft.fftshift(external_imports.np.fft.fftfreq(size, dx))
        
        #Prepare the frequencies based on the type of propagation
        if forward == True:
            freqs = freqs_x**2*4*external_imports.np.pi**2
        elif forward == False:
            freqs = -freqs_x**2*4*external_imports.np.pi**2
        else:
            print('Wrong forward parameter. Options: True / False')
        
        #Compute the solution for the output_full = False case
        if output_full == False:
            z = dz * steps
            f = fft_f0 * external_imports.np.exp(-1j/2/k*freqs*z)
            if fft_output == False:
                f = external_imports.np.fft.ifftshift(external_imports.np.fft.ifft(external_imports.np.fft.ifftshift(f)))
        #Compute he solution for the output_full = True case
        elif output_full == True:
            f = external_imports.np.ones((steps, size))+0j
            for i in external_imports.np.arange(steps):
                #Check if print_progress = True and print the iteration variable if so 
                if print_progress == True:
                    print("Step " + str(i) + "/" + str(steps))
                f[i,:] = fft_f0 * external_imports.np.exp(-1j/2/k*freqs*(i+1)*dz)
            #Return from frequency space by inverse FFT.
                if fft_output == False:
                    f[i,:] = external_imports.np.fft.ifftshift(external_imports.np.fft.ifft(external_imports.np.fft.ifftshift(f[i,:])))
        else:
            print('Wrong output_full parameter. Options: True / False')
        #Return solution
        return f
    
class numeric_fourier_solver_2d:
    """
    Numerical solver for the paraxial propagation equation. The system is 1+1-dimensional.
    """
    def linear(f0,
              wavelength = None,
              n0 = 1,
              dx = None,
              dy = None,
              dz = None,
              steps = None, 
              forward = True,
              print_progress = False,
              output_full = True, 
              fft_input = False,
              fft_output = False):
        """
        Computes the solution for the partial differential equation (PDE) for the 2-dimensional wave equation in the paraxial approximation using the Fourier Transform.
        Inputs:
            f0 = 2-dimensional array; complex - The initial condition of the PDE.
            wavelength = float; real - The wavelength of the electromagnetic wave in vacuum.
            n0 = float; real - The refractive index of the medium. 
            dx = float; real - The step for one of the spatial transverse axis.
            dy = float; real - The step for the other spatial transverse axis.
            dz = float; real - The step for the spatial propagation axis.
            steps = int; natural - The number of iterations the solver has to compute.
            forward = boolean - If False, the propagation direction is reversed.
            print_progress = boolean - If True, the iteration at which the solver computing is printed.
            output_full = boolean - If True, the function returns the solution for each iteration. If False, only the last iteration is returned.
            fft_input = boolean - If True, f0 is considered to be the FFT of the initial condition, so that no extra FFT is applied on it.
            fft_output = boolean - If True, the output is returned in Fourier space.
        Outputs:
            CONDITION: 
                if output_full = True
                    f = 3-dimensional array; complex - The solution of the PDE for every iteration. The shape is steps x size_on_x(f0) x size_on_y(f0)
                    !!! Consumes lots of RAM !!!
                if output_full = False
                    f = 2-dimensional array; complex - The solution of the PDE for the last iteration. 
        """
        #Check if required parameters are inserted. If not, they are imported from the numeric_parameters.py file
        if wavelength == None:
            wavelength = internal_imports.p.wavelength
        if dx == None:
            dx = internal_imports.p.dx
        if dy == None:
            dy = internal_imports.p.dy
        if dz == None:
            dz = internal_imports.p.dz
        if steps == None:
            steps = internal_imports.p.N_z
        if external_imports.np.shape(f0) == (internal_imports.p.N_x, internal_imports.p.N_y):
            size_x = internal_imports.p.N_x
            size_y = internal_imports.p.N_y
        else:
            size_x = external_imports.np.shape(f0)[0]
            size_y = external_imports.np.shape(f0)[1]
        
        #Compute the wavenumber
        k = 2*external_imports.np.pi/wavelength*n0
        
        #Compute the Fast Fourier Transform (FFT) of the initial condition
        if fft_input == True:
            fft_f0 = external_imports.np.copy(f0)
        else:
            fft_f0 = external_imports.np.fft.fftshift(external_imports.np.fft.fft2(external_imports.np.fft.fftshift(f0)))
            
        #Compute the frequencies associated with the transvers spatial grid for both axis
        freqs_x = external_imports.np.fft.fftshift(external_imports.np.fft.fftfreq(size_x, dx))
        freqs_y = external_imports.np.fft.fftshift(external_imports.np.fft.fftfreq(size_y, dy))
        x_freqs, y_freqs = external_imports.np.meshgrid(freqs_x, freqs_y)
        
        #Prepare the frequencies based on the type of propagation
        if forward == True:
            freqs = (x_freqs**2 + y_freqs**2)*4*external_imports.np.pi**2
        elif forward == False:
            freqs = -(x_freqs**2 + y_freqs**2)*4*external_imports.np.pi**2
        else:
            print('Wrong forward parameter. Options: True / False')

        #Compute the solution for the output_full = False case
        if output_full == False:
            z = dz * steps
            f = fft_f0 * external_imports.np.exp(-1j/2/k*freqs*z)
            if fft_output == False:
                f = external_imports.np.fft.ifftshift(external_imports.np.fft.ifft2(external_imports.np.fft.ifftshift(f)))
        #Compute he solution for the output_full = True case
        elif output_full == True:
            f = external_imports.np.ones((steps, size_x, size_y))+0j
            for i in range(steps):
                #Check if print_progress = True and print the iteration variable if so 
                if print_progress == True:
                    print("Step " + str(i) + "/" + str(steps))
                f[i,:,:] = fft_f0 * external_imports.np.exp(-1j/2/k*freqs*(i+1)*dz)
                #Return from frequency space by inverse FFT.
                if fft_output == False:
                    f[i,:,:] = external_imports.np.fft.ifftshift(external_imports.np.fft.ifft2(external_imports.np.fft.ifftshift(f[i,:,:])))
        else:
            print('Wrong output_full parameter. Options: True / False')
        #Return solution
        return f
