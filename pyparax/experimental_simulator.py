import pyparax.external_imports as external_imports
import pyparax.internal_imports as internal_imports

class experimental_simulator_1d:
    """
    Collection of functions that extend the capabilities of the numerical solver in solver.numerical_fourier_solver_1d.
    
    Extensions include:
        - introduction of optical elements (e.g. free space domains, lenses, phase masks, amplitude masks)
        - propagation through optical systems that are composed of known optical elements
        - reduction of numerical artifacts due to alias in the computation of lens phase

    OPTICAL ELEMENTS
-------------------------------------------------------------------------------    
    List of optical elements and their definition:
        free space domain = float; real; <SYNTAX: d>, where
            d = distance of the free space domain (same unit as in numeric_parameters.py)
        lens = list; <SYNTAX: ['l', f, error_x, error_y]>, where:
            'l' = the label for lens
            f = the focal length (same unit as in numeric_parameters.py) 
            error_x = shift along the X axis
            error_y = shift along the Y axis
        phase mask = list; <SYNTAX: ['mp', M, error_x, error_y]>, where:
            'mp' = the label for phase mask
            M = The phase mask (array with shape identical to that of the transverse spatial domain)
            error_x = shift along the X axis
            error_y = shift along the Y axis
        amplitude mask = list; <SYNTAX: ['ma', M, error_x, error_y]>, where:
            'ma' = the label for amplitude mask
            M = The amplitude mask (array with shape identical to that of the transverse spatial domain)
            error_x = shift along the X axis
            error_y = shift along the Y axis
-------------------------------------------------------------------------------

    OPTICAL SYSTEMS
-------------------------------------------------------------------------------    
    An optical system is defined as a list that contains optical elements in the same order as the direction of propagation e.g.
        
        <SYNTAX: optical_system = [100, ['l', 50, 0.5, 0.5], 400, ['mp', M, -0.2, 0.2], 200]>
        
    Assuming that the measurement unit that is used in numeric_parameters.py is mm, the optical system above is composed of:
        - 100: free space domain of 100mm
        - ['l', 50, 0.5, 0.5]: lens with 50mm focal length that is shifted +0.5mm on the X and Y axes
        - 400: free space domain of 400mm
        - ['mp', M, -0.2, 0.2]: phase mask given by array M that is shifted -0.2mm on the X axis and 0.2mm on the Y axis
        - 200: free space domain of 200mm
        
    Node: IT IS HIGHLY RECOMMENDED THAT ALL OPTICAL ELEMENTS EXCEPT FREE SPACE DOMAINS TO BE SEPARATED BY A FREE SPACE DOMAIN. 
-------------------------------------------------------------------------------

    ERRORS OF OPTICAL ELEMENTS
-------------------------------------------------------------------------------
    Any optical system has associated with an error that can be considered for system analysis. Error types based on the optical elements types are:
        - free space domains: error of distance measurement
        - lenses and masks: error of focal length, accidental X and Y axes shifts
        
    The errors are given by a list, similar to the way an optical system is defined e.g.
        
        <SYNTAX: errors_optical_system = [5, [2, 0.5, 0.4], 4, [0.3, 0.2], 2]>
        
    for the optical system
    
        optical_system = [100, ['l', 50, 0.5, 0.5], 400, ['mp', M, -0.2, 0.2], 200].
        
    Interpretation of the errors_optical_system syntax given optical_system is as follows:
        - 5: the free space domain interval is (100-5, 100+5)
        - [2, 0.5, 0.5]: focal length interval is (50-2, 50+2), lens shift interval on the X axis is (0.5-0.5, 0.5+0.5), lens shift interval on the Y axis is (0.5-0.4, 0.5+0.4)
        - 4: the free space domain interval is (400-4, 400+4)
        - [0.3, 0.2]: mask shift interval on the X axis is (-0.2-0.3, -0.2+0.3), mask shift interval on the Y axis is (0.2-0.2, 0.2+0.2)
        - 2: the free space domain interval is (200-2, 200+2)
            
    Tilts of lenses and masks are not considered.
-------------------------------------------------------------------------------

    ALIAS AND PROPAGATION 
-------------------------------------------------------------------------------        
    The reduction of numerical artifacts is based on the identification of aliases in the phase mask of the lenses that are used to define the optical system. The following scenarios are covered:
        - if the lens has an alias in normal space, then its phase is computed in Fourier space
        - if the lens has an alias in fourier space, then its phase is computed in normal space
    This is used when propagating the beam profile in order to avoid useless FFTs and iFFTs applications. Consider the optical system
    
        optical_system = [d1, ['l', f, 0, 0], d2].
    
    These scenarios arise:
        - if the phase is computed in Fourier space, the propagation through the free space of length d1 is expressed in Fourier space. Also, the initial condition for the propagation through the free space of length d2 is considered in Fourier space
        - if the phase is computed in normal space, the propagation through the free space of length d1 is expressed in normal space. Also, the initial condition for the propagation through the free space of length d2 is considered in normal space
    """
    def check_lens_for_alias(focal, x_dom = None, use_prints = True):
        """
        Based on the parameters in numeric_parameters.py and the desired focal length, the lens phase mask is checked for alias. The check is done in normal and fourier space.
        Inputs:
            focal = float; real - The focal length of the lens
            x_dom = 1_dimensional array; float - The transversal spatial domain on the X axis. 
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            output = string - Returns one of the 4 possible scenarios:
                1) 'pass' - if no alias is found in both normal and Fourier space
                2) 'n' - if alias is found in normal space
                3) 'f' - if alias is found in Fourier space
        Note: Based on my experience, scenarios 2 and 3 are the most common. Scenario 1 is expected for small values of internal_imports.p.N_x variable.
        """
        
        #Initialize output variable
        output = 'pass'
        
        #Compute wavenumber
        k = 2*external_imports.np.pi/internal_imports.p.wavelength
        
        #Compute space domain
        if x_dom  is None:
            x_dom = internal_imports.ini.standard_initial_conditions.spatial_domain_generator(dim = 1)
        
        #Compute the phase function in normal space 
        phase = k/2/focal * x_dom**2
        
        #Differentiate the phase
        diff_phase_x = external_imports.np.abs(external_imports.np.diff(phase))
        
        #Check if the phase changes faster than pi. If True, then it is considered that an alias has been found in normal space
        if external_imports.np.max(diff_phase_x)>external_imports.np.pi:
            if use_prints == True:
                print('Alias in normal space.')
            output = 'n'
        
        #Compute the frequency domain
        freqs_x = external_imports.np.fft.fftshift(external_imports.np.fft.fftfreq(internal_imports.p.N_x, internal_imports.p.dx))
        
        #compute the phase function in Fourier space
        phase = 2*external_imports.np.pi**2*focal/k*(freqs_x**2)
        
        #Differentiate the phase
        diff_phase_x = external_imports.np.abs(external_imports.np.diff(phase))
        
        #Check if the phase changes faster than pi. If True, then it is considered that an alias has been found in Fourier space
        if external_imports.np.max(diff_phase_x)>external_imports.np.pi:
            if use_prints == True:
                print('Alias in frequency space.')
            #Check if an alias in normal space has already been found.
            if output == 'n':
                x_dom_size = external_imports.np.size(x_dom)
                x_dom = x_dom[int(x_dom_size/4):int(3*x_dom_size/4)]
                
                #Recall function for smaller domain to check which alias still appears
                output = experimental_simulator_1d.check_lens_for_alias(focal, x_dom = x_dom, use_prints = use_prints)
            else:
                output = 'f'
        return output
    
    def check_lenses_in_system_for_alias(system, use_prints = True):
        """
        Identifies all the aliases for each lens in an optical system by using check_lens_for_alias function. Based on the alias information, the lens phase maskes are computed in the space where no alias is found.
        Inputs:
            system = list - The optical system. Check class documentation for more information on how the optical system is defined.
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            system_warnings = list - A list that speficies the space in which the computation is made.
            
        Note: The output format is related to the optical system. Consider the optical system 
        
            optical_system = [100, ['l', 10, 0, 0], 110, ['l', 100, 0, 0], 100]
            
        and consider that the first lens has an alias in normal space, while the second lens has an alias in fourier space. The output in this case is
        
            system_warnings = [['normal', 'fourier'], 'n', ['fourier', 'normal'], 'f', ['normal', 'normal']].
        
        See solver.numeric_fourier_solver_1d() variables fft_input and fft_output, and experimental_simulator_1d.propagate() function for more information.
        """
        
        #Initialize the output variable
        system_warnings = []
        for i in system:
            #Check if optical element is a free space domain and if True append the fft_input/fft_output term
            if external_imports.np.size(i) == 1:
                system_warnings.append(['normal', 'normal'])
            #Check if optical element is anything but a free space domain
            if external_imports.np.size(i) > 1:
                #If it is lens, append the type of alias it has. Otherwise, append 'pass'.
                if i[0] == 'l':
                    system_warnings.append(experimental_simulator_1d.check_lens_for_alias(i[1], use_prints = use_prints))
                else:
                    system_warnings.append('pass')
        system_length = len(system_warnings)
        
        #Change the fft_input/fft_output terms acording to the types of alias found above
        for i in external_imports.np.arange(system_length):
            if system_warnings[i] == 'n':
                if i>0:
                    system_warnings[i-1][1] = 'fourier'
                if i<system_length-1:
                    system_warnings[i+1][0] = 'fourier'
            if system_warnings[i] == 'f':
                if i>0:
                    system_warnings[i-1][1] = 'normal'
                if i<system_length-1:
                    system_warnings[i+1][0] = 'normal'
        return system_warnings                    
    
    def propagate(f0, system, forward = True, norm = True, output_full = True, print_output = False, use_prints = True):
        """
        Propagates the beam through the optical system.
        Inputs:
            f0 = 1-dimensional array; complex - The input beam profile.
            system = list - The optical system.
            forward = boolean - If False, the propagation direction through the optical system is reversed.
            norm = boolean - If True, the beam profile is normalized at each step on the propagation axis when print_output = True. Good for visualising the profile on the entire 1+1-dimensional space.
            output_full = boolean - If True, the function returns the solution for each iteration. If False, only the last iteration is returned.
            print_output = boolean - If True, the solution is ploted. Depends on output_full variable.
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            CONDITION: 
                if output_full = True
                    f_total = 2-dimensional array; complex - The beam profile for the entire optical system.
                if output_full = False
                    f_final = 1-dimensional array; complex - The beam profile at the end of the optical system. 
        """
        
        #Check if the optical system given by [0].
        if system == [0]:
            return f0
        
        #Check if the first and the last elements are anything but free space domains and if False, add a free space domain of length dz (Check class description, OPTICAL SYSTEMS, Note)
        if type(system[-1]) == list:
            system.append(1*internal_imports.p.dz)
        system.reverse()
        if type(system[-1]) == list:
            system.append(1*internal_imports.p.dz)
        system.reverse()
        
        #Compute the size of the optical system
        system_length = external_imports.np.size(system)
        
        #Main computation done by cases 
        if output_full == True:
            if forward == True:
                #Case 1: output_full == True and forward == True
                
                #Compute system_warnings and print, if needed, for debuging
                system_warnings = experimental_simulator_1d.check_lenses_in_system_for_alias(system, use_prints = use_prints)
                if use_prints == True:
                    print(system_warnings)
                
                #Initialize the output array
                f_total = external_imports.np.reshape(f0, (1,external_imports.np.size(f0))) + 0j
                
                #Propagate beam through the optical system optical element by optical element. 
                for i in external_imports.np.arange(system_length):
                    
                    #Check if optical element is free space domain and propagate with solver.numeric_fourier_solver_1d()
                    if type(system[i]) == int or type(system[i]) == float:
                        
                        #Print operation.
                        if use_prints == True:
                            print('Propagate for '+str(system[i])+' units [mm]')
                        
                        #Check for non-zero length of free space domain
                        if system[i] != 0:
                            
                            #Propagate according to all the input parameters. Use system_warnings for the fft_input/fft_output varibles
                            f = internal_imports.s.numeric_fourier_solver_1d.linear(f0, wavelength=internal_imports.p.wavelength, n0 = internal_imports.p.n0, print_progress=False, steps = int(system[i]/internal_imports.p.dz), output_full = True, fft_input=(system_warnings[i][0]=='fourier'), fft_output=(system_warnings[i][1]=='fourier'))
                            if system_warnings[i][1] == 'fourier':
                                temp_f = external_imports.np.zeros_like(f)
                                for j in range(external_imports.np.shape(f)[0]):
                                    temp_f[j,:] = external_imports.np.fft.ifftshift(external_imports.np.fft.ifft(external_imports.np.fft.ifftshift(f[j,:])))
                                f_total = external_imports.np.append(f_total,temp_f, axis = 0)
                                del temp_f
                            if system_warnings[i][1] == 'normal':
                                f_total = external_imports.np.append(f_total, f, axis = 0)
                                
                            #Keep the last profile for the next iteration.
                            f0 = f[-1,:]
                    
                    #Check if optical element is not a free space domain.
                    if type(system[i]) == list:
                        
                        #Check if optical element is lens
                        if system[i][0] == 'l':
                            
                            #Check if alias in normal space
                            if system_warnings[i] == 'n':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                
                                #Compute phase in Fourier space 
                                phase = internal_imports.ini.optical_elements.lens_theory(system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], dim = 1, fft_flag = True)
                                
                                #Compute normalization constant for initial beam profile
                                norm_const = external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                                #Apply the lens phase mask by convolution with the initial beam profile  
                                f0 = external_imports.sps.convolve(phase, f0, mode = 'same')
                                
                                #Renormalize the resulting profile
                                f0 = norm_const*f0/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                            
                            #Check if alias in Fourier space
                            if system_warnings[i] == 'f':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                
                                #Compute phase in normal space
                                phase = internal_imports.ini.optical_elements.lens_theory(system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], dim = 1, fft_flag = False)
                                
                                #Apply the lens phase mask by product
                                f0 = f0*phase
                                
                        #Check if phase mask
                        if system[i][0] == 'mp':
                            if use_prints == True:    
                                print('Apply mask phase.')
                            
                            #Shift mask
                            mask = mask_generator_1d.mask_shift(system[i][1], system[i][2])
                            
                            #Apply phase mask by product
                            f0 = f_total[-1,:]*external_imports.np.exp(1j*mask)
                            
                        #Check if amplitude mask
                        if system[i][0] == 'ma':
                            if use_prints == True:
                                print('Apply mask amplitude')
                            
                            #Shift mask
                            mask = mask_generator_1d.mask_shift(system[i][1], system[i][2])
                            
                            #Apply amplitude mask by product
                            f0 = f_total[-1,:]*system[i][1]
                            
            if forward == False:
                #Case 2: output_full == True and forward == False
                
                #Initialize the output array
                f_total = external_imports.np.reshape(f0, (1,external_imports.np.size(f0))) + 0j
                
                #Reverse order of optical elements in system
                system.reverse()
                
                #Compute system_warnings and print, if needed, for debuging
                system_warnings = experimental_simulator_1d.check_lenses_in_system_for_alias(system, use_prints = use_prints)
                if use_prints == True:
                    print(system_warnings)
                    
                #Propagate beam through the optical system optical element by optical element. 
                for i in external_imports.np.arange(system_length):
                    
                    #Check if optical element is free space domain and propagate with solver.numeric_fourier_solver_1d()
                    if type(system[i]) == int or type(system[i]) == float:
                        
                        #Print operation.
                        if use_prints == True:
                            print('Propagate for '+str(system[i])+' units [mm]')
                        
                        #Check for non-zero length of free space domain
                        if system[i] != 0:
                            
                            #Propagate according to all the input parameters. Use system_warnings for the fft_input/fft_output varibles
                            f = internal_imports.s.numeric_fourier_solver_1d.linear(f0, wavelength=internal_imports.p.wavelength, n0 = internal_imports.p.n0, print_progress=False, steps = int(system[i]/internal_imports.p.dz), output_full = True, forward = False, fft_input=(system_warnings[i][0]=='fourier'), fft_output=(system_warnings[i][1]=='fourier'))
                            if system_warnings[i][1] == 'fourier':
                                temp_f = external_imports.np.zeros_like(f)
                                for j in range(external_imports.np.shape(f)[0]):
                                    temp_f[j,:] = external_imports.np.fft.ifftshift(external_imports.np.fft.ifft(external_imports.np.fft.ifftshift(f[j,:])))
                                f_total = external_imports.np.append(f_total,temp_f, axis = 0)
                                del temp_f
                            if system_warnings[i][1] == 'normal':
                                f_total = external_imports.np.append(f_total, f, axis = 0)
                            
                            #Keep the last profile for the next iteration.
                            f0 = f[-1,:]
                    
                    #Check if optical element is not a free space domain.
                    if type(system[i]) == list:
                        
                        #Check if optical element is lens
                        if system[i][0] == 'l':
                            
                            #Check if alias in normal space
                            if system_warnings[i] == 'n':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in Fourier space 
                                phase = internal_imports.ini.optical_elements.lens_theory(-system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], dim = 1, fft_flag = True)
                                
                                #Compute normalization constant for initial beam profile
                                norm_const = external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                                #Apply the lens phase mask by convolution with the initial beam profile  
                                f0 = external_imports.sps.convolve(phase, f0, mode = 'same')
                                
                                #Renormalize the resulting profile
                                f0 = norm_const*f0/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                            #Check if alias in Fourier space
                            if system_warnings[i] == 'f':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                
                                #Compute phase in normal space
                                phase = internal_imports.ini.optical_elements.lens_theory(-system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], dim = 1, fft_flag = False)
                                
                                #Apply the lens phase mask by product
                                f0 = f0*phase
                            
                        #Check if phase mask
                        if system[i][0] == 'mp':
                            if use_prints == True:
                                print('Apply mask phase.')
                                
                            #Shift mask
                            mask = mask_generator_1d.mask_shift(system[i][1], system[i][2])
                            
                            #Apply phase mask by product
                            f0 = f_total[-1,:]*external_imports.np.exp(-1j*mask)
                        
                        #Check if amplitude mask
                        if system[i][0] == 'ma':
                            if use_prints == True:
                                print('Apply mask amplitude')
                                
                            #Shift mask
                            mask = mask_generator_1d.mask_shift(system[i][1], system[i][2])
                            
                            #Apply amplitude mask by product
                            f0 = f_total[-1,:]*mask
                            
                #Reverse order of optical elements in system back to the original order
                system.reverse()
            
            #Plot the full beam profile
            if print_output == True:
                
                #Width of the optical elements in the plot representation
                width = 5
                
                #Copy the beam profile in order to not change the original values
                f_temp = external_imports.np.copy(f_total)
                
                #Normalize the beam profile if norm == True. Maximum amplitude is 1
                if norm == True:
                    for i in range(external_imports.np.shape(f_temp)[0]):
                        f_temp[i,:]=f_temp[i,:]/external_imports.np.max(external_imports.np.abs(f_temp[i,:]))
                
                #Initialize idx valiable
                idx = 0
                
                #Add some specific patterns in order to faster identify each non free space domain optical element.
                try:
                    
                    #Reverse the system if needed
                    if forward == False:
                        system.reverse()
                        
                    #Check each element in the optical system
                    for i in system:
                        
                        #Check if free space domain and increase idx variable
                        if type(i) == int or type(i) == float:
                            idx = idx + int(i/internal_imports.p.dz)
                        
                        #Check if non free space domain
                        if type(i) == list:
                            
                            #Check if lens and if True mark a region of width (2*width) variable with value 1
                            if i[0] == 'l':
                                f_temp[idx-width:idx+width,:] = external_imports.np.ones((width*2, external_imports.np.shape(f_temp)[1])) 
                                
                            #Check if phase mask and if True mark a region of width equal to (2*width) with alternating values of 0 and 1
                            if i[0] == 'mp':
                                f_temp[idx-width:idx+width,:] = external_imports.np.ones((width*2, external_imports.np.shape(f_temp)[1])) * external_imports.np.transpose(external_imports.np.arange(external_imports.np.shape(f_temp)[1]))%2
                                
                    #Plot the resulting beam with the specific patterns added to it
                    external_imports.plt.imshow(external_imports.np.abs(external_imports.np.transpose(f_temp))**2)
                    external_imports.plt.xlabel(r"$z/\Delta z, \Delta z = $" + str(internal_imports.p.dz) + str(internal_imports.p.unit))
                    external_imports.plt.ylabel(r"$x/\Delta x, \Delta x = $" + str(internal_imports.p.dx) + str(internal_imports.p.unit))
                    
                    #Reverse the system if needed
                    if forward == False:
                        system.reverse()
                        
                #If try block fails, just plot the beam profile without specific patterns
                except:
                    external_imports.plt.imshow(external_imports.np.abs(external_imports.np.transpose(f_temp))**2)
                    external_imports.plt.xlabel(r"$z/\Delta z, \Delta z = $" + str(internal_imports.p.dz) + str(internal_imports.p.unit))
                    external_imports.plt.ylabel(r"$x/\Delta x, \Delta x = $" + str(internal_imports.p.dx) + str(internal_imports.p.unit))
                del f_temp

            return f_total
        else:
            if forward == True:
                #Case 3: output_full == False and forward == True
                
                #Compute system_warnings and print, if needed, for debuging
                system_warnings = experimental_simulator_1d.check_lenses_in_system_for_alias(system, use_prints = use_prints)
                if use_prints == True:
                    print(system_warnings)
                
                #Initialize the output array
                f_final = external_imports.np.reshape(f0, (1,external_imports.np.size(f0))) + 0j
                
                #Propagate beam through the optical system optical element by optical element. 
                for i in external_imports.np.arange(system_length):
                    
                    #Check if optical element is free space domain and propagate with solver.numeric_fourier_solver_1d()
                    if type(system[i]) == int or type(system[i]) == float:
                        
                        #Print operation
                        if use_prints == True:
                            print('Propagate for '+str(system[i])+' units [mm]')
                            
                        #Check for non-zero length of free space domain
                        if system[i] != 0:
                            
                            #Propagate according to all the input parameters. Use system_warnings for the fft_input/fft_output varibles
                            f_final = internal_imports.s.numeric_fourier_solver_1d.linear(f0, wavelength=internal_imports.p.wavelength, n0 = internal_imports.p.n0, print_progress=True, steps = int(system[i]/internal_imports.p.dz), output_full = False, fft_input=(system_warnings[i][0]=='fourier'), fft_output=(system_warnings[i][1]=='fourier'))
                            
                            #Copy the output profile
                            f0 = external_imports.np.copy(f_final)
                            
                    #Check if optical element is not a free space domain
                    if type(system[i]) == list:
                        
                        #Check if optical element is lens
                        if system[i][0] == 'l':
                            
                            #Check if alias in normal space
                            if system_warnings[i] == 'n':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in Fourier space 
                                phase = internal_imports.ini.optical_elements.lens_theory(system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], dim = 1, fft_flag = True)
                                
                                #Compute normalization constant for initial beam profile
                                norm_const = external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                                #Apply the lens phase mask by convolution with the initial beam profile  
                                f0 = external_imports.sps.convolve(phase, f0, mode = 'same')
                                
                                #Renormalize the resulting profile
                                f0 = norm_const*f0/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                            #Check if alias in Fourier space
                            if system_warnings[i] == 'f':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                
                                #Compute phase in normal space
                                phase = internal_imports.ini.optical_elements.lens_theory(system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], dim = 1, fft_flag = False)
                                
                                #Apply the lens phase mask by product
                                f0 = f0*phase
                        
                        #Check if phase mask
                        if system[i][0] == 'mp':
                            if use_prints == True:
                                print('Apply mask phase.')
                                
                            #Shift mask
                            print(system[i])
                            mask = mask_generator_1d.mask_shift(system[i][1], system[i][2])
                            
                            #Apply phase mask by product
                            f0 = f0*external_imports.np.exp(1j*mask)
                            
                        #Check if amplitude mask
                        if system[i][0] == 'ma':
                            if use_prints == True:
                                print('Apply mask amplitude')
                                
                            #Shift mask
                            mask = mask_generator_1d.mask_shift(system[i][1], system[i][2])
                                
                            #Apply amplitude mask by product
                            f0 = f0*mask
                            
                            
            if forward == False:
                #Case 4: output_full == False and forward == False
                
                #Initialize the output array
                f_final = external_imports.np.reshape(f0, (1,external_imports.np.size(f0))) + 0j
                
                #Reverse order of optical elements in system
                system.reverse()
                
                #Compute system_warnings and print, if needed, for debuging
                system_warnings = experimental_simulator_1d.check_lenses_in_system_for_alias(system, use_prints = use_prints)
                if use_prints == True:
                    print(system_warnings)
                    
                #Propagate beam through the optical system optical element by optical element. 
                for i in external_imports.np.arange(system_length):
                    
                    #Check if optical element is free space domain and propagate with solver.numeric_fourier_solver_1d()
                    if type(system[i]) == int or type(system[i]) == float:
                        
                        #Print operation.
                        if use_prints == True:
                            print('Propagate for '+str(system[i])+' units [mm]')
                        
                        #Check for non-zero length of free space domain
                        if system[i] != 0:
                            
                            #Propagate according to all the input parameters. Use system_warnings for the fft_input/fft_output varibles
                            f_final = internal_imports.s.numeric_fourier_solver_1d.linear(f0, wavelength=internal_imports.p.wavelength, n0 = internal_imports.p.n0, print_progress=True, steps = int(system[i]/internal_imports.p.dz), output_full = False, forward = False, fft_input=(system_warnings[i][0]=='fourier'), fft_output=(system_warnings[i][1]=='fourier'))
                            
                            #Copy the output profile
                            f0 = external_imports.np.copy(f_final)
                    
                    #Check if optical element is not a free space domain.
                    if type(system[i]) == list:
                        
                        #Check if optical element is lens
                        if system[i][0] == 'l':
                            
                            #Check if alias in normal space
                            if system_warnings[i] == 'n':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in Fourier space 
                                phase = internal_imports.ini.optical_elements.lens_theory(-system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], dim = 1, fft_flag = True)
                                
                                #Compute normalization constant for initial beam profile
                                norm_const = external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                                #Apply the lens phase mask by convolution with the initial beam profile  
                                f0 = external_imports.sps.convolve(phase, f0, mode = 'same')
                                
                                #Renormalize the resulting profile
                                f0 = norm_const*f0/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                            #Check if alias in Fourier space
                            if system_warnings[i] == 'f':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in normal space
                                phase = internal_imports.ini.optical_elements.lens_theory(-system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], dim = 1, fft_flag = False)
                                
                                #Apply the lens phase mask by product
                                f0 = f0*phase
                            
                        #Check if phase mask
                        if system[i][0] == 'mp':
                            if use_prints == True:
                                print('Apply mask phase.')
                                
                            #Shift mask
                            mask = mask_generator_1d.mask_shift(system[i][1], system[i][2])
                            
                            #Apply phase mask by product
                            f0 = f_final*external_imports.np.exp(-1j*mask)
                            
                        #Check if amplitude mask
                        if system[i][0] == 'ma':
                            if use_prints == True:
                                print('Apply mask amplitude')
                                
                            #Shift mask
                            mask = mask_generator_1d.mask_shift(system[i][1], system[i][2])    
                            
                            #Apply amplitude mask by product
                            f0 = f_final*mask
                            
                #Reverse order of optical elements in system back to the original order
                system.reverse()
            
            #Plot the beam profile
            if print_output == True:                    
                external_imports.plt.plot(external_imports.np.abs(f_final))
                external_imports.plt.xlabel(r"$x/\Delta x, \Delta x = $" + str(internal_imports.p.dx) + str(internal_imports.p.unit))
            
            return f_final
        
    def monte_carlo_precision_test(f_in, system, system_errors, number_of_samples):
        """
        Simulate the propagation of a beam considering errors regarding the positioning of optical elements. Based of the errors, various scenarios are considered and simulated in order to check how errors influence the optical system.
        
        The sampling assumes a uniform distribution inside the intervals.
        
        Inputs:
            f_in = 1-dimensional array; complex - The input beam profile.
            system = list - The optical system.
            system_errors = list - The errors related to the optical elements.
            number_of_samples = float - Number of parameters sampled and of simulations done.
        Outputs:
            f_out = 1-dimensional array; complex - The beam profile at the end of the optical system.
        """
        
        #Check if system and system_errors are the same length
        if len(system) != len(system_errors):
            print('len(system) != len(system_errors)')
        
        system_length = len(system)
        system_size = 0
        for i in range(system_length):
            if type(system[i]) == int or type(system[i]) == float:
                system_size += system[i]
        f_out = external_imports.np.zeros((int(system_size/internal_imports.p.dz+1), internal_imports.p.N_x))
        
        #Compute the solutions for each case and add together the resulting output amplitude profiles. Good approach for fast visualisation. 
        for i in external_imports.np.arange(number_of_samples):
            
            #Initialize temporal optical system
            temp_system = []
            
            #Sample values for parameters of the temporal optical system
            for j in external_imports.np.arange(system_length):
                
                #Sample value for free space domain
                if type(system[j]) == int or type(system[j]) == float:
                    temp_system.append(system[j] + int(((external_imports.np.random.random()-0.5)*2*system_errors[j]) // internal_imports.p.dz) * internal_imports.p.dz)
                
                #Sample values for other optical elements
                if type(system[j]) == list:
                    
                    #Sample values for lens
                    if system[j][0] == 'l':
                        
                        #Initialize optical element and append it to the temporal optical system
                        temp_system.append([])
                        
                        #Append lens tag to the optical element
                        temp_system[j].append('l')
                        
                        #Compute and append focal length value to the optical element
                        val = float(external_imports.np.copy(system[j][1]) + round(((external_imports.np.random.random()-0.5)*2*system_errors[j][1])//internal_imports.p.dx) * internal_imports.p.dx)
                        temp_system[j].append(val)
                        
                        #Compute and append shift value on the X axis 
                        val = float(external_imports.np.copy(system[j][2]) + round(((external_imports.np.random.random()-0.5)*2*system_errors[j][2])//internal_imports.p.dx) * internal_imports.p.dx)
                        temp_system[j].append(val)
                        
                    #Sample values for phase mask
                    if system[j][0] == 'mp':
                        
                        #Initialize optical element and append it to the temporal optical system
                        temp_system.append([])
                        
                        #Append lens tag to the optical element
                        temp_system[j].append('mp')
                        
                        #Append phase mask
                        shifted_mask = external_imports.np.copy(system[j][1])
                        
                        #Shift the phase mask and append the result to the optical element 
                        shifted_mask = mask_generator_1d.mask_shift(shifted_mask, external_imports.np.copy(system[j][2]) + (external_imports.np.random.random()-0.5)*2*system_errors[j][2])
                        temp_system[j].append(shifted_mask)
                        
                        val = float(external_imports.np.copy(system[j][2]) + round(((external_imports.np.random.random()-0.5)*2*system_errors[j][2])//internal_imports.p.dx) * internal_imports.p.dx)
                        temp_system[j].append(val)
                        
                    #Sample values for amplitude mask
                    if system[j][0] == 'ma':
                        
                        #Initialize optical element and append it to the temporal optical system
                        temp_system.append([])
                        
                        #Append lens tag to the optical element
                        temp_system[j].append('ma')
                        
                        #Append phase mask
                        shifted_mask = external_imports.np.copy(system[j][1])
                        
                        #Shift the phase mask and append the result to the optical element
                        shifted_mask = mask_generator_2d.mask_shift(shifted_mask, external_imports.np.copy(system[j][2]) + (external_imports.np.random.random()-0.5)*2*system_errors[j][2])
                        temp_system[j].append(shifted_mask)
                        
                        val = float(external_imports.np.copy(system[j][2]) + round(((external_imports.np.random.random()-0.5)*2*system_errors[j][2])//internal_imports.p.dx) * internal_imports.p.dx)
                        temp_system[j].append(val)
                        
            print(temp_system)
            
            #Compute the full solution for the temporal optical system with sampled errors
            temp_f_out = external_imports.np.abs(experimental_simulator_1d.propagate(f_in, temp_system, forward = True, norm = True, output_full = True, print_output = False, use_prints = False))**2
            f_out = f_out + temp_f_out
            
            #Compute the size of the temporal solution along the propagation axis
            size_temp_f_out = external_imports.np.shape(temp_f_out)[0]
            
            #Initialize an array to store the positions of the maximum amplitude
            temp_max_f_out = external_imports.np.zeros(size_temp_f_out)
            
            #Find the positions of the intensity maxima of the full solution            
            for j in external_imports.np.arange(size_temp_f_out):
                temp_max_f_out[j] = external_imports.np.where(temp_f_out[j, :] == external_imports.np.max(temp_f_out[j,:]))[0][0]
            
            #Plot the trajectory of the intensity maxima
            external_imports.plt.plot(temp_max_f_out)
            external_imports.plt.ylim([0, internal_imports.p.N_x])
            
            #Print step
            print("Step: " + str(i+1) + "/"+str(number_of_samples))
        return f_out

class mask_generator_1d:
    def compute_mask_dual_system(system1, system2, f_in, f_out, check_mask = False, print_output = False, norm = True, use_prints = True):
        """
        Computes the phase mask for a specific optical system that is split in 2 subsystems.
        
        System diagram:
                  Subsystem1               Subsystem2
        Input[------------------]MASK[-------------------]Output
        
        The entire optical system is given by:
            
            optical_system = system1 + ['mp', MASK] + system2
        
        Inputs:
            system1 = list - Subsystem1.
            system2 = list - Subsystem2.
            f_in = 1-dimensional array; complex - The initial condition.
            f_out = 1-dimensional array; complex - The desired output beam.
            check_mask = boolean - If True, calls check_output_for_mask function.
            print_output = boolean - If True, the whole beam profile is ploted.
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            mask = 1-dimensional array; real - The phase mask
        """
        
        #Normalize in input and output beam profiles
        norm_in = external_imports.np.sum(external_imports.np.abs(f_in)**2)
        norm_out = external_imports.np.sum(external_imports.np.abs(f_out)**2)
        
        f_in = f_in/external_imports.np.sqrt(norm_in)
        f_out = f_out/external_imports.np.sqrt(norm_out)
        
        #Propagate the input beam forwards through system1
        f1 = experimental_simulator_1d.propagate(f_in, system1, forward = True, norm = False, output_full=False)+0j
        
        #Propagate the output beam backwards through system2
        f2 = experimental_simulator_1d.propagate(f_out, system2, forward = False, norm = False, output_full=False)+0j

        #Compute the phase mask
        try:
            mask = external_imports.np.angle(f2/f1)
        except:
            mask = external_imports.np.angle(external_imports.np.exp(1j*external_imports.np.angle(f2))/external_imports.np.exp(1j*external_imports.np.angle(f1)))
        
        #Check the phase mask
        if check_mask == True:
            mask_generator_1d.check_output_for_mask(system1 + [['mp', mask, 0, 0]] + system2, f_in, f_out, use_prints = True, output = False)
        
        #Print the entire beam through the optical system
        if print_output == True:
            f = experimental_simulator_1d.propagate(f_in, system1 + [['mp', mask,0,0]] + system2, forward = True, norm = norm, output_full=True, print_output = True)
        
        return mask
    
    def check_output_for_mask(system, f_in, f_out, output = True, use_prints = True, ploting = False):
        """
        Computes the beam profile through an optical system given by
            
            optical_system = system1 + [['mp', mask]] + system2
            
        and then compare the computed output with the desired one (f_out).
        
        Inputs:
            system = list - The entire optical system with phase mask.
            f_in = 1-dimensional array; complex - The initial condition.
            f_out = 1-dimensional array; complex - The desired output beam.
            output = boolean - If True, the function returns the convolution of the 2 outputs and the shift.
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            CONDITION:
                if output = True
                    corr = 1-dimensional array; complex - The cross-correlation between the computed output and the desired one
                    shift = int - The difference in position between the 2 outputs, measured in pixels. 
                if output = False
                    None
        """
        
        #Print operation
        if use_prints == True:
            print("Checking mask.")
        
        #Propagate input profile through the entire optical system
        f = experimental_simulator_1d.propagate(f_in, system, forward = True, norm = False, output_full=False)
        
        #Plot the computed and desired amplitude profiles for qualitative comparison
        if ploting == True:
            external_imports.plt.figure()
            external_imports.plt.plot(external_imports.np.abs(f_out)**2)
            external_imports.plt.plot(external_imports.np.abs(f)**2)
            external_imports.plt.title(r'$f_{out} vs. f_{retrieved}$')
        
        #Normalize the 2 outputs
        f_norm = f/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f)**2))
        print(external_imports.np.shape(f_norm))
        f_out_norm = f_out/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f_out)**2))
        print(external_imports.np.shape(f_out_norm))
        
        #Compute the cross-correlation between them
        corr = external_imports.sps.correlate(f_out_norm, f_norm, mode = 'same')
        
        #Print the maximum cross-correlation value 
        if use_prints == True:
            print('Max correlation: ' + str(external_imports.np.max(external_imports.np.abs(corr))))
        
        #Compute the difference in position between the 2 outputs 
        shift = external_imports.np.where(external_imports.np.abs(corr) == external_imports.np.max(external_imports.np.abs(corr)))[0][0] - int(external_imports.np.shape(corr)[0]/2)
        
        #Print it
        if use_prints == True:
            print('Shift: ' + str(shift))
        
        #Return the outputs
        if output == True:
            return corr, shift 
        
    def compute_mask_triple_system(system1, system2, system3, f_in, f_out, check_mask = False, print_output = False, norm = True, use_prints = True):
        """
        Computes the phase mask for a specific optical system that is split in 3 subsystems.
        
        System diagram:
                  Subsystem1                 Subsystem2                 Subsystem3
        Input[------------------]MASK_1[-------------------]MASK_2[-------------------]Output
        
        The entire optical system is given by:
            
            optical_system = system1 + [['mp', MASK_1]] + system2 + [['mp', MASK_2]] + system3
        
        Inputs:
            system1 = list - Subsystem1.
            system2 = list - Subsystem2.
            system3 = list - Subsystem3.
            f_in = 1-dimensional array; complex - The initial condition.
            f_out = 1-dimensional array; complex - The desired output beam.
            check_mask = boolean - If True, calls check_output_for_mask function.
            print_output = boolean - If True, the whole beam profile is ploted.
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            mask1 = 1-dimensional array; real - The phase mask
            mask2 = 1-dimensional array; real - The phase mask
        """
        
        #Normalize in input and output beam profiles
        norm_in = external_imports.np.sum(external_imports.np.abs(f_in)**2)
        norm_out = external_imports.np.sum(external_imports.np.abs(f_out)**2)
        
        f_in = f_in/external_imports.np.sqrt(norm_in)
        f_out = f_out/external_imports.np.sqrt(norm_out)
        
        #Propagate the output beam backwards through system3
        f3 = experimental_simulator_1d.propagate(f_out, system3, forward = False, norm = False, output_full=False)
        
        #Compute MASK_2 considering f3 as output of optical_system = system1 + [['mp, MASK_1]] + system2 
        mask1 = mask_generator_1d.compute_mask_dual_system(system1, system2, f_in, external_imports.np.abs(f3), check_mask = False)
        
        #Compute the MASK_1
        mask2 = mask_generator_1d.compute_mask_dual_system(system1 + [['mp', mask1, 0, 0]] + system2, system3, f_in, f_out, check_mask = False)
        
        ##Check the phase masks
        if check_mask == True:
            mask_generator_1d.check_output_for_mask(system1 + [['mp', mask1, 0, 0]] + system2 + [['mp', mask2, 0, 0]] + system3, f_in, f_out, use_prints = use_prints, output = False)
        
        #Print the entire beam through the optical system
        if print_output == True:
            _ = experimental_simulator_1d.propagate(f_in, system1 + [['mp', mask1, 0, 0]] + system2 + [['mp', mask2, 0, 0]] + system3, forward = True, norm = norm, output_full=True, print_output=True, use_prints=use_prints)
            
        return mask1, mask2
        
    def mask_shift(mask, shift_x):
        """
        Shift the phase mask on the X direction by an amount given by the user.
        
        Inputs:
            mask = 1-dimensional array; real - The phase mask.
            shift_x = float; real - Shift on the X axis. Units are the same as in numeric_parameters.py file. 
        Outputs:
            mask = 1-dimensional array; real - The shifted phase mask.
        """
        
        #Transform the shift from the implied measurement unit into pixels
        shift_x_pixels = int(shift_x / internal_imports.p.dx)
        
        #Apply the shift on the mask
        mask = external_imports.np.roll(mask, shift_x_pixels, axis = 0)
        
        return mask
        
class experimental_simulator_2d:
    """
    Same functions as in experimental_simulator_1d but for 2-dimensional transversal space. Check experimental_simulator_1d for more information.
    """
    
    def check_lens_for_alias(focal, x_dom = None, y_dom = None, use_prints = True):
        """
        Based on the parameters in numeric_parameters.py and the desired focal length, the lens phase mask is checked for alias. The check is done in normal and fourier space.
        Inputs:
            focal = float; real - The focal length of the lens
            x_dom = 1-dimensional array; float - The transversal spatial domain on the X axis. 
            y_dom = 1_dimensional array; float - The transversal spatial domain on the Y axis. 
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            output = string - Returns one of the 4 possible scenarios:
                1) 'pass' - if no alias is found in both normal and Fourier space
                2) 'n' - if alias is found in normal space
                3) 'f' - if alias is found in Fourier space
        Note: Based on my experience, scenarios 2 and 3 are the most common. Scenario 1 is expected for small values of internal_imports.p.N_x variable.
        """
        
        #Initialize output variable
        output = 'pass'
        
        #Compute wavenumber
        k = 2*external_imports.np.pi/internal_imports.p.wavelength
        
        #Compute space domain
        if x_dom is None or y_dom is None:
            x_dom, y_dom = internal_imports.ini.standard_initial_conditions.spatial_domain_generator(dim = 2)
        else:
            x_dom, y_dom = external_imports.np.meshgrid(x_dom, y_dom)
        
        #Compute the phase function in normal space 
        phase = k/2/focal * (x_dom**2+y_dom**2)
        
        #Differentiate the phase
        diff_phase_x = external_imports.np.abs(external_imports.np.diff(phase))
        diff_phase_y = external_imports.np.abs(external_imports.np.diff(external_imports.np.transpose(phase)))
        
        #Check if the phase changes faster than pi. If True, then it is considered that an alias has been found in normal space
        if external_imports.np.max(diff_phase_x)>external_imports.np.pi or external_imports.np.max(diff_phase_y)>external_imports.np.pi:
            if use_prints == True:
                print('Alias in normal space.')
            output = 'n'
            
        #Compute the frequency domain
        freqs_x = external_imports.np.fft.fftfreq(internal_imports.p.N_x, internal_imports.p.dx)
        freqs_y = external_imports.np.fft.fftfreq(internal_imports.p.N_y, internal_imports.p.dy)
        x_freqs, y_freqs = external_imports.np.meshgrid(freqs_x, freqs_y)
        
        #compute the phase function in Fourier space
        phase = 2*external_imports.np.pi**2*focal/k*(x_freqs**2+y_freqs**2)
        
        #Differentiate the phase
        diff_phase_x = external_imports.np.abs(external_imports.np.diff(phase))
        diff_phase_y = external_imports.np.abs(external_imports.np.diff(external_imports.np.transpose(phase)))
        
        #Check if the phase changes faster than pi. If True, then it is considered that an alias has been found in Fourier space
        if external_imports.np.max(diff_phase_x)>external_imports.np.pi or external_imports.np.max(diff_phase_y)>external_imports.np.pi:
            if use_prints == True:
                print('Alias in frequency space.')
                
            #Check if an alias in normal space has already been found.
            if output == 'n':
                x_dom_size = external_imports.np.size(x_dom)
                x_dom = x_dom[int(x_dom_size/4):int(3*x_dom_size/4)]
                y_dom_size = external_imports.np.size(y_dom)
                y_dom = y_dom[int(y_dom_size/4):int(3*y_dom_size/4)]
                
                #Recall function for smaller domain to check which alias still appears 
                output = experimental_simulator_2d.check_lens_for_alias(focal, x_dom = x_dom, y_dom = y_dom, use_prints = use_prints)
            else:
                output = 'f'
        return output
                
    def check_lenses_in_system_for_alias(system, use_prints = True):
        """
        Identifies all the aliases for each lens in an optical system by using check_lens_for_alias function. Based on the alias information, the lens phase maskes are computed in the space where no alias is found.
        Inputs:
            system = list - The optical system. Check class documentation for more information on how the optical system is defined.
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            system_warnings = list - A list that speficies the space in which the computation is made.
            
        Note: The output format is related to the optical system. Consider the optical system 
        
            optical_system = [100, ['l', 10, 0, 0], 110, ['l', 100, 0, 0], 100]
            
        and consider that the first lens has an alias in normal space, while the second lens has an alias in fourier space. The output in this case is
        
            system_warnings = [['normal', 'fourier'], 'n', ['fourier', 'normal'], 'f', ['normal', 'normal']].
        
        See solver.numeric_fourier_solver_1d() variables fft_input and fft_output, and experimental_simulator_1d.propagate() function for more information.
        """
        #Initialize the output variable
        system_warnings = []
        for i in system:
            #Check if optical element is a free space domain and if True append the fft_input/fft_output term
            if external_imports.np.size(i) == 1:
                system_warnings.append(['normal', 'normal'])
            #Check if optical element is anything but a free space domain
            if external_imports.np.size(i) > 1:
                #If it is lens, append the type of alias it has. Otherwise, append 'pass'.
                if i[0] == 'l':
                    system_warnings.append(experimental_simulator_2d.check_lens_for_alias(i[1], use_prints = use_prints))
                else:
                    system_warnings.append('pass')
        system_length = len(system_warnings)
        
        #Change the fft_input/fft_output terms acording to the types of alias found above
        for i in external_imports.np.arange(system_length):
            if system_warnings[i] == 'n':
                if i>0:
                    system_warnings[i-1][1] = 'fourier'
                if i<system_length-1:
                    system_warnings[i+1][0] = 'fourier'
            if system_warnings[i] == 'f':
                if i>0:
                    system_warnings[i-1][1] = 'normal'
                if i<system_length-1:
                    system_warnings[i+1][0] = 'normal'
        return system_warnings                    
    
    
    def propagate(f0, system, forward = True, norm = True, output_full = True, print_output = False, use_prints = True):
        """
        Propagates the beam through the optical system.
        Inputs:
            f0 = 2-dimensional array; complex - The input beam profile.
            system = list - The optical system.
            forward = boolean - If False, the propagation direction through the optical system is reversed.
            norm = boolean - If True, the beam profile is normalized at each step on the propagation axis.
            output_full = boolean - If True, the function returns the solution for each iteration. If False, only the last iteration is returned.
            print_output = boolean - If True, the solution is ploted. Depends on output_full variable.
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            CONDITION: 
                if output_full = True
                    f_total = 3-dimensional array; complex - The beam profile for the entire optical system.
                if output_full = False
                    f_final = 2-dimensional array; complex - The beam profile at the end of the optical system. 
        """
        
        #Check if the optical system given by [0].
        if system == [0]:
            return f0
        
        #Check if the first and the last elements are anything but free space domains and if False, add a free space domain of length dz (Check class description, OPTICAL SYSTEMS, Note)
        if type(system[-1]) == list:
            system.append(1*internal_imports.p.dz)
        system.reverse()
        if type(system[-1]) == list:
            system.append(1*internal_imports.p.dz)
        system.reverse()
        
        #Compute the size of the optical system
        system_length = external_imports.np.size(system)
        
        #Main computation done by cases 
        if output_full == True:
            if forward == True:
                #Case 1: output_full == True and forward == True
                
                #Compute system_warnings and print, if needed, for debuging
                system_warnings = experimental_simulator_2d.check_lenses_in_system_for_alias(system, use_prints = use_prints)
                if use_prints == True:
                    print(system_warnings)
                    
                #Initialize the output array
                f_total = external_imports.np.reshape(f0, (1, external_imports.np.shape(f0)[0], external_imports.np.shape(f0)[1])) + 0j

                #Propagate beam through the optical system optical element by optical element.
                for i in external_imports.np.arange(system_length):
                    
                    #Check if optical element is free space domain and propagate with solver.numeric_fourier_solver_2d()
                    if type(system[i]) == int or type(system[i]) == float:
                        #Print operation.
                        if use_prints == True:
                            print('Propagate for '+str(system[i])+' units [mm]')
                            
                        #Check for non-zero length of free space domain
                        if system[i] != 0:
                            
                            #Propagate according to all the input parameters. Use system_warnings for the fft_input/fft_output varibles
                            f = internal_imports.s.numeric_fourier_solver_2d.linear(f0, wavelength=internal_imports.p.wavelength, n0 = internal_imports.p.n0, print_progress = use_prints, steps = int(system[i]/internal_imports.p.dz), output_full = True, fft_input=(system_warnings[i][0]=='fourier'), fft_output=(system_warnings[i][1]=='fourier'))
                            if system_warnings[i][1] == 'fourier':
                                temp_f = external_imports.np.zeros_like(f)
                                for j in range(external_imports.np.shape(f)[0]):
                                    temp_f[j,:] = external_imports.np.fft.ifftshift(external_imports.np.fft.ifft2(external_imports.np.fft.ifftshift(f[j,:])))
                                f_total = external_imports.np.append(f_total,temp_f, axis = 0)
                                del temp_f
                            if system_warnings[i][1] == 'normal':
                                f_total = external_imports.np.append(f_total, f, axis = 0)
                                
                            #Keep the last profile for the next iteration
                            f0 = f[-1,:]
                            
                    #Check if optical element is not a free space domain
                    if type(system[i]) == list:
                        
                        #Check if optical element is lens
                        if system[i][0] == 'l':
                            
                            #Check if alias in normal space
                            if system_warnings[i] == 'n':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in Fourier space 
                                phase = internal_imports.ini.optical_elements.lens_theory(system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], offset_y = system[i][3], dim = 2, fft_flag = True)
                                
                                #Compute normalization constant for initial beam profile
                                norm_const = external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                                #Apply the lens phase mask by convolution with the initial beam profile
                                f0 = external_imports.sps.convolve(phase, f0, mode = 'same')
                                
                                #Renormalize the resulting profile
                                f0 = norm_const*f0/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                            #Check if alias in Fourier space
                            if system_warnings[i] == 'f':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in normal space
                                phase = internal_imports.ini.optical_elements.lens_theory(system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], offset_y = system[i][3], dim = 2, fft_flag = False)
                                
                                #Apply the lens phase mask by product
                                f0 = f0*phase
                            
                        #Check if phase mask
                        if system[i][0] == 'mp':
                            if use_prints == True:
                                print('Apply mask phase.')
                                
                            #Shift mask
                            mask = mask_generator_2d.mask_shift(system[i][1], system[i][2], system[i][3])
                            
                            #Apply phase mask by product
                            f0 = f0*external_imports.np.exp(1j*mask)
                            
                        #Check if amplitude mask
                        if system[i][0] == 'ma':
                            if use_prints == True:
                                print('Apply mask amplitude')
                            
                            #Shift mask
                            mask = mask_generator_2d.mask_shift(system[i][1], system[i][2], system[i][3])
                            
                            #Apply amplitude mask by product
                            f0 = f0*mask
                            
            if forward == False:
                #Case 2: output_full == True and forward == False
                
                #Initialize the output array
                f_total = external_imports.np.reshape(f0, (1, external_imports.np.shape(f0)[0], external_imports.np.shape(f0)[1])) + 0j
                
                #Reverse order of optical elements in system
                system.reverse()
                
                #Compute system_warnings and print, if needed, for debuging
                system_warnings = experimental_simulator_2d.check_lenses_in_system_for_alias(system, use_prints = use_prints)
                if use_prints == True:
                    print(system_warnings)
                    
                #Propagate beam through the optical system optical element by optical element
                for i in external_imports.np.arange(system_length):
                    
                    #Check if optical element is free space domain and propagate with solver.numeric_fourier_solver_2d()
                    if type(system[i]) == int or type(system[i]) == float:
                        
                        #Print operation.
                        if use_prints == True:
                            print('Propagate for '+str(system[i])+' units [mm]')
                        
                        #Check for non-zero length of free space domain
                        if system[i] != 0:
                            
                            #Propagate according to all the input parameters. Use system_warnings for the fft_input/fft_output varibles
                            f = internal_imports.s.numeric_fourier_solver_2d.linear(f0, wavelength=internal_imports.p.wavelength, n0 = internal_imports.p.n0, print_progress=use_prints, steps = int(system[i]/internal_imports.p.dz), output_full = True, forward = False, fft_input=(system_warnings[i][0]=='fourier'), fft_output=(system_warnings[i][1]=='fourier'))
                            if system_warnings[i][1] == 'fourier':
                                temp_f = external_imports.np.zeros_like(f)
                                for j in range(external_imports.np.shape(f)[0]):
                                    temp_f[j,:] = external_imports.np.fft.ifftshift(external_imports.np.fft.ifft2(external_imports.np.fft.ifftshift(f[j,:])))
                                f_total = external_imports.np.append(f_total,temp_f, axis = 0)
                                del temp_f
                            if system_warnings[i][1] == 'normal':
                                f_total = external_imports.np.append(f_total, f, axis = 0)
                            
                            #Keep the last profile for the next iteration.
                            f0 = f[-1,:]
                    
                    #Check if optical element is not a free space domain.
                    if type(system[i]) == list:
                        
                        #Check if optical element is lens
                        if system[i][0] == 'l':
                            
                            #Check if alias in normal space
                            if system_warnings[i] == 'n':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in Fourier space 
                                phase = internal_imports.ini.optical_elements.lens_theory(-system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], offset_y = system[i][3], dim = 2, fft_flag = True)
                                
                                #Compute normalization constant for initial beam profile
                                norm_const = external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                                #Apply the lens phase mask by convolution with the initial beam profile
                                f0 = external_imports.sps.convolve(phase, f0, mode = 'same')
                                
                                #Renormalize the resulting profile
                                f0 = norm_const*f0/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                            
                            #Check if alias in Fourier space
                            if system_warnings[i] == 'f':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in normal space
                                phase = internal_imports.ini.optical_elements.lens_theory(-system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], offset_y = system[i][3], dim = 2, fft_flag = False)
                                
                                #Apply the lens phase mask by product
                                f0 = f0*phase
                            
                        #Check if phase mask
                        if system[i][0] == 'mp':
                            if use_prints == True:
                                print('Apply mask phase.')
                               
                            #Shift mask
                            mask = mask_generator_2d.mask_shift(system[i][1], system[i][2], system[i][3])
                                
                            #Apply phase mask by product
                            f0 = f0*external_imports.np.exp(-1j*mask)
                            
                        #Check if amplitude mask
                        if system[i][0] == 'ma':
                            if use_prints == True:    
                                print('Apply mask amplitude')
                                
                            #Shift mask
                            mask = mask_generator_2d.mask_shift(system[i][1], system[i][2], system[i][3])
                                
                            #Apply amplitude mask by product
                            f0 = f0*mask
                            
                #Reverse order of optical elements in system back to the original order
                system.reverse()
            
            #Plot the beam profile at the last position on the propagation axis
            if print_output == True:
                
                #Plot the normalized values using surface plots
                try:
                    import mayavi.mlab as mayavi_mlab
                    mayavi_mlab.clf()
                    mayavi_mlab.contour3d(external_imports.np.abs(f_total)**2, transparent = True, contours = 10)
                except:
                    plotters.full_beam_plot_1d(f_total[:,:,int(external_imports.np.shape(f_total)[2]/2.)], system, forward = forward, norm = norm)
                
            return f_total
        else:
            if forward == True:
                #Case 3: output_full == False and forward == True
                
                #Compute system_warnings and print, if needed, for debuging
                system_warnings = experimental_simulator_2d.check_lenses_in_system_for_alias(system, use_prints = use_prints)
                if use_prints == True:
                    print(system_warnings)
                    
                #Initialize the output array
                f_final = external_imports.np.copy(f0)
                
                #Propagate beam through the optical system optical element by optical element
                for i in external_imports.np.arange(system_length):
                    
                    #Check if optical element is free space domain and propagate with solver.numeric_fourier_solver_2d()
                    if type(system[i]) == int or type(system[i]) == float:
                        
                        #Print operation
                        if use_prints == True:
                            print('Propagate for '+str(system[i])+' units [mm]')
                            
                        #Check for non-zero length of free space domain
                        if system[i] != 0:
                            
                            #Propagate according to all the input parameters. Use system_warnings for the fft_input/fft_output varibles
                            f_final = internal_imports.s.numeric_fourier_solver_2d.linear(f0, wavelength=internal_imports.p.wavelength, n0 = internal_imports.p.n0, print_progress=False, steps = int(system[i]/internal_imports.p.dz), output_full = False, fft_input=(system_warnings[i][0]=='fourier'), fft_output=(system_warnings[i][1]=='fourier'))
                            
                            #Copy the output profile
                            f0 = external_imports.np.copy(f_final)
                            
                    #Check if optical element is not a free space domain
                    if type(system[i]) == list:
                        
                        #Check if optical element is lens
                        if system[i][0] == 'l':
                            
                            #Check if alias in normal space
                            if system_warnings[i] == 'n':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in Fourier space 
                                phase = internal_imports.ini.optical_elements.lens_theory(system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], offset_y = system[i][3], dim = 2, fft_flag = True)
                                
                                #Compute normalization constant for initial beam profile
                                norm_const = external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                                #Apply the lens phase mask by convolution with the initial beam profile
                                f0 = external_imports.sps.convolve(phase, f0, mode = 'same')
                                
                                #Renormalize the resulting profile
                                f0 = norm_const*f0/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                            
                            #Check if alias in Fourier space
                            if system_warnings[i] == 'f':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in normal space
                                phase = internal_imports.ini.optical_elements.lens_theory(system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], offset_y = system[i][3], dim = 2, fft_flag = False)
                                
                                #Apply the lens phase mask by product
                                f0 = f0*phase
                            
                        #Check if phase mask
                        if system[i][0] == 'mp':
                            if use_prints == True:
                                print('Apply mask phase.')
                                
                            #Shift mask
                            mask = mask_generator_2d.mask_shift(system[i][1], system[i][2], system[i][3])
                            
                            #Apply phase mask by product
                            f0 = f0*external_imports.np.exp(1j*mask)
                            
                        #Check if amplitude mask
                        if system[i][0] == 'ma':
                            if use_prints == True:
                                print('Apply mask amplitude')
                                
                            #Shift mask
                            mask = mask_generator_2d.mask_shift(system[i][1], system[i][2], system[i][3])
                                
                            #Apply amplitude mask by product
                            f0 = f0*mask
        
            if forward == False:
                #Case 4: output_full == False and forward == False
                
                #Initialize the output array
                f_final = external_imports.np.copy(f0)
                
                #Reverse order of optical elements in system
                system.reverse()
                
                #Compute system_warnings and print, if needed, for debuging
                system_warnings = experimental_simulator_2d.check_lenses_in_system_for_alias(system, use_prints = use_prints)
                if use_prints == True:
                    print(system_warnings)
                    
                #Propagate beam through the optical system optical element by optical element.
                for i in external_imports.np.arange(system_length):
                    
                    #Check if optical element is free space domain and propagate with solver.numeric_fourier_solver_1d()
                    if type(system[i]) == int or type(system[i]) == float:
                        
                        #Print operation.
                        if use_prints == True:
                            print('Propagate for '+str(system[i])+' units [mm]')
                        
                        #Check for non-zero length of free space domain
                        if system[i] != 0:
                            
                            #Propagate according to all the input parameters. Use system_warnings for the fft_input/fft_output varibles
                            f_final = internal_imports.s.numeric_fourier_solver_2d.linear(f0, wavelength=internal_imports.p.wavelength, n0 = internal_imports.p.n0, print_progress=False, steps = int(system[i]/internal_imports.p.dz), output_full = False, forward=False, fft_input=(system_warnings[i][0]=='fourier'), fft_output=(system_warnings[i][1]=='fourier'))
                            
                            #Copy the output profile
                            f0 = external_imports.np.copy(f_final)
                            
                    #Check if optical element is not a free space domain.
                    if type(system[i]) == list:
                        
                        #Check if optical element is lens
                        if system[i][0] == 'l':
                            
                            #Check if alias in normal space
                            if system_warnings[i] == 'n':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in Fourier space 
                                phase = internal_imports.ini.optical_elements.lens_theory(-system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], offset_y = system[i][3], dim = 2, fft_flag = True)
                                
                                #Compute normalization constant for initial beam profile
                                norm_const = external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                                #Apply the lens phase mask by convolution with the initial beam profile
                                f0 = external_imports.sps.convolve(phase, f0, mode = 'same')
                                
                                #Renormalize the resulting profile
                                f0 = norm_const*f0/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f0)**2))
                                
                            #Check if alias in Fourier space
                            if system_warnings[i] == 'f':
                                if use_prints == True:
                                    print('Apply lens phase.')
                                    
                                #Compute phase in normal space
                                phase = internal_imports.ini.optical_elements.lens_theory(-system[i][1], wavelength=internal_imports.p.wavelength, offset_x = system[i][2], offset_y = system[i][3], dim = 2, fft_flag = False)
                                
                                #Apply the lens phase mask by product
                                f0 = f0*phase
                                
                        #Check if phase mask
                        if system[i][0] == 'mp':
                            if use_prints == True:
                                print('Apply mask phase.')
                            
                            #Shift mask
                            mask = mask_generator_2d.mask_shift(system[i][1], system[i][2], system[i][3])
                            
                            #Apply phase mask by product
                            f0 = f_final*external_imports.np.exp(-1j*mask)
                            
                        #Check if amplitude mask
                        if system[i][0] == 'ma':
                            if use_prints == True:
                                print('Apply mask amplitude')
                            
                            #Shift mask
                            mask = mask_generator_2d.mask_shift(system[i][1], system[i][2], system[i][3])
                                
                            #Apply amplitude mask by product
                            f0 = f_final*mask
                
                #Reverse order of optical elements in system back to the original order
                system.reverse()
                
            #Plot the beam profile
            if print_output == True:
                plotters.full_beam_plot_1d(f_total[:,:,int(external_imports.np.shape(f_total)[2]/2.)], system, forward = forward, norm = norm)
            
        return f_final
        
    def monte_carlo_precision_test(f_in, system, system_errors, number_of_samples):
        """
        Simulate the propagation of a beam considering errors regarding the positioning of optical elements. Based of the errors, various scenarios are considered and simulated in order to check how errors influence the optical system.
        
        The sampling assumes a uniform distribution inside the intervals.
        
        Inputs:
            f_in = 2-dimensional array; complex - The input beam profile.
            system = list - The optical system.
            system_errors = list - The errors related to the optical elements.
            number_of_samples = float - Number of parameters sampled and of simulations done.
        Outputs:
            f_out = 2-dimensional array; complex - The beam profile at the end of the optical system.
        """
        
        #Check if system and system_errors are the same length
        if len(system) != len(system_errors):
            print('len(system) != len(system_errors)')
        
        system_length = len(system)
        f_out = external_imports.np.zeros((internal_imports.p.N_x, internal_imports.p.N_y))
        
        #Compute the solutions for each case and add together the resultng output amplitude profiles. Good approach for fast visualisation. 
        for i in external_imports.np.arange(number_of_samples):
            
            #Initialize temporal optical system
            temp_system = []
            
            #Sample values for parameters of the temporal optical system
            for j in external_imports.np.arange(system_length):
                
                #Sample value for free space domain
                if type(system[j]) == int or type(system[j]) == float:
                    temp_system.append(system[j] + (external_imports.np.random.random()-0.5)*2*system_errors[j])
                
                #Sample values for other optical elements
                if type(system[j]) == list:
                    
                    #Sample values for lens
                    if system[j][0] == 'l':
                        
                        #Initialize optical element and append it to the temporal optical system
                        temp_system.append([])
                        
                        #Append lens tag to the optical element
                        temp_system[j].append('l')
                        
                        #Compute and append focal length value to the optical element
                        val = float(external_imports.np.copy(system[j][1]) + (external_imports.np.random.random()-0.5)*2*system_errors[j][0])
                        temp_system[j].append(val)
                        
                        #Compute and append shift value on the X axis 
                        val = float(external_imports.np.copy(system[j][2]) + (external_imports.np.random.random()-0.5)*2*system_errors[j][1])
                        temp_system[j].append(val)
                        
                        #Compute and append shift value on the Y axis
                        val = float(external_imports.np.copy(system[j][3]) + (external_imports.np.random.random()-0.5)*2*system_errors[j][2])
                        temp_system[j].append(val)
                        
                    #Sample values for phase mask
                    if system[j][0] == 'mp':
                        
                        #Initialize optical element and append it to the temporal optical system
                        temp_system.append([])
                        
                        #Append lens tag to the optical element
                        temp_system[j].append('mp')
                        
                        #Append phase mask
                        shifted_mask = external_imports.np.copy(system[j][1])
                        
                        #Shift the phase mask and append the result to the optical element 
                        shifted_mask = mask_generator_2d.mask_shift(shifted_mask, external_imports.np.copy(system[j][2]) + (external_imports.np.random.random()-0.5)*2*system_errors[j][0], external_imports.np.copy(system[j][3]) + (external_imports.np.random.random()-0.5)*2*system_errors[j][1])
                        temp_system[j].append(shifted_mask)
                        
                    #Sample values for amplitude mask
                    if system[j][0] == 'ma':
                        
                        #Initialize optical element and append it to the temporal optical system
                        temp_system.append([])
                        
                        #Append lens tag to the optical element
                        temp_system[j].append('ma')
                        
                        #Append phase mask
                        shifted_mask = external_imports.np.copy(system[j][1])
                        
                        #Shift the phase mask and append the result to the optical element
                        shifted_mask = mask_generator_2d.mask_shift(shifted_mask, external_imports.np.copy(system[j][2]) + (external_imports.np.random.random()-0.5)*2*system_errors[j][0], external_imports.np.copy(system[j][3]) + (external_imports.np.random.random()-0.5)*2*system_errors[j][1])
                        temp_system[j].append(shifted_mask)
            
            #Add the amplitude profile from the current iteration
            f_out = f_out + external_imports.np.abs(experimental_simulator_2d.propagate(f_in, temp_system, forward = True, norm = True, output_full = False, print_output = False, use_prints = False))**2
            print("Step: " + str(i+1) + "/"+str(number_of_samples))
        return f_out
    
class mask_generator_2d:
    def compute_mask_dual_system(system1, system2, f_in, f_out, check_mask = False, print_output = False, use_prints = True):
        """
        Computes the phase mask for a specific optical system that is split in 2 subsystems.
        
        System diagram:
                  Subsystem1               Subsystem2
        Input[------------------]MASK[-------------------]Output
        
        The entire optical system is given by:
            
            optical_system = system1 + ['mp', MASK] + system2
        
        Inputs:
            system1 = list - Subsystem1.
            system2 = list - Subsystem2.
            f_in = 2-dimensional array; complex - The initial condition.
            f_out = 2-dimensional array; complex - The desired output beam.
            check_mask = boolean - If True, calls check_output_for_mask function.
            print_output = boolean - If True, the beam profile at the end of the optical system is ploted.
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            mask = 2-dimensional array; real - The phase mask
        """
        
        #Normalize in input and output beam profiles
        norm_in = external_imports.np.sum(external_imports.np.abs(f_in)**2)
        norm_out = external_imports.np.sum(external_imports.np.abs(f_out)**2)
        
        f_in = f_in/external_imports.np.sqrt(norm_in)
        f_out = f_out/external_imports.np.sqrt(norm_out)
        
        #Propagate the input beam forwards through system1
        f1 = experimental_simulator_2d.propagate(f_in, system1, forward = True, norm = False, output_full=False)
        
        #Propagate the output beam backwards through system2
        f2 = experimental_simulator_2d.propagate(f_out, system2, forward = False, norm = False, output_full=False)
        
        #Compute the phase mask
        try:
            mask = external_imports.np.angle(f2/f1)
        except:
            mask = external_imports.np.angle(external_imports.np.exp(1j*external_imports.np.angle(f2))/external_imports.np.exp(1j*external_imports.np.angle(f1)))
        
        if check_mask == True:
            mask_generator_2d.check_output_for_mask(system1 + [['mp', mask, 0, 0]] + system2, f_in, f_out, use_prints = True, output = False)
                
        #Print the entire beam through the optical system using 3d contour plots
        if print_output == True:
            f = experimental_simulator_2d.propagate(f_in, system1 + [['mp', mask, 0, 0]] + system2, forward = True, norm = False, output_full=False)
            external_imports.plt.figure()
            external_imports.plt.imshow(external_imports.np.abs(f)**2)
        
        return mask
            
    def check_output_for_mask(system, f_in, f_out, output = True, use_prints = True, ploting = False):
        """
        Computes the beam profile through an optical system given by
            
            optical_system = system1 + [['mp', mask]] + system2
            
        and then compare the computed autput with the desired one (f_out).
        
        Inputs:
            system1 = list - Subsystem1.
            mask = 2-dimensional array; real - The phase mask.
            system2 = list - Subsystem2.
            f_in = 2-dimensional array; complex - The initial condition.
            f_out = 2-dimensional array; complex - The desired output beam.
            output = boolean - If True, the function returns the convolution of the 2 outputs and the shift.
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            CONDITION:
                if output = True
                    corr = 2-dimensional array; complex - The cross-correlation between the computed output and the desired one
                    shift = int - The difference in position between the 2 outputs, measured in pixels. 
                if output = False
                    None
        """
        
        #Print operation
        if use_prints == True:
            print("Checking mask.")
        
        #Propagate input profile through the entire optical system
        f = experimental_simulator_2d.propagate(f_in, system, forward = True, norm = False, output_full=False)
        
        #Plot the computed and desired amplitude profiles for qualitative comparison
        if ploting == True:
            external_imports.plt.figure()
            external_imports.plt.imshow(external_imports.np.abs(f_out)**2)
            external_imports.plt.figure()
            external_imports.plt.imshow(external_imports.np.abs(f)**2)
        
        #Normalize the 2 outputs
        f_norm = f/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f)**2))
        f_out_norm = f_out/external_imports.np.sqrt(external_imports.np.sum(external_imports.np.abs(f_out)**2))
        
        #Compute the cross-correlation between them
        corr = external_imports.sps.correlate(f_out_norm, f_norm, mode = 'same')
        
        #Print the maximum cross-correlation value 
        if use_prints == True:
            print('Max correlation: ' + str(external_imports.np.max(external_imports.np.abs(corr))))
        
        #Compute the difference in position between the 2 outputs 
        shift = external_imports.np.array([external_imports.np.where(external_imports.np.abs(corr) == external_imports.np.max(external_imports.np.abs(corr)))[0][0],external_imports.np.where(external_imports.np.abs(corr) == external_imports.np.max(external_imports.np.abs(corr)))[1][0]]) - external_imports.np.array([int(external_imports.np.shape(corr)[0]/2), int(external_imports.np.shape(corr)[1]/2)])
        
        #Print it
        if use_prints == True:
            print('Shift: ' + str(shift))
        
        #Return the outputs
        if output == True:
            return corr, shift 

    def compute_mask_triple_system(system1, system2, system3, f_in, f_out, check_mask = False, print_output = False, use_prints = True):
        """
        Computes the phase mask for a specific optical system that is split in 3 subsystems.
        
        System diagram:
                  Subsystem1                 Subsystem2                 Subsystem3
        Input[------------------]MASK_1[-------------------]MASK_2[-------------------]Output
        
        The entire optical system is given by:
            
            optical_system = system1 + [['mp', MASK_1]] + system2 + [['mp', MASK_2]] + system3
        
        Inputs:
            system1 = list - Subsystem1.
            system2 = list - Subsystem2.
            system3 = list - Subsystem3.
            f_in = 2-dimensional array; complex - The initial condition.
            f_out = 2-dimensional array; complex - The desired output beam.
            check_mask = boolean - If True, calls check_output_for_mask function.
            print_output = boolean - If True, the whole beam profile is ploted.
            use_prints = boolean - If True, all the print calls in the function are activated. Used for debugging.
        Outputs:
            mask1 = 2-dimensional array; real - The phase mask
            mask2 = 2-dimensional array; real - The phase mask
        """
        
        #Normalize in input and output beam profiles
        norm_in = external_imports.np.sum(external_imports.np.abs(f_in)**2)
        norm_out = external_imports.np.sum(external_imports.np.abs(f_out)**2)
        
        f_in = f_in/external_imports.np.sqrt(norm_in)
        f_out = f_out/external_imports.np.sqrt(norm_out)
        
        #Propagate the output beam backwards through system3
        f3 = experimental_simulator_2d.propagate(f_out, system3, forward = False, norm = False, output_full=False)
        
        #Compute MASK_2 considering f3 as output of optical_system = system1 + [['mp, MASK_1]] + system2 
        mask1 = mask_generator_2d.compute_mask_dual_system(system1, system2, f_in, external_imports.np.abs(f3), check_mask = False)
        
        #Compute the MASK_1
        mask2 = mask_generator_2d.compute_mask_dual_system(system1 + [['mp', mask1, 0, 0]] + system2, system3, f_in, f_out, check_mask = False)
        
        ##Check the phase masks
        if check_mask == True:
            mask_generator_2d.check_output_for_mask(system1 + [['mp', mask1, 0, 0]] + system2 + [['mp', mask2, 0, 0]] + system3, f_in, f_out, use_prints = True, output = False)
        
        #Print the entire beam through the optical system using 3d contour plots
        if print_output == True:
            f = experimental_simulator_2d.propagate(f_in, system1 + [['mp', mask1, 0, 0]] + system2 + [['mp', mask2, 0, 0]] + system3, forward = True, norm = False, output_full=True)
            external_imports.plt.figure()
            external_imports.plt.imshow(external_imports.np.abs(f)**2)
        return mask1, mask2
    
    def mask_shift(mask, shift_x, shift_y):
        """
        Shift the phase mask on the X and Y directions by an amount given by the user.
        
        Inputs:
            mask = 2-dimensional array; real - The phase mask.
            shift_x = float; real - Shift on the X axis. Units are the same as in numeric_parameters.py file. 
            shift_y = float; real - Shift on the Y axis. Units are the same as in numeric_parameters.py file. 
        Outputs:
            mask = 2-dimensional array; real - The shifted phase mask.
        """
        
        #Transform the shifts from the implied measurement unit into pixels
        shift_x_pixels = int(shift_x / internal_imports.p.dx)
        shift_y_pixels = int(shift_y / internal_imports.p.dy)
        
        #Apply the shifts on the mask
        mask = external_imports.np.roll(mask, shift_x_pixels, axis = 0)
        mask = external_imports.np.roll(mask, shift_y_pixels, axis = 1)
        
        return mask
    
class computing_system_estimators:
    def memory_estimator(domain_size, system, output_full = True, size_unit = "GB"):
        """
        Computes the required minimum RAM needed for storage of the optical system based on the size of the spatial domain and the optical components in the system.
        
        Note: This function does not estimate the required RAM for the calculations needed during propagation. This implies that the value returned is a lower bound of the RAM requirement.
        
        Inputs:
            domain_size = array; int - The shape of the spatial domain e.g. (internal_imports.p.N_x) for a 1-dimensional array with internal_imports.p.N_x elemens, or (internal_imports.p.N_x, internal_imports.p.N_y) for a 2-dimensional array
            system = list; The optical system
            output_full = boolean - If True, the spatial domain for the entire propagation axis is considered.
            size_unit = string - The unit for the memory space required. Possible inputs are: B for bytes, KB for kilobytes, MG for megabytes, and GB for gigabytes.
        Outputs:
            number_of_bytes = float - The minimum memory space required to store the variables.
        """
        
        #Initialize the number_of_bytes varible
        number_of_bytes = 0
        
        #Compute for 1-dimensional transversal case
        if external_imports.np.size(domain_size) == 1:
            #Compute total number of free space domains
            if output_full == True:
                distance = 0
                for i in system:
                    if type(i) == int or type(i) == float:
                        distance += i
                steps = int(distance/internal_imports.p.dz)
                #Compute the memory required for the propagation steps
                temp_bytes = (external_imports.np.zeros(domain_size)+0j).nbytes
                number_of_bytes += temp_bytes * steps
            else:
                temp_bytes = (external_imports.np.zeros(domain_size)+0j).nbytes
                number_of_bytes += temp_bytes*2
                
            #Compute the memory required for phase masks in system
            for i in system:
                if type(i) == list:
                    if i[0] == 'mp':
                        temp_bytes = (external_imports.np.zeros(domain_size)+0j).nbytes
                        number_of_bytes += temp_bytes
            
            #Compute the memory required for lenses
            for i in system:
                if type(i) == list:
                    if i[0] == 'l':
                        temp_bytes = (external_imports.np.zeros(domain_size)+0j).nbytes
                        number_of_bytes += temp_bytes
        
        #Compute for 2-dimensional transversal case
        if external_imports.np.size(domain_size) == 2:
            
            #C+ompute total number of free space domains
            if output_full == True:
                distance = 0
                for i in system:
                    if type(i) == int or type(i) == float:
                        distance += i
                steps = int(distance/internal_imports.p.dz)
                #compute the memory req for the propagation steps
                temp_bytes = (external_imports.np.zeros(domain_size)+0j).nbytes
                number_of_bytes += temp_bytes * steps
            else:
                temp_bytes = (external_imports.np.zeros(domain_size)+0j).nbytes
                number_of_bytes += temp_bytes*2
            #compute the memory req for phase masks in system
            for i in system:
                if type(i) == list:
                    if i[0] == 'mp':
                        temp_bytes = (external_imports.np.zeros(domain_size)+0j).nbytes
                        number_of_bytes += temp_bytes
            #compute the memory req for lenses
            for i in system:
                if type(i) == list:
                    if i[0] == 'l':
                        temp_bytes = (external_imports.np.zeros(domain_size)+0j).nbytes
                        number_of_bytes += temp_bytes
        
        if size_unit == 'B':
            return number_of_bytes
        if size_unit == 'KB':
            return number_of_bytes/2**10
        if size_unit == 'MB':
            return number_of_bytes/2**20
        if size_unit == 'GB':
            return number_of_bytes/2**30
        
class plotters:
    def full_beam_plot_1d(f, system, coords = ['z','x'], forward = True, x_number_of_ticks = None, y_number_of_ticks = None, norm = False, which = 'abs'):
        """
        Plots the amplitude of a computed 1-dimensional beam profile using the physical spatial scaling specified in numeric_parameters.py.
        
        Inputs:
            f = 2-dimensional arraay; complex - The computed beam profile.
            system = list - The optical system.
            coords = list - The meaning of the axes. Usually the solver outputs an array with the order of axis ['z', 'x'].
            x_number_of_ticks = int - The number of major ticks on the X axis.
            y_number_of_ticks = int - The number of major ticks on the Y axis.
            norm = boolean - If true, the amplitude profile is normalized such that the maximum value becomes 1 at every step on the propagation axis. 
            which = string - Accepts 3 inputs:
                'abs' - plots the amplitude profile
                'phase' - plots the phase profile
                'both' - plots the product between the amplitude profile and the phase profile
        Outputs:
            None
        """
        
        import matplotlib.ticker as mpl_t
        #Create the figure
        fig = external_imports.plt.figure()
        ax = fig.gca()
        
        #Initialize the number of ticks if needed
        if x_number_of_ticks is None:
            x_number_of_ticks = 20
        if y_number_of_ticks is None:
            y_number_of_ticks = 20   
        
        if coords == ['z', 'x']:
            f = external_imports.np.transpose(f)
            
        #Normalize the amplitude profile at each step on the propagation axis
        for i in range(external_imports.np.shape(f)[1]):
            f[:,i] = f[:,i]/external_imports.np.max(external_imports.np.abs(f[:,i]))
            
        if forward == False:
            system = system[::-1]
            
        distance_pixels = 0
        for i in system:
            if type(i) == int or type(i) == float:
                distance_pixels += i/internal_imports.p.dz
            elif type(i) == list:
                if i[0] == 'l':
                    f[:,int(distance_pixels-2):int(distance_pixels+3)] = external_imports.np.max(external_imports.np.abs(f))
                if i[0] == 'mp':
                    f[::2,int(distance_pixels-2):int(distance_pixels+3)] = external_imports.np.max(external_imports.np.abs(f))
                if i[0] == 'ma':
                    f[::4,int(distance_pixels-2):int(distance_pixels+3)] = external_imports.np.max(external_imports.np.abs(f))
            
        #Plot the desired profile
        if which == 'abs':
            ax.imshow(external_imports.np.abs(f))
        elif which == 'phase':
            ax.imshow(external_imports.np.angle(f))
        elif which == 'both':
            ax.imshow(external_imports.np.abs(f)*external_imports.np.angle(f))
        else:
            print("Select between 'abs', 'phase' or 'both' for parameter: which")        
        
        #Calculate the physical transverse domain
        x_init = internal_imports.ini.standard_initial_conditions.spatial_domain_generator(dim = 1)[0]
        
        #Compute distance between 2 neighboring ticks in pixels on each axis based on the size of f, x_number_of_ticks, and y_number_of_ticks
        x_size, y_size = external_imports.np.shape(f)
        
        x_tick_size = round(x_size / x_number_of_ticks)
        y_tick_size = round(y_size / y_number_of_ticks)
        
        #Compute the physical distance between 2 neighboring ticks on each axis
        x_tick_size_physx = x_tick_size * internal_imports.p.dx
        y_tick_size_physx = y_tick_size * internal_imports.p.dz
        
        #Compute the positions of the ticks in pixels
        x_positions = (external_imports.np.arange(x_number_of_ticks + 1)*x_tick_size).tolist()
        y_positions = (external_imports.np.arange(y_number_of_ticks + 1)*y_tick_size).tolist()

        #Compute the physical positions of the ticks
        x_labels = (external_imports.np.around(external_imports.np.arange(x_number_of_ticks + 1)*x_tick_size_physx + x_init, decimals = 2)).tolist()
        if forward == True:
            y_labels = (external_imports.np.around(external_imports.np.arange(y_number_of_ticks + 1)*y_tick_size_physx, decimals = 2)).tolist()
        elif forward == False:
            y_labels = (-1*external_imports.np.around(external_imports.np.arange(y_number_of_ticks + 1)*y_tick_size_physx, decimals = 2)).tolist()
        else:
            print("Select between 'True' or 'False' for parameter: forward")
        
        #Set the new labels and positions
        x_positions = mpl_t.FixedLocator(x_positions)
        y_positions = mpl_t.FixedLocator(y_positions)
        
        x_labels = mpl_t.FixedFormatter(x_labels)
        y_labels = mpl_t.FixedFormatter(y_labels)
        
        ax.xaxis.set_major_formatter(y_labels)
        ax.yaxis.set_major_formatter(x_labels)
        ax.xaxis.set_major_locator(y_positions)
        ax.yaxis.set_major_locator(x_positions)
        
        #Label the axis.
        ax.set_xlabel(r"$z[$"+internal_imports.p.unit+r"$]$")
        ax.set_ylabel(r"$x[$"+internal_imports.p.unit+r"$]$")
        
        ax.tick_params(axis = 'x', labelrotation = 90)
                    