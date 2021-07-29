# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:28:57 2020

@author: Victor
"""

import pyparax as parax
import numpy as np

"""
Collections of functions that showcase the mask computation procedure for the 1-dimensional case.

All cases have been optimized for good reconstruction of the output profile given by a maximum cross-correlation or over 0.95 out of 1.
"""

parax.numeric_parameters.N_x = 1000
parax.numeric_parameters.N_y = 1000
parax.numeric_parameters.dz = 0.5

def dual_system_optimization_case_1_initial_phase():
    """
    ---------------------------------------------------------------------------
    General description of the step
    ---------------------------------------------------------------------------
    In the first step the following optical systems are considered: S1 = [0], S2 = [100]. 
    
    The input Gaussian beam has a standard deviation of 0.2mm and the output Airy beam is rescaled by a factor of 40.
    
    The phase mask is computed using the corresponding built-in function.
    
    Using the phase mask, the resulting output profile is computed for the entire optical system.
    
    --------------------------------------------------------------------------
    Interpretation of results
    --------------------------------------------------------------------------
    The computed output profile does not resemble the desired Airy profile, thus some changes have to be made to either the optical system or the input beam shape.
    
    The input beam shape approach is presented in dual_system_optimization_case_1_optimization_ver_1(), while the optical system change is presented in dual_system_optimization_case_1_optimization_ver_2() and dual_system_optimization_case_1_optimization_ver_3().    
    """
    f_in = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0, 0.2,0.2)
    f_out = parax.initial_conditions.standard_initial_condition_generation.generate_airy_2d(4*10**1, 0, 10**-1)
    system1 = [0]
    system2 = [100]
    mask = parax.experimental_simulator.mask_generator_2d.compute_mask_dual_system(system1, system2, f_in, f_out, check_mask = True)
    
    #Check the beam profile propagation through the optical system
    system = system1 + [['mp', mask, 0,0]] + system2
    f = parax.experimental_simulator.experimental_simulator_2d.propagate(f_in, system, output_full = True, forward = True, print_output=True, norm = True)
    
    del f
    
def dual_system_optimization_case_1_optimization_ver_1():
    """
    ---------------------------------------------------------------------------
    General description of the step
    ---------------------------------------------------------------------------
    First we consider the optimization of the input Gaussian beam width. 
    
    In order to select qualitatively the appropriate width, the following steps are considered:
        1) backward propagate the output profile through optical system 2
        2) since optical system 1 consists of nothing, select the Gaussian mean (position) and width such that a good match between it and the backward propagated beam is satisfied
        3) run the phase mask computation procedure for that choice of the input Gaussian beam parameters
    
    --------------------------------------------------------------------------
    Interpretation of results
    --------------------------------------------------------------------------
    By changing the width parameter for the Gaussian beam, the optimal value is found to be around 0.56mm, which gives a cross-correlation of 0.9930 out of 1.
    
    This result is satisfactory.
    """
    f_in = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.6,0.6)
    f_out = parax.initial_conditions.standard_initial_condition_generation.generate_airy_2d(4*10**1, 0, 10**-1)
    system1 = [0]
    system2 = [100]
    
    # 1) backward propagate the output profile through optical system 2
    f_backward = parax.experimental_simulator.experimental_simulator_2d.propagate(f_out, system2, output_full = False, forward = False, print_output=False)
    
    # 2) since optical system 1 consists of nothing, select the gaussian mean (position) and width such that a good match between it and the backward propagated beam is satisfied
    parax.external_imports.plt.figure()
    parax.external_imports.plt.imshow(np.abs(f_backward)/np.max(np.abs(f_backward)))
    parax.external_imports.plt.figure()
    parax.external_imports.plt.imshow(np.abs(f_in)/np.max(np.abs(f_in)))
    
    # By trial and error, the width 0.56mm is found to be optimal
    f_in = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.56,0.56)
    
    # 3) run the phase mask computation procedure for that choice of the input gaussian beam parameters
    mask = parax.experimental_simulator.mask_generator_2d.compute_mask_dual_system(system1, system2, f_in, f_out, check_mask = True)
    
    del f_backward
    
def dual_system_optimization_case_1_optimization_ver_2():
    """
    ---------------------------------------------------------------------------
    General description of the step
    ---------------------------------------------------------------------------
    This function consideres the optimization procedure by changeing optical system 2. 
    
    Similar steps are being used here as well:
        1) backward propagate the output profile through optical system 2
        2) since optical system 1 consists of nothing, select the appropriate distance in optical system 2 such that a good match between it and the backward propagated beam is satisfied
        3) run the phase mask computation procedure for that choice of the optical system 2 parameter
    
    --------------------------------------------------------------------------
    Interpretation of results
    --------------------------------------------------------------------------
    The length of the free space in optical system 2 has been varied until the optimal value of 75mm has been found, which gives a cross-correlation of 0.9864 out of 1.
    
    This result is satisfactory, although of lower quality when compared with the one in dual_system_optimization_case_1_optimization_ver_1().
    """
    f_in = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.4,0.4)
    f_out = parax.initial_conditions.standard_initial_condition_generation.generate_airy_2d(4*10**1, 0, 10**-1)
    system1 = [0]
    system2 = [100]
    
    # 1) backward propagate the output profile through optical system 2
    f_backward = parax.experimental_simulator.experimental_simulator_2d.propagate(f_out, system2, output_full = False, forward = False, print_output=False)
    
    # 2) since optical system 1 consists of nothing, select the gaussian mean (position) and width such that a good match between it and the backward propagated beam is satisfied
    parax.external_imports.plt.figure()
    parax.external_imports.plt.imshow(np.abs(f_backward)/np.max(np.abs(f_backward)))
    parax.external_imports.plt.figure()
    parax.external_imports.plt.imshow(np.abs(f_in)/np.max(np.abs(f_in)))
    
    # By trial and error, the optimal optical system 2 is of 41mm
    system2 = [75]
    
    # 3) run the phase mask computation procedure for that choice of the input gaussian beam parameters
    mask = parax.experimental_simulator.mask_generator_2d.compute_mask_dual_system(system1, system2, f_in, f_out, check_mask = True)
    
    del f_backward
    
def dual_system_optimization_case_1_optimization_ver_3():
    """
    ---------------------------------------------------------------------------
    General description of the step
    ---------------------------------------------------------------------------
    This function consideres the optimization procedure by changeing optical system 1. 
    
    The aim here is to change the amplitude profile of the forward propagated beam through optical system 1 such that it matches the backward propagated one.
    For this purpose, a lens is inserted in optical system 1, after which the beam is focused according to the choice of the focal length. The focusing changes the witdh of the beam in a rather small region so a phase mask can be placed ar the corresponding position to match the desired width.
    
    The steps that are being used here are:
        1) backward propagate the output profile through optical system 2
        2) build the optical system 1 
        3) choose the position and focal length of the lens such that the effect of it is to satisfy the amplitude match at the phase mask (checked by forward propagation)
        3) run the phase mask computation procedure for that choice of the optical system 1 parameters
    
    --------------------------------------------------------------------------
    Interpretation of results
    --------------------------------------------------------------------------
    For a choice of an optical system that consistes of 3 components e.i. free space, lens, free space; where the first free space is of 10mm and the lens has a focal length of 20mm, the optimal choice for the last free space domain is of 76mm, which gives a cross-correlation of 0.9920 out of 1.
    
    This result is satisfactory, although of lower quality when compared with the one in dual_system_optimization_case_1_optimization_ver_1().
    """
    f_in = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.2,0.2)
    f_out = parax.initial_conditions.standard_initial_condition_generation.generate_airy_2d(4*10**1, 0, 10**-1)
    system1 = [10, ['l', 20, 0, 0], 10]
    system2 = [100]
    
    # 1) backward propagate the output profile through optical system 2
    f_backward = parax.experimental_simulator.experimental_simulator_2d.propagate(f_out, system2, output_full = False, forward = False, print_output=False)
    f_forward = parax.experimental_simulator.experimental_simulator_2d.propagate(f_in, system1, output_full = False, forward = False, print_output=False)
    
    # 2) since optical system 1 consists of nothing, select the gaussian mean (position) and width such that a good match between it and the backward propagated beam is satisfied
    parax.external_imports.plt.figure()
    parax.external_imports.plt.imshow(np.abs(f_backward)/np.max(np.abs(f_backward)))
    parax.external_imports.plt.figure()
    parax.external_imports.plt.imshow(np.abs(f_forward)/np.max(np.abs(f_forward)))
    
    # By trial and error, the optimal optical system 1 is [10, ['l', 20, 0, 0], 76]
    system1 = [10, ['l', 20, 0, 0], 76]
    
    # 3) run the phase mask computation procedure for that choice of the input gaussian beam parameters
    mask = parax.experimental_simulator.mask_generator_2d.compute_mask_dual_system(system1, system2, f_in, f_out, check_mask = True)
    
    del f_backward