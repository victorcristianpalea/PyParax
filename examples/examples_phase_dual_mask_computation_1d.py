# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 08:18:18 2020

@author: Victor
"""

import pyparax as parax
import numpy as np

"""
Collections of functions that showcase the procedure for the computation of 2 phase mask, 1-dimensional case.

All cases have been optimized for good reconstruction of the output profile given by a maximum cross-correlation or over 0.99 out of 1.
"""

parax.numeric_parameters.N_x = 3000
parax.numeric_parameters.N_y = 3000

def triple_system_mask_computation_1():
    """
    First example computes the phase mask required in order to transform a Gaussian beam into an Airy beam for the optical systems S1 = [100], S2 = [100], S3 = [100].
    """
    f_in = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_1d(0, 0.01)
    f_out = parax.initial_conditions.standard_initial_condition_generation.generate_airy_1d(4.5*10**1, 0, 2*10**-1)
    system1 = [100]
    system2 = [100]
    system3 = [100]
    mask1, mask2 = parax.experimental_simulator.mask_generator_1d.compute_mask_triple_system(system1, system2, system3, f_in, f_out, check_mask = True)
    
    #Check the beam profile propagation through the optical system
    system = system1 + [['mp', mask1, 0,0]] + system2 + [['mp', mask2, 0,0]] + system3
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f_in, system, output_full = True, forward = True, print_output=False)
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system)
    
    del f
    
def triple_system_mask_computation_2():
    """
    Second example computes the phase mask required in order to transform a Gaussian beam into an Airy beam for the optical systems S1 = [50, ['l', 20, 0, 0], 60], S2 = [100].
    """
    f_in = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_1d(0,0.5)
    f_out = parax.initial_conditions.standard_initial_condition_generation.generate_airy_1d(4*10**1, 0, 10**-1)
    system1 = [50, ['l', 20, 0, 0], 40]
    system2 = [100]
    system3 = [100]
    mask1, mask2 = parax.experimental_simulator.mask_generator_1d.compute_mask_triple_system(system1, system2, system3, f_in, f_out, check_mask = True)
    
    #Check the beam profile propagation through the optical system
    system = system1 + [['mp', mask1, 0,0]] + system2 + [['mp', mask2, 0,0]] + system3
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f_in, system, output_full = True, forward = True, print_output=False)
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system)
    
    del f
    
def triple_system_mask_computation_3():
    """
    Third example computes the phase mask required in order to transform a Gaussian beam into an Airy beam for the optical systems S1 = [50, ['l', 20, 0, 0], 50], S2 = [80, ['l', 40, 0, 0], 60].
    """
    f_in = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_1d(0,0.25)
    f_out = parax.initial_conditions.standard_initial_condition_generation.generate_airy_1d(4*10**1, 0, 10**-1)
    system1 = [50, ['l', 20, 0, 0], 40]
    system2 = [20, ['l', 40, 0, 0], 80]
    system3 = [100]
    mask1, mask2 = parax.experimental_simulator.mask_generator_1d.compute_mask_triple_system(system1, system2, system3, f_in, f_out, check_mask = True)
    
    #Check the beam profile propagation through the optical system
    system = system1 + [['mp', mask1, 0,0]] + system2 + [['mp', mask2, 0,0]] + system3
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f_in, system, output_full = True, forward = True, print_output=False)
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system)
    
    del f
    
def triple_system_mask_computation_4():
    """
    Forth example computes the phase mask required in order to transform an off-centered Gaussian beam into an Airy beam for the optical systems S1 = [50, ['l', 20, 0, 0], 50], S2 = [80, ['l', 40, 0, 0], 60].
    """
    f_in = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_1d(0.1 ,0.25)
    f_out = parax.initial_conditions.standard_initial_condition_generation.generate_airy_1d(4*10**1, -0.21, 10**-1)
    system1 = [50, ['l', 20, 0, 0], 40]
    system2 = [20, ['l', 40, 0, 0], 80]
    system3 = [100]
    mask1, mask2 = parax.experimental_simulator.mask_generator_1d.compute_mask_triple_system(system1, system2, system3, f_in, f_out, check_mask = True)
    
    #Check the beam profile propagation through the optical system
    system = system1 + [['mp', mask1, 0,0]] + system2 + [['mp', mask2, 0,0]] + system3
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f_in, system, output_full = True, forward = True, print_output=False)
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system)
    
    del f