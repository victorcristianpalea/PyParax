#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:16:15 2020

@author: victor
"""

import pyparax as parax
import numpy as np

"""
Collections of functions that showcase some simple 2-dimensional scenarios.
"""

parax.numeric_parameters.N_x = 600
parax.numeric_parameters.N_y = 600
parax.numeric_parameters.dz = 0.5

def propagate_forward_freespace():
    """
    Forward propagates a gaussian beam through an optical system S = [300]. The physical units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.1,0.1)
    system = [300]
    f = parax.experimental_simulator.experimental_simulator_2d.propagate(f0, system, output_full = True, forward = True, print_output=True)
    del f
    
def propagate_backward_freespace():
    """
    Backward propagates a gaussian beam through an optical system S = [300]. The units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.1,0.1)
    system = [300]
    f = parax.experimental_simulator.experimental_simulator_2d.propagate(f0, system, output_full = True, forward = False, print_output=True)
    del f

def propagate_forward_lens():
    """
    Forward propagates a gaussian beam through an optical system S = [100, ['l', 75, 0, 0], 200]. The units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.5,0.5)
    system = [100, ['l', 75, 0, 0], 200]
    f = parax.experimental_simulator.experimental_simulator_2d.propagate(f0, system, output_full = True, forward = True, print_output=True)
    del f
    
def propagate_backward_lens():
    """
    Backward propagates a gaussian beam through an optical system S = [50, ['l', 75, 0, 0], 150]. The units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.5,0.5)
    system = [100, ['l', 75, 0, 0], 200]
    f = parax.experimental_simulator.experimental_simulator_2d.propagate(f0, system, output_full = True, forward = False, print_output=True)
    del f
    
def propagate_forward_telescope():
    """
    Forward propagates a gaussian beam through an optical system S = [50, ['l', 20, 0, 0], 80, ['l', 60,0,0], 170]. The units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.1,0.1)
    system = [50, ['l', 20, 0, 0], 80, ['l', 60,0,0], 170]
    f = parax.experimental_simulator.experimental_simulator_2d.propagate(f0, system, output_full = True, forward = True, print_output=True)
    del f
    
def propagate_backward_telescope():
    """
    Forward propagates a gaussian beam through an optical system S = [50, ['l', 20, 0, 0], 80, ['l', 60,0,0], 170]. The units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.3,0.3)
    system = [50, ['l', 20, 0, 0], 80, ['l', 60,0,0], 170]
    f = parax.experimental_simulator.experimental_simulator_2d.propagate(f0, system, output_full = True, forward = False, print_output=True)
    del f
    
def propagate_forward_apperture():
    """
    Forward propagates a gaussian beam through an optical system S = [150, ['ma', M, 0, 0], 150], where M is an apperture of width 1 units. The units are set in the numeric_parameteres.py file. 
    
    Observation: Abrupt borders given by the apperture might introduce numerical artifacts. 
    """
    f0 = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.8,0.8)
    M = parax.initial_conditions.optical_elements.amplitude_mask_circular(0.5, smooth = 4, dim = 2)
    system = [150, ['ma', M, 0, 0], 150]
    f = parax.experimental_simulator.experimental_simulator_2d.propagate(f0, system, output_full = True, forward = True, print_output=True)
    del f
    
def propagate_backward_apperture():
    """
    Backward propagates a gaussian beam through an optical system S = [150, ['ma', M, 0, 0], 150], where M is an apperture of width 1 units. The units are set in the numeric_parameteres.py file. 
    
    Observation: This operation although allowed by the solver, it is not physically correct if the beam has a bigger width than the apperture. It should be used with care. 
    """
    f0 = parax.initial_conditions.standard_initial_condition_generation.generate_gauss_2d(0,0,0.8,0.8)
    M = parax.initial_conditions.optical_elements.amplitude_mask_circular(0.5, smooth = 4, dim = 2)
    system = [150, ['ma', M, 0, 0], 150]
    f = parax.experimental_simulator.experimental_simulator_2d.propagate(f0, system, output_full = True, forward = False, print_output=True)
    del f