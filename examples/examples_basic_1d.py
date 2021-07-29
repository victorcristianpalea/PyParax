# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:07:59 2020

@author: Victor
"""

import pyparax as parax
import numpy as np

"""
Collections of functions that showcase some simple 1-dimensional scenarios.
"""

def propagate_forward_freespace():
    """
    Forward propagates a gaussian beam through an optical system S = [300]. The physical units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.function_generator.standard_initial_conditions.generate_gauss_1d(0,0.1)
    system = [300]
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f0, system, output_full = True, forward = True, print_output=False)
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system, which = 'abs')
    del f
    
def propagate_backward_freespace():
    """
    Backward propagates a gaussian beam through an optical system S = [300]. The units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.function_generator.standard_initial_conditions.generate_gauss_1d(0,0.1)
    system = [300]
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f0, system, output_full = True, forward = False, print_output=False)
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system, forward = False, which = 'abs')
    del f
    
def propagate_forward_lens():
    """
    Forward propagates a gaussian beam through an optical system S = [100, ['l', 75, 0, 0], 200]. The units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.function_generator.standard_initial_conditions.generate_gauss_1d(0,0.5)
    system = [100, ['l', 75, 0, 0], 200]
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f0, system, output_full = True, forward = True, print_output=False)
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system, which = 'abs')
    del f
    
def propagate_backward_lens():
    """
    Backward propagates a gaussian beam through an optical system S = [50, ['l', 75, 0, 0], 150]. The units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.function_generator.standard_initial_conditions.generate_gauss_1d(0,0.5)
    system = [100, ['l', 75, 0, 0], 200]
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f0, system, output_full = True, forward = False, print_output=False)
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system, forward = False, which = 'abs')
    del f
    
def propagate_forward_telescope():
    """
    Forward propagates a gaussian beam through an optical system S = [50, ['l', 20, 0, 0], 80, ['l', 60,0,0], 170]. The units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.function_generator.standard_initial_conditions.generate_gauss_1d(0,0.2)
    system = [50, ['l', 20, 0, 0], 80, ['l', 60,0,0], 170]
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f0, system, output_full = True, forward = True, print_output=False)
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system, which = 'abs')
    del f
    
def propagate_backward_telescope():
    """
    Forward propagates a gaussian beam through an optical system S = [50, ['l', 20, 0, 0], 80, ['l', 60,0,0], 170]. The units are set in the numeric_parameteres.py file. 
    """
    f0 = parax.function_generator.standard_initial_conditions.generate_gauss_1d(0,0.6)
    system = [50, ['l', 20, 0, 0], 80, ['l', 60,0,0], 170]
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f0, system, output_full = True, forward = False, print_output=False)
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system, forward = False, which = 'abs')
    del f
    
def propagate_forward_apperture():
    """
    Forward propagates a gaussian beam through an optical system S = [150, ['ma', M, 0, 0], 150], where M is an apperture of width 1 units. The units are set in the numeric_parameteres.py file. 
    
    Observation: Abrupt borders given by the apperture might introduce numerical artifacts. 
    """
    f0 = parax.function_generator.standard_initial_conditions.generate_gauss_1d(0,0.8)
    M = parax.function_generator.optical_elements.amplitude_mask_circular(1, smooth = 4, dim = 1)
    system = [150, ['ma', M, 0, 0], 150]
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f0, system, output_full = True, forward = True, print_output=False)
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system, which = 'abs')
    del f
    
def propagate_backward_apperture():
    """
    Backward propagates a gaussian beam through an optical system S = [150, ['ma', M, 0, 0], 150], where M is an apperture of width 1 units. The units are set in the numeric_parameteres.py file. 
    
    Observation: This operation although allowed by the solver, it is not physically correct if the beam has a bigger width than the apperture. It should be used with care. 
    """
    f0 = parax.function_generator.standard_initial_conditions.generate_gauss_1d(0,0.8)
    M = parax.function_generator.optical_elements.amplitude_mask_circular(1, smooth = 4, dim = 1)
    system = [150, ['ma', M, 0, 0], 150]
    f = parax.experimental_simulator.experimental_simulator_1d.propagate(f0, system, output_full = True, forward = False, print_output=False)
    print("next to plot")
    parax.experimental_simulator.plotters.full_beam_plot_1d(f, system, forward = False, which = 'abs')
    del f
    