"""Exercise example"""

import pickle
import numpy as np
import matplotlib.animation as manimation
from simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog


def exercise_example(timestep):
    """Exercise example"""
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0,0,0.1],  # Robot position in [m]
            spawn_orientation=[0,0,0],  # Orientation in Euler angles [rad]
            drive=3,  # An example of parameter part of the grid search
            amplitude_gradient = None,
            #amplitudes=[1, 2, 3],  # Just an example
            phase_lag=2*np.pi/10,  # or np.zeros(n_joints) for example
            # Another example
            frequency = 1,
            # ...
        )
        for drive in np.linspace(1, 2, 2)
        # for amplitudes in ...
        # for ...
    ]

    # Grid search
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground' or 'amphibious'
            # fast=True,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video, see below for saving
            # video_distance=1.5,  # Set distance of camera to robot
            # video_yaw=0,  # Set camera yaw for recording
            # video_pitch=-45,  # Set camera pitch for recording
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
        # Save video
        if sim.options.record:
            if 'ffmpeg' in manimation.writers.avail:
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation.mp4',
                    iteration=sim.iteration,
                    writer='ffmpeg',
                )
            elif 'html' in manimation.writers.avail:
                # FFmpeg might not be installed, use html instead
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation.html',
                    iteration=sim.iteration,
                    writer='html',
                )
            else:
                pylog.error('No known writers, maybe you can use: {}'.format(
                    manimation.writers.avail
                ))


if __name__ == '__main__':
    exercise_example(timestep=1e-2)
