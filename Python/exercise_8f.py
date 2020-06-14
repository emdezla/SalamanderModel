"""Exercise 8f"""

import pickle
import numpy as np
from simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_8f1(timestep):
    """Exercise 8f"""
    # Use exercise_example.py for reference
    parameter_set = [
        SimulationParameters(
            duration=10,
            timestep=timestep,
            spawn_position=[0,0,0.1],  # Robot position in [m]
            spawn_orientation=[0,0,0],  # Orientation in Euler angles [rad]
            amplitude_gradient = True,
            drive=1.5,
            amplitudes=[0.4,0.4],
            frequency=1,
            nominal_radius=0.3,
            phase_lag = phase
        ) for phase in np.linspace(0,2*np.pi,30)
    ]

    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_8f1/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='ground',  # Can also be 'ground' or 'amphibious'
            fast=True,  # For fast mode (not real-time)
            #headless=True,  # For headless mode (No GUI, could be faster)
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

def exercise_8f2(timestep):
    """Exercise 8f"""
    # Use exercise_example.py for reference
    parameter_set = [
        SimulationParameters(
            duration=10,
            timestep=timestep,
            spawn_position=[0,0,0.1],  # Robot position in [m]
            spawn_orientation=[0,0,0],  # Orientation in Euler angles [rad]
            amplitude_gradient = True,
            drive=1.5,
            amplitudes=[0.4,1],
            frequency=1,
            nominal_radius=radius,
            phase_lag = 1.95
        ) for radius in np.linspace(0,1.0,30)
    ]

    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_8f2/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='ground',  # Can also be 'ground' or 'amphibious'
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
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
    exercise_8f1(timestep=1e-2)
    exercise_8f2(timestep=1e-2)
