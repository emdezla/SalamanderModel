"""Run network without Pybullet"""

import time
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog
from network import SalamandraNetwork
from save_figures import save_figures
from parse_args import save_plots
from simulation_parameters import SimulationParameters
from plot_results import plot_joint_angles


def run_network(duration, update=False, drive=0):
    """Run network without Pybullet and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        description
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    timestep = 1e-2
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    sim_parameters = SimulationParameters(
        drive=drive,
        amplitude_gradient=None,
        phase_lag=None,
        turn=None,
    )
    network = SalamandraNetwork(sim_parameters, n_iterations)
    osc_left = np.arange(10)
    osc_right = np.arange(10, 20)
    osc_legs = np.arange(20, 24)

    # Logs
    phases_log = np.zeros([n_iterations,len(network.state.phases(iteration=0))])
    phases_log[0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([ n_iterations,len(network.state.amplitudes(iteration=0))])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([n_iterations,len(network.robot_parameters.freqs) ])
    freqs_log[0, :] = network.robot_parameters.freqs
    outputs_log = np.zeros([n_iterations,len(network.get_motor_position_output(iteration=0))])
    outputs_log[0, :] = network.get_motor_position_output(iteration=0)
    drive_log = np.zeros([n_iterations,1])

    # Run network ODE and log data
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            drive = drive+timestep*0.2
            network.robot_parameters.update(
                SimulationParameters(
                     drive = drive ,
                )
            )
        network.step(i, time0, timestep)
        drive_log[i+1] = drive
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        outputs_log[i+1, :] = network.get_motor_position_output(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs
    # # Alternative option
    # phases_log[:, :] = network.state.phases()
    # amplitudes_log[:, :] = network.state.amplitudes()
    # outputs_log[:, :] = network.get_motor_position_output()
    toc = time.time()

    weights = network.robot_parameters.coupling_weights
    bias = network.robot_parameters.phase_bias
    # Network performance
    pylog.info("Time to run simulation for {} steps: {} [s]".format(
        n_iterations,
        toc - tic
    ))

    # Implement plots of network results

    plot_joint_angles(times,outputs_log, osc_right, osc_left, osc_legs)
    
    plt.figure()
    plt.plot(drive_log,freqs_log)
    plt.legend(('Body','Limb'))
    plt.xlabel('Drive')
    plt.ylabel('Frequency [Hz]')
    plt.grid(True)
    
    plt.figure()
    plt.plot(drive_log,phases_log)
    #plt.legend(('Body','Limb'))
    plt.xlabel('Drive')
    plt.ylabel('Phases')
    plt.grid(True)
    
    plt.figure()
    plt.plot(drive_log,amplitudes_log)
    plt.legend(('Body','Limb'))
    plt.xlabel('Drive')
    plt.ylabel('Nominal Amplitude')
    plt.grid(True)
    
    plt.figure()
    plt.plot(times,drive_log)
    plt.xlabel('Time(s)')
    plt.ylabel('Drive')
    plt.grid(True)
    
            
        
    #plot(times,output_log[])
    #pylog.warning("Implement plots")

def main(plot):
    """Main"""

    run_network(duration=30,update=True)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

