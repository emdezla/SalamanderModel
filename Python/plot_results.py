"""Plot results"""

import pickle
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from save_figures import save_figures
from parse_args import save_plots
from salamandra_simulation.data import AnimatData


def plot_joint_angles(time,output, osc_left, osc_legs):

    plt.figure()
    n_spine = len(osc_left)
    offset = output[:, :n_spine].max()-output[:, :n_spine].min()

    for i in osc_left:
        plt.plot(time,output[:,i]+(n_spine-i-1)*offset, label=f'X{i}')

    plt.grid()
    plt.legend()
    plt.title("Body_Joints_Angles")

    plt.figure()
    n_limbs = len(osc_legs)
    offset = output[:, :n_limbs].max()-output[:, :n_limbs].min()

    for i in osc_legs:
        plt.plot(time,output[:,i]+(n_limbs-i-1)*offset, label=f'X{i}')

    plt.grid()
    plt.legend()
    plt.title("Limb_Joints_Angles")

def plot_3d_variable(times, variable_log, variable_name):
    """ Plots a variable as a 3D surface depending on joint_number (Y axis) and time(X axis)"""
    nb_joints = variable_log.shape[1]

    # 3D data containers
    times_3d = np.zeros_like(variable_log)
    joint_number_3d = np.zeros_like(variable_log)

    for i in range(nb_joints):
        times_3d[:, i] = times

    for i in range(len(times)):
        joint_number_3d[i, :] = np.arange(nb_joints)

    # 3D plots
    fig = plt.figure(variable_name.replace(" ", "_")+"_plot")
    plt.title(variable_name)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_2d(times_3d, joint_number_3d, variable_log, cmap='coolwarm')
    ax.set_xlabel('time')
    ax.set_ylabel('joint number')
    ax.set_zlabel(variable_name)


def exercise_8b_plot_gridsearch():

    amplitudes=[]
    phase_lags=[]
    speeds=[]
    energies=[]
    performance=[]


    for i in range(0,100):
        path='logs/exercise_8b/simulation_'+str(i)
        data = AnimatData.from_file(path+'.h5', 2*14)
        with open(path+'.pickle', 'rb') as param_file:
            parameters = pickle.load(param_file)

        times = data.times
        timestep = times[1] - times[0]
        amplitudes = np.append(amplitudes,parameters.amplitudes)
        phase_lags = np.append(phase_lags,parameters.phase_lag)

        # General features
        osc_phases = data.state.phases_all()
        osc_amplitudes = data.state.amplitudes_all()
        links_positions = data.sensors.gps.urdf_positions()
        head_positions = links_positions[:, 0, :]
        tail_positions = links_positions[:, 10, :]
        joints_positions = data.sensors.proprioception.positions_all()
        joints_velocities = data.sensors.proprioception.velocities_all()
        joints_torques = data.sensors.proprioception.motor_torques()

        # Speed
        xvelocity = np.diff(head_positions[:,0], axis=0) / timestep
        yvelocity = np.diff(head_positions[:,1], axis=0) / timestep
        speeds= np.append(speeds,np.mean(xvelocity)+np.mean(yvelocity))

        # Energy
        energies = np.append(energies, np.cumsum(abs(np.asarray(joints_torques)*np.asarray(joints_velocities)*timestep))[-1])

        #Performance
        performance=np.append(performance,speeds[-1]/energies[-1])

    plt.figure("Speed")
    results_speed = np.concatenate((phase_lags.reshape(-1,1),amplitudes.reshape(-1,1),speeds.reshape(-1,1)),axis=1)
    plot_2d(results_speed,["Phase_lag","Oscillation amplitude","Salamander Speed"],cmap='coolwarm')
    plt.title("Speed over phase lag and amplitude")


    plt.figure("Energy")
    results_energy = np.concatenate((phase_lags.reshape(-1,1),amplitudes.reshape(-1,1),energies.reshape(-1,1)),axis=1)
    plot_2d(results_energy, ["Phase_lag","Oscillation amplitude","Salamander Energy"],cmap='coolwarm')
    plt.title("Energy over phase lag and amplitude")

    plt.figure("Performance")
    results_performance = np.concatenate((phase_lags.reshape(-1,1),amplitudes.reshape(-1,1),performance.reshape(-1,1)),axis=1)
    plot_2d(results_performance, ["Phase_lag","Oscillation amplitude","Parameter performance"],cmap='coolwarm')
    plt.title("Performance over phase lag and amplitude")

    index = np.where(performance == performance.max())[0].item()
    print('Phase lag={}, Amplitude={}'.format(phase_lags[index], amplitudes[index]))
    print('Speed={}, Energy={}'.format(speeds[index], energies[index]))

    return index

def exercise_8c_plot_gridsearch():

    Rhead=[]
    Rtail=[]
    speeds=[]
    energies=[]
    performance=[]


    for i in range(0,100):
        path='logs/exercise_8c/simulation_'+str(i)
        data = AnimatData.from_file(path+'.h5', 2*14)
        with open(path+'.pickle', 'rb') as param_file:
            parameters = pickle.load(param_file)

        times = data.times
        timestep = times[1] - times[0]
        Rhead = np.append(Rhead,parameters.amplitudes[0])
        Rtail = np.append(Rtail,parameters.amplitudes[1])

        # General features
        osc_phases = data.state.phases_all()
        osc_amplitudes = data.state.amplitudes_all()
        links_positions = data.sensors.gps.urdf_positions()
        head_positions = links_positions[:, 0, :]
        tail_positions = links_positions[:, 10, :]
        joints_positions = data.sensors.proprioception.positions_all()
        joints_velocities = data.sensors.proprioception.velocities_all()
        joints_torques = data.sensors.proprioception.motor_torques()

        # Speed
        xvelocity = np.diff(head_positions[:,0], axis=0) / timestep
        yvelocity = np.diff(head_positions[:,1], axis=0) / timestep
        speeds= np.append(speeds,np.mean(xvelocity)+np.mean(yvelocity))

        # Energy
        energies = np.append(energies, np.cumsum(abs(np.asarray(joints_torques)*np.asarray(joints_velocities)*timestep))[-1])

        #Performance
        performance=np.append(performance,speeds[-1]/energies[-1])

    plt.figure("Speed")
    results_speed = np.concatenate((Rhead.reshape(-1,1),Rtail.reshape(-1,1),speeds.reshape(-1,1)),axis=1)
    plot_2d(results_speed,["Rhead","Rtail","Salamander Speed"],cmap='coolwarm')
    plt.title("Speed over amplitude's gradient")


    plt.figure("Energy")
    results_energy = np.concatenate((Rhead.reshape(-1,1),Rtail.reshape(-1,1),energies.reshape(-1,1)),axis=1)
    plot_2d(results_energy, ["Rhead","Rtail","Salamander Energy"],cmap='coolwarm')
    plt.title("Energy over amplitude's gradient")

    plt.figure("Performance")
    results_performance = np.concatenate((Rhead.reshape(-1,1),Rtail.reshape(-1,1),performance.reshape(-1,1)),axis=1)
    plot_2d(results_performance, ["Rhead","Rtail","Parameter performance"],cmap='coolwarm')
    plt.title("Performance over amplitude's gradient")

    index = np.where(performance == performance.max())[0].item()
    print('Rhead={}, Rtail={}'.format(Rhead[index], Rtail[index]))
    print('Speed={}, Energy={}'.format(speeds[index], energies[index]))

    return index
def plot_efficient_behaviour_8c(i):

    path='logs/exercise_8c/simulation_'+str(i)
    data = AnimatData.from_file(path+'.h5', 2*14)
    with open(path+'.pickle', 'rb') as param_file:
        parameters = pickle.load(param_file)

    times = data.times
    timestep = times[1] - times[0]
    amplitudes = parameters.amplitudes
    phase_lags = parameters.phase_lag

    # General features
    osc_phases = data.state.phases_all()
    osc_amplitudes = data.state.amplitudes_all()
    links_positions = data.sensors.gps.urdf_positions()
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 10, :]
    joints_positions = data.sensors.proprioception.positions_all()
    joints_velocities = data.sensors.proprioception.velocities_all()
    joints_torques = data.sensors.proprioception.motor_torques()

    plt.figure("Positions")
    plot_positions(times, head_positions)
    plt.title('Positions of the head')
    plt.figure("Trajectory")
    plot_trajectory(head_positions)
    plt.title('Trajectory of the salamander')

def plot_efficient_behaviour(i):

    path='logs/exercise_8c/simulation_'+str(i)
    data = AnimatData.from_file(path+'.h5', 2*14)
    with open(path+'.pickle', 'rb') as param_file:
        parameters = pickle.load(param_file)

    times = data.times
    timestep = times[1] - times[0]
    amplitudes = parameters.amplitudes
    phase_lags = parameters.phase_lag

    # General features
    osc_phases = data.state.phases_all()
    osc_amplitudes = data.state.amplitudes_all()
    links_positions = data.sensors.gps.urdf_positions()
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 10, :]
    joints_positions = data.sensors.proprioception.positions_all()
    joints_velocities = data.sensors.proprioception.velocities_all()
    joints_torques = data.sensors.proprioception.motor_torques()

    plt.figure("Positions")
    plot_positions(times, head_positions)
    plt.figure("Trajectory")
    plot_trajectory(head_positions)

def plot_positions(times, link_data, hline=False, yline=0):
    """Plot positions"""
    plt.figure()

    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=["x", "y", "z"][i])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.grid(True)
    if hline:
        plt.hlines(yline, 0, 20, colors='r', linestyles='dashdot', label='Transition from walk to swim')


def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear'  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], "r.")
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation="none",
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])

def plot_phase_angle_trajectory(file,param=None):
    data = AnimatData.from_file('logs/{}{}/simulation_0.h5'.format(file,param), 2*14)
    with open('logs/{}{}/simulation_0.pickle'.format(file,param), 'rb') as param_file:
        parameters = pickle.load(param_file)
    times = data.times
    timestep = times[1] - times[0]  # Or parameters.timestep
    amplitudes = parameters.amplitudes
    phase_lag = parameters.phase_lag
    osc_phases = data.state.phases_all()
    osc_amplitudes = data.state.amplitudes_all()
    links_positions = data.sensors.gps.urdf_positions()
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 10, :]
    joints_positions = data.sensors.proprioception.positions_all()
    joints_velocities = data.sensors.proprioception.velocities_all()
    joints_torques = data.sensors.proprioception.motor_torques()
    # Notes:
    # For the gps arrays: positions[iteration, link_id, xyz]
    # For the positions arrays: positions[iteration, xyz]
    # For the joints arrays: positions[iteration, joint]

    # Plot data
    #plt.figure("spine_angles")
    #plot_joint_angles(times, head_positions)
    if param ==1 :
        title= 'turning'
    elif param==2:
        title = 'backward'
    else:
        title=file
    plt.figure("Trajectory {}{}".format(file,param))
    plot_trajectory(head_positions)
    plt.title('Trajectory {}'.format(title))
    plt.legend(['Head Position'])
    plot_joint_angles(times,np.sin(osc_phases),[0,1,2,3,4,5,6,7,8,9], [10,11,12,13])


def plot_8f1():
    """Plot velocity over phase lag"""

    velocities = []
    phase_lags = []

    for sim in range(0,30):
        data = AnimatData.from_file('logs/exercise8f1/simulation{0}.h5'.format(sim), 214)
        with open('logs/exercise8f1/simulation{0}.pickle'.format(sim), 'rb') as param_file:
            parameters = pickle.load(param_file)
        velocities.append(getSpeedOfSim(data, parameters.timestep))
        phase_lags.append(parameters.phase_lag)

    print(np.max(velocities), phase_lags[np.argmax(velocities)])
    plt.plot(phase_lags,velocities)
    plt.title("Mean Velocity Over Phase Lag")
    plt.xlabel("Phase Lag [s]")
    plt.ylabel("Velocity [m/s]")
    plt.grid(True)

def plot_8f2():
    velocities = []
    oscillations = np.linspace(0,1.0,30)

    for sim in range(0,30):
        data = AnimatData.from_file('logs/exercise8f2/simulation{0}.h5'.format(sim), 214)
        with open('logs/exercise8f2/simulation{0}.pickle'.format(sim), 'rb') as param_file:
            parameters = pickle.load(param_file)

        velocities.append(getSpeedOfSim(data, parameters.timestep))

    print(np.max(velocities), oscillations[np.argmax(velocities)])
    plt.plot(oscillations,velocities)
    plt.title("Mean Velocity Over Oscillation Amplitude")
    plt.xlabel("Oscillation Amplitude [m]")
    plt.ylabel("Velocity [m/s]")
    plt.grid(True)

def plot_8g():
    for sim in range(0,2):
        data = AnimatData.from_file('logs/exercise_8g/simulation_{}.h5'.format(sim), 2*14)

        links_positions = data.sensors.gps.urdf_positions()
        head_positions = links_positions[:, 0, :]
        plot_joint_angles(data.times,np.sin(data.state.phases_all()),[0,1,2,3,4,5,6,7,8,9], [10,11,12,13])

        if (sim == 0):
            plot_positions(data.times, head_positions, True, -2.3)
        else:
            plot_positions(data.times, head_positions, True, -1.4)


def main(plot=True):
    """Main"""
    # Load data

    # 8b
    #efficient_index = exercise_8b_plot_gridsearch()
    #plot_efficient_behaviour(efficient_index)

    # 8c
    #efficient_index = exercise_8c_plot_gridsearch()
    #plot_efficient_behaviour_8c(efficient_index)

    # 8d
    #plot_phase_angle_trajectory('exercise_8d',1)
    #plot_phase_angle_trajectory('exercise_8d',2)

    # 8f
    #plot_8f1()
    #plot_8f2()

    # 8g
    #plot_8g()




        # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())
