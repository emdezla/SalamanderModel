"""Animat data"""

import os
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from farms_bullet.data.data import (
    SensorsData,
    ContactsArray,
    ProprioceptionArray,
    GpsArray,
    HydrodynamicsArray,
)

NPDTYPE = np.float64
NPITYPE = np.uintc


def to_array(array, iteration=None):
    """To array or None"""
    if array is not None:
        array = np.array(array)
        if iteration is not None:
            array = array[:iteration]
    return array


class AnimatData:
    """Animat data"""

    def __init__(self, times, state, sensors):
        super(AnimatData, self).__init__()
        self.times = times
        self.state = state
        self.sensors = sensors

    @classmethod
    def from_dict(cls, dictionary, n_oscillators=2*14):
        """Load data from dictionary"""
        return cls(
            times=dictionary['times'],
            state=OscillatorNetworkState(dictionary['state'], n_oscillators),
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    @classmethod
    def from_file(cls, filename, n_oscillators=2*14):
        """From file"""
        return cls.from_dict(dd.io.load(filename), n_oscillators)

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        return {
            'times': self.times,
            'state': to_array(self.state.array),
            'sensors': self.sensors.to_dict(iteration),
        }

    def to_file(self, filename, iteration=None):
        """Save data to file"""
        directory = os.path.dirname(filename)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        dd.io.save(filename, self.to_dict(iteration))

    def plot(self):
        """Plot"""
        self.state.plot(self.times)
        self.sensors.plot(self.times)


class OscillatorNetworkState:
    """Network state"""

    def __init__(self, state, n_oscillators):
        super(OscillatorNetworkState, self).__init__()
        self.array = state
        self.n_oscillators = n_oscillators

    @classmethod
    def from_options(cls, state, animat_options):
        """From options"""
        return cls(
            state=state,
            n_oscillators=2*animat_options.morphology.n_joints()
        )

    @classmethod
    def from_solver(cls, solver, n_oscillators):
        """From solver"""
        return cls(solver.state, n_oscillators)

    def phases(self, iteration):
        """Phases"""
        return self.array[iteration, :self.n_oscillators]

    def phases_all(self):
        """Phases"""
        return self.array[:, :self.n_oscillators]

    def amplitudes(self, iteration):
        """Amplitudes"""
        return self.array[iteration, self.n_oscillators:]

    def amplitudes_all(self):
        """Phases"""
        return self.array[:, self.n_oscillators:]

    @classmethod
    def from_state(cls, state):
        """From initial state"""
        state_size = np.shape(state)[1]
        return cls(state, n_oscillators=state_size//2)

    def plot(self, times):
        """Plot"""
        self.plot_phases(times)
        self.plot_amplitudes(times)

    def plot_phases(self, times):
        """Plot phases"""
        plt.figure('Network state phases')
        for data in np.transpose(self.phases_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Phases [rad]')
        plt.grid(True)

    def plot_amplitudes(self, times):
        """Plot amplitudes"""
        plt.figure('Network state amplitudes')
        for data in np.transpose(self.amplitudes_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Amplitudes')
        plt.grid(True)


class SalamandraData(AnimatData):
    """Salamandra network parameter"""

    @classmethod
    def from_options(
            cls,
            times,
            state,
            morphology,
            n_iterations
    ):
        """Default salamandra newtwork parameters"""
        state = OscillatorNetworkState.from_state(state)
        sensors = SensorsData(
            contacts=ContactsArray.from_size(
                morphology.n_legs,
                n_iterations,
            ),
            proprioception=ProprioceptionArray.from_size(
                morphology.n_joints(),
                n_iterations,
            ),
            gps=GpsArray.from_size(
                morphology.n_links(),
                n_iterations,
            ),
            hydrodynamics=HydrodynamicsArray.from_size(
                morphology.n_links_body(),
                n_iterations,
            )
        )
        return cls(times, state, sensors)

