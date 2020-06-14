"""Robot parameters"""

import numpy as np
import farms_pylog as pylog
from numpy import genfromtxt


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""
        #pylog.warning("Coupling weights must be set")

        if ((parameters.drive <= parameters.dhigh_B) and (parameters.drive >= parameters.dlow_B)):
            self.freqs[:self.n_oscillators_body] = parameters.Cv1_B*parameters.drive + parameters.Cv0_B
        else:
            self.freqs[:self.n_oscillators_body] = parameters.vsat_B

        if ((parameters.drive <= parameters.dhigh_L) and (parameters.drive >= parameters.dlow_L)):
            self.freqs[-self.n_oscillators_legs:] = parameters.Cv1_L*parameters.drive + parameters.Cv0_L
        else:
            self.freqs[-self.n_oscillators_legs:] = parameters.vsat_L
            
        self.freqs *= parameters.frequency

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""

        self.coupling_weights = genfromtxt('weights.csv', delimiter=',')

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        
        self.phase_bias = genfromtxt('phase_lags.csv', delimiter=',')

        if parameters.phase_lag is not None:
            self.phase_bias[0:5,20] = parameters.phase_lag
            self.phase_bias[5:10,21] = parameters.phase_lag
            self.phase_bias[10:15,22] = parameters.phase_lag
            self.phase_bias[15:20,23] = parameters.phase_lag
            
        if parameters.backward:
            self.phase_bias *= -1

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        #pylog.warning("Convergence rates must be set")
        self.rates = np.ones(self.n_oscillators)*20

    def set_nominal_amplitudes(self, p):
        """Set nominal amplitudes"""
        #pylog.warning("Nominal amplitudes must be set")
        if p.nominal_radius != None:
            self.nominal_amplitudes[:self.n_oscillators_body] = p.nominal_radius
        elif ((p.drive <= p.dhigh_B) and (p.drive >= p.dlow_B)):
            self.nominal_amplitudes[:self.n_oscillators_body] = p.CR1_B*p.drive + p.CR0_B
        else:
            self.nominal_amplitudes[:self.n_oscillators_body] = p.Rsat_B

        if ((p.drive <= p.dhigh_L) and (p.drive >= p.dlow_L)):
            self.nominal_amplitudes[-self.n_oscillators_legs:] = p.CR1_L*p.drive + p.CR0_L
        else:
            self.nominal_amplitudes[-self.n_oscillators_legs:] = p.Rsat_L

        if p.amplitude_gradient != None:
            grad_start = p.amplitudes[0]
            grad_end = p.amplitudes[1]
            self.nominal_amplitudes[:self.n_body_joints] *= np.linspace(grad_start, grad_end,(self.n_body_joints))
            self.nominal_amplitudes[self.n_body_joints:2*self.n_body_joints] *= np.linspace(grad_start, grad_end,(self.n_body_joints))

        if(p.turn[1] == 'Right'):
            self.nominal_amplitudes[10:20] = self.nominal_amplitudes[10:20]*p.turn[0]
        elif(p.turn[1] == 'Left'):
            self.nominal_amplitudes[0:10] = self.nominal_amplitudes[0:10]*p.turn[0]
