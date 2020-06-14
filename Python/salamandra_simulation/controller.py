"""Network controller"""

import numpy as np
from farms_bullet.model.control import ModelController


class SalamandraController(ModelController):
    """Salamandra network"""

    def __init__(self, joints, animat_data, network):
        super(SalamandraController, self).__init__(
            joints=joints,
            use_position=True,
            use_torque=False,
        )
        self.network = network
        self.animat_data = animat_data
        size = len(joints)
        self.gain_amplitude = np.ones(size)
        self.gain_offset = np.ones(size)
        self.joints_bias = np.zeros(size)

    def step(self, iteration, time, timestep):
        """Control step"""
        self.network.step(iteration, time, timestep)

    def positions(self, iteration):
        """Postions"""
        return self.network.get_motor_position_output(iteration)

