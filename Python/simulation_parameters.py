"""Simulation parameters"""


class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 10
        self.n_legs_joints = 4
        self.duration = 30
        self.timestep=None  # Simulation timestep in [s]
        self.spawn_position = None # Robot position in [m]
        self.spawn_orientation = None  # Orientation in Euler angles [rad]
        self.frequency = None
        self.amplitudes = [0.5, 0.5]
        self.amplitude_gradient = None
        self.turn = [1, 'Right'] # 1 = no turning
        self.phase_lag = None
        self.nominal_radius = None
        self.walk = False
        self.backward = False

        self.drive = None

        #Body oscillators
        self.dlow_B = 1
        self.dhigh_B = 5

        self.Cv1_B = 0.2
        self.Cv0_B = 0.3
        self.vsat_B = 0.0

        self.CR1_B = 0.065
        self.CR0_B = 0.196
        self.Rsat_B = 0.0

        #Limb oscillators
        self.dlow_L = 1
        self.dhigh_L = 3

        self.Cv1_L = 0.2
        self.Cv0_L = 0.0
        self.vsat_L = 0.0

        self.CR1_L = 0.131
        self.CR0_L = 0.131
        self.Rsat_L = 0.0

        # ...
        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)
