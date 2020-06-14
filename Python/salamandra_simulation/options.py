"""Animat options"""

from farms_bullet.simulation.options import Options


class SalamandraConvention:
    """Salamandra convention"""

    def __init__(self, **kwargs):
        super(SalamandraConvention, self).__init__()
        self.n_joints_body = kwargs.pop('n_joints_body')
        self.n_dof_legs = kwargs.pop('n_dof_legs')
        self.n_legs = kwargs.pop('n_legs')

    def bodyosc2index(self, joint_i, side=0):
        """body2index"""
        n_body_joints = self.n_joints_body
        assert 0 <= joint_i < n_body_joints, 'Joint must be < {}, got {}'.format(
            n_body_joints, joint_i)
        return 2*joint_i + side

    def legosc2index(self, leg_i, side_i, joint_i, side=0):
        """legosc2index"""
        n_legs = self.n_legs
        n_body_joints = self.n_joints_body
        n_legs_dof = self.n_dof_legs
        assert 0 <= leg_i < n_legs, 'Leg must be < {}, got {}'.format(
            n_legs//2, leg_i)
        assert 0 <= side_i < 2, 'Body side must be < 2, got {}'.format(side_i)
        assert 0 <= joint_i < n_legs_dof, 'Joint must be < {}, got {}'.format(
            n_legs_dof, joint_i)
        assert 0 <= side < 2, 'Oscillator side must be < 2, got {}'.format(
            side)
        return (
            2*n_body_joints
            + leg_i*2*n_legs_dof*2  # 2 oscillators, 2 legs
            + side_i*n_legs_dof*2  # 2 oscillators
            + 2*joint_i
            + side
        )

    def oscindex2information(self, index):
        """Oscillator index information"""
        information = {}
        n_joints = self.n_joints_body + self.n_legs*self.n_dof_legs
        n_oscillators = 2*n_joints
        n_body_oscillators = 2*self.n_joints_body
        assert 0 <= index < n_oscillators, (
            'Index {} bigger than number of oscillator (n={})'.format(
                index,
                n_oscillators,
            )
        )
        information['body'] = index < n_body_oscillators
        information['side'] = index % 2
        if information['body']:
            information['body_link'] = index//2
        else:
            index_i = index - n_body_oscillators
            n_osc_leg = 2*self.n_dof_legs
            n_osc_leg_pair = 2*n_osc_leg
            information['leg'] = index_i // n_osc_leg
            information['leg_i'] = index_i // n_osc_leg_pair
            information['side_i'] = (
                0 if (index_i % n_osc_leg_pair) < n_osc_leg else 1
            )
            information['joint_i'] = (index_i % n_osc_leg)//2
        return information

    def bodylink2name(self, link_i):
        """bodylink2name"""
        n_body = self.n_joints_body+1
        assert 0 <= link_i < n_body, 'Body must be < {}, got {}'.format(
            n_body, link_i)
        return 'link_body_{}'.format(link_i)

    def leglink2index(self, leg_i, side_i, joint_i):
        """leglink2index"""
        n_legs = self.n_legs
        n_body_links = self.n_joints_body+1
        n_legs_dof = self.n_dof_legs
        assert 0 <= leg_i < n_legs, 'Leg must be < {}, got {}'.format(
            n_legs//2, leg_i)
        assert 0 <= side_i < 2, 'Body side must be < 2, got {}'.format(side_i)
        assert 0 <= joint_i < n_legs_dof, 'Joint must be < {}, got {}'.format(
            n_legs_dof, joint_i)
        return (
            n_body_links - 1
            + leg_i*2*n_legs_dof
            + side_i*n_legs_dof
            + joint_i
        )

    def bodyjoint2name(self, link_i):
        """bodyjoint2name"""
        n_body = self.n_joints_body + 1
        assert 0 <= link_i < n_body, 'Body must be < {}, got {}'.format(
            n_body, link_i)
        return 'joint_body_{}'.format(link_i)

    def leglink2name(self, leg_i, side_i, joint_i):
        """leglink2name"""
        n_legs = self.n_legs
        n_legs_dof = self.n_dof_legs
        assert 0 <= leg_i < n_legs, 'Leg must be < {}, got {}'.format(
            n_legs//2, leg_i)
        assert 0 <= side_i < 2, 'Body side must be < 2, got {}'.format(side_i)
        assert 0 <= joint_i < n_legs_dof, 'Joint must be < {}, got {}'.format(
            n_legs_dof, joint_i)
        return 'link_leg_tibia_{}_{}_{}'.format(
            leg_i, 'R' if side_i else 'L', joint_i)

    def bodyjoint2index(self, joint_i):
        """bodyjoint2index"""
        n_body = self.n_joints_body
        assert 0 <= joint_i < n_body, 'Body joint must be < {}, got {}'.format(
            n_body, joint_i)
        return joint_i

    def legjoint2index(self, leg_i, side_i, joint_i):
        """legjoint2index"""
        n_legs = self.n_legs
        n_body_joints = self.n_joints_body
        n_legs_dof = self.n_dof_legs
        assert 0 <= leg_i < n_legs, 'Leg must be < {}, got {}'.format(
            n_legs//2, leg_i)
        assert 0 <= side_i < 2, 'Body side must be < 2, got {}'.format(side_i)
        assert 0 <= joint_i < n_legs_dof, 'Joint must be < {}, got {}'.format(
            n_legs_dof, joint_i)
        return (
            n_body_joints
            + leg_i*2*n_legs_dof
            + side_i*n_legs_dof
            + joint_i
        )

    def legjoint2name(self, leg_i, side_i, joint_i):
        """legjoint2index"""
        n_legs = self.n_legs
        n_legs_dof = self.n_dof_legs
        link_name = self.leglink2name(
            leg_i,
            side_i,
            joint_i
        )
        assert 0 <= leg_i < n_legs, 'Leg must be < {}, got {}'.format(
            n_legs//2, leg_i)
        assert 0 <= side_i < 2, 'Body side must be < 2, got {}'.format(side_i)
        assert 0 <= joint_i < n_legs_dof, 'Joint must be < {}, got {}'.format(
            n_legs_dof, joint_i)
        assert 'link_' in link_name, 'Link_ not in {}'.format(link_name)
        return link_name.replace('link_', 'joint_')

    def contactleglink2index(self, leg_i, side_i):
        """Contact leg link 2 index"""
        n_legs = self.n_legs
        assert 0 <= leg_i < n_legs, 'Leg must be < {}, got {}'.format(
            n_legs//2, leg_i)
        assert 0 <= side_i < 2, 'Body side must be < 2, got {}'.format(side_i)
        return 2*leg_i + side_i

    def joint_names(self):
        """Joint names"""
        return [
            self.bodyjoint2name(i)
            for i in range(self.n_joints_body)
        ] + [
            self.legjoint2name(leg_i, side_i, joint_i)
            for leg_i in range(self.n_legs)
            for side_i in range(2)
            for joint_i in range(self.n_dof_legs)
        ]


class SalamandraOptions(Options):
    """Simulation options"""

    def __init__(self, **kwargs):
        super(SalamandraOptions, self).__init__()
        self.morphology = SalamandraMorphologyOptions(
            **kwargs.pop('morphology'))
        self.spawn = SalamandraSpawnOptions(**kwargs.pop('spawn'))
        self.physics = SalamandraPhysicsOptions(**kwargs.pop('physics'))
        self.joints = SalamandraJointsOptions(**kwargs.pop('joints'))
        self.collect_gps = kwargs.pop('collect_gps')
        self.show_hydrodynamics = kwargs.pop('show_hydrodynamics')
        self.transition = kwargs.pop('transition')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def default(cls):
        """Deafault options"""
        return cls.from_options({})

    @classmethod
    def from_options(cls, kwargs=None):
        """From options"""
        options = {}
        options['morphology'] = kwargs.pop(
            'morphology',
            SalamandraMorphologyOptions.from_options(kwargs)
        )
        options['spawn'] = kwargs.pop(
            'spawn',
            SalamandraSpawnOptions.from_options(kwargs)
        )
        options['physics'] = kwargs.pop(
            'physics',
            SalamandraPhysicsOptions.from_options(kwargs)
        )
        options['joints'] = kwargs.pop(
            'joints',
            SalamandraJointsOptions.from_options(kwargs)
        )
        options['collect_gps'] = kwargs.pop('collect_gps', False)
        options['show_hydrodynamics'] = kwargs.pop('show_hydrodynamics', False)
        options['transition'] = kwargs.pop('transition', False)
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))
        return cls(**options)


class SalamandraMorphologyOptions(Options):
    """Salamandra morphology options"""

    def __init__(self, **kwargs):
        super(SalamandraMorphologyOptions, self).__init__()
        self.mesh_directory = kwargs.pop('mesh_directory')
        self.density = kwargs.pop('density')
        self.n_joints_body = kwargs.pop('n_joints_body')
        self.n_dof_legs = kwargs.pop('n_dof_legs')
        self.n_legs = kwargs.pop('n_legs')
        self.links = kwargs.pop('links')
        self.joints = kwargs.pop('joints')
        self.feet = kwargs.pop('feet')
        self.links_swimming = kwargs.pop('links_swimming')
        self.links_no_collisions = kwargs.pop('links_no_collisions')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['mesh_directory'] = kwargs.pop('mesh_directory', '')
        options['density'] = kwargs.pop('density', 1000.0)
        options['n_joints_body'] = kwargs.pop('n_joints_body', 10)
        options['n_dof_legs'] = kwargs.pop('n_dof_legs', 1)
        options['n_legs'] = kwargs.pop('n_legs', 4)
        convention = SalamandraConvention(**options)
        options['links'] = kwargs.pop('links', [
            convention.bodylink2name(i)
            for i in range(options['n_joints_body']+1)
        ] + [
            convention.leglink2name(leg_i, side_i, link_i)
            for leg_i in range(options['n_legs']//2)
            for side_i in range(2)
            for link_i in range(options['n_dof_legs'])
        ])
        options['joints'] = kwargs.pop('joints', [
            convention.bodyjoint2name(i)
            for i in range(options['n_joints_body'])
        ] + [
            convention.legjoint2name(leg_i, side_i, joint_i)
            for leg_i in range(options['n_legs']//2)
            for side_i in range(2)
            for joint_i in range(options['n_dof_legs'])
        ])
        options['feet'] = kwargs.pop('feet', [
            convention.leglink2name(
                leg_i=leg_i,
                side_i=side_i,
                joint_i=options['n_dof_legs']-1
            )
            for leg_i in range(options['n_legs']//2)
            for side_i in range(2)
        ])
        options['links_swimming'] = kwargs.pop('links_swimming', [
            convention.bodylink2name(body_i)
            for body_i in range(options['n_joints_body']+1)
        ])
        options['links_no_collisions'] = kwargs.pop('links_no_collisions', [
            convention.bodylink2name(body_i)
            for body_i in range(1, options['n_joints_body'])
        ] + [
            convention.leglink2name(leg_i, side_i, joint_i)
            for leg_i in range(options['n_legs']//2)
            for side_i in range(2)
            for joint_i in range(options['n_dof_legs']-1)
        ])
        return cls(**options)

    def n_joints(self):
        """Number of joints"""
        return self.n_joints_body + self.n_legs*self.n_dof_legs

    def n_joints_legs(self):
        """Number of legs joints"""
        return self.n_legs*self.n_dof_legs

    def n_links_body(self):
        """Number of body links"""
        return self.n_joints_body + 1

    def n_links(self):
        """Number of links"""
        return self.n_links_body() + self.n_joints_legs()


class SalamandraSpawnOptions(Options):
    """Salamandra spawn options"""

    def __init__(self, **kwargs):
        super(SalamandraSpawnOptions, self).__init__()
        self.position = kwargs.pop('position')
        self.orientation = kwargs.pop('orientation')
        self.velocity_lin = kwargs.pop('velocity_lin')
        self.velocity_ang = kwargs.pop('velocity_ang')
        self.joints_positions = kwargs.pop('joints_positions')
        self.joints_velocities = kwargs.pop('joints_velocities')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        # Position in [m]
        options['position'] = kwargs.pop('spawn_position', [0, 0, 0.1])
        # Orientation in [rad] (Euler angles)
        options['orientation'] = kwargs.pop('spawn_orientation', [0, 0, 0])
        # Linear velocity in [m/s]
        options['velocity_lin'] = kwargs.pop('spawn_velocity_lin', [0, 0, 0])
        # Angular velocity in [rad/s] (Euler angles)
        options['velocity_ang'] = kwargs.pop('spawn_velocity_ang', [0, 0, 0])
        # Joints positions
        options['joints_positions'] = kwargs.pop('joints_positions', None)
        options['joints_velocities'] = kwargs.pop('joints_velocities', None)
        return cls(**options)


class SalamandraPhysicsOptions(Options):
    """Salamandra physics options"""

    def __init__(self, **kwargs):
        super(SalamandraPhysicsOptions, self).__init__()
        self.drag = kwargs.pop('drag')
        self.drag_coefficients = kwargs.pop('drag_coefficients')
        self.buoyancy = kwargs.pop('buoyancy')
        self.water_surface = kwargs.pop('water_surface')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['drag'] = kwargs.pop('drag', True)
        options['drag_coefficients'] = kwargs.pop(
            'drag_coefficients',
            [
                [-1e-1, -1e1, -1e1],
                [-1e-6, -1e-6, -1e-6],
            ],
        )
        options['buoyancy'] = kwargs.pop(
            'buoyancy',
            True
        )
        options['water_surface'] = kwargs.pop(
            'water_surface',
            0.0
        )
        return cls(**options)


class SalamandraJointsOptions(Options):
    """Salamandra joints options"""

    def __init__(self, **kwargs):
        super(SalamandraJointsOptions, self).__init__()
        self.offsets = kwargs.pop('offsets')
        self.rates = kwargs.pop('rates')
        self.gain_amplitude = kwargs.pop('gain_amplitude')
        self.gain_offset = kwargs.pop('gain_offset')
        self.offsets_bias = kwargs.pop('offsets_bias')
        if kwargs:
            raise Exception('Unknown kwargs: {}'.format(kwargs))

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['offsets'] = kwargs.pop('offsets', None)
        options['rates'] = kwargs.pop('rates', None)
        options['gain_amplitude'] = kwargs.pop('gain_amplitude', None)
        options['gain_offset'] = kwargs.pop('gain_offset', None)
        options['offsets_bias'] = kwargs.pop('offsets_bias', None)
        return cls(**options)

