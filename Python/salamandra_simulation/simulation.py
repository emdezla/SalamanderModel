"""Simulation"""

import os
import time
import numpy as np
import pybullet

from farms_bullet.simulation.simulation import Simulation
from farms_bullet.simulation.simulator import real_time_handing
from farms_bullet.model.animat import Animat
from farms_bullet.model.model import SimulationModels
from farms_bullet.interface.interface import Interfaces
from farms_bullet.interface.interface import UserParameters, DebugParameter
from farms_bullet.sensors.sensors import (
    Sensors,
    JointsStatesSensor,
    ContactsSensors,
    LinksStatesSensor,
)
from farms_bullet.plugins.swimming import (
    drag_forces,
    swimming_motion,
    swimming_debug
)
import farms_pylog as pylog

from .controller import SalamandraController
from .data import SalamandraData


def initial_pose(identity, spawn_options, units):
    """Initial pose"""
    pybullet.resetBasePositionAndOrientation(
        identity,
        spawn_options.position,
        pybullet.getQuaternionFromEuler(
            spawn_options.orientation
        )
    )
    pybullet.resetBaseVelocity(
        objectUniqueId=identity,
        linearVelocity=np.array(spawn_options.velocity_lin)*units.velocity,
        angularVelocity=np.array(spawn_options.velocity_ang)/units.seconds
    )
    if (
            spawn_options.joints_positions is not None
            or spawn_options.joints_velocities is not None
    ):
        if spawn_options.joints_positions is None:
            spawn_options.joints_positions = np.zeros_like(
                spawn_options.joints_velocities
            ).tolist()
        if spawn_options.joints_velocities is None:
            spawn_options.joints_velocities = np.zeros_like(
                spawn_options.joints_positions
            ).tolist()
        for joint_i, (position, velocity) in enumerate(zip(
                spawn_options.joints_positions,
                spawn_options.joints_velocities
        )):
            pybullet.resetJointState(
                bodyUniqueId=identity,
                jointIndex=joint_i,
                targetValue=position,
                targetVelocity=velocity/units.seconds
            )


def swimming_step(iteration, animat):
    """Swimming step"""
    physics_options = animat.options.physics
    if physics_options.drag:
        water_surface = physics_options.water_surface
        animat.drag_swimming_forces(
            iteration,
            water_surface=water_surface,
            coefficients=physics_options.drag_coefficients,
            buoyancy=physics_options.buoyancy,
        )
        animat.apply_swimming_forces(
            iteration,
            water_surface=water_surface,
        )
        if animat.options.show_hydrodynamics:
            animat.draw_hydrodynamics(
                iteration,
                water_surface=water_surface,
            )


class SalamandraSimulation(Simulation):
    """Salamandra simulation"""

    def __init__(self, simulation_options, animat, arena):
        super(SalamandraSimulation, self).__init__(
            models=SimulationModels([animat, arena]),
            options=simulation_options
        )
        # Interface
        self.interface = Interfaces(
            user_params=SalamandraUserParameters(self.animat().options)
        )
        if not self.options.headless:
            self.interface.init_camera(
                target_identity=(
                    self.animat().identity()
                    if not self.options.free_camera
                    else None
                ),
                timestep=self.options.timestep,
                rotating_camera=self.options.rotating_camera,
                top_camera=self.options.top_camera,
                pitch=simulation_options.video_pitch,
                yaw=simulation_options.video_yaw,
            )

        if self.options.record:
            self.interface.init_video(
                target_identity=self.animat().identity(),
                simulation_options=simulation_options,
                pitch=simulation_options.video_pitch,
                yaw=simulation_options.video_yaw,
                motion_filter=1e-1,
                rotating_camera=self.options.rotating_camera,
            )
        # Real-time handling
        self.tic_rt = np.zeros(2)
        # Simulation state
        self.simulation_state = None
        self.save()

    def animat(self):
        """Salamander animat"""
        return self.models[0]

    def pre_step(self, iteration):
        """New step"""
        play = True
        if not self.options.headless:
            play = self.interface.user_params.play().value
            if not iteration % 100:
                self.interface.user_params.update()
            if not play:
                time.sleep(0.5)
                self.interface.user_params.update()
        return play

    def step(self, iteration):
        """Simulation step"""
        self.tic_rt[0] = time.time()
        animat = self.animat()

        # Interface
        if not self.options.headless:
            self.animat_interface(iteration)

        # Animat sensors
        animat.sensors.update(iteration)

        # Physics step
        if iteration < self.options.n_iterations-1:
            # Swimming
            swimming_step(iteration, animat)

            # Update animat controller
            if animat.controller is not None:
                animat.controller.step(
                    iteration=iteration,
                    time=iteration*self.options.timestep,
                    timestep=self.options.timestep,
                )

    def post_step(self, iteration):
        """Post step"""

        # Camera
        if not self.options.headless:
            self.interface.camera.update()
        if self.options.record:
            self.interface.video.record(iteration)

        # Real-time
        if not self.options.headless:
            self.tic_rt[1] = time.time()
            if (
                    not self.options.fast
                    and self.interface.user_params.rtl().value < 2.99
            ):
                real_time_handing(
                    self.options.timestep,
                    self.tic_rt,
                    rtl=self.interface.user_params.rtl().value
                )

    def animat_interface(self, iteration):
        """Animat interface"""
        animat = self.animat()

        # Camera zoom
        if self.interface.user_params.zoom().changed:
            self.interface.camera.set_zoom(
                self.interface.user_params.zoom().value
            )

        # Drives
        if self.interface.user_params.drive_left().changed:
            pylog.debug(
                'Switch of behaviour can be implemented here (OPTIONAL)')
            # Example: animat.data.network.drives...
            self.interface.user_params.drive_left().changed = False
        if self.interface.user_params.drive_right().changed:
            pylog.debug(
                'Switch of behaviour can be implemented here (OPTIONAL)')
            # Example: animat.data.network.drives...
            self.interface.user_params.drive_right().changed = False

    def postprocess(
            self,
            iteration,
            log_path='',
            plot=False,
            video='',
            **kwargs
    ):
        """Plot after simulation"""
        times = np.arange(
            0,
            self.options.timestep*self.options.n_iterations,
            self.options.timestep
        )[:iteration]
        animat = self.animat()

        # Log
        if log_path:
            pylog.info('Saving data to {}'.format(log_path))
            animat.data.to_file(
                os.path.join(
                    log_path,
                    'simulation.hdf5'
                ),
                iteration,
            )
            self.options.save(os.path.join(
                log_path,
                'simulation_options.yaml'
            ))
            animat.options.save(os.path.join(
                log_path,
                'animat_options.yaml'
            ))

        # Plot
        if plot:
            animat.data.plot(times)

        # Record video
        if video:
            self.interface.video.save(
                video,
                iteration=iteration,
                writer=kwargs.pop('writer', 'ffmpeg')
            )


class SalamandraUserParameters(UserParameters):
    """Salamandra user parameters"""

    def __init__(self, options):
        super(SalamandraUserParameters, self).__init__()
        self['drive_left'] = DebugParameter(
            'Drive left',
            0,
            0, 6
        )
        self['drive_right'] = DebugParameter(
            'Drive right',
            0,
            0, 6
        )

    def drive_left(self):
        """Drive left"""
        return self['drive_left']

    def drive_right(self):
        """Drive right"""
        return self['drive_right']


class Salamandra(Animat):
    """Salamandra animat"""

    def __init__(self, sdf, options, controller, timestep, iterations, units):
        super(Salamandra, self).__init__(options=options)
        self.sdf = sdf
        self.timestep = timestep
        self.n_iterations = iterations
        self.controller = controller
        self.data = (
            controller.animat_data
            if controller is not None
            else None
        )
        # Hydrodynamic forces
        n_links = 11
        self.masses = np.zeros(n_links)
        self.hydrodynamics = None
        # Sensors
        self.sensors = Sensors()
        # Physics
        self.units = units

    def links_identities(self):
        """Links"""
        return [
            self._links[link]
            for link in self.options.morphology.links
        ]

    def joints_identities(self):
        """Joints"""
        return [self._joints[joint]
                for joint in self.options.morphology.joints]

    def spawn(self):
        """Spawn salamandra"""
        # Spawn
        self.spawn_sdf()
        # Sensors
        if self.data:
            self.add_sensors()
        # Body properties
        self.set_body_properties()
        # Debug
        self.hydrodynamics = [
            pybullet.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, 0],
                lineColorRGB=[0, 0, 0],
                lineWidth=3*self.units.meters,
                lifeTime=0,
                parentObjectUniqueId=self.identity(),
                parentLinkIndex=i
            )
            for i in range(self.options.morphology.n_links_body())
        ]

    def spawn_sdf(self, verbose=False):
        """Spawn sdf"""
        if verbose:
            pylog.debug(self.sdf)
        self._identity = pybullet.loadSDF(
            self.sdf,
            useMaximalCoordinates=0,
            globalScaling=1
        )[0]
        initial_pose(self._identity, self.options.spawn, self.units)
        for joint_i in range(pybullet.getNumJoints(self.identity())):
            joint_info = pybullet.getJointInfo(self.identity(), joint_i)
            self._links[joint_info[12].decode('UTF-8')] = joint_i
            self._joints[joint_info[1].decode('UTF-8')] = joint_i
        if self.options.morphology.links is not None:
            for link in self.options.morphology.links:
                if link not in self._links:
                    self._links[link] = -1
                    break
            for link in self.options.morphology.links:
                assert link in self._links, 'Link {} not in {}'.format(
                    link,
                    self._links,
                )
        if verbose:
            self.print_information()

    def add_sensors(self):
        """Add sensors"""
        # Links
        if self.options.morphology.links is not None:
            self.sensors.add({
                'links': LinksStatesSensor(
                    array=self.data.sensors.gps.array,
                    animat_id=self.identity(),
                    links=self.links_identities(),
                    units=self.units
                )
            })
        # Joints
        if self.options.morphology.joints is not None:
            self.sensors.add({
                'joints': JointsStatesSensor(
                    self.data.sensors.proprioception.array,
                    self._identity,
                    self.joints_identities(),
                    self.units,
                    enable_ft=True
                )
            })
        # Contacts
        if (
                self.options.morphology.links is not None
                and self.options.morphology.feet is not None
        ):
            self.sensors.add({
                'contacts': ContactsSensors(
                    self.data.sensors.contacts.array,
                    [
                        self._identity
                        for _ in self.options.morphology.feet
                    ],
                    [
                        self._links[foot]
                        for foot in self.options.morphology.feet
                    ],
                    self.units.newtons
                )
            })

    def set_body_properties(self, verbose=False):
        """Set body properties"""
        # Masses
        n_links = pybullet.getNumJoints(self.identity())+1
        self.masses = np.zeros(n_links)
        for i in range(n_links):
            self.masses[i] = pybullet.getDynamicsInfo(self.identity(), i-1)[0]
        if verbose:
            pylog.debug('Body mass: {} [kg]'.format(np.sum(self.masses)))
        # Deactivate collisions
        if self.options.morphology.links_no_collisions is not None:
            self.set_collisions(
                self.options.morphology.links_no_collisions,
                group=0,
                mask=0
            )
        # Deactivate damping
        small = 0
        self.set_links_dynamics(
            self._links.keys(),
            linearDamping=small,
            angularDamping=small,
            jointDamping=small
        )
        # Friction
        self.set_links_dynamics(
            self._links.keys(),
            lateralFriction=0.5,
            spinningFriction=small,
            rollingFriction=small,
        )
        if self.options.morphology.feet is not None:
            self.set_links_dynamics(
                self.options.morphology.feet,
                lateralFriction=0.9,
                spinningFriction=small,
                rollingFriction=small,
                # contactStiffness=1e3,
                # contactDamping=1e6
            )

    def drag_swimming_forces(self, iteration, water_surface, **kwargs):
        """Animat swimming physics"""
        drag_forces(
            iteration,
            self.data.sensors.gps,
            self.data.sensors.hydrodynamics.array,
            [
                link_i
                for link_i in range(self.options.morphology.n_links_body())
                if (
                    self.data.sensors.gps.com_position(iteration, link_i)[2]
                    < water_surface
                )
            ],
            masses=self.masses,
            **kwargs
        )

    def apply_swimming_forces(
            self, iteration, water_surface, link_frame=True, debug=False
    ):
        """Animat swimming physics"""
        links = self.options.morphology.links
        links_swimming = self.options.morphology.links_swimming
        swimming_motion(
            iteration,
            self.data.sensors.hydrodynamics.array,
            self.identity(),
            [
                [links.index(name), self._links[name]]
                for name in links_swimming
                if (
                    self.data.sensors.gps.com_position(
                        iteration,
                        links.index(name)
                    )[2] < water_surface
                )
            ],
            link_frame=link_frame,
            units=self.units
        )
        if debug:
            swimming_debug(
                iteration,
                self.data.sensors.gps,
                [
                    [links.index(name), self._links[name]]
                    for name in links_swimming
                ]
            )

    def draw_hydrodynamics(self, iteration, water_surface, margin=0.01):
        """Draw hydrodynamics forces"""
        gps = self.data.sensors.gps
        links = self.options.morphology.links
        for i, (line, name) in enumerate(zip(
                self.hydrodynamics,
                self.options.morphology.links_swimming
        )):
            if (
                    gps.com_position(iteration, links.index(name))[2]
                    < water_surface + margin
            ):
                force = self.data.sensors.hydrodynamics.array[iteration, i, :3]
                self.hydrodynamics[i] = pybullet.addUserDebugLine(
                    lineFromXYZ=[0, 0, 0],
                    lineToXYZ=np.array(force),
                    lineColorRGB=[0, 0, 1],
                    lineWidth=7*self.units.meters,
                    parentObjectUniqueId=self.identity(),
                    parentLinkIndex=i-1,
                    replaceItemUniqueId=line
                )


def simulation_setup(
        animat_sdf,
        arena,
        animat_options,
        simulation_options,
        network,
):
    """Simulation setup"""

    # Animat data
    animat_data = SalamandraData.from_options(
        np.arange(
            0,
            simulation_options.duration(),
            simulation_options.timestep,
        ),
        network.state,
        animat_options.morphology,
        simulation_options.n_iterations,
    )

    # Animat controller
    animat_controller = SalamandraController(
        joints=animat_options.morphology.joints,
        animat_data=animat_data,
        network=network,
    )

    # Creating animat
    animat = Salamandra(
        sdf=animat_sdf,
        options=animat_options,
        controller=animat_controller,
        timestep=simulation_options.timestep,
        iterations=simulation_options.n_iterations,
        units=simulation_options.units,
    )

    # Setup simulation
    pylog.info('Creating simulation')
    sim = SalamandraSimulation(
        simulation_options=simulation_options,
        animat=animat,
        arena=arena,
    )
    return sim, animat_data

