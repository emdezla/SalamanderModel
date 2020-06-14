"""Test"""

import time
import sched
import pybullet
from tqdm import tqdm
import numpy as np


def create_floor():
    """Create floor"""
    return pybullet.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_PLANE,
            planeNormal=[0, 0, 1],
        ),
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
    )


class Robot:
    """Robot"""

    def __init__(self):
        sdf = 'salamandra_robotica.sdf'
        self.identity = pybullet.loadSDF(sdf)[0]
        pybullet.resetBasePositionAndOrientation(
            bodyUniqueId=self.identity,
            posObj=[0, 0, 0.1],
            ornObj=[0, 0, 0, 1],
        )

    def step(self):
        """Update"""
        self.update_sensors()
        self.control()

    def update_sensors(self):
        """Update sensors"""

    def control(self):
        """Control"""


def set_physics(timestep, gravity):
    """Set physics"""
    pybullet.setTimeStep(timestep)
    pybullet.setGravity(*gravity)


class Simulation:
    """Simulation"""

    def __init__(self, duration):
        super(Simulation, self).__init__()
        # Simulation
        self.iteration = 0
        self.timestep = 1e-3
        self.n_iterations = np.rint(duration/self.timestep).astype(int)
        self.rtf = 1.0  # np.inf
        self.timer = None
        self.initial_time = None
        # Create pybullet physics
        pybullet.connect(pybullet.GUI)
        # Create floor
        self.floor = create_floor()
        # Create robot
        self.robot = Robot()
        # Physics
        set_physics(
            timestep=self.timestep,
            gravity=[0, 0, -9.81]
        )
        # Scheduler
        self.scheduler = sched.scheduler(time.time, time.sleep)

    def run(self):
        """Run"""
        for iteration in range(self.n_iterations):
            self.scheduler.enter(
                delay=self.timestep*iteration/self.rtf,
                priority=1,
                action=self.step,
                argument=(iteration,)
            )
        with tqdm(
                total=self.n_iterations,
        ) as self.timer:
            self.initial_time = time.time()
            self.scheduler.run()
        print("Total simulation time: {} [s]".format(
            time.time() - self.initial_time
        ))

    def step(self, iteration, verbose=False):
        """Step"""
        assert iteration == self.iteration, (
            'Error at timestep {} != {}'.format(self.timestep)
        )
        if verbose:
            print(
                "Iteration {}/{} at time {:3f} [s] (Expected {:3f} [s])".format(
                    self.iteration,
                    self.n_iterations,
                    time.time() -
                    self.initial_time,
                    self.iteration *
                    self.timestep,
                ))
        self.robot.step()
        pybullet.stepSimulation()
        self.iteration += 1
        self.timer.update()

    @staticmethod
    def end():
        """End simulation"""
        pybullet.disconnect()


def main():
    """Main"""
    simulation = Simulation(duration=30)
    simulation.run()
    simulation.end()


if __name__ == '__main__':
    main()

