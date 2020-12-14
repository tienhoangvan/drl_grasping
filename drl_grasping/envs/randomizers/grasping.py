import abc
from typing import Union, List
from gym_ignition import utils
from gym_ignition.utils import misc
from gym_ignition import randomizers
from scenario import gazebo as scenario
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition.randomizers.model.sdf import Method, Distribution, UniformParams
from gym_ignition.rbd import conversions
from scipy.spatial.transform import Rotation

from drl_grasping.envs import tasks, models
from drl_grasping.utils import Tf2Broadcaster

import numpy as np

# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[tasks.Grasping]


class GraspingGazeboEnvRandomizerImpl(randomizers.abc.TaskRandomizer,
                                      randomizers.abc.PhysicsRandomizer,
                                      randomizers.abc.ModelDescriptionRandomizer,
                                      abc.ABC):
    """
    Mixin that collects the implementation of task, model and physics randomizations for
    grasping environments.
    """

    def __init__(self, num_physics_rollouts: int = 0):

        # Initialize base classes
        randomizers.abc.TaskRandomizer.__init__(self)
        randomizers.abc.PhysicsRandomizer.__init__(
            self, randomize_after_rollouts_num=num_physics_rollouts)
        randomizers.abc.ModelDescriptionRandomizer.__init__(self)

        self._tf2_broadcaster = Tf2Broadcaster()

        self._panda_name = None
        self._camera_name = None
        self._is_populated = False

        self._object_names = []
        self._plane_name = None

        self._plane_change_counter = 0

        # SDF randomizer
        # self._sdf_randomizer = None

    # ===========================
    # PhysicsRandomizer interface
    # ===========================

    def get_engine(self):

        return scenario.PhysicsEngine_dart

    def randomize_physics(self, task: SupportedTasks, **kwargs) -> None:

        gravity_z = task.np_random.normal(loc=-9.8, scale=0.2)

        if not task.world.to_gazebo().set_gravity((0, 0, gravity_z)):
            raise RuntimeError("Failed to set the gravity")

    # ========================
    # TaskRandomizer interface
    # ========================

    def randomize_task(self, task: SupportedTasks, **kwargs) -> None:

        # Get gazebo instance associated with the task
        if "gazebo" not in kwargs:
            raise ValueError("gazebo kwarg not passed to the task randomizer")
        gazebo = kwargs["gazebo"]

        # Generate a random model description
        # random_model = self.randomize_model_description(task=task)

        # Insert a new model in the world
        if not self._is_populated:
            self._populate_world(task=task)
            self._is_populated = True
        else:
            self._randomize_camera_pose(task=task)

        self._reset_random_objects(task=task,
                                   gazebo=gazebo,
                                   number_of_objects=1,
                                   spawn_centre=(1.0, 0.0),
                                   spawn_width=0.5,
                                   height=0.2)

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    # ====================================
    # ModelDescriptionRandomizer interface
    # ====================================

    def randomize_model_description(self, task: SupportedTasks, **kwargs) -> str:

        pass

        # randomizer = self._get_sdf_randomizer(task=task)
        # sdf = misc.string_to_file(randomizer.sample())
        # return sdf

    # ===============
    # Private Methods
    # ===============

    def _get_sdf_randomizer(self, task: SupportedTasks) -> \
            randomizers.model.sdf.SDFRandomizer:

        pass

        # if self._sdf_randomizer is not None:
        #     return self._sdf_randomizer

        # # Get the model file
        # urdf_model_file = grasping.grasping.get_model_file()

        # # Convert the URDF to SDF
        # sdf_model_string = scenario.urdffile_to_sdfstring(urdf_model_file)

        # # Write the SDF string to a temp file
        # sdf_model = utils.misc.string_to_file(sdf_model_string)

        # # Create and initialize the randomizer
        # sdf_randomizer = randomizers.model.sdf.SDFRandomizer(
        #     sdf_model=sdf_model)

        # # Use the RNG of the task
        # sdf_randomizer.rng = task.np_random

        # # Randomize the mass of all links
        # sdf_randomizer.new_randomization() \
        #     .at_xpath("*/link/inertial/mass") \
        #     .method(Method.Additive) \
        #     .sampled_from(Distribution.Uniform, UniformParams(low=-0.2, high=0.2)) \
        #     .force_positive() \
        #     .add()

        # # Process the randomization
        # sdf_randomizer.process_data()
        # assert len(sdf_randomizer.get_active_randomizations()) > 0

        # # Store and return the randomizer
        # self._sdf_randomizer = sdf_randomizer
        # return self._sdf_randomizer

    def _populate_world(self, task: SupportedTasks, model: str = None) -> None:

        robot_pos = [0, 0, 0]
        robot_quat = [1, 0, 0, 0]
        robot = models.Panda(world=task.world,
                             position=robot_pos,
                             orientation=robot_quat,
                             initial_joint_positions=[0, 0, 0, -1.57, 0, 1.57, 0.79, 0, 0])
        self._tf2_broadcaster.broadcast_tf(translation=robot_pos,
                                           rotation=robot_quat,
                                           child_frame_id="panda_link0")
        self._robot_name = robot.name()

        camera_pos, camera_rpy = self._random_camera_pose(centre=[1.0, 0.0, 0.0],
                                                          distance=0.5)

        camera_quat = conversions.Quaternion.to_wxyz(
            xyzw=Rotation.from_euler('xyz', camera_rpy).as_quat())

        camera = models.RealsenseD435(world=task.world,
                                      position=camera_pos,
                                      orientation=camera_quat)
        self._tf2_broadcaster.broadcast_tf(translation=camera_pos,
                                           rotation=camera_quat,
                                           child_frame_id="realsense_d435/d435/camera")
        self._camera_name = camera.name()

    def _randomize_camera_pose(self, task: SupportedTasks, model: str = None) -> None:

        camera_pos, camera_rpy = self._random_camera_pose(centre=[1.0, 0.0, 0.0],
                                                          distance=0.5)

        camera_quat = conversions.Quaternion.to_wxyz(
            xyzw=Rotation.from_euler('xyz', camera_rpy).as_quat())

        camera = task.world.to_gazebo().get_model(self._camera_name)
        camera.to_gazebo().reset_base_pose(camera_pos, camera_quat)

        self._tf2_broadcaster.broadcast_tf(translation=camera_pos,
                                           rotation=camera_quat,
                                           child_frame_id="realsense_d435/d435/camera")

    def _random_camera_pose(self, centre=[0.0, 0.0, 0.0], distance=1.0, camera_height=(0.5, 1.0)) -> ([float], (float)):
        # Range [0;pi] [-pi;pi]
        theta = np.random.uniform(0.0, 1.0) * np.pi
        phi = np.random.uniform(-1.0, 1.0) * np.pi

        # Switch to cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.random.uniform(camera_height[0], camera_height[1])

        pitch = np.arctan2(z, np.sqrt(x**2+y**2))
        yaw = np.arctan2(y, x) + np.pi

        x = x*distance + centre[0]
        y = y*distance + centre[1]
        z = z*distance + centre[2]

        return [x, y, z], [0.0, pitch, yaw]

    def _reset_random_objects(self,
                              task: SupportedTasks,
                              gazebo,
                              number_of_objects=1,
                              spawn_centre=(0.0, 0.0),
                              spawn_width=1.0,
                              height=1.0,
                              plane_reset_frequency=100000) -> None:


        if self._plane_change_counter % plane_reset_frequency == 0:
            if self._plane_name is not None:
                if not task.world.to_gazebo().remove_model(self._plane_name):
                    print("Failed to remove plane from the world")
                if not gazebo.run(paused=True):
                    raise RuntimeError("Failed to execute a paused Gazebo run")

            plane = models.RandomPlanePBR(world=task.world,
                                          position=(0, 0, 0),
                                          orientation=(1, 0, 0, 0),
                                          size=(3.0, 3.0),
                                          texture_dir='/home/andrej/Pictures/pbr_textures')
            self._plane_name = plane.name()

            self._plane_change_counter = 0
    
        self._plane_change_counter += 1

        for model_name in self._object_names:
            if not task.world.to_gazebo().remove_model(model_name):
                print("Failed to remove object from the world")

        if not gazebo.run(paused=False):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        self._object_names.clear()

        for _ in range(number_of_objects):
            x = np.random.uniform(-spawn_width/2,
                                  spawn_width/2) + spawn_centre[0]
            y = np.random.uniform(-spawn_width/2,
                                  spawn_width/2) + spawn_centre[1]
            z = height
            roll = np.random.uniform(-1, 1) * np.pi
            pitch = np.random.uniform(-1, 1) * np.pi
            yaw = np.random.uniform(-1, 1) * np.pi

            object_quat = conversions.Quaternion.to_wxyz(
                xyzw=Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat())
            try:
                model = models.RandomObject(world=task.world,
                                            position=(x, y, z),
                                            orientation=object_quat)
                self._object_names.append(model.name())
            except:
                # TODO: figure out what to do when the object cannot be inserted into the world
                pass

        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")


class GraspingGazeboEnvRandomizer(gazebo_env_randomizer.GazeboEnvRandomizer,
                                  GraspingGazeboEnvRandomizerImpl):
    """
    Concrete implementation of grasping environments randomization.
    """

    def __init__(self,
                 env: MakeEnvCallable,
                 num_physics_rollouts: int = 0):

        # Initialize the mixin
        GraspingGazeboEnvRandomizerImpl.__init__(
            self, num_physics_rollouts=num_physics_rollouts)

        # Initialize the environment randomizer
        gazebo_env_randomizer.GazeboEnvRandomizer.__init__(
            self, env=env, physics_randomizer=self)
