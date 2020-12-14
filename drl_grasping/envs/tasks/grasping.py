import abc
import gym
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple
from gym_ignition.base import task
from scenario import core as scenario
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace


from drl_grasping.control import MoveIt2
from drl_grasping.perception import PointCloudSub

# from gym_ignition.runtimes.gazebo_runtime import GazeboRuntime


class Grasping(task.Task, abc.ABC):

    def __init__(self,
                 agent_rate: float,
                 **kwargs):

        # Initialize the Task base class
        task.Task.__init__(self, agent_rate=agent_rate)

        # Task-specific interfaces with actuators and sensors
        # These interfaces should be independent of the used runtime
        self.moveit2 = MoveIt2()
        self.point_cloud_sub = PointCloudSub()

        # Other variables
        # TODO: set from somewhere that has access to the model itself
        self._number_of_joints = 9

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

        # Action space
        action_space = gym.spaces.Dict(self._get_action_space_dict())

        # Observation space
        observation_space = gym.spaces.Dict(self._get_observation_space_dict())

        return action_space, observation_space

    def set_action(self, action: Action) -> None:

        # Set position goal
        pos_xyz = action['pos_xyz']
        self.moveit2.set_position_goal(pos_xyz)

        # Normalize quaternions and use to as orientation goal
        quat_xyzw = Rotation.from_quat(action['quat_xyzw']).as_quat()
        self.moveit2.set_orientation_goal(quat_xyzw)

        # TODO: add gripper closing and opening
        # gripper_width = action['gripper_width]
        # gripper_force = action['gripper_force]
        # self.moveit2.gripper_close(width=0.08*gripper_width, force=20*gripper_force)

        self.moveit2.plan_kinematic_path(allowed_planning_time=0.1)
        self.moveit2.execute()

    def get_observation(self) -> Observation:

        joint_state = self.moveit2.get_joint_state()

        joint_positions = joint_state.position
        joint_velocities = joint_state.velocity

        ee_pose = self.moveit2.compute_fk(
            joint_state=joint_state).pose_stamped[0].pose
        ee_pos = [ee_pose.position.x,
                  ee_pose.position.y,
                  ee_pose.position.z]
        ee_quat = [ee_pose.orientation.x,
                   ee_pose.orientation.y,
                   ee_pose.orientation.z,
                   ee_pose.orientation.w]

        # Create the observation
        observation = Observation(np.array([joint_positions,
                                            joint_velocities,
                                            ee_pos,
                                            ee_quat,
                                            ]))

        # Return the observation
        return observation

    def get_reward(self) -> Reward:

        # TODO: reward
        reward = 1.0

        return Reward(reward)

    def is_done(self) -> bool:

        # TODO: done
        # observation = self.get_observation()

        done = False

        return done

    def reset_task(self) -> None:

        # TODO: reset

        pass

# Private methods

    def _get_action_space_dict(self) -> dict:

        actions: dict = {}

        # Grasp end-effector position (x, y, z)
        actions['pos_xyz'] = gym.spaces.Box(low=np.array((-0.5, -0.5, 0.25)),
                                            high=np.array((0.5, 0.5, 1.0)),
                                            # shape=(3,),
                                            dtype=np.float32)

        # Grasp end-effector orientation (x, y, z, w)
        actions['quat_xyzw'] = gym.spaces.Box(low=-1.0,
                                              high=1.0,
                                              shape=(4,),
                                              dtype=np.float32)

        # Gripper action
        actions['gripper_width'] = gym.spaces.Box(low=0.0,
                                                  high=1.0,
                                                  shape=(1,),
                                                  dtype=np.float32)
        actions['gripper_force'] = gym.spaces.Box(low=0.0,
                                                  high=1.0,
                                                  shape=(1,),
                                                  dtype=np.float32)

        return actions

    def _get_observation_space_dict(self) -> dict:

        observations: dict = {}
        inf = np.finfo(np.float32).max

        # Joint positions
        observations['joint_pos'] = gym.spaces.Box(low=-inf,
                                                   high=inf,
                                                   shape=(
                                                       self._number_of_joints,),
                                                   dtype=np.float32)

        # Joint velocities
        observations['joint_vel'] = gym.spaces.Box(low=-inf,
                                                   high=inf,
                                                   shape=(
                                                       self._number_of_joints,),
                                                   dtype=np.float32)

        # End-effector position (x, y, z)
        observations['pos_xyz'] = gym.spaces.Box(low=-1.0,
                                                 high=1.0,
                                                 shape=(3,),
                                                 dtype=np.float32)

        # End-effector orientation (x, y, z, w)
        observations['quat_xyzw'] = gym.spaces.Box(low=-1.0,
                                                   high=1.0,
                                                   shape=(4,),
                                                   dtype=np.float32)

        point_cloud = self.point_cloud_sub.get_point_cloud()

        # TODO: Perception for object to be grasped

        # TODO: Perception for obstable avoidance

        return observations
