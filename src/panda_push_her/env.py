import os
import copy
import threading

import numpy as np

import gym
from gym.utils import seeding
from gym import utils, error, spaces, GoalEnv

import rclpy
from rclpy.node import Node

from moveit2 import MoveIt2Interface
from ign_py import IgnitionInterface

from geometry_msgs.msg import Pose, Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from scipy.spatial.transform import Rotation


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PandaPushEnv(GoalEnv, Node):
    def __init__(
            self, n_actions=3, sim_steps=40, distance_threshold=0.07, reward_type='sparse', position_change_multiplier=0.05, obj_range=0.15, target_range=0.15, max_episode_length=50, sim_step_size=0.001):
        """Initializes a new Panda environment.
        Args:
            n_actions (int): number of actions
            sim_steps (int): number of substeps the simulation runs on every call to step
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        GoalEnv.__init__(self)
        Node.__init__(self, "panda_push_env")
        self.n_actions = n_actions
        self.sim_steps = sim_steps
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.position_change_multiplier = position_change_multiplier
        self.obj_range = obj_range
        self.target_range = target_range
        self.max_episode_length_ = max_episode_length
        self.sim_steps_ = sim_steps
        self.sim_step_size_ = sim_step_size
        self.previous_ee_position_ = np.array([0.0, 0.0, 0.0])
        self.previous_object_position_ = np.array([0.0, 0.0, 0.0])

        self.episode_iteration_ = 0
        self.reward_sum_ = 0

        # Pose of box
        self.box_pose_ = Pose()
        self.box_pose_mutex_ = threading.Lock()
        self.box_pose_sub_ = self.create_subscription(Pose, '/model/box/pose',
                                                      self.box_pose_callback, 1)

        self.ign_py_ = IgnitionInterface()

        self.moveit2_ = MoveIt2Interface()

        self.executor_ = rclpy.executors.MultiThreadedExecutor(2)
        self.executor_.add_node(self)
        self.executor_.add_node(self.moveit2_)

        thread = threading.Thread(target=self.executor_.spin, args=())
        thread.daemon = True  # Daemonize thread
        thread.start()       # Start the execution

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
        }

        # Init
        self.seed()
        self.goal = self._sample_goal()

        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1.,
                                       shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf,
                                    shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf,
                                     shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf,
                                   shape=obs['observation'].shape, dtype='float32'),
        ))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Take action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        # Step the simulation
        is_stepped = False
        while not is_stepped:
            is_stepped = self.ign_py_.step(self.sim_steps_)
        self.episode_iteration_ = self.episode_iteration_ + 1

        obs = self._get_obs()

        is_success = self._is_success(obs['achieved_goal'], self.goal)

        if is_success or self.episode_iteration_ >= self.max_episode_length_:
            done = True
        else:
            done = False

        info = {
            'is_success': is_success,
        }

        info = {
            'is_success': is_success,
            'episode_reward': self.reward_sum_,
        }

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        self.reward_sum_ += reward

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        position_control_relative = action[:3]
        # limit maximum change in position
        position_control_relative *= self.position_change_multiplier

        current_position = self.moveit2_.compute_fk(
        ).pose_stamped[0].pose.position
        quaternion = Rotation.from_euler('xyz',
                                         [180, 0, 0],
                                         degrees=True).as_quat()

        target_pose = Pose()
        target_pose.position.x = current_position.x + \
            position_control_relative[0]
        target_pose.position.y = current_position.y + \
            position_control_relative[1]
        target_pose.position.z = current_position.z + \
            position_control_relative[2]

        # Do not go to the opposite side
        target_pose.position.x = max(target_pose.position.x, 0.15)

        # Do not go too far to either side
        target_pose.position.y = max(min(target_pose.position.y, 0.3), -0.3)

        # Do not go below ground
        target_pose.position.z = min(max(target_pose.position.z, 0.0), 0.5)

        target_pose.orientation.x = quaternion[0]
        target_pose.orientation.y = quaternion[1]
        target_pose.orientation.z = quaternion[2]
        target_pose.orientation.w = quaternion[3]

        joint_state = self.moveit2_.compute_ik(
            target_pose).solution.joint_state
        self.moveit2_.move_to_joint_state(joint_state,
                                          set_position=True,
                                          set_velocity=False,
                                          set_effort=False)

    def _get_obs(self):
        # Get positions and convert go np array
        target_pos = self.goal
        object_pos = self.get_box_pose().position
        ee_pos = self.moveit2_.compute_fk().pose_stamped[0].pose.position
        target_position = np.array(target_pos)
        object_position = np.array([object_pos.x, object_pos.y, object_pos.z])
        ee_position = np.array([ee_pos.x, ee_pos.y, ee_pos.z])

        # Compute velocity of end effector and object
        dt = float(self.sim_steps_)*self.sim_step_size_
        ee_velocity = np.array([0.0, 0.0, 0.0])
        if self.previous_ee_position_.sum() != 0.0:
            ee_velocity = (ee_position - self.previous_ee_position_) / dt
        object_velocity = np.array([0.0, 0.0, 0.0])
        if self.previous_object_position_.sum() != 0.0:
            object_velocity = (object_position -
                               self.previous_object_position_) / dt
        self.previous_ee_position_ = ee_position
        self.previous_object_position_ = object_position

        # Get object position and velocity relative to end effector
        object_position_relative = object_position - ee_position
        object_velocity_relative = object_velocity - ee_velocity

        # Get target position relative to the object
        target_position_relative = target_position - object_position

        achieved_goal = np.squeeze(object_position.copy())
        desired_goal = target_position.copy()

        obs = np.concatenate([
            ee_position.ravel(),
            object_position.ravel(),
            object_position_relative.ravel(),
            object_velocity_relative.ravel(),
            target_position_relative.ravel()
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy(),
        }

    def reset(self):
        super(PandaPushEnv, self).reset()

        # Reset sim
        self.ign_py_.remove_markers()

        self.ign_py_.set_box_position(-100.0, -100.0, 0.025)

        home_pose = Pose()
        home_pose.position.x = 0.6
        home_pose.position.y = 0.0
        home_pose.position.z = 0.2
        quaternion = Rotation.from_euler('xyz',
                                         [180, 0, 0],
                                         degrees=True).as_quat()
        home_pose.orientation.x = quaternion[0]
        home_pose.orientation.y = quaternion[1]
        home_pose.orientation.z = quaternion[2]
        home_pose.orientation.w = quaternion[3]

        joint_state = self.moveit2_.compute_ik(
            home_pose).solution.joint_state
        self.moveit2_.move_to_joint_state(joint_state,
                                          set_position=True,
                                          set_velocity=False,
                                          set_effort=False)

        is_home = False
        i = 0
        while not is_home:
            self.ign_py_.step(100)
            i += 1
            ee_position = self.moveit2_.compute_fk(
            ).pose_stamped[0].pose.position
            if i > 100 or (abs(ee_position.x - home_pose.position.x) < 0.05 and abs(ee_position.y - home_pose.position.y) < 0.05 and abs(ee_position.z - home_pose.position.z) < 0.05):
                is_home = True

        # Randomly set position of the object
        ee_position_xy = np.array([home_pose.position.x, home_pose.position.y])
        object_xy_pos = ee_position_xy

        while np.linalg.norm(object_xy_pos - ee_position_xy) < 0.1:
            object_xy_pos = ee_position_xy + \
                self.np_random.uniform(-self.obj_range,
                                       self.obj_range, size=2)

        self.ign_py_.set_box_position(
            object_xy_pos[0], object_xy_pos[1], 0.025)
        self.episode_iteration_ = 0
        self.reward_sum_ = 0

        # Sample a new goal
        is_success = True
        while is_success:
            goal = ee_position_xy + \
                self.np_random.uniform(-self.target_range,
                                       self.target_range, size=2)
            is_success = self._is_success(object_xy_pos, goal)
            self.goal = np.array([goal[0], goal[1], 0.025])

        self.ign_py_.add_marker(self.goal[0], self.goal[1], self.goal[2])

        self.previous_ee_position_ = np.array([0.0, 0.0, 0.0])
        self.previous_object_position_ = np.array([0.0, 0.0, 0.0])

        # Get initial observation
        obs = self._get_obs()
        return obs

    def _sample_goal(self):
        ee_position = self.moveit2_.compute_fk().pose_stamped[0].pose.position
        object_position = self.get_box_pose().position
        object_position_np = np.array([object_position.x, object_position.y])

        is_success = True
        while is_success:
            goal = np.array([ee_position.x, ee_position.y]) + \
                self.np_random.uniform(-self.target_range,
                                       self.target_range, size=2)
            is_success = self._is_success(object_position_np, goal)

        self.ign_py_.add_marker(goal[0], goal[1], 0.025)

        return np.array([goal[0], goal[1], 0.025])

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def close(self):
        pass

    def box_pose_callback(self, msg):
        self.box_pose_mutex_.acquire()
        self.box_pose_ = msg
        self.box_pose_mutex_.release()

    def get_box_pose(self) -> Pose:
        self.box_pose_mutex_.acquire()
        box_pose = self.box_pose_
        self.box_pose_mutex_.release()
        return box_pose
