#!/usr/bin/env python3

import drl_grasping

import time
import enum
import numpy as np
import gym_ignition
from typing import List
from functools import partial
from gym_ignition.rbd import conversions
from scenario import core as scenario_core
from scenario import gazebo as scenario_gazebo
from scipy.spatial.transform import Rotation


# Configure verbosity
scenario_gazebo.set_verbosity(scenario_gazebo.Verbosity_error)
# Configure numpy output
np.set_printoptions(precision=4, suppress=True)


# Get the simulator and the world
gazebo = scenario_gazebo.GazeboSimulator(
    step_size=0.01, rtf=1.0, steps_per_run=50)

gazebo.insert_world_from_sdf(
    "/home/andrej/uni/repos/drl_grasping/drl_grasping/envs/worlds/training_grounds.sdf")

gazebo.initialize()
world = gazebo.get_world()


# Open the GUI
gazebo.gui()
gazebo.run(paused=True)

world = gazebo.get_world()


object1_pos = [0.0, 0.0, 1.25]
object1_quat = conversions.Quaternion.to_wxyz(
    xyzw=Rotation.from_euler('xyz', [0, 0, 0]).as_quat())
object1 = drl_grasping.envs.models.RandomObject(world=world,
                                               position=object1_pos,
                                               orientation=object1_quat)


# tf2_broadcaster = drl_grasping.utils.Tf2Broadcaster()

# # Insert the Panda manipulator
# robot_pos = [0, 0, 0]
# robot_quat = [1, 0, 0, 0]
# robot = drl_grasping.envs.models.Panda(world=world,
#                                        position=robot_pos,
#                                        orientation=robot_quat,
#                                        initial_joint_positions=[0, 0, 0, -1.57, 0, 1.57, 0.79, 0, 0])
# tf2_broadcaster.broadcast_tf(translation=robot_pos,
#                              rotation=robot_quat,
#                              child_frame_id="panda_link0")
# robot_name = robot.name()

# camera_pos = [0.25, -0.5, 0.25]
# camera_quat = conversions.Quaternion.to_wxyz(
#     xyzw=Rotation.from_euler('xyz', [0, 0.5236, 0.7854]).as_quat())
# camera = drl_grasping.envs.models.RealsenseD435(world=world,
#                                                 position=camera_pos,
#                                                 orientation=camera_quat)
# tf2_broadcaster.broadcast_tf(translation=camera_pos,
#                              rotation=camera_quat,
#                              child_frame_id="realsense_d435/d435/camera")
# camera_name = camera.name()




# moveit = drl_grasping.control.MoveIt2()
# ps_sub = drl_grasping.perception.PointCloudSub()

# i = 1
while True:
    gazebo.run(paused=False)
#     if i == 1:
#         moveit.gripper_close()
#         i = 0
#     else:
#         moveit.gripper_open()
#         i = 1
    # print(ps_sub.get_point_cloud())

gazebo.close()
