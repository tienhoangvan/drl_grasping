

from typing import List
from threading import Thread, Lock
from networkx.algorithms.cuts import volume

import numpy as np
import random

import os

from pcg_gazebo.parsers import parse_sdf
from pcg_gazebo.parsers.sdf import create_sdf_element
from pcg_gazebo.parsers.types import XMLBase


from scenario import core as scenario
from scenario import gazebo as scenario_gazebo
from gym_ignition.scenario import model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name
from gym_ignition.utils import logger
from gym_ignition.utils import misc

import glob
import re

import trimesh

from shutil import copyfile

from drl_grasping.utils.model_collection_randomizer import ModelCollectionRandomizer

# TODO: set RNG seed somehow

# mesh = trimesh.load('model.obj', force='mesh')
# collision_mesh = mesh.simplify_quadratic_decimation(len(mesh.faces)/20)


class RandomPlanePBR(model_wrapper.ModelWrapper):

    def __init__(self,
                 world: scenario.World,
                 position: List[float] = (0, 0, 0),
                 orientation: List[float] = (1, 0, 0, 0),
                 size: List[float] = (1.0, 1.0),
                 texture_dir='/home/andrej/Pictures/pbr_textures'):
        # Get a unique model name
        model_name = get_unique_model_name(world, 'plane')

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get list of the available textures
        textures = os.listdir(texture_dir)

        # Choose a random texture from these
        random_texture_dir = os.path.join(texture_dir,
                                          np.random.choice(textures))

        # List all files
        texture_files = os.listdir(random_texture_dir)

        # Extract the appropriate files
        albedo_map = None
        normal_map = None
        roughness_map = None
        metalness_map = None
        for texture in texture_files:
            texture_lower = texture.lower()
            if 'basecolor' in texture_lower or 'albedo' in texture_lower:
                albedo_map = os.path.join(random_texture_dir, texture)
            elif 'normal' in texture_lower:
                normal_map = os.path.join(random_texture_dir, texture)
            elif 'roughness' in texture_lower:
                roughness_map = os.path.join(random_texture_dir, texture)
            elif 'specular' in texture_lower or 'metalness' in texture_lower:
                metalness_map = os.path.join(random_texture_dir, texture)

        # Create SDF string for the model
        sdf = \
            """<sdf version="1.7">
            <model name="%s">
                <static>true</static>
                <link name="%s_link">
                    <collision name="%s_collision">
                        <geometry>
                            <plane>
                                <normal>0 0 1</normal>
                            </plane>
                        </geometry>
                    </collision>
                    <visual name="%s_visual">
                        <geometry>
                            <plane>
                                <normal>0 0 1</normal>
                                <size>%f %f</size>
                            </plane>
                        </geometry>
                        <material>
                            <ambient>1 1 1 1</ambient>
                            <diffuse>1 1 1 1</diffuse>
                            <specular>1 1 1 1</specular>
                            <pbr>
                                <metal>
                                    %s
                                    %s
                                    %s
                                    %s
                                </metal>
                            </pbr>
                        </material>
                    </visual>
                </link>
            </model>
        </sdf>""" % (model_name, model_name, model_name, model_name,
                    size[0],
                    size[1],
                    '<albedo_map>%s</albedo_map>' % albedo_map if albedo_map is not None else '',
                    '<normal_map>%s</normal_map>' % normal_map if normal_map is not None else '',
                    '<roughness_map>%s</roughness_map>' % roughness_map if roughness_map is not None else '',
                    '<metalness_map>%s</metalness_map>' % metalness_map if metalness_map is not None else '')

        # Convert it into a file
        sdf_file = misc.string_to_file(sdf)

        # Insert the model
        ok_model = world.to_gazebo().insert_model(sdf_file,
                                                  initial_pose,
                                                  model_name)
        if not ok_model:
            raise RuntimeError('Failed to insert ' + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        model_wrapper.ModelWrapper.__init__(self, model=model)
