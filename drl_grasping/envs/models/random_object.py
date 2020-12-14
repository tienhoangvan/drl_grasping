

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


import glob
import re

import trimesh

from shutil import copyfile

from drl_grasping.utils.model_collection_randomizer import ModelCollectionRandomizer

# TODO: set RNG seed somehow

# mesh = trimesh.load('model.obj', force='mesh')
# collision_mesh = mesh.simplify_quadratic_decimation(len(mesh.faces)/20)


class RandomObject(model_wrapper.ModelWrapper):

    _model_paths = None

    def __init__(self,
                 world: scenario.World,
                 position: List[float] = (0, 0, 0),
                 orientation: List[float] = (1, 0, 0, 0),
                 model_paths=None,
                 owner='GoogleResearch',
                 collection='Google Scanned Objects',
                 server='https://fuel.ignitionrobotics.org',
                 server_version='1.0',
                 new_collection=False):
        # Get a unique model name
        model_name = get_unique_model_name(world, 'object')

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)
        
        model_collection_randomizer = ModelCollectionRandomizer()

        modified_sdf_file = model_collection_randomizer.random_model()

        # Insert the model
        ok_model = world.to_gazebo().insert_model(modified_sdf_file,
                                                  initial_pose,
                                                  model_name)
        if not ok_model:
            raise RuntimeError('Failed to insert ' + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        model_wrapper.ModelWrapper.__init__(self, model=model)
