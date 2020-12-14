#!/usr/bin/env python3

import drl_grasping

import gym
import time
import functools
from gym_ignition.utils import logger
from drl_grasping.envs.randomizers import GraspingGazeboEnvRandomizer


# Set verbosity
logger.set_level(gym.logger.WARN)
# logger.set_level(gym.logger.DEBUG)

# Available tasks
env_id = "Grasping-Gazebo-v0"


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import gym_ignition_environments
    return gym.make(env_id, **kwargs)


# Create a partial function passing the environment id
make_env = functools.partial(make_env_from_id, env_id=env_id)

# env = make_env_from_id(env_id=env_id)

# Wrap the environment with the randomizer.
# This is a simple example no randomization are applied.
env = GraspingGazeboEnvRandomizer(env=make_env, num_physics_rollouts=0)

# Wrap the environment with the randomizer.
# This is a complex example that randomizes both the physics and the model.
# env = randomizers.cartpole.CartpoleEnvRandomizer(
#     env=make_env, seed=42, num_physics_rollouts=5)

# Enable the rendering
# env.render('human')

# Initialize the seed
env.seed(0)

for epoch in range(10000):

    # Reset the environment
    observation = env.reset()

    # Initialize returned values
    done = False
    totalReward = 0

    while not done:

        # Execute a random action
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

        # Render the environment.
        # It is not required to call this in the loop if physics is not randomized.
        # env.render('human')

        # Accumulate the reward
        totalReward += reward

    # print(f"Reward episode #{epoch}: {totalReward}")

env.close()
time.sleep(5)
