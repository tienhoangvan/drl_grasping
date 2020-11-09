#!/usr/bin/env python3

from stable_baselines3 import HER, DDPG
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import CheckpointCallback

from env import PandaPushEnv

import rclpy
import os


def main(args=None):
    rclpy.init(args=args)

    training_steps = 8000000

    training_dir = "./training/ddpg_her_push/"
    model_name = "model0"
    model_path = os.path.join(training_dir, model_name)
    tensorboard_path = os.path.join(training_dir, "tensorboard")

    # Create Ignition environment
    max_episode_length = 50
    env = PandaPushEnv(n_actions=3, sim_steps=40, distance_threshold=0.07, reward_type='sparse', position_change_multiplier=0.05,
                       obj_range=0.15, target_range=0.15, max_episode_length=max_episode_length, sim_step_size=0.001)

    # Initialize or load the model
    model = HER(policy="MlpPolicy",
                env=env,
                model_class=DDPG,
                n_sampled_goal=4,
                goal_selection_strategy=GoalSelectionStrategy.FUTURE,
                online_sampling=True,
                buffer_size=200000,
                batch_size=256,
                learning_starts=100000,
                learning_rate=0.001,
                gamma=0.95,
                max_episode_length=max_episode_length,
                tensorboard_log=tensorboard_path,
                verbose=1)
    # model = HER.load(path=os.path.join(training_dir, "checkpoint00", "model0_XXXXX_steps"),
    #                  env=env,
    #                  tensorboard_log=tensorboard_path,
    #                  verbose=1)

    # Train the model (save regularly)
    checkpoint_callback = CheckpointCallback(save_freq=50*max_episode_length,
                                             save_path=os.path.join(
                                                 training_dir, "checkpoint01"),
                                             name_prefix=model_name,
                                             verbose=1)
    model.learn(total_timesteps=training_steps,
                callback=checkpoint_callback,
                log_interval=10)

    model.save(model_path)

    test_trained_model = True
    if test_trained_model:
        episodes = 1000
        episode_reward = 0
        succeded = 0
        episode = 0
        obs = env.reset()
        while episode < episodes:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done or info.get('is_success', False):
                print("Reward:", episode_reward,
                      "Success?", info.get('is_success', False))
                if info.get('is_success', False):
                    succeded += 1
                episode += 1
                episode_reward = 0
                obs = env.reset()
    print("Success rate: ", succeded/episodes)


if __name__ == "__main__":
    main()
