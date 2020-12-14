from .grasping import Grasping

from gym.envs.registration import register

register(
    id='Grasping-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=3,
    kwargs={'task_cls': Grasping,
            'agent_rate': 2,
            'physics_rate': 200,
            'real_time_factor': 1.0,
            'world': '/home/andrej/uni/repos/drl_grasping/drl_grasping/envs/worlds/training_grounds.sdf'
            })
