from typing import List
from threading import Thread

from scenario import core as scenario
from scenario import gazebo as scenario_gazebo
from gym_ignition.utils.scenario import get_unique_model_name
from gym_ignition.scenario import model_wrapper, model_with_file


class Panda(model_wrapper.ModelWrapper,
            model_with_file.ModelWithFile):

    def __init__(self,
                 world: scenario.World,
                 position: List[float] = [0, 0, 0],
                 orientation: List[float] = [1, 0, 0, 0],
                 model_file: str = None,
                 initial_joint_positions: List[float] =
                 [0, 0, 0, -1.57, 0, 1.57, 0.79, 0, 0]):
        # Get a unique model name
        model_name = get_unique_model_name(world, "panda")

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Get the default model description (URDF or SDF) allowing to pass a custom model
        if model_file is None:
            model_file = self.get_model_file(False)

        # Insert the model
        ok_model = world.to_gazebo().insert_model(model_file,
                                                  initial_pose,
                                                  model_name)
        if not ok_model:
            raise RuntimeError("Failed to insert " + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Set initial joint configuration
        self.__set_initial_joint_positions(initial_joint_positions)
        model.to_gazebo().reset_joint_positions(
            self.get_initial_joint_positions(),
            self.get_joint_names())

        # Add JointStatePublisher to Panda
        self.__add_joint_state_publisher(model)

        # Add JointTrajectoryController to Panda
        self.__add_joint_trajectory_controller(model)

        # Initialize base class
        super().__init__(model=model)

    @classmethod
    def get_model_file(self, fuel=True) -> str:
        if fuel:
            return scenario_gazebo.get_model_file_from_fuel(
                "https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/panda")
        else:
            return "panda"

    @classmethod
    def get_joint_names(self) -> [str]:
        return ["panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_finger_joint1",
                "panda_finger_joint2"]

    def get_initial_joint_positions(self) -> [float]:
        return self.__initial_joint_positions

    def __set_initial_joint_positions(self, initial_joint_positions):
        self.__initial_joint_positions = initial_joint_positions

    def __add_joint_state_publisher(self, model) -> bool:
        """Add JointTrajectoryController"""
        model.to_gazebo().insert_model_plugin(
            "libignition-gazebo-joint-state-publisher-system.so",
            "ignition::gazebo::systems::JointStatePublisher",
            self.__get_joint_state_publisher_config()
        )

    @classmethod
    def __get_joint_state_publisher_config(self) -> str:
        return \
            """
            <sdf version="1.7">
            %s
            </sdf>
            """ \
            % " ".join(("<joint_name>" + joint + "</joint_name>" for joint in self.get_joint_names()))

    def __add_joint_trajectory_controller(self, model) -> bool:
        """Add JointTrajectoryController"""
        model.to_gazebo().insert_model_plugin(
            "libignition-gazebo-joint-trajectory-controller-system.so",
            "ignition::gazebo::systems::JointTrajectoryController",
            self.__get_joint_trajectory_controller_config()
        )

    def __get_joint_trajectory_controller_config(self) -> str:
        return \
            """
            <sdf version="1.7">
            <topic>joint_trajectory</topic>

            <joint_names>%s</joint_names>
            <initial_positions>%s</initial_positions>

            <position_p_gain>  3000  9500  6500  6000  2750 2500  2000  250  250</position_p_gain>
            <position_d_gain>  15    47.5  32.5  30    2.75 2.5   2     0.2  0.2</position_d_gain>
            <position_i_gain>  1650  5225  3575  3300  1515 1375  1100  50   50 </position_i_gain>
            <position_i_min>  -15   -47.5 -32.5 -30   -6.88 -6.25 -5   -10  -10 </position_i_min>
            <position_i_max>   15    47.5  32.5  30    6.88  6.25  5    10   10 </position_i_max>
            <position_cmd_min>-87   -87   -87   -87   -12   -12   -12  -20  -20 </position_cmd_min>
            <position_cmd_max> 87    87    87    87    12    12    12   20   20 </position_cmd_max>
            </sdf>
            """ % \
            (" ".join(self.get_joint_names()),
             " ".join(map(str, self.get_initial_joint_positions())))
