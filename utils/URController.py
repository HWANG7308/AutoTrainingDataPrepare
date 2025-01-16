import URBasic
import math3d as m3d
import math
import numpy as np
import time


class UR5RobotController:
    def __init__(
        self, robot_ip, acceleration=0.9, velocity=0.8, init_position_index=None
    ):
        """
        Initialize the UR5 robot controller.

        Parameters:
        robot_ip (str): The IP address of the robot.
        acceleration (float): Robot acceleration value.
        velocity (float): Robot speed value.
        """
        self.robot_model = URBasic.robotModel.RobotModel()
        self.robot = None
        self.robot_ip = robot_ip
        self.acc = acceleration
        self.vel = velocity

        self.home_position = [
            math.radians(0),
            math.radians(-90),
            math.radians(0),
            math.radians(-90),
            math.radians(0),
            math.radians(0),
        ]

        self.init_positions = {
            1: [
                math.radians(0),
                math.radians(-25),
                math.radians(-155),
                math.radians(0),
                math.radians(0),
                math.radians(0),
            ],
            2: [
                math.radians(149.91),
                math.radians(-83.37),
                math.radians(152.57),
                math.radians(-69.2),
                math.radians(59.91),
                math.radians(180),
            ],
            3: [
                math.radians(-90),
                math.radians(-90),
                math.radians(-90),
                math.radians(-90),
                math.radians(90),
                math.radians(0),
            ],
        }

        self.init_position = self.init_positions.get(
            init_position_index, self.home_position
        )
        self.init_robot()

    def init_robot(self):
        print("Initializing robot...")
        self.robot = URBasic.urScriptExt.UrScriptExt(
            host=self.robot_ip, robotModel=self.robot_model
        )
        self.robot.reset_error()
        self.robot.movej(q=self.init_position, a=self.acc, v=self.vel)
        time.sleep(0.5)
        self.robot.init_realtime_control()
        time.sleep(0.5)
        print("Robot initialized!")

    def set_lookorigin(self):
        """
        Creates a new coordinate system at the current robot TCP position.

        Returns:
        orig (math3d.Transform): The new coordinate system in reference to the base coordinate system.
        """
        position = self.robot.get_actual_tcp_pose()
        return m3d.Transform(position)

    def get_joints(self, type="deg"):
        """
        Get the current joint positions.

        Parameters:
        type (str): The unit type for joint positions ('deg' or 'rad').

        Returns:
        np.ndarray: The current joint positions.
        """
        current_joints = self.robot.get_actual_joint_positions()
        if type == "deg":
            return np.degrees(current_joints)
        elif type == "rad":
            return current_joints
        else:
            print("get_joints: Invalid type. Use 'deg' or 'rad'.")
            return -1

    def is_moving(self):
        raise NotImplementedError

    def is_home(self, eps=0.02):
        """
        Check if the robot is at the home position.

        Parameters:
        eps (float): Tolerance for checking the home position.

        Returns:
        bool: True if the robot is at the home position, False otherwise.
        """
        return np.all(np.abs(np.array(self.home_position) - self.get_joints()) < eps)

    def at_target(self, target, type="deg", eps=0.02):
        """
        Check if the robot is at the target position.

        Parameters:
        target (list): The target joint positions.
        type (str): The unit type for joint positions ('deg' or 'rad').
        eps (float): Tolerance for checking the target position.

        Returns:
        bool: True if the robot is at the target position, False otherwise.
        """
        return np.all(np.abs(np.array(target) - self.get_joints(type=type)) < eps)

    def move_robot(
        self, type="j", pose=None, joint=None, check_joint=False, return_joint=False
    ):
        """
        Move the robot to the specified pose.

        Parameters:
        type (str): The movement type ('j' for joint, 'l' for linear, 'p' for path).
        pose (list): The target pose.
        joint (list): The target joint positions.
        check_joint (bool): Whether to check joint limits before moving.
        return_joint (bool): Whether to return the joint positions after moving.

        Returns:
        int or np.ndarray: 1 if successful, or the joint positions if return_joint is True.
        """
        if check_joint:
            self.check_joint_limit()

        move_func = {
            "j": self.robot.movej,
            "l": self.robot.movel,
            "p": self.robot.movep,
        }.get(type)
        if move_func is None:
            raise ValueError("Invalid movement type. Use 'j', 'l', or 'p'.")

        if pose is not None:
            print(f"Moving the robot to pose: {pose}...")
        elif joint is not None:
            print(f"Moving the robot joints to: {joint}...")
        else:
            raise ValueError("No valid robot pose/joint position is given")

        move_func(q=joint, pose=pose)
        time.sleep(0.5)
        print("Robot moved to position!")

        if return_joint:
            return self.get_joints()

        return 1

    def go_home(self):
        """
        Move the robot to the home position.

        Returns:
        int: 1 if successful.
        """
        print("Going to home position...")
        self.robot.movej(q=self.home_position, a=self.acc, v=self.vel)
        time.sleep(0.5)
        print("Robot moved to home position!")
        return 1

    def go_init(self):
        """
        Move the robot to the initial position.

        Returns:
        int: 1 if successful.
        """
        print("Going to the initial position...")
        self.robot.movej(q=self.init_position, a=self.acc, v=self.vel)
        time.sleep(0.5)
        print("Robot moved to the initial position!")
        return 1

    def check_joint_limit(self, joint_limit=300.0):
        """
        Check if the robot joints are within safe limits.

        Returns:
        int: 1 if within limits, otherwise moves to initial position.
        """
        print("Checking joint limits...")
        joints = self.get_joints()
        ranges = [(joint_limit, 360.0), (-360.0, -joint_limit)]
        if any(
            any(lower <= joint <= upper for lower, upper in ranges) for joint in joints
        ):
            print("Approaching joint limits!")
            self.go_init()
        return 1

    def close(self):
        """
        Close the robot connection.
        """
        if self.robot:
            print("Closing robot connection")
            self.robot.close()
        else:
            raise ValueError("No robot instance found!")


if __name__ == "__main__":
    UR5 = UR5RobotController("192.168.2.144")  # URSim

    try:
        print(UR5)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing robot connection.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Closing robot connection.")
        UR5.close()
