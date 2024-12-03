import URBasic
import math3d as m3d
import math
import numpy as np
import time


class UR5RobotController:
    def __init__(self, robot_ip, acceleration=0.9, velocity=0.8):
        """
        robot_ip: the ip address of the robot
        acceleration: robot acceleration value
        velocity: robot speed value
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

        self.init_position_1 = [
            math.radians(0),
            math.radians(-25),
            math.radians(-155),
            math.radians(0),
            math.radians(0),
            math.radians(0),
        ]

        self.init_position_2 = [
            math.radians(149.91),
            math.radians(-83.37),
            math.radians(152.57),
            math.radians(-69.2),
            math.radians(59.91),
            math.radians(180),
        ]

        self.init_position_3 = [
            math.radians(-90),
            math.radians(-90),
            math.radians(-90),
            math.radians(-90),
            math.radians(90),
            math.radians(0),
        ]

        self.init_robot()

    def init_robot(self):
        print("Initializing robot...")

        self.robot = URBasic.urScriptExt.UrScriptExt(
            host=self.robot_ip, robotModel=self.robot_model
        )
        self.robot.reset_error()
        self.robot.movej(q=self.init_position_3, a=self.acc, v=self.vel)
        self.robot.waitRobotIdleOrStopFlag()
        self.robot.init_realtime_control()
        time.sleep(0.5)

        print("Robot initialized!")

    def set_lookorigin(self):
        """
        Creates a new coordinate system at the current robot tcp position.
        This coordinate system is the basis of the face following.
        It describes the midpoint of the plane in which the robot follows faces.

        Return Value:
            orig: math3D Transform Object
                characterises location and rotation of the new coordinate system in reference to the base coordinate system

        """
        position = self.robot.get_actual_tcp_pose()
        orig = m3d.Transform(position)
        return orig

    def get_joints(self, type="deg"):
        currentJoints = self.robot.get_actual_joint_positions()
        if type == "deg":
            currentJoints = np.degrees(currentJoints)
        elif type != "rad":
            print("get_joints: wrong type")
            return -1
        return currentJoints

    def is_moving(self):
        raise NotImplementedError

    def is_home(self, eps=0.02):
        j = self.get_joints()
        t = np.array(self.home_position)
        d = np.abs(t - j)
        return np.all(d < eps)

    def at_target(self, t, type="deg", eps=0.02):
        j = self.get_joints(type=type)
        return np.all(np.abs(np.array(t) - j) < eps)

    def move_robot(self, next_pose, type="j"):

        self.check_joint_limit()

        # inv_kin = pose["inverse kinematic"]
        # robot.set_realtime_pose(next_pose)
        if type == "j":
            self.robot.movej(pose=next_pose)
        elif type == "l":
            self.robot.movel(pose=next_pose)
        elif type == "p":
            self.robot.movep(pose=next_pose)
        else:
            raise TypeError("Only j, l, and p are allowed for moving type")

        self.robot.waitRobotIdleOrStopFlag()
        return 1

    def go_home(self):
        self.robot.movej(q=self.home_position, a=self.acc, v=self.vel)
        self.robot.waitRobotIdleOrStopFlag()
        return 1

    def check_joint_limit(self):
        joints = self.get_joints()
        ranges = [(300, 360), (-360, -300)]
        if any(
            any(lower <= joint <= upper for lower, upper in ranges) for joint in joints
        ):
            print("Approaching joint limits! Go back to home position...")
            self.go_home()
            self.robot.waitRobotIdleOrStopFlag()

        return 1

    def close(self):
        if self.robot:
            self.robot.close()


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
