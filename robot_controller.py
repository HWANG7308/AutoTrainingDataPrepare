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
        ]  # the joint values in degree when the robot is at home position

        # self.init_position = [
        #     math.radians(149.91),
        #     math.radians(-83.37),
        #     math.radians(152.57),
        #     math.radians(-69.2),
        #     math.radians(59.91),
        #     math.radians(180)
        # ]  # The joint positions the robot starts at
        self.init_position = [
            math.radians(2),
            math.radians(-27),
            math.radians(-158),
            math.radians(5),
            math.radians(-2),
            math.radians(0),
        ]  # The joint positions the robot starts at

        self.init_robot()

    def init_robot(self):
        print("Initializing robot...")

        self.robot = URBasic.urScriptExt.UrScriptExt(
            host=self.robot_ip, robotModel=self.robot_model
        )
        self.robot.reset_error()
        self.robot.movej(q=self.home_position, a=self.acc, v=self.vel)

        self.robot.waitRobotIdleOrStopFlag()

        self.robot.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller

        time.sleep(0.5)  # just a short wait to make sure everything is initialised

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
        if type == "deg":
            currentJoins = self.robot.get_actual_joint_positions()
            currentJoints = np.degrees(currentJoins)
        elif type == "rad":
            currentJoints = self.robot.get_actual_joint_positions()
        else:
            print("get_joints: wrong type")
            currentJoints = -1

        return currentJoints

    def is_moving(self):
        raise NotImplemented

    def is_home(self, eps=0.02):
        j = self.get_joints()
        t = np.array(self.home_position)
        d = np.abs(t - j)
        home = True
        for q in d:
            if q > eps:
                home = False
                break

        return home

    def at_target(self, t, type="deg", eps=0.02):
        j = self.get_joints(type=type)
        if (
            t[0] + eps > j[0] > t[0] - eps
            and j[1] < t[1] + eps
            and j[1] > t[1] - eps
            and j[2] < t[2] + eps
            and j[2] > t[2] - eps
            and j[3] < t[3] + eps
            and j[3] > t[3] - eps
            and j[4] < t[4] + eps
            and j[4] > t[4] - eps
            and j[5] < t[5] + eps
            and j[5] > t[5] - eps
        ):
            at_target = True
        else:
            at_target = False
        return at_target

    def move_robot(self, next_pose):
        # inv_kin = pose["inverse kinematic"]
        # robot.set_realtime_pose(next_pose)
        self.robot.movej(pose=next_pose)
        self.robot.waitRobotIdleOrStopFlag()

        return 1


if __name__ == "__main__":

    UR5 = UR5RobotController("192.168.2.144")  # URSim

    try:
        print(UR5)

    except KeyboardInterrupt:
        print("closing robot connection")
        # Remember to always close the robot connection, otherwise it is not possible to reconnect
        UR5.close()

    except:
        print("closing robot connection")
        # Remember to always close the robot connection, otherwise it is not possible to reconnect
        UR5.close()

    finally:
        print("closing robot connection")
        # Remember to always close the robot connection, otherwise it is not possible to reconnect
        UR5.close()
