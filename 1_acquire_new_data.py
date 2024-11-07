"""
TODO fix the introduction here
UR robot movement based on RTDE
Python program for realtime movement of a Universal Robot (tested with UR5cb)

Created by Hao Wang
License: TODO create a license

"""

import os
import json
from pathlib import Path
import time
import datetime
import math
import math3d as m3d
import cv2
from utils import PoseGenerator
from robot_controller import UR5RobotController
from depth_camera import D435

root = str(Path(__file__).resolve().parent)

# ROBOT_IP = "192.168.2.144"  # URSim
ROBOT_IP = "192.168.2.196"  # UR5

# The configurations of the robot system (the object position regarding the robot, the camera position regarding the end effector)
# The transformation from the robot base to the object (static, UR5)
T_rob2obj = m3d.Transform(
    m3d.Orientation.new_rotation_vector((math.pi / 2, 0, 0)), m3d.Vector(0, -0.6, 0)
)
# print('The transformation from the robot base to the object:', T_rob2obj)

# The transformation from the end effector to the camera (static)
T_end2cam = m3d.Transform(
    m3d.Orientation.new_rotation_vector((0, 0, 0)), m3d.Vector(0, 0, 0.05)
)
# print('The transformation from the end effector to the camera:', T_end2cam)


def acquire_new_data_from_object():

    # create a UR5
    UR5 = UR5RobotController(ROBOT_IP)

    # Generate the end-effector positions to capture object images from various defined views
    poses = PoseGenerator(T_rob2obj, T_end2cam).generate_position_example(
        theta=-math.pi / 4, phi=-math.pi / 4
    )  # for test only

    # poses = PoseGenerator(T_rob2obj, T_end2cam).generate_positions()

    # create a depth camera
    DC = D435()

    data_dir = os.path.join(root, "results/acquired_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    names = list(os.listdir(data_dir))

    while True:
        print("____________________________________________________________________")
        name = input("Enter name of the new object: ")
        if name in names:
            print(
                "An object with the name, {}, already exists. Please find a different name.".format(
                    name
                )
            )
            continue
        print("Current name is:", name)
        break

    data_save_dir = os.path.join(data_dir, name)
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    for n, pose in enumerate(poses):
        if n > 10:
            break

        next_pose = pose.get("next pose")

        print("Moving the robot to {}...".format(next_pose))
        UR5.move_robot(next_pose)
        print("Robot moved to position!")

        print("Getting data from the camera...")
        out, success = DC.get_frames(
            return_intrinsics=True,
            with_repair=False,
            return_first_try=True,
            return_first=True,
            check_state=True,
        )
        if not success:
            print("Failed to get data at this position!")
            continue

        print("Saving data to:", data_save_dir)
        cv2.imwrite(data_save_dir + "/color_{:06d}.png".format(n), out.get("color"))
        cv2.imwrite(data_save_dir + "/depth_{:06d}.png".format(n), out.get("depth"))

        # get meta data
        meta = {}
        meta["class"] = name
        meta["time"] = datetime.datetime.today().strftime("%Y-%m-%d, %H:%M:%S")
        meta["view_point_id"] = n
        meta["robot_arm_joints"] = UR5.get_joints().tolist()
        meta["object_pose"] = pose.get("T_obj2cam").inverse.get_matrix().tolist()
        meta["tf_rob2end"] = pose.get("T_rob2end").get_matrix().tolist()
        meta["intrinsics_color"] = out.get("color_intr")
        meta["depth_scale"] = out.get("depth_scale")
        # meta['hand_eye_calibration'] = None  # TODO fix the hand-eye calibration

        with open(data_save_dir + "/meta_{:06d}.json".format(n), "w") as f:
            json.dump(meta, f, indent=4)

        print("Data sample saved!")

    print("Closing robot connection")
    # Remember to always close the robot connection, otherwise it is not possible to reconnect
    UR5.robot.close()

    print("Closing camera")
    DC.pipe.stop()

    # try:

    # except KeyboardInterrupt:
    #     print('Closing robot connection')
    #     # Remember to always close the robot connection, otherwise it is not possible to reconnect
    #     UR5.robot.close()

    #     print('Closing camera')
    #     DC.pipe.stop()

    # except:
    #     print('Closing robot connection')
    #     # Remember to always close the robot connection, otherwise it is not possible to reconnect
    #     UR5.robot.close()

    #     print('Closing camera')
    #     DC.pipe.stop()

    # finally:
    #     print('Closing robot connection')
    #     # Remember to always close the robot connection, otherwise it is not possible to reconnect
    #     UR5.robot.close()

    #     print('Closing camera')
    #     DC.pipe.stop()


def main():
    acquire_new_data_from_object()


if __name__ == "__main__":
    main()
