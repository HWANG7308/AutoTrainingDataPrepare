"""
TODO fix the introduction here
UR robot movement based on RTDE
Python program for realtime movement of a Universal Robot (tested with UR5cb)

Created by Hao Wang
License: MIT License

"""

import os
import json
from pathlib import Path
import time
import datetime
import math
import math3d as m3d
import cv2
from PoseGenerator import PoseGenerator
from URController import UR5RobotController
from DepthCamera import D435
from DataAnnotator import (
    Annotator2DBBox,
    Annotator3DBBox,
    Annotator6DPose,
)
from utils import *


def acquire_new_data_from_object():
    """
    Acquire new images from an object by taking images with given robot poses
    """

    # create a UR5
    UR5 = UR5RobotController(ROBOT_IP)

    # create a depth camera
    DC = D435()

    # Generate the end-effector positions to capture object images from various defined views
    robot_poses = PoseGenerator(T_rob2obj, T_end2cam).generate_position_example(
        theta=-math.pi / 4, phi=-math.pi / 4
    )  # for test only
    # robot_poses = PoseGenerator(T_rob2obj, T_end2cam).generate_positions()

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

    try:
        for n, pose in enumerate(robot_poses):
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

    except KeyboardInterrupt:
        print("Closing robot connection")
        # Remember to always close the robot connection, otherwise it is not possible to reconnect
        UR5.robot.close()

        print("Closing camera")
        DC.pipe.stop()

    except:
        print("Closing robot connection")
        # Remember to always close the robot connection, otherwise it is not possible to reconnect
        UR5.robot.close()

        print("Closing camera")
        DC.pipe.stop()

    finally:
        print("Closing robot connection")
        # Remember to always close the robot connection, otherwise it is not possible to reconnect
        UR5.robot.close()

        print("Closing camera")
        DC.pipe.stop()


def create_labels_2dbbox():
    color_img_path = "results/acquired_data/test/color_000000.png"
    depth_img_path = None

    # Annotate 2D bounding boxes
    DA_2dbbox = Annotator2DBBox(color_img_path, depth_img_path)
    DA_2dbbox.remove_bkg_chroma_key(show_result=True)
    DA_2dbbox.annotate(show_result=True)


def create_labels_6dpose():
    color_img_path = "results/acquired_data/test/color_000000.png"
    depth_img_path = None
    meta_path_ = os.path.join(
        os.path.dirname(color_img_path),
        os.path.basename(color_img_path).replace("color", "meta"),
    )
    meta_path = os.path.splitext(meta_path_)[0] + ".json"

    # Annotate 6D poses
    DA_6dpose = Annotator6DPose(color_img_path, depth_img_path, meta_path)
    DA_6dpose.annotate(show_result=True)


def create_labels_3dbbox():
    color_img_path = "results/acquired_data/test/color_000000.png"
    depth_img_path = None
    meta_path_ = os.path.join(
        os.path.dirname(color_img_path),
        os.path.basename(color_img_path).replace("color", "meta"),
    )
    meta_path = os.path.splitext(meta_path_)[0] + ".json"

    # Annotate 3D bounding boxes
    DA_2DBBox_top = Annotator2DBBox(color_img_path, depth_img_path)
    DA_2DBBox_top.annotate()
    DA_2DBBox_front = Annotator2DBBox(color_img_path, depth_img_path)
    DA_2DBBox_front.annotate()
    annotation_top = DA_2DBBox_top.annotation
    annotation_front = DA_2DBBox_front.annotation
    DA_3dbbox = Annotator3DBBox(color_img_path, depth_img_path, meta_path)
    oriented_3dbbox = DA_3dbbox.reconstruct_oriented_3dbbox(
        annotation_top, annotation_front
    )
    DA_3dbbox.visualize_3dbbox(oriented_3dbbox)


def create_labels_img_seg():
    raise NotImplemented


def train_object_detection():
    raise NotImplemented


def train_pose_estimation():
    raise NotImplemented


def run_live_prediction_obj_detect():
    raise NotImplemented


def run_live_prediction_pose_estimate():
    raise NotImplemented


def visualize():
    raise NotImplemented


def hand_eye_calibration():
    raise NotImplemented


def main():
    s = {
        "Acquire New Data from Object": acquire_new_data_from_object,
        "Create Labels (2D BBox)": create_labels_2dbbox,
        "Create Labels (6D Pose)": create_labels_6dpose,
        "Create Labels (3D BBox)": create_labels_3dbbox,
        "Create Labels (Image Segmentation)": create_labels_img_seg,
        "Train Object Detection Model": train_object_detection,
        "Train Pose Estimation Model": train_pose_estimation,
        "Run Live Prediction (Object Detection)": run_live_prediction_obj_detect,
        "Run Live Prediction (Pose Estimation)": run_live_prediction_pose_estimate,
        "Visualize": visualize,
        "Hand-Eye Calibration": hand_eye_calibration,
    }

    while True:
        print("____________________________________________________________________")
        selection = get_selection(
            list(sorted(s.keys())), "Main Menu", with_exit=True, with_return=False
        )
        if selection == "exit":
            break
        else:
            s[selection]()


if __name__ == "__main__":
    root = str(Path(__file__).resolve().parent)

    ROBOT_IP = "192.168.2.144"  # URSim
    # ROBOT_IP = "192.168.2.196"  # UR5

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

    main()
