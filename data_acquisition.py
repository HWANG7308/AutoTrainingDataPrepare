"""
Data acquisition functions

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
import numpy as np
from utils.PoseGenerator import PoseGenerator
from utils.URController import UR5RobotController
from utils.CameraController import D435
from utils.utils import get_selection


def save_data_sample(
    data_save_dir, n, name, pose, out, UR5, T_end2cam, img_crop_size=(480, 640)
):
    """
    Save the data sample including images and metadata.

    Parameters:
    data_save_dir (str): Directory to save the data.
    n (int): Index of the current sample.
    name (str): Name of the object.
    pose (dict): Pose information.
    out (dict): Output from the depth camera.
    UR5 (UR5RobotController): The robot controller instance.
    T_end2cam (m3d.Transform): Transformation from the end effector to the camera.
    """

    save_dir = {
        key: os.path.join(data_save_dir, key.split("_")[0])
        for key in [
            "color_img_dir",
            "color-cropped_img_dir",
            "depth_img_dir",
            "meta_info_dir",
        ]
    }

    for dir_path in save_dir.keys():
        os.makedirs(save_dir.get(dir_path), exist_ok=True)

    color_image = out.get("color")
    depth_image = out.get("depth")

    if color_image is not None:
        cv2.imwrite(
            os.path.join(save_dir.get("color_img_dir"), f"color_{n:06d}.png"),
            color_image,
        )
        if img_crop_size is not None:
            if (
                color_image.shape[0] >= img_crop_size[0]
                and color_image.shape[1] >= img_crop_size[1]
            ):
                color_cropped_image = color_image[
                    (color_image.shape[0] - img_crop_size[0])
                    // 2 : (color_image.shape[0] + img_crop_size[0])
                    // 2,
                    (color_image.shape[1] - img_crop_size[1])
                    // 2 : (color_image.shape[1] + img_crop_size[1])
                    // 2,
                ]
                cv2.imwrite(
                    os.path.join(
                        save_dir.get("color-cropped_img_dir"),
                        f"color-cropped_{n:06d}.png",
                    ),
                    color_cropped_image,
                )
            # TODO: Add a warning if the image is too small to be cropped
            # TODO: Add a function to white out the background instead of cropping
            else:
                print(
                    f"Warning: Color image for sample {n} is too small to be cropped."
                )
    else:
        print(f"Warning: Color image for sample {n} is None and will not be saved.")

    if depth_image is not None:
        cv2.imwrite(
            os.path.join(save_dir.get("depth_img_dir"), f"depth_{n:06d}.png"),
            depth_image,
        )
    else:
        print(f"Warning: Depth image for sample {n} is None and will not be saved.")

    meta = {
        "class": name,
        "time": datetime.datetime.today().strftime("%Y-%m-%d, %H:%M:%S"),
        "view_point_id": n,
        "robot_arm_joints": UR5.get_joints().tolist(),
        "object_pose": pose.get("T_obj2cam").get_inverse().get_matrix().tolist(),
        "tf_rob2end": pose.get("T_rob2end").get_matrix().tolist(),
        "intrinsics_color": out.get("color_intr"),
        "depth_scale": out.get("depth_scale"),
        "hand_eye_calibration": T_end2cam.get_matrix().tolist(),
        "color_img_saved": color_image is not None,
        "depth_img_saved": depth_image is not None,
    }
    with open(
        os.path.join(save_dir.get("meta_info_dir"), f"meta_{n:06d}.json"), "w"
    ) as f:
        json.dump(meta, f, indent=4)

    print("Data sample saved!")


def acquire_new_data_from_object():
    """
    Acquire new images from an object by taking images with given robot poses.
    """
    UR5 = UR5RobotController(ROBOT_IP)
    DC = D435()

    robot_poses = PoseGenerator(T_rob2obj, T_end2cam).generate_positions(
        change_first="azimuth"
    )

    data_dir = os.path.join(root, "result/acquired_data")
    os.makedirs(data_dir, exist_ok=True)

    names = list(os.listdir(data_dir))
    while True:
        print("\n" + "_" * 70)
        name = input("Enter name of the new object: ")
        if name in names:
            print(
                f"An object with the name, {name}, already exists. Please find a different name."
            )
            continue
        print("Current name is:", name)
        break

    data_save_dir = os.path.join(data_dir, name)
    os.makedirs(data_save_dir, exist_ok=True)
    print("Saving data to:", data_save_dir)

    try:
        for n, pose in enumerate(robot_poses):
            print(f"Position {n}:")
            next_pose = pose.get("next pose")
            _ = UR5.move_robot(pose=next_pose)
            print("Getting data from the camera...")
            out = DC.get_frames(return_intrinsics=True, with_repair=False)
            save_data_sample(data_save_dir, n, name, pose, out, UR5, T_end2cam)
            # UR5.go_init() #TODO find a way to optimize the robot path otherwise go back to initial position every time
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing connections.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        UR5.robot.close()
        print("Closing camera")
        DC.pipe.stop()


def acquire_new_data_from_object_demo():
    """
    Acquire new images from an object by taking images with given robot poses (for demo).
    """
    UR5 = UR5RobotController(ROBOT_IP)
    DC = D435()

    pose_top = PoseGenerator(T_rob2obj, T_end2cam).generate_position_example()
    pose_mid = PoseGenerator(T_rob2obj, T_end2cam).generate_position_example(
        phi=math.pi / 4
    )
    pose_front = PoseGenerator(T_rob2obj, T_end2cam).generate_position_example(
        phi=math.pi / 2
    )
    robot_poses = pose_top + pose_mid + pose_front

    data_dir = os.path.join(root, "result/acquired_data")
    os.makedirs(data_dir, exist_ok=True)

    names = list(os.listdir(data_dir))

    while True:
        print("\n" + "_" * 70)
        name = input("Enter name of the new object: ")
        if name in names:
            print(
                f"An object with the name, {name}, already exists. Please find a different name."
            )
            continue
        print("Current name is:", name)
        break

    data_save_dir = os.path.join(data_dir, name)
    print("Saving data to:", data_save_dir)
    os.makedirs(data_save_dir, exist_ok=True)

    try:
        for n, pose in enumerate(robot_poses):
            print(f"Position {n}:")
            next_pose = pose.get("next pose")
            UR5.move_robot(pose=next_pose)
            print("Getting data from the camera...")
            out = DC.get_frames(return_intrinsics=True, with_repair=False)
            save_data_sample(data_save_dir, n, name, pose, out, UR5, T_end2cam)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing connections.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        UR5.robot.close()
        print("Closing camera")
        DC.pipe.stop()


def acquire_new_data_from_object_with_joints():
    """
    Acquire new images from an object by taking images with given robot joint positions.
    """
    UR5 = UR5RobotController(ROBOT_IP)
    DC = D435(color_width=1920, color_height=1080, depth_width=1280, depth_height=720)

    robot_time = 0.0
    camera_time = 0.0
    save_img_time = 0.0

    robot_poses = PoseGenerator(T_rob2obj, T_end2cam).generate_positions(
        change_first="azimuth"
    )

    robot_poses[16], robot_poses[17] = (
        robot_poses[17],
        robot_poses[16],
    )  # swap the positions of the robot poses due to the robot's limitation (the robot joints were set manually) TODO: find a way to optimize the robot path

    with open("data/robot_joints_demo3.json", "r") as f:
        robot_joints = json.load(f)

    data_dir = os.path.join(root, "result/acquired_data")
    os.makedirs(data_dir, exist_ok=True)

    names = list(os.listdir(data_dir))

    while True:
        print("\n" + "_" * 70)
        name = input("Enter name of the new object: ")
        if name in names:
            print(
                f"An object with the name, {name}, already exists. Please find a different name."
            )
            continue
        print("Current name is:", name)
        break

    data_save_dir = os.path.join(data_dir, name)
    print("Saving data to:", data_save_dir)
    os.makedirs(data_save_dir, exist_ok=True)

    try:
        for n, pose in enumerate(robot_poses):
            print("\n" + "_" * 70)
            print(
                f"Taking the image with robot joint configurations {n} ({robot_joints.get(str(n))})..."
            )

            print("Moving the robot to the target position...")
            robot_start_time = time.time()
            while not UR5.at_target(robot_joints.get(str(n))):
                UR5.move_robot(joint=np.radians(robot_joints.get(str(n))))
            robot_end_time = time.time()
            robot_time += robot_end_time - robot_start_time

            print("Getting data from the camera...")
            camera_start_time = time.time()
            out = DC.get_frames(return_intrinsics=True, with_repair=False)
            camera_end_time = time.time()
            camera_time += camera_end_time - camera_start_time

            print("Saving the data...")
            save_img_start_time = time.time()
            save_data_sample(data_save_dir, n, name, pose, out, UR5, T_end2cam)
            save_img_end_time = time.time()
            save_img_time += save_img_end_time - save_img_start_time

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing connections.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        UR5.robot.close()
        print("Closing camera")
        DC.pipe.stop()

    print(
        f"Robot time:{robot_time}\nCamera time: {camera_time}\nSave image time: {save_img_time}"
    )
    time_report = {
        "Robot time": robot_time,
        "Camera time": camera_time,
        "Save img time": save_img_time,
    }
    with open(os.path.join(data_save_dir, f"time.json"), "w") as f:
        json.dump(time_report, f, indent=4)


def main():
    s = {
        "Acquire New Data from Object": acquire_new_data_from_object,
        "Acquire New Data from Object (Demo)": acquire_new_data_from_object_demo,
        "Acquire New Data from Object with Given Robot Joint Positions": acquire_new_data_from_object_with_joints,
    }

    while True:
        print("\n" + "_" * 70)
        selection = get_selection(
            list(sorted(s.keys())), "Main Menu", with_exit=True, with_return=False
        )
        if selection == "exit":
            break
        else:
            print(f"Selected: {selection}")
            s[selection]()


if __name__ == "__main__":
    root = str(Path(__file__).resolve().parent)

    # ROBOT_IP = "192.168.2.144"  # URSim
    ROBOT_IP = "192.168.2.196"  # UR5cb3

    T_rob2obj = m3d.Transform(
        m3d.Orientation.new_euler((math.pi / 2, 0, math.pi), "XYZ"),
        m3d.Vector(0, -0.7, 0),
    )
    T_end2cam = m3d.Transform(
        m3d.Orientation.new_rotation_vector((0, 0, 0)), m3d.Vector(0, 0, 0.05)
    )

    main()
