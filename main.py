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
import datetime
import math
import math3d as m3d
import cv2
import HandEyeCalibration
from PoseGenerator import PoseGenerator
from URController import UR5RobotController
from DepthCamera import D435
from DataAnnotator import Annotator2DBBox, Annotator3DBBox, Annotator6DPose
from utils import get_selection


def save_data_sample(data_save_dir, n, name, pose, out, UR5):
    """
    Save the data sample including images and metadata.

    Parameters:
    data_save_dir (str): Directory to save the data.
    n (int): Index of the current sample.
    name (str): Name of the object.
    pose (dict): Pose information.
    out (dict): Output from the depth camera.
    UR5 (UR5RobotController): The robot controller instance.
    """

    save_dir = {
        "color_img_dir": os.path.join(data_save_dir, "color"),
        "depth_img_dir": os.path.join(data_save_dir, "depth"),
        "meta_info_dir": os.path.join(data_save_dir, "meta"),
    }

    for dir_path in save_dir.keys():
        os.makedirs(dir_path, exist_ok=True)

    cv2.imwrite(
        os.path.join(save_dir.get("color_img_dir"), f"color_{n:06d}.png"),
        out.get("color"),
    )
    cv2.imwrite(
        os.path.join(save_dir.get("depth_img_dir"), f"depth_{n:06d}.png"),
        out.get("depth"),
    )
    meta = {
        "class": name,
        "time": datetime.datetime.today().strftime("%Y-%m-%d, %H:%M:%S"),
        "view_point_id": n,
        "robot_arm_joints": UR5.get_joints().tolist(),
        "object_pose": pose.get("T_obj2cam").inverse.get_matrix().tolist(),
        "tf_rob2end": pose.get("T_rob2end").get_matrix().tolist(),
        "intrinsics_color": out.get("color_intr"),
        "depth_scale": out.get("depth_scale"),
        "hand_eye_calibration": T_end2cam.get_matrix().tolist(),
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
            save_data_sample(data_save_dir, n, name, pose, out, UR5)
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
            save_data_sample(data_save_dir, n, name, pose, out, UR5)
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
    TODO fix this function
    """
    import numpy as np

    UR5 = UR5RobotController(ROBOT_IP)
    DC = D435()

    robot_poses = PoseGenerator(T_rob2obj, T_end2cam).generate_positions(
        change_first="azimuth"
    )

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
            print(f"Robot joint {n}:", robot_joints.get(str(n)))
            while not UR5.at_target(robot_joints.get(str(n))):
                UR5.move_robot(joint=np.radians(robot_joints.get(str(n))))
            print("Getting data from the camera...")
            out = DC.get_frames(return_intrinsics=True, with_repair=False)
            save_data_sample(data_save_dir, n, name, pose, out, UR5)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing connections.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        UR5.robot.close()
        print("Closing camera")
        DC.pipe.stop()


def save_annotations(data_save_dir, n, annotations):
    """
    Save the annotations to a JSON file.

    Parameters:
    data_save_dir (str): Directory to save the annotations.
    n (int): Index of the current sample.
    annotations (dict): Annotations to save.
    """
    os.makedirs(data_save_dir, exist_ok=True)
    with open(os.path.join(data_save_dir, f"meta_{n:06d}.json"), "w") as f:
        json.dump(annotations, f, indent=4)
    print("Data annotation saved!")


def create_labels(annotation_type, save_vis=False):
    """
    Create labels for the acquired data.

    Parameters:
    annotation_type (str): Type of annotation ("2dbbox", "6dpose", "3dbbox", "img_seg").
    """
    raw_data_dir = os.path.join(root, "result/acquired_data")
    data_save_dir = os.path.join(root, "result/annotated_data")

    names = list(os.listdir(raw_data_dir))

    for _, name in enumerate(names):
        color_img_dir = os.path.join(raw_data_dir, name, "color")
        color_imgs = list(os.listdir(color_img_dir))
        for n, color_img in enumerate(color_imgs):
            color_img_path = os.path.join(color_img_dir, color_img)
            depth_img_path = color_img_path.replace("color", "depth")
            meta_path_ = color_img_path.replace("color", "meta")
            meta_path = os.path.splitext(meta_path_)[0] + ".json"

            save_vis_dir = (
                os.path.join(data_save_dir, name, annotation_type, "vis")
                if save_vis
                else None
            )

            if annotation_type == "2dbbox":
                annotator = Annotator2DBBox(color_img_path, depth_img_path, meta_path)
                _ = annotator.remove_bkg_chroma_key(save_vis_dir=save_vis_dir)
                annotations = annotator.annotate(save_vis_dir=save_vis_dir)
            elif annotation_type == "6dpose":
                annotator = Annotator6DPose(color_img_path, depth_img_path, meta_path)
                annotations = annotator.annotate(save_vis_dir=save_vis_dir)
            elif annotation_type == "3dbbox":
                raise ValueError("Yet to be merged")  # TODO merge the function
            elif annotation_type == "img_seg":
                raise ValueError("Yet to be merged")  # TODO merge the function
            else:
                raise ValueError(
                    "Invalid annotation type. Use '2dbbox', '6dpose', '3dbbox', or 'img_seg'."
                )

            save_annotations(
                os.path.join(data_save_dir, name, annotation_type, "label"),
                n,
                annotations,
            )


def create_labels_2dbbox():
    """
    Create 2D bounding box labels for the acquired data.
    """
    create_labels("2dbbox")


def create_labels_6dpose():
    """
    Create 6D pose labels for the acquired data.
    """
    create_labels("6dpose")


def create_labels_3dbbox():
    raw_data_dir = os.path.join(root, "result/acquired_data")
    data_save_dir = os.path.join(root, "result/annotated_data")
    os.makedirs(data_save_dir, exist_ok=True)

    names = list(os.listdir(raw_data_dir))

    for n, name in enumerate(names):

        # top_color_img_path = "result/acquired_data/test_bkp/color_000000.png"
        # front_color_img_path = "result/acquired_data/test/color_000002.png"
        # depth_img_path = None
        # test_img_path = "result/acquired_data/test_old/color_000000.png"

        # TODO fix the paths of top view and front view here
        top_color_img_path = os.path.join(raw_data_dir, name, "color_000000.png")
        top_depth_img_path = os.path.join(
            os.path.dirname(top_color_img_path),
            os.path.basename(top_color_img_path).replace("color", "depth"),
        )
        front_color_img_path = os.path.join(raw_data_dir, name, "color_000002.png")
        front_depth_img_path = os.path.join(
            os.path.dirname(front_color_img_path),
            os.path.basename(front_color_img_path).replace("color", "depth"),
        )

        color_img_path = os.path.join(raw_data_dir, name, f"color_{n:06d}.png")
        depth_img_path = os.path.join(
            os.path.dirname(color_img_path),
            os.path.basename(color_img_path).replace("color", "depth"),
        )
        meta_path_ = os.path.join(
            os.path.dirname(color_img_path),
            os.path.basename(color_img_path).replace("color", "meta"),
        )
        meta_path = os.path.splitext(meta_path_)[0] + ".json"

        # Annotate 3D bounding boxes
        DA_2DBBox_top = Annotator2DBBox(top_color_img_path, top_depth_img_path)
        DA_2DBBox_top.annotate()
        DA_2DBBox_front = Annotator2DBBox(front_color_img_path, front_depth_img_path)
        DA_2DBBox_front.annotate()
        annotation_top = DA_2DBBox_top.annotation
        annotation_front = DA_2DBBox_front.annotation

        DA_3dbbox = Annotator3DBBox(color_img_path, depth_img_path, meta_path)
        oriented_3dbbox = DA_3dbbox.reconstruct_oriented_3dbbox(
            annotation_front, annotation_top
        )
        DA_3dbbox.visualize_3dbbox(oriented_3dbbox)

        # TODO save the meta information


def create_labels_img_seg():
    raise NotImplementedError


def train(task_type):
    if task_type == "2d_object_detection":
        raise ValueError("Yet to be merged")  # TODO merge the function
    elif task_type == "6d_pose_estimation":
        raise ValueError("Yet to be merged")  # TODO merge the function
    elif task_type == "3d_object_detection":
        raise ValueError("Yet to be merged")  # TODO merge the function
    elif task_type == "img_segmentation":
        raise ValueError("Yet to be merged")  # TODO merge the function
    else:
        raise ValueError(
            "Invalid task type. Use '2d_object_detection', '6d_pose_estimation', '3d_object_detection', or 'img_segmentation'."
        )


def train_2d_object_detection():
    train("2d_object_detection")


def train_6d_pose_estimation():
    train("6d_pose_estimation")


def prediction(task_type):
    if task_type == "2d_object_detection":
        raise ValueError("Yet to be merged")  # TODO merge the function
    elif task_type == "6d_pose_estimation":
        raise ValueError("Yet to be merged")  # TODO merge the function
    elif task_type == "3d_object_detection":
        raise ValueError("Yet to be merged")  # TODO merge the function
    elif task_type == "img_segmentation":
        raise ValueError("Yet to be merged")  # TODO merge the function
    else:
        raise ValueError(
            "Invalid task type. Use '2d_object_detection', '6d_pose_estimation', '3d_object_detection', or 'img_segmentation'."
        )


def run_live_prediction_2d_obj_detect():
    prediction("2d_object_detection")


def run_live_prediction_6d_pose_estimate():
    prediction("6d_pose_estimation")


def visualize():
    raise NotImplementedError


def perform_hand_eye_calibration():
    """
    Perform hand-eye calibration using the robot and camera.
    """
    UR5 = UR5RobotController(ROBOT_IP)
    DC = D435()

    robot_poses = PoseGenerator(
        T_rob2obj, T_end2cam
    ).generate_positions_hand_eye_calibration()

    images = HandEyeCalibration.get_images(robot_poses, UR5, DC)

    camera_poses = HandEyeCalibration.get_camera_poses(images, DC, method="chessboard")

    T_end2cam_calib = HandEyeCalibration(robot_poses, camera_poses)

    print("Calibrated T_end2cam:\n", T_end2cam_calib)


def main():
    s = {
        "Acquire New Data from Object": acquire_new_data_from_object,
        "Acquire New Data from Object (Demo)": acquire_new_data_from_object_demo,
        "Acquire New Data from Object with Given Robot Joint Positions": acquire_new_data_from_object_with_joints,
        "Create Labels (2D BBox)": create_labels_2dbbox,
        "Create Labels (6D Pose)": create_labels_6dpose,
        "Create Labels (3D BBox)": create_labels_3dbbox,
        # "Create Labels (Image Segmentation)": create_labels_img_seg,
        # "Train 2D Object Detection Model": train_2d_object_detection,
        # "Train 6D Pose Estimation Model": train_6d_pose_estimation,
        # "Run Live Prediction (Object Detection)": run_live_prediction_obj_detect,
        # "Run Live Prediction (Pose Estimation)": run_live_prediction_pose_estimate,
        # "Visualize": visualize,
        "Hand-Eye Calibration": perform_hand_eye_calibration,
    }

    while True:
        print("\n" + "_" * 70)
        selection = get_selection(
            list(sorted(s.keys())), "Main Menu", with_exit=True, with_return=False
        )
        if selection == "exit":
            break
        else:
            s[selection]()


if __name__ == "__main__":
    root = str(Path(__file__).resolve().parent)

    # ROBOT_IP = "192.168.2.144"  # URSim
    ROBOT_IP = "192.168.2.196"  # UR5

    T_rob2obj = m3d.Transform(
        m3d.Orientation.new_euler((math.pi / 2, 0, math.pi), "XYZ"),
        m3d.Vector(0, -0.7, 0),
    )
    T_end2cam = m3d.Transform(
        m3d.Orientation.new_rotation_vector((0, 0, 0)), m3d.Vector(0, 0, 0.05)
    )

    main()
