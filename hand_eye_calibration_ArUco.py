"""
Hand-eye calibration using ArUco markers with OpenCV

TODO debugging
"""

import cv2
import numpy as np
import math3d as m3d
from PoseGenerator import PoseGenerator
from URController import UR5RobotController
from DepthCamera import D435


def get_images(robot_poses, UR5, DC):
    images = []
    try:
        for pose in robot_poses:
            next_pose = pose.get("next pose")
            UR5.move_robot(next_pose)
            out, success = DC.get_frames(
                return_intrinsics=True,
                with_repair=False,
                return_first_try=True,
                return_first=True,
                check_state=True,
            )
            if success:
                images.append(out.get("color"))
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        UR5.close()
        DC.pipe.stop()
    return images


def get_camera_poses_with_aruco(images, DC):
    camera_poses = []
    camera_intrinsics = DC.get_color_intrinsics()
    cam_K = np.array(
        [
            [camera_intrinsics.get("fx"), 0, camera_intrinsics.get("ppx")],
            [0, camera_intrinsics.get("fy"), camera_intrinsics.get("ppy")],
            [0, 0, 1],
        ]
    )
    dist_coeffs = camera_intrinsics.get("coeffs")
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params
        )
        if ids is not None:
            ret, rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.05, cam_K, dist_coeffs
            )
            if ret:
                R_c, _ = cv2.Rodrigues(rvec[0])
                T_c = tvec[0]
                camera_poses.append((R_c, T_c))
    return camera_poses


def hand_eye_calibration(robot_poses, camera_poses):
    if len(camera_poses) != len(robot_poses):
        raise ValueError(
            "Number of camera poses does not match number of robot poses. Check input data."
        )
    R_e_list = [pose.get("T_rob2end")._o for pose in robot_poses]
    T_e_list = [pose.get("T_rob2end")._v for pose in robot_poses]
    R_c_list = [pose[0] for pose in camera_poses]
    T_c_list = [pose[1] for pose in camera_poses]
    R_ec, T_ec = cv2.calibrateHandEye(
        R_e_list, T_e_list, R_c_list, T_c_list, method=cv2.CALIB_HAND_EYE_TSAI
    )
    T_end2cam = m3d.Transform(m3d.Orientation(R_ec), m3d.Vector(T_ec))
    return T_end2cam


if __name__ == "__main__":
    ROBOT_IP = "192.168.2.144"
    UR5 = UR5RobotController(ROBOT_IP)
    DC = D435()
    T_rob2obj = m3d.Transform(
        m3d.Orientation.new_rotation_vector((math.pi / 2, 0, 0)), m3d.Vector(0, -0.6, 0)
    )
    T_end2cam_temp = m3d.Transform(
        m3d.Orientation.new_rotation_vector((0, 0, 0)), m3d.Vector(0, 0, 0.05)
    )
    robot_poses = PoseGenerator(
        T_rob2obj, T_end2cam_temp
    ).generate_positions_hand_eye_calibration()
    images = get_images(robot_poses, UR5, DC)
    camera_poses = get_camera_poses_with_aruco(images, DC)
    T_end2cam = hand_eye_calibration(robot_poses, camera_poses)
    print(T_end2cam)
