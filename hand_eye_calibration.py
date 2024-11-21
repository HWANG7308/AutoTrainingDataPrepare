"""
Hand-Eye Calibration Using a Chessboard

TODO debug this script for hand-eye calibration
"""

import cv2
import numpy as np
import math
import math3d as m3d
from PoseGenerator import PoseGenerator
from URController import UR5RobotController
from DepthCamera import D435


def get_images(robot_poses, UR5, DC):
    """
    Take images of a chessboard with given robot end effector poses

    robot_poses (list): a list of robot end effector poses
    UR5 (UR5RobotController): the robot controller instance
    DC (D435): the depth camera instance

    return images (list of np.ndarray): a list of captured images
    """
    images = []

    try:
        for pose in robot_poses:
            next_pose = pose.get("next pose")

            print(f"Moving the robot to {next_pose}...")
            UR5.move_robot(next_pose)
            print("Robot moved to position!")

            print("Getting data from the camera...")
            out, success = DC.get_frames(
                return_intrinsics=True,
                with_repair=False,
            )

            if not success:
                print("Failed to get data at this position!")
                continue

            images.append(out.get("color"))

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing connections.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        UR5.close()
        DC.pipe.stop()

    return images


def get_camera_poses_with_chessboard(images, DC):
    """
    Get the camera poses regarding a list of pictures of a chessboard captured from a specific view

    images (list of np.ndarray): the images of a chessboard captured from specific views

    return camera_poses (list of np.ndarray): a list of the camera poses represented by a 4x4 transformation matrix
    """
    camera_poses = []

    # Get camera intrinsics
    camera_intrinsics = DC.get_color_intrinsics()
    cam_K = np.array(
        [
            [camera_intrinsics.get("fx"), 0, camera_intrinsics.get("ppx")],
            [0, camera_intrinsics.get("fy"), camera_intrinsics.get("ppy")],
            [0, 0, 1],
        ]
    )
    dist_coeffs = camera_intrinsics.get("coeffs")

    # Define the dimensions of the chessboard
    chessboard_size = (9, 6)
    square_size = 0.025

    # Prepare 3D points in the chessboard coordinate system
    object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[
        0 : chessboard_size[0], 0 : chessboard_size[1]
    ].T.reshape(-1, 2)
    object_points *= square_size

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(
                gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria
            )

            # Solve PnP to get the rotation and translation vectors
            ret, rvec, tvec = cv2.solvePnP(object_points, corners, cam_K, dist_coeffs)

            if ret:
                # Convert rotation vector to rotation matrix
                R_c, _ = cv2.Rodrigues(rvec)
                T_c = tvec

                camera_poses.append((R_c, T_c))
            else:
                print("PnP solution failed for this image.")
        else:
            print("Chessboard corners not found in image.")

    return camera_poses


def hand_eye_calibration(robot_poses, camera_poses):
    """
    Get the transformation from the robot end effector to the in-hand camera using hand-eye calibration.

    robot_poses (list): a list of robot end effector poses to take pictures of a chessboard
    camera_poses (list): a list of camera poses matching the list of robot end effector poses

    return T_end2cam (math3d.Transform): the transformation from the robot end effector to the camera
    """
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

    print("Transformation from end-effector to camera:")
    print("Rotation Matrix (R_ec):\n", R_ec)
    print("Translation Vector (T_ec):\n", T_ec)

    T_end2cam = m3d.Transform(m3d.Orientation(R_ec), m3d.Vector(T_ec))

    return T_end2cam


if __name__ == "__main__":
    # Step 1: Create an UR5 instance and a D435 instance
    ROBOT_IP = "192.168.2.144"  # URSim
    # ROBOT_IP = "192.168.2.196"  # UR5
    UR5 = UR5RobotController(ROBOT_IP)
    DC = D435()

    # Step 2: Generate a list of robot end effector poses to take images of a chessboard
    T_rob2obj = m3d.Transform(
        m3d.Orientation.new_rotation_vector((math.pi / 2, 0, 0)), m3d.Vector(0, -0.6, 0)
    )
    T_end2cam_temp = m3d.Transform(
        m3d.Orientation.new_rotation_vector((0, 0, 0)), m3d.Vector(0, 0, 0.05)
    )
    robot_poses = PoseGenerator(
        T_rob2obj, T_end2cam_temp
    ).generate_positions_hand_eye_calibration()

    # Step 3: Take images with given robot end effector poses
    images = get_images(robot_poses, UR5, DC)

    # Step 4: Detect chessboard corners and calculate camera poses
    camera_poses = get_camera_poses_with_chessboard(images, DC)

    # Step 5: Perform hand-eye calibration and calculate the transformation matrix from the robot end effector to the camera
    T_end2cam = hand_eye_calibration(robot_poses, camera_poses)

    print(T_end2cam)
