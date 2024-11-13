"""
Hand-Eye Calibration Using a Chessboard

TODO create the script for hand-eye calibration
"""

import cv2
import numpy as np
import glob
import math3d as m3d
from utils import PoseGenerator
from RobotController.UR import UR5RobotController
from DepthCamera.depth_camera import D435

"""
# Define the dimensions of the chessboard
chessboard_size = (9, 6)  # Inner corners (width x height) of the chessboard pattern
square_size = 1.0  # Size of each square (in consistent units, e.g., centimeters)

# Prepare object points (3D points in chessboard frame)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in world coordinate system
imgpoints = []  # 2D points in image plane

# Load images of the chessboard
images = glob.glob(
    "path_to_chessboard_images/*.jpg"
)  # Update this path with your images

for image_file in images:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, refine corner positions and store points
    if ret:
        objpoints.append(objp)
        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        imgpoints.append(refined_corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, refined_corners, ret)
        cv2.imshow("Chessboard Corners", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration to get camera matrix and distortion coefficients
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Check if camera calibration was successful
if not ret:
    print("Camera calibration failed. Exiting.")
    exit()

# Camera calibration successful
print("Camera calibration was successful!")
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# Placeholder for robot's end-effector poses and camera poses
# Each pose is represented as a (4x4) transformation matrix
# Replace with actual robot poses corresponding to each image
robot_poses = []  # List of robot end-effector transformation matrices (4x4)
camera_poses = []  # List of camera transformation matrices (4x4)

# Generate transformation matrices for each captured pose
for i in range(len(images)):
    # Get rotation and translation from camera calibration for each image
    rvec, tvec = rvecs[i], tvecs[i]
    rot_matrix, _ = cv2.Rodrigues(rvec)

    # Camera pose as transformation matrix
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = rot_matrix
    camera_pose[:3, 3] = tvec.flatten()

    camera_poses.append(camera_pose)

    # Generate or load corresponding robot end-effector pose for each image
    # For example:
    robot_pose = np.eye(4)  # Replace with actual robot end-effector pose
    robot_poses.append(robot_pose)

# Ensure we have matching number of robot and camera poses
assert len(robot_poses) == len(
    camera_poses
), "Mismatch between robot and camera pose counts."

# Hand-eye calibration (robot -> camera transformation)
retval, R_hand_eye, t_hand_eye = cv2.calibrateHandEye(
    robot_poses, camera_poses, method=cv2.CALIB_HAND_EYE_TSAI
)

if retval:
    print("Hand-eye calibration was successful!")
    print("Rotation (R_hand_eye):\n", R_hand_eye)
    print("Translation (t_hand_eye):\n", t_hand_eye)
else:
    print("Hand-eye calibration failed.")
    
"""


def hand_eye_calibration(robot_poses, camera_poses, images):
    """
    Get the transformation from the robot end effector to the in-hand camera using hand-eye calibration.

    robot_poses (np.array): a list of robot end effector poses to take pictures of a chessboard
    camera_poses (np.array): a list of camera poses matching the list of robot end effector poses
    images: a list of images taken from each camera pose

    return T_end2cam (math3d.Transform): the transformation from the robot end effector to the camera
    """

    T_end2cam = m3d.Transform()

    return T_end2cam


def get_camera_pose_with_chessboard(image):
    """
    Get the camera pose regarding a picture of a chessboard captured from a specific view

    image (np.ndarray): the image of a chessboard captured from a specific view

    return pose (np.ndarray of shape (4, 4)): the camera pose represented by a 4x4 transformation matrix
    """

    pose = np.eye(4)

    return pose


def get_chessboard():
    """
    Initalize the chessboard used for getting camera poses

    return chessboard (dictionary): {"chessboard_size", "square_size"}
    """
    pass


if __name__ == "__main__":

    # ROBOT_IP = "192.168.2.144"  # URSim
    ROBOT_IP = "192.168.2.196"  # UR5

    # create a UR5
    UR5 = UR5RobotController(ROBOT_IP)

    poses = []  # TODO fix a list of robot end effector poses here

    # create a depth camera
    DC = D435()
