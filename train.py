# TODO: create training code in this file (2D object detection, 3D object detection, and 6D object pose estimation)

"""
====================
Training functions
====================
"""

from utils.utils import get_selection
from pathlib import Path


def train(task_type):
    if task_type == "2d_object_detection":
        # TODO: Implement the training function for 2D object detection
        print("Training 2D object detection model...")
        # Add your training code here
        # Example: train_2d_object_detection_model()
        raise NotImplementedError(
            "2d object detection training function is not implemented yet."
        )
    elif task_type == "6d_pose_estimation":
        # Implement the training function for 6D pose estimation
        print("Training 6D pose estimation model...")
        # Add your training code here
        # Example: train_6d_pose_estimation_model()
        raise NotImplementedError(
            "6D pose estimation training function is not implemented yet."
        )
    elif task_type == "img_segmentation":
        # TODO: Implement the training function for image segmentation
        print("Training image segmentation model...")
        # Add your training code here
        # Example: train_img_segmentation_model()
        raise NotImplementedError(
            "Image segmentation training function is not implemented yet."
        )
    elif task_type == "3d_object_detection":
        # Example: train_img_segmentation_model()
        print("Training image segmentation model...")
        # Add your training code here
        # Example: train_3d_object_detection_model()
        raise NotImplementedError(
            "3D object detection training function is not implemented yet."
        )
    else:
        raise ValueError(
            "Invalid task type. Use '2d_object_detection', '6d_pose_estimation', '3d_object_detection', or 'img_segmentation'."
        )


def train_2d_object_detection():
    train("2d_object_detection")


def train_6d_pose_estimation():
    train("6d_pose_estimation")


def train_3d_object_detection():
    train("3d_object_detection")


def train_img_segmentation():
    train("img_segmentation")


def main():
    s = {
        "Train 2D Object Detection Model": train_2d_object_detection,
        "Train 6D Pose Estimation Model": train_6d_pose_estimation,
        "Train 3D Object Detection Model": train_3d_object_detection,
        "Train Image Segmentation Model": train_img_segmentation,
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

    main()
