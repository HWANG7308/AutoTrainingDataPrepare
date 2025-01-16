# TODO: create test code in this file (2D object detection, 3D object detection, and 6D object pose estimation)

"""
====================
Testing functions
====================
"""

from utils.utils import get_selection
from pathlib import Path


def prediction(task_type):
    if task_type == "2d_object_detection":
        raise NotImplementedError("Yet to be merged")  # TODO merge the function
    elif task_type == "6d_pose_estimation":
        raise NotImplementedError("Yet to be merged")  # TODO merge the function
    elif task_type == "3d_object_detection":
        raise NotImplementedError("Yet to be merged")  # TODO merge the function
    elif task_type == "img_segmentation":
        raise NotImplementedError("Yet to be merged")  # TODO merge the function
    else:
        raise ValueError(
            "Invalid task type. Use '2d_object_detection', '6d_pose_estimation', '3d_object_detection', or 'img_segmentation'."
        )


def run_live_prediction_2d_obj_detect():
    prediction("2d_object_detection")


def run_live_prediction_6d_pose_estimate():
    prediction("6d_pose_estimation")


def run_live_prediction_3d_obj_detect():
    prediction("3d_object_detection")


def run_live_prediction_img_seg():
    prediction("img_segmentation")


def main():
    s = {
        "Run Live Prediction for 2D Object Detection": run_live_prediction_2d_obj_detect,
        "Run Live Prediction for 6D Pose Estimation": run_live_prediction_6d_pose_estimate,
        "Run Live Prediction for 3D Object Detection": run_live_prediction_3d_obj_detect,
        "Run Live Prediction for Image Segmentation": run_live_prediction_img_seg,
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
