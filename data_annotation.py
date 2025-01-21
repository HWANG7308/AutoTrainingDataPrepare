"""
Data annotation functions for 2D bounding box, 6D pose, 3D bounding box, and image segmentation.

Created by Hao Wang
License: MIT License
"""

import os
import json
from pathlib import Path
import time
import datetime
import numpy as np
from tqdm import tqdm
from utils.DataAnnotator import Annotator2DBBox, Annotator3DBBox, Annotator6DPose
from utils.utils import get_selection


def save_annotations(annotation_type, data_save_dir, n, annotation):
    """
    Save the annotations to a JSON file.

    Parameters:
    annotion_type (str): Type of annotation ("2dbbox", "6dpose", "3dbbox", "img_seg", "remove_bkg_chroma_key").
    data_save_dir (str): Directory to save the annotations.
    n (int): Index of the current sample.
    annotation (dict): Annotation to save.
    """
    os.makedirs(data_save_dir, exist_ok=True)
    with open(os.path.join(data_save_dir, f"{annotation_type}_{n:06d}.json"), "w") as f:
        json.dump(annotation, f, indent=4)
    # print(f"Data annotation {n} saved in {data_save_dir}")


def calculate_T_objc2obj(annotation_front, annotation_top, meta_path):
    """
    Calculate the translation from the center of the object bottom to the object center.
    """

    dist_cam2obj = 0.3  # distance from camera to object bottom

    bbox_front = annotation_front.get("shapes")[0].get("points")
    bbox_top = annotation_top.get("shapes")[0].get("points")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    fx = meta.get("intrinsics_color").get("fx")
    fy = meta.get("intrinsics_color").get("fy")
    ppx = meta.get("intrinsics_color").get("ppx")
    ppy = meta.get("intrinsics_color").get("ppy")

    # TODO: check this calculation, especially the scaling factor (from image pixels to actual distance)
    objc_x = (bbox_front[0][0] + bbox_front[1][0]) / 2
    objc_y = (bbox_front[0][1] + bbox_front[1][1]) / 2
    objc_z = (bbox_top[0][1] + bbox_top[1][1]) / 2

    translation_x = (objc_x - ppx) * dist_cam2obj / fx
    translation_y = (objc_y - ppy) * dist_cam2obj / fy
    translation_z = (objc_z - ppy) * dist_cam2obj / fy

    T_objc2obj = np.array(
        [
            [1, 0, 0, translation_x],
            [0, 1, 0, translation_y],
            [0, 0, 1, translation_z],
            [0, 0, 0, 1],
        ]
    )

    print(f"Translation from object bottom to object center: {T_objc2obj}")

    return T_objc2obj


def create_labels(annotation_type, save_vis=False):
    """
    Create labels for the acquired data.

    Parameters:
    annotation_type (str): Type of annotation ("2dbbox", "6dpose", "3dbbox", "img_seg", "remove_bkg_chroma_key").
    """
    raw_data_dir = os.path.join(root, "result/acquired_data")
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(
            "Please acquire data first. Run 'data_acquisition.py' to acquire data."
        )

    data_save_dir = os.path.join(root, "result/annotated_data")

    names = sorted(os.listdir(raw_data_dir))

    annotation_start_time = time.time()

    for _, name in enumerate(names):

        color_img_dir = os.path.join(raw_data_dir, name, "color")
        color_imgs = sorted(os.listdir(color_img_dir))

        if annotation_type == "3dbbox" or annotation_type == "6dpose":
            front_view_id = 0
            top_view_id = 64  # TODO: find a robust method to get the top view images

            # Annotation for front view
            front_color_img_path = os.path.join(
                color_img_dir, f"color_{front_view_id:06d}.png"
            )
            front_depth_img_path = front_color_img_path.replace("color", "depth")
            front_meta_path_ = front_color_img_path.replace("color", "meta")
            front_meta_path = os.path.splitext(front_meta_path_)[0] + ".json"
            front_annotator = Annotator2DBBox(
                front_color_img_path, front_depth_img_path, front_meta_path
            )
            _ = front_annotator.remove_bkg_chroma_key()
            annotation_front = front_annotator.annotate()

            # Annotation for top view
            top_color_img_path = os.path.join(
                color_img_dir, f"color_{top_view_id:06d}.png"
            )
            top_depth_img_path = top_color_img_path.replace("color", "depth")
            top_meta_path_ = top_color_img_path.replace("color", "meta")
            top_meta_path = os.path.splitext(top_meta_path_)[0] + ".json"
            top_annotator = Annotator2DBBox(
                top_color_img_path, top_depth_img_path, top_meta_path
            )
            _ = top_annotator.remove_bkg_chroma_key()
            annotation_top = top_annotator.annotate()

            T_objc2obj = calculate_T_objc2obj(
                annotation_front, annotation_top, meta_path=front_meta_path
            )

        for n, color_img in enumerate(
            tqdm(color_imgs, desc=f"Processing data for {name}")
        ):
            color_img_path = os.path.join(color_img_dir, color_img)
            depth_img_path = color_img_path.replace("color", "depth")
            meta_path_ = color_img_path.replace("color", "meta")
            meta_path = os.path.splitext(meta_path_)[0] + ".json"

            if save_vis:
                save_vis_dir = os.path.join(data_save_dir, name, "vis")
                os.makedirs(save_vis_dir, exist_ok=True)

            if annotation_type == "2dbbox":
                annotator = Annotator2DBBox(color_img_path, depth_img_path, meta_path)
                _ = annotator.remove_bkg_chroma_key(save_vis_dir=save_vis_dir)
                annotation = annotator.annotate(save_vis_dir=save_vis_dir)
            elif annotation_type == "remove_bkg_chroma_key_grayscale":
                annotator = Annotator2DBBox(color_img_path, depth_img_path, meta_path)
                annotation = annotator.remove_bkg_chroma_key(save_vis_dir=save_vis_dir)
                continue
            elif annotation_type == "remove_bkg_chroma_key_hsv":
                annotator = Annotator2DBBox(color_img_path, depth_img_path, meta_path)
                annotation = annotator.remove_bkg_chroma_key_HSV(
                    save_vis_dir=save_vis_dir
                )
                continue
            elif annotation_type == "6dpose":
                annotator = Annotator6DPose(
                    color_img_path, depth_img_path, meta_path, T_objc2obj
                )
                annotation = annotator.annotate(save_vis_dir=save_vis_dir)
            elif annotation_type == "3dbbox":
                annotator = Annotator3DBBox(
                    color_img_path, depth_img_path, meta_path, T_objc2obj
                )
                _ = annotator.init_front_top_views(annotation_front, annotation_top)
                annotation = annotator.annotate(save_vis_dir=save_vis_dir)
            elif annotation_type == "img_seg":
                raise ValueError("Yet to be merged")  # TODO merge the function
            else:
                raise ValueError(
                    "Invalid annotation type. Use '2dbbox', '6dpose', '3dbbox', 'img_seg', or 'remove_bkg_chroma_key'."
                )

            annotation["class"] = name
            annotation["time"] = (
                datetime.datetime.today().strftime("%Y-%m-%d, %H:%M:%S"),
            )

            save_annotations(
                annotation_type,
                os.path.join(data_save_dir, name, "label", annotation_type),
                n,
                annotation,
            )

    annotation_end_time = time.time()
    annotation_time = annotation_end_time - annotation_start_time
    time_report = {"Annotation time": annotation_time}
    with open(os.path.join(data_save_dir, f"time_{annotation_type}.json"), "w") as f:
        json.dump(time_report, f, indent=4)


def create_labels_2dbbox():
    """
    Create 2D bounding box labels for the acquired data.
    """
    create_labels("2dbbox", save_vis=True)


def create_labels_6dpose():
    """
    Create 6D pose labels for the acquired data.
    """
    create_labels("6dpose", save_vis=True)


def create_labels_3dbbox():
    """
    Create 3D bounding box labels for the acquired data.
    """
    create_labels("3dbbox", save_vis=True)


def create_labels_img_seg():
    """
    Create segmentation mask labels for the acquired data.
    """
    # create_labels("img_seg", save_vis=True)
    raise NotImplementedError


def remove_bkg_chroma_key_grayscale():
    """
    Create background removed data for the acquired data based on chroma key.
    """
    create_labels("remove_bkg_chroma_key_grayscale", save_vis=True)


def remove_bkg_chroma_key_hsv():
    """
    Create background removed data for the acquired data based on chroma key.
    """
    create_labels("remove_bkg_chroma_key_hsv", save_vis=True)


def main():
    s = {
        "Create Labels (2D BBox)": create_labels_2dbbox,
        "Create Labels (3D BBox)": create_labels_3dbbox,
        "Create Labels (6D Pose)": create_labels_6dpose,
        "Create Labels (Image Segmentation)": create_labels_img_seg,
        "Remove Background (Chroma Key, Grayscale)": remove_bkg_chroma_key_grayscale,
        "Remove Background (Chroma Key, HSV)": remove_bkg_chroma_key_hsv,
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
