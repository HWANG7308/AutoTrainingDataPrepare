"""
Data annotation functions

Created by Hao Wang
License: MIT License
"""

import os
import json
from pathlib import Path
import time
import datetime
from tqdm import tqdm
from DataAnnotator import Annotator2DBBox, Annotator3DBBox, Annotator6DPose
from utils import get_selection


def save_annotations(data_save_dir, n, annotation):
    """
    Save the annotations to a JSON file.

    Parameters:
    data_save_dir (str): Directory to save the annotations.
    n (int): Index of the current sample.
    annotation (dict): Annotation to save.
    """
    os.makedirs(data_save_dir, exist_ok=True)
    with open(os.path.join(data_save_dir, f"meta_{n:06d}.json"), "w") as f:
        json.dump(annotation, f, indent=4)
    print(f"Data annotation {n} saved in {data_save_dir}")


def create_labels(annotation_type, save_vis=False):
    """
    Create labels for the acquired data.

    Parameters:
    annotation_type (str): Type of annotation ("2dbbox", "6dpose", "3dbbox", "img_seg", "remove_bkg_chroma_key").
    """
    raw_data_dir = os.path.join(root, "result/acquired_data")
    data_save_dir = os.path.join(root, "result/annotated_data")

    names = sorted(os.listdir(raw_data_dir))

    for _, name in enumerate(names):

        color_img_dir = os.path.join(raw_data_dir, name, "color")
        color_imgs = os.listdir(color_img_dir)

        if annotation_type == "3dbbox":
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
            front_annotation = front_annotator.annotate()

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
            top_annotation = top_annotator.annotate()

        for n, color_img in enumerate(
            tqdm(color_imgs, desc=f"Processing data for {name}")
        ):
            color_img_path = os.path.join(color_img_dir, color_img)
            depth_img_path = color_img_path.replace("color", "depth")
            meta_path_ = color_img_path.replace("color", "meta")
            meta_path = os.path.splitext(meta_path_)[0] + ".json"

            if save_vis:
                save_vis_dir = os.path.join(data_save_dir, name, "vis", annotation_type)
                os.makedirs(save_vis_dir, exist_ok=True)

            if annotation_type == "2dbbox":
                annotator = Annotator2DBBox(color_img_path, depth_img_path, meta_path)
                _ = annotator.remove_bkg_chroma_key(save_vis_dir=save_vis_dir)
                annotation = annotator.annotate(save_vis_dir=save_vis_dir)
            elif annotation_type == "remove_bkg_chroma_key":
                annotator = Annotator2DBBox(color_img_path, depth_img_path, meta_path)
                annotation = annotator.remove_bkg_chroma_key(save_vis_dir=save_vis_dir)
                continue
            elif annotation_type == "6dpose":
                annotator = Annotator6DPose(color_img_path, depth_img_path, meta_path)
                annotation = annotator.annotate(save_vis_dir=save_vis_dir)
            elif annotation_type == "3dbbox":
                annotator = Annotator3DBBox(color_img_path, depth_img_path, meta_path)
                _ = annotator.init_front_top_views(front_annotation, top_annotation)
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
                os.path.join(data_save_dir, name, "label", annotation_type),
                n,
                annotation,
            )


def remove_bkg_chroma_key():
    """
    Create background removed data for the acquired data based on chroma key.
    """
    create_labels("remove_bkg_chroma_key", save_vis=True)


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


def create_labels_3dbbox_standalone():  # TODO: remove this function if the one works correctly in create_labels.
    raw_data_dir = os.path.join(root, "result/acquired_data")
    data_save_dir = os.path.join(root, "result/annotated_data")

    names = list(os.listdir(raw_data_dir))

    front_view_id = 0
    top_view_id = 64  # TODO: find a robust method to get the top view images
    annotation_type = "3dbbox"
    save_vis = True

    for _, name in enumerate(names):
        color_img_dir = os.path.join(raw_data_dir, name, "color")
        color_imgs = list(os.listdir(color_img_dir))

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
        front_annotation = front_annotator.annotate()

        # Annotation for top view
        top_color_img_path = os.path.join(color_img_dir, f"color_{top_view_id:06d}.png")
        top_depth_img_path = top_color_img_path.replace("color", "depth")
        top_meta_path_ = top_color_img_path.replace("color", "meta")
        top_meta_path = os.path.splitext(top_meta_path_)[0] + ".json"
        top_annotator = Annotator2DBBox(
            top_color_img_path, top_depth_img_path, top_meta_path
        )
        _ = top_annotator.remove_bkg_chroma_key()
        top_annotation = top_annotator.annotate()

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

            DA_3dbbox = Annotator3DBBox(color_img_path, depth_img_path, meta_path)
            _ = DA_3dbbox.init_top_front_views(front_annotation, top_annotation)
            annotations = DA_3dbbox.annotate(save_vis_dir=save_vis_dir)

            save_annotations(
                os.path.join(data_save_dir, name, annotation_type, "label"),
                n,
                annotations,
            )


def create_labels_img_seg():
    """
    Create segmentation mask labels for the acquired data.
    """
    # create_labels("img_seg", save_vis=True)
    raise NotImplementedError


def main():
    s = {
        "Remove Background (Chroma Key)": remove_bkg_chroma_key,
        "Create Labels (2D BBox)": create_labels_2dbbox,
        "Create Labels (6D Pose)": create_labels_6dpose,
        "Create Labels (3D BBox)": create_labels_3dbbox,
        "Create Labels (Image Segmentation)": create_labels_img_seg,
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
            annotation_start_time = time.time()
            s[selection]()
            annotation_end_time = time.time()
            annotation_time = annotation_end_time - annotation_start_time
            print(f"Annotation time: {annotation_time}")


if __name__ == "__main__":
    root = str(Path(__file__).resolve().parent)

    main()
