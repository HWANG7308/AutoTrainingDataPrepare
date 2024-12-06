import os
import json
from pathlib import Path


def save_annotations(data_save_dir, n, annotations):
    """
    Save the annotations to a JSON file.

    Parameters:
    data_save_dir (str): Directory to save the annotations.
    n (int): Index of the current sample.
    annotations (dict): Annotations to save.
    """
    with open(os.path.join(data_save_dir, f"meta_{n:06d}.json"), "w") as f:
        json.dump(annotations, f, indent=4)
    print("Data annotation saved!")


def create_labels(annotation_type):
    """
    Create labels for the acquired data.

    Parameters:
    annotation_type (str): Type of annotation ('2dbbox' or '6dpose').
    """
    raw_data_dir = os.path.join(root, "result/acquired_data")
    data_save_dir = os.path.join(root, "result/annotated_data")
    os.makedirs(data_save_dir, exist_ok=True)

    names = list(os.listdir(raw_data_dir))

    for n, name in enumerate(names):
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

        if annotation_type == "2dbbox":
            annotator = Annotator2DBBox(color_img_path, depth_img_path, meta_path)
            _ = annotator.remove_bkg_chroma_key(show_result=True)
            annotations = annotator.annotate(show_result=True)
        elif annotation_type == "6dpose":
            annotator = Annotator6DPose(color_img_path, depth_img_path, meta_path)
            annotations = annotator.annotate(show_result=True)
        else:
            raise ValueError("Invalid annotation type. Use '2dbbox' or '6dpose'.")

        save_annotations(data_save_dir, n, annotations)


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
