import os
import numpy as np
from pathlib import Path
from utils.utils import get_selection


def make_train_and_test_dataset(
    object_names,
    data_set_type,
    save_name,
    p_test=0.2,
    mode="pred",
    use_extra_data=False,
):
    """
    Create training and testing datasets from the given object names.

    Parameters:
    object_names (list): List of object names to include in the dataset.
    data_set_type (str): Type of the dataset (e.g., 'segmentation').
    save_name (str): Name to save the dataset.
    p_test (float): Proportion of the dataset to include in the test split.
    mode (str): Mode for the dataset (default is 'pred').
    use_extra_data (bool): Whether to use extra data for training (default is False).
    """

    train_samples = []
    test_samples = []
    extra_train_samples = []

    given_mode = mode

    save_dir = os.path.join(root, "result/data_sets", data_set_type, save_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for object_name in object_names:
        object_path = os.path.join(root, "result/annotated_data", object_name)
        dirs = os.listdir(object_path)
        if "extra" in dirs:
            if data_set_type == "segmentation":
                i = dirs.index("extra")
                del dirs[i]
            else:
                if not use_extra_data:
                    i = dirs.index("extra")
                    del dirs[i]

        for d in dirs:
            dir_path = os.path.join(object_path, d)
            samples = sorted(os.listdir(dir_path))
            if samples:

                if d == "extra":
                    mode = "new_pred"
                else:
                    mode = given_mode

                tag = ".{}.label.png".format(mode)
                l = len(tag)
                samples = [s[:-l] for s in samples if tag in s]

                if d != "extra":
                    step = int(np.round(len(samples) / (len(samples) * p_test), 0))
                    iii = []
                    for i, s in enumerate(samples):
                        if i % step == 0:
                            test_samples.append(os.path.join(object_name, d, s))
                        else:
                            train_samples.append(os.path.join(object_name, d, s))
                        if object_name == "Disk":
                            iii.append(i)
                else:
                    for s in samples:
                        extra_train_samples.append(os.path.join(object_name, d, s))

    print("number of train samples: {}".format(len(train_samples)))
    print("number of train samples: {}".format(len(test_samples)))
    print("number of train samples: {}".format(len(extra_train_samples)))

    with open(os.path.join(save_dir, "train_data_list.txt"), "w") as f:
        for item in train_samples:
            f.write("%s\n" % item)

    with open(os.path.join(save_dir, "test_data_list.txt"), "w") as f:
        for item in test_samples:
            f.write("%s\n" % item)

    if use_extra_data:
        with open(os.path.join(save_dir, "extra_train_data_list.txt"), "w") as f:
            for item in extra_train_samples:
                f.write("%s\n" % item)

    with open(os.path.join(save_dir, "classes.txt"), "w") as f:
        for item in object_names:
            f.write("%s\n" % item)


def create_dataset():

    data_set_types = [
        "2d_obj_detection",
        "3d_obj_detection",
        "6d_obj_pose_estimation",
        "img_segmentation",
    ]

    while True:
        print("____________________________________________________________________")
        data_set_type = get_selection(data_set_types, "Select the data set type")
        if not data_set_type:
            return print("Returning to Main Menu")

        data_set_path = os.path.join(root, "result/data_sets", data_set_type)
        if not os.path.exists(data_set_path):
            os.makedirs(data_set_path)
        names = os.listdir(data_set_path)

        if data_set_type == "2d_obj_detection":
            while True:
                print(
                    "____________________________________________________________________"
                )
                name = input("Enter name of the new data set: ")
                if name in names:
                    print(
                        f"Dataset '{name}' already exists. Please choose a different name."
                    )
                    continue

                selection = input(
                    f"The new data set name is: '{name}'.\nType 'r' to rename, 'b' to return, or hit any other key to continue."
                )
                if selection == "r":
                    continue
                elif selection == "b":
                    break

                path = os.path.join(root, "result/annotated_data")
                objects = sorted(os.listdir(path))
                while True:
                    print(
                        "____________________________________________________________________"
                    )
                    object_names = get_selection(
                        objects.append,
                        "Select objects to include into the new dataset. "
                        '\n Select multiple objects by separating them with a comma. (e.g. "1,2")',
                        multi_selection=True,
                    )
                    if not object_names:
                        break

                    if isinstance(object_names, str):
                        object_names = [object_names]

                    if object_names == ["all"]:
                        object_names = objects

                    make_train_and_test_dataset(object_names, data_set_type, name)
                    print(
                        "____________________________________________________________________"
                    )
                    print(
                        'Created new "{}" data set "{}", with "{}" objects: '.format(
                            data_set_type, name, len(object_names)
                        )
                    )
                    for i, object_name in enumerate(object_names):
                        print("{}   : {}".format(i + 1, object_name))
                    return print("Returning to Main Menu")


if __name__ == "__main__":
    root = str(Path(__file__).resolve().parent)

    create_dataset()
