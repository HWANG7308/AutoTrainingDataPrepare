import os
import random
import json
from pathlib import Path
from utils.utils import get_selection


def make_train_and_test_dataset(
    object_names,
    data_set_type,
    save_name,
    p_val=0.1,
    p_test=0.1,
    randon_seed=100,
):
    """
    Create training and testing datasets from the given object names.

    Parameters:
    object_names (list): List of object names to include in the dataset.
    data_set_type (str): Type of the dataset (e.g., 'segmentation').
    save_name (str): Name to save the dataset.
    p_val (float): Proportion of the dataset to include in the validation split.
    p_test (float): Proportion of the dataset to include in the test split.
    randon_seed (int): Random seed for reproducibility.
    """

    random.seed(randon_seed)

    train_samples = {}
    val_samples = {}
    test_samples = {}

    save_dir = os.path.join(root, "result/datasets", data_set_type)
    os.makedirs(save_dir, exist_ok=True)

    for object_name in object_names:
        color_img_dir = os.path.join(root, "result/acquired_data", object_name, "color")
        num_img = len(os.listdir(color_img_dir))

        num_val = int(p_val * num_img)
        num_test = int(p_test * num_img)
        num_train = num_img - num_val - num_test

        train_list = random.sample(range(num_img), num_train)
        val_list = random.sample(list(set(range(num_img)) - set(train_list)), num_val)
        test_list = list(set(range(num_img)) - set(train_list) - set(val_list))

        train_samples[object_name] = sorted(train_list)
        val_samples[object_name] = sorted(val_list)
        test_samples[object_name] = sorted(test_list)

    print(f"{num_train} samples for training.")
    print(f"{num_val} samples for validation.")
    print(f"{num_test} samples for testing.")

    data_set = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "class": object_names,
    }

    with open(os.path.join(save_dir, f"{save_name}.json"), "w") as f:
        json.dump(data_set, f, indent=4)


def create_dataset(data_set_type):

    data_set_path = os.path.join(root, "result/datasets", data_set_type)
    os.makedirs(data_set_path, exist_ok=True)

    names = [os.path.splitext(file)[0] for file in os.listdir(data_set_path)]

    while True:
        print("\n" + "_" * 70)
        name = input("Enter name of the new data set: ")
        if name in names:
            print(f"Dataset '{name}' already exists. Please choose a different name.")
            continue

        selection = input(
            f"The new data set name is: '{name}'.\nType 'r' to rename, 'b' to return, or hit any other key to continue."
        )
        if selection == "r":
            continue
        elif selection == "b":
            break

        path = os.path.join(root, "result/annotated_data")
        objects = sorted(
            [obj for obj in os.listdir(path) if os.path.isdir(os.path.join(path, obj))]
        )
        while True:
            print("\n" + "_" * 70)
            object_names = get_selection(
                objects + ["all"],
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

            print("\n" + "_" * 70)
            print(
                'Created new "{}" data set "{}", with "{}" objects: '.format(
                    data_set_type, name, len(object_names)
                )
            )
            print(object_names)
            return print("Returning to Main Menu")


def main():

    data_set_types = [
        "2dbbox",
        "3dbbox",
        "6dpose",
        # "img_seg",
    ]

    while True:
        print("\n" + "_" * 70)
        data_set_type = get_selection(data_set_types, "Select the data set type")
        if not data_set_type:
            return print("Returning to Main Menu")
        create_dataset(data_set_type)


if __name__ == "__main__":
    root = str(Path(__file__).resolve().parent)

    main()
