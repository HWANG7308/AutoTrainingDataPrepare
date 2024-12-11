def save_data_sample(data_save_dir, n, name, pose, out, UR5):
    """
    Save the data sample including images and metadata.

    Parameters:
    data_save_dir (str): Directory to save the data.
    n (int): Index of the current sample.
    name (str): Name of the object.
    pose (dict): Pose information.
    out (dict): Output from the depth camera.
    UR5 (UR5RobotController): The robot controller instance.
    """

    save_dirs = {
        "color_img_dir": os.path.join(data_save_dir, "color"),
        "depth_img_dir": os.path.join(data_save_dir, "depth"),
        "meta_info_dir": os.path.join(data_save_dir, "meta"),
    }

    for dir_path in save_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    cv2.imwrite(
        os.path.join(save_dirs["color_img_dir"], f"color_{n:06d}.png"),
        out.get("color"),
    )
    cv2.imwrite(
        os.path.join(save_dirs["depth_img_dir"], f"depth_{n:06d}.png"),
        out.get("depth"),
    )
    meta = {
        "class": name,
        "time": datetime.datetime.today().strftime("%Y-%m-%d, %H:%M:%S"),
        "view_point_id": n,
        "robot_arm_joints": UR5.get_joints().tolist(),
        "object_pose": pose.get("T_obj2cam").inverse.get_matrix().tolist(),
        "tf_rob2end": pose.get("T_rob2end").get_matrix().tolist(),
        "intrinsics_color": out.get("color_intr"),
        "depth_scale": out.get("depth_scale"),
        # "hand_eye_calibration": None  # TODO fix the hand-eye calibration
    }
    with open(os.path.join(save_dirs["meta_info_dir"], f"meta_{n:06d}.json"), "w") as f:
        json.dump(meta, f, indent=4)
    print("Data sample saved!")
