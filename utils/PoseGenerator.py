import math
import math3d as m3d
import numpy as np
import json


class PoseGenerator:
    def __init__(
        self,
        T_rob2obj=m3d.Transform(m3d.Orientation(np.eye(3)), m3d.Vector(0, 0, 0)),
        T_end2cam=m3d.Transform(m3d.Orientation(np.eye(3)), m3d.Vector(0, 0, 0)),
        radius=0.3,
        num_azi=16,
        num_polar=5,
    ):
        """
        radius: the radius of the hemisphere in meters, the focal length of D435, >=0.3 m for D435 due to the camera's characteristic
        num_azi: the number of points on the same horizontal plane of the hemisphere
        num_polar: the number of points on half of the vertical plane of the hemisphere
        """
        self.T_rob2obj = T_rob2obj
        self.T_end2cam = T_end2cam
        self.radius = radius
        self.num_azi = num_azi
        self.num_polar = num_polar
        self.theta_step = 2 * math.pi / self.num_azi
        self.phi_step = math.pi / 2 / (self.num_polar - 1)

    def generate_position_example(self, theta=0, phi=0, show_result=False):
        """Create an example end effector position in spherical coordinate system with defined theta (azimuth) and phi (polar)"""
        print(f"Generating the end effector pose with theta: {theta} and phi: {phi}")

        T_obj2cam = self._compute_transform(
            theta, phi
        )  # The transformation from the object to the camera (changing, the camera is moving on the hemisphere with the object located at the center of the hemisphere)
        T_rob2end = self.T_rob2obj * T_obj2cam * self.T_end2cam.inverse
        next_pose = T_rob2end.get_logarithm().get_array()

        print("End effector poses generated!")

        pos_list = [
            {"T_obj2cam": T_obj2cam, "T_rob2end": T_rob2end, "next pose": next_pose}
        ]

        if show_result:
            self.visualization(pos_list)

        return pos_list

    def generate_positions(self, change_first="polar", show_result=False):
        """Generate a list of end effector positions in spherical coordinate system"""
        # TODO: optimize the order of the position list
        print("Generating end effector poses...")

        if change_first == "polar":
            # change polar angle first
            pos_list = [
                self._create_pose(
                    self.theta_step * i,
                    self.phi_step * (j if i % 2 == 0 else self.num_polar - 1 - j),
                )
                for i in range(self.num_azi)
                for j in range(self.num_polar)
            ]
        elif change_first == "azimuth":
            # change azimuth angle first
            pos_list = [
                self._create_pose(
                    self.theta_step * (i if j % 2 == 0 else self.num_azi - 1 - i),
                    self.phi_step * j,
                )
                for j in range(self.num_polar)
                for i in range(self.num_azi)
            ]
        else:
            TypeError("Only polar and azimuth are allowed for change_first")

        print("End effector poses generated!")

        if show_result:
            self.visualization(pos_list)

        return pos_list

    def generate_positions_move_obj(self, show_result=False):
        """Generate a list of end effector positions in spherical coordinate system for rotating the object directly"""
        # TODO This function has not been debugged!!!
        print("Generating end effector poses...")

        pos_list = [
            self._create_pose_move_obj(
                self.theta_step * i,
                self.phi_step * (j if i % 2 == 0 else self.num_polar - 1 - j),
            )
            for i in range(self.num_azi)
            for j in range(self.num_polar)
        ]

        print("End effector poses generated!")

        if show_result:
            self.visualization(pos_list)

        return pos_list

    def generate_positions_hand_eye_calibration(self, show_result=False):
        """Generate a list of end effector positions in spherical coordinate system for hand-eye calibration"""
        # TODO: optimize the order of the position list
        print("Generating end effector poses...")

        pos_list = [
            self._create_pose(
                self.theta_step * i,
                self.phi_step * (j if i % 2 == 0 else 1 - j),
            )
            for i in range(self.num_azi // 2)
            for j in range(2)
        ]

        print("End effector poses generated!")

        if show_result:
            self.visualization(pos_list)

        return pos_list

    def _compute_transform(self, theta, phi):
        """Compute the transformation from object (center of a hemisphere) to camera (a point on the hemisphere)"""
        return m3d.Transform(
            m3d.Orientation.new_euler((theta, -phi, 0), "ZXY"),
            m3d.Vector(
                self.radius * math.sin(phi) * math.sin(theta),
                -self.radius * math.sin(phi) * math.cos(theta),
                -self.radius * math.cos(phi),
            ),
        )

    def _create_pose(self, theta, phi):
        T_obj2cam = self._compute_transform(theta, phi)
        T_rob2end = self.T_rob2obj * T_obj2cam * self.T_end2cam.inverse
        next_pose = T_rob2end.get_logarithm().get_array()
        return {"T_obj2cam": T_obj2cam, "T_rob2end": T_rob2end, "next pose": next_pose}

    def _create_pose_move_obj(self, theta, phi):
        """
        TODO debug this function
        """
        T_end2obj_canonical = m3d.Transform(
            m3d.Orientation.new_rotation_vector((0, 0, 0)),
            m3d.Vector(0, 0, 0.35),
        )
        T_obj2obj_canonical = m3d.Transform(
            m3d.Orientation.new_euler((-theta, -phi, 0), "ZXY"),
            m3d.Vector(0, 0, 0),
        )
        T_rob2end = self.T_rob2obj * T_obj2obj_canonical * T_end2obj_canonical.inverse
        next_pose = T_rob2end.get_logarithm().get_array()
        return {"T_rob2end": T_rob2end, "next pose": next_pose}

    def _optimize_trajectory(self, poses):
        """Find the optimal sequence of defined poses in which the UR5 moves based on a greedy algorithm"""
        raise NotImplementedError

    def visualization(self, pose_list):
        import matplotlib.pyplot as plt

        # Example PoseVector list
        pose_vectors = [pose.get("T_rob2end") for pose in pose_list]
        # pose_vectors.append(self.T_rob2obj)  # Add the object pose for reference

        # Prepare 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Visualize each PoseVector
        positions = []
        for index, pose in enumerate(pose_vectors):
            # Extract position
            position = pose.pos  # Get the translation vector (x, y, z)
            x, y, z = position[0], position[1], position[2]
            positions.append([x, y, z])

            # Extract orientation
            # Use one direction vector to represent orientation (e.g., x-axis of the rotation matrix)
            orientation_matrix = pose.orient.matrix  # Get the 3x3 rotation matrix
            direction_vector = orientation_matrix[
                :, 0
            ]  # Take the x-axis as a representative vector

            # Plot the position as a point
            ax.scatter(x, y, z, c="black", marker=".", s=5)

            # # Plot the orientation as an arrow
            # ax.quiver(
            #     x,
            #     y,
            #     z,  # Starting point of the arrow
            #     direction_vector[0],  # x-component of the direction
            #     direction_vector[1],  # y-component of the direction
            #     direction_vector[2],  # z-component of the direction
            #     length=0.05,  # Scale arrow length (adjustable)
            #     normalize=True,
            #     color="r",
            # )

            # Add index label
            # ax.text(x, y, z, f"{index}", color="black", fontsize=10)

            # Plot x, y, z axes for orientation
            for i in range(3):  # Iterate over the 3 axes
                direction_vector = orientation_matrix[:, i]
                ax.quiver(
                    x,
                    y,
                    z,
                    direction_vector[0],
                    direction_vector[1],
                    direction_vector[2],
                    length=0.03,
                    color=["r", "g", "b"][i],
                    alpha=0.2,
                )

        # Equalize the scale of the axes
        positions = np.array(positions)
        x_limits = [positions[:, 0].min(), positions[:, 0].max()]
        y_limits = [positions[:, 1].min(), positions[:, 1].max()]
        z_limits = [positions[:, 2].min(), positions[:, 2].max()]

        # Find the maximum range for all axes
        max_range = max(
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0],
        )

        # Center the axes and set limits
        x_middle = sum(x_limits) / 2
        y_middle = sum(y_limits) / 2
        z_middle = sum(z_limits) / 2

        ax.set_xlim(x_middle - max_range / 2, x_middle + max_range / 2)
        ax.set_ylim(y_middle - max_range / 2, y_middle + max_range / 2)
        ax.set_zlim(z_middle - max_range / 2, z_middle + max_range / 2)

        # # Set axis labels
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_title("PoseVector Visualization (Equal Scale)")

        # Set the aspect ratio to be equal
        ax.set_box_aspect([1, 1, 1])

        # Show the plot
        # plt.show()

        # Show left view (view from -X axis)
        # ax.view_init(elev=0, azim=180)
        # plt.show()
        # Show front view (view from -Y axis)
        # ax.view_init(elev=0, azim=90)
        # plt.show()
        # Show top view (view from +Z axis)
        # ax.view_init(elev=90, azim=-90)
        # plt.show()

        save_dir = "../doc/cam_pose_group"
        # Save the current 3D plot as a subplot image, cropped to subplot boundaries
        fig.savefig(
            f"{save_dir}/{self.num_azi}_{self.num_polar}.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )


def visualize_robot_position():
    """
    Visualize robot end effector positions generated by PoseGenerator.
    Allows user to specify azimuth, polar, and radius parameters.
    """
    T_rob2obj = m3d.Transform(
        m3d.Orientation.new_euler((math.pi / 2, 0, math.pi), "XYZ"),
        m3d.Vector(0, -0.7, 0),
    )
    T_end2cam = m3d.Transform(
        m3d.Orientation.new_rotation_vector((0, 0, 0)), m3d.Vector(0, 0, 0.05)
    )

    def get_input(prompt, default, cast_type):
        val = input(f"{prompt} (default {default}): ")
        return cast_type(val) if val.strip() else default

    num_azi = get_input("Enter the number of azimuth positions", 16, int)
    num_polar = get_input("Enter the number of polar positions", 5, int)
    radius = get_input("Enter the radius in meters", 0.3, float)

    poses = PoseGenerator(
        T_rob2obj, T_end2cam, radius=radius, num_azi=num_azi, num_polar=num_polar
    ).generate_positions(show_result=True)

    # # Visualize for multiple configurations
    # num_azi_list = [16, 30, 60, 120]
    # num_polar_list = [5, 9, 16, 31]
    # for num_azi in num_azi_list:
    #     for num_polar in num_polar_list:
    #         print(f"Visualizing for num_azi={num_azi}, num_polar={num_polar}")
    #         poses = PoseGenerator(
    #             T_rob2obj, T_end2cam, radius=0.3, num_azi=num_azi, num_polar=num_polar
    #         ).generate_positions(show_result=True)

    return poses


def test_robot_position(save_joints=False, step_check=True):
    """
    Test the robot positions by moving the robot to various end effector positions
    and checking the joint positions.

    The robot positions are generated based on the PoseGenerator class, which creates a set of poses
    for the robot to move to. The robot will then move to each pose and check if the joint positions
    are within a certain threshold of the target joint positions.
    The joint positions can be saved to a JSON file if 'save_joints' is set to True.
    The 'step_check' parameter allows for manual stepping through the poses, where the user can confirm
    whether to proceed to the next pose or return to the initial position.
    """
    from URController import UR5RobotController

    print("--- Testing robot positions...")
    print("- Save joints: ", save_joints)
    print("- Step check: ", step_check)

    UR5 = UR5RobotController(ROBOT_IP, acceleration=1, velocity=1)

    T_rob2obj = m3d.Transform(
        m3d.Orientation.new_euler((math.pi / 2, 0, math.pi), "XYZ"),
        m3d.Vector(0, -0.7, 0),
    )
    T_end2cam = m3d.Transform(
        m3d.Orientation.new_rotation_vector((0, 0, 0)), m3d.Vector(0, 0, 0.05)
    )
    robot_poses = PoseGenerator(
        T_rob2obj,
        T_end2cam,
        # num_azi=120,
        # num_polar=31,
    ).generate_positions(change_first="azimuth")

    print(f"Number of poses: {len(robot_poses)}")

    robot_joints = {} if save_joints else None

    return_init_index = [16, 17, 32, 41, 44, 59, 76]
    try:
        for n, pose in enumerate(robot_poses):
            print("\n" + "_" * 70)

            if step_check:
                while True:
                    if n in return_init_index:
                        UR5.go_init()
                    m = (
                        input(
                            "Go to next position? ['y': yes, 'n': back to init] (default: y) "
                        )
                        or "y"
                    )
                    if m == "y":
                        print("Move on")
                        break
                    elif m == "n":
                        UR5.go_init()

            next_pose = pose.get("next pose")
            print(f"Pose {n}: {next_pose}")

            target_joints = np.degrees(UR5.robot.get_inverse_kin(next_pose))
            print(f"Target robot joints: {target_joints}")

            robot_joint = UR5.move_robot(
                pose=next_pose, check_joint=True, return_joint=save_joints
            )
            print(f"Current robot joints: {robot_joint}")

            if step_check:
                while True:
                    r = input("Redo the move? ['y': yes, 'n': no] (default: n) ") or "n"
                    if r == "y":
                        robot_joint = UR5.move_robot(
                            pose=next_pose, check_joint=True, return_joint=save_joints
                        )
                    elif r == "n":
                        break

            # at_target = np.all(np.abs(np.array(target_joints) - robot_joint) < 0.02)
            # print(f"Robot at target joint positions? {at_target}")
            # while not at_target:
            #     print("Not yet...")
            #     robot_joint = UR5.move_robot(pose=next_pose, return_joint=save_joints)
            #     at_target = np.all(np.abs(np.array(target_joints) - robot_joint) < 0.02)

            # print(f"Robot at target joint positions!")

            if save_joints:
                robot_joints[n] = robot_joint.tolist()

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing robot connection.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        UR5.robot.close()

    if save_joints:
        print("Saving robot joints...")
        with open("result/robot_joints/robot_joints.json", "w") as f:
            json.dump(robot_joints, f, indent=4)
        print("Robot joints saved!")


def test_robot_joint():
    """
    Test the robot joint positions based on the saved joint positions in the JSON file.
    """
    from URController import UR5RobotController

    UR5 = UR5RobotController(ROBOT_IP, acceleration=1, velocity=1)

    with open("result/robot_joints/robot_joints_demo3.json", "r") as f:
        robot_joints = json.load(f)

    try:
        for n in range(len(robot_joints)):
            print(f"Robot joint {n}:", robot_joints.get(str(n)))
            while not UR5.at_target(robot_joints.get(str(n))):
                UR5.move_robot(joint=np.radians(robot_joints.get(str(n))))
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing robot connection.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        UR5.robot.close()


if __name__ == "__main__":

    from utils import get_selection

    # Create a UR5 robot controller
    # ROBOT_IP = "192.168.2.144"  # URSim
    ROBOT_IP = "192.168.2.196"  # UR5

    s = {
        "Visualize robot positions": visualize_robot_position,
        "Test robot positions": test_robot_position,
        "Test robot joints": test_robot_joint,
    }

    while True:
        print("\n" + "_" * 70)
        selection = get_selection(
            list(sorted(s.keys())), "Main Menu", with_exit=True, with_return=False
        )
        if selection == "exit":
            break
        else:
            s[selection]()
