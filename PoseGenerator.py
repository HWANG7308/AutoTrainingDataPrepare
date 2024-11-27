import math
import math3d as m3d
import numpy as np


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

    def generate_positions(self, show_result=False):
        """Generate a list of end effector positions in spherical coordinate system"""
        # TODO: optimize the order of the position list
        print("Generating end effector poses...")

        # change polar angle first
        pos_list = [
            self._create_pose(
                self.theta_step * i,
                self.phi_step * (j if i % 2 == 0 else self.num_polar - 1 - j),
            )
            for i in range(self.num_azi)
            for j in range(self.num_polar)
        ]

        # change azimuth angle first
        # pos_list = [
        #     self._create_pose(
        #         self.theta_step * (i if j % 2 == 0 else self.num_azi - 1 - i),
        #         self.phi_step * j,
        #     )
        #     for j in range(self.num_polar)
        #     for i in range(self.num_azi)
        # ]

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
                self.phi_step * ((j if i % 2 == 0 else 1 - j) + 1),
            )
            for i in range(self.num_azi)
            for j in range(2)
        ]

        print("End effector poses generated!")

        if show_result:
            self.visualization(pos_list)

        return pos_list

    def _compute_transform(self, theta, phi):
        """Compute the transformation from object (center of a hemisphere) to camera (a point on the hemisphere)"""
        return m3d.Transform(
            m3d.Orientation.new_euler((theta, phi, 0), "ZXY"),
            m3d.Vector(
                -self.radius * math.sin(phi) * math.sin(theta),
                self.radius * math.sin(phi) * math.cos(theta),
                -self.radius * math.cos(phi),
            ),
        )

    def _create_pose(self, theta, phi):
        T_obj2cam = self._compute_transform(theta, phi)
        T_rob2end = self.T_rob2obj * T_obj2cam * self.T_end2cam.inverse
        next_pose = T_rob2end.get_logarithm().get_array()
        return {"T_obj2cam": T_obj2cam, "T_rob2end": T_rob2end, "next pose": next_pose}

    def _create_pose_move_obj(self, theta, phi):
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
        pass

    def visualization(self, pose_list):
        import matplotlib.pyplot as plt

        # from mpl_toolkits.mplot3d import Axes3D

        # Example PoseVector list
        pose_vectors = [pose.get("T_rob2end") for pose in pose_list]

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
            ax.scatter(x, y, z, c="b", marker="o")

            # Plot the orientation as an arrow
            ax.quiver(
                x,
                y,
                z,  # Starting point of the arrow
                direction_vector[0],  # x-component of the direction
                direction_vector[1],  # y-component of the direction
                direction_vector[2],  # z-component of the direction
                length=0.05,  # Scale arrow length (adjustable)
                normalize=True,
                color="r",
            )

            # Add index label
            ax.text(x, y, z, f"{index}", color="black", fontsize=10)

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
                    length=0.05,
                    color=["r", "g", "b"][i],
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

        # Set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("PoseVector Visualization (Equal Scale)")

        # Show the plot
        plt.show()


def visualize_robot_position():
    T_rob2obj = m3d.Transform(
        m3d.Orientation.new_rotation_vector((math.pi / 2, 0, 0)),
        m3d.Vector(0, -0.6, 0),
    )
    T_end2cam = m3d.Transform(
        m3d.Orientation.new_rotation_vector((0, 0, 0)), m3d.Vector(0, 0, 0.05)
    )

    # Generate the end effector positions to capture object images from various defined views
    _ = PoseGenerator(T_rob2obj, T_end2cam).generate_positions(show_result=True)


def test_robot_position():
    from URController import UR5RobotController

    # Create a UR5 robot controller
    ROBOT_IP = "192.168.2.144"  # URSim
    UR5 = UR5RobotController(ROBOT_IP)

    T_rob2obj = m3d.Transform(
        m3d.Orientation.new_rotation_vector((math.pi / 2, 0, 0)),
        m3d.Vector(0, -0.6, 0),
    )
    T_end2cam = m3d.Transform(
        m3d.Orientation.new_rotation_vector((0, 0, 0)), m3d.Vector(0, 0, 0.05)
    )

    # Generate the end effector positions to capture object images from various defined views
    robot_poses = PoseGenerator(
        T_rob2obj, T_end2cam  # , num_azi=36, num_polar=36
    ).generate_positions()

    try:
        for _, pose in enumerate(robot_poses):
            next_pose = pose.get("next pose")
            print(f"Moving the robot to {next_pose}...")
            UR5.move_robot(next_pose)
            print("Robot moved to position!")
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing robot connection.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        UR5.robot.close()


if __name__ == "__main__":
    visualize_robot_position()

    # test_robot_position()
