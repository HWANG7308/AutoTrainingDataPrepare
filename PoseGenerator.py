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

    def generate_position_example(self, theta=0, phi=0):
        """Create an example end-effector position in spherical coordinate system with defined theta (azimuth) and phi (polar)"""
        print(f"Generating the end-effector pose with theta: {theta} and phi: {-phi}")

        T_obj2cam = self._compute_transform(
            theta, phi
        )  # The transformation from the object to the camera (changing, the camera is moving on the hemisphere with the object located at the center of the hemisphere)
        T_rob2end = self.T_rob2obj * T_obj2cam * self.T_end2cam.inverse
        next_pose = T_rob2end.get_logarithm().get_array()

        print("End-effector poses generated!")

        return [
            {"T_obj2cam": T_obj2cam, "T_rob2end": T_rob2end, "next pose": next_pose}
        ]

    def generate_positions(self):
        """Generate a list of end-effector positions in spherical coordinate system"""
        # TODO: optimize the order of the position list
        print("Generating end-effector poses...")

        theta_step = 2 * math.pi / self.num_azi
        phi_step = math.pi / 2 / (self.num_polar - 1)
        pos_list = []

        for i in range(self.num_azi):
            for j in range(self.num_polar):
                theta = theta_step * i
                phi = -phi_step * (j if i % 2 == 0 else self.num_polar - 1 - j)
                T_obj2cam = self._compute_transform(theta, phi)
                T_rob2end = self.T_rob2obj * T_obj2cam * self.T_end2cam.inverse
                next_pose = T_rob2end.get_logarithm().get_array()
                pos_list.append(
                    {
                        "T_obj2cam": T_obj2cam,
                        "T_rob2end": T_rob2end,
                        "next pose": next_pose,
                    }
                )

        print("End-effector poses generated!")
        return pos_list

    def generate_positions_move_obj(self):
        """Generate a list of end-effector positions in spherical coordinate system for rotating the object directly"""
        theta_step = 2 * math.pi / self.num_azi
        phi_step = math.pi / 2 / (self.num_polar - 1)
        pos_list = []

        for i in range(self.num_azi):
            for j in range(self.num_polar):
                theta = theta_step * i
                phi = -phi_step * (j if i % 2 == 0 else self.num_polar - 1 - j)
                T_end2obj_canonical = m3d.Transform(
                    m3d.Orientation.new_rotation_vector((0, 0, 0)),
                    m3d.Vector(0, 0, 0.35),
                )  # transformation from the tcp to the camera (static)
                T_obj2obj_canonical = m3d.Transform(
                    m3d.Orientation.new_euler((-theta, -phi, 0), "ZXY"),
                    m3d.Vector(0, 0, 0),
                )  # transformation from the tcp to the camera (static)
                T_rob2end = (
                    self.T_rob2obj * T_obj2obj_canonical * T_end2obj_canonical.inverse
                )
                next_pose = T_rob2end.get_logarithm().get_array()
                pos_list.append({"T_rob2end": T_rob2end, "next pose": next_pose})

        return pos_list

    def generate_positions_hand_eye_calibration(self):
        """Generate a list of end-effector positions in spherical coordinate system for hand-eye calibration"""
        # TODO: optimize the order of the position list
        print("Generating end-effector poses...")

        theta_step = 2 * math.pi / self.num_azi
        phi_step = math.pi / 2 / (self.num_polar - 1)
        pos_list = []

        for i in range(self.num_azi // 2):
            for j in range(self.num_polar // 2):
                theta = theta_step * i
                phi = -phi_step * (j if i % 2 == 0 else self.num_polar - 1 - j)
                T_obj2cam = self._compute_transform(theta, phi)
                T_rob2end = self.T_rob2obj * T_obj2cam * self.T_end2cam.inverse
                next_pose = T_rob2end.get_logarithm().get_array()
                pos_list.append(
                    {
                        "T_obj2cam": T_obj2cam,
                        "T_rob2end": T_rob2end,
                        "next pose": next_pose,
                    }
                )

        print("End-effector poses generated!")
        return pos_list

    def _compute_transform(self, theta, phi):
        """Compute the transformation from object (center of a hemisphere) to camera (a point on the hemisphere)"""
        return m3d.Transform(
            m3d.Orientation.new_euler((-theta, -phi, 0), "ZXY"),
            m3d.Vector(
                -self.radius * math.sin(phi) * math.sin(theta),
                -self.radius * math.sin(phi) * math.cos(theta),
                -self.radius * math.cos(phi),
            ),
        )


if __name__ == "__main__":
    T_rob2obj = m3d.Transform(
        m3d.Orientation.new_rotation_vector((math.pi / 2, 0, 0)), m3d.Vector(0, -0.6, 0)
    )
    print("The transformation from the robot base to the object:", T_rob2obj)

    T_end2cam = m3d.Transform(
        m3d.Orientation.new_rotation_vector((0, 0, 0)), m3d.Vector(0, 0, 0.05)
    )
    print("The transformation from the end effector to the camera:", T_end2cam)

    poses = PoseGenerator(T_rob2obj, T_end2cam).generate_positions()
    print(poses[0])
