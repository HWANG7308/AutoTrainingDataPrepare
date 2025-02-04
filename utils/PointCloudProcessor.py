import numpy as np
import open3d as o3d
import os
import json


class PointCloudProcessor:
    def __init__(self, rgb_dir, depth_threshold=300):
        """
        Attributes:
        rgbd_images (list): List to store RGBD images.
        transformation_matrices (list): List to store transformation matrices.
        camera_intrinsics (o3d.camera.PinholeCameraIntrinsic): Camera intrinsics.
        Initialize the PointCloudProcessor.

        Parameters:
        rgb_dir (str): Directory containing RGB images.
        depth_threshold (int): Threshold for depth values in millimeter. Default is 300.
        """
        self.rgb_dir = rgb_dir
        self.depth_threshold = depth_threshold

        self.rgbd_images = []
        self.transformation_matrices = []
        self.camera_intrinsics = None

        self.load_images_and_metadata()

    def load_images_and_metadata(self):
        obj = self.rgb_dir.split("/")[-2]

        rgb_files = os.listdir(self.rgb_dir)

        for rgb_file in rgb_files:
            color_img_path = os.path.join(self.rgb_dir, rgb_file)
            depth_img_path = color_img_path.replace("color", "depth")
            mask_file_path = (
                color_img_path.replace("acquired_data", "annotated_data")
                .replace(obj, obj + "/label")
                .replace("color", "img_seg")
                .replace(".png", ".json")
            )
            meta_file_path = color_img_path.replace("color", "meta").replace(
                ".png", ".json"
            )
            color = o3d.io.read_image(color_img_path)
            depth = o3d.io.read_image(depth_img_path)
            with open(mask_file_path, "r") as f:
                mask_data = json.load(f)
            mask = np.array(mask_data["seg_mask"], dtype=np.uint8)

            with open(meta_file_path, "r") as f:
                meta_data = json.load(f)

            transformation = np.linalg.inv(np.array(meta_data["object_pose"]))
            self.transformation_matrices.append(transformation)

            # Apply mask to color and depth images
            color_np = np.asarray(color)
            depth_np = np.asarray(depth)
            color_np[mask == 0] = 0
            depth_np[mask == 0] = 0
            depth_np[depth_np > self.depth_threshold] = 0

            color = o3d.geometry.Image(color_np)
            depth = o3d.geometry.Image(depth_np)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth
            )
            self.rgbd_images.append(rgbd_image)

        meta_file_path = (
            os.path.join(self.rgb_dir, rgb_files[0])
            .replace("color", "meta")
            .replace(".png", ".json")
        )
        with open(meta_file_path, "r") as f:
            meta_data = json.load(f)

        # Camera intrinsics
        color_intrinsics = meta_data["intrinsics_color"]
        self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            color_intrinsics["width"],
            color_intrinsics["height"],
            color_intrinsics["fx"],
            color_intrinsics["fy"],
            color_intrinsics["ppx"],
            color_intrinsics["ppy"],
        )

    def reconstruct_point_cloud(self):
        pcd = o3d.geometry.PointCloud()
        for rgbd_image, transformation in zip(
            self.rgbd_images, self.transformation_matrices
        ):
            pcd_partial = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.camera_intrinsics
            )
            pcd_partial.transform(transformation)
            pcd += pcd_partial

        return pcd

    def post_process_point_cloud(
        self,
        pcd,
        voxel_size=1e-3,
        remove_outliers=False,
        nb_neighbors=20,
        std_ratio=2.0,
        smooth=False,
        smooth_max_nn=30,
        smooth_radius=1e-2,
    ):
        # Downsample the point cloud
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

        if remove_outliers:
            # Remove statistical outliers
            _, ind = pcd_down.remove_statistical_outlier(nb_neighbors, std_ratio)
            pcd_down = pcd_down.select_by_index(ind)

        if smooth:
            # Smooth the point cloud using a radius outlier removal
            pcd_down, _ = pcd_down.remove_radius_outlier(
                nb_points=smooth_max_nn, radius=smooth_radius
            )

        return pcd_down

    def save_point_cloud(
        self,
        pcd,
        save_dir="point_clouds",
        filename="reconstructed_point_cloud",
        save_format="ply",
    ):
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, filename + "." + save_format)
        o3d.io.write_point_cloud(filename, pcd)

    def visualize_point_cloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])

    def reconstruct_mesh_from_point_cloud(self, pcd, radius=0.01, max_nn=30, depth=9):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn)
        )
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth)

        return mesh

    def save_mesh(
        self,
        mesh,
        save_dir="meshes",
        filename="reconstructed_mesh",
        save_format="obj",
    ):
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, filename + "." + save_format)
        o3d.io.write_triangle_mesh(filename, mesh)

    def visualize_mesh(self, mesh):
        o3d.visualization.draw_geometries([mesh])


def main():
    processor = PointCloudProcessor(rgb_dir="color")
    pcd = processor.reconstruct_point_cloud()
    processor.save_point_cloud(pcd)


if __name__ == "__main__":
    main()
