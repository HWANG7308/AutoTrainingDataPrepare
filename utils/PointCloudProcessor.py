import numpy as np
import open3d as o3d
import os
import json


class PointCloudProcessor:
    def __init__(self, rgb_dir, depth_threshold=300):
        self.rgb_dir = rgb_dir
        self.depth_threshold = depth_threshold
        self.rgbd_images = []
        self.transformation_matrices = []
        self.camera_intrinsics = None

    def load_images_and_metadata(self):
        rgb_files = sorted(os.listdir(self.rgb_dir))

        for rgb_file in rgb_files:
            color_img_path = os.path.join(self.rgb_dir, rgb_file)
            depth_img_path = color_img_path.replace("color", "depth")
            mask_file_path = color_img_path.replace("color", "img_seg").replace(
                ".png", ".json"
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

    def save_point_cloud(self, pcd, filename="reconstructed_point_cloud.ply"):
        o3d.io.write_point_cloud(filename, pcd)


def main():
    processor = PointCloudProcessor(rgb_dir="color")
    processor.load_images_and_metadata()
    pcd = processor.reconstruct_point_cloud()
    processor.save_point_cloud(pcd)


if __name__ == "__main__":
    main()
