import os
import numpy as np
import open3d as o3d
import cv2
import json


class PointCloudProcessor:
    def __init__(self, rgb_path, depth_path, meta_path, bbox_path):
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.meta_path = meta_path
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.intrinsics = self.meta["intrinsics_color"]
        self.depth_scale = self.meta["depth_scale"]
        self.bbox_path = bbox_path

    def load_rgb_depth(self):
        rgb_image = cv2.imread(self.rgb_path)
        depth_image = cv2.imread(self.depth_path, cv2.IMREAD_UNCHANGED)
        return rgb_image, depth_image

    def create_point_cloud(self, rgb_image, depth_image):
        h, w = depth_image.shape
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["ppx"]
        cy = self.intrinsics["ppy"]

        points = []
        colors = []

        for v in range(h):
            for u in range(w):
                z = depth_image[v, u] * self.depth_scale
                if z == 0 or z > 0.4:
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                colors.append(rgb_image[v, u] / 255.0)

        points = np.array(points)
        colors = np.array(colors)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        return point_cloud

    def crop_point_cloud(self, point_cloud):
        with open(self.bbox_path, "r") as f:
            bbox = json.load(f)
        min_bound = bbox["oriented_3dbbox"][0]
        max_bound = bbox["oriented_3dbbox"][1]
        cropped_point_cloud = point_cloud.crop(
            o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        )
        return cropped_point_cloud

    def visualize_point_cloud(self, point_cloud):
        o3d.visualization.draw_geometries([point_cloud])

    def process(self, crop=False):
        rgb_image, depth_image = self.load_rgb_depth()
        point_cloud = self.create_point_cloud(rgb_image, depth_image)
        if not crop:
            self.visualize_point_cloud(point_cloud)
            return
        cropped_point_cloud = self.crop_point_cloud(point_cloud)
        self.visualize_point_cloud(cropped_point_cloud)


if __name__ == "__main__":
    rgb_path = "./result/acquired_data/A0/color/color_000000.png"
    depth_path = "./result/acquired_data/A0/depth/depth_000000.png"
    meta_path = "./result/acquired_data/A0/meta/meta_000000.json"
    bbox_path = "./result/annotated_data/A0/label/3dbbox/3dbbox_000000.json"

    processor = PointCloudProcessor(rgb_path, depth_path, meta_path, bbox_path)
    processor.process()
