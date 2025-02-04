"""
1. Load an RGB image
2. Load the corresponding meta data
3. Create 2D object detection annotation
4. Create image segmentation annotation (optional)
5. Create 3D object detection annotation
6. Create 6D object pose estimation annotation
"""

import cv2
import numpy as np
import json
import os


class Annotator2DBBox:
    def __init__(self, color_img_path, depth_img_path, meta_path):
        self.color_img_path = color_img_path
        self.color_img_bgr = cv2.imread(self.color_img_path)
        if self.color_img_bgr is None:
            raise FileNotFoundError(f"Color image not found at {self.color_img_path}")
        self.color_img_rgb = cv2.cvtColor(self.color_img_bgr, cv2.COLOR_BGR2RGB)

        self.depth_img_path = depth_img_path
        self.depth_img = cv2.imread(self.depth_img_path)
        if self.depth_img is None:
            raise FileNotFoundError(f"Depth image not found at {self.depth_img_path}")

        self.meta_path = meta_path
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

        self.depth_scale = self.meta.get("depth_scale")

        self.annotation = {}

    def remove_bkg_chroma_key(
        self,
        white_range=(170, 255),
        show_result=False,
        save_vis_dir=None,
    ):
        """
        Extract the object in the foreground based on chroma key

        Parameters:
        white_range (tuple): the grayscale range to be determined as white background.
        show_result (bool): Indicator of showing the visualization or not.
        save_vis_dir (str): The directory for saving the visualization.
        """

        rgb_img = self.color_img_rgb
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, white_range[0], white_range[1], cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask)
        colors_image = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
        rgba_image = cv2.cvtColor(colors_image, cv2.COLOR_RGB2RGBA)
        rgba_image[np.all(rgba_image[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]

        if show_result:
            cv2.imshow(
                "Result of background removal based on chroma key (grayscale)",
                rgba_image,
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_vis_dir is not None:
            os.makedirs(
                os.path.join(save_vis_dir, "remove_bkg_chroma_key_grayscale"),
                exist_ok=True,
            )
            filename = os.path.basename(self.color_img_path).replace(
                "color", "remove_bkg_chroma_key"
            )
            cv2.imwrite(
                os.path.join(save_vis_dir, "remove_bkg_chroma_key_grayscale", filename),
                cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGB),
            )

        return rgba_image

    def remove_bkg_chroma_key_HSV(
        self,
        lower_white=np.array([0, 0, 168]),
        upper_white=np.array([172, 111, 255]),
        show_result=False,
        save_vis_dir=None,
    ):
        """
        Extract the object in the foreground based on chroma key

        Parameters:
        show_result (bool): Indicator of showing the visualization or not.
        save_vis_dir (str): The directory for saving the visualization.
        """

        rgb_img = self.color_img_rgb
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        mask_inv = cv2.bitwise_not(mask)
        mask_inv_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        colors_image = cv2.bitwise_and(rgb_img, mask_inv_3ch)
        rgba_image = cv2.cvtColor(colors_image, cv2.COLOR_RGB2RGBA)
        rgba_image[np.all(rgba_image[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]

        if show_result:
            cv2.imshow(
                "Result of background removal based on chroma key (HSV)", rgba_image
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_vis_dir is not None:
            os.makedirs(
                os.path.join(save_vis_dir, "remove_bkg_chroma_key_hsv"),
                exist_ok=True,
            )
            filename = os.path.basename(self.color_img_path).replace(
                "color", "remove_bkg_chroma_key_hsv"
            )
            cv2.imwrite(
                os.path.join(save_vis_dir, "remove_bkg_chroma_key_hsv", filename),
                cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGB),
            )

        return rgba_image

    def remove_bkg_depth_value(
        self,
        clipping_distance_in_meters=0.3,
        show_result=False,
        save_vis_dir=None,
    ):
        """
        Remove background based on depth value
        We will be removing the background of objects more than clipping_distance_in_meters meters away
        TODO This method is not robust enough with small objects like connector terminals, fix this background removal based on depth value
        """

        color_image = self.color_img_bgr
        depth_img = self.depth_img

        clipping_distance = clipping_distance_in_meters / self.depth_scale

        gray_color = 255
        bg_removed = np.where(
            (depth_img > clipping_distance) | (depth_img <= 0),
            gray_color,
            color_image,
        )

        if show_result:
            cv2.imshow("Result of background removal based on depth value", bg_removed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_vis_dir is not None:
            os.makedirs(
                os.path.join(save_vis_dir, "remove_bkg_depth_value"),
                exist_ok=True,
            )
            filename = os.path.basename(self.color_img_path).replace(
                "color", "remove_bkg_depth_value"
            )
            cv2.imwrite(
                os.path.join(save_vis_dir, "remove_bkg_depth_value", filename),
                cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB),
            )

        return bg_removed

    def annotate(self, show_result=False, save_vis_dir=None, save_mask=False):
        """Draw 2D BBox (rectangle) around the object given the result of chroma_key(raw_rgb_img)"""
        rgba_image = self.remove_bkg_chroma_key()

        _, _, _, alpha = cv2.split(rgba_image)
        contours, _ = cv2.findContours(
            alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        annotations = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Discard the contour that is too small (potentially noise)
            if w * h < 500:
                continue

            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h

            annotation = {
                "shapes": [
                    {
                        "points": [[x_min, y_min], [x_max, y_max]],
                        "shape_type": "rectangle",
                    }
                ],
                "img_path": os.path.basename(self.color_img_path),
                "img_height": self.color_img_bgr.shape[0],
                "img_width": self.color_img_bgr.shape[1],
            }

            annotations.append(annotation)

        if annotations:
            self.annotation = annotations[0]

        vis = self.visualize_2dbbox(show_result)

        if save_vis_dir is not None:
            os.makedirs(
                os.path.join(save_vis_dir, "2dbbox"),
                exist_ok=True,
            )
            filename = os.path.basename(self.color_img_path).replace("color", "2dbbox")
            cv2.imwrite(
                os.path.join(save_vis_dir, "2dbbox", filename),
                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
            )

        if save_mask:
            self.convert_to_bw(rgba_image, show_result, save_vis_dir)

        return self.annotation

    def visualize_2dbbox(self, show=False):
        img = self.color_img_rgb.copy()
        annotation = self.annotation

        if not annotation:
            raise ValueError("No annotation found")

        shape = annotation.get("shapes")[0]
        [x_min, y_min] = shape.get("points")[0]
        [x_max, y_max] = shape.get("points")[1]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        if show:
            cv2.imshow("2D BBox annotation", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img

    def convert_to_bw(self, rgba_image, show_result=False, save_vis_dir=None):
        """
        Convert the result of remove_bkg_chroma_key into a black-white image

        Parameters:
        rgba_image (numpy.ndarray): The RGBA image from remove_bkg_chroma_key

        Returns:
        bw_image (numpy.ndarray): The black-white image
        """
        # Create a mask where the alpha channel is greater than 0
        mask = rgba_image[:, :, 3] > 0

        # Create a black-white image
        bw_image = np.zeros((rgba_image.shape[0], rgba_image.shape[1]), dtype=np.uint8)
        bw_image[mask] = 255

        # Apply the bounding box annotation mask
        if self.annotation:
            shape = self.annotation.get("shapes")[0]
            [x_min, y_min] = shape.get("points")[0]
            [x_max, y_max] = shape.get("points")[1]
            bbox_mask = np.zeros_like(bw_image, dtype=bool)
            bbox_mask[y_min:y_max, x_min:x_max] = True
            bw_image[~bbox_mask] = 0

        if show_result:
            cv2.imshow("Black-white image", bw_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_vis_dir is not None:
            os.makedirs(
                os.path.join(save_vis_dir, "seg_mask"),
                exist_ok=True,
            )
            filename = os.path.basename(self.color_img_path).replace(
                "color", "seg_mask"
            )
            cv2.imwrite(
                os.path.join(save_vis_dir, "seg_mask", filename),
                bw_image,
            )

        return bw_image


class Annotator6DPose:
    def __init__(self, color_img_path, depth_img_path, meta_path, T_obj2objc=np.eye(4)):
        self.color_img_path = color_img_path
        self.depth_img_path = depth_img_path
        self.meta_path = meta_path
        self.Tobj2objc = T_obj2objc

        self.color_img_bgr = cv2.imread(self.color_img_path)
        if self.color_img_bgr is None:
            raise FileNotFoundError(f"Color image not found at {self.color_img_path}")
        self.color_img_rgb = cv2.cvtColor(self.color_img_bgr, cv2.COLOR_BGR2RGB)

        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

        self.object_pose = np.array(self.meta.get("object_pose")) @ self.Tobj2objc

        self.transformation_matrix = self.object_pose
        self.rotation_matrix = self.object_pose[:3, :3]
        self.translation_vector = self.object_pose[:3, 3]

        self.cam_K = np.array(
            [
                [
                    self.meta.get("intrinsics_color").get("fx"),
                    0,
                    self.meta.get("intrinsics_color").get("ppx"),
                ],
                [
                    0,
                    self.meta.get("intrinsics_color").get("fy"),
                    self.meta.get("intrinsics_color").get("ppy"),
                ],
                [0, 0, 1],
            ]
        )

        self.annotation = {}

    def annotate(self, show_result=False, save_vis_dir=None):
        """
        Generate the 6D pose annotation given meta data
        """
        annotation = {
            "transformation_matrix": self.transformation_matrix.tolist(),
            "rotation": self.rotation_matrix.tolist(),
            "translation": self.translation_vector.tolist(),
            "img_path": os.path.basename(self.color_img_path),
            "img_height": self.color_img_bgr.shape[0],
            "img_width": self.color_img_bgr.shape[1],
        }  # TODO: fix the annotation format

        self.annotation = annotation

        vis = self.visualize_6dpose(show_result)

        if save_vis_dir is not None:
            os.makedirs(
                os.path.join(save_vis_dir, "6dpose"),
                exist_ok=True,
            )
            filename = os.path.basename(self.color_img_path).replace("color", "6dpose")
            cv2.imwrite(
                os.path.join(save_vis_dir, "6dpose", filename),
                vis,
            )

        return self.annotation

    def visualize_6dpose(self, show=False):
        """Draw xyz-axis"""
        color = self.color_img_bgr.copy()

        vis = self.draw_xyz_axis(
            color,
            ob_in_cam=np.asarray(self.object_pose),
            scale=0.02,
            K=self.cam_K,
            thickness=3,
            transparency=0,
        )

        if show:
            cv2.imshow("6D pose annotation", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis

    def project_3d_to_2d(self, pt, K, ob_in_cam):
        """
        Project 3D point to 2D
        From FoundationPose: https://github.com/NVlabs/FoundationPose
        """
        pt = pt.reshape(4, 1)
        projected = K @ ((ob_in_cam @ pt)[:3, :])
        projected = projected.reshape(-1)
        projected = projected / projected[2]
        return projected.reshape(-1)[:2].round().astype(int)

    def draw_xyz_axis(
        self,
        color_img,
        ob_in_cam,
        scale=0.1,
        K=np.eye(3),
        thickness=3,
        transparency=0,
        is_input_rgb=False,
    ):
        """
        Draw xyz-axis on the image
        From FoundationPose: https://github.com/NVlabs/FoundationPose
        @color_img: BGR
        """
        if is_input_rgb:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        xx = np.array([1, 0, 0, 1]).astype(float)
        yy = np.array([0, 1, 0, 1]).astype(float)
        zz = np.array([0, 0, 1, 1]).astype(float)
        xx[:3] = xx[:3] * scale
        yy[:3] = yy[:3] * scale
        zz[:3] = zz[:3] * scale
        origin = tuple(self.project_3d_to_2d(np.array([0, 0, 0, 1]), K, ob_in_cam))
        xx = tuple(self.project_3d_to_2d(xx, K, ob_in_cam))
        yy = tuple(self.project_3d_to_2d(yy, K, ob_in_cam))
        zz = tuple(self.project_3d_to_2d(zz, K, ob_in_cam))
        line_type = cv2.LINE_AA
        arrow_len = 0
        tmp = color_img.copy()
        tmp1 = tmp.copy()
        tmp1 = cv2.arrowedLine(
            tmp1,
            origin,
            xx,
            color=(0, 0, 255),
            thickness=thickness,
            line_type=line_type,
            tipLength=arrow_len,
        )
        mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
        tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
        tmp1 = tmp.copy()
        tmp1 = cv2.arrowedLine(
            tmp1,
            origin,
            yy,
            color=(0, 255, 0),
            thickness=thickness,
            line_type=line_type,
            tipLength=arrow_len,
        )
        mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
        tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
        tmp1 = tmp.copy()
        tmp1 = cv2.arrowedLine(
            tmp1,
            origin,
            zz,
            color=(255, 0, 0),
            thickness=thickness,
            line_type=line_type,
            tipLength=arrow_len,
        )
        mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
        tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
        tmp = tmp.astype(np.uint8)
        if is_input_rgb:
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

        return tmp


class Annotator3DBBox:
    def __init__(self, color_img_path, depth_img_path, meta_path, T_obj2objc=np.eye(4)):
        self.color_img_path = color_img_path
        self.depth_img_path = depth_img_path
        self.meta_path = meta_path
        self.T_obj2objc = T_obj2objc

        self.color_img_bgr = cv2.imread(self.color_img_path)
        if self.color_img_bgr is None:
            raise FileNotFoundError(f"Color image not found at {self.color_img_path}")
        self.color_img_rgb = cv2.cvtColor(self.color_img_bgr, cv2.COLOR_BGR2RGB)

        self.annotation_top = None
        self.annotation_front = None

        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

        self.object_pose = np.array(self.meta.get("object_pose")) @ self.T_obj2objc

        self.cam_K = np.array(
            [
                [
                    self.meta.get("intrinsics_color").get("fx"),
                    0,
                    self.meta.get("intrinsics_color").get("ppx"),
                ],
                [
                    0,
                    self.meta.get("intrinsics_color").get("fy"),
                    self.meta.get("intrinsics_color").get("ppy"),
                ],
                [0, 0, 1],
            ]
        )

        self.annotation = {}

    def init_front_top_views(self, annotation_front, annotation_top):
        self.annotation_front = annotation_front
        self.annotation_top = annotation_top
        return 1

    def reconstruct_oriented_3dbbox(self, annotation_front, annotation_top):
        """Generate 3D BBox (cuboid) given the 2D BBoxes of the front and top view"""
        bbox_front = annotation_front.get("shapes")[0].get("points")
        bbox_top = annotation_top.get("shapes")[0].get("points")

        cuboid_width = bbox_top[1][0] - bbox_top[0][0]
        cuboid_height = bbox_front[1][1] - bbox_front[0][1]
        cuboid_depth = bbox_top[1][1] - bbox_top[0][1]

        # Convert from the pixel space to real-world space
        dist_cam2obj = 0.3  # The distance between the camera and the object, i.e., radius in PoseGenerator
        fx = self.meta.get("intrinsics_color").get("fx")
        fy = self.meta.get("intrinsics_color").get("fy")

        cuboid_width = cuboid_width * dist_cam2obj / fx
        cuboid_height = cuboid_height * dist_cam2obj / fy
        cuboid_depth = cuboid_depth * dist_cam2obj / fy

        extents = np.asarray((cuboid_width, cuboid_height, cuboid_depth))

        oriented_3dbbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        return oriented_3dbbox

    def annotate(self, show_result=False, save_vis_dir=None):

        oriented_3dbbox = self.reconstruct_oriented_3dbbox(
            self.annotation_front, self.annotation_top
        )

        vis = self.visualize_3dbbox(oriented_3dbbox, show_result)

        annotation = {
            "oriented_3dbbox": oriented_3dbbox.tolist(),
            "ob_in_cam": np.asarray(self.object_pose).tolist(),
            "img_path": os.path.basename(self.color_img_path),
            "img_height": self.color_img_bgr.shape[0],
            "img_width": self.color_img_bgr.shape[1],
        }  # TODO: fix the annotation format

        self.annotation = annotation

        if save_vis_dir is not None:
            os.makedirs(
                os.path.join(save_vis_dir, "3dbbox"),
                exist_ok=True,
            )
            filename = os.path.basename(self.color_img_path).replace("color", "3dbbox")
            cv2.imwrite(
                os.path.join(save_vis_dir, "3dbbox", filename),
                vis,
            )

        return self.annotation

    def visualize_3dbbox(self, oriented_3dbbox, show=False):
        """Visualize 3D bounding box"""
        color = self.color_img_bgr.copy()

        vis = self.draw_posed_3d_box(
            img=color,
            ob_in_cam=np.asarray(self.object_pose),
            bbox=oriented_3dbbox,
            K=self.cam_K,
        )

        if show:
            cv2.imshow("3D bounding box", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis

    def to_homo(self, pts):
        """
        Convert points to homogeneous coordinates
        From FoundationPose: https://github.com/NVlabs/FoundationPose
        @pts: (N,3 or 2) will homogeneliaze the last dimension
        """
        assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
        homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
        return homo

    def draw_posed_3d_box(
        self, img, ob_in_cam, bbox, K=np.eye(3), line_color=(0, 255, 255), linewidth=2
    ):
        """
        Draw 3D bounding box on the image
        From FoundationPose: https://github.com/NVlabs/FoundationPose
        Revised from 6pack dataset/inference_dataset_nocs.py::projection
        @bbox: (2,3) min/max
        @line_color: BGR
        """
        min_xyz = bbox.min(axis=0)
        xmin, ymin, zmin = min_xyz
        max_xyz = bbox.max(axis=0)
        xmax, ymax, zmax = max_xyz

        def draw_line3d(start, end, img):
            pts = np.stack((start, end), axis=0).reshape(-1, 3)
            pts = (ob_in_cam @ self.to_homo(pts).T).T[:, :3]  # (2,3)
            projected = (K @ pts.T).T
            uv = np.round(projected[:, :2] / projected[:, 2].reshape(-1, 1)).astype(
                int
            )  # (2,2)
            # print("uv", uv[0].tolist(), uv[1].tolist())
            img = cv2.line(
                img,
                uv[0].tolist(),
                uv[1].tolist(),
                color=line_color,
                thickness=linewidth,
                lineType=cv2.LINE_AA,
            )
            return img

        for y in [ymin, ymax]:
            for z in [zmin, zmax]:
                start = np.array([xmin, y, z])
                end = start + np.array([xmax - xmin, 0, 0])
                img = draw_line3d(start, end, img)

        for x in [xmin, xmax]:
            for z in [zmin, zmax]:
                start = np.array([x, ymin, z])
                end = start + np.array([0, ymax - ymin, 0])
                img = draw_line3d(start, end, img)

        for x in [xmin, xmax]:
            for y in [ymin, ymax]:
                start = np.array([x, y, zmin])
                end = start + np.array([0, 0, zmax - zmin])
                img = draw_line3d(start, end, img)

        return img


class AnnotatorImgSeg:
    def __init__(self, color_img_path, depth_img_path, meta_path):
        self.color_img_path = color_img_path
        self.color_img_bgr = cv2.imread(self.color_img_path)
        if self.color_img_bgr is None:
            raise FileNotFoundError(f"Color image not found at {self.color_img_path}")
        self.color_img_rgb = cv2.cvtColor(self.color_img_bgr, cv2.COLOR_BGR2RGB)

        self.depth_img_path = depth_img_path
        self.depth_img = cv2.imread(self.depth_img_path)
        if self.depth_img is None:
            raise FileNotFoundError(f"Depth image not found at {self.depth_img_path}")

        self.meta_path = meta_path
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

        self.depth_scale = self.meta.get("depth_scale")

        self.annotation = {}

        self.annotator_2dbbox = Annotator2DBBox(
            color_img_path, depth_img_path, meta_path
        )

    def annotate(self, show_result=False, save_vis_dir=None):
        """
        Transform the chroma-key processed image to segmentation annotation (optional)
        TODO fix this annotator
        """

        rgba_image = self.annotator_2dbbox.remove_bkg_chroma_key()
        seg_mask = self.annotator_2dbbox.convert_to_bw(
            rgba_image, show_result, save_vis_dir
        )

        self.annotation = {
            "seg_mask": seg_mask.tolist(),
            "img_path": os.path.basename(self.color_img_path),
            "img_height": self.color_img_bgr.shape[0],
            "img_width": self.color_img_bgr.shape[1],
        }

        if save_vis_dir is not None:
            os.makedirs(
                os.path.join(save_vis_dir, "mask"),
                exist_ok=True,
            )
            filename = os.path.basename(self.color_img_path).replace("color", "mask")
            cv2.imwrite(
                os.path.join(save_vis_dir, "mask", filename),
                seg_mask,
            )

        if show_result:
            cv2.imshow("Segmentation mask", seg_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return self.annotation


def test_2DBBox():
    annotator = Annotator2DBBox(color_img_path, depth_img_path, meta_path)
    _ = annotator.remove_bkg_chroma_key(show_result=True)
    _ = annotator.remove_bkg_depth_value(show_result=True)
    annotations = annotator.annotate(show_result=True)
    print(annotations)


def test_6DPose():
    annotator = Annotator6DPose(color_img_path, depth_img_path, meta_path)
    annotation = annotator.annotate(show_result=True)
    print(annotation)


def test_3DBBox():
    DA_2DBBox_top = Annotator2DBBox(color_img_path, depth_img_path)
    _ = DA_2DBBox_top.annotate()
    DA_2DBBox_front = Annotator2DBBox(color_img_path, depth_img_path)
    _ = DA_2DBBox_front.annotate()
    annotation_top = DA_2DBBox_top.annotation
    annotation_front = DA_2DBBox_front.annotation

    DA_3DBBox = Annotator3DBBox(color_img_path, depth_img_path, meta_path)

    oriented_3dbbox = DA_3DBBox.reconstruct_oriented_3dbbox(
        annotation_top, annotation_front
    )

    DA_3DBBox.visualize_3dbbox(oriented_3dbbox)


def test_bkg_remove(method):
    """
    Test background removal functions
    """
    annotator = Annotator2DBBox(color_img_path, depth_img_path, meta_path)

    if method == "grayscale":
        _ = annotator.remove_bkg_chroma_key(show_result=True)
    elif method == "HSV":
        _ = annotator.remove_bkg_chroma_key_HSV(show_result=True)
    elif method == "depth":
        _ = annotator.remove_bkg_depth_value(show_result=True)
    else:
        raise ValueError("Invalid method")


if __name__ == "__main__":
    color_img_path = "doc/acquired_data/A0/color/color_000000.png"
    depth_img_path = color_img_path.replace("color", "depth")
    meta_path_ = color_img_path.replace("color", "meta")
    meta_path = os.path.splitext(meta_path_)[0] + ".json"

    test_bkg_remove("grayscale")
    test_bkg_remove("HSV")
