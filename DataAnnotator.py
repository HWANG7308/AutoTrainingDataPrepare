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
    def __init__(self, color_img_path, depth_img_path):
        self.color_img_path = color_img_path
        self.ori_color = cv2.imread(self.color_img_path)
        self.rgb_img = cv2.cvtColor(self.ori_color, cv2.COLOR_BGR2RGB)

        self.depth_img_path = depth_img_path
        # self.ori_depth = cv2.imread(self.depth_img_path)

        self.annotation = {}

    def remove_bkg_chroma_key(self, white_range=(100, 255), show_result=False):
        """extract the object in the foreground based on chroma key"""

        # HSV space
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(self.raw_color_img, cv2.COLOR_BGR2HSV)
        # Define lower and upper bounds for white in HSV space
        lower_white = np.array([42, 12, 88]) #96968F
        upper_white = np.array([96, 50, 250]) #545855
        # Create the mask to detect white color
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # Invert the mask to segment the object
        mask_inv = cv2.bitwise_not(mask)
        # Extract the object
        segmented_object = cv2.bitwise_and(self.raw_color_img, self.raw_color_img, mask=mask_inv)
        """

        rgb_img = self.rgb_img
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, white_range[0], white_range[1], cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask)
        colors_image = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
        rgba_image = cv2.cvtColor(colors_image, cv2.COLOR_RGB2RGBA)
        rgba_image[np.all(rgba_image[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]

        colors_result = rgba_image.copy()  # [:, :, :3].astype(np.uint8)

        if show_result:
            cv2.imshow(
                "Result of background removal based on chroma key", colors_result
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # cv2.imwrite(
            #     "Result of background removal based on chroma key.png", colors_result
            # )

        return colors_result

    """
    def remove_bkg_depth_value(
        self, show_result=False
    ):  # TODO fix this background removal based on depth value
        depth_value = 300
        DEPTH_RANGE = 310

        rgb_image = cv2.cvtColor(self.ori_color, cv2.COLOR_BGR2RGB)

        rgbd_array = np.concatenate(
            (rgb_image, np.expand_dims(depth_value, axis=2)), axis=2
        )
        rgbd_array[
            (rgbd_array[..., 3] < 10) | (rgbd_array[..., 3] > DEPTH_RANGE), :3
        ] = [255, 255, 255]
        new_rgb_image = rgbd_array[:, :, :3].astype(np.uint8)
        rgba_image = cv2.cvtColor(new_rgb_image, cv2.COLOR_RGB2RGBA)
        rgba_image[np.all(rgba_image[:, :, :3] == [255, 255, 255], axis=-1)] = [
            255,
            255,
            255,
            0,
        ]
        depth_result = rgba_image.copy()  # [:, :, :3].astype(np.uint8)
    """

    def annotate(self, show_result=False):
        """draw 2D BBox (rectangle) around the object given the result of chroma_key(raw_rgb_img)"""

        image = self.remove_bkg_chroma_key()

        _, _, _, alpha = cv2.split(image)

        contours, _ = cv2.findContours(
            alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        annotations = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # discard the contour that is too small (potentially noise)
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
                "img_height": self.ori_color.shape[0],
                "img_width": self.ori_color.shape[1],
            }

            annotations.append(annotation)

        self.annotation = annotations[0]

        if show_result:
            self.visualize_2dbbox()

        return annotations

    def visualize_2dbbox(self):
        img = self.rgb_img
        annotation = self.annotation

        shape = annotation.get("shapes")[0]
        [x_min, y_min] = shape.get("points")[0]
        [x_max, y_max] = shape.get("points")[1]

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.imshow("2D BBox annotation", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite("2D BBox annotation.png", img)


class Annotator6DPose:  # TODO fix this annotator
    def __init__(self, color_img_path, depth_img_path, meta_path):

        self.color_img_path = color_img_path
        self.depth_img_path = depth_img_path
        self.meta_path = meta_path

        self.ori_color = cv2.imread(self.color_img_path)
        self.rgb_img = cv2.cvtColor(self.ori_color, cv2.COLOR_BGR2RGB)

        # self.ori_depth = cv2.imread(self.depth_img_path)

        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

        self.object_pose = np.matrix(self.meta.get("object_pose"))

        self.transformation_matrix = np.matrix(self.meta.get("object_pose"))
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

    def annotate(self, show_result=False):
        """generate the 6D pose annotation given meta data"""

        # TODO fix the annotation format here
        annotation = {
            "transformation_matrix": self.transformation_matrix.tolist(),
            "rotation": self.rotation_matrix.tolist(),
            "translation": self.translation_vector.tolist(),
            "img_path": os.path.basename(self.color_img_path),
            "img_height": self.ori_color.shape[0],
            "img_width": self.ori_color.shape[1],
        }

        if show_result:
            self.visualize_6dpose()

        self.annotation = annotation

        return annotation

    def visualize_6dpose(self):
        """
        draw xyz-axis
        """

        color = self.ori_color
        # color = cv2.resize(color, (640, 480))

        vis = self.draw_xyz_axis(
            color,
            ob_in_cam=np.asarray(self.object_pose),
            scale=0.1,
            K=self.cam_K,
            thickness=3,
            transparency=0,
        )

        cv2.imshow("6D pose annotation", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite("6D pose annotation.png", vis)

    def project_3d_to_2d(self, pt, K, ob_in_cam):
        """From FoundationPose: https://github.com/NVlabs/FoundationPose"""

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
        """From FoundationPose: https://github.com/NVlabs/FoundationPose"""

        """
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


class Annotator3DBBox:  # TODO fix this annotator
    def __init__(self, color_img_path, depth_img_path, meta_path):

        self.color_img_path = color_img_path
        self.depth_img_path = depth_img_path
        self.meta_path = meta_path

        self.ori_color = cv2.imread(self.color_img_path)
        self.rgb_img = cv2.cvtColor(self.ori_color, cv2.COLOR_BGR2RGB)

        # self.ori_depth = cv2.imread(self.depth_img_path)

        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

        self.object_pose = np.matrix(self.meta.get("object_pose"))

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

    def reconstruct_oriented_3dbbox(self, annotation_top, annotation_front):
        """generate 3D BBox (cuboid) given the 2D BBoxes of the front and top view"""

        bbox_top = annotation_top.get("shapes")[0].get("points")
        bbox_front = annotation_front.get("shapes")[0].get("points")

        length = bbox_top[1][0] - bbox_top[0][0]
        width = bbox_top[1][1] - bbox_top[0][1]
        height = bbox_front[1][1] - bbox_front[0][1]

        # convert from the pixel space to real-world space
        img_height, img_width, _ = self.ori_color.shape
        dist_cam2obj = 0.3  # the distance between the camera and the object, i.e., radius in PoseGenerator

        length = length / img_width * dist_cam2obj
        width = width / img_height * dist_cam2obj
        height = height / img_height * dist_cam2obj

        extents = np.asarray((length, height, width))  # (width, height, depth)

        oriented_3dbbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        self.annotation = {"oriented_3dbbox": oriented_3dbbox}

        return oriented_3dbbox

    def annotate(self):
        raise NotImplemented

    def visualize_3dbbox(self, oriented_3dbbox):
        """
        visualize 3d bounding box
        """

        color = self.ori_color
        # color = cv2.resize(color, (640, 480))

        vis = self.draw_posed_3d_box(
            img=color,
            ob_in_cam=np.asarray(self.object_pose),
            bbox=oriented_3dbbox,
            K=self.cam_K,
        )

        cv2.imshow("3D bounding box", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite("3D bounding box.png", vis)

    def to_homo(self, pts):
        """
        @pts: (N,3 or 2) will homogeneliaze the last dimension
        """
        assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
        homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
        return homo

    def draw_posed_3d_box(
        self, img, ob_in_cam, bbox, K=np.eye(3), line_color=(0, 255, 255), linewidth=2
    ):
        """Revised from 6pack dataset/inference_dataset_nocs.py::projection
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
            print("uv", uv[0].tolist(), uv[1].tolist())
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


class Annotator_img_seg:  # TODO fix this annotator
    def __init__(self):
        pass

    def annotate(self):
        """transform the chroma-key processed image to segmentation annotation (optional)"""
        raise NotImplemented


def test_2DBBox():
    DA = Annotator2DBBox(color_img_path, depth_img_path)
    DA.remove_bkg_chroma_key(show_result=True)
    DA.annotate(show_result=True)


def test_6DPose():
    DA = Annotator6DPose(color_img_path, depth_img_path, meta_path)
    DA.annotate(show_result=True)


def test_3DBBox():
    DA_2DBBox_top = Annotator2DBBox(color_img_path, depth_img_path)
    DA_2DBBox_top.annotate()
    DA_2DBBox_front = Annotator2DBBox(color_img_path, depth_img_path)
    DA_2DBBox_front.annotate()
    annotation_top = DA_2DBBox_top.annotation
    annotation_front = DA_2DBBox_front.annotation

    DA_3DBBox = Annotator3DBBox(color_img_path, depth_img_path, meta_path)

    oriented_3dbbox = DA_3DBBox.reconstruct_oriented_3dbbox(
        annotation_top, annotation_front
    )

    DA_3DBBox.visualize_3dbbox(oriented_3dbbox)


if __name__ == "__main__":
    color_img_path = "results/acquired_data/test/color_000000.png"
    depth_img_path = None
    meta_path_ = os.path.join(
        os.path.dirname(color_img_path),
        os.path.basename(color_img_path).replace("color", "meta"),
    )
    meta_path = os.path.splitext(meta_path_)[0] + ".json"

    test_2DBBox()

    test_6DPose()

    test_3DBBox()
