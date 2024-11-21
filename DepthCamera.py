"""
Adapt from https://github.com/KochPJ/AutoPoseEstimation/blob/main/depth_camera/DepthCam.py
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time


class D435:
    """
    Intel RealSense depth camera D435
    """
    def __init__(
        self,
        fps=30,
        color_width=640,
        color_height=480,
        depth_width=640,
        depth_height=480,
    ):
        self.color_height = color_height
        self.color_width = color_width
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.fps = fps

        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()
        self.repairing = False

        self.init_pipeline()

    def init_pipeline(self):
        self.config.enable_stream(
            rs.stream.depth,
            self.depth_width,
            self.depth_height,
            rs.format.z16,
            self.fps,
        )
        self.config.enable_stream(
            rs.stream.color,
            self.color_width,
            self.color_height,
            rs.format.bgr8,
            self.fps,
        )
        self.profile = self.pipe.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.color_sensor = self.profile.get_device().query_sensors()[1]
        self.color_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.color_sensor.set_option(rs.option.enable_auto_exposure, False)
        self.color_sensor.set_option(rs.option.exposure, 200.0)
        self.color_sensor.set_option(rs.option.white_balance, 3200.0)

    def stream(
        self,
        fps=30,
        show_color=True,
        show_depth=False,
        show_depth_color=False,
        show_added=False,
    ):
        while True:
            return_depth_colorized = show_depth_color or show_added
            out = self.get_frames(return_depth_colorized=return_depth_colorized)
            color = out.get("color")
            depth = out.get("depth")

            if show_color:
                show = color
            if show_depth:
                show = np.array(depth / 2000 * 255, dtype=np.uint8)
            if show_depth_color:
                show = out.get("depth_colorized")
            if show_added:
                depth_colorized = out.get("depth_colorized")
                show = cv2.addWeighted(color, 0.7, depth_colorized, 0.3, 0)

            cv2.imshow("stream", cv2.cvtColor(show, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(max(0, 1 / fps - (time.time() % (1 / fps))))

        self.pipe.stop()

    def get_frames(
        self, return_intrinsics=False, return_depth_colorized=False, with_repair=True
    ):
        while True:
            success, frames = self.pipe.try_wait_for_frames()
            if success:
                try:
                    frames = self.align.process(frames)
                    out = {"frames": frames}
                    if return_intrinsics:
                        out["color_intr"] = self.get_color_intrinsics()
                        out["depth_intr"] = self.get_depth_intrinsics()
                        out["depth_scale"] = self.get_depth_scale()

                    color = np.array(frames.get_color_frame().get_data())
                    depth = np.array(frames.get_depth_frame().get_data())
                    out["color"] = color
                    out["depth"] = depth

                    if return_depth_colorized:
                        depth_colorized = np.array(
                            self.colorizer.colorize(frames.get_depth_frame()).get_data()
                        )
                        out["depth_colorized"] = depth_colorized

                    if with_repair:
                        self.repairing = False

                    return out

                except Exception as e:
                    print(f"Error processing frames: {e}")

            if with_repair:
                self.repairing = True
                while True:
                    try:
                        self.init_pipeline()
                        break
                    except Exception as e:
                        print(f"Pipeline initialization failed: {e}")
                        time.sleep(1)

    def get_color_intrinsics(self):
        color_intr_ = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        return {
            "width": color_intr_.width,
            "height": color_intr_.height,
            "ppx": color_intr_.ppx,
            "ppy": color_intr_.ppy,
            "fx": color_intr_.fx,
            "fy": color_intr_.fy,
            "coeffs": color_intr_.coeffs,
        }

    def get_depth_intrinsics(self):
        return (
            self.profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )

    def get_depth_scale(self):
        return self.depth_sensor.get_depth_scale()


if __name__ == "__main__":
    DC = D435(fps=15, color_width=1280, color_height=720)
    intr = DC.get_color_intrinsics()
    print(intr)
    print(DC.get_depth_scale() * 1000)
    DC.stream(show_color=True, show_added=True, show_depth=True)
