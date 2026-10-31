import threading
import time

import cv2
import numpy as np
import PIL
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

from scripts.model import AutopilotNet
from scripts.dataset import preprocessing_transform


class AutopilotDiffDrive(Node):
    """
    Differential-drive version of the autopilot node.

    Subscribes to camera images, runs the network in a separate thread (older frames are dropped) and
    publishes a geometry_msgs/Twist on /cmd_vel. The network outputs two normalised values,
    [velocity, angular velocity], which are scaled by max_linear_velocity / max_angular_velocity
    (these MUST equal the values the training dataset was normalised with, see distill_meta.json).

    Extra reflex for sharp corners: when the line is not visible at all (the next branch of a 90 degree
    corner is out of view, or the robot was knocked off the line) the robot rotates in place towards
    the side it was last turning to until the line re-appears.
    """

    def __init__(self):
        super().__init__("autopilot_diffdrive")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("image_width", 128)
        self.declare_parameter("image_height", 96)
        self.declare_parameter("model_path", "/tmp/diffdrivebot/diffdrive_model.pt")
        self.declare_parameter("max_linear_velocity", 0.4)
        self.declare_parameter("max_angular_velocity", 2.0)
        self.declare_parameter("smoothing", 0.0)                  # 0 = off, 0.5 = average with previous command
        self.declare_parameter("lost_line_search", True)
        self.declare_parameter("lost_line_min_fraction", 0.002)   # same default as make_distill_dataset.py --min-line-frac
        self.declare_parameter("search_angular_velocity", 1.0)
        self.declare_parameter("search_linear_velocity", 0.0)

        p = self.get_parameter
        self.image_topic = p("image_topic").value
        self.cmd_vel_topic = p("cmd_vel_topic").value
        self.image_width = p("image_width").value
        self.image_height = p("image_height").value
        self.model_path = p("model_path").value
        self.max_linear = float(p("max_linear_velocity").value)
        self.max_angular = float(p("max_angular_velocity").value)
        self.smoothing = float(p("smoothing").value)
        self.lost_line_search = bool(p("lost_line_search").value)
        self.lost_min_fraction = float(p("lost_line_min_fraction").value)
        self.search_angular = float(p("search_angular_velocity").value)
        self.search_linear = float(p("search_linear_velocity").value)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Loading model from: {self.model_path}")
        self.model = AutopilotNet()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model.to(self.device)

        self.bridge = CvBridge()
        self.transform = preprocessing_transform(self.image_height, self.image_width)

        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                 history=QoSHistoryPolicy.KEEP_LAST,
                                 depth=1)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 1)
        self.create_subscription(Image, self.image_topic, self._image_callback, qos_profile)

        self.prev_v = 0.0
        self.prev_w = 0.0
        self.search_dir = 1.0          # direction of the last clear turn (+1 = left)

        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

        self.get_logger().info("AutopilotDiffDrive initialized")

    def _image_callback(self, msg: Image):
        with self.lock:
            self.latest_frame = msg

    def _inference_loop(self):
        while self.running and rclpy.ok():
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame
                    self.latest_frame = None
            if frame is not None:
                self._process_frame(frame)
            else:
                time.sleep(0.005)

    @staticmethod
    def _line_fraction(gray, rel=0.6, min_contrast=25):
        """Fraction of clearly-dark pixels (the line). Keep identical to distill_lib.line_pixel_fraction."""
        med = float(np.median(gray))
        g = gray.astype(np.float32)
        return float(((g < rel * med) & ((med - g) > min_contrast)).mean())

    @staticmethod
    def _line_side(gray, rel=0.6, min_contrast=25):
        """Where the line is in the lower part of the image: -1 (left) .. +1 (right), None if hardly any line."""
        med = float(np.median(gray))
        g = gray.astype(np.float32)
        m = (g < rel * med) & ((med - g) > min_contrast)
        low = m[int(0.65 * m.shape[0]):, :]
        if low.sum() < 20:
            return None
        xs = np.nonzero(low)[1]
        return float((xs.mean() - 0.5 * m.shape[1]) / (0.5 * m.shape[1]))

    def _publish(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _process_frame(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        side = self._line_side(gray)
        if side is not None and abs(side) > 0.08:
            self.search_dir = -1.0 if side > 0 else 1.0     # line on the right -> turn right (negative z)

        # Line completely out of view: rotate towards the last turning direction.
        if self.lost_line_search and self._line_fraction(gray) < self.lost_min_fraction:
            self._publish(self.search_linear, self.search_dir * self.search_angular)
            return

        input_tensor = self.transform(PIL.Image.fromarray(gray)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            v_norm = float(output[0, 0].item())
            w_norm = float(output[0, 1].item())

        v_norm = min(max(v_norm, 0.0), 1.0)
        w_norm = min(max(w_norm, -1.0), 1.0)
        if abs(w_norm) > 0.15:
            self.search_dir = 1.0 if w_norm > 0 else -1.0

        v = v_norm * self.max_linear
        w = w_norm * self.max_angular
        if self.smoothing > 0.0:
            v = self.smoothing * self.prev_v + (1.0 - self.smoothing) * v
            w = self.smoothing * self.prev_w + (1.0 - self.smoothing) * w
        self.prev_v, self.prev_w = v, w
        self._publish(v, w)

    def destroy_node(self):
        self.running = False
        self.inference_thread.join()
        try:
            self._publish(0.0, 0.0)          # stop the robot
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AutopilotDiffDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
