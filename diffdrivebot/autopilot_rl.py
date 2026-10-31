import os

import cv2
import numpy as np
import torch
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from cv_bridge import CvBridge


class AutopilotRL(Node):
    """
    Runs the reinforcement-learning line-following policy (exported by rl/export_policy.py as TorchScript).

    Every `control_period` seconds of SIMULATED time (taken from the image header, so a slow real-time factor does not
    change the control rate) it stacks the last `frame_stack` grayscale frames, runs the policy and publishes a Twist:
        v = (a0 + 1) / 2 * max_linear_velocity        w = a1 * max_angular_velocity        (+ = turn left)
    These numbers (frame size, stack, period, limits) MUST match the environment the policy was trained in
    (rl_env.py defaults / diffdrive_bot.xacro).
    """

    def __init__(self):
        super().__init__("autopilot_rl")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("model_path", "/tmp/diffdrivebot/rl_policy.pt")
        self.declare_parameter("max_linear_velocity", 0.4)
        self.declare_parameter("max_angular_velocity", 2.0)
        self.declare_parameter("frame_stack", 3)
        self.declare_parameter("control_period", 0.1)
        self.declare_parameter("obs_width", 128)
        self.declare_parameter("obs_height", 96)
        self.declare_parameter("debug_dir", "")        # if set, saves what the policy sees every 20th step

        p = self.get_parameter
        self.max_linear = float(p("max_linear_velocity").value)
        self.max_angular = float(p("max_angular_velocity").value)
        self.stack = int(p("frame_stack").value)
        self.period = float(p("control_period").value)
        self.W, self.H = int(p("obs_width").value), int(p("obs_height").value)
        self.debug_dir = p("debug_dir").value
        model_path = p("model_path").value

        self.get_logger().info(f"Loading policy from: {model_path}")
        self.model = torch.jit.load(model_path, map_location="cpu").eval()
        self.bridge = CvBridge()
        self.frames = deque(maxlen=self.stack)
        self.last_t = None
        self.count = 0
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.cmd_pub = self.create_publisher(Twist, p("cmd_vel_topic").value, 1)
        self.create_subscription(Image, p("image_topic").value, self._on_image, qos)
        self.get_logger().info("AutopilotRL initialized")

    def _publish(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _on_image(self, msg: Image):
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if t == 0.0:                                         # no simulated stamp: fall back to wall time
            t = self.get_clock().now().nanoseconds * 1e-9
        if self.last_t is not None:
            if t < self.last_t - 1e-3:                       # simulation was reset
                self.frames.clear()
                self.last_t = None
            elif t - self.last_t < self.period - 1e-6:
                return                                       # not time for the next control step yet
        self.last_t = t

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape != (self.H, self.W):
            gray = cv2.resize(gray, (self.W, self.H), interpolation=cv2.INTER_AREA)
        if not self.frames:
            self.frames.extend([gray] * self.stack)
        else:
            self.frames.append(gray)

        obs = np.stack(self.frames)[None]                    # (1, C, H, W) uint8, same layout as training
        with torch.no_grad():
            a = self.model(torch.from_numpy(obs))[0].numpy()
        v = min(max((float(a[0]) + 1.0) / 2.0, 0.0), 1.0) * self.max_linear
        w = min(max(float(a[1]), -1.0), 1.0) * self.max_angular
        self._publish(v, w)

        self.count += 1
        if self.debug_dir and self.count % 20 == 0:
            cv2.imwrite(os.path.join(self.debug_dir, f"obs_{self.count:06d}.png"), np.hstack(list(self.frames)))

    def destroy_node(self):
        try:
            self._publish(0.0, 0.0)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AutopilotRL()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
