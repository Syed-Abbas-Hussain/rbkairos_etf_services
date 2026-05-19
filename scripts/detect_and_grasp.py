#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from rbkairos_etf_services.get_base_pose import Tracker3D


class GraspController:
    def __init__(self):
        rospy.init_node("fruit_grasp_controller")

        self.buffer = []
        self.buffer_size = int(rospy.get_param("~buffer_size", 8))
        self.min_samples = int(rospy.get_param("~min_samples", 4))
        #self.deviation_threshold = float(rospy.get_param("~deviation_threshold", 0.04))
        self.publish_topic = rospy.get_param("~publish_topic", "/vision/latest_fruit_point")

        self.tracker = Tracker3D()
        self.point_pub = rospy.Publisher(self.publish_topic, PointStamped, queue_size=1)

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.process_detections()
            rate.sleep()

    def publish_point(self, avg_pos):
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "fr3_link0"
        msg.point.x = float(avg_pos[0])
        msg.point.y = float(avg_pos[1])
        msg.point.z = float(avg_pos[2])
        self.point_pub.publish(msg)

    def process_detections(self):
        det = self.tracker.get_last_detection()

        if det is None:
            self.buffer = []
            return

        self.buffer.append([det.x, det.y, det.z])

        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]

        if len(self.buffer) < self.min_samples:
            return

        avg_pos = np.mean(self.buffer, axis=0)
        std_dev = np.std(self.buffer, axis=0)

        # always publish the smoothed grasp point
        self.publish_point(avg_pos)

        rospy.loginfo_throttle(
            1.0,
            f"Vision target avg={avg_pos.round(4).tolist()} std={std_dev.round(4).tolist()}"
        )

if __name__ == "__main__":
    GraspController()