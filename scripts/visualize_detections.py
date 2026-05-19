#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection2DArray


class EdgeVisualizer:
    def __init__(self):
        self.latest_detections = None

        rospy.loginfo("Starting visualizer...")
        rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.image_callback, queue_size=1)
        rospy.Subscriber("/fruit_detections", Detection2DArray, self.det_callback, queue_size=1)

        rospy.loginfo("Visualizer online. Waiting for image and detection topics...")

    def det_callback(self, msg):
        self.latest_detections = msg

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                rospy.logwarn_throttle(2.0, "Could not decode compressed image.")
                return

            if self.latest_detections is not None:
                for det in self.latest_detections.detections:
                    w = int(det.bbox.size_x)
                    h = int(det.bbox.size_y)
                    cx = int(det.bbox.center.x)
                    cy = int(det.bbox.center.y)

                    x1 = cx - w // 2
                    y1 = cy - h // 2
                    x2 = cx + w // 2
                    y2 = cy + h // 2

                    if len(det.results) > 0:
                        class_id = det.results[0].id
                        score = det.results[0].score
                        label = "ID: {} ({:.2f})".format(class_id, score)
                    else:
                        label = "No ID"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            cv2.imshow("RB-Kairos Edge Vision", frame)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr_throttle(2.0, f"Visualizer image callback error: {e}")


if __name__ == '__main__':
    rospy.init_node('laptop_visualizer')
    viz = EdgeVisualizer()
    rospy.spin()