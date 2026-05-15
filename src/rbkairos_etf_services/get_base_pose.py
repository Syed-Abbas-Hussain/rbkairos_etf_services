"""
#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2

class Tracker3D:
    def __init__(self):
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.K = None
        self.last_log_time = rospy.Time.now()
        self.last_detection = None
        rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.info_cb)
        
        det_sub = message_filters.Subscriber("/fruit_detections", Detection2DArray)
        depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        
        # 100ms window for asynchronous messages from jetson nano and camera
        self.ts = message_filters.ApproximateTimeSynchronizer([det_sub, depth_sub], 10, 0.1)
        self.ts.registerCallback(self.cb)
        
        rospy.loginfo("Tracker active")
        #banana and apple
        self.valid_IDs = [47, 49]

    def info_cb(self, msg):
        # K[0] = fx, K[2] = cx, K[4] = fy, K[5] = cy
        self.K = msg.K

    def cb(self, det_msg, depth_msg):
        if self.K is None: return
    
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        self.last_detection = None
        # if coded to mm, scale to m
        if depth_msg.encoding == "16UC1":
            depth_img = depth_img.astype(float) * 0.001
        
        fx, cx, fy, cy = self.K[0], self.K[2], self.K[4], self.K[5]
        
        valid_detections = [d for d in det_msg.detections if d.results and d.results[0].id in self.valid_IDs]

        if not valid_detections:
            rospy.loginfo(f"Nema ID : {self.valid_IDs}")
            self.last_detection = None
            return

        for det in valid_detections:
            x1, y1 = int(det.bbox.center.x - det.bbox.size_x/2), int(det.bbox.center.y - det.bbox.size_y/2)
            w, h = int(det.bbox.size_x), int(det.bbox.size_y)
            
            
            roi = depth_img[max(0, y1):min(depth_img.shape[0], y1+h), 
                           max(0, x1):min(depth_img.shape[1], x1+w)]
            if roi.size == 0: continue

            roi_norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            _, mask = cv2.threshold(roi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = (mask > 0) 
            
            pts = np.argwhere(mask)
            rospy.loginfo(f"maska :{pts}")
            if len(pts) > 50:
                mean_pts = pts.mean(axis=0)
                v, u = mean_pts[0] + max(0, y1), mean_pts[1] + max(0, x1)
                z = roi[mask].mean()
                
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                
                pt = PointStamped()
                pt.header.frame_id = "camera_link_proxy" 
                pt.header.stamp = depth_msg.header.stamp
                pt.point.x, pt.point.y, pt.point.z = x, y, z
                
                    
                try:
                    target_pt = self.tf_buffer.transform(pt, "fr3_link0", timeout=rospy.Duration(0.2))
                    self.last_detection= target_pt.point
                except Exception as e:
                    self.last_detection=None
                    rospy.loginfo(f"Couldn't form coordinates: {e}")
    def get_last_detection(self):
        return self.last_detection


if __name__ == '__main__':
    rospy.init_node('tracker3d_test')
    tracker = Tracker3D()
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        det = tracker.get_last_detection()
        if det is not None:
            rospy.loginfo(f"3D fruit point: x={det.x:.3f}, y={det.y:.3f}, z={det.z:.3f}")
        rate.sleep()
#if __name__ == '__main__':
#    tracker = Tracker3D()
"""

#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2

class Tracker3D:
    def __init__(self):
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.K = None
        self.last_detection = None

        rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.info_cb)

        det_sub = message_filters.Subscriber("/fruit_detections", Detection2DArray)
        depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([det_sub, depth_sub], 10, 0.1)
        self.ts.registerCallback(self.cb)

        rospy.loginfo("Tracker3D test node active")

        # set this to the class IDs you want to track
        self.valid_IDs = [49, 75, 32]

    def info_cb(self, msg):
        self.K = msg.K

    def cb(self, det_msg, depth_msg):
        if self.K is None:
            rospy.logwarn_throttle(2.0, "Waiting for camera intrinsics K")
            return

        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        self.last_detection = None

        if depth_msg.encoding == "16UC1":
            depth_img = depth_img.astype(float) * 0.001

        fx, cx, fy, cy = self.K[0], self.K[2], self.K[4], self.K[5]

        valid_detections = [d for d in det_msg.detections if d.results and d.results[0].id in self.valid_IDs]

        if not valid_detections:
            rospy.loginfo_throttle(2.0, f"No valid class IDs found. Looking for: {self.valid_IDs}")
            return

        for det in valid_detections:
            x1 = int(det.bbox.center.x - det.bbox.size_x / 2)
            y1 = int(det.bbox.center.y - det.bbox.size_y / 2)
            w = int(det.bbox.size_x)
            h = int(det.bbox.size_y)

            roi = depth_img[max(0, y1):min(depth_img.shape[0], y1 + h),
                            max(0, x1):min(depth_img.shape[1], x1 + w)]

            if roi.size == 0:
                rospy.logwarn_throttle(2.0, "Depth ROI is empty")
                continue

            roi_norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            _, mask = cv2.threshold(roi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = (mask > 0)

            pts = np.argwhere(mask)
            if len(pts) <= 50:
                rospy.logwarn_throttle(2.0, f"Too few valid depth points: {len(pts)}")
                continue

            mean_pts = pts.mean(axis=0)
            v = mean_pts[0] + max(0, y1)
            u = mean_pts[1] + max(0, x1)
            z = roi[mask].mean()

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            pt = PointStamped()
            pt.header.frame_id = "camera_link_proxy"
            pt.header.stamp = depth_msg.header.stamp
            pt.point.x = x
            pt.point.y = y
            pt.point.z = z

            rospy.loginfo_throttle(
                1.0,
                f"Camera frame point: x={pt.point.x:.3f}, y={pt.point.y:.3f}, z={pt.point.z:.3f}"
            )

            try:
                target_pt = self.tf_buffer.transform(pt, "fr3_link0", timeout=rospy.Duration(0.2))
                self.last_detection = target_pt.point
                rospy.loginfo_throttle(
                    1.0,
                    f"Base frame point: x={target_pt.point.x:.3f}, y={target_pt.point.y:.3f}, z={target_pt.point.z:.3f}"
                )
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"TF transform failed: {e}")

    def get_last_detection(self):
        return self.last_detection

if __name__ == '__main__':
    rospy.init_node("tracker3d_test")
    tracker = Tracker3D()
    rospy.spin()