#!/usr/bin/env python3
import time
import numpy as np
import cv2
import rospy
import onnxruntime as ort
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

class YoloLaptopNode:
    def __init__(self):
        self.bridge = CvBridge()

        self.model_path = rospy.get_param("~model_path")
        self.input_size = int(rospy.get_param("~input_size", 640))
        self.obj_thres = float(rospy.get_param("~obj_thres", 0.20))
        self.cls_thres = float(rospy.get_param("~cls_thres", 0.15))
        self.nms_thres = float(rospy.get_param("~nms_thres", 0.45))

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.det_pub = rospy.Publisher("/fruit_detections", Detection2DArray, queue_size=1)

        self.sub = rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self.image_cb,
            queue_size=1,
            buff_size=2**24
        )

        rospy.loginfo(f"Loaded ONNX model with ONNX Runtime: {self.model_path}")
        rospy.loginfo(f"Input: {self.session.get_inputs()[0].shape}")
        rospy.loginfo(f"Output: {self.session.get_outputs()[0].shape}")

    def image_cb(self, msg):
        start = time.time()

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        orig_h, orig_w = frame.shape[:2]

        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # NCHW

        output = self.session.run([self.output_name], {self.input_name: img})[0]
        output = np.squeeze(output)

        rospy.loginfo_throttle(5.0, f"Raw ONNX output shape: {np.array(output).shape}")

        # Expect YOLOv5-style output: (25200, 85)
        if output.ndim != 2:
            rospy.logwarn_throttle(5.0, f"Unexpected output rank: {output.shape}")
            return

        boxes = []
        confidences = []
        class_ids = []

        sx = float(orig_w) / float(self.input_size)
        sy = float(orig_h) / float(self.input_size)

        for row in output:
            if len(row) < 6:
                continue

            obj_conf = float(row[4])
            if obj_conf < self.obj_thres:
                continue

            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            max_class_score = float(class_scores[class_id])

            if max_class_score < self.cls_thres:
                continue

            cx, cy, w, h = row[:4]

            x = int((cx - w / 2.0) * sx)
            y = int((cy - h / 2.0) * sy)
            w = int(w * sx)
            h = int(h * sy)

            boxes.append([x, y, w, h])
            confidences.append(obj_conf * max_class_score)
            class_ids.append(class_id)

        det_msg = Detection2DArray()
        det_msg.header = msg.header

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.cls_thres, self.nms_thres)

            if len(indices) > 0:
                indices = np.array(indices).reshape(-1)

                for idx in indices:
                    x, y, w, h = boxes[idx]

                    d = Detection2D()
                    d.bbox.center.x = x + w / 2.0
                    d.bbox.center.y = y + h / 2.0
                    d.bbox.size_x = float(w)
                    d.bbox.size_y = float(h)

                    hyp = ObjectHypothesisWithPose()
                    hyp.id = int(class_ids[idx])
                    hyp.score = float(confidences[idx])

                    d.results.append(hyp)
                    det_msg.detections.append(d)

        if len(class_ids) > 0:
            rospy.loginfo_throttle(1.0, f"Detected class IDs: {class_ids}")

        self.det_pub.publish(det_msg)
        rospy.loginfo_throttle(2.0, f"Frame time: {(time.time()-start)*1000:.1f} ms")

if __name__ == "__main__":
    rospy.init_node("yolo_node_laptop")
    node = YoloLaptopNode()
    rospy.spin()