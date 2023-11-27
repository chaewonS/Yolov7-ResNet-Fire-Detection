import cv2
import torch
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging
from utils.torch_utils import select_device, TracedModel

import rospkg
import rospy

from sensor_msgs.msg import Image
from yolov7.msg import Detector2DArray
from yolov7.msg import Detector2D
from yolov7.msg import Bbox2D

from cv_bridge import CvBridge, CvBridgeError

class DETECT(object):
    def __init__(self, weights, img_b_flag):

        self.weights_path = weights
        self.img_size = 640
        self.conf_thres = 0.25
        self.nms_thres = 0.45
        self.trace = True
        self.augment = False
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.img_b_pub = img_b_flag

        self.model, self.stride = self._get_model()
        self.names = self._get_name(self.model)
        self.class_color = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # setup pub node
        self.bbox_pub = rospy.Publisher("/video_source/bbox", Detector2DArray, queue_size=1)
        if self.img_b_pub is True:
            self.img_pub = rospy.Publisher("/video_source/raw_box", Image, queue_size=10)


    def _get_model(self):
        # Load model
        model = attempt_load(self.weights_path, map_location='cuda:0')  # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=stride)  # check img_size

        if self.trace:
            model = TracedModel(model, self.device, self.img_size)
        if self.half:
            model.half()  # to FP16

        return model, stride


    def _get_name(self, model):
        name = model.module.names if hasattr(model, 'module') else model.names
        return name

    
    def resize_image(self, img):
        ih = img.shape[0] % 32
        iw = img.shape[1] % 32
        id = img.shape[2]
        img = np.resize(img, (img.shape[0]-ih, img.shape[1]-iw, id))

        return img


    def detect_image(self, img):
        img = self.resize_image(img)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            # Inference
            pred = self.model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.nms_thres, agnostic=False)
            # pred = x1, y1, x2, y2, conf, cls_pred

        return pred[0]


    def draw_bbox_image(self, img, detections):
        # Bounding-box colors
        for x1, y1, x2, y2, conf, cls_pred in detections:
            text = f"{self.names[int(cls_pred)]}:{conf.item():0.2f}"
            box_w = x2 - x1
            box_h = y2 - y1
            # Create a Rectangle patch
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.class_color[int(cls_pred)], 2)
            img = cv2.line(img, (int((x1+x2)/2), int((y1+y2)/2)), (int((x1+x2)/2), int((y1+y2)/2)), self.class_color[int(cls_pred)], 4)
            font =  cv2.FONT_HERSHEY_PLAIN
            img = cv2.putText(img, text, (int(x1), int(y1)-5), font, 2, self.class_color[int(cls_pred)], 1, cv2.LINE_AA)
        
        return img


    def publisher(self, img, bridge, pred):
        """Publish a video as ROS messages."""
        print ("Publish Bbox")
        
        try:
            # Publish.
            detection_ary = Detector2DArray()
            detection_ary.header.stamp = rospy.Time.now()
            for i in range(len(pred)):
                # pred: x1, y1, x2, y2, conf, cls_pred
                detection = Detector2D()

                detection.header.stamp = rospy.Time.now()
                detection.id = int(pred[i,5])
                detection.confidence_score = float(pred[i,4])

                detection.bbox.header.stamp = rospy.Time.now()
                detection.bbox.center_x = int(pred[i][0] + (pred[i][2] - pred[i][0])/2)
                detection.bbox.center_y = int(pred[i][1] + (pred[i][3] - pred[i][1])/2)

                detection.bbox.size_x = int(abs(pred[i][0] - pred[i][2]))
                detection.bbox.size_y = int(abs(pred[i][1] - pred[i][3]))
                
                detection_ary.detections.append(detection)
            
            # Publish image.
            if self.img_b_pub is True:
                img_out = self.draw_bbox_image(img, pred)

                img_msg = bridge.cv2_to_imgmsg(img_out, "rgb8")
                img_msg.header.stamp = rospy.Time.now()
                self.img_pub.publish(img_msg)
            self.bbox_pub.publish(detection_ary)

        except CvBridgeError as err:
            print(err)

        return
