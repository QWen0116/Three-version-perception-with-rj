# -*- coding: utf-8 -*-

"""
Since multiple CAV normally use the same ML/DL model,
here we have this class to enable different CAVs share the same model to
 avoid duplicate memory consumption.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import cv2
import torch
import numpy as np
import copy
from pytorchfi.core import fault_injection
from pytorchfi.neuron_error_models import random_neuron_location
from pytorchfi.neuron_error_models import random_neuron_inj, random_inj_per_layer
import torchvision
from pytorchfi.core import fault_injection
from pytorchfi import weight_error_models
from pytorchfi import neuron_error_models
from collections import Counter
import torch.nn as nn


class YoloDetector:
    def __init__(self, model, name):
        self.model = model
        self.name = name

def pyfi(model):
    batch_size=128
    input_shape=[3, 640, 640]
    model = fault_injection(model, batch_size=batch_size, input_shape=input_shape)
     # Declare weight injection
    corrupted_model =  weight_error_models.random_weight_inj(model, -1, -100, 300)
    return corrupted_model

class MLManager(object):
    """
    A class that should contain all the ML models you want to initialize.

    Attributes
    -object_detector : torch_detector
        The YoloV5 detector load from pytorch.

    """


    def __init__(self):
# 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model1 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)
        model2 = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True).to(self.device)
        model3 = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True).to(self.device)
        self.object_detector1=YoloDetector(model1,"ObjectDetector1_Healthy")
        self.object_detector2=YoloDetector(model2,"ObjectDetector2_Healthy")
        self.object_detector3=YoloDetector(model3,"ObjectDetector3_Healthy")

        corrupted_model1 = copy.deepcopy(model1)
        corrupted_model2 = copy.deepcopy(model2)
        corrupted_model3 = copy.deepcopy(model3)

        pyfi(corrupted_model1)
        pyfi(corrupted_model2)
        pyfi(corrupted_model3)

        
        self.object_detector4=YoloDetector(corrupted_model1,"ObjectDetector1_Faulty")
        self.object_detector5=YoloDetector(corrupted_model2,"ObjectDetector2_Faulty")
        self.object_detector6=YoloDetector(corrupted_model3,"ObjectDetector3_Faulty")


        


    def get_matched_detection(self,detections):
        outputs = [detections_to_numpy(det) for det in detections]
        matched_detections = {}
        for output in outputs:
            for det in output:
                found_match = False
                for key, value in matched_detections.items():
                    for bbox in value['bboxes']:
                        if is_match_detections(bbox, det):
                            value['bboxes'].append(det[:4])
                            value['confidences'].append(det[4])
                            value['classes'].append(det[5])
                            found_match = True
                            break
                    if found_match:
                        break
                if not found_match:
                    matched_detections[len(matched_detections)] = {'bboxes': [det[:4]],'confidences': [det[4]], 'classes': [det[5]]}
        return matched_detections


    def get_final_detection(self,matched_detections):
        final_detections = []
        skipflag = 0 
        for key, value in matched_detections.items():
            if len(value['bboxes']) <= 1:
                continue
            else: #len(value['bboxes']) > 1
                class_counts = Counter(value['classes'])
                most_common_class, most_common_count = class_counts.most_common(1)[0]
                if most_common_count > 1:
                    final_class = most_common_class
                    highest_confidence_idx = np.argmax([conf for cls, conf in zip(value['classes'], value['confidences']) if cls == most_common_class])
                    # final_bbox = value['bboxes'][highest_confidence_idx]
                    # highest_confidence = value['confidences'][highest_confidence_idx]
                    final_bbox = value['bboxes'][highest_confidence_idx]
                    highest_confidence = value['confidences'][highest_confidence_idx]
                    final_detections.append((final_bbox,highest_confidence, final_class))
                else:
                    skipflag=1
                    continue

        flattened_detections = np.array([np.concatenate([bbox, [conf, cls]]) for bbox, conf, cls in final_detections])
        return flattened_detections,skipflag



    def draw_2d_box(self, result, rgb_image,label_names):
        """
        Draw 2d bounding box based on the yolo detection.

        Args:
            -result (yolo.Result):Detection result from yolo 5.
            -rgb_image (np.ndarray): Camera rgb image.
            -index(int): Indicate the index.

        Returns:
            -rgb_image (np.ndarray): camera image with bbx drawn.
        """
        bounding_box=result

        for i in range(bounding_box.shape[0]):
            detection = bounding_box[i]

            # the label has 80 classes, which is the same as coco dataset
            label = int(detection[5])
            # label_name = result.names[label]
            label_name = label_names[label]

            if is_vehicle_cococlass(label):
                label_name = 'vehicle'

            x1, y1, x2, y2 = int(
                detection[0]), int(
                detection[1]), int(
                detection[2]), int(
                detection[3])
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw text on it
            cv2.putText(rgb_image, label_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

        return rgb_image


def detections_to_numpy(detection):
    if detection.is_cuda:
        output = detection.cpu().detach().numpy()
    else:
        output = detection.detach().numpy()
    return output


def bbox_iou(bbox1, bbox2):

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def is_match_detections(det1, det2, iou_threshold=0.7):

    bbox1 = det1[:4]
    bbox2 = det2[:4]
    iou = bbox_iou(bbox1, bbox2)
    return iou >= iou_threshold


def is_vehicle_cococlass(label):
    """
    Check whether the label belongs to the vehicle class according
    to coco dataset.
    Args:
        -label(int): yolo detection prediction.
    Returns:
        -is_vehicle: bool
            whether this label belongs to the vehicle class
    """
    vehicle_class_array = np.array([1, 2, 3, 5, 7], dtype=np.int)
    return True if 0 in (label - vehicle_class_array) else False
