# -*- coding: utf-8 -*-
"""
Static Obstacle base class
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import sys
import math
import cv2
import numpy as np
import carla
import torch
import copy
from pytorchfi.core import fault_injection
from pytorchfi.neuron_error_models import random_neuron_location
from pytorchfi.neuron_error_models import random_neuron_inj, random_inj_per_layer
import torchvision

# yolov5s = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
# model =fasterrcnn_resnet50 = torch.hub.load('pytorch/vision:v0.6.0', 'fasterrcnn_resnet50_fpn', pretrained=True)
# model= torch.hub.load('ultralytics/yolov3', 'yolov3')
model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # official model

# injection_strength=20
# injection_strength = injection_strength

# for param in model.parameters():
#     param.data = param.data + injection_strength * torch.randn_like(param.data)

# # 步骤5：使用 PyTorchFi 注入故障
# # 复制模型
# pfi_model = copy.deepcopy(model)

# # 初始化 fault_injection 类
# batch_size=128

# fi = fault_injection(pfi_model, batch_size=batch_size,input_shape=[3,640, 640])

# def custom_fault(val):
#     return val * 0.5

# # 定义神经元注入参数
# neuron_injection_params = {
# #     "function":random_neuron_inj,
#     "layer_num": [0,1],  # 选择要注入的卷积层索引
#     "batch": [2,3],      # 批次索引
#     "dim1": [0,1],       # 维度1索引
#     "dim2": [0,1],       # 维度2索引
#     "dim3": [0,1],       # 维度3索引
#     "value": [1,1],    # 注入的值
# }

# # 声明神经元注入
# fi.declare_neuron_fi(**neuron_injection_params)
# corrupted_model = fi.get_corrupted_model()