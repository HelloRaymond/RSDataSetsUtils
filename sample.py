#!/usr/bin/env python
# -*- coding: utf-8 -*-
from RSDataSetsUtils import RSDataSetsUtils

DOTAUtils = RSDataSetsUtils(dataset_name="DOTA", dataset_type="DOTA",
                            images_folder_path="TestDataSet/DOTA/train/images", labels_folder_path="TestDataSet/DOTA/train/labelTxt")
DOTAUtils.convert2VOC()
DOTAUtils.convert2Yolo()
DOTAUtils.Prepare4Darknet("DOTA4Darknet", "train")
DOTAUtils.Prepare4YoloV5("DOTA4YoloV5", "train")

VisDroneUtils = RSDataSetPrepare(dataset_name="VisDrone2019", dataset_type="VisDrone",
                                 images_folder_path="TestDataSet/VisDrone2019/train/images", labels_folder_path="TestDataSet/VisDrone2019/train/annotations")
VisDroneUtils.convert2VOC()
VisDroneUtils.convert2Yolo()
VisDroneUtils.Prepare4Darknet("VisDrone20194Darknet", "train")
VisDroneUtils.Prepare4YoloV5("VisDrone20194YoloV5", "train")

VisDroneVOCUtils = RSDataSetsUtils(dataset_type="VOC", images_folder_path="TestDataSet/VisDrone2019_VOC/train/images",
                                   labels_folder_path="TestDataSet/VisDrone2019_VOC/train/VOC")
VisDroneVOCUtils.convert2Yolo(classes=["ignored regions", "pedestrian", "people", "bicycle",
                              "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"])

VHRUtils = RSDataSetsUtils(dataset_type="VHR", images_folder_path="TestDataSet/NWPU VHR-10/train/positive image set",
                           labels_folder_path="TestDataSet/NWPU VHR-10/train/ground truth")
VHRUtils.convert2VOC()
VHRUtils.convert2Yolo()
VHRUtils.Prepare4Darknet("VHR4Darknet", "train")
VHRUtils.Prepare4YoloV5("VHR4YoloV5", "train")
