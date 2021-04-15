#!/usr/bin/env python
# -*- coding: utf-8 -*-

from RSDataSetsUtils import RSDataSetsConverter, RSDataSetsSpliter

DOTAConverter = RSDataSetsConverter(dataset_name="DOTA", dataset_type="DOTA", images_folder_path="TestDataSet/DOTA/train/images", labels_folder_path="TestDataSet/DOTA/train/labelTxt")
DOTAConverter.convert2VOC()
DOTAConverter.convert2Yolo()
DOTAConverter.Prepare4Darknet("DOTA4Darknet", "train")
DOTAConverter.Prepare4YoloV5("DOTA4YoloV5", "train")

VisDroneConverter = RSDataSetsConverter(dataset_name="VisDrone2019", dataset_type="VisDrone", images_folder_path="TestDataSet/VisDrone2019/train/images", labels_folder_path="TestDataSet/VisDrone2019/train/annotations")
VisDroneConverter.convert2VOC()
VisDroneConverter.convert2Yolo()
VisDroneConverter.Prepare4Darknet("VisDrone20194Darknet", "train")
VisDroneConverter.Prepare4YoloV5("VisDrone20194YoloV5", "train")

VisDroneVOCConverter = RSDataSetsConverter(dataset_name="VisDrone2019_VOC", dataset_type="VOC", images_folder_path="TestDataSet/VisDrone2019_VOC/train/images", labels_folder_path="TestDataSet/VisDrone2019_VOC/train/VOC")
VisDroneVOCConverter.convert2Yolo(classes=["ignored regions", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"])

VHRConverter = RSDataSetsConverter(dataset_name="NWPU VHR-10", dataset_type="VHR", images_folder_path="TestDataSet/NWPU VHR-10/train/positive image set", labels_folder_path="TestDataSet/NWPU VHR-10/train/ground truth")
VHRConverter.convert2VOC()
VHRConverter.convert2Yolo()
VHRConverter.Prepare4Darknet("VHR4Darknet", "train")
VHRConverter.Prepare4YoloV5("VHR4YoloV5", "train")

DOTASpilter = RSDataSetsSpliter(oringin_imgs_folder_path="TestDataSet/DOTA/train/images", oringin_labels_folder_path="TestDataSet/DOTA/train/VOC")
DOTASpilter.splitImgs(img_dst_path="TestDataSet/DOTA/train/devided_images", size_w=412, size_h=412, step=412)
DOTASpilter.splitLabels(img_floder="TestDataSet/DOTA/train/images", out_img_floder="TestDataSet/DOTA/train/devided_images", label_floder="TestDataSet/DOTA/train/VOC", out_label_floder="TestDataSet/DOTA/train/devided_labels", size_w=412, size_h=412)
