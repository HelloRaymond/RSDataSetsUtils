#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

RSDataSetUtils

~~~~~~~~~~~~~~~~

This script provides some utils for remote sensing image data sets processing;

The types of data sets currently supported are: PASCAL VOC, VisDrone, DOTA, NWPU VHR-10, YOLO.

"""

import os
import cv2
import time
from xml.dom import minidom
import xml.etree.ElementTree as ET


class YoloBoudingBox:
    def __init__(self, center_x, center_y, width, height, cls_id):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.cls_id = cls_id


class VOCBoudingBox:
    def __init__(self, x_min, x_max, y_min, y_max, cls_name, is_difficult=False, is_truncated=False):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cls_name = cls_name
        self.is_difficult = is_difficult
        self.is_truncated = is_truncated


class VOCLabel:
    def __init__(self, img_path, label_path, clss):
        self.img_path = img_path
        self.label_path = label_path
        self.clss = clss
        self.BBoxes = []
        self.img_w = cv2.imread(img_path).shape[1]
        self.img_h = cv2.imread(img_path).shape[0]
        self.img_d = cv2.imread(img_path).shape[2]

    def getBBoxes(label_path):
        with open(label_path, "r") as input_label_file:
            tree = ET.parse(input_label_file)
            root = tree.getroot()
            for obj in root.iter("object"):
                try:
                    is_difficult = False if obj.find(
                        "difficult").text == "0" else True
                except:
                    is_difficult = False
                try:
                    is_truncated = False if obj.find(
                        "truncated").text == "0" else True
                except:
                    is_truncated = False
                cls_name = obj.find("name").text
                xmlbox = obj.find("bndbox")
                x_min = float(xmlbox.find("xmin").text)
                x_max = float(xmlbox.find("xmax").text)
                y_min = float(xmlbox.find("ymin").text)
                y_max = float(xmlbox.find("ymax").text)
                self.BBoxes.append(BBoxes.append(VOCBoudingBox(
                    img_w, img_h, img_d, x_min, x_max, y_min, y_max, cls_name, is_difficult, is_truncated)))

    def Convert2Yolo(self):
        dw = 1.0/self.img_w
        dh = 1.0/self.img_h
        BBoxesYolo = []
        for box in self.BBoxes:
            x = (box.x_min + box.x_max)/2.0 - 1
            y = (box.y_min + box.y_max)/2.0 - 1
            w = box.x_max - box.x_min
            h = box.y_max - box.y_min
            x = x*dw
            w = w*dw
            y = y*dh
            h = h*dh
            BBoxesYolo.append(YoloBoudingBox(
                x, y, w, h, self.clss.index(box.cls_name)))
        newYoloLabel = YoloLabel(self.img_path, self.label_path, self.clss)
        newYoloLabel.BBoxes = BBoxesYolo
        return newYoloLabel


class YoloLabel(VOCLabel):
    def __init__(self, img_path, label_path, clss):
        super(YoloLabel, self).__init__(img_path, label_path, clss)

    def getBBoxes(self):
        with open(self.label_path, "r") as input_label_file:
            lines = [input_label_file.readlines()]
            for line in lines:
                for boxes in line:
                    box = boxes.strip("\n")
                    box = box.split(" ")
                    cls_name = int(box[0])
                    x_center = float(box[1])
                    y_center = float(box[2])
                    width = float(box[3])
                    height = float(box[4])
                    cls_id = self.clss.index(cls_name)
                    self.BBoxes.append(YoloBoudingBox(
                        x_center, y_center, width, height, cls_id))

    def Convert2VOC(self):
        BBoxesVOC = []
        for box in self.BBoxes:
            xmin = int(box.center_x * self.img_w -
                       box.bbox_width * self.img_w / 2)
            ymin = int(box.center_y * self.img_h -
                       box.bbox_height * self.img_h / 2)
            xmax = int(box.center_x * self.img_w +
                       box.bbox_width * self.img_w / 2)
            ymax = int(box.center_y * self.img_h +
                       box.bbox_height * self.img_h / 2)
            BBoxesVOC.append(VOCBoudingBox(self.img_w, self.img_h, self.img_w,
                             box.xmin, box.xmax, box.ymin, box.ymax, clss[box.cls_id]))
        newVOCLabel = VOCLabel(self.img_path, self.label_path, self.clss)
        newVOCLabel.BBoxes = BBoxesVOC
        return newVOCLabel


class RSDataSetsUtils:

    def __init__(self, dataset_name, dataset_type, images_folder_path, labels_folder_path):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type.lower()
        if self.dataset_type not in ["visdrone", "voc", "dota", "yolo", "vhr"]:
            print("Not yet supported!")
            self.__del__()
        self.images_folder_path = images_folder_path
        self.labels_folder_path = labels_folder_path

    def __WriteVOCLabelXml(self, voc_label, xml_path):
        img_folder = os.path.split(voc_label.img_path)[0]
        img_name = os.path.split(voc_label.img_path)[1]
        doc = minidom.Document()
        annotation = doc.createElement("annotation")
        doc.appendChild(annotation)
        folder = doc.createElement("folder")
        folder.appendChild(doc.createTextNode(img_folder))
        annotation.appendChild(folder)
        filename = doc.createElement("filename")
        filename.appendChild(doc.createTextNode(img_name))
        annotation.appendChild(filename)
        source = doc.createElement("source")
        database = doc.createElement("database")
        database.appendChild(doc.createTextNode("Unknown"))
        source.appendChild(database)
        annotation.appendChild(source)
        size = doc.createElement("size")
        width = doc.createElement("width")
        width.appendChild(doc.createTextNode(str(voc_label.img_w)))
        size.appendChild(width)
        height = doc.createElement("height")
        height.appendChild(doc.createTextNode(str(voc_label.img_h)))
        size.appendChild(height)
        depth = doc.createElement("depth")
        depth.appendChild(doc.createTextNode(str(voc_label.img_d)))
        size.appendChild(depth)
        annotation.appendChild(size)
        segmented = doc.createElement("segmented")
        segmented.appendChild(doc.createTextNode("0"))
        annotation.appendChild(segmented)
        for box in voc_label.BBoxes:
            object = doc.createElement("object")
            nm = doc.createElement("name")
            nm.appendChild(doc.createTextNode(box.cls_name))
            object.appendChild(nm)
            pose = doc.createElement("pose")
            pose.appendChild(doc.createTextNode("Unspecified"))
            object.appendChild(pose)
            truncated = doc.createElement("truncated")
            truncated.appendChild(doc.createTextNode(
                "1" if box.is_truncated else "0"))
            object.appendChild(truncated)
            difficult = doc.createElement("difficult")
            difficult.appendChild(doc.createTextNode(
                "1" if box.is_difficult else "0"))
            object.appendChild(difficult)
            bndbox = doc.createElement("bndbox")
            xmin = doc.createElement("xmin")
            xmin.appendChild(doc.createTextNode(str(box.x_min)))
            bndbox.appendChild(xmin)
            ymin = doc.createElement("ymin")
            ymin.appendChild(doc.createTextNode(str(box.y_min)))
            bndbox.appendChild(ymin)
            xmax = doc.createElement("xmax")
            xmax.appendChild(doc.createTextNode(str(box.x_max)))
            bndbox.appendChild(xmax)
            ymax = doc.createElement("ymax")
            ymax.appendChild(doc.createTextNode(str(box.y_max)))
            bndbox.appendChild(ymax)
            object.appendChild(bndbox)
            annotation.appendChild(object)
            with open(xml_path, "w") as output_label_file:
                output_label_file.write(doc.toprettyxml())

    def __WritrYoloLabelTxt(self, yolo_label, label_txt_path):
        with open(label_txt_path, "w") as output_label_file:
            for box in yolo_label.BBoxes:
                output_label_file.write("%d %.16f %.16f %.16f %.16f\n" % (
                    box.cls_id, box.center_x, box.center_y, box.width, box.height))

    def __VOC2YOLO(self, img_path, label_path, classes):
        output_file_name = os.path.basename(label_path).split(".")[0] + ".txt"
        save_path = os.path.join(self.labels_folder_path, "../YOLO")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        voc_label = VOCLabel(img_path, label_path, classes)
        voc_label.getBBoxes(label_path)
        yolo_label = voc_label.Convert2Yolo()
        self.__WritrYoloLabelTxt(
            yolo_label, os.path.join(save_path, output_file_name))

    def __DOTA2VOC(self, img_path, label_path):
        output_file_name = os.path.splitext(
            os.path.basename(label_path))[0] + ".xml"
        save_path = os.path.join(self.labels_folder_path, "../VOC")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        voc_label = VOCLabel(img_path, label_path, ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", "harbor",
                             "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool", "container-crane", "airport", "helipad"])
        with open(label_path, "r") as input_label_file:
            lines = [input_label_file.readlines()]
            for line in lines:
                for boxes in line:
                    if "imagesource" in boxes or "gsd" in boxes:
                        continue
                    box = boxes.strip("\n")
                    box = box.split(" ")
                    x_min = min(float(box[0]), float(
                        box[2]), float(box[4]), float(box[6]))
                    y_min = min(float(box[1]), float(
                        box[3]), float(box[5]), float(box[7]))
                    x_max = max(float(box[0]), float(
                        box[2]), float(box[4]), float(box[6]))
                    y_max = max(float(box[1]), float(
                        box[3]), float(box[5]), float(box[7]))
                    cls_name = box[8]
                    is_difficult = False if box[9] == "0" else True
                    voc_label.BBoxes.append(VOCBoudingBox(
                        x_min, x_max, y_min, y_max, cls_name, is_difficult))
        self.__WriteVOCLabelXml(
            voc_label, os.path.join(save_path, output_file_name))

    def __DOTA2YOLO(self, img_path, label_path, classes=["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool", "container-crane", "airport", "helipad"]):
        output_file_name = os.path.basename(label_path)
        save_path = os.path.join(self.labels_folder_path, "../YOLO")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        voc_label = VOCLabel(img_path, label_path, classes)
        with open(label_path, "r") as input_label_file:
            lines = [input_label_file.readlines()]
            for line in lines:
                for boxes in line:
                    if "imagesource" in boxes or "gsd" in boxes:
                        continue
                    box = boxes.strip("\n")
                    box = box.split(" ")
                    x_min = min(float(box[0]), float(
                        box[2]), float(box[4]), float(box[6]))
                    y_min = min(float(box[1]), float(
                        box[3]), float(box[5]), float(box[7]))
                    x_max = max(float(box[0]), float(
                        box[2]), float(box[4]), float(box[6]))
                    y_max = max(float(box[1]), float(
                        box[3]), float(box[5]), float(box[7]))
                    voc_label.BBoxes.append(VOCBoudingBox(
                        int(x_min), int(x_max), int(y_min), int(y_max), box[8]))
                    yolo_label = voc_label.Convert2Yolo()
        self.__WritrYoloLabelTxt(
            yolo_label, os.path.join(save_path, output_file_name))

    def __VisDrone2VOC(self, img_path, label_path, classes=["ignored regions", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"]):
        output_file_name = os.path.basename(label_path).split(".")[0] + ".xml"
        save_path = os.path.join(self.labels_folder_path, "../VOC")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        voc_label = VOCLabel(img_path, label_path, classes)
        with open(label_path, "r") as input_label_file:
            lines = [input_label_file.readlines()]
            for line in lines:
                for boxes in line:
                    box = boxes.strip("\n")
                    box = box.split(",")
                    x_min = box[0]
                    y_min = box[1]
                    x_max = int(box[0]) + int(box[2])
                    y_max = int(box[1]) + int(box[3])
                    cls_name = classes[int(box[5])]
                    is_truncated = True if int(
                        box[6]) > 0 or int(box[7]) > 0 else False
                    is_difficult = True if int(box[7]) > 1 else False
                    voc_label.BBoxes.append(VOCBoudingBox(
                        x_min, x_max, y_min, y_max, cls_name, is_difficult, is_truncated))
        self.__WriteVOCLabelXml(
            voc_label, os.path.join(save_path, output_file_name))

    def __VisDrone2YOLO(self, img_path, label_path):
        output_file_name = os.path.basename(label_path)
        save_path = os.path.join(self.labels_folder_path, "../YOLO")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        voc_label = VOCLabel(img_path, label_path, ["ignored regions", "pedestrian", "people",
                             "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"])
        with open(label_path, "r") as input_label_file:
            lines = [input_label_file.readlines()]
            for line in lines:
                for boxes in line:
                    box = boxes.strip("\n")
                    box = box.split(",")
                    x_min = int(box[0])
                    y_min = int(box[1])
                    x_max = int(box[0]) + int(box[2])
                    y_max = int(box[1]) + int(box[3])
                    cls_id = int(box[5])
                    voc_label.BBoxes.append(VOCBoudingBox(
                        x_min, x_max, y_min, y_max, voc_label.clss[cls_id]))
                    yolo_label = voc_label.Convert2Yolo()
            self.__WritrYoloLabelTxt(
                yolo_label, os.path.join(save_path, output_file_name))

    def __VHR2VOC(self, img_path, label_path, classes=["airplane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor", "bridge", "vehicle"]):
        output_file_name = os.path.basename(label_path).split(".")[0] + ".xml"
        save_path = os.path.join(self.labels_folder_path, "../VOC")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        voc_label = VOCLabel(img_path, label_path, classes)
        with open(label_path, "r") as input_label_file:
            lines = [input_label_file.readlines()]
            for line in lines:
                for boxes in line:
                    box = boxes.strip("\n")
                    box = box.split(",")
                    x_min = int(box[0].replace("(", ""))
                    y_min = int(box[1].replace(")", ""))
                    x_max = int(box[2].replace("(", ""))
                    y_max = int(box[3].replace(")", ""))
                    cls_name = classes[int(box[4])]
                    voc_label.BBoxes.append(VOCBoudingBox(
                        x_min, x_max, y_min, y_max, cls_name))
        self.__WriteVOCLabelXml(
            voc_label, os.path.join(save_path, output_file_name))

    def __VHR2YOLO(self, img_path, label_path):
        output_file_name = os.path.basename(label_path)
        save_path = os.path.join(self.labels_folder_path, "../YOLO")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        voc_label = VOCLabel(img_path, label_path, ["airplane", "ship", "storage tank", "baseball diamond",
                             "tennis court", "basketball court", "ground track field", "harbor", "bridge", "vehicle"])
        with open(label_path, "r") as input_label_file:
            lines = [input_label_file.readlines()]
            for line in lines:
                for boxes in line:
                    box = boxes.strip("\n")
                    box = box.split(",")
                    x_min = int(box[0].replace("(", ""))
                    y_min = int(box[1].replace(")", ""))
                    x_max = int(box[2].replace("(", ""))
                    y_max = int(box[3].replace(")", ""))
                    voc_label.BBoxes.append(VOCBoudingBox(
                        x_min, x_max, y_min, y_max, voc_label.clss[int(box[4])]))
                    yolo_label = voc_label.Convert2Yolo()
            self.__WritrYoloLabelTxt(
                yolo_label, os.path.join(save_path, output_file_name))

    def convert2VOC(self):
        txt_file = os.listdir(self.labels_folder_path)
        if self.dataset_type == "dota":
            for txt in txt_file:
                txt_full_path = os.path.join(self.labels_folder_path, txt)
                img_full_path = os.path.join(
                    self.images_folder_path, txt.split(".")[0] + ".png")
                self.__DOTA2VOC(img_full_path, txt_full_path)
        elif self.dataset_type == "visdrone":
            for txt in txt_file:
                txt_full_path = os.path.join(self.labels_folder_path, txt)
                img_full_path = os.path.join(
                    self.images_folder_path, txt.split(".")[0] + ".jpg")
                self.__VisDrone2VOC(img_full_path, txt_full_path)
        elif self.dataset_type == "vhr":
            for txt in txt_file:
                txt_full_path = os.path.join(self.labels_folder_path, txt)
                img_full_path = os.path.join(
                    self.images_folder_path, txt.split(".")[0] + ".jpg")
                self.__VHR2VOC(img_full_path, txt_full_path)
        else:
            print("Not yet supported!")

    def convert2Yolo(self, classes=None):
        txt_file = os.listdir(self.labels_folder_path)
        if self.dataset_type == "voc":
            for txt in txt_file:
                txt_full_path = os.path.join(self.labels_folder_path, txt)
                img_full_path = os.path.join(
                    self.images_folder_path, txt.split(".")[0] + ".jpg")
                self.__VOC2YOLO(img_full_path, txt_full_path, classes)
        elif self.dataset_type == "visdrone":
            for txt in txt_file:
                txt_full_path = os.path.join(self.labels_folder_path, txt)
                img_full_path = os.path.join(
                    self.images_folder_path, txt.split(".")[0] + ".jpg")
                self.__VisDrone2YOLO(img_full_path, txt_full_path)
        elif self.dataset_type == "dota":
            for txt in txt_file:
                txt_full_path = os.path.join(self.labels_folder_path, txt)
                img_full_path = os.path.join(
                    self.images_folder_path, txt.split(".")[0] + ".png")
                self.__DOTA2YOLO(img_full_path, txt_full_path)
        elif self.dataset_type == "vhr":
            for txt in txt_file:
                txt_full_path = os.path.join(self.labels_folder_path, txt)
                img_full_path = os.path.join(
                    self.images_folder_path, txt.split(".")[0] + ".jpg")
                self.__VHR2YOLO(img_full_path, txt_full_path)
        else:
            print("Not yet supported!")

    def Prepare4Darknet(self, dst_path, part):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        os.makedirs(os.path.join(
            dst_path, self.dataset_name, part))
        for i in os.listdir(self.images_folder_path):
            os.symlink(os.path.join(self.images_folder_path, i), os.path.join(
                dst_path, self.dataset_name, part, i))
            print(os.path.join(self.images_folder_path, i), os.path.join(
                dst_path, self.dataset_name, part, i))
        for i in os.listdir(self.labels_folder_path):
            os.symlink(os.path.join(self.images_folder_path, i), os.path.join(
                dst_path, self.dataset_name, part, i))
        with open(os.path.join(dst_path, self.dataset_name, part + ".txt"), "w") as output_list_file:
            for i in os.listdir(self.images_folder_path):
                output_list_file.write(os.path.abspath(
                    os.path.join(self.images_folder_path, i)) + "\n")

    def Prepare4YoloV5(self, dst_path, sub_folder):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        os.makedirs(os.path.join(dst_path, self.dataset_name,
                                 "image", part))
        os.makedirs(os.path.join(dst_path, self.dataset_name,
                                 "label", part))
        for i in os.listdir(self.images_folder_path):
            os.symlink(os.path.join(self.images_folder_path, i), os.path.join(
                dst_path, self.dataset_name, "image", part, i))
        for i in os.listdir(self.labels_folder_path):
            os.symlink(os.path.join(self.images_folder_path, i), os.path.join(
                dst_path, self.dataset_name, "label", part, i))

    def __del__(self):
        del self
        print("bye!")


if __name__ == "__main__":
    pass
