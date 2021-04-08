"""
This script is used to process remote sensing image data sets;
The types of data sets currently supported are: PASCAL VOC, VisDrone, DOTA, NWPU VHR-10, YOLO
"""

import os
import cv2
import time
from xml.dom import minidom
import xml.etree.ElementTree as ET


class RSDataConverter:
    def __init__(self, dataset_type, images_folder_path, labels_folder_path):
        self.dataset_type = dataset_type.lower()
        if self.dataset_type not in ["visdrone", "voc", "dota", "yolo", "vhr"]:
            print("Not yet supported!")
            self.__del__()
        self.images_folder_path = images_folder_path
        self.labels_folder_path = labels_folder_path

    def __BoxConvert4Yolo(self, img_w, img_h, x_min, x_max, y_min, y_max):
        dw = 1.0/img_w
        dh = 1.0/img_h
        x = (x_min + x_max)/2.0 - 1
        y = (y_min + y_max)/2.0 - 1
        w = x_max - x_min
        h = y_max - y_min
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return x, y, w, h

    def __WriteVOCLabelXml(self, img_name, img_folder, img_w, img_h, img_d, objects, xml_path):
        # dict objects has "object_name", "is_truncated", "is_difficult", "x_min", "y_min", "x_max", "y_max" keys
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
        width.appendChild(doc.createTextNode(str(img_w)))
        size.appendChild(width)
        height = doc.createElement("height")
        height.appendChild(doc.createTextNode(str(img_h)))
        size.appendChild(height)
        depth = doc.createElement("depth")
        depth.appendChild(doc.createTextNode(str(img_d)))
        size.appendChild(depth)
        annotation.appendChild(size)
        segmented = doc.createElement("segmented")
        segmented.appendChild(doc.createTextNode("0"))
        annotation.appendChild(segmented)
        for box in objects:
            object = doc.createElement("object")
            nm = doc.createElement("name")
            try:
                nm.appendChild(doc.createTextNode(box["object_name"]))
            except KeyError:
                print("object_name key is necessary!")
                continue
            object.appendChild(nm)
            pose = doc.createElement("pose")
            pose.appendChild(doc.createTextNode("Unspecified"))
            object.appendChild(pose)
            truncated = doc.createElement("truncated")
            try:
                truncated.appendChild(
                    doc.createTextNode(str(box["is_truncated"])))
            except KeyError:
                truncated.appendChild(doc.createTextNode("0"))
            object.appendChild(truncated)
            difficult = doc.createElement("difficult")
            try:
                difficult.appendChild(
                    doc.createTextNode(str(box["is_difficult"])))
            except KeyError:
                difficult.appendChild(doc.createTextNode("0"))
            object.appendChild(difficult)
            bndbox = doc.createElement("bndbox")
            xmin = doc.createElement("xmin")
            try:
                xmin.appendChild(doc.createTextNode(str(int(box["x_min"]))))
            except KeyError:
                print("x_min key is necessary!")
                continue
            bndbox.appendChild(xmin)
            ymin = doc.createElement("ymin")
            try:
                ymin.appendChild(doc.createTextNode(str(int(box["y_min"]))))
            except KeyError:
                print("y_min key is necessary!")
                continue
            bndbox.appendChild(ymin)
            xmax = doc.createElement("xmax")
            try:
                xmax.appendChild(doc.createTextNode(str(int(box["x_max"]))))
            except KeyError:
                print("x_max key is necessary!")
                continue
            bndbox.appendChild(xmax)
            ymax = doc.createElement("ymax")
            try:
                ymax.appendChild(doc.createTextNode(str(int(box["y_max"]))))
            except KeyError:
                print("y_max key is necessary!")
                continue
            bndbox.appendChild(ymax)
            object.appendChild(bndbox)
            annotation.appendChild(object)
            with open(xml_path, "w") as output_label_file:
                output_label_file.write(doc.toprettyxml())

    def __WritrYoloLabelTxt(self, objects, label_txt_path):
        with open(label_txt_path, "w") as output_label_file:
            for box in objects:
                try:
                    output_label_file.write("%d %.16f %.16f %.16f %.16f\n" % (
                        box["cls_id"], box["x_center"], box["y_center"], box["width"], box["height"]))
                except KeyError:
                    print(
                        "cls_id, x_center, y_center, width, height keys are necessary!")
                    continue

    def __VOC2YOLO(self, img_path, label_path, classes):
        output_file_name = os.path.basename(label_path).split(".")[0] + ".txt"
        save_path = os.path.join(self.labels_folder_path, "../YOLO")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        objects = []
        with open(label_path, "r") as input_label_file:
            tree = ET.parse(input_label_file)
            root = tree.getroot()
            size = root.find("size")
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)
            for obj in root.iter("object"):
                difficult = obj.find("difficult").text
                cls = obj.find("name").text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find("bndbox")
                x_min = float(xmlbox.find("xmin").text)
                x_max = float(xmlbox.find("xmax").text)
                y_min = float(xmlbox.find("ymin").text)
                y_max = float(xmlbox.find("ymax").text)
                x, y, w, h = self.__BoxConvert4Yolo(
                    img_w, img_h, x_min, x_max, y_min, y_max)
                objects.append({"cls_id": cls_id, "x_center": x,
                               "y_center": y, "width": w, "height": h})
        self.__WritrYoloLabelTxt(
            objects, os.path.join(save_path, output_file_name))

    def __DOTA2VOC(self, img_path, label_path):
        output_file_name = os.path.basename(label_path).split(".")[0] + ".xml"
        save_path = os.path.join(self.labels_folder_path, "../VOC")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img = cv2.imread(img_path)
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_d = img.shape[2]
        objects = []
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
                    object_name = box[8]
                    is_difficult = 0 if box[9] == "0" else 1
                    objects.append({"object_name": object_name, "is_truncated": 0, "is_difficult": is_difficult,
                                   "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        self.__WriteVOCLabelXml(os.path.basename(img_path), "images", img_w,
                                img_h, img_d, objects, os.path.join(save_path, output_file_name))

    def __DOTA2YOLO(self, img_path, label_path, classes=["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool", "container-crane", "airport", "helipad"]):
        output_file_name = os.path.basename(label_path)
        save_path = os.path.join(self.labels_folder_path, "../YOLO")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img = cv2.imread(img_path)
        img_w = img.shape[1]
        img_h = img.shape[0]
        objects = []
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
                    object_num = classes.index(box[8])
                    x, y, w, h = self.__BoxConvert4Yolo(
                        img_w, img_h, x_min, x_max, y_min, y_max)
                    objects.append(
                        {"cls_id": object_num, "x_center": x, "y_center": y, "width": w, "height": h})
        self.__WritrYoloLabelTxt(
            objects, os.path.join(save_path, output_file_name))

    def __VisDrone2VOC(self, img_path, label_path, classes=["ignored regions", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"]):
        output_file_name = os.path.basename(label_path).split(".")[0] + ".xml"
        save_path = os.path.join(self.labels_folder_path, "../VOC")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        objects = []
        img = cv2.imread(img_path)
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_d = img.shape[2]
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
                    object_name = classes[int(box[5])]
                    is_truncated = 1 if int(
                        box[6]) > 0 or int(box[7]) > 0 else 0
                    is_difficult = 1 if int(box[7]) > 1 else 0
                    objects.append({"object_name": object_name, "is_truncated": is_truncated,
                                   "is_difficult": is_difficult, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        self.__WriteVOCLabelXml(os.path.basename(img_path), "images", img_w,
                                img_h, img_d, objects, os.path.join(save_path, output_file_name))

    def __VisDrone2YOLO(self, img_path, label_path):
        output_file_name = os.path.basename(label_path)
        save_path = os.path.join(self.labels_folder_path, "../YOLO")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        objects = []
        img = cv2.imread(img_path)
        img_w = img.shape[1]
        img_h = img.shape[0]
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
                    x, y, w, h = self.__BoxConvert4Yolo(
                        img_w, img_h, x_min, x_max, y_min, y_max)
                    objects.append(
                        {"cls_id": cls_id, "x_center": x, "y_center": y, "width": w, "height": h})
            self.__WritrYoloLabelTxt(
                objects, os.path.join(save_path, output_file_name))

    def __VHR2VOC(self, img_path, label_path, classes=["airplane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor", "bridge", "vehicle"]):
        output_file_name = os.path.basename(label_path).split(".")[0] + ".xml"
        save_path = os.path.join(self.labels_folder_path, "../VOC")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        objects = []
        img = cv2.imread(img_path)
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_d = img.shape[2]
        with open(label_path, "r") as input_label_file:
            lines = [input_label_file.readlines()]
            for line in lines:
                for boxes in line:
                    box = boxes.strip("\n")
                    box = box.split(",")
                    x_min = int(box[0].replace('(', ''))
                    y_min = int(box[1].replace(')', ''))
                    x_max = int(box[2].replace('(', ''))
                    y_max = int(box[3].replace(')', ''))
                    object_name = classes[int(box[4])]
                    objects.append({"object_name": object_name, "is_truncated": 0, "is_difficult": 0,
                                   "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        self.__WriteVOCLabelXml(os.path.basename(img_path), "images", img_w,
                                img_h, img_d, objects, os.path.join(save_path, output_file_name))

    def __VHR2YOLO(self, img_path, label_path):
        output_file_name = os.path.basename(label_path)
        save_path = os.path.join(self.labels_folder_path, "../YOLO")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        objects = []
        img = cv2.imread(img_path)
        img_w = img.shape[1]
        img_h = img.shape[0]
        with open(label_path, "r") as input_label_file:
            lines = [input_label_file.readlines()]
            for line in lines:
                for boxes in line:
                    box = boxes.strip("\n")
                    box = box.split(",")
                    x_min = int(box[0].replace('(', ''))
                    y_min = int(box[1].replace(')', ''))
                    x_max = int(box[2].replace('(', ''))
                    y_max = int(box[3].replace(')', ''))
                    cls_id = int(box[4])
                    x, y, w, h = self.__BoxConvert4Yolo(
                        img_w, img_h, x_min, x_max, y_min, y_max)
                    objects.append(
                        {"cls_id": cls_id, "x_center": x, "y_center": y, "width": w, "height": h})
            self.__WritrYoloLabelTxt(
                objects, os.path.join(save_path, output_file_name))

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

    def __del__(self):
        del self
        print("bye!")


if __name__ == "__main__":
    pass
