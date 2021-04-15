#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

RSDataSetsUtils

~~~~~~~~~~~~~~~~

This script provides some utils for remote sensing image data sets processing;

The types of data sets currently supported are: PASCAL VOC, VisDrone, DOTA, NWPU VHR-10, YOLO.

"""

import os
import cv2


class YoloBoudingBox(object):

    def __init__(self, center_x, center_y, width, height, cls_id):
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.width = float(width)
        self.height = float(height)
        self.cls_id = int(cls_id)


class VOCBoudingBox(object):

    def __init__(self, x_min, x_max, y_min, y_max, cls_name, is_difficult=False, is_truncated=False):
        self.x_min = int(x_min)
        self.x_max = int(x_max)
        self.y_min = int(y_min)
        self.y_max = int(y_max)
        self.cls_name = str(cls_name)
        self.is_difficult = bool(is_difficult)
        self.is_truncated = bool(is_truncated)


class VOCLabel(object):

    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path
        self.BBoxes = []
        self.img_w = 0
        self.img_h = 0
        self.img_d = 0

    def updateSize(self):
        img = cv2.imread(self.img_path)
        self.img_w = img.shape[1]
        self.img_h = img.shape[0]
        self.img_d = img.shape[2]
        return self.img_h, self.img_w, self.img_d

    def updateBBoxes(self):
        import xml.etree.ElementTree as ET
        with open(self.label_path, "r") as input_label_file:
            tree = ET.parse(input_label_file)
            root = tree.getroot()
            for obj in root.iter("object"):
                try:
                    diffi = False if obj.find("difficult").text == "0" else True
                except:
                    diffi = False
                try:
                    trunc = False if obj.find("truncated").text == "0" else True
                except:
                    trunc = False
                clsname = obj.find("name").text
                xmlbox = obj.find("bndbox")
                x1 = float(xmlbox.find("xmin").text)
                x2 = float(xmlbox.find("xmax").text)
                y1 = float(xmlbox.find("ymin").text)
                y2 = float(xmlbox.find("ymax").text)
                self.BBoxes.append(VOCBoudingBox(x1, x2, y1, y2, clsname, diffi, trunc))
        return self.BBoxes

    def convert2YoloLabel(self, clss):
        self.updateSize()
        dw = 1.0/self.img_w
        dh = 1.0/self.img_h
        yolo_label = YoloLabel(self.img_path, self.label_path)
        for box in self.BBoxes:
            x = (box.x_min + box.x_max)/2.0 - 1
            y = (box.y_min + box.y_max)/2.0 - 1
            w = box.x_max - box.x_min
            h = box.y_max - box.y_min
            x = x*dw
            w = w*dw
            y = y*dh
            h = h*dh
            yolo_label.BBoxes.append(YoloBoudingBox(x, y, w, h, clss.index(box.cls_name)))
        return yolo_label


class YoloLabel(VOCLabel):

    def __init__(self, img_path, label_path):
        super(YoloLabel, self).__init__(img_path, label_path)

    def updateBBoxes(self):
        with open(self.label_path, "r") as input_label_file:
            lines = [input_label_file.readlines()]
            for line in lines:
                for boxes in line:
                    box = boxes.strip("\n")
                    box = box.split(" ")
                    clsid = int(box[0])
                    x = float(box[1])
                    y = float(box[2])
                    w = float(box[3])
                    h = float(box[4])
                    self.BBoxes.append(YoloBoudingBox(x, y, w, h, clsid))
        return self.BBoxes

    def convert2VOCLabel(self, clss):
        self.updateSize()
        voc_label = VOCLabel(self.img_path, self.label_path)
        for box in self.BBoxes:
            x1 = int(box.center_x * self.img_w - box.bbox_width * self.img_w / 2)
            x2 = int(box.center_x * self.img_w + box.bbox_width * self.img_w / 2)
            y1 = int(box.center_y * self.img_h - box.bbox_height * self.img_h / 2)
            y2 = int(box.center_y * self.img_h + box.bbox_height * self.img_h / 2)
            voc_label.BBoxes.append(VOCBoudingBox(x1, x2, y1, y2, clss[box.cls_id]))
        return voc_label


class RSDataSetsUtils:

    @staticmethod
    def writeVOCXmlLabel(voc_label, xml_label_path):
        from xml.dom import minidom
        voc_label.updateSize()
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
            difficult.appendChild(doc.createTextNode("1" if box.is_difficult else "0"))
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
            with open(xml_label_path, "w") as output_label_file:
                output_label_file.write(doc.toprettyxml())

    @staticmethod
    def writeYoloTxtLabel(yolo_label, label_txt_path):
        with open(label_txt_path, "w") as output_label_file:
            for box in yolo_label.BBoxes:
                output_label_file.write("%d %.16f %.16f %.16f %.16f\n" % (box.cls_id, box.center_x, box.center_y, box.width, box.height))

    @staticmethod
    def getDOTA(img_path, label_path):
        voc_label = VOCLabel(img_path, label_path)
        with open(label_path, "r") as input_label_file:
            lines = [input_label_file.readlines()]
            for line in lines:
                for boxes in line:
                    if "imagesource" in boxes or "gsd" in boxes:
                        continue
                    box = boxes.strip("\n")
                    box = box.split(" ")
                    x_min = min(float(box[0]), float(box[2]), float(box[4]), float(box[6]))
                    y_min = min(float(box[1]), float(box[3]), float(box[5]), float(box[7]))
                    x_max = max(float(box[0]), float(box[2]), float(box[4]), float(box[6]))
                    y_max = max(float(box[1]), float(box[3]), float(box[5]), float(box[7]))
                    cls_name = box[8]
                    is_difficult = False if box[9] == "0" else True
                    voc_label.BBoxes.append(VOCBoudingBox(x_min, x_max, y_min, y_max, cls_name, is_difficult))
        return voc_label

    @staticmethod
    def getVisDrone(img_path, label_path):
        classes = ["ignored regions", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"]
        voc_label = VOCLabel(img_path, label_path)
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
                    is_truncated = True if int(box[6]) > 0 or int(box[7]) > 0 else False
                    is_difficult = True if int(box[7]) > 1 else False
                    voc_label.BBoxes.append(VOCBoudingBox(x_min, x_max, y_min, y_max, cls_name, is_difficult, is_truncated))
        return voc_label

    @staticmethod
    def getVHR(img_path, label_path):
        classes = ["airplane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor", "bridge", "vehicle"]
        voc_label = VOCLabel(img_path, label_path)
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
                    voc_label.BBoxes.append(VOCBoudingBox(x_min, x_max, y_min, y_max, cls_name))
        return voc_label

    @staticmethod
    def fillRight(img, size_w):
        size = img.shape
        img_fill_right = cv2.copyMakeBorder(img, 0, 0, 0, size_w - size[1], cv2.BORDER_CONSTANT, value=(107, 113, 115))
        return img_fill_right

    @staticmethod
    def fillBottom(img, size_h):
        size = img.shape
        img_fill_bottom = cv2.copyMakeBorder(img, 0, size_h - size[0], 0, 0, cv2.BORDER_CONSTANT, value=(107, 113, 115))
        return img_fill_bottom

    @staticmethod
    def fillRightBottom(img, size_w, size_h):
        size = img.shape
        img_fill_right_bottom = cv2.copyMakeBorder(img, 0, size_h - size[0], 0, size_w - size[1], cv2.BORDER_CONSTANT, value=(107, 113, 115))
        return img_fill_right_bottom

    @staticmethod
    def isBetween(a, b):
        flag = True
        for i in a:
            if i < b[0] or i > b[1]:
                flag = False
        return flag

    @staticmethod
    def plotLabeledImg(voc_label):
        img = cv2.imread(voc_label.img_path)
        for box in voc_label.BBoxes:
            cv2.putText(img,box.cls_name,(box.x_min, box.y_min), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 255), thickness=1)
            cv2.rectangle(img, (box.x_min, box.y_min), (box.x_max, box.y_max), (0, 255, 0), 1)
        cv2.imshow(voc_label.img_path, img)
        cv2.waitKey()

    @staticmethod
    def getImgExtName(name, path):
        ext_list = [".jpg", ".png", ".bmp", ".jpeg", ".jpe", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif", ".exr", "jp2"]
        for i in ext_list:
            if os.path.exists(path + name):
                return i


class RSDataSetsConverter(RSDataSetsUtils):

    def __init__(self, dataset_name, dataset_type, images_folder_path, labels_folder_path):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type.lower()
        if self.dataset_type not in ["visdrone", "voc", "dota", "yolo", "vhr"]:
            print("type: \"%s\", Not yet supported!" %(dataset_type))
            self.__del__()
        self.images_folder_path = images_folder_path
        self.labels_folder_path = labels_folder_path

    def convert2VOC(self):
        file_list = os.listdir(self.labels_folder_path)
        if self.dataset_type == "yolo":
            for label_filename in file_list:
                if os.path.splitext(label_filename)[-1] != ".txt":
                    continue
                label_full_path = os.path.join(self.labels_folder_path, label_filename)
                img_full_path = os.path.join(self.images_folder_path, os.path.splitext(label_filename)[0] + self.getImgExtName(os.path.splitext(label_filename)[0], self.images_folder_path))
                yolo_label = YoloLabel(img_full_path, label_full_path)
                yolo_label.updateSize()
                yolo_label.updateBBoxes()
                voc_label = yolo_label.convert2VOCLabel(classes)
                output_file_name = os.path.splitext(os.path.basename(img_full_path))[0] + ".txt"
                save_path = os.path.join(self.labels_folder_path, "../YOLO")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.writeYoloTxtLabel(yolo_label, os.path.join(save_path, output_file_name))
        if self.dataset_type == "dota":
            for label_filename in file_list:
                if os.path.splitext(label_filename)[-1] != ".txt":
                    continue
                label_full_path = os.path.join(self.labels_folder_path, label_filename)
                img_full_path = os.path.join(self.images_folder_path, os.path.splitext(label_filename)[0] + ".png")
                voc_label = self.getDOTA(img_full_path, label_full_path)
                output_file_name = os.path.splitext(os.path.basename(img_full_path))[0] + ".xml"
                save_path = os.path.join(self.labels_folder_path, "../VOC")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.writeVOCXmlLabel(voc_label, os.path.join(save_path, output_file_name))
        elif self.dataset_type == "visdrone":
            for label_filename in file_list:
                if os.path.splitext(label_filename)[-1] != ".txt":
                    continue
                label_full_path = os.path.join(self.labels_folder_path, label_filename)
                img_full_path = os.path.join(self.images_folder_path, os.path.splitext(label_filename)[0] + ".jpg")
                voc_label = self.getVisDrone(img_full_path, label_full_path)
                output_file_name = os.path.splitext(os.path.basename(img_full_path))[0] + ".xml"
                save_path = os.path.join(self.labels_folder_path, "../VOC")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.writeVOCXmlLabel(voc_label, os.path.join(save_path, output_file_name))
        elif self.dataset_type == "vhr":
            for label_filename in file_list:
                if os.path.splitext(label_filename)[-1] != ".txt":
                    continue
                label_full_path = os.path.join(self.labels_folder_path, label_filename)
                img_full_path = os.path.join(self.images_folder_path, os.path.splitext(label_filename)[0] + ".jpg")
                voc_label = self.getVHR(img_full_path, label_full_path)
                output_file_name = os.path.splitext(os.path.basename(img_full_path))[0] + ".xml"
                save_path = os.path.join(self.labels_folder_path, "../VOC")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.writeVOCXmlLabel(voc_label, os.path.join(save_path, output_file_name))
        else:
            print("type: \"%s\" to voc, Not yet supported!" %(self.dataset_type))

    def convert2Yolo(self, classes = None):
        file_list = os.listdir(self.labels_folder_path)
        if self.dataset_type == "voc":
            for label_filename in file_list:
                if os.path.splitext(label_filename)[-1] != ".xml":
                    continue
                label_full_path = os.path.join(self.labels_folder_path, label_filename)
                img_full_path = os.path.join(self.images_folder_path, os.path.splitext(label_filename)[0] + ".jpg")
                voc_label = VOCLabel(img_full_path, label_full_path)
                voc_label.updateSize()
                voc_label.updateBBoxes()
                yolo_label = voc_label.convert2YoloLabel(classes)
                output_file_name = os.path.splitext(os.path.basename(img_full_path))[0] + ".txt"
                save_path = os.path.join(self.labels_folder_path, "../YOLO")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.writeYoloTxtLabel(yolo_label, os.path.join(save_path, output_file_name))
        if self.dataset_type == "dota":
            classes = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool", "container-crane", "airport", "helipad"]
            for label_filename in file_list:
                if os.path.splitext(label_filename)[-1] != ".xml":
                    continue
                label_full_path = os.path.join(self.labels_folder_path, label_filename)
                img_full_path = os.path.join(self.images_folder_path, os.path.splitext(label_filename)[0] + ".png")
                voc_label = self.getDOTA(img_full_path, label_full_path)
                yolo_label = voc_label.convert2YoloLabel(classes)
                output_file_name = os.path.splitext(os.path.basename(img_full_path))[0] + ".txt"
                save_path = os.path.join(self.labels_folder_path, "../YOLO")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.writeYoloTxtLabel(yolo_label, os.path.join(save_path, output_file_name))
        elif self.dataset_type == "visdrone":
            classes = ["ignored regions", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"]
            for label_filename in file_list:
                if os.path.splitext(label_filename)[-1] != ".xml":
                    continue
                label_full_path = os.path.join(self.labels_folder_path, label_filename)
                img_full_path = os.path.join(self.images_folder_path, os.path.splitext(label_filename)[0] + ".jpg")
                voc_label = self.getVisDrone(img_full_path, label_full_path)
                yolo_label = voc_label.convert2YoloLabel(classes)
                output_file_name = os.path.splitext(os.path.basename(img_full_path))[0] + ".txt"
                save_path = os.path.join(self.labels_folder_path, "../YOLO")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.writeYoloTxtLabel(yolo_label, os.path.join(save_path, output_file_name))
        elif self.dataset_type == "vhr":
            classes = ["airplane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor", "bridge", "vehicle"]
            for label_filename in file_list:
                if os.path.splitext(label_filename)[-1] != ".xml":
                    continue
                label_full_path = os.path.join(self.labels_folder_path, label_filename)
                img_full_path = os.path.join(
                    self.images_folder_path, os.path.splitext(label_filename)[0] + ".jpg")
                voc_label = self.getVHR(img_full_path, label_full_path)
                yolo_label = voc_label.convert2YoloLabel(classes)
                output_file_name = os.path.splitext(os.path.basename(img_full_path))[0] + ".txt"
                save_path = os.path.join(self.labels_folder_path, "../YOLO")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.writeYoloTxtLabel(yolo_label, os.path.join(save_path, output_file_name))
        else:
            print("type: \"%s\" to yolo, Not yet supported!" %(self.dataset_type))

    def Prepare4Darknet(self, dst_path, part):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        os.makedirs(os.path.join(dst_path, self.dataset_name, part))
        for i in os.listdir(self.images_folder_path):
            os.symlink(os.path.join(self.images_folder_path, i), os.path.join(dst_path, self.dataset_name, part, i))
        for i in os.listdir(self.labels_folder_path):
            os.symlink(os.path.join(self.images_folder_path, i), os.path.join(dst_path, self.dataset_name, part, i))
        with open(os.path.join(dst_path, self.dataset_name, part + ".txt"), "w") as output_list_file:
            for i in os.listdir(self.images_folder_path):
                output_list_file.write(os.path.abspath(os.path.join(self.images_folder_path, i)) + "\n")

    def Prepare4YoloV5(self, dst_path, part):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        os.makedirs(os.path.join(dst_path, self.dataset_name, "image", part))
        for i in os.listdir(self.images_folder_path):
            os.symlink(os.path.abspath(os.path.join(self.images_folder_path, i)), os.path.join(dst_path, self.dataset_name, "image", part, i))
        for i in os.listdir(self.labels_folder_path):
            os.symlink(os.path.abspath(os.path.join(self.labels_folder_path, i)), os.path.join(dst_path, self.dataset_name, "label", part, i))


class RSDataSetsSpliter(RSDataSetsUtils):

    def __init__(self, oringin_imgs_folder_path, oringin_labels_folder_path):
        self.oringin_imgs_folder_path = oringin_imgs_folder_path
        self.oringin_labels_folder_path = oringin_labels_folder_path

    def splitImgs(self, img_dst_path, size_w, size_h, step):
        if not os.path.exists(img_dst_path):
            os.makedirs(img_dst_path)
        img_list = os.listdir(self.oringin_imgs_folder_path)
        for img_name in img_list:
            name = os.path.splitext(img_name)[0]
            img = cv2.imread(os.path.join(self.oringin_imgs_folder_path, img_name))
            size = img.shape
            if size[0] >= size_h and size[1] >= size_w:
                for h in range(0, size[0] - 1, step):
                    start_h = h
                    for w in range(0, size[1] - 1, step):
                        start_w = w
                        end_h = start_h + size_h
                        if end_h > size[0]:
                            start_h = size[0] - size_h
                            end_h = start_h + size_h
                        end_w = start_w + size_w
                        if end_w > size[1]:
                            start_w = size[1] - size_w
                        end_w = start_w + size_w
                        cropped = img[start_h: end_h, start_w: end_w]
                        name_img = name + "_" + \
                            str(start_h) + "_" + str(start_w)
                        cv2.imwrite(os.path.join(img_dst_path, name_img + ".png"), cropped)
            elif size[0] >= size_h and size[1] < size_w:
                img0 = self.fillRight(img, size_w)
                for h in range(0, size[0] - 1, step):
                    start_h = h
                    start_w = 0
                    end_h = start_h + size_h
                    if end_h > size[0]:
                        start_h = size[0] - size_h
                        end_h = start_h + size_h
                    end_w = start_w + size_w
                    cropped = img0[start_h: end_h, start_w: end_w]
                    name_img = name + "_" + str(start_h) + "_" + str(start_w)
                    cv2.imwrite(os.path.join(img_dst_path, name_img + ".png"), cropped)
            elif size[0] < size_h and size[1] >= size_w:
                img0 = self.fillBottom(img, size_h)
                for w in range(0, size[1] - 1, step):
                    start_h = 0
                    start_w = w
                    end_w = start_w + size_w
                    if end_w > size[1]:
                        start_w = size[1] - size_w
                        end_w = start_w + size_w
                    end_h = start_h + size_h
                    cropped = img0[start_h: end_h, start_w: end_w]
                    name_img = name + "_" + str(start_h) + "_" + str(start_w)
                    cv2.imwrite(os.path.join(img_dst_path, name_img + ".png"), cropped)
            elif size[0] < size_h and size[1] < size_w:
                img0 = self.fillRightBottom(img, size_w, size_h)
                cropped = img0[0: size_h, 0: size_w]
                name_img = name + "_" + "0" + "_" + "0"
                cv2.imwrite(os.path.join(img_dst_path, name_img + ".png"), cropped)

    def getVOCLabelfromOrig(self, oringin_voc_label, out_img_floder, out_label_floder, start_x, start_y, size_w, size_h):
        oringin_voc_label.updateBBoxes()
        img_name = os.path.splitext(os.path.split(oringin_voc_label.img_path)[1])
        label_name = os.path.splitext(os.path.split(oringin_voc_label.label_path)[1])
        img_name = os.path.join(out_img_floder, img_name[0] + "_%s_%s" %(start_y, start_x) + img_name[1])
        label_name = os.path.join(out_label_floder, label_name[0] + "_%s_%s" %(start_y, start_x) + label_name[1])
        voc_label = VOCLabel(img_name, label_name)
        for box in oringin_voc_label.BBoxes:
            if self.isBetween((box.x_min, box.x_max), (start_x, start_x + size_w)) and self.isBetween((box.y_min, box.y_max), (start_y, start_y + size_h)):
                voc_label.BBoxes.append(VOCBoudingBox(box.x_min - start_x, box.x_max - start_x, box.y_min - start_y, box.y_max - start_y, box.cls_name, box.is_difficult, box.is_truncated))
        return voc_label

    def splitLabels(self, img_floder, out_img_floder, label_floder, out_label_floder, size_h, size_w):
        if not os.path.exists(out_label_floder):
            os.makedirs(out_label_floder)
        img_list = os.listdir(out_img_floder)
        for img_name in img_list:
            name = os.path.splitext(img_name)[0]
            name_list = name.split('_')
            txt_name = name_list[0]
            y = int(name_list[1])
            x = int(name_list[2])
            oringin_voc_label = VOCLabel(os.path.join(img_floder, txt_name + ".png"), os.path.join(label_floder, txt_name + ".xml"))
            voc_label = self.getVOCLabelfromOrig(oringin_voc_label, out_img_floder, out_label_floder, x, y, size_w, size_h)
            self.writeVOCXmlLabel(voc_label, os.path.join(out_label_floder, name + ".xml"))

    def deleteEmptySample(self, images_folder_path, labels_folder_path):
        txt_list = os.listdir(images_folder_path)
        for txt_name in txt_list:
            name = os.path.splitext(
                (os.path.split(images_folder_path)[1]))[0]
            txt_path = os.path.join(labels_folder_path, txt_name)
            img_path = os.path.join(images_folder_path, name + ".png")
            with open(txt_path, "r") as f:
                data = f.read()
                f.close()
                if(data == ""):
                    os.remove(txt_path)
                    os.remove(img_path)

    def __del__(self):
        del self
        print("bye!")


if __name__ == "__main__":
    pass
