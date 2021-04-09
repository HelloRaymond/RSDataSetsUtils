from RSDataConvert import RSDataConverter, RSDataSetPrepare

DOTAConvert = RSDataConverter(
    dataset_type="DOTA", images_folder_path="TestDataSet/DOTA/train/images", labels_folder_path="TestDataSet/DOTA/train/labelTxt")
DOTAConvert.convert2VOC()
DOTAConvert.convert2Yolo()
DOTAYolo = RSDataSetPrepare(dataset_name="DOTA", dataset_type="train",
                            images_folder_path="TestDataSet/DOTA/train/images", labels_folder_path="TestDataSet/DOTA/train/labelTxt")
DOTAYolo.Prepare4Darknet("Darknet")
DOTAYolo.Prepare4YoloV5("YoloV5")

VisDroneConvert = RSDataConverter(dataset_type="VisDrone", images_folder_path="TestDataSet/VisDrone2019/train/images",
                                  labels_folder_path="TestDataSet/VisDrone2019/train/annotations")
VisDroneConvert.convert2VOC()
VisDroneConvert.convert2Yolo()
VisDroneYolo = RSDataSetPrepare(dataset_name="VisDrone2019", dataset_type="train",
                                images_folder_path="TestDataSet/VisDrone2019/train/images", labels_folder_path="TestDataSet/VisDrone2019/train/annotations")
VisDroneYolo.Prepare4Darknet("Darknet")
VisDroneYolo.Prepare4YoloV5("YoloV5")

VisDroneVOCConvert = RSDataConverter(
    dataset_type="VOC", images_folder_path="TestDataSet/VisDrone2019_VOC/train/images", labels_folder_path="TestDataSet/VisDrone2019_VOC/train/VOC")
VisDroneVOCConvert.convert2Yolo(classes=["ignored regions", "pedestrian", "people", "bicycle",
                                "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"])

VHRConvert = RSDataConverter(dataset_type="VHR", images_folder_path="TestDataSet/NWPU VHR-10/train/positive image set",
                             labels_folder_path="TestDataSet/NWPU VHR-10/train/ground truth")
VHRConvert.convert2VOC()
VHRConvert.convert2Yolo()
VHRYolo = RSDataSetPrepare(dataset_name="VHR-10", dataset_type="train", images_folder_path="TestDataSet/NWPU VHR-10/train/positive image set",
                           labels_folder_path="TestDataSet/NWPU VHR-10/train/ground truth")
VHRYolo.Prepare4Darknet("Darknet")
VHRYolo.Prepare4YoloV5("YoloV5")
