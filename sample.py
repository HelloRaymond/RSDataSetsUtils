from RSDataConvert import RSDataConverter

DOTAConvert = RSDataConverter(
    dataset_type='DOTA', images_folder_path='TestDataSet/DOTA/train/images', labels_folder_path='TestDataSet/DOTA/train/labelTxt')
DOTAConvert.convert2VOC()
DOTAConvert.convert2Yolo()

VisDroneConvert = RSDataConverter(dataset_type='VisDrone', images_folder_path='TestDataSet/VisDrone2019/train/images',
                                  labels_folder_path='TestDataSet/VisDrone2019/train/annotations')
VisDroneConvert.convert2VOC()
VisDroneConvert.convert2Yolo()

VisDroneVOCConvert = RSDataConverter(
    dataset_type='VOC', images_folder_path='TestDataSet/VisDrone2019_VOC/train/images', labels_folder_path='TestDataSet/VisDrone2019_VOC/train/VOC')
VisDroneVOCConvert.convert2Yolo(classes=["ignored regions", "pedestrian", "people", "bicycle",
                                "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"])

VHRConvert = RSDataConverter(dataset_type='VHR', images_folder_path='TestDataSet/NWPU VHR-10/train/positive image set',
                             labels_folder_path='TestDataSet/NWPU VHR-10/train/ground truth')
VHRConvert.convert2VOC()
VHRConvert.convert2Yolo()
