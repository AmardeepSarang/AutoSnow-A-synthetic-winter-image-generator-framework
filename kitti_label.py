
'''
==================================================================================================================================================================
AUTHOR: code from https://github.com/oberger4711/kitti_for_yolo, then modified to keep truck, car, cyclist, ped lables
DESCRIPTION: Used to convert KITTI object label into form YOLO expects the labels in the form: <object-class> <x> <y> <width> <height>
                Also removes any object label not of the classes 'Pedestrian',Cyclist,'Car','Truck'.
                Outputs numpy lists of file # to be split into testing and training data

ARGS:
    label_dir: Path to KITTI download folder for trainig lables in label_2 subfolder, example:  path/to/data_object_label_2/training/label_2 
    image_2_dir: Path to KITTI download folder for trainig lables in label_2 subfolder, example: path/to/data_object_image_2/training/image_2 
    --training-samples (float): Percentage of the samples to be used for training between 0.0 and 1.0, default=0.8
    --use-dont-care (bool flag): Will not ignore 'DontCare' labels when set
EXAMPLE USAGE:
    python kitti_label.py path/to/kitti/data_object_label_2/training/label_2 path/to/kitti/data_object_image_2/training/image_2 --training-samples 0.7

==================================================================================================================================================================
'''
from PIL import Image
import argparse
import os
import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split


OUT_LABELS_DIR = "Data/labels"

KEY_PEDESTRIAN = "Pedestrian"
KEY_CYCLIST = "Cyclist"
KEY_CAR = "Car"
KEY_VAN = "Van"
KEY_MISC = "Misc"
KEY_TRUCK = "Truck"
KEY_PERSON_SITTING = "Person_sitting"
KEY_TRAM = "Tram"
KEY_DONT_CARE = "DontCare"

CLASS_NUMBERS = {
            #'Pedestrian',Cyclist,'Car','Truck' 
            KEY_PEDESTRIAN : 0,
            KEY_CYCLIST : 1,
            KEY_CAR : 2,
            KEY_TRUCK : 3

        }

def getSampleId(path):
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]

def resolveClassNumberOrNone(lable_class, use_dont_care):
    if lable_class == KEY_CYCLIST:
        return CLASS_NUMBERS[KEY_CYCLIST]
    elif lable_class == KEY_TRUCK:
        return CLASS_NUMBERS[KEY_TRUCK]
    #if class in (KEY_PEDESTRIAN, KEY_PERSON_SITTING):
    elif lable_class == KEY_PEDESTRIAN or lable_class==KEY_PERSON_SITTING:
        return CLASS_NUMBERS[KEY_PEDESTRIAN]
    #if class in (KEY_CAR, KEY_VAN):
    elif lable_class == KEY_CAR or lable_class == KEY_VAN:
        return CLASS_NUMBERS[KEY_CAR]
    else:
        return None

def convertToYoloBBox(bbox, size):
    # Yolo uses bounding bbox coordinates and size relative to the image size.
    # This is taken from https://pjreddie.com/media/files/voc_label.py .
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (bbox[0] + bbox[1]) / 2.0
    y = (bbox[2] + bbox[3]) / 2.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def readRealImageSize(img_path):
    # This loads the whole sample image and returns its size.
    return Image.open(img_path).size

def readFixedImageSize():
    # This is not exact for all images but most (and it should be faster).
    return (1242, 375)

def parseSample(lbl_path, img_path, use_dont_care):
    with open(lbl_path) as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=["type", "truncated", "occluded", "alpha", "bbox2_left", "bbox2_top", "bbox2_right", "bbox2_bottom", "bbox3_height", "bbox3_width", "bbox3_length", "bbox3_x", "bbox3_y", "bbox3_z", "bbox3_yaw", "score"], delimiter=" ")
        yolo_labels = []
        for row in reader:
            class_number = resolveClassNumberOrNone(row["type"], use_dont_care)
            if class_number is not None:
                size = readRealImageSize(img_path)
                #size = readFixedImageSize()
                # Image coordinate is in the top left corner.
                bbox = (
                        float(row["bbox2_left"]),
                        float(row["bbox2_right"]),
                        float(row["bbox2_top"]),
                        float(row["bbox2_bottom"])
                       )
                yolo_bbox = convertToYoloBBox(bbox, size)
                # Yolo expects the labels in the form:
                # <object-class> <x> <y> <width> <height>.
                yolo_label = (class_number,) + yolo_bbox
                yolo_labels.append(yolo_label)
    return yolo_labels

def parseArguments():
    parser = argparse.ArgumentParser(description="Generates labels for training darknet on KITTI.")
    parser.add_argument("label_dir", help="data_object_label_2/training/label_2 directory; can be downloaded from KITTI.")
    parser.add_argument("image_2_dir", help="data_object_image_2/training/image_2 directory; can be downloaded from KITTI.")
    parser.add_argument("--training-samples", type=float, default=0.8, help="percentage of the samples to be used for training between 0.0 and 1.0.")
    parser.add_argument("--use-dont-care", action="store_true", help="do not ignore 'DontCare' labels.")
    args = parser.parse_args()
    if args.training_samples < 0 or args.training_samples > 1:
        print("Invalid argument {} for --training-samples. Expected a percentage value between 0.0 and 1.0.")
        exit(-1)
    return args

def main():
    args = parseArguments()

    if not os.path.exists(OUT_LABELS_DIR):
        os.makedirs(OUT_LABELS_DIR)

    print("Generating darknet labels...")
    sample_img_pathes = []
    for dir_path, sub_dirs, files in os.walk(args.label_dir):
        for file_name in files:
            if file_name.endswith(".txt"):
                lbl_path = os.path.join(dir_path, file_name)                
                sample_id = getSampleId(lbl_path)
                img_path = os.path.join(args.image_2_dir, "{}.png".format(sample_id))
                sample_img_pathes.append(img_path)
                yolo_labels = parseSample(lbl_path, img_path, args.use_dont_care)
                with open(os.path.join(OUT_LABELS_DIR, "{}.txt".format(sample_id)), "w") as yolo_label_file:
                    for lbl in yolo_labels:
                        yolo_label_file.write("{} {} {} {} {}\n".format(*lbl))

    print("Writing training and test sample ids...")
    #create files that splits the images between training and testing

    file_range = np.arange(len(sample_img_pathes))
    range_train ,range_test = train_test_split(file_range,test_size=args.training_samples)
    np.save("kitti_train.txt",range_train)
    np.save("kitti_test.txt",range_test)


if __name__ == "__main__":
    main()
