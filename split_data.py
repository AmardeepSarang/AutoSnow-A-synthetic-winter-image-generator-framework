
'''
==================================================================================================================================================================
AUTHOR: Amardeep Sarang
DESCRIPTION: Used move KITTI dataset images and labels into training and testing sets. Splits data based on numpy lists created by kitti_label.py. Also creates the
				Testing and training folders are created before files are moved, folders made according to YOLO requirement in following structure:
				Data/
				└── clear_weather/
					├── images/
					│   ├── train/
					│   │   ├── img01.jpg
					│   │   ├── img03.jpg
					│   │   └── ...
					│   └── val/
					│       ├── img02.jpg
					│       ├── img04.jpg
					│       └── ..
					└── labels/
						├── train/
						│   ├── lbl01.txt
						│   └── lbl03.txt
						└── val/
							├── lbl02.txt
							└── lbl04.txt


ARGS (Hard coded):
	Path to numpy list of training files (hard coded to 'kitti_train.txt.npy')
	Path to numpy list of testing files (hard coded to 'kitti_test.txt.npy')

==================================================================================================================================================================
'''

import os
import csv
import random
import numpy as np

WEATHER='clear_weather'#top level folder


def create_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)

#backwards on purpose to fix mistake 
val_list=np.load('kitti_train.txt.npy')

train_list=np.load('kitti_test.txt.npy')

print("Train size: ",np.shape(train_list)[0])
print("Val size: ",np.shape(val_list)[0])
#make train and test data folder
train_path_img=os.path.join("Data",WEATHER,"images","train")
val_path_img=os.path.join("Data",WEATHER,"images","val")

train_path_lb=os.path.join("Data",WEATHER,"labels","train")
val_path_lb=os.path.join("Data",WEATHER,"labels","val")

for path in [train_path_lb,train_path_img,val_path_lb,val_path_img]:
	create_folder(path)

print("Moving train data...")
#move training dataset images and labes to its own folder
for data_i in train_list:

	old_file_lb = os.path.join("Data","labels", "{:06d}.txt".format(data_i))
	old_file_img = os.path.join("Data","images","image_2", "{:06d}.png".format(data_i))

	new_file_lb = os.path.join(train_path_lb, "{:06d}.txt".format(data_i))
	new_file_img = os.path.join(train_path_img, "{:06d}.png".format(data_i))

	os.replace(old_file_img,new_file_img)
	os.replace(old_file_lb,new_file_lb)

print("Moving val data...")
#move val dataset images and labes to its own folder
for data_i in val_list:

	old_file_lb = os.path.join("Data","labels", "{:06d}.txt".format(data_i))
	old_file_img = os.path.join("Data","images","image_2", "{:06d}.png".format(data_i))

	new_file_lb = os.path.join(val_path_lb, "{:06d}.txt".format(data_i))
	new_file_img = os.path.join(val_path_img, "{:06d}.png".format(data_i))

	os.replace(old_file_img,new_file_img)
	os.replace(old_file_lb,new_file_lb)