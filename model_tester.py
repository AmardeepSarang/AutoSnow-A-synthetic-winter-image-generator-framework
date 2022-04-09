#Script to run trained model on set of images and saves its predictions
'''
args:
	model_path: path to model file
	val_img_folder: path to images folder
	results_folder: folder where results should be stored
'''
#
# myls.py
# Import the argparse library
import argparse
import torch
#from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import sys
from tqdm import tqdm

# winter data types
CLEAR_ICE = "ice_buildup_clear" #IC
ICE_BUILDUP_LIGHT = "ice_buildup_light" #IBL
ICE_BUILDUP_HEAVY = "ice_buildup_heavy" #IBH
WHITE_OUT_SNOW = "white_out_snow" #WOS
FINE_SNOW_HEAVY = "fine_snow_heavy" #FSH
FINE_SNOW_MED = "fine_snow_med" #FSM
FLUFFY_SNOW = "fluffy_snow" #FLS
STREAK_SNOW = "streaking_snow" #SS

ALL_CONDITIONS = [CLEAR_ICE,ICE_BUILDUP_HEAVY,ICE_BUILDUP_LIGHT, WHITE_OUT_SNOW, FINE_SNOW_HEAVY, FINE_SNOW_MED, 
                  FLUFFY_SNOW, STREAK_SNOW]


def getImgId(path):
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]



def load_model(model_path='clear_weather_model.pt'):
    #load model
    print("Loading model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom',path=model_path, force_reload=True) 
    print()
    print("Loaded model...")
    return model

def make_auto_snow_dir(condition_name, root_data_dir="Data",printit=True):
    mode = 0o666
    #make parent folder
    data_root_path = os.path.join(root_data_dir, "AutoSnow_"+condition_name)
    if not os.path.exists(data_root_path):
        os.mkdir(data_root_path, mode)
    
    if printit:
        print("Directory '{}' created".format(data_root_path))
    
    #make images subdirctory
    path_images = os.path.join(root_data_dir, "AutoSnow_"+condition_name,'images')
    if not os.path.exists(path_images):
        os.mkdir(path_images, mode)
    
    if printit:
        print("Directory '{}' created".format(path_images))
    
    #make lable directory
    path_labels = os.path.join(root_data_dir, "AutoSnow_"+condition_name,'prediction_labels')
    if not os.path.exists(path_labels):
        os.mkdir(path_labels, mode)
    if printit:
        print("Directory '{}' created".format(path_labels))
    
    return data_root_path, path_images, path_labels

def make_dir(path):
    mode = 0o666
    if not os.path.exists(path):
        os.mkdir(path, mode)
    
def predict(prediction_img_folder,predictions_results_folder, model):
    #iterate over all images in test folder
    print("Starting prediction")
    count=0
    for dir_path, sub_dirs, files in os.walk(prediction_img_folder):
            for file_name in tqdm(files, 'prediction labels'):
                if file_name.endswith(".png"):
                    
                    #make prediction 
                    img = os.path.join(prediction_img_folder,file_name)
                    results = model(img).xyxy

                    results = results[0].cpu().tolist()

                    #print detected labels to file for that image
                    img_id = getImgId(file_name)
                    path = os.path.join(predictions_results_folder, "{}.txt".format(img_id))
                    if os.path.exists(path):
                        os.remove(path)

                    with open(path, "w") as yolo_label_file:
                        for lbl in results:

                            # default format from .xyxy is: 
                            #<xmin> <ymin> <xmax> <ymax> <confidence> <class_id>

                            #But we need format:
                            #<class_id> <confidence> <xmin> <ymin> <xmax> <ymax> 
                            #this change is made in the .format  
                            yolo_label_file.write("{} {} {} {} {} {}\n".format(int(lbl[5]),lbl[4],int(lbl[0]),int(lbl[1]),int(lbl[2]),int(lbl[3])))





# Create the parser
parser = argparse.ArgumentParser(description='Enter the model file, test and results folder')

# Add the arguments
parser.add_argument('--mode',
                       type=str,
                       help='The mode to run in',default='d')
parser.add_argument('--model',
                       type=str,
                       help='the path to the model',default='clear_weather_model.pt')
parser.add_argument('--img_folder',
                       type=str,
                       help='the path to the image folder', default = 'Data/clear_weather/images/val')
parser.add_argument('--results_folder',
                       type=str,
                       help='the path to the results folder', default ='Data/clear_weather/labels/detection')


# parse arguments

args = parser.parse_args()

mode = args.mode

if mode == 'd':
    model = load_model(args.model)
    #run in default mode on clear weather dataset
    print("Running on clear weather data (default)")
    predict(args.img_folder,args.results_folder,model=model)
elif mode == "s":
    model = load_model(args.model)
    print("Running on all AutoSnow datasets...")

    #go over all weather condition
    for weather in ALL_CONDITIONS:
        #get directory
        root_weather_dir, img_dir, lable_dir = make_auto_snow_dir(weather,printit=False)
        
        print()
        print('Run {} data'.format(weather))
        predict(img_dir,lable_dir,model=model)

elif mode == "w":
    #run winter model on clear weather, cadcd winter data, and mixed Auto snow data
    

    print("Running winter model")
    model_name = "winter_weather_model.pt"
    model = load_model(model_name)
    
    #clear_weather
    img_dir = "Data/clear_weather/images/val"
    results_dir = "Data/clear_weather/labels/winter_detection"
    make_dir(results_dir)
    predict(img_dir,results_dir,model=model)

    #cadcd winter data
    img_dir = "Data/cadcd/images/val"
    results_dir = "Data/cadcd/labels/winter_detection"
    make_dir(results_dir)
    predict(img_dir,results_dir,model=model)


    #synthetic mix winter data
    img_dir = "Data/AutoSnow_mixed/images"
    results_dir = "Data/AutoSnow_mixed/winter_prediction"
    make_dir(results_dir)
    predict(img_dir,results_dir,model=model)

    #mixed winter and clear data
    img_dir = "Data/AutoSnow_training_data_mixed/images/val"
    results_dir = "Data/AutoSnow_training_data_mixed/labels/winter_detection"
    make_dir(results_dir)
    predict(img_dir,results_dir,model=model)
    