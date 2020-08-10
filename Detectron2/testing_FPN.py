import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch


from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from matplotlib.pyplot import imshow
from PIL import Image
import os
import numpy as np
import json
from detectron2.structures import BoxMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader

from detectron2.config import get_cfg

from detectron2.data import DatasetCatalog, MetadataCatalog

######################################################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--model_dir', type=str, default=None)

parser.add_argument('--out_dir', type=str, default=None)

parser.add_argument('--num_of_data', type=int, default=42000)
parser.add_argument('--data_random', type=bool, default=False)

parser.add_argument('-f')
args = parser.parse_args()
######################################################################################

data_path = args.data_dir

model_path = args.model_dir

out_path = args.out_dir

num_of_data = args.num_of_data

data_random_select = args.data_random

def get_pedestrain_dict(image_list, json_list):
    dataset_dicts = []
    
    for i,path in tqdm(enumerate(image_list)):
        filename = path
        #img = cv2.imread(path)
        # height, width = cv2.imread(filename).shape[:2]
        record = {}
        record['file_name'] = filename
        #record['file_img'] = img
        record['image_id'] = i #path.split('/')[-1][:-5]
        #id is like 000000 or 000001
        record['height']= 480
        record['width']= 640
        
        #for i in data_list[1] to get bbox and category
        objs = []
        
        people = json_list[i]
        with open(people) as p:
            json_context = json.load(p)
            for person in json_context:
                boxes = list(map(float, person['pos']))
                obj = {
                    "bbox": boxes,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    #"segmentation": [poly], To draw a line, along to ballon
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts #list of dicts

def take_path(path):
    tmp = []
    for json_number in tqdm(sorted(os.listdir(path))):
        if json_number == '.ipynb_checkpoints':
            pass
        else: tmp.append(os.path.join(path, json_number))
    return tmp
    

data_pathlist_json = take_path(data_path+'street_json/')

if not data_random_select:
    data_pathlist_json = data_pathlist_json[:num_of_data]
else:
    data_pathlist_json = random.sample(data_pathlist_json, num_of_data)
    

test_people_jsonlist = data_pathlist_json
test_people_imagelist = [i.replace('street_json', 'street').replace('json', 'jpg') for i in data_pathlist_json]

DatasetCatalog.register("pedestrain_test", lambda : get_pedestrain_dict(test_people_imagelist, test_people_jsonlist))
MetadataCatalog.get("pedestrain_test").set(thing_classes=["person"])

cfg = get_cfg()
cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("pedestrain_test", )
cfg.OUTPUT_DIR = out_path

predictor = DefaultPredictor(cfg)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

evaluator = COCOEvaluator("pedestrain_test", cfg, False, output_dir = cfg.OUTPUT_DIR)

val_loader = build_detection_test_loader(cfg, "pedestrain_test")
inference_on_dataset(predictor.model, val_loader, evaluator)

