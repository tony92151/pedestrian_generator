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

######################################################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--out_dir', type=str, default=None)

parser.add_argument('--num_of_data', type=int, default=42000)
parser.add_argument('--data_random', type=bool, default=False)

parser.add_argument('-f')
args = parser.parse_args()
######################################################################################


data_path = args.data_dir

out_path = args.out_dir

num_of_data = args.num_of_data

data_random_select = args.data_random


# data_path = '/root/notebooks/final/caltech_origin_data_refine/'

# out_path = '/root/notebooks/final/detectron2_output/baseline_FPN_final_v2_part2/'

# num_of_data = 42000

# data_random_select = False

#test = os.listdir('/root/notebooks/final/caltech_origin_data_refine/street_json/')

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
    
# print(train_data_pathlist_json[:10])

# print(len(train_data_pathlist_json))
# print(train_data_pathlist_json[78978][:-5])
# with open(train_data_pathlist_json[0], 'r') as t :
#     print(json.load(t))
    
# people_jsonlist = []
# people_imagelist = []
# with open('./people_jsonlist_final_v2', 'r') as j:
#     lines = j.readlines()
#     for line in lines:
#         people_jsonlist.append('/home/tedbest2/Desktop/repo/'+line[29:-1])
# with open('./people_imagelist_final_v2', 'r') as i:
#     lines = i.readlines()
#     for line in lines:
#         people_imagelist.append('/home/tedbest2/Desktop/repo/'+line[29:-1])

# print(len(people_jsonlist), len(people_imagelist))
# print(people_imagelist[:2])
# print(people_jsonlist[:2])


train_people_jsonlist = data_pathlist_json
train_people_imagelist = [i.replace('street_json', 'street').replace('json', 'jpg') for i in data_pathlist_json]

test_people_jsonlist = data_pathlist_json
test_people_imagelist = [i.replace('street_json', 'street').replace('json', 'jpg') for i in data_pathlist_json]

# print(len(train_people_imagelist))
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

from detectron2.data import DatasetCatalog, MetadataCatalog

DatasetCatalog.register("pedestrain_train", lambda : get_pedestrain_dict(train_people_imagelist, train_people_jsonlist))
MetadataCatalog.get("pedestrain_train").set(thing_classes=["person"])
DatasetCatalog.register("pedestrain_test", lambda : get_pedestrain_dict(test_people_imagelist, test_people_jsonlist))
MetadataCatalog.get("pedestrain_test").set(thing_classes=["person"])

pedestrain_metadata = MetadataCatalog.get("pedestrain_train")

# Traning 

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Base-RCNN-FPN.yaml"))
cfg.DATASETS.TRAIN = ("pedestrain_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 3
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER =  50000   # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = out_path


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()