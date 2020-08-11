import detectron2
from detectron2.utils.logger import setup_logger

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

from detectron2.config import get_cfg

from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.modeling import build_model

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import csv 
from defaults import DefaultBatchPredictor
######################################################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--model_dir', type=str, default=None)

parser.add_argument('--batch', type=int, default=24)

parser.add_argument('--num_of_data', type=int, default=42000)
parser.add_argument('--data_random', type=bool, default=False)

parser.add_argument('-f')
args = parser.parse_args()

######################################################################################

data_path = args.data_dir

model_path = args.model_dir

num_of_data = args.num_of_data

data_random_select = args.data_random

batch = args.batch


def take_path(path):
    tmp = []
    for json_number in tqdm(sorted(os.listdir(path))):
        if json_number == '.ipynb_checkpoints':
            pass
        else: tmp.append(os.path.join(path, json_number))
    return tmp


data_pathlist_json = take_path(data_path+'street_json/')

data_pathlist_json = data_pathlist_json[42000:(42000+num_of_data)]

train_people_jsonlist = data_pathlist_json
random.shuffle(train_people_jsonlist)
train_people_imagelist = [i.replace('street_json', 'street').replace('/json/', '/output/').replace('.json', '.jpg') for i in train_people_jsonlist]

#dataset_dicts = get_pedestrain_dict(train_people_imagelist, train_people_jsonlist)

#print(dataset_dicts)
print("Total images : "+str(len(train_people_imagelist)))


cfg = get_cfg()

#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

pre = DefaultBatchPredictor(cfg, modelPath=os.path.join(model_path, "model_final.pth"))
                            
pre.create_batch_loader(train_people_imagelist, train_people_jsonlist, batch)

result, result_ = pre.batchPredictor()


# print(dataset_dicts[0])


    



