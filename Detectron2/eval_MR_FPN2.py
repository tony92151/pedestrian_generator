import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import cv2
import random
import pdb
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
parser.add_argument('--num_worker', type=int, default=4)

parser.add_argument('--data_random', type=bool, default=False)

parser.add_argument('-f')
args = parser.parse_args()

######################################################################################

data_path = args.data_dir

model_path = args.model_dir

num_of_data = args.num_of_data

data_random_select = args.data_random

batch = args.batch

worker = args.num_worker


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


iou = 0.5
def compute_TF(outputs, d, iou=0.5):
    TF_list = []
    for box in outputs['instances'].pred_boxes.to("cpu"):
        
        for annotation in d['annotations']:
            curr_iou = 0
            flag = False
            
            top, left = annotation['bbox'][0], annotation['bbox'][1]
            bottom, right = top+annotation['bbox'][2], left+annotation['bbox'][3] 
            
            curr_iou = compute_iou((top, left, bottom, right), box)
            
            if curr_iou >= iou:
                TF_list.append('T')
                flag = True
                break
        if flag == False:
            TF_list.append('F')
    return TF_list

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0


#from：https://blog.csdn.net/leviopku/java/article/details/81629492

def compute_XY(r, r_dic):
    MR_list =[]
    FP_list = []
    
    TP = 0
    for i in range(len(r)):
        outputs = r[i]
        d = r_dic[i]

        TP = 0

        TF_list = compute_TF(outputs, d, iou)
        for TF in TF_list:
            if TF == 'T':
                TP += 1
        if len(d['annotations']) > TP:
            MR = (len(d['annotations']) - TP) / len(d['annotations'])
            FP = len(outputs['instances']) - TP
        else: 
            MR = 0
            FP = len(outputs['instances']) - len(d['annotations'])

        MR_list.append(MR)
        FP_list.append(FP)
    x = sum(FP_list)/len(FP_list)
    y = sum(MR_list)/len(MR_list)
    print('X:',x)
    print('Y:',y)
    return x,y
    


def compuete_MR(threadh=0.6):
    cfg = get_cfg()

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threadh
    
    cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")
    
    cfg.merge_from_file(model_zoo.get_config_file("Base-RCNN-FPN.yaml"))
    cfg.DATALOADER.NUM_WORKERS = worker
    cfg.SOLVER.IMS_PER_BATCH = 3
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER =  50000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    pre = DefaultBatchPredictor(cfg)

    _= pre.create_batch_loader(train_people_imagelist, train_people_jsonlist, batch, worker)

    result, result_dic = pre.batchPredictor()
#     rr=[]
#     for i in range(100):
#         rer = pre(cv2.imread(train_people_imagelist[i]))
#         rr.append(rer)
    
    #print(result[:3])
    #print(result_dic[:3])
    
    x,y = compute_XY(result, result_dic)
    return x, y #FP, MR


## Start inference

MR_plot = []
count = 1

gap = 2
for threshold in range(40,105,gap):
    print("Threshold test : ",threshold/100.0," (" ,count,'/',len(range(40,100+gap,gap)), ")")
    x,y = compuete_MR(threshold/100.0)
    MR_plot.append([x,y])
    count+=1
    #break


with open(os.path.join(model_path, "model_result.csv") , 'w+', newline ='') as f:
    write = csv.writer(f) 
    write.writerows(MR_plot) 
    
print(MR_plot)
print("CSV file save at : ", os.path.join(model_path, "model_result.csv"))
print("OK")

    



