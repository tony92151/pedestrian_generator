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

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from matplotlib.pyplot import imshow
from PIL import Image

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)



def detectron_mask_img(img_path,resize=None):
    psize = 400
    img = cv2.imread(img_path)
    if (resize!=None):
        img = cv2.resize(img, resize)

    backg = np.zeros([psize,psize,3]).astype(np.uint8)    
    backg[int((psize/2)-(img.shape[0]/2)):int((psize/2)+(img.shape[0]/2)),int((psize/2)-(img.shape[1]/2)):int((psize/2)+(img.shape[1]/2)),:] = img
    outputs = predictor(backg)
    mask = outputs["instances"].pred_masks.cpu().numpy()[0][int((psize/2)-(img.shape[0]/2)):int((psize/2)+(img.shape[0]/2)),int((psize/2)-(img.shape[1]/2)):int((psize/2)+(img.shape[1]/2))]
    
    return mask


def detectron_mask_img_composite(img_path,resize=None):
    img = cv2.imread(img_path)
    if (resize!=None):
        img = cv2.resize(img, resize)

    backg = np.zeros([256,256,3]).astype(np.uint8)    
    backg[int((256/2)-(img.shape[0]/2)):int((256/2)+(img.shape[0]/2)),int((256/2)-(img.shape[1]/2)):int((256/2)+(img.shape[1]/2)),:] = img
    outputs = predictor(backg)
    mask = outputs["instances"].pred_masks.cpu().numpy()[0]
    mask = mask[int((256/2)-(img.shape[0]/2)):int((256/2)+(img.shape[0]/2)),int((256/2)-(img.shape[1]/2)):int((256/2)+(img.shape[1]/2))]
    
    result = cv2.cvtColor(mask.astype(np.uint8) , cv2.COLOR_GRAY2RGB)*img
    
    return result

def detectron_multi_mask_img(img_path,resize=None):
    psize = 300
    img = cv2.imread(img_path)
    if (resize!=None):
        img = cv2.resize(img, resize)

    plain_mask = np.zeros([psize,psize]).astype(np.bool) 

    backg = np.zeros([psize,psize,3]).astype(np.uint8)    
    backg[int((psize/2)-(img.shape[0]/2)):int((psize/2)+(img.shape[0]/2)),int((psize/2)-(img.shape[1]/2)):int((psize/2)+(img.shape[1]/2)),:] = img
    outputs = predictor(backg)

    cla = outputs['instances'].pred_classes.cpu().numpy()

    masks = []
    for i in range(len(cla)):
        if (cla[i] == 0): # person
            masks.append(outputs["instances"].pred_masks.cpu().numpy()[i])

        # if (cla[i] == 1): # bicycle
        #     masks.append(outputs["instances"].pred_masks.cpu().numpy()[i])

        if (cla[i] == 27): # handbag
            masks.append(outputs["instances"].pred_masks.cpu().numpy()[i])

        if (cla[i] == 25): # backpack
            masks.append(outputs["instances"].pred_masks.cpu().numpy()[i])

    #mask = outputs["instances"].pred_masks.cpu().numpy()[0][int((psize/2)-(img.shape[0]/2)):int((psize/2)+(img.shape[0]/2)),int((psize/2)-(img.shape[1]/2)):int((psize/2)+(img.shape[1]/2))]

    for i in masks:
        plain_mask += i
    
    mask = plain_mask[int((psize/2)-(img.shape[0]/2)):int((psize/2)+(img.shape[0]/2)),int((psize/2)-(img.shape[1]/2)):int((psize/2)+(img.shape[1]/2))]
    return mask
    
    
        
    