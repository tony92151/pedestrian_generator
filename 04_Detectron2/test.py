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
import sys

######################################################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--out_dir', type=str, default=None)
parser.add_argument('--data_p', type=float, default=1.0)

parser.add_argument('--second_data_dir', type=str, default=None)

parser.add_argument('--num_of_data', type=int, default=42000)
parser.add_argument('--data_random', type=bool, default=False)

parser.add_argument('-f')
args = parser.parse_args()
######################################################################################


data_path = args.data_dir

out_path = args.out_dir

num_of_data = args.num_of_data

data_random_select = args.data_random

print(args.data_p)
print(type(args.data_p))

