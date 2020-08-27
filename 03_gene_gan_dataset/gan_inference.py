import sys
sys.path.append('../libs')

from tqdm import tqdm
from models import CompletionNetwork, ContextDiscriminator, GlobalDiscriminator_P
from datasets_inference import ImageDataset
from losses import completion_network_loss, completion_network_loss_P
from noise import AddGaussianNoise
from utils import (
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
    poisson_blend,
    poisson_blend_m,
)
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
import os
import argparse
import numpy as np
import json
import time

import cv2
import matplotlib.pyplot as plt
import glob

from multiprocessing import Process, Queue, Pool
import time

# data_path = "/root/notebooks/final/caltech_origin_mask2_20000"
# out_path = "/root/notebooks/final/caltech_origin_mask_20000/result_6-10_part2"

data_path = "/root/notebooks/pedestrian_generator_data/caltech_origin_mask10_100000"

model_path = "/root/notebooks/pedestrian_generator_data/result_mask10/phase_3/1_model_cn_step20000"

out_path = "/root/notebooks/pedestrian_generator_data/caltech_origin_mask10_100000"

####################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--result_dir', type=str, default=None)
parser.add_argument('--recursive_search', action='store_true', default=False)
parser.add_argument('--init_model_cn', type=str, default=None)
parser.add_argument('--init_model_cd', type=str, default=None)
parser.add_argument('--steps_1', type=int, default=8000)
parser.add_argument('--steps_2', type=int, default=4000)
parser.add_argument('--steps_3', type=int, default=3000)
parser.add_argument('--snaperiod_1', type=int, default=800)
parser.add_argument('--snaperiod_2', type=int, default=400)
parser.add_argument('--snaperiod_3', type=int, default=300)
parser.add_argument('--max_holes', type=int, default=1)
parser.add_argument('--hole_min_w', type=int, default=48)
parser.add_argument('--hole_max_w', type=int, default=96)
parser.add_argument('--hole_min_h', type=int, default=48)
parser.add_argument('--hole_max_h', type=int, default=96)
parser.add_argument('--cn_input_size', type=int, default=256)
parser.add_argument('--ld_input_size', type=int, default=256)
parser.add_argument('--optimizer', type=str, choices=['adadelta', 'adam'], default='adam')
parser.add_argument('--bsize', type=int, default=10)
parser.add_argument('--bdivs', type=int, default=1)
parser.add_argument('--data_parallel', action='store_true')
parser.add_argument('--num_test_completions', type=int, default=3)
parser.add_argument('--mpv', nargs=3, type=float, default=None)
parser.add_argument('--alpha', type=float, default=4e-3)
parser.add_argument('--arc', type=str, choices=['celeba', 'places2'], default='celeba')

parser.add_argument('-f')

args = parser.parse_args()

data_path = args.data_dir

model_path = args.init_model_cn

out_path = args.result_dir

# ================================================
# Preparation
# ================================================
args.data_dir = os.path.expanduser(args.data_dir)
args.result_dir = os.path.expanduser(args.result_dir)
if args.init_model_cn != None:
    args.init_model_cn = os.path.expanduser(args.init_model_cn)


if torch.cuda.is_available() == False:
    raise Exception('At least one gpu must be available.')
else:
    gpu = torch.device('cuda:0')

# create result directory (if necessary)
if os.path.exists(args.result_dir) == False:
    os.makedirs(args.result_dir)

    
#code below not used    
for s in ['output']:
    if os.path.exists(os.path.join(args.result_dir, s)) == False:
        os.makedirs(os.path.join(args.result_dir, s))


####################################################################################


# inference dataset

trnsfm = transforms.Compose([
    transforms.ToTensor(),
    
])

trnsfm2 = transforms.Compose([
    transforms.ToTensor(),
])
print('loading dataset... (it may take a few minutes)')
train_dset = ImageDataset(args.data_dir, trnsfm,trnsfm2, load2meme = False)
train_loader = DataLoader(train_dset, batch_size=(args.bsize // args.bdivs), shuffle=False, num_workers=5)

alpha = torch.tensor(args.alpha).to(gpu)


# Create model G

model_cn = CompletionNetwork()
if args.data_parallel:
    model_cn = DataParallel(model_cn)
    
if args.init_model_cn != None:
    model_cn.load_state_dict(torch.load(args.init_model_cn, map_location='cpu'))

    
model_cn = model_cn.to(gpu)

transPIL = transforms.ToPILImage()

def inference_G():
    cnt_bdivs = 0
    #Pro.start()
    pbar = tqdm(total=len(train_loader))

    for street_img, mask_poeple, mask, img_names in train_loader:
        
        street_img = street_img.to(gpu)
        mask_poeple = mask_poeple.to(gpu)
        mask = mask

        input = torch.cat((street_img, mask_poeple), dim=1)
        with torch.no_grad():
            output = model_cn(input)
        mask_ = (mask).le(0.5).to(torch.uint8) * 255
        
        completed = poisson_blend_m(output, street_img, mask_)
        for i in range(len(completed)):
            img = completed[i]
            
            street_img_pil = transforms.functional.to_pil_image(img)
            
            imgpath = str(img_names[i])
            imgpath = imgpath.replace('street', 'output')
            #print(imgpath)
            #save_image(img, imgpath)
            street_img_pil.save(imgpath)
            #Pro.push([street_img_pil, imgpath])
        pbar.update()
            
        #pbar.set_description('%d | phase 1 | train loss: %.5f' % (n,loss.cpu()))
        
    pbar.close()
    #Pro.kill()

inference_G()