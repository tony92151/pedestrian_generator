import torch.utils.data as data
import os
import torch
import tqdm
import imghdr
import random
from PIL import Image
import glob
import json
import numpy as np
import math

from tqdm import tqdm

class ImageDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, load2meme = False):
        super(ImageDataset, self).__init__()
        self.transform = transform
        
        self.street_imgpaths = glob.glob(data_dir+'/street_view/*')
        self.people_imgpaths = glob.glob(data_dir+'/people/*')
        self.position_jsonpaths = glob.glob(data_dir+'/json/*')
        
        self.load2meme = load2meme
        if (self.load2meme):
            #print("Load all images to mem")
            self.pre_tre_imgs, self.pre_peo_imgs, self.left_top= self.img_in_mem(self.street_imgpaths, self.people_imgpaths, self.position_jsonpaths)
            

    def __len__(self):
        return len(self.street_imgpaths)

    def __getitem__(self, index):
        if (self.load2meme):
            out1, out2, out3 = self.pre_tre_imgs[index], self.pre_peo_imgs[index], np.array(self.left_top[index])
        else:
            # Load street image
            img = Image.open(self.street_imgpaths[index])
            people_img = Image.open(self.people_imgpaths[index])
            img = img.convert('RGB')

            # Create plain mask image
            mask = Image.fromarray(np.zeros((img.size[1],img.size[0]),dtype="uint8"))
            mask = mask.convert('RGB')

            # Load data information from .json file
            with open(self.position_jsonpaths[index]) as json_file:
                data = json.load(json_file)

            box_x, box_y, box_w, box_h = int(data['people']['pos'][0]['0']), int(data['people']['pos'][0]['0']), 64, 128

            cw, ch = math.floor(img.size[0]/2), math.floor(img.size[1]/2)

            mask.paste(people_img,(math.floor(cw-(box_w/2)),math.floor(ch-(box_h/2))))
            
            lt = [96,64]

            input_img = img.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom
            mask_with_poeple = mask.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom       

            input_img = input_img.convert('RGB')
            if self.transform is not None:
                input_img = self.transform(input_img)
                mask_with_poeple = self.transform(mask_with_poeple)
            out1, out2, out3 = input_img, mask_with_poeple, np.array(lt)
            
        return out1, out2, out3


    def img_in_mem(self, str_img, people_img, position_json):
        str_imgs = []
        mask_people_imgs = []
        left_top = []
        pbar = tqdm(total=len(str_img))
        
        for index in range(len(str_img)):
            img = Image.open(str_img[index])
            peo_img = Image.open(people_img[index])
            img = img.convert('RGB')

            # Create plain mask image
            mask = Image.fromarray(np.zeros((img.size[1],img.size[0]),dtype="uint8"))
            mask = mask.convert('RGB')

            # Load data information from .json file
            with open(position_json[index]) as json_file:
                data = json.load(json_file)

            box_x, box_y, box_w, box_h = int(data['people']['pos'][0]['0']), int(data['people']['pos'][0]['0']), 64, 128

            cw, ch = math.floor(img.size[0]/2), math.floor(img.size[1]/2)

            mask.paste(peo_img,(math.floor(img.size[0]/2-(box_w/2)),math.floor(img.size[1]/2-(box_h/2))))

            input_img = img.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom
            mask_with_poeple = mask.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom    
            
            lt = [math.floor(img.size[0]/2-(box_w/2)),math.floor(img.size[1]/2-(box_h/2))]

            input_img = input_img.convert('RGB')
            mask_with_poeple = mask_with_poeple.convert('RGB')
            if self.transform is not None:
                input_img = self.transform(input_img)
                mask_with_poeple = self.transform(mask_with_poeple)
                
            str_imgs.append(input_img)
            mask_people_imgs.append(mask_with_poeple)
            left_top.append(lt)
            pbar.update()
        pbar.close()
        return str_imgs, mask_people_imgs, left_top