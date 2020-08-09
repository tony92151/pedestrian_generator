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
#from detectron_pro import mask_img,mask_img_composite
from tqdm import tqdm
import cv2

class ImageDataset(data.Dataset):
    def __init__(self, data_dir, transform=None,transform2=None):
        super(ImageDataset, self).__init__()
        self.transform = transform
        self.transform2 = transform2
        
        self.street_imgpaths = sorted(glob.glob(data_dir+'/street/*'), key=lambda x: x[-10:-3])

        
        self.people_imgpaths = [i.replace('street', 'people') for i in self.street_imgpaths]
        self.position_jsonpaths = [i.replace('street', 'json').replace('jpg', 'json') for i in self.street_imgpaths]
        #self.poeple_maskspaths = [i.replace('street', 'mask') for i in self.street_imgpaths]
        
        self.people_imgpaths_without_index = data_dir+'/people/'
        self.poeple_maskspaths_without_index = data_dir+'/mask/'

#         for i in self.street_imgpaths:
#             js = i.replace('street', 'json')
#             js = js.replace('jpg', 'json')
#             self.position_jsonpaths.append(js)
        

            

    def __len__(self):
        return len(self.street_imgpaths)

    def __getitem__(self, index):

        # Load street image
        street_img = Image.open(self.street_imgpaths[index])
        #people_img = Image.open(self.people_imgpaths[index])
        #masks_img = Image.open(self.poeple_maskspaths[index])
            
        #img = img.convert('RGB')

        # Create plain mask image
        plain_mask = Image.fromarray(np.zeros((street_img.size[1],street_img.size[0]),dtype="uint8"))
        plain_mask = plain_mask.convert('RGB')
            
        street_img2 = street_img.copy()
            

        # Load data information from .json file
        with open(self.position_jsonpaths[index]) as json_file:
            data = json.load(json_file)
            
        people_img = Image.open(self.people_imgpaths_without_index + data[0]['img'])
        masks_img = Image.open(self.poeple_maskspaths_without_index + data[0]['img'])

        #box_x, box_y, box_w, box_h = int(data[0]['pos'][0]), int(data[0]['pos'][1]), int(data[0]['pos'][2]),int(data[0]['pos'][3])

        box_x, box_y, box_w, box_h = math.floor(street_img.size[0]/2)-32, math.floor(street_img.size[1]/2)-64, 64, 128

        cw, ch = math.floor(street_img.size[0]/2), math.floor(street_img.size[1]/2)

        street_img2.paste(people_img,(box_x,box_y),masks_img)
        plain_mask.paste(people_img,(box_x,box_y),masks_img)
            
        lt = [box_x,box_y]

        #input_img = street_img.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom
        #mask_with_poeple = street_img2.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom    

        input_img = street_img
        mask_with_poeple = street_img2

        plain_mask.paste(masks_img,(box_x,box_y))
        #plain_mask = plain_mask.crop((cw-128, ch-128, cw+128, ch+128))

        #input_img = input_img.convert('RGB')
        if self.transform is not None:
            input_img = self.transform(input_img)
            mask_with_poeple = self.transform2(mask_with_poeple)
            plain_mask = self.transform2(plain_mask)
        out1, out2, out3, out4 = input_img, mask_with_poeple, plain_mask ,np.array(lt)
            
        return out1, out2, out3, out4