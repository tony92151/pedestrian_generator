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
    def __init__(self, data_dir, transform=None,transform2=None, load2meme = False):
        super(ImageDataset, self).__init__()
        self.transform = transform
        self.transform2 = transform2
        
        self.street_imgpaths = sorted(glob.glob(data_dir+'/street/*'), key=lambda x: x[-10:-3])
        self.people_imgpaths = sorted(glob.glob(data_dir+'/people/*'), key=lambda x: x[-10:-3])
        self.position_jsonpaths = sorted(glob.glob(data_dir+'/json/*'), key=lambda x: x[-10:-3])
        self.poeple_masks = sorted(glob.glob(data_dir+'/mask/*'), key=lambda x: x[-10:-3])
        
        self.load2meme = load2meme
        if (self.load2meme):
            #print("Load all images to mem")
            self.pre_tre_imgs, self.pre_peo_imgs, self.left_top= self.img_in_mem(self.street_imgpaths, self.people_imgpaths, self.position_jsonpaths, self.poeple_masks)
            

    def __len__(self):
        return len(self.street_imgpaths)

    def __getitem__(self, index):
        if (self.load2meme):
            out1, out2, out3 = self.pre_tre_imgs[index], self.pre_peo_imgs[index], np.array(self.left_top[index])
        else:
            # Load street image
            img = Image.open(self.street_imgpaths[index])
            people_img = Image.open(self.people_imgpaths[index])
            masks_imgs = Image.open(self.poeple_masks[index])
            #people_img = Image.fromarray(mask_img_composite(self.people_imgpaths[index],(64,128))[:,:,[2,1,0]])
            #masks_imgs = Image.fromarray((cv2.cvtColor(np.asarray(masks_imgs).astype(np.uint8) , cv2.COLOR_GRAY2RGB)*np.asarray(people_img))[:,:,[2,1,0]])
            
            people_mask = Image.open(self.poeple_masks[index])
            
            #img = img.convert('RGB')

            # Create plain mask image
            plain_mask = Image.fromarray(np.zeros((img.size[1],img.size[0]),dtype="uint8"))
            plain_mask = plain_mask.convert('RGB')

            # Load data information from .json file
            with open(self.position_jsonpaths[index]) as json_file:
                data = json.load(json_file)

            box_x, box_y, box_w, box_h = int(data[0]['pos'][0]), int(data[0]['pos'][1]), int(data[0]['pos'][2]),int(data[0]['pos'][3])

            cw, ch = math.floor(img.size[0]/2), math.floor(img.size[1]/2)

            plain_mask.paste(people_img,(box_x,box_y),masks_imgs)
            
            lt = [box_x-cw+128,box_y-ch+128]

            input_img = img.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom
            mask_with_poeple = plain_mask.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom       

            #input_img = input_img.convert('RGB')
            if self.transform is not None:
                input_img = self.transform(input_img)
                mask_with_poeple = self.transform2(mask_with_poeple)
            out1, out2, out3 = input_img, mask_with_poeple, np.array(lt)
            
        return out1, out2, out3


    def img_in_mem(self, str_img, people_img, position_json, mask_poeple):
        str_imgs = []
        mask_people_imgs = []
        left_top = []
        pbar = tqdm(total=len(str_img))
        
        for index in range(len(str_img)):
#             img = Image.open(str_img[index])
#             #peo_img = Image.open(people_img[index])
#             peo_img = Image.fromarray(mask_img_composite(self.people_imgpaths[index],(64,128))[:,:,[2,1,0]])
#             img = img.convert('RGB')

#             # Create plain mask image
#             plain_mask = Image.fromarray(np.zeros((img.size[1],img.size[0]),dtype="uint8"))
#             plain_mask = mask.convert('RGB')

#             # Load data information from .json file
#             with open(position_json[index]) as json_file:
#                 data = json.load(json_file)

#             box_x, box_y, box_w, box_h = int(data[0]['pos'][0]), int(data[0]['pos'][1]), int(data[0]['pos'][2]),int(data[0]['pos'][3])

#             cw, ch = math.floor(img.size[0]/2), math.floor(img.size[1]/2)

#             palin_mask.paste(people_img,(box_x,box_y),masks_imgs)

#             input_img = img.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom
#             mask_with_poeple = mask.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom    
            
#             lt = [96,64]
            
#             input_img = input_img.convert('RGB')
#             mask_with_poeple = mask_with_poeple.convert('RGB')
            # Load street image
            img = Image.open(self.street_imgpaths[index])
            people_img = Image.open(self.people_imgpaths[index])
            masks_imgs = Image.open(self.poeple_masks[index])
            #people_img = Image.fromarray(mask_img_composite(self.people_imgpaths[index],(64,128))[:,:,[2,1,0]])
            #masks_imgs = Image.fromarray((cv2.cvtColor(np.asarray(masks_imgs).astype(np.uint8) , cv2.COLOR_GRAY2RGB)*np.asarray(people_img))[:,:,[2,1,0]])
            
            people_mask = Image.open(self.poeple_masks[index])
            
            #img = img.convert('RGB')

            # Create plain mask image
            plain_mask = Image.fromarray(np.zeros((img.size[1],img.size[0]),dtype="uint8"))
            plain_mask = plain_mask.convert('RGB')

            # Load data information from .json file
            with open(self.position_jsonpaths[index]) as json_file:
                data = json.load(json_file)

            box_x, box_y, box_w, box_h = int(data[0]['pos'][0]), int(data[0]['pos'][1]), int(data[0]['pos'][2]),int(data[0]['pos'][3])

            cw, ch = math.floor(img.size[0]/2), math.floor(img.size[1]/2)

            plain_mask.paste(people_img,(box_x,box_y),masks_imgs)
            
            lt = [box_x-cw+128,box_y-ch+128]

            input_img = img.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom
            mask_with_poeple = plain_mask.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom       

            #input_img = input_img.convert('RGB')
            
            if self.transform is not None:
                input_img = self.transform(input_img)
                mask_with_poeple = self.transform2(mask_with_poeple)
                
            str_imgs.append(input_img)
            mask_people_imgs.append(mask_with_poeple)
            left_top.append(lt)
            pbar.update()
        pbar.close()
        return str_imgs, mask_people_imgs, left_top