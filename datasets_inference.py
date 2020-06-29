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

        
        self.people_imgpaths_without_index = data_dir+'/people/'
        
        self.poeple_maskspaths_without_index = data_dir+'/mask/'

        self.position_jsonpaths = []
        
        for i in self.street_imgpaths:
            js = i.replace('street', 'json')
            js = js.replace('jpg', 'json')
            self.position_jsonpaths.append(js)
        
        self.load2meme = load2meme
        if (self.load2meme):
            #print("Load all images to mem")
            self.pre_tre_imgs, self.pre_peo_imgs, self.pre_mask, self.pre_left_top= self.img_in_mem(
                self.street_imgpaths, 
                self.people_imgpaths, 
                # self.position_jsonpaths, 
                self.poeple_maskspaths )
            

    def __len__(self):
        return len(self.street_imgpaths)

    def __getitem__(self, index):
        if (self.load2meme):
            # out1: street img
            # out2: people with mask img
            # out3: mask img
            # out4: poeple position
            out1, out2, out3, out4 = self.pre_tre_imgs[index], self.pre_peo_imgs[index], self.pre_mask[index], np.array(self.pre_left_top[index])
        else:
            # Load street image
            street_img = Image.open(self.street_imgpaths[index])
            #people_img = Image.open(self.people_imgpaths[index])
            #masks_img = Image.open(self.poeple_maskspaths[index])
            
            #img = img.convert('RGB')

            # Create plain mask image
            plain_mask = Image.fromarray(np.zeros((street_img.size[1],street_img.size[0]),dtype="uint8"))
            plain_mask = plain_mask.convert('RGBA')
            
            plain_mask2 = plain_mask.copy()
            
            plain_people = Image.fromarray(np.zeros((street_img.size[1],street_img.size[0]),dtype="uint8"))
            plain_people = plain_people.convert('RGBA')
            
            street_img2 = street_img.copy()
            

            # Load data information from .json file
            with open(self.position_jsonpaths[index]) as json_file:
                data = json.load(json_file)
                
            out_plain_mask = Image.fromarray(np.ones((street_img.size[1],street_img.size[0]),dtype="uint8"))
            plain_mask = plain_mask.convert('RGBA')
                
            for i in range(len(data)):
                size = (int(data[i]['pos'][2]),int(data[i]['pos'][3]))
                imgp = Image.open(self.people_imgpaths_without_index+data[i]['img']).resize(size, Image.BILINEAR )
                imgm = Image.open(self.poeple_maskspaths_without_index+data[i]['img']).resize(size, Image.BILINEAR )
                
                #plain_people.paste(imgp,(int(data[i]['pos'][0]), int(data[i]['pos'][1])), imgm)
                cpm = plain_mask.copy().paste(imgm,(int(data[i]['pos'][0]), int(data[i]['pos'][1])))
                #cpp = plain_mask.copy().paste(imgp,(int(data[i]['pos'][0]), int(data[i]['pos'][1])), imgm)
                
                plain_mask.paste(imgm,(int(data[i]['pos'][0]), int(data[i]['pos'][1])), imgm)
                street_img2.paste(imgp,(int(data[i]['pos'][0]), int(data[i]['pos'][1])),imgm)
                
                
                #out_plain_mask = Image.composite(cpm, out_plain_mask, cpm)
                
                #Image.composite(plain_mask, plain_mask2, plain_mask.convert("L"))

            #plain_mask = out_plain_mask
            #box_x, box_y, box_w, box_h = int(data[0]['pos'][0]), int(data[0]['pos'][1]), int(data[0]['pos'][2]),int(data[0]['pos'][3])
            #box_x, box_y, box_w, box_h = int(data[0]['pos'][0]), int(data[0]['pos'][1]), int(data[0]['pos'][2]),int(data[0]['pos'][3])

            #h_pick = random.randint(0,4) # index
            #w_pick = random.randint(128,512) #  128 <= w <=512 (center of people image)

            #cw, ch = int(box_y+box_w/2), int(box_y+box_h/2)
            
            #box_x, box_y, box_w, box_h = int(cw - h_scale[h_pick][0]/2), int(ch - h_scale[h_pick][1]/2), h_scale[h_pick][0], h_scale[h_pick][1]

            #masks_img = masks_img.resize((box_w, box_h), Image.BILINEAR )
            #people_img = people_img.resize((box_w, box_h), Image.BILINEAR )
            #cw, ch = math.floor(street_img.size[0]/2), math.floor(street_img.size[1]/2)
            #cw, ch = w_pick, h_list[h_pick]
            #print(masks_img.size)

            #street_img2.paste(plain_people,(0,0),plain_mask) # paste from top-left point
            #plain_mask2.paste(plain_people,(0,0),plain_mask)
            
            #lt = [cw, ch]

            #input_img = street_img.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom
            #mask_with_poeple = street_img2.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom  

            input_img = street_img
            mask_with_poeple = street_img2

            #plain_mask.paste(masks_img,(box_x,box_y))
            #plain_mask = plain_mask.crop((cw-128, ch-128, cw+128, ch+128))

            #input_img = input_img.convert('RGB')
            if self.transform is not None:
                input_img = self.transform(input_img)
                mask_with_poeple = self.transform2(mask_with_poeple)
                plain_mask = self.transform2(plain_mask)
            out1, out2, out3, out4 = input_img, mask_with_poeple, plain_mask , self.street_imgpaths[index]
            
        return out1, out2, out3, out4


    def img_in_mem(self, str_img, people_img, position_json, mask_poeple):
        str_imgs = []
        mask_people_imgs = []
        pre_left_top = []
        pre_mask = []
        pbar = tqdm(total=len(str_img))
        
        for index in range(len(str_img)):
            # Load street image
            street_img = Image.open(self.street_imgpaths[index])
            people_img = Image.open(self.people_imgpaths[index])
            masks_img = Image.open(self.poeple_maskspaths[index])
            
            #img = img.convert('RGB')

            # Create plain mask image
            plain_mask = Image.fromarray(np.zeros((street_img.size[1],street_img.size[0]),dtype="uint8"))
            plain_mask = plain_mask.convert('RGB')

            street_img2 = street_img.copy()

            # Load data information from .json file
            with open(self.position_jsonpaths[index]) as json_file:
                data = json.load(json_file)

            box_x, box_y, box_w, box_h = int(data[0]['pos'][0]), int(data[0]['pos'][1]), int(data[0]['pos'][2]),int(data[0]['pos'][3])

            cw, ch = math.floor(street_img.size[0]/2), math.floor(street_img.size[1]/2)

            street_img2.paste(people_img,(box_x,box_y),masks_img)
            plain_mask.paste(people_img,(box_x,box_y),masks_img)
            
            lt = [box_x-cw+128,box_y-ch+128]

            input_img = street_img.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom
            mask_with_poeple = street_img2.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom      

            plain_mask.paste(masks_img,(box_x,box_y))
            plain_mask = plain_mask.crop((cw-128, ch-128, cw+128, ch+128))        
            
            if self.transform is not None:
                input_img = self.transform(input_img)
                mask_with_poeple = self.transform2(mask_with_poeple)
                plain_mask = self.transform2(plain_mask)
                
            str_imgs.append(input_img)
            mask_people_imgs.append(mask_with_poeple)
            pre_left_top.append(lt)
            pre_mask.append(plain_mask)
            pbar.update()
        pbar.close()
        return str_imgs, mask_people_imgs, pre_mask, pre_left_top