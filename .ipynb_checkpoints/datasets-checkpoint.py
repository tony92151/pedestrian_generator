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


class ImageDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, recursive_search=False, preload2mem = False):
        super(ImageDataset, self).__init__()
        #self.data_dir = data_dir
        self.transform = transform
        # self.imgpaths = self.__load_imgpaths_from_dir(self.data_dir, walk=recursive_search)
        self.street_imgpaths = glob.glob(data_dir+'/street_view/*')
        self.people_imgpaths = glob.glob(data_dir+'/people/*')
        self.position_jsonpaths = glob.glob(data_dir+'/json/*')

    def __len__(self):
        return len(self.street_imgpaths)

    def __getitem__(self, index, color_format='RGB'):
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

        mask.paste(people_img,(math.floor(img.size[0]/2-(box_w/2)),math.floor(img.size[1]/2-(box_h/2))))

        input_img = img.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom
        mask_with_poeple = mask.crop((cw-128, ch-128, cw+128, ch+128)) #left, top, right, bottom       

        img = img.convert(color_format)
        if self.transform is not None:
            img = self.transform(img)
        return input_img, mask_with_poeple#, content_img

    def load_img_from_index():
        pass

    def preload_images():
        pass

    # def __is_imgfile(self, filepath):
    #     filepath = os.path.expanduser(filepath)
    #     if os.path.isfile(filepath) and imghdr.what(filepath):
    #         return True
    #     else:
    #         return False

    # def __load_imgpaths_from_dir(self, dirpath, walk=False, allowed_formats=None):
    #     imgpaths = []
    #     dirpath = os.path.expanduser(dirpath)
    #     if walk:
    #         for (root, dirs, files) in os.walk(dirpath):
    #             for file in files:
    #                 file = os.path.join(root, file)
    #                 if self.__is_imgfile(file):
    #                     imgpaths.append(file)
    #     else:
    #         for path in os.listdir(dirpath):
    #             path = os.path.join(dirpath, path)
    #             if self.__is_imgfile(path) == False:
    #                 continue
    #             imgpaths.append(path)
    #     return imgpaths
