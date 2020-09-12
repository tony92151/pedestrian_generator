import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import os
import random
import glob
# from detectron_pro import detectron_mask_img,detectron_mask_img_composite
import shutil
import cv2

import random
from tqdm import tqdm

######################################################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None)

parser.add_argument('--pedestrian_dir', type=str, default=None)

parser.add_argument('--out_dir', type=str, default=None)

parser.add_argument('--num_of_data', type=int, default=42000)

parser.add_argument('-f')
args = parser.parse_args()
######################################################################################

data_path = args.data_dir
if data_path[-1]=='/': data_path = data_path[:-1]

origin_data_dir = args.data_dir
if origin_data_dir[-1]=='/': origin_data_dir = origin_data_dir[:-1]

people_dir = args.pedestrian_dir
if people_dir[-1]=='/': people_dir = people_dir[:-1]

# Image save dir
save_dir = args.out_dir
if save_dir[-1]=='/': save_dir = save_dir[:-1]

num_imgs = args.num_of_data

######################################################################################

# to get location that stickimg will sticked on jpg_dir center or random
def center_location(stickimg_dir):
    #im = np.array(Image.open(jpg_dir), dtype=np.uint8)
    x_center,y_center = im.shape[1]/2,im.shape[0]/2
    #im_stick = np.array(Image.open(stickimg_dir), dtype=np.uint8)
    #im_stick_shape = im_stick.shape
    im_stick_shape = (128, 64)
    bd_box_x,bd_box_y = x_center-(im_stick_shape[1]/2),y_center-(im_stick_shape[0]/2)
    bd_box_length,bd_box_height =im_stick_shape[1],im_stick_shape[0]
    
    result = [[bd_box_x,bd_box_y,bd_box_length,bd_box_height]]
    return result

def random_location(num = None):
    #im = np.array(Image.open(jpg_dir), dtype=np.uint8)
    # x boundary
    #rangeX = 400 
    #x_left_bound,x_right_bound = rangeX, im.shape[1]-rangeX
    result = []
    
    seq = [[320, [120, 240]], [280, [80, 160]], [240, [64, 128]], [220, [54, 108]], [200, [48, 96]], [190, [36, 72]], [180, [30, 60]], [175, [28, 56]], [173, [22, 44]], [175, [28, 56]], [173, [22, 44]], [175, [28, 56]], [173, [22, 44]], [175, [28, 56]], [173, [22, 44]], [175, [28, 56]], [173, [22, 44]], [175, [28, 56]], [173, [22, 44]]]
    
    if num == None:
        re = random.sample(seq, 1)
    else:
        re = random.sample(seq, len(num))
    
    x_left_bound,x_right_bound = 128,512
    
    for i in range(len(re)):
        x_center = random.randint(x_left_bound, x_right_bound)
        #result.append([x_center-int(re[i][1][0]/2), re[i][0]-int(re[i][1][1]/2), re[i][1][0], re[i][1][1], num[i]])
        result.append([x_center-int(re[i][1][0]/2), re[i][0]-int(re[i][1][1]/2), int((re[i][1][0])*4/5), re[i][1][1], num[i]]) # make people thinner
    
    return result

def create_json_file(jpg_dir,street_json,results_dir,function='center',num_of_people = 1):
    if function == 'center':
        re = center_location(stickimg_dir)
    elif function == 'random':
        re = random_location(num = num_of_people)
    
        
    input_file = open (street_json)
    json_array = json.load(input_file)
    
    data = []
    for box in re:
        data.append({
        'end':0,
        'hide':0,
        'id':0,
        'init':0,
        'lbl':"pasted_person",
        'lock':0,
        'occl':0,
        'pos':[
        box[0],
        box[1],
        box[2],
        box[3]],    
        'posv':[
        0,
        0,
        0,
        0],
        'str':0,
        'img':box[4]
        })
    
    if json_array != []:
        for item in json_array:
            data.append(item)
            
    with open(results_dir, 'w') as outfile:
        json.dump(data, outfile)
        

img_data = glob.glob(origin_data_dir+'/street/*.jpg', recursive=True)

json_data = glob.glob(origin_data_dir+'/street_json/*.json', recursive=True)


# Check dir folder exit
# If not, create one
if os.path.exists(save_dir) == False:
    os.makedirs(save_dir)

for s in ['people', 'mask', 'street', 'street_json','json']:
    if os.path.exists(os.path.join(save_dir, s)) == False:
        os.makedirs(os.path.join(save_dir, s))
        

street_imgs = img_data

if num_imgs<len(street_imgs):
    street_imgs = random.sample(street_imgs, num_imgs)
else:
    street_imgs.append(random.choices(street_imgs, num_imgs-len(street_imgs)))
    
    
random.shuffle(street_imgs)

mask_imgs_ = glob.glob(people_dir+'/market_mask_refine_6467/*.jpg', recursive=True)

people_imgs_ = [i.replace('market_mask_refine_6467', 'people') for i in mask_imgs_]


r_t_imgs = []
r_m_imgs = []


pbar = tqdm(total=len(people_imgs_))

for i in range(len(people_imgs_)):
    shutil.copyfile(mask_imgs_[i], save_dir+'/mask/'+str('{0:06}'.format(i))+'.jpg')
    shutil.copyfile(people_imgs_[i], save_dir+'/people/'+str('{0:06}'.format(i))+'.jpg')
    pbar.update()
    
pbar.close()



for i in tqdm(range(num_imgs)):
    
#     if (i%100==0):
#         print("Process (",i,"/",num_imgs,")  ","{:.2f}".format(100*i/num_imgs)," %")
        
    people_pick = random.randint(0,len(people_imgs_)-1)

    
    # save street img
    street_img = cv2.imread(street_imgs[i])
    street_img = cv2.resize(street_img,(640,480))
    cv2.imwrite(save_dir+'/street/'+str('{0:06}'.format(i))+'.jpg', street_img)
    
    ################################################################
    img_path = street_imgs[i]
    json_dir = img_path.replace('street', 'street_json')
    json_dir = json_dir.replace('jpg', 'json')
    shutil.copyfile(json_dir, save_dir+'/street_json/'+str('{0:06}'.format(i))+'.json')
    ################################################################
    
    # save poeple img
    people_img = people_imgs_[people_pick]

    
    if random.randint(0,100)>50:
        nump = [str('{0:06}'.format(random.randint(0,len(people_imgs_)-1)))+'.jpg' for i in range(2)]
    else:
        nump = [str('{0:06}'.format(random.randint(0,len(people_imgs_)-1)))+'.jpg' for i in range(3)]
    
    # create json file and save
    create_json_file(save_dir+'/street/'+str('{0:06}'.format(i))+'.jpg',
                     save_dir+'/street_json/'+str('{0:06}'.format(i))+'.json',
                     save_dir+'/json/'+str('{0:06}'.format(i))+'.json',
                     function="random",
                     num_of_people = nump)
