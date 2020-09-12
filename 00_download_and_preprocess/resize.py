from PIL import Image
import glob
import os
from tqdm import tqdm
import argparse

####################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--result_dir', type=str, default=None)

parser.add_argument('--w', type=int, default=None)
parser.add_argument('--h', type=int, default=None)

parser.add_argument('-f')

args = parser.parse_args()

data_path = args.data_dir

out_path = args.result_dir

img_w = args.w

img_h = args.h

####################################################################################

if os.path.exists(args.result_dir) == False:
    os.makedirs(args.result_dir)
    
    
img_dir_ = glob.glob(data_path+'/*.png', recursive=True)

#print(img_dir_[10:])


for i,_ in tqdm(enumerate(range(len(img_dir_)))):
    img = Image.open(img_dir_[i])
    img_ = img.resize((img_w, img_h))
    
    img_.save(os.path.join(out_path, img_dir_[i].replace(os.path.dirname(img_dir_[i]),'')[1:]))