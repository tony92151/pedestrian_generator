import detectron2
from detectron2.utils.logger import setup_logger

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

from detectron2.config import get_cfg

from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.modeling import build_model

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import csv 

######################################################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--model_dir', type=str, default=None)

parser.add_argument('--batch', type=int, default=24)

parser.add_argument('--num_of_data', type=int, default=42000)
parser.add_argument('--data_random', type=bool, default=False)

parser.add_argument('-f')
args = parser.parse_args()
######################################################################################

data_path = args.data_dir

model_path = args.model_dir

num_of_data = args.num_of_data

data_random_select = args.data_random

batch = args.batch



def get_pedestrain_dict(image_list, json_list):
    dataset_dicts = []
    
    for i,path in tqdm(enumerate(image_list)):
        filename = path
        #img = cv2.imread(path)
        # height, width = cv2.imread(filename).shape[:2]
        record = {}
        record['file_name'] = filename
        #record['file_img'] = img
        record['image_id'] = i #path.split('/')[-1][:-5]
        #id is like 000000 or 000001
        record['height']= 480
        record['width']= 640
        
        #for i in data_list[1] to get bbox and category
        objs = []
        
        people = json_list[i]
        with open(people) as p:
            json_context = json.load(p)
            for person in json_context:
                boxes = list(map(float, person['pos']))
                obj = {
                    "bbox": boxes,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    #"segmentation": [poly], To draw a line, along to ballon
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts #list of dicts

def take_path(path):
    tmp = []
    for json_number in tqdm(sorted(os.listdir(path))):
        if json_number == '.ipynb_checkpoints':
            pass
        else: tmp.append(os.path.join(path, json_number))
    return tmp


data_pathlist_json = take_path(data_path+'street_json/')

data_pathlist_json = data_pathlist_json[42000:(42000+num_of_data)]

train_people_jsonlist = data_pathlist_json
random.shuffle(train_people_jsonlist)
train_people_imagelist = [i.replace('street_json', 'street').replace('/json/', '/output/').replace('.json', '.jpg') for i in train_people_jsonlist]

dataset_dicts = get_pedestrain_dict(train_people_imagelist, train_people_jsonlist)

print("Total images : "+str(len(train_people_imagelist)))

class inference_Dataset(Dataset):
    def __init__(self, train_d, dic):
        self.filenames = train_d
        self.dic = dic
        
        self.transform = transforms.Compose([transforms.ToTensor(), ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(img)
        return image, self.dic[idx]
    
data_test = inference_Dataset(train_people_imagelist, dataset_dicts)

# _,a,b = data_test.__getitem__(0)
# print(a)
# print(b)
    
infer_dataloader = DataLoader(inference_Dataset(train_people_imagelist, dataset_dicts),batch_size=batch,num_workers=6)

# test_people_jsonlist = data_pathlist_json
# test_people_imagelist = [i.replace('street_json', 'street').replace('json', 'jpg') for i in data_pathlist_json]

# DatasetCatalog.register("pedestrain_test", lambda : get_pedestrain_dict(test_people_imagelist, test_people_jsonlist))
# MetadataCatalog.get("pedestrain_test").set(thing_classes=["person"])

# cfg = get_cfg()
# cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# cfg.DATASETS.TEST = ("pedestrain_test", )
# cfg.OUTPUT_DIR = out_path

from detectron2.checkpoint import DetectionCheckpointer


#img = cv2.imread(image_path)
def getmodel(threadh=0.7):
    cfg = get_cfg()
    #cfg.merge_from_file("model_config.yaml")
    cfg.merge_from_file(model_zoo.get_config_file("Base-RCNN-FPN.yaml"))
    #cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final_fix.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threadh 
    #cfg.MODEL.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE='cuda'

    model = build_model(cfg)
    
    DetectionCheckpointer(model).load(os.path.join(model_path, "model_final.pth"))

    #model_dict = torch.load(cfg.MODEL.WEIGHTS, map_location=torch.device(cfg.MODEL.DEVICE))
    #model.load_state_dict(model_dict['model'] )
    model.to(cfg.MODEL.DEVICE)
    model.train(False)
    
    return model

def dataset_batch(dataset, bat):
    batch_data = []
    temp = []
    for i in dataset:
        if len(temp)<bat:
            temp.append(i)
        elif(i==dataset[-1]):
            batch_data.append(temp)
        else:
            batch_data.append(temp)
            temp = []
            temp.append(i)
    return batch_data


        
        
####################################################################################################
#https://github.com/facebookresearch/maskrcnn-benchmark/issues/430
def delete_net_weights_for_finetune(
	model_file, 
	out_file, 
	rpn_final_convs=False, 
	bbox_final_fcs=True, 
	mask_final_conv=True
):
	del_keys = []
	checkpoint = torch.load(model_file)
	print("keys: {}".format(checkpoint.keys()))
	m = checkpoint['model']

	if rpn_final_convs:
		# 'module.rpn.anchor_generator.cell_anchors.0', 
		# 'module.rpn.anchor_generator.cell_anchors.1', 
		# 'module.rpn.anchor_generator.cell_anchors.2', 
		# 'module.rpn.anchor_generator.cell_anchors.3', 
		# 'module.rpn.anchor_generator.cell_anchors.4'
		# 'module.rpn.head.cls_logits.weight', 
		# 'module.rpn.head.cls_logits.bias', 
		# 'module.rpn.head.bbox_pred.weight', 
		# 'module.rpn.head.bbox_pred.bias',
		del_keys.extend([
			k for k in m.keys() if k.find("rpn.anchor_generator") is not -1
		])
		del_keys.extend([
			k for k in m.keys() if k.find("rpn.head.cls_logits") is not -1
		])
		del_keys.extend([
			k for k in m.keys() if k.find("rpn.head.bbox_pred") is not -1
		])

	if bbox_final_fcs:
		# 'module.roi_heads.box.predictor.cls_score.weight', 
		# 'module.roi_heads.box.predictor.cls_score.bias', 
		# 'module.roi_heads.box.predictor.bbox_pred.weight', 
		# 'module.roi_heads.box.predictor.bbox_pred.bias',
		del_keys.extend([
			k for k in m.keys() if k.find(
				"roi_heads.box.predictor.cls_score"
			) is not -1
		])
		del_keys.extend([
			k for k in m.keys() if k.find(
				"roi_heads.box.predictor.bbox_pred"
			) is not -1
		])

	if mask_final_conv:
		# 'module.roi_heads.mask.predictor.mask_fcn_logits.weight', 
		# 'module.roi_heads.mask.predictor.mask_fcn_logits.bias',
		del_keys.extend([
			k for k in m.keys() if k.find(
				"roi_heads.mask.predictor.mask_fcn_logits"
			) is not -1
		])
	
	for k in del_keys:
		print("del k: {}".format(k))
		del m[k]

	# checkpoint['model'] = m
	#print("f: {}\nout_file: {}".format(f, out_file))
	#recursively_mkdirs(os.path.dirname(out_file))
	torch.save({"model": m}, out_file)
    
def fix_model_build():
    if  not os.path.isfile(os.path.join(model_path, "model_final_fix.pth")):
        print("Convert model ...")
        delete_net_weights_for_finetune(os.path.join(model_path, "model_final.pth"), os.path.join(model_path, "model_final_fix.pth"))
        print("Fixed model save at : " + str(os.path.join(model_path, "model_final_fix.pth")))
############################################################
def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0


#from：https://blog.csdn.net/leviopku/java/article/details/81629492

def compute_TF(outputs, d, iou):
    TF_list = []
    for box in outputs['instances'].pred_boxes.to("cpu"):
        
        for annotation in d['annotations']:
            #print(annotation)
            curr_iou = 0
            flag = False
            
            top, left = annotation['bbox'][0], annotation['bbox'][1]
            bottom, right = top+annotation['bbox'][2], left+annotation['bbox'][3] 
            
            #print("Here")
            #print(top, left, bottom, right)
            #print(box)
            curr_iou = compute_iou((top, left, bottom, right), box)
            
            if curr_iou >= iou:
                TF_list.append('T')
                flag = True
                break
        if flag == False:
            TF_list.append('F')
    return TF_list
############################################################

#dataset_dicts_batch = dataset_batch(dataset_dicts, batch)

#fix_model_build()

#predictor = getmodel(0.5)

#outputs = predictor(dataset_dicts_batch[0])


#print(dataset_dicts_batch[0])
#print(len(dataset_dicts_batch[0]))

iou = 0.5


matplot_x_test = []
matplot_y_test = []

MR_plot = []

for threshold in range(40,105,2):
    MR_list =[]
    FP_list = []
    score_threshold = threshold /100
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    #predictor = DefaultPredictor(cfg)
    predictor = getmodel(score_threshold)
    
    for x, dic in tqdm(infer_dataloader):
        imgs = Variable(x).cuda()
        d= dic

        #print(d)
        TP = 0
        #im = cv2.imread(d["file_name"])
#         im = d["file_img"]
        inputs = []  
        for i in imgs:
            inputs.append({"image":i})
        
        outputs = predictor(inputs)
        #print(outputs)
        for i in outputs:
#             print(i.['instances'])
            TF_list = compute_TF(i, d, iou)
            for TF in TF_list:
                if TF == 'T':
                    TP += 1
            if len(d['annotations']) > TP:
                MR = (len(d['annotations']) - TP) / len(d['annotations'])
                FP = len(i['instances']) - TP
            else: 
                MR = 0
                FP = len(i['instances']) - len(d['annotations'])

            MR_list.append(MR)
            FP_list.append(FP)
    matplot_x_test.append(sum(FP_list)/len(FP_list))
    matplot_y_test.append(sum(MR_list)/len(MR_list))
    
    MR_plot.append([sum(FP_list)/len(FP_list), sum(MR_list)/len(MR_list)])
    break
    #mean the MR_list and FP_list
    
with open(os.path.join(model_path, "model_result.csv") , 'w+', newline ='') as f:
    write = csv.writer(f) 
    write.writerows(MR_plot) 

print(matplot_x_test)
print(matplot_y_test)
print("OK")