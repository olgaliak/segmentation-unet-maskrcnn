"""This script is to evaluate model performance over N of epoches
Given different N values, save the jaccard and dice coefficient dictionary in the output directory
"""

import os
import numpy as np
import pickle
import argparse
import time

# import maskRCNN utils
from mrcnn_config import modelConfig as MrcnnConfig
from mrcnn_config import inputConfig, inferenceConfig
from mrcnn_dataset import LolDataset 
import model as modellib

parser = argparse.ArgumentParser(description='mask RCNN')
parser.add_argument('--modellog', 
                    help='the directory of saved maskRCNN weights')
parser.add_argument('--data', 
                    help='the directory of the validation jpg dataset')
parser.add_argument('--hill', 
                    help='boolean parameter for loading hillshade data or not')
parser.add_argument('--output', 
                    help='the direcotry to store object detection results')
parser.add_argument('--epoch', nargs='+', type=int,
                    help='the number of epoches')
args = parser.parse_args()


if not args.modellog:
	raise ImportError('The --modellog parameter needs to be provided (directory to model weights)')
else:
	logs = args.modellog

if not args.data:
	raise ImportError('The --data parameter needs to be provided (directory to test dataset)')
else:
	val_dir = args.data

if not args.hill:
	raise ImportError('The --hill parameter needs to be provided (dataset to use)')
else:
	hill = args.hill

if not args.output:
    print ('the output will be saved in the folder %s under current directory'% inputConfig.OUTPUT_DIR)
    output_dir = inputConfig.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
else:
	output_dir = args.output

if not args.epoch:
	raise ImportError('The --epoch parameter needs to be provided')
else:
    EPOCHES = args.epoch

#os.environ["CUDA_VISIBLE_DEVICES"]="2" # select one GPU

model_dir = inputConfig.MODEL_DIR

# maskRCNN config
mConfig = MrcnnConfig()

print ('read validation data for maskRCNN ...')

print ('debug', hill == 'True')
dataset_val = LolDataset()
dataset_val.load_LOL(val_dir, hill=='True')
dataset_val.prepare()

def NametoID(image_name):
    for i in range(len(dataset_val.image_info)):
        if image_name == dataset_val.image_info[i]['path'].split('/')[-1]:
            image_id = dataset_val.image_info[i]['id']
    return image_id

## jaccard coefficient: https://en.wikipedia.org/wiki/Jaccard_index
def jaccard_coef(y_true, y_pred):
    intersec = y_true*y_pred
    union = np.logical_or(y_true, y_pred).astype(int)
    if intersec.sum() == 0:
        jac_coef = 0
    else:
        jac_coef = round(intersec.sum()/union.sum(), 2)
    return jac_coef

## dice coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
def dice_coef(y_true, y_pred):
    intersec = y_true*y_pred
    union = y_true+y_pred
    if intersec.sum() == 0:
        dice_coef = 0
    else:
        dice_coef = round(intersec.sum()*2/union.sum(), 2)
    return dice_coef

def coeff_per_image(metric_name, image_id, pred, gt_mask, gt_class_id):
    
    coeff_dict = {}
    
    for clsid in list(inputConfig.CLASS_DICT.keys()):         
        coeff_dict[clsid] = []
        gt_index = np.where(gt_class_id == clsid)
        
        # if there is no groundtruth or no predicted mask, the coefficient is equal to zero
        if gt_index[0].size ==0 or len(pred['masks']) == 0:
            coeff_dict[clsid].append(0)
        else:
            # get the union of all groundtruth masks belong to clsid
            gt_mask_per_class = gt_mask[:,:,gt_index[0]] # get groundtruth mask

            _gt_sum = np.zeros((gt_mask.shape[0],gt_mask.shape[1]))

            for gt_num in range(gt_mask_per_class.shape[2]): # as there may be over one mask per class
                _gt =  gt_mask_per_class[:,:,gt_num]
                _gt_sum = _gt_sum + _gt

            _gt_union = (_gt_sum>0).astype(int)

            # get the union of all predicted masks belong to clsid
            pred_index = np.where(pred['class_ids'] == clsid)
            pred_mask_per_class = pred['masks'][:,:,pred_index[0]]

            _mask_sum = np.zeros((pred['masks'].shape[0],pred['masks'].shape[1]))

            for num in range(pred_mask_per_class.shape[2]):
                _mask = pred_mask_per_class[:,:,num]
                _mask_sum = _mask_sum + _mask

            _mask_union = (_mask_sum>0).astype(int)
            
            if metric_name == 'jaccard index':
                coeff_dict[clsid].append(jaccard_coef(_mask_union, _gt_union))
            elif metric_name == 'dice':
                coeff_dict[clsid].append(dice_coef(_mask_union, _gt_union))
            
    return coeff_dict

# get inference config
inference_config = inferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=model_dir)

# calculate jaccard coefficient for all test images over each epoch
start_time = time.time()

for epoch in EPOCHES:
    # load the model
    print ('epoch number is {}'.format(epoch))
    model_path = os.path.join(model_dir, logs + '/mask_rcnn_coco_'+"{0:0=4d}".format(epoch)+'.h5')
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    
    jaccard_dic = {}
    dice_dic = {}
    for image_id in dataset_val.image_ids:
        
        print ('processing image_id {}'.format(image_id))
        # load groundtruth and prediction per image
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, 
                               image_id, use_mini_mask=False)
        results = model.detect([original_image], verbose=1)
        r = results[0]
        
        jaccard_dic[image_id] = coeff_per_image('jaccard index', image_id, r, gt_mask, gt_class_id)
        dice_dic[image_id] = coeff_per_image('dice', image_id, r, gt_mask, gt_class_id)

    jaccard_path = os.path.join(output_dir, logs+'_jaccard_epoch_'+str(epoch)+'.p')
    dice_path = os.path.join(output_dir, logs+'_dice_epoch_'+str(epoch)+'.p')
    
    print ("save jaccard results into", jaccard_path)
    pickle.dump(jaccard_dic, open(jaccard_path, 'wb'))
    
    print ("save jaccard results into", dice_path)
    pickle.dump(dice_dic, open(dice_path, 'wb'))

print("--- %s seconds ---" % (time.time() - start_time))