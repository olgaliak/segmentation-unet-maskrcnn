import itertools
import sys
import os
import numpy as np
import pickle

from unet_config import Config as UnetConfig
from dataset_lol import Dataset as UnetDataset

from dataset_lol import load_image_gt as unet_load_image_gt
from dataset_lol import data_generator as unet_data_generator
from script_LOL import get_unet
from script_LOL import get_patches as unet_get_patches
from script_LOL import add_hillshade_channel

from model_eval import predict_unet, unet_img_prep

#os.environ["CUDA_VISIBLE_DEVICES"]="1" # select one GPU

## data and output directories
val_dir = '<test data directory>' # test dataset
WEIGHTS_FLD = '<unet weights directory>' # unet model weights
OUTPUT_UNET = '<output direcotry>'
os.makedirs(OUTPUT_UNET, exist_ok = True)

## unet configuration
uConfig = UnetConfig()
dataset_val_unet = UnetDataset()
dataset_val_unet.load_LOL(val_dir)
dataset_val_unet.prepare()

## parameter settings: thresholds and model name list
TEST_IMAGES = next(os.walk(os.path.join(val_dir, 'jpg')))[2]
treshholds = [[0.3, 0.3, 0.3, 0.3],\
              [0.3, 0.1, 0.3, 0.3],\
              [0.3, 0.1, 0.4, 0.25],\
              [0.1, 0.3, 0.5, 0.3],\
              [0.1, 0.1, 0.1, 0.1]
              ] 
model_name_list = ['unet_cl4_step0_e50_tr10000_v500_jk0.9826', 'unet_cl4_step3_e50_tr10000_v500_jk0.9832']

## Helper function
def NametoID(dataset_val, image_name):
    for i in range(len(dataset_val.image_info)):
        if image_name == dataset_val.image_info[i]['path'].split('/')[-1]:
            image_id = dataset_val.image_info[i]['id']
    return image_id

# jaccard coefficient: https://en.wikipedia.org/wiki/Jaccard_index
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

def coeff_per_image_unet(metric_name, image_id, pred, gt_mask, gt_class_id):
    
    coeff_dict = {}
    for clsid in [1,2,3,4]:
        coeff_dict[clsid] = []
        gt_index = list(gt_class_id).index(clsid)
        gt_per_mask = gt_mask[:,:,gt_index].astype(int)
        pred_per_mask = pred[:,:,clsid-1].astype(int)
        
        if metric_name == 'jaccard index':
            coeff_dict[clsid].append(jaccard_coef(gt_per_mask, pred_per_mask))
        elif metric_name == 'dice':
            coeff_dict[clsid].append(dice_coef(gt_per_mask, pred_per_mask))
            
    return coeff_dict
    
def choose_trs(model, amt, x, i, treshholds, image_id, gt_mask, gt_class_id):
        ave_trs = []
        for trs in treshholds:  
            pred_res =  predict_unet(model, amt, trs, x,  ISZ=224, N_Cls=4)         
            r = pred_res[i,:, :, :]

            # calculate jaccard coefficient per image
            jaccard_coeff = coeff_per_image_unet('jaccard index', image_id, r, gt_mask, gt_class_id)
            dice_coeff = coeff_per_image_unet('dice', image_id, r, gt_mask, gt_class_id)
            
            ave_metric = np.mean(list(itertools.chain(*jaccard_coeff.values())))
            ave_trs.append(ave_metric)
        
        best_trs = treshholds[np.argmax(ave_trs)]
        return best_trs

## main function to choose the best model and best threshold
for model_name in model_name_list:
    model_path = os.path.join(WEIGHTS_FLD, model_name)
    print("Loading weights from ", model_path)
    model = get_unet()
    model.load_weights(model_path)

    jaccard_dic = {}
    dice_dic = {}
    trs_dic = {}

    for i in range(0, len(TEST_IMAGES), 10):
        batch_images_names = TEST_IMAGES[i: i+10]
        amt = len(batch_images_names)
        N_Channels = 4
        batch_images, batch_gt_class_ids, batch_gt_masks, batch_hills = unet_data_generator( dataset_val_unet, uConfig, \
                                                                        batch_size=amt, shuffle= False, \
                                                                        verbose=False, imageNames = batch_images_names,\
                                                                        add_hill=True, min_truth_sum = 0)


        x = unet_img_prep(batch_images, batch_hills, amt, N_Channels, uConfig)

        for i in range(amt):
            print("****************************")
            print("unet predictions for ", batch_images_names[i])
            image_id = NametoID(dataset_val_unet, batch_images_names[i])
            original_image = batch_images[i]
            # gt_class_id = batch_gt_class_ids[i]
            gt_mask =  batch_gt_masks[i]

            if batch_images_names[i].startswith('waterways'):
                gt_class_id = [1,2,3,4]
            elif batch_images_names[i].startswith('fieldborders'):
                gt_class_id = [2,1,3,4]
            elif batch_images_names[i].startswith('terraces'):
                gt_class_id = [3,1,2,4]
            else:
                gt_class_id = [4,1,2,3]
            # choose the best trs for the image
            best_trs = choose_trs(model, amt, x, i, treshholds, image_id, gt_mask, gt_class_id)
            # unet prediction using the best trs
            pred_bs =  predict_unet(model, amt, best_trs, x,  ISZ=224, N_Cls=4)         
            r_bs = pred_bs[i,:, :, :]

            # calculate jaccard coefficient per image
            jaccard_dic[batch_images_names[i]] = coeff_per_image_unet('jaccard index', image_id, r_bs, gt_mask, gt_class_id)
            dice_dic[batch_images_names[i]] = coeff_per_image_unet('dice', image_id, r_bs, gt_mask, gt_class_id)
            trs_dic[batch_images_names[i]] = best_trs

    # save results to pickle files
    jaccard_path = os.path.join(OUTPUT_UNET, model_name+'_jaccard.p')
    dice_path = os.path.join(OUTPUT_UNET, model_name+'_dice.p')
    trs_path = os.path.join(OUTPUT_UNET, model_name+'_trs.p')

    print ("save jaccard results into", jaccard_path)
    pickle.dump(jaccard_dic, open(jaccard_path, 'wb'))

    print ("save jaccard results into", dice_path)
    pickle.dump(dice_dic, open(dice_path, 'wb'))

    print ("save trs results into", trs_path)
    pickle.dump(trs_dic, open(trs_path, 'wb'))