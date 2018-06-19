"""This script is to train the maskRCNN model
Given the input data (aerial images, masks and hill shade data), train an instance segmentation model to detect sustainable farming.
"""

import os
import numpy as np
import pickle
import argparse
import time

from mrcnn_config import modelConfig as MrcnnConfig
from mrcnn_config import inputConfig, modelConfig
from mrcnn_dataset import LolDataset 
import model as modellib

parser = argparse.ArgumentParser(description='mask RCNN')
# parser.add_argument('--modellog', 
#                     help='the directory of saved maskRCNN weights')
parser.add_argument('--train', 
                    help='the directory of the training jpg dataset')
parser.add_argument('--valid', 
                    help='the directory of the validation jpg dataset')
parser.add_argument('--hill', 
                    help='boolean parameter for loading hillshade data or not')
parser.add_argument('--epoch', type=int,
                    help='the number of epoches')
parser.add_argument('--output', 
                    help='the direcotry to store object detection results')

args = parser.parse_args()


# if not args.modellog:
# 	raise ImportError('The --modellog parameter needs to be provided (directory to model weights)')
# else:
# 	logs = args.modellog

if not args.train:
	raise ImportError('The --data parameter needs to be provided (directory to test dataset)')
else:
	train_dir = args.train

if not args.valid:
	raise ImportError('The --data parameter needs to be provided (directory to test dataset)')
else:
	val_dir = args.valid

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

os.environ["CUDA_VISIBLE_DEVICES"]="2" # select one GPU

# Directory to save logs and trained model
MODEL_DIR = output_dir

# Path to COCO trained weights
COCO_MODEL_PATH = "./mask_rcnn_coco.h5"

# maskRCNN config
mConfig = MrcnnConfig()

print ('read training data for maskRCNN ...')
dataset_train = LolDataset()
dataset_train.load_LOL(train_dir, hill=='True')
dataset_train.prepare()

print ('read validating data for maskRCNN ...')
dataset_val = LolDataset()
dataset_val.load_LOL(val_dir, hill=='True')
dataset_val.prepare()

# Create model in training mode# Create 
model = modellib.MaskRCNN(mode="training", config=modelConfig(),
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=0.01, 
            epochs=EPOCHES, 
            layers='all')
