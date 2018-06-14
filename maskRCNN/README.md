# MaskRCNN to detect sustainbale farming

## Prerequisites
- Python 3.4+
- [Imgaug](https://github.com/aleju/imgaug)
- Tensorflow 1.3+
- Keras 2.0.8+

## Get started
[LOL_prepare_data.ipynb](https://github.com/olgaliak/segmentation-unet-maskrcnn/blob/master/maskRCNN/LOL_prepare_data.ipynb): data augmentation using flip, rotate90, flip up down, flip left right.

[maskRCNN_training.ipynb](https://github.com/olgaliak/segmentation-unet-maskrcnn/blob/master/maskRCNN/maskRCNN_training.ipynb): train [maskRCNN model](https://github.com/matterport/Mask_RCNN) 

[main_eval.py](https://github.com/olgaliak/segmentation-unet-maskrcnn/blob/master/maskRCNN/main_eval.py): evaluate new image using the built maskRCNN model. 

eg: python main_eval.py --data ./tstData --hill True --output ./output --epoch [50]

## Prepare data

There are three datasets here: aerial imagery, human-labeled masks for target sustainable farming, hillshaded data. 

### tiles (aerial imagery) and corresponding masks
To get the training dataset the aerial imagery was labeled manually using  desktop ArcGIS tool. Then images got split to tiles of 224x224 pixel  size. 
The data for 7 suitable practices were prepared (see description below).  For training POC  we have focused on 4 classes that have the most of labeled data:
-  Grassed waterways (5.7K tiles)
-  Terraces (2.7K
- Water and Sediment Control Basins or WSBs (1K) 
- Field Borders (1K).

<!-- <img src="loldata.png" width="400"> -->
<img width="1465" alt="loldata" src="https://user-images.githubusercontent.com/30126508/41442961-e42f5bb6-6fee-11e8-8783-84e03fe910bd.png">

There are two image sets here: title jpg images and label masks as shown in the following figure. The left image is the tile and the middle and right images are the label masks.

<!-- <img src="sampledata.png" width="400"> -->
<img width="1028" alt="sampledata" src="https://user-images.githubusercontent.com/30126508/41442963-e706c072-6fee-11e8-9fb1-062e1d884188.png">

We augmented dataset (both tiles and corresponding masks) by flipping and rotating using [Imgaug](https://github.com/aleju/imgaug).

### hillshade data
[NRCS](https://www.nrcs.usda.gov/wps/portal/nrcs/site/national/home/) Experts heavily use DEM/hill shade data when analyzing suitability practices. For example, terrace is the combination of ridge and channel and contour buffer strips go around hill slope. Therefore, we added hill shade data to the dataset and applied the same data augmentation techniques to it as well.

## Preprocess data
As maskRCNN model only accept three channels images, we mreged hill shade data with areial images. We put both satellite and hill shade images in one blank image then used [gamma correction](https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/) to highlight hill shade information on the images. 

<!-- <img src="hillmerge.png" width="400"> -->
![hillmerge](https://user-images.githubusercontent.com/30126508/41442941-c617f7f0-6fee-11e8-9dac-a7ec915ca5a1.png)

## Train maskRCNN
In general, we need millions of images to train a deep learning model from scratch. As discussed in the above, we have unbalanced training dataset and the training dataset size is not enough for training a MaskRCNN model from scratch.

To leverage training time and the tiny training dataset, transfer learning is performed in the project. Transfer learning is a machine learning technique where a model trained from one task is repurposed on another related task. In our case, we used Feature Pyramid Network (FPN) and a ResNet101 backbone. The [initial weights](https://github.com/matterport/Mask_RCNN/releases) that have been trained on [COCO dataset](http://cocodataset.org/#home) were used. These pre-trained weights already learned common features in natural images. Then the model was fine-tuned using the custom tiny dataset. 

The detailed steps are described in [maskRCNN_training.ipynb](https://github.com/olgaliak/segmentation-unet-maskrcnn/blob/master/maskRCNN/maskRCNN_training.ipynb). To run this notebook, first clone [mask-RCNN repo](https://github.com/matterport/Mask_RCNN), then run the notebook by updating your own data path. 

## Evaluate model

Our primary metric for model evaluation was Jaccard Index and Dice Similarity Coefficient. They measure how close predicted mask is to the manually marked masks, ranging from 0 (no overlap) to 1 (complete congruence).

[Jaccard Similarity Index](https://en.wikipedia.org/wiki/Jaccard_index) is the most intuitive ratio between the intersection and union:

> ![](https://latex.codecogs.com/svg.latex?J%28A%2CB%29%20%3D%20%5Cfrac%7B%7CA%5Ccap%20B%7C%7D%7B%7CA%7C&plus;%7CB%7C-%7CA%5Ccap%20B%7C%7D)

[Dice Coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) is a popular metric, it's numerically less sensitive to mismatch when there is a reasonably strong overlap:

> ![](https://latex.codecogs.com/svg.latex?DSC%28A%2CB%29%20%3D%20%5Cfrac%7B2%7CA%5Ccap%20B%7C%7D%7B%7CA%7C&plus;%7CB%7C%7D)

Regarding loss functions we started out with using classical Binary Cross Entropy (BCE) that is available as prebuilt loss function in Keras.
Also we have explored incorporating Dice Similarity Coefficient  into loss function and got inspiration from [this repo](https://github.com/killthekitten/kaggle-carvana-2017) related to Kaggle's Carvana challenge. 

> ![](https://latex.codecogs.com/svg.latex?BCE_%7BDSC%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7DBCE&plus;%5Cfrac%7B1%7D%7B2%7D%7B%281-DSC%29%7D)

To get the jaccard and dice coefficient for each test image, run the script [main_eval.py](https://github.com/olgaliak/segmentation-unet-maskrcnn/blob/master/maskRCNN/main_eval.py). These values will be saved in pickle files. By turn the hill parrameter (True/False), you can choose model with/without hill shade data.


