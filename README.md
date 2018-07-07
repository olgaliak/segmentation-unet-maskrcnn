# segmentation-unet-maskrcnn
Code for  satellite image segmentation using Unet or Mask RCNN and comparing these two approaches.

Please, see more details in the blog post [Satellite Images Segmentation and Sustainable Farming](
https://www.microsoft.com/developerblog/2018/07/05/satellite-images-segmentation-sustainable-farming/)

## Get started
[LOL_prepare_data.ipynb](https://github.com/olgaliak/segmentation-unet-maskrcnn/blob/master/maskRCNN/LOL_prepare_data.ipynb): data augmentation using flip (mirror effect), rotate90, flip up to down, flip left to right.

## Data

### input data folder structure for maskRCNN
* train/valid/test
    * jpg (aerial images)
    * jpg4 (aerial images + hill shade data)
    * polygon (masks)

### input data folder structure for unet
* train/valid/test
    * jpg (aerial images)
    * hill (hill shade data)
    * polygon (masks)