import  numpy as np
from dataset_helper import Dataset
from dataset_helper import data_generator
from unet_config import Config

NUM_STABILITY_NORM = 1e-5

CACHED_DATASETS = {}

def get_dataste_key(dir, amt):
    return dir + str(amt)

def get_cached_dataset(folder, amt):
    global CACHED_DATASETS
    dataset = None
    k = get_dataste_key(folder, amt)
    if k in CACHED_DATASETS.keys():
        print("Using cached dataset ", k)
        dataset = CACHED_DATASETS[k]
    else:
        dataset = Dataset()
        dataset.load_LOL(folder)
        dataset.prepare()
        CACHED_DATASETS[k] = dataset
    return dataset

def prep_data_dor_unet(batch_size, batch_gt_masks, class_ids, config):
   print("number of masks to load:", len(batch_gt_masks))
   masks = np.zeros((batch_size,  config.NUM_CLASSES, config.ISZ, config.ISZ))
   for i in range(len(batch_gt_masks)): # loop though masks "sets" corresponding to every image:
       #print("GT class_ids:", class_ids[i])
       for j in range(config.NUM_CLASSES):
           mask_indx_class = class_ids[i, j]
           #print("Taking elem at position {0} from ground truth class array: {1}".format(j, mask_indx_class))
           mask_indx = mask_indx_class - 1 #  pos 0 -- ww, pos 1 -- field, 2-terr, 3 -wsb
           #print("mask_indx = mask_indx_class - 1  ==> result is ", mask_indx)
          # m = None -- will skip NOne assigment and se how it goes
          # print(mask_indx)
           if (mask_indx > -1):
               #print("taking mask at position {0} from batch_gt_masks".format(j))
               m = batch_gt_masks[i, :, :, j]
               masks[i,mask_indx] = m
               #print("Putting masks in the correct Y location:", mask_indx)

   return masks

def add_hillshade_channel(batch_hills, amt, images, config):
    hills = np.rollaxis(batch_hills, 3,
                        1)  # hills shape <class 'tuple'>: (600, 3, 256, 256) and images <class 'tuple'>: (600, 3, 256, 256)
    # x = np.stack((images, hills), axis =1)
    img_arr = np.empty([amt, config.N_CHANNELS, config.ISZ, config.ISZ])  # shape <class 'tuple'>: (600, 6, 256, 256)
    img_arr[:, 0:3, :, :] = images
    img_arr[:, 3:4, :, :] = hills
    return img_arr

def get_patches_dataset(dataset, config, shuffleOn = True, amt=10000, verbose = False, norm = True, imageNames = [], min_truth_sum = 0):
    batch_images, batch_gt_class_ids, batch_gt_masks, batch_hills = data_generator(dataset, config,
                                                                      batch_size=amt, shuffle=shuffleOn,
                                                                      verbose=verbose, imageNames = imageNames,
                                                                      add_hill=True,
																	  min_truth_sum = min_truth_sum)

    #print("shapes for: batch_images, batch_gt_class_ids, batch_gt_masks", batch_images.shape, batch_gt_class_ids.shape,
    #      batch_gt_masks.shape)
    masks = prep_data_dor_unet(amt, batch_gt_masks, batch_gt_class_ids, config)
    #print("Masks after re-shaping", masks.shape)

    images = np.rollaxis(batch_images, 3, 1)
    x = add_hillshade_channel(batch_hills, amt, images, config)


    y = masks
    if norm:
        #x = x[:, :]/255.
        for i in range(config.NUM_CLASSES):
            x[:,i] = (x[:, i] - config.MEAN_PIXEL_LOL[i]) / np.sqrt(config.VARIANCE_LOL[i] + NUM_STABILITY_NORM)

    #print("****x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y)", x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))

    # for pretrained u-net Channel should be last dim
    x = np.rollaxis(x, 1, 4)
    y = np.rollaxis(y, 1, 4)
    return x, y

def get_patches_dir(directory, config, shuffleOn = True, amt=10000, verbose = False, norm = True, imageNames = [], min_truth_sum = 0):
    dataset = get_cached_dataset(directory, amt)
    return get_patches_dataset(dataset, config, shuffleOn, amt, verbose, norm, imageNames, min_truth_sum)