import utils
import os
import cv2
import numpy as np
from mrcnn_config import inputConfig
###
def generate_mask_path(mask_dir, filename):
    fn_img, ext = os.path.splitext(os.path.basename(filename))
    mask_endings = [x for x in inputConfig.CATEGORIES if x != fn_img.split('_')[0]]
    mask_path = [os.path.join(mask_dir, filename)]
    for ending in mask_endings:
        mask_path.append( os.path.join(mask_dir, fn_img + '_'+ ending + '.jpg'))
    return mask_path

class LolDataset(utils.Dataset):
    
    def load_LOL(self, datasetdir, hill=True):
        
        for i in range(inputConfig.NUM_CLASSES):
            self.add_class("shapes", i, inputConfig.CLASS_DICT[i+1] )
        if hill == True:
            image_dir = os.path.join(datasetdir, 'jpg4')
        else:
            image_dir = os.path.join(datasetdir, 'jpg')
        print ('image_dir is', image_dir)
        mask_dir = os.path.join(datasetdir, 'polygon')
        
        image_names = next(os.walk(image_dir))[2]
        for i in range(len(image_names)):
            self.add_image("shapes", image_id = i,
                    path=os.path.join(image_dir, image_names[i]),
                    mask_path = generate_mask_path(mask_dir, image_names[i]),
                    width = inputConfig.IMAGE_WIDTH,
                    height = inputConfig.IMAGE_HEIGHT)
        
    def load_image(self, image_id):
        info = self.image_info[image_id]
        image_path = info['path']
        image_BGR = cv2.imread(image_path)
        image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        valid_mask = []
        for _mask_path in mask_path:
            _mask = cv2.imread(_mask_path, 0)
            
            if _mask.max() == _mask.min():
                pass
            else:
                valid_mask.append(_mask_path)
             
        count = len(valid_mask)
        mask = np.zeros([info['height'], info['width'], count], 'uint8')
        shapes = []
        for i in range(count):
            img_array = cv2.imread(valid_mask[i], 0)
            (thresh, im_bw) = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            mask_array = (img_array < thresh).astype('uint8')
            mask[:, :, i:i+1] = np.expand_dims(mask_array, axis=2)
            fn_img, ext = os.path.splitext(valid_mask[i])

            if fn_img.split('_')[-1] == 'merged':
                shapes.append(fn_img.split('/')[-1].split('_')[0])
            else:
                shapes.append(fn_img.split('_')[-1])
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s) for s in shapes])
        
        return mask, class_ids