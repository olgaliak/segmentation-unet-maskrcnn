import os
import numpy as np

import scipy.misc

import cv2
import scipy.ndimage

from sys import platform

SLASH = '/'
if platform == "win32":
    SLASH = '\\'

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    def get_imageids(self, imagenames):
        result = []
        for i in range(len(self.image_info)):
            path = self.image_info[i]["path"]
            for imagename in imagenames:
                if imagename in path:
                    result.append(self.image_info[i]["id"])
                    if len(imagenames) == len(result):
                        print("Names: {0}, ids: {1}".format(imagenames, result))
                        return  result
                    break
        if len(result) == 0:
            print("Did not find any matching ids to: ", imagenames)
        return result

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]


# custom
    def generate_mask_path(self, mask_dir, filename):
        fn_img, ext = os.path.splitext(os.path.basename(filename))

        if fn_img.split('_')[0] == 'waterways':
            mask_path = [os.path.join(mask_dir, filename),
                         os.path.join(mask_dir, fn_img + '_fieldborders' + '.jpg'),
                         os.path.join(mask_dir, fn_img + '_terraces' + '.jpg'),
                         os.path.join(mask_dir, fn_img + '_wsb' + '.jpg'),
                         ]

        elif fn_img.split('_')[0] == 'fieldborders':
            mask_path = [os.path.join(mask_dir, filename),
                         os.path.join(mask_dir, fn_img + '_waterways' + '.jpg'),
                         os.path.join(mask_dir, fn_img + '_terraces' + '.jpg'),
                         os.path.join(mask_dir, fn_img + '_wsb' + '.jpg'),
                         ]
        elif fn_img.split('_')[0] == 'terraces':
            mask_path = [os.path.join(mask_dir, filename),
                         os.path.join(mask_dir, fn_img + '_fieldborders' + '.jpg'),
                         os.path.join(mask_dir, fn_img + '_waterways' + '.jpg'),
                         os.path.join(mask_dir, fn_img + '_wsb' + '.jpg'),
                         ]

        else:
            mask_path = [os.path.join(mask_dir, filename),
                         os.path.join(mask_dir, fn_img + '_waterways' + '.jpg'),
                         os.path.join(mask_dir, fn_img + '_fieldborders' + '.jpg'),
                         os.path.join(mask_dir, fn_img + '_terraces' + '.jpg'),
                         ]
        return mask_path

    def load_LOL(self, datasetdir):
        #         self.add_class("shapes", 1, 'terraces')
        self.add_class("shapes", 1, 'waterways')
        self.add_class("shapes", 2, 'fieldborders')
        self.add_class("shapes", 3, 'terraces')
        self.add_class("shapes", 4, 'wsb')
        #         self.add_class("shapes", 3, 'fieldborders')
        #         self.add_class("shapes", 4, 'filterstrips')
        #         self.add_class("shapes", 5, 'riparian')
        #         self.add_class("shapes", 6, 'contourbufferstrips')


        image_dir = os.path.join(datasetdir, 'jpg')
        mask_dir = os.path.join(datasetdir, 'polygon')
        hill_dir = os.path.join(datasetdir, 'hill')

        image_names = next(os.walk(image_dir))[2]
        for i in range(len(image_names)):
            msk_path = self.generate_mask_path(mask_dir, image_names[i])
            hill_path = os.path.join(hill_dir, image_names[i])
            self.add_image("shapes", image_id=i,
                           path=os.path.join(image_dir, image_names[i]),
                           mask_path=msk_path,
                           hill_path=hill_path,
                           width=224,
                           height=224)

    def load_image(self, image_id, verbose = False, isHill=False):
        info = self.image_info[image_id]
        image_path = info['path']
        if isHill:
            image_path = info['hill_path']
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            #going from {224, 224) shape to {224, 224, 1)
            image_ch = np.zeros((image.shape[0], image.shape[1], 1))
            image_ch[:, :, 0] = image
            image = image_ch
        else:
            image = cv2.imread(image_path)
        if verbose:
            print("***img path:", image_path)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        #print("*** mask path: ", mask_path)
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
            mask[:, :, i:i + 1] = np.expand_dims(mask_array, axis=2)
            fn_img, ext = os.path.splitext(valid_mask[i])

            if fn_img.split('_')[-1] == 'merged':
                shapes.append(fn_img.split(SLASH)[-1].split('_')[0])
            else:
                shapes.append(fn_img.split('_')[-1])
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s) for s in shapes])

        return mask, class_ids

def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding

def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def load_image_gt(dataset, image_id, config, verbose = False, add_hill = False):
    # Load image and mask
    image = dataset.load_image(image_id, verbose)
    mask, class_ids = dataset.load_mask(image_id)
    shape = image.shape
    image, window, scale, padding = resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    mask = resize_mask(mask, scale, padding)
    hill = None
    if add_hill:
        hill = dataset.load_image(image_id, verbose, True)
        if not hill is None:
            hill, _,_,_ = resize_image(
                hill,
                min_dim=config.IMAGE_MIN_DIM,
                max_dim=config.IMAGE_MAX_DIM)
    return image,  class_ids, mask, hill


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    # Image mean (RGB)
    mean_pix = config.MEAN_PIXEL
    return images.astype(np.float32) - mean_pix

def data_generator(dataset, config,batch_size=1, shuffle=True, verbose = False, imageNames=[], add_hill=True, min_truth_sum = 0):
    b = 0  # batch item index
    image_index = -1
    if len(imageNames) > 0:
        image_ids = dataset.get_imageids(imageNames)
    else:
        image_ids = np.copy(dataset.image_ids)
    error_count = 0
    batch_hills = []
    skipped = 0

    while True:
        # Increment index to pick next image. Shuffle if at the start of an epoch.
        image_index = (image_index + 1) % len(image_ids)
        if shuffle and image_index == 0:
            np.random.shuffle(image_ids)

        # Get GT bounding boxes and masks for image.
        image_id = image_ids[image_index]
        image, gt_class_ids, gt_masks, hill = load_image_gt(dataset,image_id,  config, verbose, add_hill)

        # Init batch arrays
        can_init_hill = (not add_hill) or (add_hill and not hill is None)
        if b == 0 and can_init_hill:
            batch_images = np.zeros(
                (batch_size,) + image.shape, dtype=np.float32)
            if add_hill:
                batch_hills= np.zeros(
                       (batch_size,) + hill.shape, dtype=np.float32)
            batch_gt_class_ids = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
            batch_gt_masks = np.zeros(
                (batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))


        if hill is None:
            continue

        # if min_truth_sum > 0:
        #     n_masks = gt_masks.shape[2]
        #     for i in range(n_masks):
        #         m = gt_masks[:, :, i]
        #         s = np.sum(m)
        #         print("mask shape, s", m.shape, s)
        #         fn_tr = "test/{0}_{1}_{2}.png".format(b, i, s)
        #         plt.imsave(fn_tr, m, cmap='hot')
        #         if s < min_truth_sum:
        #             #mask is small, skip the whole image
        #             skipped = skipped + 1
        #             continue


        batch_hills[b] = hill #mold_image(hill.astype(np.float32), config)


        # Add to batch
        batch_images[b] = image #(image.astype(np.float32), config)
        batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
        batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

        b += 1

        # Batch full?
        if b >= batch_size:
            break

    print("Skipped {0} images out of {1}, min_truth_sum = {2}".format(skipped, batch_size, min_truth_sum))
    return batch_images, batch_gt_class_ids, batch_gt_masks, batch_hills

def build_dataset(directory):
    dataset = Dataset()
    dataset.load_LOL(directory)
    dataset.prepare()
    return dataset