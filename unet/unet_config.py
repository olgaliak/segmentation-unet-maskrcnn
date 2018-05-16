import numpy as np

class Config(object):
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 224
    ISZ = 224
    N_CHANNELS = 4

    EPOCS = 5
    N_STEPS = 5
    BATCH_SIZE = 16


    AMT_TRAIN = 60
    AMT_VAL = 10
    AMT_SMALL_VAL = 10

    TRAIN_DIR = 'D:\\data\\LOL\\processedraw\\4_class_smaller\\train_aug'
    VAL_DIR = 'D:\\data\\LOL\\unet\\test'  # 'D:\\data\\LOL\\processedraw\\4_class_smaller\\test'
    SMAL_VAL_DIR = 'D:\\data\\LOL\\processedraw\\4_class_smaller\\test'


    IMAGE_PADDING = True  # currently, the False option is not supported

    # Number of classes
    NUM_CLASSES = 4
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 4

    # Image mean (RGB)
    #MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    MEAN_PIXEL_LOL = np.array([90, 91, 70, 141])
    VARIANCE_LOL = np.array([3587, 3146, 2022, 6594])
    
    IMAGES_PER_GPU = 2

    GPU_COUNT = 1

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])



    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
