from config import Config

class inputConfig():
    NUM_CLASSES = 4
    CLASS_DICT = {1: 'waterways', 2: 'fieldborders', 3: 'terraces', 4: 'wsb'}
    CATEGORIES = list(CLASS_DICT.values())
    CATEGORIES_VALUES = list(CLASS_DICT.keys())
    NUM_EPOCHES = 1
    # TRAIN_LAYERS = 'all'
    # SAVE_TRAIN = 'logs'
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    JPG_NAME = 'jpg4'
    OUTPUT_DIR = '/home/tinzha/Projects/LandOLakes/posthack/metrics'
    MODEL_DIR = '/home/tinzha/Projects/LandOLakes/logs'

class modelConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 256

    # Number of classes (including background)
    NUM_CLASSES = 1 + inputConfig.NUM_CLASSES

class inferenceConfig(modelConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1