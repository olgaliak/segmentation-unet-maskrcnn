import numpy as np
from keras.applications.vgg16 import VGG16
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, merge
from keras import backend as K
K.set_image_data_format("channels_last")
#K.set_image_dim_ordering("th")


import losses
import metrics

CACHED_UNET = None

def get_vgg_7conv(ims, nchannels, n_cls):
    input_shape_base = (None, None, 4)
    img_input = Input(input_shape_base)
    vgg16_base = VGG16(input_tensor=img_input, include_top=False, weights=None)
    #for l in vgg16_base.layers:
    #    l.trainable = True

    conv1 = vgg16_base.get_layer("block1_conv2").output
    conv2 = vgg16_base.get_layer("block2_conv2").output
    conv3 = vgg16_base.get_layer("block3_conv3").output
    pool3 = vgg16_base.get_layer("block3_pool").output

    conv4 = Conv2D(384, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block4_conv1")(pool3)
    conv4 = Conv2D(384, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block4_conv2")(conv4)
   # pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=None, name='block4_pool')(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block5_conv1")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block5_conv2")(conv5)
   # pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)

    conv6 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block6_conv1")(pool5)
    conv6 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block6_conv2")(conv6)
    #pool6 = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(conv6)
    pool6 = MaxPooling2D((2, 2), strides=(2,2), name='block6_pool')(conv6)

    conv7 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block7_conv1")(pool6)
    conv7 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block7_conv2")(conv7)

    up8 = concatenate([Conv2DTranspose(384, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='valid')(conv7), conv6], axis=3)
    #up8 = merge([Conv2DTranspose(384, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv7), conv6], mode='concat', concat_axis=3)
    conv8 = Conv2D(384, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up8)

    up9 = concatenate([Conv2DTranspose(256, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv8), conv5], axis=3)
    conv9 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up9)

    up10 = concatenate([Conv2DTranspose(192, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv9), conv4], axis=3)
    conv10 = Conv2D(192, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up10)

    up11 = concatenate([Conv2DTranspose(128, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv10), conv3], axis=3)
    conv11 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up11)

    up12 = concatenate([Conv2DTranspose(64, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv11), conv2], axis=3)
    conv12 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up12)

    up13 = concatenate([Conv2DTranspose(32, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv12), conv1], axis=3)
    conv13 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up13)

    # #Batch normalization
    #conv13 = BatchNormalization(mode=0, axis=1)(conv13)

    conv13 = Conv2D(n_cls, (1, 1), activation='sigmoid')(conv13)
    #conv13 = Conv2D(1, (1, 1))(conv13)
    #conv13 = Activation("sigmoid")(conv13)
    model = Model(img_input, conv13)

    # Recalculate weights on first layer
    conv1_weights = np.zeros((3, 3, nchannels, 64), dtype="float32")
    vgg = VGG16(include_top=False, input_shape=(ims, ims, 3))
    conv1_weights[:, :, :3, :] = vgg.get_layer("block1_conv1").get_weights()[0][:, :, :, :]
    bias = vgg.get_layer("block1_conv1").get_weights()[1]
    model.get_layer('block1_conv1').set_weights((conv1_weights, bias))
    return model

def get_loss_func(loss_mode):
    loss_func = 'binary_crossentropy'
    if loss_mode == "bcedice":
       loss_func = losses.dice_coef_loss_bce

    print("loss:", loss_func)
    return  loss_func

def get_unet_pretrained(config, loss_mode):
    model = get_vgg_7conv(config.ISZ, config.N_CHANNELS, config.NUM_CLASSES)
    loss_func = get_loss_func(loss_mode)
    model.compile(optimizer=Adam(), loss=loss_func,  metrics=[metrics.jaccard_coef,  metrics.jacard_coef_flat, metrics.jaccard_coef_int,  metrics.dice_coef, 'accuracy'])
    return model

def get_unet(config, loss_mode):
    global CACHED_UNET
    if CACHED_UNET == None:
        CACHED_UNET = get_unet_pretrained(config, loss_mode)
    return  CACHED_UNET
