import os
import time
from keras.callbacks import ModelCheckpoint, TensorBoard

from data_reader import get_patches_dataset
from dataset_helper import build_dataset
from model import get_unet
from model_metrics import calc_jacc_img_msk
from scorer import  check_predict_gold
from scorer import check_predict_small_test


def train_net(weights_folder, logs_folder, progress_predict_dir, config, loss_mode):

    print("start train net")
    train_dataset = build_dataset(config.TRAIN_DIR)
    val_dataset = build_dataset(config.VAL_DIR)
    x_trn, y_trn =  get_patches_dataset(train_dataset, config, shuffleOn=True, amt= config.AMT_TRAIN)
    x_val, y_val = get_patches_dataset(val_dataset, config, shuffleOn=False, amt= config.AMT_VAL)
    model = get_unet(config, loss_mode)
    os.makedirs(weights_folder, exist_ok=True)
    #model.load_weights('weights/unet_cl2_step0_e5_tr600_v600_jk0.6271')
    model_checkpoint = ModelCheckpoint(os.path.join(weights_folder,'unet_tmp.hdf5'), monitor='loss', save_best_only=True)
    tb_callback = TensorBoard(log_dir=logs_folder, histogram_freq=0, batch_size=config.BATCH_SIZE,
                              write_graph=True, write_grads=False, write_images=True,
                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    start_time = time.time()
    for i in range(config.N_STEPS):
        print("Step i", i)
        model.fit(x_trn, y_trn, batch_size= config.BATCH_SIZE, epochs= config.EPOCS, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint, tb_callback], validation_data=(x_val, y_val))

        print("---  Training for %s seconds ---" % (time.time() - start_time))
        score, trs = calc_jacc_img_msk(model, x_trn, y_trn, config.BATCH_SIZE, config.NUM_CLASSES)
        print('train jk', score)

        score, trs = calc_jacc_img_msk(model, x_val, y_val, config.BATCH_SIZE, config.NUM_CLASSES)
        print('val jk', score)
        score_str = '%.4f' % score
        model_name = 'unet_cl{0}_step{1}_e{2}_tr{3}_v{4}_jk{5}'.format(config.NUM_CLASSES, i, config.EPOCS,
                                                                       config.AMT_TRAIN, config.AMT_VAL,score_str)
        print("Weights: ", model_name)
        model.save_weights(os.path.join(weights_folder, model_name))

        #if (i % 10 == 0):
        check_predict_gold(model, model_name, progress_predict_dir, config, loss_mode)
        check_predict_small_test(model, model_name, progress_predict_dir, config, loss_mode)

        #Get ready for next step
        del x_trn
        del y_trn
        x_trn, y_trn = get_patches_dataset(train_dataset, config, shuffleOn=True, amt=config.AMT_TRAIN)
    return model
