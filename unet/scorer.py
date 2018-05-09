import os
import  numpy as np
import  matplotlib.pyplot as plt

from sklearn.metrics import jaccard_similarity_score

from data_reader import  get_patches_dir
from model import  get_unet

GOLD_IMG_4Classes_3 = ['fieldborders_7_4109150_merged', 'fieldborders_28_4109150_merged', 'wsb_9_4109150_merged']
GOLD_IMG_4Classes_4 = ['waterways_2285_4109150_merged', 'waterways_2633_4109150_merged',
                       'waterways_1285_4109150_merged', 'fieldborders_19_4109150_merged']
AMT_SMALL_VAL = 100


def predict_from_val(model, amt_pred, trs, config, x):
    prd = np.zeros((amt_pred, config.ISZ, config.ISZ, config.NUM_CLASSES)).astype(np.float32)
    tmp = model.predict(x, batch_size=1)
    for i in range(amt_pred):
        prd_i = tmp[i]
        for c in range(config.NUM_CLASSES):
            prd[i,:,:,c] = prd_i[:,:,c] > trs[c]
    return prd

def display_pred(pred_res, true_masks, config, modelName, amt_pred, trs, min_pred_sum, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    trs_str = "trs_none"
    if trs != None:
        trs_str = '_'.join([str(x) for x in trs])
        trs_str = "trs_" + trs_str

    #print("Saving predictions when np.sum(pred) >", min_pred_sum)
    nothing_saved = True
    for p in range(amt_pred):
        for i in range(config.NUM_CLASSES):
            pred = pred_res[p,:, :, i]
            true_mask = true_masks[p, :, :, i]

            sum_pred = np.sum(pred)
            sum_mask = np.sum(true_mask)
            if sum_pred > min_pred_sum :
                    #and sum_mask > min_pred_sum:

                jk = jaccard_similarity_score(true_mask, pred)
                #print("Calc jaccard", jk)
                fn = os.path.join(output_folder,"{4}{0}_p{1}_cl{2}_{3}.png".format(modelName, p, i, trs_str, jk))
                #print("Saving  predictions with np.sum {0} to  {1}".format(sum, fn))
                plt.imsave(fn, pred, cmap='hot')

                fn_tr= os.path.join(output_folder,"{4}{0}_p{1}_TRUE_cl{2}_{3}.png".format(modelName, p, i, trs_str, jk))
                plt.imsave(fn_tr, true_mask, cmap='hot')

                nothing_saved = False

    if (nothing_saved):
        print("All predictions did not satisfy: sum_pred > min_pred_sum, nothing saved. Min_pred_sum:", min_pred_sum)


def check_predict_folder(model, model_name, val_dir,  predict_dir,  config, loss_mode, amt_pred, verbose, imageNames =[]):
    x_val, y_val = get_patches_dir(val_dir, config, shuffleOn=False, amt=amt_pred, verbose=verbose,
                                imageNames=imageNames)
    for i in range(1,10):
        tr = []
        for j in range(config.NUM_CLASSES):
            tr.append(i / 10.0)
        check_predict_model(model, model_name, config, predic_cnt=amt_pred,
                      trs = tr, x=x_val, y=y_val,  min_pred_sum = 10,
                      output_folder= predict_dir)

def check_predict_small_test(model, model_name, predict_dir, config, loss_mode):
    verbose = False
    amt_pred = config.AMT_SMALL_VAL
    predict_dir = predict_dir + "_small_test"
    val_dir = config.SMAL_VAL_DIR

    return check_predict_folder(model, model_name, val_dir, predict_dir, config, loss_mode, amt_pred, verbose)


def check_predict_gold(model, model_name, predict_dir, config, loss_mode):
    amt_pred = 3
    verbose = False
    imageNames = GOLD_IMG_4Classes_4
    predict_dir = predict_dir + "_gold"
    val_dir = config.VAL_DIR
    return check_predict_folder(model, model_name, val_dir, predict_dir, config, loss_mode, amt_pred, verbose, imageNames= imageNames)

def check_predict_model(model, model_name, config, predic_cnt, trs, x, y, min_pred_sum, output_folder):
    pred_res = predict_from_val(model, predic_cnt, trs, config, x)
    display_pred(pred_res, y, config, modelName=model_name, amt_pred=predic_cnt, trs=trs,
                 min_pred_sum=min_pred_sum,
                 output_folder=output_folder)


def check_predict(model_name, weights_folder, config, loss_mode, predic_cnt, trs, x, y, min_pred_sum, output_folder):
    model = get_unet(config, loss_mode)
    model.load_weights(os.path.join(weights_folder, model_name))
    check_predict_model(model, model_name, config, predic_cnt, trs, x, y, min_pred_sum, output_folder)
