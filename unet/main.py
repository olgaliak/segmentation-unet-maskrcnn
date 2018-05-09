import argparse
import os
import random

from trainer import train_net
from unet_config import Config
from model_metrics import calc_jacc_img_msk
from data_reader import  get_patches_dir
from scorer import  check_predict_gold
from model import  get_unet

parser = argparse.ArgumentParser()

parser.add_argument("--lossmode", default="bce",
                    help="Loss mode: bce (binary cross entropy) or jcc (jaccard distance)" )
parser.add_argument("--cudadev", default="0",
                    help="Cuda device ID)" )

parser.add_argument("--runprefix",
                    help="Prefix to be added to the output folders" )
parser.add_argument("--runmode", default="train",
                    help="Run mode: train or test" )

args = parser.parse_args()


print("cuda device:", args.cudadev)
os.environ["CUDA_VISIBLE_DEVICES"]= args.cudadev

RUN_PREFIX = args.runprefix
print("RUN_PREFIX", RUN_PREFIX)
WEIGHTS_FLD = RUN_PREFIX + '_weights'
PRED_FLD = RUN_PREFIX + '_pred'
LOG_FLD = RUN_PREFIX + '_logs'

def train():
    print("Train mode")
    print("run_prefix", RUN_PREFIX)
    config = Config()
    loss_mode = args.lossmode
    model = train_net(WEIGHTS_FLD, LOG_FLD, PRED_FLD, config, loss_mode)

    print("calc_jacc for {}: N examples:: {}".format(dir, config.AMT_VAL))
    x_val, y_val = get_patches_dir(config.VAL_DIR, config, shuffleOn=False, amt=config.AMT_VAL)
    score, trs = calc_jacc_img_msk(model, x_val, y_val, batch_size = 4, n_classes = config.NUM_CLASSES)
    print("score, trs", score, trs)

def test():
    print("Test mode")
    config = Config()
    loss_mode = args.lossmode
    model_name = 'unet_cl4_step3_e5_tr600_v100_jk0.9817'

    model = get_unet(config, loss_mode)
    model.load_weights(os.path.join(WEIGHTS_FLD, model_name))

    check_predict_gold(model, model_name, PRED_FLD, config, loss_mode)

if __name__ == '__main__':
    random.seed(100)
    if args.runmode == "train":
        train()
    else:
        if args.runmode == "test":
            test()
        else:
            print("Unkmown run mode:", args.runmode)




