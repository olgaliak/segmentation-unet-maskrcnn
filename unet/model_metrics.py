import matplotlib as plt
import  numpy as np
from sklearn.metrics import jaccard_similarity_score
# def calc_jacc(model, dir, amt_pred, debug = False, min_truth_sum = 0):
#     print("calc_jacc for {}: N examples:: {}".format(dir, amt_pred))
#     img, msk = get_patches(directory=dir, shuffleOn=False, amt=amt_pred, min_truth_sum = min_truth_sum)
#     return calc_jacc_img_msk(model, img, msk, debug)


def calc_jacc_img_msk(model, img, msk, batch_size, n_classes):
    prd = model.predict(img, batch_size= batch_size)
    #print("prd.shape {0}, msk.shape {1}". format(prd.shape, msk.shape))
     #prd.shape, msk.shape (16, 2, 256, 256) (16, 2, 256, 256)
    avg, trs = [], []

    for i in range(n_classes):
        t_msk = msk[:, i, :, :] # t_mask shape is (Npredictions, H, W)
        t_prd = prd[:, i, :, :]
        t_msk = t_msk.reshape(msk.shape[0] * msk.shape[2], msk.shape[3]) # shape is Npredictions*W, H
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])

        m, b_tr = 0, 0
        for j in range(10):
            tr = j / 10
            pred_binary_mask = t_prd > tr

            jk = jaccard_similarity_score(t_msk, pred_binary_mask)
            if jk > m:
                m = jk
                b_tr = tr

        print("i, m, b_tr", i, m, b_tr)
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / n_classes
    return score, trs
