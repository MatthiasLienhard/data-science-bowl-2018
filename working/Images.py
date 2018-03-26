#functions from public kernel
#cv-score-calculation-takes-5-min-in-kernel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tqdm
from skimage.transform import resize
import os
import warnings
import sys
from multiprocessing import Pool
import tensorflow as tf

class Images(object):
    def __init__(self, path="../input/stage1_train",width=128, height=128, channels=3):
        self.path=path
        self.width=width
        self.height=height
        self.channels=channels
        self.ids=None
        self.images=None
        self.masks=None
        self.labeled_masks=None
        self.pred=None
        self.labeled_pred=None
        self.features=None
    def get_ids(self):
        ids=next(os.walk(self.path))[1]
        if self.ids is None:
            self.ids=ids
            self.features=pd.DataFrame(data=self.ids, columns=['ids'])
        elif not set(self.ids).issubset(set(ids)) :
            raise ValueError('ids do not match')

    def read_images(self):
        self.get_ids()
        self.images=np.zeros((len(self.ids), self.height, self.width, self.channels), dtype=np.uint8)
        dims=np.zeros(shape=(len(self.ids), 2 ))
        sys.stdout.flush()
        for n, id_ in tqdm.tqdm(enumerate(self.ids), total=len(self.ids)):
            img = scipy.misc.imread(self.path + '/' + id_ + '/images/' + id_ + '.png')[:,:,:self.channels]
            dims[n]=img.shape[:2]
            self.images[n]= resize(img, (self.height, self.width), mode='constant', preserve_range=True)
        sys.stdout.flush()
        self.features['size_x']=dims[:,0]
        self.features['size_y']=dims[:,1]


    def read_masks(self):
        # adds images labeled and unlabeled masks
        # idea: better in original size?
        self.get_ids()
        self.labeled_masks = np.zeros((len(self.ids),self.height, self.width, 1), dtype=np.uint)
        self.masks = np.zeros((len(self.ids),self.height, self.width, 1), dtype=np.bool)
        nuclei_features=np.zeros((len(self.ids),5)) #store 5 features of the mask
        sys.stdout.flush()
        for n, id_ in tqdm.tqdm(enumerate(self.ids), total=len(self.ids)):
            m_size=[]
            for k, mask_file in enumerate(next(os.walk(self.path + '/'+ id_ + '/masks/'))[2]):
                #print(mask_file)
                mask = scipy.misc.imread(self.path + '/' + id_ +  '/masks/' + mask_file).astype(np.bool)
                mask = np.expand_dims(resize(mask, (self.height, self.width), mode='constant',
                                              preserve_range=True), axis=-1).astype(np.bool)
                m_size.append(np.sum(mask))
                self.masks[n] = np.maximum(self.masks[n], mask) 
                self.labeled_masks[n] = np.maximum(self.labeled_masks[n], mask * (k+1)) #assuming they are not overlaying
            nuclei_features[n,0]=k+1
            nuclei_features[n,1]=np.mean(m_size)
            nuclei_features[n,2]=np.std(m_size)
            nuclei_features[n,3]=np.min(m_size)
            nuclei_features[n,4]=np.max(m_size)

        self.features=pd.concat((self.features,pd.DataFrame(data=nuclei_features,
                    columns=['nuclei_n', 'nuclei_meanSz','nuclei_stdSz', 'nuclei_minSz', 'nuclei_maxSz'])),
                    axis=1)
        sys.stdout.flush()

    def show_image(self, idx="random"):
        # sanity check function (e.g. plot the three images)
        # but also print score, rank, image classification, other features
        # idx can be "random", "worst", "bad", "average", "good","best"

        if idx == "random":
            idx = scipy.random.randint(0, len(self.ids))
        elif idx == "bad":
            # select image with poor performance
            idx=0 #not implemented yet
        print(self.features.iloc[[idx]])
        if not self.images is None:
            plt.subplot(221)
            plt.imshow(self.images[idx])
            #plt.grid(True)
        if not self.masks is None:
            plt.subplot(222)
            plt.imshow(np.squeeze(self.masks[idx]))
            #plt.grid(True)
        if not self.labeled_masks is None:
            plt.subplot(223)
            plt.imshow(np.squeeze(self.labeled_masks[idx]))
        if not self.pred is None:
            plt.subplot(224)
            plt.imshow(np.squeeze(self.pred[idx]))

        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
        plt.show()

    def write_submission(self, file_name):
        if self.pred is None:
            warnings.warn("")
        # Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
        def rle_encoding(x):
            dots = np.where(x.T.flatten() == 1)[0]
            run_lengths = []
            prev = -2
            for b in dots:
                if (b>prev+1): run_lengths.extend((b + 1, 0))
                run_lengths[-1] += 1
                prev = b
            return run_lengths

        def prob_to_rles(x, cutoff=0.5):

            for i in range(1, self.pred.max() + 1):
                yield rle_encoding(self.pred == i)
        ids_out = []
        rles = []
        for n, id_ in enumerate(self.ids):
            rle = list(prob_to_rles(self.upsampled(self.pred)))
            rles.extend(rle)
            ids_out.extend([id_] * len(rle))
        sub = pd.DataFrame()
        sub['ImageId'] = ids_out
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        sub.to_csv(file_name, index=False)





    #add feature functions add one or more columns to self.features
    #def add_feature_type(self):
        # type: e.g. colored vs grayscale, different stainings, clusters
    #def add_feature_nTouching(self):
        # number of touching nuclei
    #def add_feature_avgLabeledIou
    #def add_feature_unlabeledIou
    #def add_feature_avgIouScore
        # with different thresholds

def iou(truth, pred):
    # unlabled masks: returns 1 iou score (bg vs fg classification)
    # labled masks: returns array of iou scores (one per nucleus in truth)
    nclass=np.max(truth).astype(np.int)
    iou=np.zeros((nclass), dtype=float)
    for truth_c in np.arange(1,nclass+1, dtype=np.int):
        sel=np.where(truth == truth_c)
        pred_c=pred[sel].astype(np.int)
        #print(np.bincount(pred_c))

        pred_c=np.argmax(np.bincount(pred_c)) #majoritiy vote... is this always the best?+
        #print(pred_c)
        if pred_c > 0:
            isect=np.sum(np.logical_and(truth==truth_c, pred==pred_c))
            union=np.sum(np.logical_or(truth==truth_c, pred==pred_c))
            iou[truth_c-1]=isect/union
            #print('I={} U={} IoU={}'.format(isect, union, iou[truth_c-1]))
        else:
            iou[truth_c-1]=0

    return iou

def iou_score(truth, pred, th=np.arange(.5,1,.05)):
    # only on labeled masks
    # if th is in \[0.5,1\]:
        # returns array of fraction of recognized nuclei per image at threshold th
    # else
        # array with average fraction of recognized nuclei per image at all thresholds np.arange(0.5,0.95,0.05)
        # the average over all image should correspond to the score used by kaggle
    iou_vals=iou(truth, pred)
    mean_iou=[]
    for th_ in th :
        mean_iou.append(np.mean(iou_vals>th_))
    return(np.mean(mean_iou))







if __name__=='__main__':
    import ModelUNet_rep
    train=Images()
    train.get_ids()
    train.ids=train.ids[:10]
    train.features=train.features[:10]
    print("reading training images")
    train.read_images()
    print("reading training masks")
    train.read_masks()
    model=ModelUNet_rep.ModelUNet('model-dsbowl2018-1.h5')
    train.pred=model.predict_unlabeld(train)
    train.labeled_pred=model.label(train.pred)
    print(iou(train.masks[0], train.pred[0]))
    print(iou(train.labeled_masks[0], train.labeled_pred[0]))
    train.show_image()

