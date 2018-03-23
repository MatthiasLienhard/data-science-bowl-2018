#functions from public kernel
#cv-score-calculation-takes-5-min-in-kernel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tqdm
from skimage.transform import resize
import os
import sys
from multiprocessing import Pool


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
            self.features=pd.DataFrame(index=ids)
        elif set(self.ids) != set(ids):
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
        self.masks= []
        self.labeled_masks = np.zeros((len(self.ids),self.height, self.width, 1), dtype=np.uint)
        sys.stdout.flush()
        for n, id_ in tqdm.tqdm(enumerate(self.ids), total=len(self.ids)):
            self.masks.append([])
            for k, mask_file in enumerate(next(os.walk(self.path + '/'+ id_ + '/masks/'))[2]):
                #print(mask_file)
                mask = scipy.misc.imread(self.path + '/' + id_ +  '/masks/' + mask_file)
                mask = np.expand_dims(resize(mask, (self.height, self.width), mode='constant',
                                              preserve_range=True), axis=-1)
                self.masks[n].append(mask)
                self.labeled_masks = np.maximum(self.labeled_masks, mask * k) #assuming they are not overlaying
        sys.stdout.flush()

    def show_image(self, idx="random"):
        print ("not tested yet")
        if idx == "random":
            idx = scipy.random.randint(0, len(self.ids))
        elif idx == "bad":
            # select image with poor performance
            idx=0 #not implemented yet

        plt.imshow(self.imags[idx])
        plt.show()
        plt.imshow(np.squeeze(self.labeled_masks[idx]))
        plt.show()

    #add feature functions add one or more columns to self.features
    #def add_feature_type(self):
        # type: e.g. colored vs grayscale, different stainings, clusters
    #def add_feature_nTouching(self):
        # number of touching nuclei
    #def add_feature_avgLabeledIou
    #def add_feature_unlabeledIou
    #def add_feature_avgIouScore
        # with different thresholds

#def get_iou(images, labeled=True):
    # unlabled masks
        # returns array of iou scores
    # labled masks
        # returns array of iou score lists (one per nucleus in truth)

#def get_score(images, th=None):
    # only on labeled masks
    # if th is in \[0.5,1\]:
        # returns array of fraction of recognized nuclei per image at threshold th
    # else
        # array with average fraction of recognized nuclei per image at all thresholds np.arange(0.5,0.95,0.05)
        # the average over all image should correspond to the score used by kaggle
#def show_image(images, idx):
    # sanity check function (e.g. plot the three images)
    # but also print score, rank, image classification, other features
    # idx can be "random", "worst", "bad", "average", "good","best"

if __name__=='__main__':

    test=Images()
    print("reading test images")
    test.read_images()
    print("reading test masks")
    test.read_masks()
    test.show_image()
