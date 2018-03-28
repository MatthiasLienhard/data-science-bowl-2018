import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tqdm
import skimage.transform
import os
import warnings
import sys
import copy
from multiprocessing import Pool
import tensorflow as tf

class Images(object):
    def __init__(self, path="../input/stage1_train"):
        self.path=path
        self.images=None
        self.masks=None
        self.pred=None
        self.features=None
        self.add_ids()

    def add_ids(self):#idea: not all but list of ids/indices?
        ids=next(os.walk(self.path))[1]
        if self.features is None:
            self.features=pd.DataFrame(data=ids, columns=['ids'])
        elif not set(self.features['ids']).issubset(set(ids)) :
            raise ValueError('ids do not match')

    def subset(self, idx):
        ret=copy.deepcopy(self)
        ret.features=ret.features.loc[idx]
        ret.features.index=range(ret.n())
        if ret.images is not None:
            ret.images=list( ret.images[i] for i in idx )
        if ret.masks is not None:
            ret.masks=list( ret.masks[i] for i in idx )
        if ret.pred is not None:
            ret.pred=list( ret.pred[i] for i in idx )
        return ret

    def n(self):
        return self.features.shape[0]

    def rescale(self, imgs, scale=(128,128,3), dtype=np.uint8, **kwargs):
        if scale is None:
            #rescale to original size
            scaled_images=[]
            for i in range(len(imgs)):
                height = self.features['size_x'][i]
                width  = self.features['size_y'][i]
                scaled_images.append(skimage.transform.resize(imgs[i], (height, width), **kwargs).astype(dtype))

        else:
            scaled_images = np.zeros((len(imgs),) + scale , dtype=dtype)
            for i in range(len(imgs)):
                scaled_images[i] = skimage.transform.resize(imgs[i], scale[:2], **kwargs).astype(dtype)
        return scaled_images

    def get_images(self, scale=(128, 128)):
        return self.rescale(self.images, scale + (3,), dtype=np.uint8,  preserve_range=True, mode='reflect')
        #anti_alaiasing=True/False works for skimage.__version__ 0.14

    def get_masks(self,scale=(128,128), labeled=True):
        if labeled:
            dtype=np.uint16
            # requires to be done mask by mask, to avoid averages
        else:
            dtype=np.bool
        return self.rescale(self.masks, scale + (1,), dtype,  preserve_range=True, mode='reflect')

    def load_images(self):
        self.images=[]
        dims=np.zeros(shape=(self.n(), 3 ), dtype=np.int)
        sys.stdout.flush() 
        for n, id_ in tqdm.tqdm(enumerate(self.features['ids']), total=self.n()):
            img = scipy.misc.imread(self.path + '/' + id_ + '/images/' + id_ + '.png', mode='RGB') #remove alpha channel
            dims[n]=img.shape
            if np.all(img[:,:,0]==img[:,:,1]) and np.all(img[:,:,0]==img[:,:,2]):
                dims[n,2]=1
                #is it worth removing color information?
            self.images.append(img)
            #self.images[n]= resize(img, (self.height, self.width), mode='constant', preserve_range=True)
        sys.stdout.flush()
        self.features['size_x']=dims[:,0].astype(np.int)
        self.features['size_y']=dims[:,1].astype(np.int)
        self.features['n_channels']=dims[:,2].astype(np.int)

    def load_masks(self):
        # adds images labeled and unlabeled masks
        #self.add_ids()

        #self.masks = np.zeros((len(self.ids),self.height, self.width, 1), dtype=np.uint16)
        self.masks=[]
        nuclei_features=np.zeros((self.n(),5), dtype=np.int) #store 5 features of the mask: number of nuclei, mean size, sd, min, max
        #todo: more interesting features would be: circumfence
        sys.stdout.flush()
        height=self.features['size_x']
        width=self.features['size_y']
        for n, id_ in tqdm.tqdm(enumerate(self.features['ids']), total=self.n()):
            m_size=[]
            self.masks.append(np.zeros((height[n], width[n],1), dtype=np.uint16))
            for k, mask_file in enumerate(next(os.walk(self.path + '/'+ id_ + '/masks/'))[2]):
                #print(mask_file)
                mask = np.expand_dims(scipy.misc.imread(self.path + '/' + id_ +  '/masks/' + mask_file), axis=-1).astype(np.bool)
                m_size.append(np.sum(mask))
                #self.masks[n] = np.maximum(self.masks[n], mask) #unlabeled
                self.masks[n] = np.maximum(self.masks[n], mask * (k+1)) #assuming they are not overlaying

            nuclei_features[n,0]=k+1
            nuclei_features[n,1]=np.mean(m_size).astype(np.int)
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
            idx = scipy.random.randint(0, self.n())
        elif idx == "bad":
            # select image with poor performance
            idx=0 #not implemented yet
        print(self.features.iloc[[idx]])
        if not self.images is None:
            plt.subplot(221)
            plt.imshow(self.images[idx])
            plt.title('original')
            #plt.grid(True)
        if not self.masks is None:
            plt.subplot(222)
            plt.imshow(np.squeeze(self.masks[idx]>0))
            plt.title('unlabled mask')
            plt.subplot(223)
            plt.imshow(np.squeeze(self.masks[idx]))
            plt.title('labeled mask')
        if not self.pred is None:
            plt.subplot(224)
            plt.imshow(np.squeeze(self.pred[idx]))
            plt.title('prediction')

        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
        plt.show()

    def add_pred(self, pred):
        self.pred=pred
        scores=np.zeros((len(self.pred),3), dtype=np.float)
        if self.masks is not None:
            for i in range(len(pred)):
                scores[i, 0] = iou_score(self.masks[i], self.pred[i])
                scores[i, 1] = iou_score(self.masks[i], self.pred[i], th=[.5])
                scores[i, 2] = iou_score(self.masks[i], self.pred[i], th=[.95])

            self.features['iou_score']=scores[:, 0]
            self.features['iou_th50'] = scores[:, 1]
            self.features['iou_th95'] = scores[:, 2]

    def write_submission(self, file_name):

        if self.pred is None:
            warnings.warn("no labeled prediction found")
            return 1

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

        def prob_to_rles(x):
            for i in range(1, x.max() + 1):
                yield rle_encoding(x == i)
        ids_out = []
        rles = []
        for n, id_ in enumerate(self.features['ids']):
            rle = list(prob_to_rles(self.pred[n]))
            rles.extend(rle)
            ids_out.extend([id_] * len(rle))
        sub = pd.DataFrame()
        sub['ImageId'] = ids_out
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        sub.to_csv(file_name, index=False)
        return 0

    #def add_iou_score(self):





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
        if sel is None:
            pred_c=0
            warnings.warn("found mask of size 0")
        else:
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
    #train.get_ids()
    train.ids=train.ids[:10]
    train.features=train.features[:10]
    print("reading training images")
    train.read_images()
    print("reading training masks")
    train.read_masks()
    #model=ModelUNet_rep.ModelUNet('model-dsbowl2018-1.h5')
    #train.pred=model.predict_unlabeld(train)
    #train.labeled_pred=model.label(train.pred)
    #print(iou(train.masks[0], train.pred[0]))
    #print(iou(train.labeled_masks[0], train.labeled_pred[0]))
    train.show_image()

