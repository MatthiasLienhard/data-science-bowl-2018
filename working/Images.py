import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tqdm
import skimage.transform
import skimage.segmentation
import skimage.measure
import os
import warnings
import sys
import copy
from collections import OrderedDict

from multiprocessing import Pool
import tensorflow as tf

class Images(object):
    def __init__(self, path="../input/stage1_train"):
        self.path=path
        self.images=None
        self.masks=None
        self.pred=None
        self.features=None
        self.nuc_features=None
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
            ret.nuc_features=ret.nuc_features.loc[ret.nuc_features['img_id']==idx]
            ret.nuc_features.index=range(ret.nuc_features.shape[0])
        if ret.pred is not None:
            ret.pred=list( ret.pred[i] for i in idx )
        return ret

    def n(self):
        return self.features.shape[0]

    def rescale(self, what, scale=(128,128,3),idx=None, dtype=np.uint8, **kwargs):
        if type(what) is str:
            if what =="images":
                imgs=self.images
            elif what == "masks":
                imgs=self.masks
        elif type(what) is np.ndarray:
            imgs=what
        else:
            raise ValueError('"what" should be "images" or "masks", or a numpy array')

        if idx is None:
            idx=range(len(imgs))
        elif np.max(idx)>len(imgs):
            raise ValueError('there are only {} images'.len(imgs))

        if scale is None:
            #rescale to original size
            scaled_images=[]
            for i in idx:
                height = self.features['size_x'][i]
                width  = self.features['size_y'][i]
                scaled_images.append(skimage.transform.resize(imgs[i], (height, width), **kwargs).astype(dtype))
        else:
            scaled_images = np.zeros((len(imgs),) + scale , dtype=dtype)
            for i in idx:
                scaled_images[i] = skimage.transform.resize(imgs[i], scale[:2], **kwargs).astype(dtype)
        return scaled_images
    def get_mask_boundaries(self, scale=(128,128), idx=None):
        _masks=self.get_masks(scale=scale, labeled=True, idx=idx)
        if scale is None:
            mask_boundaries=[]
            for i in range(len(_masks)):
                mask_boundaries.append(skimage.segmentation.find_boundaries(_masks[i]))
        else:
            mask_boundaries=np.zeros(_masks.shape, np.bool)
            for i in range(len(_masks)):
                mask_boundaries[i]=skimage.segmentation.find_boundaries(_masks[i])
        return(mask_boundaries)

    def get_images(self, scale=(128, 128), idx=None):
        if scale is not None:
            scale=scale + (3,)
        return self.rescale(what="images", scale=scale, idx=idx, dtype=np.uint8,  preserve_range=True, mode='reflect')
        #anti_alaiasing=True/False works for skimage.__version__ 0.14

    def get_masks(self,scale=(128,128), labeled=True, idx=None):
        if scale is not None:
            scale=scale + (1,)
        if labeled:
            dtype=np.uint16
            # requires to be done mask by mask, to avoid averages
        else:
            dtype=np.bool
        return self.rescale(what="masks", scale=scale, idx=idx, dtype=dtype,  preserve_range=True, mode='reflect')

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
        #self.mask_boundaries=[]
        nuclei_features=np.zeros((self.n(),5), dtype=np.int) #store 5 features of the mask: number of nuclei, mean size, sd, min, max
        #todo: more interesting features would be: circumfence
        sys.stdout.flush()
        height=self.features['size_x']
        width=self.features['size_y']
        #self.nuc_features=pd.DataFrame(columns=['nuc_ids','img_id','size', 'boundary'])
        nucf_list=[]

        for n, id_ in tqdm.tqdm(enumerate(self.features['ids']), total=self.n()):
            #m_size=[]
            #b_size=[]
            self.masks.append(np.zeros((height[n], width[n],1), dtype=np.uint16))
            for k, mask_file in enumerate(next(os.walk(self.path + '/'+ id_ + '/masks/'))[2]):
                #print(mask_file)
                mask = np.expand_dims(scipy.misc.imread(self.path + '/' + id_ +  '/masks/' + mask_file), axis=-1).astype(np.bool)
                #m_size.append(np.sum(mask))
                #boundary=
                #b_size.append(skimage.measure.perimeter(mask))
                #self.masks[n] = np.maximum(self.masks[n], mask) #unlabeled
                self.masks[n] = np.maximum(self.masks[n], mask * (k+1)) #assuming they are not overlaying
            #self.mask_boundaries.append(skimage.segmentation.find_boundaries(self.masks[n]))
            props=skimage.measure.regionprops(self.masks[n])
            b_size=[p.perimeter for p in props]
            m_size=[p.area for p in props]
            nuclei_features[n,0]=k+1
            nuclei_features[n,1]=np.mean(m_size).astype(np.int)
            nuclei_features[n,2]=np.std(m_size)
            nuclei_features[n,3]=np.min(m_size)
            nuclei_features[n,4]=np.max(m_size)
            nucf_list.append(pd.DataFrame(OrderedDict([ ('img_id', np.repeat(n,k+1)),('nuc_i', np.arange(1,k+2)),('size', m_size),('boundary', b_size) ] ) ))
        self.features=pd.concat((self.features,pd.DataFrame(data=nuclei_features,
                    columns=['nuclei_n', 'nuclei_meanSz','nuclei_stdSz', 'nuclei_minSz', 'nuclei_maxSz'])),
                    axis=1)
        self.nuc_features=pd.concat(nucf_list, ignore_index=True)
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
        print(self.features.drop(['ids'], axis=1).iloc[[idx]])
        if not self.images is None:
            plt.subplot(321)
            plt.imshow(self.images[idx])
            plt.title('original')
            #plt.grid(True)
        if not self.masks is None:
            #plt.subplot(322)
            #plt.imshow(np.squeeze(self.masks[idx]>0))
            #plt.title('unlabled mask')
            plt.subplot(323)
            plt.imshow(np.squeeze(self.masks[idx]))
            plt.title('labeled mask')
            #plt.subplot(324)
            #plt.imshow(np.squeeze(self.get_mask_boundaries(scale=None, idx=[idx])[0]))
            #plt.title('mask boundaries')
            if self.images is not None:
                _img=copy.deepcopy(self.images[idx])
                _img_b=self.get_mask_boundaries(scale=None, idx=[idx])[0].astype(np.bool)
                _img_b.shape=_img_b.shape[:2]
                _img[_img_b]=(255,0,0)
                plt.subplot(324)
                plt.imshow(_img)
                plt.title('masked image')
        if not self.pred is None:
            plt.subplot(325)
            plt.imshow(np.squeeze(self.pred[idx]))
            plt.title('prediction')
            if self.images is not None:
                _img=copy.deepcopy(self.images[idx])
                _img_b=skimage.segmentation.find_boundaries(self.pred[idx])
                _img_b.shape=_img_b.shape[:2]
                _img[_img_b]=(255,0,0)
                plt.subplot(326)
                plt.imshow(_img)
                plt.title('masked image (pred)')
            if self.masks is not None:
                plt.subplot(322)
                plt.imshow(np.squeeze((self.pred[idx]>0).astype(np.int)*2 + (self.masks[idx]>0).astype(np.int)))
                plt.title('diff map')
        plt.subplots_adjust(top=0.9, bottom=0.08, left=0.10, right=0.95, hspace=0.35,wspace=0.25)


        #plt.figure(figsize=(18, 12), dpi= 80, facecolor='w', edgecolor='k')
        plt.show()

    def add_predictions(self, model,**kwargs):
        pred=model.predict(self, **kwargs)
        self.pred=pred
        scores=np.zeros((len(self.pred),14), dtype=np.float)
        nuc_scores=[]

        if self.masks is not None:
            print("computing scores... ")
            for i in tqdm.tqdm(range(len(pred))):
                scores[i, 0] = np.max(self.pred[i])
                props=skimage.measure.regionprops(self.pred[i])
                #b_size=[p.perimeter for p in props]
                a_size=[p.area for p in props]
                scores[i, 1] = np.mean(self.pred[i])
                scores[i, 2] = iou((self.masks[i]>0).astype(np.int), (self.pred[i]>0).astype(np.int))[0]
                #scores[i, 3] = iou_score(self.masks[i], self.pred[i])
                #for j, th in enumerate(np.arange(.5,1,.05)):
                #    scores[i, j+3] = iou_score(self.masks[i], self.pred[i], th=[th])
                scores[i, range(4,14)]= iou_score(self.masks[i], self.pred[i])
                scores[i, 3]= np.mean(scores[i, range(4,14)])
                nuc_scores += iou(truth=self.masks[i], pred=self.pred[i]).tolist()
            colnames=['n_pred','mean_size_pred','iou_fg', 'iou_score'] + ['iou_th'+str(th) for th in np.arange(50,100,5)]
            self.features=pd.concat([self.features, pd.DataFrame(scores, columns=colnames)], axis=1)
            self.nuc_features=pd.concat([self.nuc_features, pd.DataFrame({'iou': nuc_scores}) ], axis=1)

    def write_submission(self, file_name):

        if self.pred is None:
            warnings.warn("no prediction found")
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
            pred_c=np.argmax(np.bincount(pred_c)) #majoritiy vote
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
    n_truth=np.max(truth)#len(np.unique(truth))-1
    n_pred=np.max(truth)#len(np.unique(pred))-1
    for _th in th :
        true_pos=np.sum(iou_vals>_th)
        mean_iou.append(true_pos/(n_truth + n_pred - true_pos))
    return(mean_iou)







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

