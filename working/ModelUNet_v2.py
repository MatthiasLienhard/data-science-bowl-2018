
import os
import warnings

import numpy as np
import skimage.morphology
import skimage.segmentation
import skimage.feature
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout, Lambda
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, load_model
from Images import Images
import tqdm
import cv2

import scipy.ndimage

class ModelUNet(object): #maybe define prototype?
    @staticmethod
    def mean_iou(y_true, y_pred):
        #average IOU over thresholds from 0.5 to 0.95
        #BUT: on whole image, not on individual lables
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    @staticmethod
    def init_model(shape):
        inputs = Input(shape)
        s = Lambda(lambda x: x / 255) (inputs)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
        c1 = Dropout(0.1) (c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
        c2 = Dropout(0.1) (c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
        c3 = Dropout(0.2) (c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
        c4 = Dropout(0.2) (c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
        c5 = Dropout(0.3) (c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
        c6 = Dropout(0.2) (c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
        c7 = Dropout(0.2) (c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
        c8 = Dropout(0.1) (c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
        c9 = Dropout(0.1) (c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[ModelUNet.mean_iou])
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[ModelUNet.mean_iou])
        return model
        #self.model.summary()


    def __init__(self, shape=(128,128,3), name="model_unet_v2"):
        self.area_model_file=name+"_area.h5"
        self.area_fit_history=None
        self.boundary_model_file=name+"_boundary.h5"
        self.boundary_fit_history=None
        self.trained_boundary=False
        self.trained_area=False
        if os.path.isfile(self.area_model_file):
            print("found area model file "+self.area_model_file)
            self.trained_area=True
            self.area_model=load_model(self.area_model_file, custom_objects={'mean_iou': ModelUNet.mean_iou})
        if os.path.isfile(self.boundary_model_file):
            print("found boundary model file "+self.boundary_model_file)
            self.trained_boundary=True
            self.boundary_model=load_model(self.boundary_model_file, custom_objects={'mean_iou': ModelUNet.mean_iou})
        if self.trained_area and self.trained_boundary:
            self.trained=True
            self.shape=self.area_model.input_shape[1:4]
            if self.area_model.input_shape != self.boundary_model.input_shape:
                # e.g. (None, 256, 256, 3)
                warnings.warn("shapes of models do not match!!")
        else:
            # make new model
            self.shape=shape
            self.trained=False
            


    def fit_model(self, train:Images):
        self.fit_area_model(train)
        self.fit_boundary_model( train)
        self.trained=True


    def fit_area_model(self, train:Images, force=False):
        if not self.trained_area or force:
            self.area_model=ModelUNet.init_model(self.shape)
            earlystopper = EarlyStopping(patience=5, verbose=1)
            checkpointer = ModelCheckpoint(self.area_model_file, verbose=1, save_best_only=True)
            self.area_fit_history = self.area_model.fit(train.get_images(self.shape[:2]),
                        train.get_masks(self.shape[:2],labeled=False),
                        validation_split=0.1, batch_size=16, epochs=50,
                        callbacks=[earlystopper, checkpointer])
            self.trained_area=True
            self.area_model = load_model(self.area_model_file, custom_objects={'mean_iou': ModelUNet.mean_iou,})
    def fit_boundary_model(self, train:Images, force=False):
        if not self.trained_boundary or force:
            self.boundary_model=ModelUNet.init_model(self.shape)
            earlystopper = EarlyStopping(patience=5, verbose=1)
            checkpointer = ModelCheckpoint(self.boundary_model_file, verbose=1, save_best_only=True)
            self.boundary_fit_history = self.boundary_model.fit(train.get_images(self.shape[:2]),
                        train.get_mask_boundaries(self.shape[:2]),
                        validation_split=0.1, batch_size=16, epochs=50,
                        callbacks=[earlystopper, checkpointer])
            self.trained_boundary=True
            # not sure whether this is required
            # but the intention is when last training epoche was not optimal, the saved model should be better than
            self.boundary_model = load_model(self.boundary_model_file, custom_objects={'mean_iou': ModelUNet.mean_iou})

    def predict_area(self, img:Images, th=None):
        if self.trained_area:
            #self.area_model = load_model(self.area_model_file, custom_objects={'mean_iou': ModelUNet.mean_iou})
            preds = self.area_model.predict(img.get_images(self.shape[:2]), verbose=2)
            if not th is None:
                # Threshold predictions
                preds = preds > th
                # todo: da kann man sich noch was besseres einfallen lassen
        else:
            preds=None
            warnings.warn('Model not trained yet')
        return preds

    def predict_boundary(self, img:Images, th=None):
        if self.trained_boundary:
            #self.boundary_model = load_model(self.boundary_model_file, custom_objects={'mean_iou': ModelUNet.mean_iou,'iou_loss':iou_loss})
            preds = self.boundary_model.predict(img.get_images(self.shape[:2]), verbose=2)
            if not th is None:
                # Threshold predictions
                preds = preds > th
                # todo: da kann man sich noch was besseres einfallen lassen
        else:
            preds=None
            warnings.warn('Model not trained yet')
        return preds

    def predict(self, img:Images,  th=0.5,boundary_height=0.05,scale=True, chull=False):

        print('predicting area...')
        pa=self.predict_area(img).reshape((-1,)+self.shape[:2])
        print('predicting boundaries...')
        pb=self.predict_boundary(img).reshape((-1,)+self.shape[:2])
        print('labeling predictions...')
        pred=self.label(pa,pb, th,boundary_height)
        if chull:
            print('getting convex hull')
            chull=np.array([skimage.morphology.convex_hull_object(x) for x in pred])
            print('relabeling...')
            pred=self.label(chull,pb, th,boundary_height)



        if not scale:
            pred=[np.expand_dims(x,axis=2) for x in pred]
            return pred
        # else:
        pred_scaled=[]
        for i, out_shape in img.features[['size_y', 'size_x']].iterrows():
            pred_scaled.append(cv2.resize(pred[i], tuple(out_shape) ,interpolation=cv2.INTER_NEAREST))
            # nearest to avoid averaging between labels
        pred_scaled=[np.expand_dims(x,axis=2) for x in pred_scaled]
        pred_scaled=[skimage.segmentation.relabel_sequential(x)[0] for x in pred_scaled]
        #small lables might have been lost
        return(pred_scaled)






    def label(self, pa, pb,th=0.5, boundary_height=0.05):

        lab_pred=np.zeros_like(pa, dtype=np.uint)
        for i in tqdm.tqdm(range(pa.shape[0])):
            #lab_pred.append(skimage.morphology.label(unl_pred[i] > th))

            #get at least one pixel in each nuclei that is not connected to another
            #starts=skimage.feature.peak_local_max(-pb[i], indices=False, footprint=np.ones((3, 3)), labels=scipy.ndimage.label(pa[i]>th)[0])
            starts=skimage.morphology.h_maxima((1-pb[i]), h=boundary_height)
            # h=.01: a closed seperation of at least 1% boundary probability required
            starts[pa[i]<th]=0 #no peaks outside mask

            #label these starts
            starts = scipy.ndimage.label(starts)[0]
            lab_pred[i] = skimage.morphology.watershed(pb[i], starts, mask=pa[i]>th)
            #lab_pred[i][pa[i]<th]=0
            lab_pred[i]=skimage.segmentation.relabel_sequential(lab_pred[i])[0]
            #lab_pred[i]=skimage.morphology.remove_small_objects(lab_pred[i].astype(int),min_size=4)


        return lab_pred
   

def iou_loss(y_true,y_pred):
    #stolen from 
    #http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
    logits=tf.reshape(y_pred, [-1])
    trn_labels=tf.reshape(y_true, [-1])
    '''
    Eq. (1) The intersection part - tf.mul is element-wise, 
    if logits were also binary then tf.reduce_sum would be like a bitcount here.
    '''
    inter=tf.reduce_sum(tf.multiply (logits,trn_labels))
    
    '''
    Eq. (2) The union part - element-wise sum and multiplication, then vector sum
    '''
    union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)))
    
    # Eq. (4)
    loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.div(inter,union))
    return loss

    



if __name__=='__main__':

    train=Images()
    #train.ids=train.ids[:10]
    train.features=train.features[:20]
    print("reading training images")
    train.load_images()
    print("reading training masks")
    train.load_masks()
    model=ModelUNet(name='unet_v1_128x128.h5')

    train.pred=model.predict_unlabeld(train, th=None)
    train.show_image()
