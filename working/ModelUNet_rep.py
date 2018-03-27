
from Images import Images
import os
import warnings
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import skimage.morphology
import tensorflow as tf


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
        return model
        #self.model.summary()


    def __init__(self, shape=(128,128,3), m_file="model_unet_rep.h5"):
        if os.path.isfile(m_file):
            print("found model file")
            self.trained=True
            self.model=load_model(m_file, custom_objects={'mean_iou': ModelUNet.mean_iou})
            self.shape=self.model.input_shape[1:4]
        else:
            # make new model
            self.shape=shape
            self.trained=False
            
        self.m_file=m_file
        self.fit_history=None

    def fit_model(self, train:Images):
        self.model=ModelUNet.init_model(self.shape)
        earlystopper = EarlyStopping(patience=5, verbose=1)
        checkpointer = ModelCheckpoint(self.m_file, verbose=1, save_best_only=True)
        self.fit_history = self.model.fit(train.get_images(self.shape[:2]),
                    train.get_masks(self.shape[:2],labeled=False),
                    validation_split=0.1, batch_size=16, epochs=50,
                    callbacks=[earlystopper, checkpointer])
        self.trained=True

    def predict_unlabeld(self, img:Images, th=None):
        if self.trained:
            self.model = load_model(self.m_file, custom_objects={'mean_iou': ModelUNet.mean_iou})
            preds = self.model.predict(img.get_images(self.shape[:2]), verbose=2)
            if not th is None:
                # Threshold predictions
                preds = (preds > th).astype(np.bool)
                # todo: da kann man sich noch was besseres einfallen lassen
        else:
            preds=None
            warnings.warn('Model not trained yet')
        return preds

    #def predict_labeled(self, img:Images):
    #    unl_pred=self.predict_unlabeld(img, th=.5)
    #    return self.label(unl_pred)

    def label(self, unl_pred, th=0.5):
        # todo: a lot of room for improvements!!!
        lab_pred=[]#np.zeros(unl_pred.shape, dtype=np.uint)
        for i in range(len(unl_pred)):
            lab_pred.append(skimage.morphology.label(unl_pred[i] > th))
        return lab_pred



if __name__=='__main__':

    train=Images()
    train.ids=train.ids[:10]
    train.features=train.features[:10]
    print("reading training images")
    train.read_images()
    print("reading training masks")
    train.read_masks()
    model=ModelUNet('test.h5')
    train.pred=model.predict(train, th=None)
    train.show_image()
