
import numpy as np
import pandas as pd
import skimage.filters

import Images
import ModelUNet_v2 #version 2 learns areas and boundaries of nuclei
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage.feature
import skimage.morphology
import cv2

import copy
model_shape=(256,256,3) #works with shape (2^x, 2^x) only?
model_name='unet_{}x{}_v2'.format(model_shape[0],model_shape[1])

train=Images.Images("../input/stage1_train")
train=train.subset(idx=range(20))
train.load_images()
train.load_masks()
train.features.head()

model=ModelUNet_v2.ModelUNet(name=model_name, shape=model_shape)

train.add_predictions(model)
print("expected LB score(train): {}".format(np.mean(train.features['iou_score'])))



print('predicting area...')
area_pred_scaled=model.predict_area(train)
print('predicting boundaries...')
boundary_pred_scaled=model.predict_boundary(train)
print('labeling predictions...')
area_pred_scaled.shape=area_pred_scaled.shape[:3]
boundary_pred_scaled.shape=boundary_pred_scaled.shape[:3]

labels_pred_scaled=model.label(pa=area_pred_scaled,pb=boundary_pred_scaled, th=0.5)

print('rescale labels vector...')
labels_pred=train.rescale(labels_pred_scaled, scale=None, mode='reflect')
#hoffentlich macht das nicht die labels kaputt
labels_pred=[np.expand_dims(x,axis=2) for x in labels_pred]

train.add_pred(labels_pred)
# this adds iou scores to train.features
print( train.features.drop(['ids'], axis=1).head() )
print("expected LB score(train): {}".format(np.mean(train.features['iou_score'])))

###
test=Images.Images("../input/stage1_test")
test=test.subset(idx=range(8,12))

#load the image files (in original size)
test.load_images()

pa=model.predict_area(test)
pb=model.predict_boundary(test)
lab=model.predict(test, scale=False)
#lab[2][lab[2]==1]=np.max(lab[2])+1
plt.imshow(lab[2].reshape(lab[2].shape[:2]))
plt.show()

lab2=skimage.morphology.remove_small_objects(lab[2].astype(int), min_size=6)
plt.imshow(lab2.reshape(lab2.shape[:2]))
plt.show()

plt.imshow(pa[2].reshape(pa[2].shape[:2])>0.5)
plt.show()


#image no 1 looks like a good example
img=train.get_images(scale=model_shape[:2],idx=[1])[0]
ma=train.get_masks(scale=model_shape[:2],idx=[1])[0]
mb=train.get_mask_boundaries(scale=model_shape[:2],idx=[1])[0]
#predicted
pa=np.reshape(area_pred_scaled[1], model_shape[:2])
pb=np.reshape(boundary_pred_scaled[1], model_shape[:2])
plt.imshow(pa, cmap='gray')
plt.show()
plt.imshow(pb, cmap='gray')
plt.show()


hills=(skimage.filters.gaussian(pb)+(1-skimage.filters.gaussian(pa)))*128


plt.imshow(hills, cmap='gray')
plt.show()
#probability to binary!!


####
# # http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
dist=scipy.ndimage.distance_transform_edt(pa>.5)

local_maxi=skimage.feature.peak_local_max(dist, indices=False, footprint=np.ones((3, 3)), labels=pa)


markers = scipy.ndimage.label(local_maxi)[0]

plt.imshow(markers)
plt.show()
labels = skimage.morphology.watershed(-dist, markers, mask=pa)
plt.imshow(labels)
plt.show()
##problem: often multiple lables at boundaries

#same approach using predicted boundaries

local_maxi=skimage.feature.peak_local_max(-pb, indices=False, footprint=np.ones((3, 3)), labels=pa>.5)

markers = scipy.ndimage.label(local_maxi)[0]
labels = skimage.morphology.watershed(pb, markers, mask=pa>.5)

plt.imshow(labels)
plt.show()
 #looks much better


