import numpy as np
import pandas as pd
import skimage
import Images
import ModelUNet_rep
train=Images.Images()
#subselect, to make it faster

train.features=train.features[:10]

print("reading training images")
train.load_images()
print("reading training masks")
train.read_masks()
train.get_images()
#testmodel=ModelUNet_rep.ModelUNet(m_file='unet_model1.h5',shape=(128,128,3))
#testmodel.fit_model(train)
model=ModelUNet_rep.ModelUNet(m_file='model-dsbowl2018-1.h5')

scaled_pred=model.predict_unlabeld(train)
unlab_pred=train.rescale(scaled_pred, scale=None, dtype=np.float32, mode='reflect')
#probability vector

train.add_pred(model.label(unlab_pred, th=0.5))
# this adds iou scores to train.features

# now start look at correlation of other features to score


train.show_image()
# todo: color by IOU
# todo: show set diffs of mask and prediction

train.write_submission(file_name='testsubmission.csv')

##########
# scores #
##########

#unlabeled IOU
print(Images.iou(train.masks[0] > 0, unlab_pred[0]>0.5))
#labled IOUs per nucleus
print(Images.iou(train.masks[0], train.pred[0]))
#mean IOU
print(np.mean(Images.iou(train.masks[0], train.pred[0])))
# fraction of nuclei > th
print(Images.iou_score(train.masks[0], train.pred[0], th=[.5]))
print(Images.iou_score(train.masks[0], train.pred[0], th=[.95]))
# fraction of nuclei > th average over range of thresholds
print(Images.iou_score(train.masks[0], train.pred[0]))


#unlabeled IOU

train.show_image(0)


import numpy as np
import scipy.misc

img_fn="../input/stage1_train/08151b19806eebd58e5acec7e138dbfbb1761f41a1ab9620466584ecc7d5fada/images/08151b19806eebd58e5acec7e138dbfbb1761f41a1ab9620466584ecc7d5fada.png"
mask_fn="../input/stage1_train/08151b19806eebd58e5acec7e138dbfbb1761f41a1ab9620466584ecc7d5fada/masks/01f50a90213f833b490c9dfe889f66f0917d3481baf1db8e581b442ad8e4d3cc.png"
img = scipy.misc.imread(img_fn, mode='RGB') #[:,:,:self.channels]
mask=scipy.misc.imread(mask_fn, flatten=True)


print(img.shape)


