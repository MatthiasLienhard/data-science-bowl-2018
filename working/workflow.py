#this is still test mode

import numpy as np
import pandas as pd
import Images
import ModelUNet_rep
train=Images.Images()
#subselect, to make it faster
train.ids=train.ids[:10]
train.features=train.features[:10]

print("reading training images")
train.load_images()
print("reading training masks")
train.read_masks()
testmodel=ModelUNet_rep.ModelUNet(m_file='unet_model1.h5')
testmodel.fit_model(train)
#training not tested yet!!!
model=ModelUNet_rep.ModelUNet(m_file='model-dsbowl2018-1.h5')

train.pred=model.predict_unlabeld(train)
train.labeled_pred=model.label(train.pred > 0.5)

print(Images.iou(train.masks[0], train.pred[0] > 0.5 ))

print(Images.iou(train.labeled_masks[0], train.labeled_pred[0]))

print(Images.iou_score(train.labeled_masks[0], train.labeled_pred[0]))

train.show_image(0)


import numpy as np
import scipy.misc

img_fn="../input/stage1_train/08151b19806eebd58e5acec7e138dbfbb1761f41a1ab9620466584ecc7d5fada/images/08151b19806eebd58e5acec7e138dbfbb1761f41a1ab9620466584ecc7d5fada.png"
mask_fn="../input/stage1_train/08151b19806eebd58e5acec7e138dbfbb1761f41a1ab9620466584ecc7d5fada/masks/01f50a90213f833b490c9dfe889f66f0917d3481baf1db8e581b442ad8e4d3cc.png"
img = scipy.misc.imread(img_fn, mode='RGB') #[:,:,:self.channels]
mask=scipy.misc.imread(mask_fn, flatten=True)


print(img.shape)


