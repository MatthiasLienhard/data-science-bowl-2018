#this is still test mode

import numpy as np
import pandas as pd
import Images
import ModelUNet_rep
train=Images.Images()
train.get_ids()
#subselect, to make it faster
train.ids=train.ids[:10]
train.features=train.features[:10]

print("reading training images")
train.read_images()
print("reading training masks")
train.read_masks()
#training not tested yet!!!
model=ModelUNet_rep.ModelUNet('model-dsbowl2018-1.h5')

train.pred=model.predict_unlabeld(train)
train.labeled_pred=model.label(train.pred > 0.5)

print(Images.iou(train.masks[0], train.pred[0] > 0.5 ))

print(Images.iou(train.labeled_masks[0], train.labeled_pred[0]))

print(Images.iou_score(train.labeled_masks[0], train.labeled_pred[0]))

train.show_image(0)

