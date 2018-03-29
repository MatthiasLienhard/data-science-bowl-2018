
# coding: utf-8

# # General workflow
# This workflow is a starting point to demonstrate the functionality of the containers and to detect and classify difficult cases

# In[1]:



import numpy as np
import pandas as pd
import skimage
import Images
import ModelUNet_rep


# In[2]:


#some definitions
debug=False #perform on a subset --> faster
model_shape=(256,256,3) #works with shape (2^x, 2^x) only?
model_name='unet_boundaries_{}x{}'.format(model_shape[0],model_shape[1])
if debug:
    model_name += '_debug'

print('the model will be called \''+model_name+'\'')


# ## Training
# * load training data
# * train the model (if not yet done)
# * 

# In[ ]:


train=Images.Images("../input/stage1_train")
if debug:
    #subselect, to make it faster 
    train=train.subset(idx=range(20))
    #todo: overload [] to get this

#we want to learn the boundaries
masks=train.masks
train.masks=train.mask_boundaries

#set aside 10% for validation
val=train.subset(np.arange(train.n()*.9, train.n()))
train=train.subset(np.arange(train.n()*.9,))

#load the image files (in original size)    
train.load_images()
train.load_masks()
train.features.head()    



# In[ ]:


# initialize and train the model 
# to detect unlabled masks (e.g. the forground vs background)


m_file=model_name+'.h5'

model=ModelUNet_rep.ModelUNet(m_file=m_file, shape=(256,256,3))
#question: should the model design be adapted according to image dimensions?
if not model.trained:
    model.fit_model(train)


# In[ ]:



#unlabled prediction (probability of pixle belonging to forground) 
#scaled to the dimensions of the model (e.g. 256x256)
print('making predictions...')
scaled_pred=model.predict_unlabeld(train)


print('rescale probability vector...')
# rescaling performs anti-aliasing, which can disturb lable masks
# it cannot be swhiched of for my version of skimage 0.13.1, only with dev0.14
print("skimage version: {}".format(skimage.__version__))
# for the probability vector antialiasing should be rather beneficial
unlab_pred=train.rescale(scaled_pred, scale=None, dtype=np.float32, mode='reflect')

print('labeling predinctions...')
pred=model.label(unlab_pred, th=0.5) #this function should also use the images
#add labled predictions to container
train.add_pred(pred)
# this adds iou scores to train.features
print( train.features.drop(['ids'], axis=1).head() ) 
print("expected LB score(train): {}".format(np.mean(train.features['iou_score'])))

train.show_image()


# In[ ]:


#same on validation data
val.load_images()
val.load_masks()
print('making predictions...')
scaled_pred=model.predict_unlabeld(val)
print('rescale probability vector...')
unlab_pred=val.rescale(scaled_pred, scale=None, dtype=np.float32, mode='reflect')
print('labeling predinctions...')
pred=model.label(unlab_pred, th=0.5) 
val.add_pred(pred)
val.features.drop(['ids'], axis=1).head()
print("expected LB score(val): {}".format(np.mean(val.features['iou_score'])))
val.show_image()


# ## Prediction (on test data)
# * load test data
# * use model to predict masks
# * prepare submission file

# In[ ]:


test=Images.Images("../input/stage1_test")

#load the image files (in original size)    
test.load_images()

test.features.head()    


# In[ ]:


print('making predictions...')
scaled_pred=model.predict_unlabeld(test)


print('rescale probability vector...')
unlab_pred=test.rescale(scaled_pred, scale=None, dtype=np.float32, mode='reflect')

print('labeling predinctions...')
pred=model.label(unlab_pred, th=0.5)
test.add_pred(pred)


# In[ ]:


submission_file='submission' + model_name + '.csv'
test.write_submission(submission_file)



# In[ ]:



##########
# scores #
##########
print('Scores for first test image:')
#unlabeled IOU
print('unlabeled IoU: {}'.format(Images.iou(val.masks[0] > 0, unlab_pred[0]>0.5)))
#labled IOUs per nucleus
print('unlabeled IoU per nucleus: {}'.format(Images.iou(val.masks[0], val.pred[0])))
#mean IOU
print('mean IoU: {}'.format(np.mean(Images.iou(val.masks[0], val.pred[0]))))
# fraction of nuclei > th
print('fraction of nuclei has IoU > 0.5: {}'.format(Images.iou_score(val.masks[0], val.pred[0], th=[.5])))
print('fraction of nuclei has IoU > 0.95: {}'.format(Images.iou_score(val.masks[0], val.pred[0], th=[.95])))
# fraction of nuclei > th average over range of thresholds
print('IoU score (over range of thresholds): {}'.format(Images.iou_score(val.masks[0], val.pred[0])))

val.show_image(0)

