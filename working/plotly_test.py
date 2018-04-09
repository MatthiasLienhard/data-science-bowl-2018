
# coding: utf-8

# In[49]:


import plotly_test.plotly as py
from plotly_test.offline import download_plotlyjs, init_notebook_mode, plot,iplot
import plotly_test.tools
import plotly_test.graph_objs as go
import numpy as np
import pandas as pd
import skimage
import Images
import ModelUNet_rep
import ModelUNet_v2 #version 2 learns areas and boundaries of nuclei
import matplotlib.pyplot as plt

import copy


# In[2]:


train=Images.Images("../input/stage1_train")

#set aside 10% for validation
val=train.subset(np.arange(train.n()*.9, train.n()))
train=train.subset(np.arange(train.n()*.9,))

model2=ModelUNet_v2.ModelUNet(name='unet_256x256_v2')
model1=ModelUNet_rep.ModelUNet(name='unet_v1_256x256')



# In[3]:


val.load_images()
val.load_masks()
val1=copy.deepcopy(val)
val.add_predictions(model2)
val.features.drop(['ids'], axis=1).head()
print("expected LB score of v2 (val): {}".format(np.mean(val.features['iou_score'])))
#somehow its mutch worse than that...
val1.add_predictions(model1)
print("expected LB score of v1 (val): {}".format(np.mean(val1.features['iou_score'])))


# In[45]:


df=val.features
df['n']=df.index
df['description']=pd.Series("IOU: " + (df.iou_score).astype(str)+"<br>ID: " + (df.n).astype(str))
df_bw=df[(df.n_channels==1)]
df_col=df[(df.n_channels==3)]
df.head()


# In[46]:


trace_bw=go.Scatter(x=df_bw.nuclei_n, y=df_bw.n_pred, mode='markers', marker=dict(size=12, line=dict(width=1), color="black"),
                    name="grayscale", text=df_bw.description,)
trace_col=go.Scatter(x=df_col.nuclei_n, y=df_col.n_pred, mode='markers', marker=dict(size=12, line=dict(width=1), color="red"),
                    name="color", text=df_col.description,)
data=[trace_bw, trace_col]
layout=go.Layout(title="number of nuclei / image", hovermode="closest",
                xaxis=dict(title="# truth", ticklen=5, zeroline=False, gridwidth=2),
                yaxis=dict(title="# prediction", ticklen=5,  gridwidth=2),)
fig=go.Figure(data=data, layout=layout)
plot(fig, show_link=False)


# In[48]:


trace_bw=go.Scatter(x=df_bw.nuclei_n, y=df_bw.iou_score, mode='markers', marker=dict(size=12, line=dict(width=1), color="black"),
                    name="grayscale", text=df_bw.description,)
trace_col=go.Scatter(x=df_col.nuclei_n, y=df_col.iou_score, mode='markers', marker=dict(size=12, line=dict(width=1), color="red"),
                    name="color", text=df_col.description,)
data=[trace_bw, trace_col]
layout=go.Layout(title="number of nuclei / image", hovermode="closest",
                xaxis=dict(title="# truth", ticklen=5, zeroline=False, gridwidth=2),
                yaxis=dict(title="# prediction", ticklen=5,  gridwidth=2),)
fig=go.Figure(data=data, layout=layout)
plot(fig, show_link=False)
#1) the more nuclei the worse
#2) color images are really bad


# In[43]:


fig=tools.make_subplots(rows=1, cols=3, subplot_titles=('masked image','prediction','difference'))


# In[53]:


id=0

data=[dict(type='heatmap', z=val.masks[0],)]
axis=dict(zeroline=False, showgrid=False, ticklen=4)
layout=dict(width=val.features['x'], height=600,
            font=dict(family='Balto', size=12),
            xaxis= dict(axis),
            yaxis= dict(axis),
            title= "mask"
            )
fig1=dict(data=data, layout=layout)
plot(fig1)


# In[52]:




