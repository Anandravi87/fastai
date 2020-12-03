#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *
from fastai.vision.widgets import *


# In[2]:


path = Path()
learn_inf = load_learner(path/'export.pkl', cpu=True)


# In[3]:


btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()


# In[4]:


def on_data_change(change):
    lbl_pred.value = ''
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'


# In[5]:


btn_upload.observe(on_data_change, names=['data'])


# In[6]:


display(VBox([widgets.Label('Select your musical instrument (Saxaphone, guitar, drums)'), btn_upload, out_pl, lbl_pred]))

