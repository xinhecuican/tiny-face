#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

from IPython import get_ipython

import evaluate

from metrics import *
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error as mse
import glob
import os
import cv2
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import imp
import time
import random
import dlib
from imgaug import augmenters as iaa
import pdb
#imp.reload(tiny)
import metrics
imp.reload(metrics)
imp.reload(evaluate)


# In[2]:


weights_path = 'data/mat2tf/mat2tf.pkl'
data_dir = 'data/widerface/WIDER_val/images/28--Sports_Fan/'
out_dir = './Tiny_Faces_in_Tensorflow/out_test/'


# In[3]:


data_folder = 'data/widerface/WIDER_val/images/'


# In[4]:


frame = glob.glob(data_dir+'*')[15]
original_image = cv2.imread(frame)[:,:,::-1]


# ### Annotations

# In[5]:


with open('data/widerface/wider_face_split/wider_face_val_bbx_gt.txt') as f:
    annotation_file = [k.strip() for k in f.readlines()]


# In[6]:


# open annotations
image_name = [k for k in annotation_file if '--' in k]
original_annot = {} 
# original_annot[name] = [(x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose)]
for name in image_name:
    n_index = annotation_file.index(name)
    n_pictures = int(annotation_file[n_index+1])
    original_annot[name] = annotation_file[n_index+2:n_index+2+n_pictures]


# In[7]:


resized_annot = {}
dico_img = {}

# resizing ground truth and images
for r in [1, 2, 4, 8]:
    resized_annot[r] = {}
    for name, value in original_annot.items():
        temp = []
        for l in value:
            (x1, y1, w, h, _, _, _, _, _, _) = map(int, l.split())
            temp.append(list(map(lambda x:int(x/r), (x1, y1, w, h))))
        resized_annot[r][name] = temp
        
    resized_shape = tuple(map(lambda x:int(x/r), original_image.shape[:2]))
    dico_img[r] = cv2.resize(original_image, resized_shape[::-1])


# In[8]:



predictions = {}
for rate, elem in dico_img.items():
    with tf.Graph().as_default():
        predictions[rate] = evaluate.evaluate(weight_file_path=weights_path, img=elem)


# ### Plotting bounding box for 1 image

# In[9]:


pic = frame.replace(data_dir, '')
img_truth = cv2.imread(pic)[:,:,:-1]


# In[10]:


f, ax = plt.subplots(figsize=(15,20), nrows=4)

# plot the ground truth bounding boxes
for i, (rate, values) in enumerate(resized_annot.items()):
    ax[i].imshow(dico_img[rate])
    ax[i].set_title('Image at scale %.2f' % (1/rate))
    # ground truth
    for k in values[get_pic_name(pic)]:
        (x1, y1, w, h) = k
        rect = patches.Rectangle((x1, y1), w, h,linewidth=2,edgecolor='g',facecolor='none')
        ax[i].add_patch(rect)
    # pred
    for k in predictions[rate]:
        x1, y1 = k[:2]
        w, h = k[2] - k[0], k[3] - k[1]
        rect = patches.Rectangle((x1,y1),w, h,linewidth=2,edgecolor='r',facecolor='none')
        ax[i].add_patch(rect)
    #ax.text(x1-5, y1-5, '%.2f' % jd, color='r')


# ## Metrics for all the class

# In[11]:


resized_annot = {}
dico_img = {}

# resizing ground truth and images
for r in np.arange(1,10.5,0.5):
    resized_annot[r] = {}
    dico_img[r] = {}
    for name, value in original_annot.items():
        if '28--Sports_Fan' in name:
            temp = []
            for l in value:
                (x1, y1, w, h, _, _, _, _, _, _) = map(int, l.split())
                temp.append(list(map(lambda x:int(x/r), (x1, y1, w, h))))
            resized_annot[r][name] = temp

            orig_img = cv2.imread(data_folder + name)
            resized_shape = tuple(map(lambda x:int(x/r), orig_img.shape[:2]))
            dico_img[r][name] = cv2.resize(orig_img, resized_shape[::-1])


# In[12]:


predictions = {}
list_imgs = {}
for rate, img_list in dico_img.items():
    list_imgs[rate] = []
    for name in glob.glob(data_dir + '*'):
        img = img_list[get_pic_name(name)]
        list_imgs[rate].append(img)


# In[13]:


imp.reload(evaluate)
tic = time.time()   
for r in dico_img :
    with tf.compat.v1.Graph().as_default():
        preds = evaluate.evaluate(weight_file_path=weights_path, list_imgs=list_imgs[r])
    predictions[r] = preds
toc = time.time()
print('It took %d sec' % (toc-tic))


# In[14]:


imp.reload(metrics)


# In[15]:


df_tot = pd.DataFrame()
for rate in dico_img.keys():
    a, df = metrics.compute_stats(data_dir, resized_annot[rate], predictions[rate])
    df['Rate'] = rate
    df_tot = pd.concat([df_tot, df])


# In[16]:


f, ax = plt.subplots(figsize=(8,5))

df_tot = df_tot[df_tot.index != 28]
x = list(df_tot.Rate.unique())
y_bbox = []
y_jaccard = []
y_tb = []
for r in x:
    y_bbox.append(df_tot[df_tot.Rate == r].Nb_Pred_Bboxes.sum())
    y_jaccard.append(df_tot[df_tot.Rate == r].mJaccard.mean())
    y_tb.append(df_tot[df_tot.Rate == r].Nb_Truth_Bboxes.sum())

ax.plot(x,y_bbox, 'r', label='Predicted')
ax.plot(x,y_tb, 'r+', label='Ground Truth')
ax.tick_params('y', colors='r')
ax.set_ylabel('Number of Bouding Box', color='r')
ax.legend()
ax.set_xlabel('Image Scale')
f.canvas.draw()
ax.set_xticklabels(['1/'+item.get_text() for item in ax.get_xticklabels()])
ax2 = ax.twinx()
ax2.plot(x, y_jaccard, 'b')
ax2.set_ylim([0,1])
ax2.tick_params('y', colors='b')
_ = ax2.set_ylabel('Mean Jaccard similarity', color='b')


# In[19]:


idx = 20
rates = [1, 2, 4, 6]
i = 0
f_img = glob.glob(data_dir + '*')[idx]
img = cv2.imread(f_img)[:,:,::-1]
f,ax = plt.subplots(nrows=2, ncols=2,  figsize=(15,11), gridspec_kw={'wspace':0.1, 'hspace':0})


for (r, img) in dico_img.items():
    if r in rates:
        ax[i//2, i%2].imshow(img[get_pic_name(f_img)][:,:,::-1])
        ax[i//2, i%2].set_title('Scale 1/%d, %d predictions' % (r, len(predictions[r][idx])))
        for k in predictions[r][idx]:
            x1, y1 = k[:2]
            w, h = k[2] - k[0], k[3] - k[1]
            rect = patches.Rectangle((x1,y1),w, h,linewidth=2,edgecolor='g',facecolor='none')
            ax[i//2, i%2].add_patch(rect)
        i += 1


# In[ ]:




