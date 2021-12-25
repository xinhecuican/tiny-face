#!/usr/bin/env python
# coding: utf-8

# Be careful about the virtual environments. Indeed, there are different algorithms, in this notebook, in the different sections and we different dependencies.   
# You many not be able to run the full notebook with the same virtual env !

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches
# get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import imp
from metrics import *
import cv2
import time
import glob

# In[2]:


data_folder = 'data/widerface/WIDER_val/images/'

# ## Loading Ground Truth

# In[3]:


with open('data/widerface/wider_face_split/wider_face_val_bbx_gt.txt') as f:
    annotation_file = [k.strip() for k in f.readlines()]

image_name = [k for k in annotation_file if '--' in k]
d = {}
# d[name] = [(x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose)]
for name in image_name:
    n_index = annotation_file.index(name)
    n_pictures = int(annotation_file[n_index + 1])
    d[name] = annotation_file[n_index + 2:n_index + 2 + n_pictures]

# ## MXNet Faster R-CNN
# Using MXNet (Apache scalable deep learning library) for face-related algorithm, **Faster-RCNN** and **ResNet-50** (not optimized, see https://arxiv.org/pdf/1706.01061.pdf for an improved Faster R-CNN for Faces)  
# Virtual Env : `source ~/Work/mxnet/bin/activate`  
# **!! Use MXNet Kernel !! **

# In[4]:

import os
sys.path.append('F:\\python project\\ExtendedTinyFaces\\mxnet_face\\detection')
sys.path.append('F:\\python project\\ExtendedTinyFaces\\mxnet_face\\detection\\symbol')
import detection as detecter

# #### Single Metric

# In[59]:


data_dir = glob.glob(data_folder + '*')[10] + '/'
img_path = glob.glob(data_dir + '*')[0]
img = cv2.imread(img_path)
_, scale = detecter.resize(img.copy(), 600, 1000)
_, dets = detecter.main(img_path)
for i in range(dets.shape[0]):
    bbox = dets[i, :4]
    cv2.rectangle(img, (int(round(bbox[0] / scale)), int(round(bbox[1] / scale))),
                  (int(round(bbox[2] / scale)), int(round(bbox[3] / scale))), (0, 0, 255), 2)

# In[60]:


bb = [(int(round(dets[i, 0] / scale)), int(round(dets[i, 1] / scale)),
       int(round(dets[i, 2] / scale)), int(round(dets[i, 3] / scale))) for i in range(dets.shape[0])]

# In[61]:


f, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img[:, :, ::-1])
for k in d[img_path.replace(data_folder, '')]:
    _, jd = find_best_bbox(k, bb)
    (x1, y1, w, h, _, _, _, _, _, _) = map(int, k.split())
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1 - 5, y1 - 5, '%.2f' % jd, color='r')

# #### One folder metrics

# In[5]:


var = glob.glob(data_folder + '*')[40] + '/'

# ##### Parade

# In[65]:


data_dir = glob.glob(data_folder + '*')[40] + '/'
b = []
tic = time.time()
for img_path in glob.glob(data_dir + '*'):
    img = cv2.imread(img_path)
    _, scale = detecter.resize(img.copy(), 600, 1000)
    _, dets = detecter.main(img_path)
    bb = [(int(round(dets[i, 0] / scale)), int(round(dets[i, 1] / scale)),
           int(round(dets[i, 2] / scale)), int(round(dets[i, 3] / scale))) for i in range(dets.shape[0])]
    b.append(bb)
toc = time.time()
print('It took %d sec' % (toc - tic))

# In[66]:


a, df = compute_stats(data_dir, d, b)
# df.mJaccard = df.mJaccard.fillna(0)


# In[92]:


df.head()

# In[67]:


df.loc[92]

# In[7]:


print('%d/%d (%.2f) bounding boxes found over all images of the folder\nMean Jaccard : %.2f' % (df.Nb_Pred_Bboxes.sum(),
                                                                                                df.Nb_Truth_Bboxes.sum(),
                                                                                                df.Nb_Pred_Bboxes.sum() / df.Nb_Truth_Bboxes.sum(),
                                                                                                df.mJaccard.mean()))

# In[8]:


idx = 92
img = cv2.imread(glob.glob(data_dir + '*')[idx])
for i in range(len(b[idx])):
    bbox = b[idx][i]
    cv2.rectangle(img, (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]), (0, 0, 255), 2)
f, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img[:, :, ::-1])

# #### Dresses

# In[12]:


ddd = []
for i, repo in enumerate(glob.glob(data_folder + '*')):
    rep = repo.replace(data_folder, '')
    ddd.append(
        [i, rep, np.mean([len(v) for k, v in d.items() if rep in k]), len([v for k, v in d.items() if rep in k])])
# sorted(ddd, key=lambda x:x[2], reverse=False)


# In[32]:


data_dir = glob.glob(data_folder + '*')[38] + '/'
b = []
tic = time.time()
for img_path in glob.glob(data_dir + '*'):
    img = cv2.imread(img_path)
    _, scale = detecter.resize(img.copy(), 600, 1000)
    _, dets = detecter.main(img_path)
    bb = [(int(round(dets[i, 0] / scale)), int(round(dets[i, 1] / scale)),
           int(round(dets[i, 2] / scale)), int(round(dets[i, 3] / scale))) for i in range(dets.shape[0])]
    b.append(bb)
toc = time.time()
print('It took %d sec' % (toc - tic))

# In[33]:


a, df = compute_stats(data_dir, d, b)
print('%d/%d (%.2f) bounding boxes found over all images of the folder\nMean Jaccard : %.2f' % (df.Nb_Pred_Bboxes.sum(),
                                                                                                df.Nb_Truth_Bboxes.sum(),
                                                                                                df.Nb_Pred_Bboxes.sum() / df.Nb_Truth_Bboxes.sum(),
                                                                                                df.mJaccard.mean()))

# In[34]:


idx = 4
img = cv2.imread(glob.glob(data_dir + '*')[idx])
for i in range(len(b[idx])):
    bbox = b[idx][i]
    cv2.rectangle(img, (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]), (0, 0, 255), 2)
f, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img[:, :, ::-1])

# ## Tiny Faces
# **!! Use TensorFlow Kernel !! **

# In[4]:


import tensorflow as tf
# sys.path.append('./Tiny_Faces_in_Tensorflow/')
import evaluate

# import tiny_face_eval as tiny
weights_path = './Tiny_Faces_in_Tensorflow/hr_res101.pkl'

# In[13]:


imp.reload(tiny)
imp.reload(evaluate)

# In[6]:


data_dir = glob.glob(data_folder + '*')[40] + '/'
weights_path = './Tiny_Faces_in_Tensorflow/hr_res101.pkl'
tic = time.time()
with tf.Graph().as_default():
    b = evaluate.evaluate(weight_file_path=weights_path, data_dir=data_dir)
toc = time.time()
print('It took %d sec' % (toc - tic))

# In[7]:


a, df = compute_stats(data_dir, d, b)
print('%d/%d (%.2f) bounding boxes found over all images of the folder\nMean Jaccard : %.2f' % (df.Nb_Pred_Bboxes.sum(),
                                                                                                df.Nb_Truth_Bboxes.sum(),
                                                                                                df.Nb_Pred_Bboxes.sum() / df.Nb_Truth_Bboxes.sum(),
                                                                                                df.mJaccard.mean()))

# In[8]:


df.loc[92]

# In[43]:


df.head()

# In[9]:


idx = 92
img = cv2.imread(glob.glob(data_dir + '*')[idx])
for i in range(len(b[idx])):
    bbox = b[idx][i]
    cv2.rectangle(img, (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]), (0, 0, 255), 2)
f, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img[:, :, ::-1])

# ##### Dresses

# In[5]:


data_dir = glob.glob(data_folder + '*')[38] + '/'
weights_path = './Tiny_Faces_in_Tensorflow/hr_res101.pkl'
tic = time.time()
with tf.Graph().as_default():
    b = evaluate.evaluate(weight_file_path=weights_path, data_dir=data_dir)
toc = time.time()
print('It took %d sec' % (toc - tic))

# In[6]:


a, df = compute_stats(data_dir, d, b)
print('%d/%d (%.2f) bounding boxes found over all images of the folder\nMean Jaccard : %.2f' % (df.Nb_Pred_Bboxes.sum(),
                                                                                                df.Nb_Truth_Bboxes.sum(),
                                                                                                df.Nb_Pred_Bboxes.sum() / df.Nb_Truth_Bboxes.sum(),
                                                                                                df.mJaccard.mean()))

# ## HOG
# **!! Use TensorFlow Kernel !!** (or any kernel with dlib) 

# In[7]:


import dlib

# In[8]:


# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

# Loop over images for detection
b = []
tic = time.time()
for img_path in glob.glob(data_dir + '*'):
    img = cv2.imread(img_path)
    detected_faces = face_detector(img)
    b.append([(k.left(), k.top(), k.right(), k.bottom()) for k in detected_faces])
toc = time.time()
print('It took %.1f sec' % (toc - tic))

# In[9]:


a, df = compute_stats(data_dir, d, b)

# In[10]:


print('%d/%d (%.2f) bounding boxes found over all images of the folder\nMean Jaccard : %.2f' % (df.Nb_Pred_Bboxes.sum(),
                                                                                                df.Nb_Truth_Bboxes.sum(),
                                                                                                df.Nb_Pred_Bboxes.sum() / df.Nb_Truth_Bboxes.sum(),
                                                                                                df.mJaccard.mean()))

# In[39]:


idx = 92
img = cv2.imread(glob.glob(data_dir + '*')[idx])
for i in range(len(b[idx])):
    bbox = b[idx][i]
    cv2.rectangle(img, (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]), (0, 0, 255), 2)
f, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img[:, :, ::-1])

# ##### Dresses

# In[13]:


data_dir = glob.glob(data_folder + '*')[38] + '/'

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

# Loop over images for detection
b = []
tic = time.time()
for img_path in glob.glob(data_dir + '*'):
    img = cv2.imread(img_path)
    detected_faces = face_detector(img)
    b.append([(k.left(), k.top(), k.right(), k.bottom()) for k in detected_faces])
toc = time.time()
print('It took %.1f sec' % (toc - tic))

# In[14]:


a, df = compute_stats(data_dir, d, b)

# In[15]:


print('%d/%d (%.2f) bounding boxes found over all images of the folder\nMean Jaccard : %.2f' % (df.Nb_Pred_Bboxes.sum(),
                                                                                                df.Nb_Truth_Bboxes.sum(),
                                                                                                df.Nb_Pred_Bboxes.sum() / df.Nb_Truth_Bboxes.sum(),
                                                                                                df.mJaccard.mean()))

# ## Haar Cascades

# In[55]:


img = cv2.imread(glob.glob(data_dir + '*')[idx])
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# In[56]:


# load cascade classifier training file for haarcascade
haar_face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')

# In[57]:


faces = haar_face_cascade.detectMultiScale(gray_img)

# In[58]:


# load cascade classifier training file for haarcascade
haar_face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')

# Loop over images for detection
b = []
tic = time.time()
for img_path in glob.glob(data_dir + '*'):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_face_cascade.detectMultiScale(gray_img)
    b.append([(x, y, x + w, y + h) for (x, y, w, h) in faces])
toc = time.time()
print('It took %.1f sec' % (toc - tic))

# In[59]:


a, df = compute_stats(data_dir, d, b)

# In[62]:


df.loc[92]

# In[60]:


print('%d/%d (%.2f) bounding boxes found over all images of the folder\nMean Jaccard : %.2f' % (df.Nb_Pred_Bboxes.sum(),
                                                                                                df.Nb_Truth_Bboxes.sum(),
                                                                                                df.Nb_Pred_Bboxes.sum() / df.Nb_Truth_Bboxes.sum(),
                                                                                                df.mJaccard.mean()))

# In[61]:


idx = 92
img = cv2.imread(glob.glob(data_dir + '*')[idx])
for i in range(len(b[idx])):
    bbox = b[idx][i]
    cv2.rectangle(img, (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]), (0, 0, 255), 2)
f, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img[:, :, ::-1])

# ##### Dresses

# In[37]:


data_dir = glob.glob(data_folder + '*')[38] + '/'
# load cascade classifier training file for haarcascade
haar_face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')

# Loop over images for detection
b = []
tic = time.time()
for img_path in glob.glob(data_dir + '*'):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_face_cascade.detectMultiScale(gray_img)
    b.append([(x, y, x + w, y + h) for (x, y, w, h) in faces])
toc = time.time()
print('It took %.1f sec' % (toc - tic))

# In[38]:


a, df = compute_stats(data_dir, d, b)

# In[39]:


print('%d/%d (%.2f) bounding boxes found over all images of the folder\nMean Jaccard : %.2f' % (df.Nb_Pred_Bboxes.sum(),
                                                                                                df.Nb_Truth_Bboxes.sum(),
                                                                                                df.Nb_Pred_Bboxes.sum() / df.Nb_Truth_Bboxes.sum(),
                                                                                                df.mJaccard.mean()))

# ## MXNet - MTCNN
# [MXNET Implementation](https://github.com/pangyupo/mxnet_mtcnn_face_detection)  
# [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks](https://arxiv.org/abs/1604.02878)

# In[5]:


import sys

sys.path.append('./mxnet_mtcnn_face_detection/')
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

data_dir = glob.glob(data_folder + '*')[40] + '/'

# In[8]:


detector = MtcnnDetector(model_folder='./mxnet_mtcnn_face_detection/model', ctx=mx.cpu(0),
                         num_worker=4, accurate_landmark=False)

# In[24]:


b = []
tic = time.time()
for img_path in glob.glob(data_dir + '*'):
    img = cv2.imread(img_path)
    results = detector.detect_face(img)
    if results is not None:
        bb = [(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
              for b in results[0]]
    else:
        bb = []
    b.append(bb)
toc = time.time()
print('It took %d sec' % (toc - tic))

# In[43]:


a, df = compute_stats(data_dir, d, b)
# df.mJaccard = df.mJaccard.fillna(0)


# In[44]:


df.sort_values('Nb_Truth_Bboxes', ascending=False).head(10)

# In[45]:


print('%d/%d (%.2f) bounding boxes found over all images of the folder\nMean Jaccard : %.2f' % (df.Nb_Pred_Bboxes.sum(),
                                                                                                df.Nb_Truth_Bboxes.sum(),
                                                                                                df.Nb_Pred_Bboxes.sum() / df.Nb_Truth_Bboxes.sum(),
                                                                                                df.mJaccard.mean()))

# In[48]:


df.loc[92]

# In[46]:


idx = 92
img = cv2.imread(glob.glob(data_dir + '*')[idx])
for i in range(len(b[idx])):
    bbox = b[idx][i]
    cv2.rectangle(img, (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]), (0, 0, 255), 2)
f, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img[:, :, ::-1])

# ##### Dresses

# In[9]:


data_dir = glob.glob(data_folder + '*')[38] + '/'
b = []
tic = time.time()
for img_path in glob.glob(data_dir + '*'):
    img = cv2.imread(img_path)
    results = detector.detect_face(img)
    if results is not None:
        bb = [(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
              for b in results[0]]
    else:
        bb = []
    b.append(bb)
toc = time.time()
print('It took %d sec' % (toc - tic))

# In[10]:


a, df = compute_stats(data_dir, d, b)
print('%d/%d (%.2f) bounding boxes found over all images of the folder\nMean Jaccard : %.2f' % (df.Nb_Pred_Bboxes.sum(),
                                                                                                df.Nb_Truth_Bboxes.sum(),
                                                                                                df.Nb_Pred_Bboxes.sum() / df.Nb_Truth_Bboxes.sum(),
                                                                                                df.mJaccard.mean()))

# ## Extensions

# See [Face R-CNN](https://arxiv.org/pdf/1706.01061.pdf) for an improved Faster R-CNN for Faces published recently that can probably outperform with Tiny Faces :   
# Results comparison [here](http://vis-www.cs.umass.edu/fddb/results.html)

# In[ ]:
