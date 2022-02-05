#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from GetData import GetDataLoader,GetMnistData
from Model_package.MLP.MLP import MLP
from Model_package.FGSM.fgsm import get_fgsm_samples_data
from Model_package.FGSM import fgsm
import torch
import numpy as np
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt


#  # 根据层数要修改

# In[2]:


layer_num = 4                                                                           #根据层数要修改


# In[3]:


filenames={}
filenames['MLPModelPath']="./Models/mlp_layer"+str(layer_num)+".pt"
mlp = torch.load(filenames['MLPModelPath'])


# In[4]:


mlp


# # 根据层数要修改文件夹位置

# In[5]:


ad_train = np.load("./AdversarialSamples_layer"+str(layer_num)+"/adversarialDataTrian-25-model_layer"+str(layer_num)+".npy")
train = np.load("./AdversarialSamples_layer"+str(layer_num)+"/DataTrian-25-model_layer"+str(layer_num)+".npy")
train_target = np.load("./AdversarialSamples_layer"+str(layer_num)+"/TargetsTrian-25-model_layer"+str(layer_num)+".npy")


# In[6]:


ad_test = np.load("./AdversarialSamples_layer"+str(layer_num)+"/adversarialDataTest-25-model_layer"+str(layer_num)+".npy")
test = np.load("./AdversarialSamples_layer"+str(layer_num)+"/DataTest-25-model_layer"+str(layer_num)+".npy")
test_target = np.load("./AdversarialSamples_layer"+str(layer_num)+"/targetsTest-25-model_layer"+str(layer_num)+".npy")


# #  原数据的范围在0~255之间

# In[7]:


ad_train_tensor = torch.tensor(ad_train)
ad_test_tensor = torch.tensor(ad_test)     


# In[8]:


train_index = []
test_index = []
ad_train_index = []
ad_test_index = []

train_target_index = []
test_target_index = []


# In[9]:


for i in range(60000):
    if int(train_target[i]) != int(torch.argmax(mlp(ad_train_tensor[i]),dim=1).cpu().detach().numpy()):
        ad_train_index.append(ad_train[i])
        train_index.append(train[i])
        train_target_index.append(train_target[i])


# In[10]:


for i in range(10000):
    if int(test_target[i]) != int(torch.argmax(mlp(ad_test_tensor[i]),dim=1).cpu().detach().numpy()):
        ad_test_index.append(ad_test[i])
        test_index.append(test[i])
        test_target_index.append(test_target[i])


# In[11]:


ad_train_clear = np.array(ad_train_index)
train_clear = np.array(train_index)
train_target_clear = np.array(train_target_index)

ad_test_clear = np.array(ad_test_index)
test_clear = np.array(test_index)
test_target_clear = np.array(test_target_index)


#  # 根据层数要修改文件夹位置

# In[12]:


np.save("./AdversarialSamples_layer"+str(layer_num)+"/new_adversarialDataTrian-25-model_layer"+str(layer_num)+".npy", ad_train_clear)            #根据层数要修改文件夹位置
np.save("./AdversarialSamples_layer"+str(layer_num)+"/new_DataTrian-25-model_layer"+str(layer_num)+".npy", train_clear)
np.save("./AdversarialSamples_layer"+str(layer_num)+"/new_TargetsTrian-25-model_layer"+str(layer_num)+".npy", train_target_clear)
np.save("./AdversarialSamples_layer"+str(layer_num)+"/new_adversarialDataTest-25-model_layer"+str(layer_num)+".npy", ad_test_clear)
np.save("./AdversarialSamples_layer"+str(layer_num)+"/new_DataTest-25-model_layer"+str(layer_num)+".npy", test_clear)
np.save("./AdversarialSamples_layer"+str(layer_num)+"/new_TargetsTest-25-model_layer"+str(layer_num)+".npy", test_target_clear)


# In[13]:


test_clear.shape


# In[14]:


train_target_clear.shape


# In[ ]:




