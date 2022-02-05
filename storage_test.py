#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append(r'F:\神经网络的鲁棒性探究\Model_package')


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import numpy as np
from Model_package.MLP.MLP import*
from Model_package.FGSM.fgsm import get_fgsm_samples_data
from Model_package.Classifier.Classifier import Classifier
from Model_package.SimCLR.ContrastiveLoss import ContrastiveLoss
from Model_package.SimCLR.SimCLR import SimCLR
from Model_package.FullModel.FullModel import FullMLP
from Model_package import Utilites
from Model_package.Utilites import TestAccuracy
from GetData import GetDataLoader,GetMnistData
from Trainers.MLPTrainner import* 


# In[3]:


mlp_classifier = Classifier(100).cuda()
torch.save(mlp_classifier,"./Models/classifier.pt")


# In[4]:


mlp = MLP(28*28, "./Models/classifier.pt").cuda()


# In[5]:


mlp


# In[7]:


batchSize=64
EPOCHES=20
trainLoader,testLoader=GetDataLoader(*GetMnistData(),batchSize)  #数据的值在0~1的范围内。
imgSize=trainLoader.dataset[0][0].shape[1:]


# In[8]:


mlp_train_losses, mlp_test_losses, mlp_test_accuracy = TrainMLP(mlp, trainLoader, testLoader, EPOCHES,device=torch.device("cuda"))


# In[ ]:




