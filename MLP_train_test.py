#!/usr/bin/env python
# coding: utf-8

# # Train MLP
# 

# In[1]:
import sys
#sys.path.append(r'F:\神经网络的鲁棒性探究\Model_package')
print(sys.path)  
sys.path.append(r'D:\Robust_SimCLR\Model_package')
sys.path.append(r'D:\Robust_SimCLR\Models')

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
import matplotlib.pyplot as plt

# In[3]:
mlp_classifier = Classifier(100).cpu()
torch.save(mlp_classifier,"./Models/classifier.pt")


# In[4]:


mlp = MLP(28*28, "./Models/classifier.pt").cpu()



# In[5]:
print(mlp)
# In[6]:
batchSize=256
# In[7]:
EPOCHES=200
# In[8]:


trainLoader,testLoader=GetDataLoader(*GetMnistData(),batchSize)  #数据的值在0~1的范围内。
imgSize=trainLoader.dataset[0][0].shape[1:]


# In[9]:


mlp_train_losses, mlp_test_losses, mlp_test_accuracy = TrainMLP(mlp, trainLoader, testLoader, EPOCHES,device=torch.device("cpu"))


# In[10]:


plt.figure()
plt.plot(range(EPOCHES), mlp_test_accuracy)
plt.xlabel('epoches')
plt.ylabel('test accuracy')
plt.show()


# In[11]:


plt.figure()
plt.plot(mlp_train_losses)
plt.xlabel('epoches')
plt.ylabel('train loss')
plt.show()


# In[12]:



# In[13]:


MLP_extract_output = nn.Sequential(*list(mlp.children())[0])


# In[14]:


layer_num = int(len(list(MLP_extract_output.children()))/2)

print('layer_num:{}\r\n'.format(layer_num))
# In[15]:


torch.save(mlp,"./Models/mlp_layer"+str(layer_num)+".pt")


# In[16]:

int(len(list(MLP_extract_output.children()))/2)



