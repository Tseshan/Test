#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import torch.nn as nn
from GetData import GetDataLoader,GetMnistData
from Model_package.FGSM import fgsm
from Model_package.FGSM.fgsm import get_fgsm_samples_data


# In[7]:


attack_epsilon = 0.25


# In[8]:


trainData,testData=GetMnistData()
#testdataLoader=torch.utils.data.DataLoader(testData,batch_size=len(testData),shuffle=False)


# In[9]:


traindataLoader=torch.utils.data.DataLoader(trainData,batch_size=len(trainData),shuffle=False)


# In[11]:


mlp = torch.load("./Models/mlp.pt")


# In[12]:


train_final_acc, train_adv_data = fgsm.test(mlp,"cuda",traindataLoader, attack_epsilon)
test_final_acc, test_adv_data = fgsm.test(mlp,"cuda",testdataLoader, attack_epsilon)


# In[ ]:




