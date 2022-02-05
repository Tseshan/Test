#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


layer_number = 5


# In[3]:


AdTest_SimCLR_files = os.listdir(".\AdTest_SimCLR_Layers"+str(layer_number)+"_output")


# In[4]:


#adtest_SimCLR_layer1 = np.load("./AdTest_SimCLR_Layers_output/H_Layer1.npy")
#adtest_SimCLR_layer2 = np.load("./AdTest_SimCLR_Layers_output/H_Layer2.npy")
#adtest_SimCLR_layer3 = np.load("./AdTest_SimCLR_Layers_output/H_Layer3.npy")


# In[5]:


#adtest_MLP_layer1 = np.load("./AdTest_MLP_Layers_output/H_Layer1.npy")
#adtest_MLP_layer2 = np.load("./AdTest_MLP_Layers_output/H_Layer2.npy")
#adtest_MLP_layer3 = np.load("./AdTest_MLP_Layers_output/H_Layer3.npy")


# In[6]:


#adtest_SimCLR_layer1 = torch.Tensor(adtest_SimCLR_layer1)
#adtest_SimCLR_layer2 = torch.Tensor(adtest_SimCLR_layer2)
#adtest_SimCLR_layer3 = torch.Tensor(adtest_SimCLR_layer3)


# In[7]:


#adtest_MLP_layer1 = torch.Tensor(adtest_MLP_layer1)
#adtest_MLP_layer2 = torch.Tensor(adtest_MLP_layer2)
#adtest_MLP_layer3 = torch.Tensor(adtest_MLP_layer3)


# In[8]:


#adtest_MLP_layer1.size()


# In[9]:


#cos = nn.CosineSimilarity(dim = 0, eps=1e-6)


# In[10]:


#similarity_layer1 = F.cosine_similarity(adtest_SimCLR_layer1, adtest_MLP_layer1)


# In[11]:


#similarity_layer1


# In[12]:


#for i in range(100):
#    features.append(adtest_SimCLR_layer1[i])
#    features.append(adtest_MLP_layer1[i])
    
# featuresTorch=torch.stack(features)
# features2Torch=featuresTorch.squeeze()
# simlarity=F.cosine_similarity(features2Torch.unsqueeze(1),features2Torch.unsqueeze(0),dim=2)
# plt.figure()
# plt.imshow(simlarity.detach().numpy())


# In[13]:


Test_targets = np.load("./AdversarialSamples0/targetsTest-25-model0.npy")
index = []

for i in range(10):
    index_i = np.arange(0,10000)[np.array(Test_targets==i)][:15]
    index.append(index_i)
    
print(len(index[0]))


# In[14]:


for i in range(len(AdTest_SimCLR_files)):

    adtest_SimCLR_layer = np.load("./AdTest_SimCLR_Layers"+str(len(AdTest_SimCLR_files))+"_output/Layer"+str(i)+".npy")
    adtest_SimCLR_layer = torch.Tensor(adtest_SimCLR_layer)
#    print(adtest_SimCLR_layer[0].size())
    
    test_SimCLR_layer = np.load("./Test_SimCLR_Layers"+str(len(AdTest_SimCLR_files))+"_output/Layer"+str(i)+".npy")
    test_SimCLR_layer = torch.Tensor(test_SimCLR_layer)
#    print(test_SimCLR_layer[0].size())
    
    for j in range(10):

        
        features = []
        for k in index[j]:
            features.append(adtest_SimCLR_layer[k])
            features.append(test_SimCLR_layer[k])

        featuresTorch=torch.stack(features)
        features2Torch=featuresTorch.squeeze()
        simlarity=F.cosine_similarity(features2Torch.unsqueeze(1),features2Torch.unsqueeze(0),dim=2)
        plt.figure()
        plt.imshow(simlarity.detach().numpy())
        plt.title("sum ={} layer={} target={}".format(len(AdTest_SimCLR_files), i,j))
        plt.colorbar(cmap="jet")
        
        plt.savefig('./cos_image/Sum'+str(len(AdTest_SimCLR_files))+'layer'+ str(i)+'_target_'+ str(j)+'.png')
        
       
           
        

    
    


# In[15]:


simlarity


# In[ ]:





# In[ ]:




