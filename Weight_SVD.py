#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


layer_num = 5


# In[3]:


simCLR = torch.load("./Models/simCLR_layer"+str(layer_num)+".pt").cpu()
MLP = torch.load("./Models/mlp_layer"+str(layer_num)+".pt").cpu()


# In[4]:


SimCLR_extract_output = list(simCLR.children())[0]
SimCLR_Layer_List = list(list(list(SimCLR_extract_output.children())[0].children())[0].children())
SimCLR_Layer_List[0]


# In[ ]:





# In[5]:


MLP_extract_output = nn.Sequential(*list(MLP.children())[0])


# In[6]:


MLP_weight = []
i = 0
for name,param in MLP_extract_output.named_parameters():
    if name == str(i)+".weight":    
        print(name)
        print(torch.Tensor(param.cpu().detach().numpy()).size())
        MLP_weight.append(torch.Tensor(param.cpu().detach().numpy()))
        i = i+2


# In[7]:


SimCLR_weight = []
i = 0
for name,param in SimCLR_extract_output.named_parameters():
    if name == "baseModel.layers."+str(i)+".weight":
        
        print(name)
        print(torch.Tensor(param.cpu().detach().numpy()))
        SimCLR_weight.append(torch.Tensor(param.cpu().detach().numpy()))
        i = i+2


# In[8]:


torch.svd(MLP_weight[0])


# In[9]:


dimension_SimCLR_weight = []
dimension_MLP_weight = []

for i in range(len(MLP_weight)):
    dimension_MLP_weight.append((sum(torch.svd(MLP_weight[i])[1]))**2/sum(torch.svd(MLP_weight[i])[1]**2))
    dimension_SimCLR_weight.append((sum(torch.svd(SimCLR_weight[i])[1]))**2/sum(torch.svd(SimCLR_weight[i])[1]**2))
    #plt.legend(('adtest_SimCLR', 'adtest_MLP', 'SimCLR','MLP')) 

plt.plot(dimension_SimCLR_weight, label='SimCLR')
plt.plot(dimension_MLP_weight,label='MLP')

plt.ylabel('dimension')
plt.xlabel('layer')
plt.legend()

plt.savefig('./SVD_image/layer'+str(len(MLP_weight))+'.png')

plt.show()


# In[10]:


dimension_MLP_weight[2]


# In[11]:


(sum(torch.svd(torch.rand(100, 150))[1]))**2/sum(torch.svd(torch.rand(100,150))[1]**2)


# In[ ]:




