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
attack_epsilon = 0.119
device = torch.device("cpu")

#%%

def Weight_cat(weight_list):
    for i in range(len(weight_list)-1):
        print("weight:{}\r".format(weight_list[i].size()))
        weight_list[i+1] = torch.cat([weight_list[i],weight_list[i+1]], dim = 1)
        print("weight_list_i:{}, weight:{}\r".format(i+1, weight_list[i+1].size()))
    
    return weight_list[len(weight_list)-1]
 

# In[3]:
def MLP_SimCLR_Cos_Similarity(layer_num,attack_epsilon,device):
    
    simCLR = torch.load("./Models/simCLR"+str(attack_epsilon*100)+"_layer"+str(layer_num)+".pt").to(device)
    MLP = torch.load("./Models/mlp_layer"+str(layer_num)+".pt").to(device)
    #print(simCLR)
    #print(MLP)
    SimCLR_extract_output = list(simCLR.children())[0]
    SimCLR_Layer_List = list(list(list(SimCLR_extract_output.children())[0].children())[0].children())
    SimCLR_Layer_List[0]
    MLP_extract_output = nn.Sequential(*list(MLP.children())[0])
    
    MLP_weight = []
    i = 0
    for name,param in MLP_extract_output.named_parameters():
        if name == str(i)+".weight":    
            print(name)
            print(torch.Tensor(param.cpu().detach().numpy()).size())
            MLP_weight.append(torch.Tensor(param.cpu().detach().numpy()).reshape(1, -1))
            i = i+2
    
    SimCLR_weight = []
    i = 0
    for name,param in SimCLR_extract_output.named_parameters():
        if name == "baseModel.layers."+str(i)+".weight":
            
            print(name)
            print(torch.Tensor(param.cpu().detach().numpy()).size())
            SimCLR_weight.append(torch.Tensor(param.cpu().detach().numpy()).reshape(1, -1))
            i = i+2
            

    MLP_weight_cat = Weight_cat(MLP_weight)
    SimCLR_weight_cat = Weight_cat(SimCLR_weight)
    
    MLP_SimCLR_cos_similarity = F.cosine_similarity(SimCLR_weight_cat, MLP_weight_cat)

    return MLP_SimCLR_cos_similarity
    
#%%


print(MLP_SimCLR_Cos_Similarity(layer_num,attack_epsilon,device))  
        
#%%
#simCLR = torch.load("./Models/simCLR"+str(attack_epsilon*100)+"_layer"+str(layer_num)+".pt").cpu()
#MLP = torch.load("./Models/mlp_layer"+str(layer_num)+".pt").cpu()

#%%
#print(simCLR)

# In[4]:


#SimCLR_extract_output = list(simCLR.children())[0]
#SimCLR_Layer_List = list(list(list(SimCLR_extract_output.children())[0].children())[0].children())
#SimCLR_Layer_List[0]


# In[5]:


#MLP_extract_output = nn.Sequential(*list(MLP.children())[0])


# In[6]:


#MLP_weight = []
#i = 0
#for name,param in MLP_extract_output.named_parameters():
#    if name == str(i)+".weight":    
#        print(name)
#        print(torch.Tensor(param.cpu().detach().numpy()).size())
#        MLP_weight.append(torch.Tensor(param.cpu().detach().numpy()).reshape(1, -1))
#        i = i+2


# In[7]:


#SimCLR_weight = []
#i = 0
#for name,param in SimCLR_extract_output.named_parameters():
#    if name == "baseModel.layers."+str(i)+".weight":
        
#        print(name)
#        print(torch.Tensor(param.cpu().detach().numpy()))
#       SimCLR_weight.append(torch.Tensor(param.cpu().detach().numpy()).reshape(1, -1))
#        i = i+2
        
    


# In[8]:


#SimCLR_weight[1].size()


# In[ ]:





# In[9]:


#weight_similarity = []
#for j in range(len(MLP_weight)):
    
    #features_weight.append(SimCLR_weight[j])
    #features_weight.append(MLP_weight[j])
    
#    weight_similarity .append(F.cosine_similarity(SimCLR_weight[j],MLP_weight[j]))
    
#plt.plot(weight_similarity)

#plt.ylabel('Cosine similarity')
#plt.xlabel('layer')

#plt.savefig('./Weight_similarity_image/layer'+str(len(MLP_weight))+'.png')

#plt.show()    
    


  
        
        #plt.savefig('./cos_image/Sum'+str(len(AdTest_SimCLR_files))+'layer'+ str(i)+'_target_'+ str(j)+'.png')


# In[10]:


#F.cosine_similarity(SimCLR_weight[0],MLP_weight[0])


# In[11]:


#F.cosine_similarity(SimCLR_weight[1],MLP_weight[1])


# In[12]:


#F.cosine_similarity(SimCLR_weight[2],MLP_weight[2])


# In[13]:


#F.cosine_similarity(SimCLR_weight[3],MLP_weight[3])


# In[ ]:





# In[14]:


#F.cosine_similarity(torch.rand(1,156800),torch.rand(1,156800))


# In[ ]:




