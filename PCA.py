#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import AdTest_SimCLR_Layers_output 


# In[5]:


AdTest_SimCLR_files = os.listdir(".\AdTest_SimCLR_Layers"+str(3)+"_output")


# In[6]:


Layer_sum = len(AdTest_SimCLR_files)


# In[7]:


Adtest_data = np.load("./AdversarialSamples0/adversarialDataTest-25-model0.npy")
Test_data = np.load("./AdversarialSamples0/DataTest-25-model0.npy")
Test_targets = np.load("./AdversarialSamples0/targetsTest-25-model0.npy")


# In[8]:


def PCA_test(index, data, targets,):
    sample = data[index]
    labels = targets[index]
    packedSamplie = [sample, labels]
    
    featuresMeanCentered=sample-sample.mean(dim=1).reshape(-1,1)
    pca=PCA(n_components=featuresMeanCentered.shape[1])
    pca.fit(featuresMeanCentered.detach().numpy())
    
    dimension =  sum(pca.singular_values_)**2/sum(pca.singular_values_**2)
    
    return dimension


# In[6]:


#adtest_SimCLR_layer1 = np.load("./AdTest_SimCLR_Layers_output/H_Layer1.npy")
#adtest_SimCLR_layer2 = np.load("./AdTest_SimCLR_Layers_output/H_Layer2.npy")
#adtest_SimCLR_layer3 = np.load("./AdTest_SimCLR_Layers_output/H_Layer3.npy")
#adtest_SimCLR_layer4 = np.load("./AdTest_SimCLR_Layers_output/H_Layer4.npy")


# In[7]:


#test_SimCLR_layer1 = np.load("./Test_SimCLR_Layers_output/H_Layer1.npy")
#test_SimCLR_layer2 = np.load("./Test_SimCLR_Layers_output/H_Layer2.npy")
#test_SimCLR_layer3 = np.load("./Test_SimCLR_Layers_output/H_Layer3.npy")
#test_SimCLR_layer4 = np.load("./Test_SimCLR_Layers_output/H_Layer4.npy")


# In[8]:


#adtest_MLP_layer1 = np.load("./AdTest_MLP_Layers_output/H_Layer1.npy")
#adtest_MLP_layer2 = np.load("./AdTest_MLP_Layers_output/H_Layer2.npy")
#adtest_MLP_layer3 = np.load("./AdTest_MLP_Layers_output/H_Layer3.npy")
#adtest_MLP_layer4 = np.load("./AdTest_MLP_Layers_output/H_Layer4.npy")


# In[9]:


#test_MLP_layer1 = np.load("./Test_MLP_Layers_output/H_Layer1.npy")
#test_MLP_layer2 = np.load("./Test_MLP_Layers_output/H_Layer2.npy")
#test_MLP_layer3 = np.load("./Test_MLP_Layers_output/H_Layer3.npy")
#test_MLP_layer4 = np.load("./Test_MLP_Layers_output/H_Layer4.npy")


# In[10]:


#adtest_SimCLR_layer1 = torch.Tensor(adtest_SimCLR_layer1)
#adtest_SimCLR_layer2 = torch.Tensor(adtest_SimCLR_layer2)
#adtest_SimCLR_layer3 = torch.Tensor(adtest_SimCLR_layer3)
#adtest_SimCLR_layer4 = torch.Tensor(adtest_SimCLR_layer4)


#test_SimCLR_layer1 = torch.Tensor(test_SimCLR_layer1)
#test_SimCLR_layer2 = torch.Tensor(test_SimCLR_layer2)
#test_SimCLR_layer3 = torch.Tensor(test_SimCLR_layer3)
#test_SimCLR_layer4 = torch.Tensor(test_SimCLR_layer4)


# In[11]:


#adtest_MLP_layer1 = torch.Tensor(adtest_MLP_layer1)
#adtest_MLP_layer2 = torch.Tensor(adtest_MLP_layer2)
#adtest_MLP_layer3 = torch.Tensor(adtest_MLP_layer3)
#adtest_MLP_layer4 = torch.Tensor(adtest_MLP_layer4)


#test_MLP_layer1 = torch.Tensor(test_MLP_layer1)
#test_MLP_layer2 = torch.Tensor(test_MLP_layer2)
#test_MLP_layer3 = torch.Tensor(test_MLP_layer3)
#test_MLP_layer4 = torch.Tensor(test_MLP_layer4)


# In[12]:


#np.cov(adtest_SimCLR_layer1)[1]


# In[9]:


index = []


# In[10]:


for i in range(10):
    index_i = np.arange(0,10000)[np.array(Test_targets==i)][:1000]
    index.append(index_i)


# In[11]:


target_num = 8
index_num = index[target_num]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


dimension_adtest_SimCLR_target_num = []
dimension_test_SimCLR_target_num = []
dimension_adtest_MLP_target_num = []
dimension_test_MLP_target_num = []


for num in range(10):
    dimension_adtest_SimCLR = []
    dimension_test_SimCLR = []
    dimension_adtest_MLP = []
    dimension_test_MLP = []
    for i in range(Layer_sum):
        
        adtest_SimCLR_layer = np.load("./AdTest_SimCLR_Layers"+str(Layer_sum)+"_output/Layer"+str(i)+".npy")
        adtest_SimCLR_layer = torch.Tensor(adtest_SimCLR_layer)  
        dimension_adtest_SimCLR_layer = PCA_test(index_num, adtest_SimCLR_layer, Test_targets)
        dimension_adtest_SimCLR.append(dimension_adtest_SimCLR_layer)

        test_SimCLR_layer = np.load("./Test_SimCLR_Layers"+str(Layer_sum)+"_output/Layer"+str(i)+".npy")
        test_SimCLR_layer = torch.Tensor(test_SimCLR_layer)   
        dimension_test_SimCLR_layer = PCA_test(index_num, test_SimCLR_layer, Test_targets)
        dimension_test_SimCLR.append(dimension_test_SimCLR_layer)




        adtest_MLP_layer = np.load("./AdTest_MLP_Layers"+str(Layer_sum)+"_output/Layer"+str(i)+".npy")
        adtest_MLP_layer = torch.Tensor(adtest_MLP_layer)  
        dimension_adtest_MLP_layer = PCA_test(index_num, adtest_MLP_layer, Test_targets)
        dimension_adtest_MLP.append(dimension_adtest_MLP_layer)

        test_MLP_layer = np.load("./Test_MLP_Layers"+str(Layer_sum)+"_output/Layer"+str(i)+".npy")
        test_MLP_layer = torch.Tensor(test_MLP_layer)   
        dimension_test_MLP_layer = PCA_test(index_num, test_MLP_layer, Test_targets)
        dimension_test_MLP.append(dimension_test_MLP_layer)    
    
    #dimension_adtest_SimCLR_target_num.append(dimension_adtest_SimCLR)
    #dimension_test_SimCLR_target_num.append(dimension_test_SimCLR)
    #dimension_adtest_MLP_target_num.append(dimension_adtest_MLP)
    #dimension_test_MLP_target_num.append(dimension_test_MLP)
    
    plt.figure(dpi=100)

    plt.title('layer'+str(Layer_sum)+'_target = '+ str(num))
    #plt.legend(('adtest_SimCLR', 'adtest_MLP', 'SimCLR','MLP')) 

    plt.plot(dimension_adtest_SimCLR, label='ad_SimCLR')
    plt.plot(dimension_test_SimCLR, label='SimCLR')
    plt.plot(dimension_test_MLP,label='MLP')
    plt.plot(dimension_adtest_MLP, label='ad_MLP')

    plt.ylabel('h')
    plt.xlabel('layer')
    plt.legend()
   

    #plt.savefig('./PCA_image/layer'+str(Layer_sum)+'_target_'+ str(num)+'.png')
    
    plt.show()

    
    


# # SimCLR

# In[17]:


#dimension_adtest_SimCLR_layer1 = PCA_test(index_num, adtest_SimCLR_layer1, Test_targets)
#dimension_adtest_SimCLR_layer2 = PCA_test(index_num, adtest_SimCLR_layer2, Test_targets)
#dimension_adtest_SimCLR_layer3 = PCA_test(index_num, adtest_SimCLR_layer3, Test_targets)
#dimension_adtest_SimCLR_layer4 = PCA_test(index_num, adtest_SimCLR_layer4, Test_targets)  


# In[18]:


#dimension_adtest_SimCLR = []
#dimension_adtest_SimCLR = [dimension_adtest_SimCLR_layer1,
#                           dimension_adtest_SimCLR_layer2,
#                           dimension_adtest_SimCLR_layer3,
#                           dimension_adtest_SimCLR_layer4
#                          ]


# In[19]:


#dimension_test_SimCLR_layer1 = PCA_test(index_num, test_SimCLR_layer1, Test_targets)
#dimension_test_SimCLR_layer2 = PCA_test(index_num, test_SimCLR_layer2, Test_targets)
#dimension_test_SimCLR_layer3 = PCA_test(index_num, test_SimCLR_layer3, Test_targets)
#dimension_test_SimCLR_layer4 = PCA_test(index_num, test_SimCLR_layer4, Test_targets)


# In[20]:


#dimension_test_SimCLR = []
#dimension_test_SimCLR = [dimension_test_SimCLR_layer1,
#                           dimension_test_SimCLR_layer2,
#                           dimension_test_SimCLR_layer3,
#                           dimension_test_SimCLR_layer4
#                          ]


# # MLP

# In[21]:


#dimension_adtest_MLP_layer1 = PCA_test(index_num, adtest_MLP_layer1, Test_targets)
#dimension_adtest_MLP_layer2 = PCA_test(index_num, adtest_MLP_layer2, Test_targets)
#dimension_adtest_MLP_layer3 = PCA_test(index_num, adtest_MLP_layer3, Test_targets)
#dimension_adtest_MLP_layer4 = PCA_test(index_num, adtest_MLP_layer4, Test_targets)


# In[22]:


#dimension_adtest_MLP = []
#dimension_adtest_MLP = [   dimension_adtest_MLP_layer1,
#                           dimension_adtest_MLP_layer2,
#                           dimension_adtest_MLP_layer3,
#                           dimension_adtest_MLP_layer4
#                          ]


# In[23]:


#dimension_test_MLP_layer1 = PCA_test(index_num, test_MLP_layer1, Test_targets)
#dimension_test_MLP_layer2 = PCA_test(index_num, test_MLP_layer2, Test_targets)
#dimension_test_MLP_layer3 = PCA_test(index_num, test_MLP_layer3, Test_targets)
#dimension_test_MLP_layer4 = PCA_test(index_num, test_MLP_layer4, Test_targets)


# In[24]:


#dimension_test_MLP = []
#dimension_test_MLP = [   dimension_test_MLP_layer1,
#                           dimension_test_MLP_layer2,
#                           dimension_test_MLP_layer3,
#                           dimension_test_MLP_layer4
#                          ]


# In[25]:


#plt.plot(dimension_adtest_SimCLR)


# In[26]:


#plt.plot(dimension_test_SimCLR)


# In[27]:


#plt.plot(dimension_adtest_MLP)


# In[28]:


#plt.plot(dimension_test_MLP)


# In[29]:


#dimension_adtest_MLP


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




