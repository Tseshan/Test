#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append(r'F:\神经网络的鲁棒性探究\Model_package')


# In[2]:


import torch
import torch.nn as nn
from Model_package.SimCLR.SimCLR import SimCLR
from Trainers.SimCLRTrainner import*
from  Trainers.ClassifierTrainner import*

from Model_package.MLP.MLP_extractor import*
from Model_package.FullModel.FullModel import*
from GetData import*
from matplotlib import pyplot as plt
from Model_package.Utilites import*


# In[3]:


layer_num = 4
attack_epsilon = 0.25


# In[ ]:


MLP = torch.load("./Models/mlp_layer"+str(layer_num)+".pt").cuda()


# In[ ]:


mlp_extractor = MLP_extractor(28*28).cpu()
torch.save(mlp_extractor,"./Models/mlp_extractor_initial.pt")


# In[ ]:


simCLR_extractor=SimCLR("./Models/mlp_extractor_initial.pt")#.cpu()


# In[ ]:


simCLR_classifier = torch.load("./Models/classifier.pt")#.cpu()


# In[ ]:


simCLR = FullMLP(simCLR_extractor,simCLR_classifier)


# In[ ]:


simCLR


# In[ ]:


batchSize=128
epoches = 150
path1 = "./AdversarialSamples_layer"+str(layer_num)+"/new_DataTrian-25-model_layer"+str(layer_num)+".npy"
path2 = "./AdversarialSamples_layer"+str(layer_num)+"/new_TargetsTrian-25-model_layer"+str(layer_num)+".npy"
#path3 = "./AdversarialSamples_layer"+str(layer_num)+"/adversarialDataTrian-25-model_layer"+str(layer_num)+".npy"
path4 = "./AdversarialSamples_layer"+str(layer_num)+"/new_DataTest-25-model_layer"+str(layer_num)+".npy"
path5 = "./AdversarialSamples_layer"+str(layer_num)+"/new_TargetsTest-25-model_layer"+str(layer_num)+".npy"
#path6 = "./AdversarialSamples_layer"+str(layer_num)+"/new_adversarialDataTest-25-model_layer"+str(layer_num)+".npy"


# In[ ]:


#trainLoader,testLoader=GetDataLoader(*Get_New_MnistData(path1,path2,path3,path4,path5,path6),batchSize) 


# In[ ]:


trainLoader,testLoader=GetDataLoader(*Get_New_MnistData_orgin(path1,path2,path4,path5),batchSize)


# In[ ]:


trainLoader


# In[ ]:


contrastiveLoss,testloss = TrainSimCLR(simCLR_extractor,MLP,trainLoader,testLoader,epoches,trainLoader.batch_size,attack_epsilon)


# In[ ]:


plt.figure()
plt.plot(range(epoches), contrastiveLoss)
plt.xlabel('epoches')
plt.ylabel('ContrastiveLoss')
plt.show()


# In[ ]:





# In[ ]:


testLoss = []
for i in range(epoches):
    testLoss.append(testloss[i])
    #testLoss.append(testloss[i].item())


# In[ ]:


#plt.figure()
#plt.plot(range(epoches), testLoss)
#plt.xlabel('epoches')
#plt.ylabel('Test Loss')
#plt.show()


# In[ ]:


simCLR = FullMLP(simCLR_extractor,simCLR_classifier)


# In[ ]:


print(TestAccuracy_origin(simCLR,testLoader))


# In[ ]:


train_losses, test_losses, test_accuracy = TrainClassifier_SimCLR_origin(simCLR_extractor, simCLR_classifier, trainLoader,testLoader, epoches=20)


# In[ ]:


plt.figure()
plt.plot(range(epoches), train_losses)
plt.xlabel('epoches')
plt.ylabel('train losses')
plt.show()


# In[ ]:


plt.figure()
plt.plot(range(epoches), test_losses)
plt.xlabel('epoches')
plt.ylabel('test losses')
plt.show()


# In[ ]:


plt.figure()
plt.plot(range(epoches), test_accuracy)
plt.xlabel('epoches')
plt.ylabel('test_accuracy')
plt.show()


# In[ ]:


simCLR = FullMLP(simCLR_extractor,simCLR_classifier)
print(TestAccuracyWithFGSM_origin(simCLR, MLP, testLoader, 0.25))


# In[ ]:


torch.save(simCLR,"./Models/simCLR_layer"+str(layer_num)+".pt")


# In[ ]:




