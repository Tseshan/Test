#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append(r'D:\Robust_SimCLR\Model_package')
sys.path.append(r'D:\Robust_SimCLR\Models')


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
print("cuda memory = {:.3f} GBs".format(torch.cuda.max_memory_allocated() / 1024 ** 3))
#%%
layer_num = 5
attack_epsilon = 0.25
device=torch.device("cpu")
batchSize=64
epoches = 200

# In[4]:


MLP = torch.load("./Models/mlp_layer"+str(layer_num)+".pt").to(device)


# In[5]:


print(MLP)


# In[6]:


mlp_extractor = MLP_extractor(28*28).cpu()
torch.save(mlp_extractor,"./Models/mlp_extractor_initial_layer_"+str(layer_num)+".pt")


# In[7]:


#print("mlp_extractor:{}\r\n".format(mlp_extractor))


# In[8]:


simCLR_extractor=SimCLR("./Models/mlp_extractor_initial_layer_"+str(layer_num)+".pt").to(device)


# In[9]:


simCLR_classifier = torch.load("./Models/classifier.pt").to(device)


# In[10]:


simCLR = FullMLP(simCLR_extractor,simCLR_classifier)


# In[11]:


print("SimCLR:{}\r\n".format(simCLR))



# In[13]:



path1 = "./AdversarialSamples_layer"+str(layer_num)+"/DataTrain-"+str(attack_epsilon*100)+"-model_layer"+str(layer_num)+".npy"
path2 = "./AdversarialSamples_layer"+str(layer_num)+"/targetsTrain-"+str(attack_epsilon*100)+"-model_layer"+str(layer_num)+".npy"
path3 = "./AdversarialSamples_layer"+str(layer_num)+"/adversarialDataTrain-epsilon-"+str(attack_epsilon*100)+"-model_layer"+str(layer_num)+".npy"
path4 = "./AdversarialSamples_layer"+str(layer_num)+"/DataTest-"+str(attack_epsilon*100)+"-model_layer"+str(layer_num)+".npy"
path5 = "./AdversarialSamples_layer"+str(layer_num)+"/targetsTest-"+str(attack_epsilon*100)+"-model_layer"+str(layer_num)+".npy"
path6 = "./AdversarialSamples_layer"+str(layer_num)+"/adversarialDataTest-epsilon-"+str(attack_epsilon*100)+"-model_layer"+str(layer_num)+".npy"


# In[14]:


trainLoader,testLoader=GetDataLoader(*Get_New_MnistData(path1,path2,path3,path4,path5,path6),batchSize) 

#ad_trainLoader,ad_testLoader=GetDataLoader(*Get_New_MnistData_orgin(path3,path2,path6,path5),batchSize) 

# In[15]:


#for step, sample in enumerate(trainLoader):
#    print(sample[0][1].size())


# In[16]:


#eval_on_dataLoader(simCLR,MLP,testLoader,attack_epsilon)


# # 训练SimCLR特征提取器
# 

# In[17]:

#for step, sample in enumerate(ad_trainLoader):
#    print(step*64)

#%%
contrastiveLoss,testloss = TrainSimCLR(simCLR_extractor,MLP,trainLoader,testLoader,epoches,trainLoader.batch_size,attack_epsilon, device)


# In[ ]:


plt.figure()
plt.plot(range(epoches), contrastiveLoss)
plt.xlabel('epoches')
plt.ylabel('ContrastiveLoss')
plt.show()


# In[ ]:


#.cpu().detach().numpy()


# In[ ]:


#np.array(testAccs.detach().numpy())


# In[ ]:


#plt.figure()
#plt.plot(range(20), testAccs.cpu().detach().numpy())
#plt.xlabel('epoches')
#plt.ylabel('Test Accuracy')
#plt.show()


# In[ ]:


testLoss = []
for i in range(epoches):
    testLoss.append(testloss[i].item())


# In[ ]:


plt.figure()
plt.plot(range(epoches), testLoss)
plt.xlabel('epoches')
plt.ylabel('Test Loss')
plt.show()


# In[ ]:


simCLR_classifier = torch.load("./Models/classifier.pt").to(device)


# In[ ]:


simCLR = FullMLP(simCLR_extractor,simCLR_classifier)


# In[ ]:


print(TestAccuracy_SimCLR_origndata(simCLR,testLoader, device))


# # 训练SimCLR分类器

# In[ ]:


train_losses, test_losses, test_accuracy  = TrainClassifier_SimCLR_origndata(simCLR_extractor, simCLR_classifier, trainLoader,testLoader, epoches, device)


# In[ ]:


test_accuracy


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
print(TestAccuracy_SimCLR_origndata(simCLR,testLoader, device))


# In[ ]:


torch.save(simCLR,"./Models/simCLR"+str(attack_epsilon*100)+"_layer"+str(layer_num)+".pt")


# In[ ]:


print(TestAccuracy_SimCLR_adversarialdata(simCLR,testLoader, device))


# In[ ]:


#simCLR = FullMLP(simCLR_extractor,simCLR_classifier)
#print(TestAccuracy_SimCLR_origndata(simCLR,testLoader))


# In[ ]:


simCLR


# In[18]:





# In[ ]:




