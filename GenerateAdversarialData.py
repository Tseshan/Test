#!/usr/bin/env python
# coding: utf-8

# In[1]:


from GetData import GetDataLoader,GetMnistData
from Model_package.MLP.MLP import MLP
from Model_package.FGSM.fgsm import get_fgsm_samples_data
from Model_package.FGSM import fgsm
import torch
import numpy as np
import os
import torch.nn.functional as F

import sys
from matplotlib import pyplot as plt


# In[3]:


sys.path


# In[4]:


layer_num = 3

# In[5]:


def GetAvailablePath(filepath, layer_num):
    
    basename,extname=os.path.splitext(filepath)
    newPath=basename+"_layer"+str(layer_num)+extname
    
    
    #basename,extname=os.path.splitext(filepath)
    #count=0
    #newPath=basename+str(count)+extname
    #while os.path.exists(newPath):
    #   count+=1
    #    newPath=basename+str(count)+extname
    return newPath

def RecursiveCreatePath(filepath):
    dirname,basename=os.path.split(filepath)
    if not os.path.exists(dirname):
        RecursiveCreatePath(dirname)
    if not os.path.exists(filepath):
        os.mkdir(filepath)



# In[6]:

attack_epsilon = 0.368

# In[7]:


trainData,testData=GetMnistData()
testdataLoader=torch.utils.data.DataLoader(testData,batch_size=len(testData),shuffle=False)
traindataLoader=torch.utils.data.DataLoader(trainData,batch_size=len(trainData),shuffle=False)



# In[8]:

filenames={}
filenames['MLPModelPath']="./Models/mlp_layer"+str(layer_num)+".pt"

# In[9]:



filenames['rootDir']=GetAvailablePath("./AdversarialSamples", layer_num)

filenames['ClassifierMLPPath']="./Models/classifierMLP.pt"
filenames['TrainAdversarialPath']=GetAvailablePath(os.path.join(filenames['rootDir'],"adversarialDataTrain-epsilon-"+str(attack_epsilon*100)+"-model.npy"), layer_num)
filenames['TestAdversarialPath']=GetAvailablePath(os.path.join(filenames['rootDir'],"adversarialDataTest-epsilon-"+str(attack_epsilon*100)+"-model.npy"), layer_num)

RecursiveCreatePath(filenames['rootDir'])


# In[10]:


#filenames['TrainDataPath']=os.path.join(filenames['rootDir'],"DataTrian-25-model.npy"))
#filenames['TestDataPath']=os.path.join(filenames['rootDir'],"DataTest-25-model.npy")
#filenames['TrainTargetsPath']=os.path.join(filenames['rootDir'],"TargetsTrian-25-model.npy")
#filenames['TestTargetsPath']=os.path.join(filenames['rootDir'],"targetsTest-25-model.npy")



filenames['TrainDataPath']=GetAvailablePath(os.path.join(filenames['rootDir'],"DataTrain-"+str(attack_epsilon*100)+"-model.npy"), layer_num)
filenames['TestDataPath']=GetAvailablePath(os.path.join(filenames['rootDir'],"DataTest-"+str(attack_epsilon*100)+"-model.npy"),layer_num)
filenames['TrainTargetsPath']=GetAvailablePath(os.path.join(filenames['rootDir'],"targetsTrain-"+str(attack_epsilon*100)+"-model.npy"),layer_num)
filenames['TestTargetsPath']=GetAvailablePath(os.path.join(filenames['rootDir'],"targetsTest-"+str(attack_epsilon*100)+"-model.npy"), layer_num)


# In[11]:


mlp = torch.load(filenames['MLPModelPath'])


# In[12]:

print(mlp)

# In[13]:

train_final_acc, train_adv_data = fgsm.test(mlp,"cpu",traindataLoader, attack_epsilon)
test_final_acc, test_adv_data = fgsm.test(mlp,"cpu",testdataLoader, attack_epsilon)


# In[14]:

trainData.targets


# In[15]:

np.save(filenames['TrainAdversarialPath'],train_adv_data.cpu().detach().numpy())
np.save(filenames['TestAdversarialPath'],test_adv_data.cpu().detach().numpy())

# In[16]:

np.save(filenames['TrainDataPath'],trainData.data.cpu().detach().numpy())
np.save(filenames['TestDataPath'],testData.data.cpu().detach().numpy())

# In[17]:

np.save(filenames['TrainTargetsPath'],trainData.targets.cpu().detach().numpy())
np.save(filenames['TestTargetsPath'],testData.targets.cpu().detach().numpy())

# In[18]:


index=[0,1,2,3,4,5,6,7]
fig,ax=plt.subplots(2,len(index),figsize=(2*len(index),2*2))
adSample=train_adv_data[index]
orignSample=(trainData.data[index])
originLabel=trainData.targets[index]
adLabel=torch.argmax(mlp(adSample),dim=1).cpu().detach().numpy()
for i in range(len(index)):
    ax[0][i].imshow(orignSample[i],cmap="gray")
    ax[1][i].imshow(adSample[i].cpu().detach().numpy().squeeze(),cmap="gray")
    ax[0][i].axis("off")
    ax[1][i].axis("off")
    ax[0][i].set_title(str(originLabel[i].item())+"->"+str(adLabel[i]))
fig.show()


# In[19]:


plt.imshow(train_adv_data[995].cpu().detach().numpy().squeeze(),cmap="gray")


# In[20]:
#for i in range(10000):
#    if int(trainData.targets[i]) == int(torch.argmax(mlp(train_adv_data[i]),dim=1).cpu().detach().numpy()):
#        print(i)
# In[29]:
# In[22]:
mlp0 = torch.load(filenames['MLPModelPath'])
# In[23]:

a = 9516   
    
plt.imshow(train_adv_data[a].cpu().detach().numpy().squeeze(),cmap="gray")
output = mlp0(train_adv_data[a])
tidu  = F.cross_entropy(output, torch.tensor([trainData.targets[a]]))
print('tidu:{}\r\n'.format(tidu))
#print(torch.argmax(output,dim=1).cpu().detach().numpy())
print(torch.softmax(output, dim=1))
# In[24]:
    
print(trainData.targets[6109])

# In[25]:    
torch.tensor([trainData.targets[6109]])
# In[26]:

print(output)

