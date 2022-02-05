#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append(r'D:\Robust_SimCLR\Model_package')
sys.path.append(r'D:\Robust_SimCLR\Model_package\Utilites')
sys.path.append(r'D:\Robust_SimCLR\Model_package\FGSM')


# In[2]:


#sys.path.append(r'F:\\神经网络的鲁棒性探究\\Model_package\\FGSM')
#print(sys.path)


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import transforms
from Model_package.FGSM import fgsm
from Model_package.FGSM.fgsm import get_fgsm_samples_data
from Model_package.MLP.MLP import MLP
from Model_package.SimCLR.ContrastiveLoss import ContrastiveLoss
from GetData import GetDataLoader,GetMnistData
from Model_package.Classifier import Classifier
from Model_package import Utilites
from Utilites import TestAccuracy
import os
import numpy as np
from Model_package.FGSM.AdversarialData import SaveAdversarialData
from matplotlib import pyplot as plt


# In[4]:


layer_num = 5


# In[5]:


#def GetAvailablePath(filepath):
#    basename,extname=os.path.splitext(filepath)
#    count=0
#    newPath=basename+str(count)+extname
#    while os.path.exists(newPath):
#        count+=1
#        newPath=basename+str(count)+extname
#    return newPath

#def RecursiveCreatePath(filepath):
#    dirname,basename=os.path.split(filepath)
#    if not os.path.exists(dirname):
#        RecursiveCreatePath(dirname)
#    if not os.path.exists(filepath):
#        os.mkdir(filepath)


# In[6]:

    

filenames={}
#filenames['rootDir']=GetAvailablePath("./Adversarial/AdversarialSamples")
filenames['MLPModelPath'] = "./Models/mlp_layer"+str(layer_num)+".pt"
#filenames['TrainAdversarialPath']=GetAvailablePath(os.path.join(filenames['rootDir'],"adversarialDataTrian-35-model.npy"))
#filenames['TestAdversarialPath']=GetAvailablePath(os.path.join(filenames['rootDir'],"adversarialDataTest-35-model.npy"))
#RecursiveCreatePath(filenames['rootDir'])


# In[7]:
mlp = torch.load(filenames['MLPModelPath'])
# In[8]:
print(mlp)


# In[9]:


trainData,testData=GetMnistData()
traindataLoader=torch.utils.data.DataLoader(trainData,batch_size=len(trainData),shuffle=False)#数据是0~1范围内

testdataLoader=torch.utils.data.DataLoader(testData,batch_size=len(testData),shuffle=False)#数据是0~1范围内


# In[10]:


#for data, target in testdataLoader:
#    print(data[0])


# In[11]:


#if hasattr(torch.cuda, 'empty_cache'):
#    torch.cuda.empty_cache()


# In[12]:


Final_Acc = []
Epsilon = []
for i in range(100):

    softmax = nn.Softmax(dim=1)
    acc = 0
    #sample=(testData.data/255).float()
    #labels=testData.targets
    #packedSample=[sample,labels]
    #Ad_data =  get_fgsm_samples_data(mlp, "cuda", packedSample, i*0.01)
    #plt.figure()
    #plt.imshow(sample[3].cpu().detach().numpy())
    #plt.imshow(Ad_data[3].cpu().detach().numpy())
    #output = mlp(Ad_data).to("cuda")
    #predict = torch.argmax(softmax(output), dim=1)
    #print(predict)
    #acc += (predict == labels.to("cuda")).sum().item()
    #print(acc/len(testdataLoader.dataset))
    #Final_Acc.append(acc/len(testdataLoader.dataset))
    #Epsilon.append(i*0.001)
    
#plt.figure()
#plt.plot(Epsilon, Final_Acc)
#plt.xlabel('epsilon')
#plt.ylabel('accuracy')
#plt.show()    

    final_acc, adv_examples = fgsm.test(mlp,"cpu",testdataLoader, i*0.01)
    plt.figure()
    plt.imshow(adv_examples[0][0].cpu().detach().numpy(),cmap="gray")
    plt.colorbar()
    plt.title('epsilon = {}'.format(i*0.01))
    plt.show()
    output = mlp(adv_examples).to("cuda")
    predict = torch.argmax(softmax(output), dim=1)
    #print(predict)
    labels=testData.targets
    acc += (predict == labels.to("cuda")).sum().item()/len(testdataLoader.dataset)
    print(acc)
    Final_Acc.append(acc)
    Epsilon.append(i*0.01)
   

plt.figure()
plt.plot(Epsilon, Final_Acc)
plt.xlabel('epsilon')
plt.ylabel('accuracy')
plt.show()        
    


# In[13]:


#len(testdataLoader.dataset)


# In[14]:


#(testData.data)[0]#.cpu().detach().numpy()


# In[15]:


#predict


# In[16]:


#Ad_data[0].size()
#plt.figure()
#plt.imshow(Ad_data[0].cpu().detach().numpy())


# # 确定攻击强度 ε为0.25

# In[17]:


#attack_epsilon = 0.25


# In[18]:


#traindataLoader=torch.utils.data.DataLoader(trainData,batch_size=len(trainData),shuffle=False)


# In[19]:


#train_final_acc, train_adv_data = fgsm.test(mlp,"cuda",traindataLoader, attack_epsilon)
#test_final_acc, test_adv_data = fgsm.test(mlp,"cuda",testdataLoader, attack_epsilon)


# In[20]:


#test_final_acc, test_adv_data = fgsm.test(mlp,"cuda",testdataLoader, attack_epsilon)

#plt.figure()
#plt.imshow(adv_examples[0][0].cpu().detach().numpy())


# In[21]:


#adv_examples.size()


# In[22]:


#Final_Acc = []
#Epsilon = []
#for i in range(20):
#    final_acc, adv_examples = fgsm.test(mlp,"cuda",testdataLoader, i*0.01)
#    Final_Acc.append(final_acc)
#    Epsilon.append(i*0.01)
#plt.figure()
#plt.plot(Epsilon, Final_Acc)
#plt.xlabel('epsilon')
#plt.ylabel('accuracy')
#plt.show()


# In[23]:


#index=[0,1,2,3,4,5,6,7]
#fig,ax=plt.subplots(2,len(index),figsize=(2*len(index),2*2))
#adSample=train_adv_data[index]
#orignSample=(trainData.data[index])
#originLabel=trainData.targets[index]
#adLabel=torch.argmax(mlp(adSample.cuda()),dim=1).cpu().detach().numpy()
#for i in range(len(index)):
#    ax[0][i].imshow(orignSample[i])
#    ax[1][i].imshow(adv_examples[i].cpu().detach().numpy().squeeze())
#    ax[0][i].axis("off")
#    ax[1][i].axis("off")
#    ax[0][i].set_title(str(originLabel[i].item())+"->"+str(adLabel[i]))
#fig.show()


# In[24]:
#SaveAdversarialData(mlp,trainData,filenames['TrainAdversarialPath'])
#SaveAdversarialData(mlp,testData,filenames['TestAdversarialPath'])
# In[25]:


#a0 = get_fgsm_samples_data(mlp, "cuda", packedSample, 0.25)[0]

#plt.figure()
#plt.imshow(a0.cpu().detach().numpy())


# In[26]:


#trainLoader.data[0]


# In[27]:


a = torch.tensor(1e-45)


# In[28]:
torch.sign(a)


