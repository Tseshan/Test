#!/usr/bin/env python
# coding: utf-8

# In[4]:


from torchvision import datasets
import torch
from torchvision.transforms import transforms
import numpy as np
from data.New_Dataset import*

def GetMnistData():
    #读取数据
    train_data = datasets.MNIST(root="./data", train=True, download=True,transform=transforms.ToTensor())
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    return train_data,test_data
def GetDataLoader(train_data,test_data,batch_size):
    #创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = 0)
    return train_loader,test_loader


# In[1]:


def Get_New_MnistData(train_path1,train_target_path2,adtrain_path3,test_path1,test_target_path2,adtest_path3):
    #读取数据
    new_train_data = NewData(train_path1,train_target_path2,adtrain_path3)
    new_test_data =  NewData(test_path1,test_target_path2,adtest_path3)
    return new_train_data,new_test_data


# In[ ]:


def Get_New_MnistData_orgin(train_path1,train_target_path2,test_path1,test_target_path2):
    #读取数据
    new_train_data = NewData_orgin(train_path1,train_target_path2)
    new_test_data =  NewData_orgin(test_path1,test_target_path2)
    return new_train_data,new_test_data


# In[ ]:





# In[ ]:




