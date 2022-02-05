#!/usr/bin/env python
# coding: utf-8

# In[2]:


from torch.utils.data import DataLoader,Dataset
import numpy as np

class NewData(Dataset):
    def __init__(self, path1, path2, path3): 
        self.images = np.load(path1)
        self.lables = np.load(path2)
        self.ad_images = np.load(path3)
        
    def __len__(self):#返回整个数据集的大小
        return len(self.images)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]  
        return [self.images[index], self.lables[index], ad_images[index]]   


# In[ ]:




