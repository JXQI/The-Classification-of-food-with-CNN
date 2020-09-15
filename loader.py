import os
from os.path import join
import torch
from torch.utils.data import Dataset
from PIL import Image

class dataloader(Dataset):
    def __init__(self,path,transforms,train_set=True):
        self.path=path
        self.transforms=transforms
        self.train_set=train_set
        self.images=[]
        self.labels=[]

        if self.train_set:
            self.images=os.listdir(self.path)
            self.labels=[int(i.split('_')[0]) for i in self.images]
            #print(len(self.images),len(self.labels))
        else:
            self.images = os.listdir(self.path)
            #print(len(self.images), len(self.labels))
    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        image=Image.open(join(self.path,self.images[item]))
        image=self.transforms(image)
        if self.train_set:
            label = self.labels[item]
            label=torch.tensor(label)
            return image,label
        else:
            return image
if __name__=='__main__':
    data=dataloader(path='./food-11/training',train_set=True)
    print(data.len())
    print(data[0])
    data=dataloader(path='./food-11/testing',train_set=False)
    print(data.len())
    print(data[0])
    data=dataloader(path='./food-11/validation',train_set=False)
    print(data.len())
    print(data[0])