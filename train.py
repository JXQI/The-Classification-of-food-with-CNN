from model import Net
from loader import dataloader
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class Process:
    def __init__(self,device):
        self.device = device
        self.net=Net()
        self.net=self.net.to(self.device)
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        train_set=dataloader(path='./food-11/training',transforms=self.transform,train_set=True)
        val_set = dataloader(path='./food-11/validation', transforms=self.transform, train_set=True)
        self.train_loader=DataLoader(dataset=train_set,batch_size=8,shuffle=True,num_workers=2)
        self.val_loader = DataLoader(dataset=val_set, batch_size=8, shuffle=True, num_workers=2)
        self.loss=nn.CrossEntropyLoss()
        self.optim=optim.SGD(self.net.parameters(),lr=0.1,momentum=0.5)
    def train(self,epoch):
        for j in range(epoch):
            for i,data in enumerate(self.train_loader,0):
                self.optim.zero_grad()
                inputs,labels=data[0].to(self.device),data[1].to(self.device)
                output=self.net(inputs)
                #print(output,labels)
                loss=self.loss(output,labels)
                loss.backward() #计算梯度，反向传播
                self.optim.step()
                print("[%d, %d] loss:%f"%(j+1,i+1,loss))
    def validate(self):
        loss_sum=0
        total=0
        correct=0
        with torch.no_grad():
            for i,data in enumerate(self.val_loader,0):
                outputs=self.net(data[0].to(self.device))
                _,predicted=torch.max(outputs,1)
                total+=data[0].size[0]
                correct+=(predicted == data[0]).sum().item() #TODO:注意此处tensor类型支持(predicted == data[0]).sum(),列表不支持
                loss=self.loss(outputs,data[1].to(self.device))
                loss_sum+=loss
            loss=loss_sum/total
            acc=correct/total
            print("The loss is %f ,The accuarcy is %f"%(loss,acc))


if __name__=="__main__":
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    pro=Process(device)
    pro.train(epoch=200)
    pro.validate()
