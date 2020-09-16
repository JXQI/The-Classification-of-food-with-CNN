from model import Net
from loader import dataloader
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from until import Accuracy,drawline
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class Process:
    def __init__(self,device):
        self.device = device
        self.net=Net()
        self.net=self.net.to(self.device)
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        train_set=dataloader(path='./food-11/training',transforms=self.transform,train_set=True)
        val_set = dataloader(path='./food-11/validation', transforms=self.transform, train_set=True)

        print(len(train_set),len(val_set))
        self.train_loader=DataLoader(dataset=train_set,batch_size=8,shuffle=True,num_workers=0)
        self.val_loader = DataLoader(dataset=val_set, batch_size=8, shuffle=True, num_workers=0)
        self.loss=nn.CrossEntropyLoss()
        self.optim=optim.SGD(self.net.parameters(),lr=0.1,momentum=0.9)
    def train(self,epoch):
        loss_list=[]
        acc_list=[]
        for j in range(epoch):
            running_loss=0
            for i,data in enumerate(self.train_loader,0):
                if i%100==99:
                    self.optim.zero_grad()
                    inputs,labels=data[0].to(self.device),data[1].to(self.device)
                    output=self.net(inputs)
                    #print(output,labels)
                    loss=self.loss(output,labels)
                    loss.backward() #计算梯度，反向传播
                    self.optim.step()
                    running_loss+=loss
                    if i%100==99:
                        print("[%d, %d] loss:%f"%(j+1,i+1,running_loss/100))
                        running_loss=0
            loss_temp,acc_temp=Accuracy(self.net,self.train_loader,self.loss,self.device)
            loss_list.append(loss_temp)
            acc_list.append(acc_temp)
            print("%d epoch the loss is %f,the accuarcy is %f " %(j,loss_temp,acc_temp))
        drawline(range(epoch),loss_list,"epoch","loss","the loss of train")
        drawline(range(epoch),acc_list, "epoch","accuarcy", "the accuracy of train")

    def validate(self):
        # loss_sum=0
        # total=0
        # correct=0
        # with torch.no_grad():
        #     for i,data in enumerate(self.val_loader,0):
        #         outputs=self.net(data[0].to(self.device))
        #         _,predicted=torch.max(outputs,1)
        #         total+=data[0].size(0)
        #         correct+=(predicted == data[1]).sum().item() #TODO:注意此处tensor类型支持(predicted == data[0]).sum(),列表不支持
        #         loss=self.loss(outputs,data[1].to(self.device))
        #         loss_sum+=loss
        #     loss=loss_sum/total
        #     acc=correct/total
        #     print("The loss is %f ,The accuarcy is %f"%(loss,acc))
        val_loss,val_acc=Accuracy(self.net,self.val_loader,self.loss,self.device)
        print("The loss is %f ,The accuarcy is %f"%(val_loss,val_acc))

if __name__=="__main__":
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    pro=Process(device)
    pro.train(epoch=2)
    pro.validate()
