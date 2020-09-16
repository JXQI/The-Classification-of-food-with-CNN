import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5) #pannel,kernel数目,kernel数目
        self.conv2=nn.Conv2d(6,16,5)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(16*53*53,30)
        self.fc2=nn.Linear(30,11)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x=x.view(-1,16*53*53)   #这里是将所有的特征flatten,是需要计算的，而不是随便给的
        x=F.relu(self.fc1(x))
        #x=F.relu(self.fc2(x))  #TODO:注意最后一层不能加激活函数
        x=self.fc2(x)

        return x
