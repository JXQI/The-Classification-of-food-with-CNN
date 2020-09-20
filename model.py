import torch.nn as nn
import torch.nn.functional as F
import torch
#
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5) #pannel,kernel数目,kernel大小
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
#
class Alex_Net(nn.Module):
    def __init__(self):
        super(Alex_Net,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            # TODO:LRN
            nn.Conv2d(96,256,kernel_size=5,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            #TODO LRN
            nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=256*6*6,out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(),  #为何这里没有Dropout，因为这里只是一个简单的映射，不能丢失特征信息
            nn.Linear(in_features=4096,out_features=1000),
        )

    def forward(self,x):
            x=self.features(x)
            x = torch.flatten(x, 1)
            x=self.classifier(x)
            return  x

if __name__=="__main__":
    # net=Net()
    # print(net.fc1)
    alexnet=Alex_Net()
    print(alexnet.features)
