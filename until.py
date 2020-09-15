import torch

def Accuracy(outputs,labels):
    _,predicted=torch.max(outputs,1)
    return torch.sum([1 for i in range(len(labels)) if outputs[i]==labels[i]])/len(labels)